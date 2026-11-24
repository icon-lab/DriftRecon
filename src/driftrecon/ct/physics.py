"""A dependency-light, matched parallel-beam CT operator.

The discretization is pixel driven: each image pixel contributes linearly to
the two nearest detector bins.  Using the same interpolation weights in the
backprojector makes :meth:`adjoint` the exact transpose of :meth:`forward`.
That property is more important to the unrolled CG solve than pairing an
unmatched rotate-and-sum forward projector with an approximate backprojector.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

# On CUDA the projector is stored as a sparse CSR matrix ``M`` (views*detectors
# x H*W), so forward = M x and adjoint = M^T s are single sparse-dense
# products instead of a Python loop over views.  The weights and sums are the
# same as in the loop.  Set DRIFTRECON_CT_SPARSE=0 to use the loop everywhere.
_USE_SPARSE = os.environ.get("DRIFTRECON_CT_SPARSE", "1") != "0"


class _SparseApply(torch.autograd.Function):
    """``Y = M @ X`` with gradient ``M^T @ dY`` (``M``, ``Mt`` CSR; ``X`` dense (n, k))."""

    @staticmethod
    def forward(ctx, x, matrix, matrix_t):
        ctx.matrix_t = matrix_t
        return torch.sparse.mm(matrix, x)

    @staticmethod
    def backward(ctx, grad):
        return torch.sparse.mm(ctx.matrix_t, grad.contiguous()), None, None


class ParallelBeamCT(nn.Module):
    """Batched 2-D sparse-view parallel-beam CT physics.

    Images have shape ``(B, C, H, W)`` and sinograms have shape
    ``(B, C, views, detectors)``.  The operator is real valued and supports
    any channel count, although DriftReconCT uses one channel.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int],
        num_views: int,
        num_detectors: Optional[int] = None,
        *,
        angles: Optional[Sequence[float] | torch.Tensor] = None,
        pixel_size: float = 1.0,
        detector_spacing: float = 1.0,
    ) -> None:
        super().__init__()
        height, width = (int(image_shape[0]), int(image_shape[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f"image_shape must be positive, got {image_shape}.")
        if int(num_views) <= 0:
            raise ValueError(f"num_views must be positive, got {num_views}.")
        if not math.isfinite(float(pixel_size)) or float(pixel_size) <= 0:
            raise ValueError("pixel_size must be finite and positive.")
        if not math.isfinite(float(detector_spacing)) or float(detector_spacing) <= 0:
            raise ValueError("detector_spacing must be finite and positive.")

        if num_detectors is None:
            physical_diagonal = math.hypot(height * pixel_size, width * pixel_size)
            num_detectors = int(math.ceil(physical_diagonal / detector_spacing)) + 1
        if int(num_detectors) < 2:
            raise ValueError("num_detectors must be at least 2.")

        self.image_shape = (height, width)
        self.num_views = int(num_views)
        self.num_detectors = int(num_detectors)
        self.pixel_size = float(pixel_size)
        self.detector_spacing = float(detector_spacing)

        if angles is None:
            angle_tensor = torch.arange(self.num_views, dtype=torch.float32)
            angle_tensor = angle_tensor * (math.pi / self.num_views)
        else:
            angle_tensor = torch.as_tensor(angles, dtype=torch.float32)
            if angle_tensor.ndim != 1 or angle_tensor.numel() != self.num_views:
                raise ValueError(
                    f"angles must have {self.num_views} values, got "
                    f"shape {tuple(angle_tensor.shape)}."
                )
            if not torch.isfinite(angle_tensor).all():
                raise ValueError("angles must be finite.")
        self.register_buffer("angles", angle_tensor)

        # Precompute the two detector bins and weights touched by each pixel.
        # Buffers follow the module to CUDA and avoid geometry work in every CG
        # iteration.  Their shape is (views, H*W).
        y = (
            torch.arange(height, dtype=torch.float32) - (height - 1) / 2.0
        ) * self.pixel_size
        x = (
            torch.arange(width, dtype=torch.float32) - (width - 1) / 2.0
        ) * self.pixel_size
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        positions = (
            torch.cos(angle_tensor)[:, None] * xx.flatten()[None]
            + torch.sin(angle_tensor)[:, None] * yy.flatten()[None]
        )
        detector_coordinate = (
            positions / self.detector_spacing + (self.num_detectors - 1) / 2.0
        )
        lower_unclamped = torch.floor(detector_coordinate)
        upper_unclamped = lower_unclamped + 1.0
        valid_lower = (lower_unclamped >= 0) & (
            lower_unclamped < self.num_detectors
        )
        valid_upper = (upper_unclamped >= 0) & (
            upper_unclamped < self.num_detectors
        )
        lower = lower_unclamped.clamp(0, self.num_detectors - 1).long()
        upper = upper_unclamped.clamp(0, self.num_detectors - 1).long()
        upper_weight = detector_coordinate - lower_unclamped
        lower_weight = 1.0 - upper_weight
        lower_weight = lower_weight * valid_lower
        upper_weight = upper_weight * valid_upper

        self.register_buffer("lower_index", lower)
        self.register_buffer("upper_index", upper)
        self.register_buffer("lower_weight", lower_weight)
        self.register_buffer("upper_weight", upper_weight)
        self.projection_scale = self.pixel_size**2 / self.detector_spacing

    def extra_repr(self) -> str:
        return (
            f"image_shape={self.image_shape}, num_views={self.num_views}, "
            f"num_detectors={self.num_detectors}"
        )

    def _check_image(self, image: torch.Tensor) -> None:
        if image.ndim != 4 or tuple(image.shape[-2:]) != self.image_shape:
            raise ValueError(
                "Expected images shaped (B, C, H, W) with H,W="
                f"{self.image_shape}, got {tuple(image.shape)}."
            )
        if not torch.is_floating_point(image):
            raise TypeError("CT images must use a floating-point dtype.")

    def _check_sinogram(self, sinogram: torch.Tensor) -> None:
        expected = (self.num_views, self.num_detectors)
        if sinogram.ndim != 4 or tuple(sinogram.shape[-2:]) != expected:
            raise ValueError(
                "Expected sinograms shaped (B, C, views, detectors) with "
                f"views,detectors={expected}, got {tuple(sinogram.shape)}."
            )
        if not torch.is_floating_point(sinogram):
            raise TypeError("CT sinograms must use a floating-point dtype.")

    def _sparse_matrices(self, device: torch.device):
        """Cached CSR projector ``M`` and its transpose on ``device``."""
        cache = self.__dict__.setdefault("_sparse_cache", {})
        key = str(device)
        if key not in cache:
            views = self.num_views
            pixels = self.image_shape[0] * self.image_shape[1]
            offset = (torch.arange(views, device=device) * self.num_detectors)[:, None]
            rows = torch.cat(
                [
                    (self.lower_index.to(device) + offset).flatten(),
                    (self.upper_index.to(device) + offset).flatten(),
                ]
            )
            cols = torch.arange(pixels, device=device).repeat(views)
            cols = torch.cat([cols, cols])
            values = torch.cat(
                [
                    self.lower_weight.to(device).flatten(),
                    self.upper_weight.to(device).flatten(),
                ]
            ).float()
            keep = values != 0
            coo = torch.sparse_coo_tensor(
                torch.stack([rows[keep], cols[keep]]),
                values[keep],
                (views * self.num_detectors, pixels),
            ).coalesce()
            cache[key] = (coo.to_sparse_csr(), coo.t().coalesce().to_sparse_csr())
        return cache[key]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the sparse-view ray transform."""
        self._check_image(image)
        batch, channels = image.shape[:2]
        if _USE_SPARSE and image.is_cuda:
            matrix, matrix_t = self._sparse_matrices(image.device)
            flat = image.reshape(batch * channels, -1).float().t().contiguous()
            out = _SparseApply.apply(flat, matrix, matrix_t).t()
            return (
                out.reshape(batch, channels, self.num_views, self.num_detectors).to(image.dtype)
                * self.projection_scale
            )
        pixels = image.flatten(start_dim=2)
        projections = []
        for view in range(self.num_views):
            projection = image.new_zeros(batch, channels, self.num_detectors)
            lower = self.lower_index[view].view(1, 1, -1).expand(batch, channels, -1)
            upper = self.upper_index[view].view(1, 1, -1).expand(batch, channels, -1)
            lower_weight = self.lower_weight[view].to(image.dtype).view(1, 1, -1)
            upper_weight = self.upper_weight[view].to(image.dtype).view(1, 1, -1)
            projection.scatter_add_(-1, lower, pixels * lower_weight)
            projection.scatter_add_(-1, upper, pixels * upper_weight)
            projections.append(projection)
        return torch.stack(projections, dim=2) * self.projection_scale

    A = forward

    def adjoint(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Apply the exact discrete transpose (unfiltered backprojection)."""
        self._check_sinogram(sinogram)
        batch, channels = sinogram.shape[:2]
        if _USE_SPARSE and sinogram.is_cuda:
            matrix, matrix_t = self._sparse_matrices(sinogram.device)
            flat = sinogram.reshape(batch * channels, -1).float().t().contiguous()
            out = _SparseApply.apply(flat, matrix_t, matrix).t()
            return (
                out.reshape(batch, channels, *self.image_shape).to(sinogram.dtype)
                * self.projection_scale
            )
        backprojection = sinogram.new_zeros(
            batch, channels, self.image_shape[0] * self.image_shape[1]
        )
        for view in range(self.num_views):
            projection = sinogram[:, :, view]
            lower = self.lower_index[view].view(1, 1, -1).expand(batch, channels, -1)
            upper = self.upper_index[view].view(1, 1, -1).expand(batch, channels, -1)
            lower_weight = self.lower_weight[view].to(sinogram.dtype).view(1, 1, -1)
            upper_weight = self.upper_weight[view].to(sinogram.dtype).view(1, 1, -1)
            backprojection = backprojection + projection.gather(-1, lower) * lower_weight
            backprojection = backprojection + projection.gather(-1, upper) * upper_weight
        return backprojection.reshape(batch, channels, *self.image_shape) * self.projection_scale

    A_adjoint = adjoint

    def normal(self, image: torch.Tensor) -> torch.Tensor:
        """Apply ``A^T A``."""
        return self.adjoint(self.forward(image))

    def power_iteration_seed(self, reference: torch.Tensor) -> torch.Tensor:
        """All-ones start vector for power iteration.

        ``A^T A`` is entrywise nonnegative for this discretization, so the
        all-ones vector overlaps its dominant eigenvector.
        """
        self._check_image(reference)
        return torch.ones_like(reference, dtype=torch.float32)

    def geometry_dict(self) -> dict:
        """Serializable constructor arguments for checkpoints."""
        return {
            "image_shape": self.image_shape,
            "num_views": self.num_views,
            "num_detectors": self.num_detectors,
            "angles": self.angles.detach().cpu().tolist(),
            "pixel_size": self.pixel_size,
            "detector_spacing": self.detector_spacing,
        }

