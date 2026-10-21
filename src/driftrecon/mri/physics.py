"""Multi-coil SENSE operators for DriftReconMRI.

``SenseOperator`` wraps :class:`deepinv.physics.MultiCoilMRI` with the tensor
layout used throughout DriftRecon:

=========  =====================  ===============================
tensor     shape                  meaning
=========  =====================  ===============================
image      ``(B, 2, H, W)``       real/imaginary complex image
k-space    ``(B, 2, N, H, W)``    masked multi-coil k-space
mask       ``(B, H, W)``          sampling mask
coil maps  ``(B, N, H, W)``       complex sensitivity maps
=========  =====================  ===============================

``PaddedOperator`` and ``MultiscaleOperator`` adapt it to the padded,
multi-resolution feature maps of the backbone.
"""

from __future__ import annotations

from typing import Sequence, Union

import deepinv as dinv
import torch
import torch.nn.functional as F


def _mask_layout(mask: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` mask to the ``(B, 2, H, W)`` layout of the SENSE physics."""
    mask = mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected a (B, H, W) mask, got {tuple(mask.shape)}.")
    return torch.stack([mask, mask], dim=1)


class SenseOperator:
    """Batched SENSE forward ``A = M F S`` and adjoint ``A^H``."""

    def __init__(
        self,
        mask: torch.Tensor,
        coil_maps: torch.Tensor,
        *,
        device: Union[str, torch.device],
    ) -> None:
        mask = _mask_layout(mask.to(device))
        coil_maps = coil_maps.to(device=device)
        if not torch.is_complex(coil_maps):
            raise TypeError("coil_maps must be a complex tensor (B, N, H, W).")
        self.mask = mask[:, 0]
        self.physics = dinv.physics.MultiCoilMRI(
            mask=mask,
            coil_maps=coil_maps.to(torch.complex64),
            img_size=tuple(int(v) for v in mask.shape[-2:]),
            device=device,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.physics.A(image)

    def adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        return self.physics.A_adjoint(kspace)

    A = forward
    A_adjoint = adjoint

    def power_iteration_seed(self, reference: torch.Tensor) -> torch.Tensor:
        """Deterministic start vector: ones on the real, minus ones on the imaginary channel."""
        vector = torch.ones_like(reference.detach(), dtype=torch.float32)
        vector[:, 1::2] *= -1.0
        return vector

    def null_project(self, image: torch.Tensor) -> torch.Tensor:
        """Adjoint-based correction ``(I - A^H A) image`` used by the final assembly."""
        return image - self.adjoint(self.forward(image))


def expand_operator(
    operator: SenseOperator, mask: torch.Tensor, coil_maps: torch.Tensor, generations: int
) -> SenseOperator:
    """SENSE operator for a batch repeated ``generations`` times."""
    if int(generations) == 1:
        return operator
    return SenseOperator(
        mask.repeat_interleave(int(generations), dim=0),
        coil_maps.repeat_interleave(int(generations), dim=0),
        device=mask.device,
    )


class PaddedOperator:
    """Operator on images zero-padded at the top/left to a multiple of 8."""

    def __init__(self, base, pad: tuple[int, int]) -> None:
        self.base = base
        self.pad = (int(pad[0]), int(pad[1]))

    def A(self, image: torch.Tensor) -> torch.Tensor:
        return self.base.A(image[..., self.pad[0]:, self.pad[1]:])

    def A_adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        image = self.base.A_adjoint(kspace)
        return F.pad(image, (self.pad[1], 0, self.pad[0], 0))

    def remove_pad(self, image: torch.Tensor) -> torch.Tensor:
        return image[..., self.pad[0]:, self.pad[1]:]


class MultiscaleOperator:
    """``A`` acting on a coarse image: ``A_s = A U_s`` and ``A_s^H = D_s A^H``.

    ``U_s`` is sinc upsampling by ``2**s`` and ``D_s`` its adjoint; scale 0 is
    the full-resolution operator.
    """

    def __init__(
        self,
        base,
        image_shape: Sequence[int],
        *,
        factors: Sequence[int] = (2, 4, 8),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.base = base
        self.resamplers = [
            dinv.physics.Downsampling(
                img_size=tuple(int(v) for v in image_shape),
                filter="sinc",
                factor=int(factor),
                device=device,
            )
            for factor in factors
        ]

    def A(self, image: torch.Tensor, scale: int = 0) -> torch.Tensor:
        if scale == 0:
            return self.base.A(image)
        return self.base.A(self.resamplers[scale - 1].A_adjoint(image))

    def A_adjoint(self, kspace: torch.Tensor, scale: int = 0) -> torch.Tensor:
        image = self.base.A_adjoint(kspace)
        if scale == 0:
            return image
        return self.resamplers[scale - 1].A(image)
