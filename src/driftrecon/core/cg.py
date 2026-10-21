"""Fixed-step H1 conjugate-gradient pseudoinverse shared by CT and MRI.

With a detached power-iteration estimate ``L`` of ``lambda_max(A^H A)`` the
module solves

    (A^H A / L + alpha D^H D + floor I) x = rhs / L

for ``rhs = A^H y`` (pseudoinverse ``A^dagger y``) or ``rhs = A^H A x``
(soft null projector ``(I - A^dagger A) x = x - A^dagger A x``).  ``D`` is the
periodic finite-difference gradient and ``alpha`` is a positive, optionally
learnable weight.

Operators must provide ``forward(x)``, ``adjoint(y)`` and
``power_iteration_seed(reference)`` on real image tensors ``(B, C, H, W)``.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def batch_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-sample real inner product with float64 reductions."""
    return (left.double() * right.double()).flatten(start_dim=1).sum(dim=1)


def batch_view(value: torch.Tensor, ndim: int) -> torch.Tensor:
    return value.reshape(value.shape[0], *([1] * (ndim - 1)))


def periodic_gradient_gram(image: torch.Tensor) -> torch.Tensor:
    """Normalized periodic finite-difference ``D^H D`` (spectrum <= 1)."""
    laplacian = 4.0 * image
    laplacian = laplacian - torch.roll(image, 1, -2) - torch.roll(image, -1, -2)
    laplacian = laplacian - torch.roll(image, 1, -1) - torch.roll(image, -1, -1)
    return laplacian / 8.0


def _autocast_off(tensor: torch.Tensor):
    # CG always runs in float32: normal operators and CG recurrences are
    # fragile in reduced precision.
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


class H1CGPseudoinverse(nn.Module):
    """Regularized pseudoinverse ``A^dagger`` and soft null projector.

    One instance is shared by the model input anchor, the input null noise,
    the null-space losses and the final reconstruction, so the operator
    definitions cannot silently diverge.
    """

    def __init__(
        self,
        *,
        cg_steps: int = 15,
        h1_alpha_init: float = 0.5,
        h1_floor: float = 1e-6,
        power_iterations: int = 20,
        learnable_alpha: bool = True,
    ) -> None:
        super().__init__()
        if int(cg_steps) <= 0:
            raise ValueError("cg_steps must be positive.")
        if not math.isfinite(float(h1_alpha_init)) or float(h1_alpha_init) <= 0:
            raise ValueError("h1_alpha_init must be finite and positive.")
        if not math.isfinite(float(h1_floor)) or float(h1_floor) <= 0:
            raise ValueError("h1_floor must be finite and positive.")
        if int(power_iterations) <= 0:
            raise ValueError("power_iterations must be positive.")
        self.cg_steps = int(cg_steps)
        self.h1_floor = float(h1_floor)
        self.power_iterations = int(power_iterations)
        # Inverse softplus so that softplus(raw_alpha) == h1_alpha_init.
        raw = torch.tensor(math.log(math.expm1(float(h1_alpha_init))), dtype=torch.float32)
        if learnable_alpha:
            self.raw_alpha = nn.Parameter(raw)
        else:
            self.register_buffer("raw_alpha", raw)

    @property
    def h1_alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

    @staticmethod
    def _normal(operator, value: torch.Tensor) -> torch.Tensor:
        return operator.adjoint(operator.forward(value)).float()

    @torch.no_grad()
    def estimate_spectral_scale(self, operator, reference: torch.Tensor) -> torch.Tensor:
        """Deterministic per-sample estimate of ``lambda_max(A^H A)``.

        ``reference`` only fixes the image shape; its values are not used.
        """
        with _autocast_off(reference):
            vector = operator.power_iteration_seed(reference).float()
            eps = torch.finfo(torch.float64).eps
            norm = torch.sqrt(batch_inner(vector, vector).clamp_min(eps))
            vector = vector / batch_view(norm.to(vector.dtype), vector.ndim)
            for _ in range(self.power_iterations):
                normal_vector = self._normal(operator, vector)
                normal_norm = torch.sqrt(
                    batch_inner(normal_vector, normal_vector).clamp_min(eps)
                )
                vector = normal_vector / batch_view(
                    normal_norm.to(vector.dtype), vector.ndim
                )
            normal_vector = self._normal(operator, vector)
            numerator = batch_inner(vector, normal_vector)
            denominator = batch_inner(vector, vector).clamp_min(eps)
        return (numerator / denominator).clamp_min(eps).to(reference.dtype)

    def solve_rhs(
        self,
        operator,
        rhs: torch.Tensor,
        *,
        spectral_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the normalized H1 system for an image-domain right side."""
        if spectral_scale.shape != (rhs.shape[0],):
            raise ValueError(
                f"spectral_scale must have shape ({rhs.shape[0]},), got "
                f"{tuple(spectral_scale.shape)}."
            )
        with _autocast_off(rhs):
            rhs = rhs.float()
            scale = spectral_scale.detach().to(rhs).clamp_min(
                torch.finfo(rhs.dtype).eps
            )
            scale_image = batch_view(scale, rhs.ndim)
            alpha = self.h1_alpha.to(rhs)

            def system(value: torch.Tensor) -> torch.Tensor:
                return (
                    self._normal(operator, value) / scale_image
                    + alpha * periodic_gradient_gram(value)
                    + self.h1_floor * value
                )

            normalized_rhs = rhs / scale_image
            solution = torch.zeros_like(normalized_rhs)
            residual = normalized_rhs
            direction = residual
            residual_energy = batch_inner(residual, residual)
            tiny = torch.finfo(residual_energy.dtype).eps
            # Fixed unroll: no tolerance-dependent branching enters the graph.
            for _ in range(self.cg_steps):
                system_direction = system(direction)
                denominator = batch_inner(direction, system_direction).clamp_min(tiny)
                active = residual_energy > tiny
                step = torch.where(
                    active,
                    residual_energy / denominator,
                    torch.zeros_like(residual_energy),
                )
                step_image = batch_view(step.to(rhs.dtype), rhs.ndim)
                solution = solution + step_image * direction
                new_residual = residual - step_image * system_direction
                new_energy = batch_inner(new_residual, new_residual)
                beta = torch.where(
                    active,
                    new_energy / residual_energy.clamp_min(tiny),
                    torch.zeros_like(new_energy),
                )
                direction = new_residual + batch_view(
                    beta.to(rhs.dtype), rhs.ndim
                ) * direction
                residual = new_residual
                residual_energy = new_energy
            return solution

    def reconstruct(
        self,
        operator,
        measurement: torch.Tensor,
        *,
        reference: torch.Tensor,
        spectral_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ``A^dagger y``; ``reference`` gives the image shape."""
        if spectral_scale is None:
            spectral_scale = self.estimate_spectral_scale(operator, reference)
        with _autocast_off(measurement):
            rhs = operator.adjoint(measurement.float()).float()
        return self.solve_rhs(operator, rhs, spectral_scale=spectral_scale)

    def null_project(
        self,
        operator,
        image: torch.Tensor,
        *,
        spectral_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the soft null projector ``(I - A^dagger A)``."""
        if spectral_scale is None:
            spectral_scale = self.estimate_spectral_scale(operator, image)
        with _autocast_off(image):
            work = image.float()
            rhs = self._normal(operator, work)
        return work - self.solve_rhs(operator, rhs, spectral_scale=spectral_scale)

    def config_dict(self) -> dict:
        return {
            "cg_steps": self.cg_steps,
            "h1_alpha_init": float(self.h1_alpha.detach().cpu()),
            "h1_floor": self.h1_floor,
            "power_iterations": self.power_iterations,
            "learnable_alpha": isinstance(self.raw_alpha, nn.Parameter),
        }
