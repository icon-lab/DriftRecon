"""Model-input construction and reconstruction assembly for MRI."""

from __future__ import annotations

import torch

from ..core.cg import H1CGPseudoinverse
from .physics import SenseOperator


def build_model_input(
    anchor: torch.Tensor,
    noise: torch.Tensor,
    operator: SenseOperator,
    p_inverse: H1CGPseudoinverse,
    *,
    spectral_scale: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Return ``A^dagger y + sigma (I - A^dagger A) epsilon``."""
    if anchor.shape != noise.shape:
        raise ValueError("anchor and noise must have identical shapes.")
    if float(sigma) < 0:
        raise ValueError("sigma must be nonnegative.")
    null_noise = p_inverse.null_project(operator, noise, spectral_scale=spectral_scale)
    return anchor + float(sigma) * null_noise


def reconstruct(x_us: torch.Tensor, model_output: torch.Tensor, operator: SenseOperator) -> torch.Tensor:
    """Assemble ``A^H y + (I - A^H A) model_output``.

    The measurement enters through the adjoint image and the network output
    through ``I - A^H A``.  With several coils ``A^H A`` is not an orthogonal
    projector, so this is a fixed linear correction rather than exact
    k-space replacement; training uses the H1-CG soft projector instead.
    """
    if x_us.shape != model_output.shape:
        raise ValueError("x_us and model_output must have identical shapes.")
    return x_us + operator.null_project(model_output.float())


def magnitude(image: torch.Tensor) -> torch.Tensor:
    """``(B, 2, H, W)`` complex image to ``(B, H, W)`` magnitude."""
    return torch.sqrt(image[:, 0].float() ** 2 + image[:, 1].float() ** 2 + 1e-12)


def magnitude_psnr(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-slice magnitude PSNR with ``data_range = max |reference|``."""
    pred = magnitude(prediction).flatten(1)
    ref = magnitude(reference).flatten(1)
    data_range = ref.max(dim=1).values.clamp_min(1e-12)
    mse = (pred - ref).square().mean(dim=1).clamp_min(1e-20)
    return 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse)
