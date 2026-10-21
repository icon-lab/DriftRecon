"""Model-input construction and reconstruction assembly for CT."""

from __future__ import annotations

import torch

from ..core.cg import H1CGPseudoinverse
from .physics import ParallelBeamCT


def build_model_input(
    anchor: torch.Tensor,
    noise: torch.Tensor,
    operator: ParallelBeamCT,
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
    null_noise = p_inverse.null_project(
        operator, noise, spectral_scale=spectral_scale
    )
    return anchor + float(sigma) * null_noise


def reconstruct(
    anchor: torch.Tensor,
    model_output: torch.Tensor,
    operator: ParallelBeamCT,
    p_inverse: H1CGPseudoinverse,
    *,
    spectral_scale: torch.Tensor,
) -> torch.Tensor:
    """Assemble ``A^dagger y + (I - A^dagger A) model_output``."""
    if anchor.shape != model_output.shape:
        raise ValueError("anchor and model_output must have identical shapes.")
    return anchor + p_inverse.null_project(
        operator, model_output, spectral_scale=spectral_scale
    )
