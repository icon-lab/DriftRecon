"""Model-input construction and reconstruction assembly for MRI."""

from __future__ import annotations

import torch
import torch.nn.functional as F

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


def _gaussian_window(size: int = 11, sigma: float = 1.5, *, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def magnitude_ssim(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-slice magnitude SSIM (11x11 Gaussian, sigma 1.5) with ``data_range = max |reference|``.

    Used for checkpoint selection only; the paper metrics come from
    ``tools/mri_metrics.py``.
    """
    pred = magnitude(prediction)[:, None]
    ref = magnitude(reference)[:, None]
    data_range = ref.flatten(1).max(dim=1).values.clamp_min(1e-12)[:, None, None, None]
    window = _gaussian_window(device=pred.device)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_p = F.conv2d(pred, window)
    mu_r = F.conv2d(ref, window)
    var_p = F.conv2d(pred * pred, window) - mu_p**2
    var_r = F.conv2d(ref * ref, window) - mu_r**2
    cov = F.conv2d(pred * ref, window) - mu_p * mu_r
    ssim_map = ((2 * mu_p * mu_r + c1) * (2 * cov + c2)) / (
        (mu_p**2 + mu_r**2 + c1) * (var_p + var_r + c2)
    )
    return ssim_map.flatten(1).mean(dim=1)
