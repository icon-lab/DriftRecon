"""Learned single-pass CFG conditioning shared by the DriftRecon backbones.

The condition vector is ``domain + 0.02 * RMSNorm(MLP(sinusoid(cfg)))``.  The
domain term is a single learned token (CT) or a class embedding (MRI
contrasts).  Backbones apply it with zero-initialized FiLM in every residual
block, so a fresh model ignores the guidance scale.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_scalar_embedding(value: torch.Tensor, dim: int) -> torch.Tensor:
    """Fourier embedding of a ``(B,)`` scalar batch into ``(B, dim)``."""
    if value.ndim != 1:
        raise ValueError(f"Expected scalars shaped (B,), got {tuple(value.shape)}.")
    if int(dim) < 2:
        raise ValueError("Conditioning dimension must be at least 2.")
    half = int(dim) // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=value.device, dtype=torch.float32)
        / float(half)
    )
    phase = value.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)
    if int(dim) % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class RMSNorm(nn.Module):
    """RMS normalization; ``eps=None`` uses the input dtype's machine epsilon."""

    def __init__(self, dim: int, eps: Optional[float] = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(dim)))
        self.eps = None if eps is None else float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(value.dtype).eps if self.eps is None else self.eps
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        return normalized * self.weight


class CFGConditionEncoder(nn.Module):
    """Domain/class token plus learned CFG-scale conditioning.

    ``num_classes=None`` uses one learned domain token (no label input);
    otherwise ``labels`` index a class embedding.
    """

    def __init__(
        self,
        cond_dim: int = 256,
        *,
        num_classes: Optional[int] = None,
        norm_eps: Optional[float] = 1e-6,
    ) -> None:
        super().__init__()
        if int(cond_dim) < 2:
            raise ValueError("cond_dim must be at least 2.")
        self.cond_dim = int(cond_dim)
        self.num_classes = None if num_classes is None else int(num_classes)
        if self.num_classes is None:
            self.domain_token = nn.Parameter(torch.empty(self.cond_dim))
            nn.init.normal_(self.domain_token)
        else:
            if self.num_classes <= 0:
                raise ValueError("num_classes must be positive.")
            self.class_embed = nn.Embedding(self.num_classes, self.cond_dim)
        self.cfg_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        self.cfg_norm = RMSNorm(self.cond_dim, eps=norm_eps)

    def forward(
        self,
        cfg_scale: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cfg_scale.ndim != 1:
            raise ValueError(f"Expected cfg_scale shaped (B,), got {tuple(cfg_scale.shape)}.")
        cfg_scale = cfg_scale.float()
        if not torch.isfinite(cfg_scale).all() or torch.any(cfg_scale < 1.0):
            raise ValueError("cfg_scale values must be finite and at least 1.")
        embedding = sinusoidal_scalar_embedding(cfg_scale, self.cond_dim)
        cfg_condition = self.cfg_norm(self.cfg_mlp(embedding))
        if self.num_classes is None:
            domain = self.domain_token[None]
        else:
            if labels is None:
                raise ValueError("Class-conditioned encoder requires labels.")
            if labels.shape != cfg_scale.shape:
                raise ValueError("labels and cfg_scale must both be shaped (B,).")
            domain = self.class_embed(labels.long())
        return domain + 0.02 * cfg_condition


class FiLM(nn.Module):
    """Zero-initialized feature-wise affine modulation from a condition vector."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), 2 * int(channels)))
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 2 or cond.shape[0] != features.shape[0]:
            raise ValueError(
                "Expected cond shaped (B, D) with the feature batch size, "
                f"got {tuple(cond.shape)} for features {tuple(features.shape)}."
            )
        scale, shift = self.affine(cond.float()).to(features.dtype).chunk(2, dim=1)
        return features * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
