"""DriftReconMRI: complex-valued U-Net with multi-scale SENSE conditioning."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..core.conditioning import CFGConditionEncoder, FiLM
from .data.m4raw import CONTRASTS
from .physics import MultiscaleOperator, PaddedOperator

IMAGE_CHANNELS = 2  # real / imaginary
WIDTHS = (64, 128, 256, 512)


def _conv3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)


class KrylovEncoder(nn.Module):
    """Two-layer residual encoder from Krylov features to block features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_in = _conv3(in_channels, in_channels)
        self.skip_in = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv_a = _conv3(in_channels, in_channels)
        self.conv_b = _conv3(in_channels, out_channels)
        self.skip_out = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv_in(value) + self.skip_in(value)
        hidden = F.relu(self.conv_a(value))
        return self.conv_b(hidden) + self.skip_out(value)


class MeasurementBlock(nn.Module):
    """Krylov measurement features ``[v, s^2 A_s^H A_s v, ...]`` at one scale.

    One sequence starts from ``A_s^H y`` and one from each of
    ``decoded_images`` complex images decoded from the block features;
    ``s = 2**scale`` compensates the energy of the coarse operator.
    """

    def __init__(self, channels: int, *, krylov_steps: int = 2, decoded_images: int = 2) -> None:
        super().__init__()
        if int(krylov_steps) <= 0:
            raise ValueError("krylov_steps must be positive.")
        self.krylov_steps = int(krylov_steps)
        self.decoded_images = int(decoded_images)
        self.decode = _conv3(channels, IMAGE_CHANNELS * self.decoded_images)
        in_channels = IMAGE_CHANNELS * self.krylov_steps * (self.decoded_images + 1)
        self.encode = KrylovEncoder(in_channels, channels)

    def _krylov(self, initial, operator: MultiscaleOperator, scale: int) -> torch.Tensor:
        weight = float(2**scale) ** 2
        values = [initial]
        current = initial
        for _ in range(self.krylov_steps - 1):
            current = operator.A_adjoint(operator.A(current, scale), scale) * weight
            values.append(current)
        return torch.cat(values, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        kspace: torch.Tensor,
        operator: MultiscaleOperator,
        scale: int,
    ) -> torch.Tensor:
        decoded = self.decode(features)
        sequences = [self._krylov(operator.A_adjoint(kspace, scale), operator, scale)]
        for index in range(self.decoded_images):
            image = decoded[:, IMAGE_CHANNELS * index : IMAGE_CHANNELS * (index + 1)]
            sequences.append(self._krylov(image, operator, scale))
        return F.relu(self.encode(torch.cat(sequences, dim=1)))


class ResidualBlock(nn.Module):
    """``x + conv2(FiLM(relu(conv1 x))) + gain * measurement(FiLM(...))``.

    ``measurement_conditioning=False`` (ablation) drops the measurement term
    and its gain.
    """

    def __init__(
        self,
        channels: int,
        *,
        cond_dim: int,
        krylov_steps: int,
        measurement_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3(channels, channels)
        self.conv2 = _conv3(channels, channels)
        self.film = FiLM(cond_dim, channels)
        if measurement_conditioning:
            self.gain = nn.Parameter(torch.tensor([1.0]))
            self.measurement = MeasurementBlock(channels, krylov_steps=krylov_steps)
        else:
            self.gain = None
            self.measurement = None

    def forward(self, features, cond, kspace, operator: MultiscaleOperator, scale: int):
        value = self.film(F.relu(self.conv1(features)), cond)
        residual = self.conv2(value)
        if self.measurement is None:
            return features + residual
        measured = self.measurement(value, kspace, operator, scale)
        return features + residual + self.gain * measured


class Stage(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        blocks: int,
        cond_dim: int,
        krylov_steps: int,
        use_checkpoint: bool,
        measurement_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels,
                    cond_dim=cond_dim,
                    krylov_steps=krylov_steps,
                    measurement_conditioning=measurement_conditioning,
                )
                for _ in range(int(blocks))
            ]
        )
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, features, cond, kspace, operator, scale: int):
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Non-tensor arguments are captured by the closure.
                def run(x, c, current_block=block):
                    return current_block(x, c, kspace, operator, scale)

                features = checkpoint(run, features, cond, use_reentrant=False)
            else:
                features = block(features, cond, kspace, operator, scale)
        return features


class DriftReconMRI(nn.Module):
    """DriftRecon generator for multi-coil MRI.

    A four-level U-Net (widths 64/128/256/512, four residual blocks per
    stage) on ``(B, 2, H, W)`` complex images.  Each stage at depth ``s``
    is conditioned on Krylov features of the SENSE operator restricted to
    resolution ``2**-s``, and every block receives the contrast/CFG condition.
    Inputs are zero-padded to a multiple of 8 internally.
    ``measurement_conditioning=False`` is the ablation without the
    operator-aware (Krylov) blocks.
    """

    def __init__(
        self,
        *,
        krylov_steps: int = 2,
        blocks_per_stage: int = 4,
        cond_dim: int = 256,
        num_classes: int = len(CONTRASTS),
        use_checkpoint: bool = False,
        measurement_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.krylov_steps = int(krylov_steps)
        self.blocks_per_stage = int(blocks_per_stage)
        self.cond_dim = int(cond_dim)
        self.num_classes = int(num_classes)
        self.use_checkpoint = bool(use_checkpoint)
        self.measurement_conditioning = bool(measurement_conditioning)
        self.condition_encoder = CFGConditionEncoder(
            self.cond_dim, num_classes=self.num_classes, norm_eps=None
        )

        def stage(width: int) -> Stage:
            return Stage(
                width,
                blocks=self.blocks_per_stage,
                cond_dim=self.cond_dim,
                krylov_steps=self.krylov_steps,
                use_checkpoint=self.use_checkpoint,
                measurement_conditioning=self.measurement_conditioning,
            )

        w0, w1, w2, w3 = WIDTHS
        self.head = _conv3(IMAGE_CHANNELS, w0)
        self.enc1, self.enc2, self.enc3 = stage(w0), stage(w1), stage(w2)
        self.body = stage(w3)
        self.dec3, self.dec2, self.dec1 = stage(w2), stage(w1), stage(w0)
        self.down1 = nn.Conv2d(w0, w1, 2, stride=2, bias=False)
        self.down2 = nn.Conv2d(w1, w2, 2, stride=2, bias=False)
        self.down3 = nn.Conv2d(w2, w3, 2, stride=2, bias=False)
        self.up3 = nn.ConvTranspose2d(w3, w2, 2, stride=2, bias=False)
        self.up2 = nn.ConvTranspose2d(w2, w1, 2, stride=2, bias=False)
        self.up1 = nn.ConvTranspose2d(w1, w0, 2, stride=2, bias=False)
        self.tail = _conv3(w0, IMAGE_CHANNELS)

    def forward(
        self,
        model_input: torch.Tensor,
        kspace: torch.Tensor,
        operator,
        cfg_scale: torch.Tensor,
        contrast_id: torch.Tensor,
    ) -> torch.Tensor:
        """``model_input`` ``(B, 2, H, W)``; ``kspace`` ``(B, 2, N, H, W)``; ``operator`` a ``SenseOperator``."""
        if model_input.ndim != 4 or model_input.shape[1] != IMAGE_CHANNELS:
            raise ValueError(f"Expected input (B, 2, H, W), got {tuple(model_input.shape)}.")
        cond = self.condition_encoder(cfg_scale, contrast_id)
        pad = (-model_input.shape[-2] % 8, -model_input.shape[-1] % 8)
        padded = PaddedOperator(operator, pad)
        x0 = F.pad(model_input, (pad[1], 0, pad[0], 0)) if any(pad) else model_input
        multiscale = MultiscaleOperator(padded, x0.shape[1:], device=x0.device)

        x1 = self.head(x0)
        x2 = self.down1(self.enc1(x1, cond, kspace, multiscale, 0))
        x3 = self.down2(self.enc2(x2, cond, kspace, multiscale, 1))
        x4 = self.down3(self.enc3(x3, cond, kspace, multiscale, 2))
        x = self.body(x4, cond, kspace, multiscale, 3)
        x = self.dec3(self.up3(x + x4), cond, kspace, multiscale, 2)
        x = self.dec2(self.up2(x + x3), cond, kspace, multiscale, 1)
        x = self.dec1(self.up1(x + x2), cond, kspace, multiscale, 0)
        return padded.remove_pad(self.tail(x + x1))

    def config_dict(self) -> dict:
        return {
            "krylov_steps": self.krylov_steps,
            "blocks_per_stage": self.blocks_per_stage,
            "cond_dim": self.cond_dim,
            "num_classes": self.num_classes,
            "use_checkpoint": self.use_checkpoint,
            "measurement_conditioning": self.measurement_conditioning,
        }
