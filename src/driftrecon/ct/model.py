"""DriftReconCT: one-channel U-Net with sinogram-physics conditioning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..core.conditioning import CFGConditionEncoder, FiLM
from .physics import ParallelBeamCT


def _resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


class CTMeasurementBlock(nn.Module):
    """Krylov measurement features ``[v, A^T A v, ...]`` for one residual block.

    Two sequences are built: one from the backprojected sinogram ``A^T y`` and
    one from a single-channel image decoded from the block features.
    """

    def __init__(self, channels: int, krylov_steps: int = 2) -> None:
        super().__init__()
        if int(krylov_steps) <= 0:
            raise ValueError("krylov_steps must be positive.")
        self.krylov_steps = int(krylov_steps)
        self.decode = nn.Conv2d(int(channels), 1, 3, padding=1, bias=False)
        self.encode = nn.Sequential(
            nn.Conv2d(2 * self.krylov_steps, int(channels), 3, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(int(channels), int(channels), 3, padding=1, bias=False),
        )
        self.gain = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _normalizer(operator: ParallelBeamCT) -> float:
        # Parallel-beam A^T A has a leading scale close to views times the
        # image diameter; dividing by it keeps the Krylov features at a stable
        # magnitude across resolutions and view counts.
        return float(
            operator.num_views
            * max(operator.image_shape)
            * operator.projection_scale**2
        )

    def _normal_at_size(
        self,
        image: torch.Tensor,
        operator: ParallelBeamCT,
        size: tuple[int, int],
    ) -> torch.Tensor:
        full_image = _resize(image, operator.image_shape)
        full_normal = operator.normal(full_image) / self._normalizer(operator)
        return _resize(full_normal, size)

    def _krylov(
        self,
        initial: torch.Tensor,
        operator: ParallelBeamCT,
        size: tuple[int, int],
    ) -> torch.Tensor:
        values = [initial]
        current = initial
        for _ in range(self.krylov_steps - 1):
            current = self._normal_at_size(current, operator, size)
            values.append(current)
        return torch.cat(values, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        sinogram: torch.Tensor,
        operator: ParallelBeamCT,
    ) -> torch.Tensor:
        size = tuple(int(v) for v in features.shape[-2:])
        measured = _resize(operator.adjoint(sinogram) / self._normalizer(operator), size)
        decoded = self.decode(features)
        embedding = torch.cat(
            [
                self._krylov(measured, operator, size),
                self._krylov(decoded, operator, size),
            ],
            dim=1,
        )
        return self.gain * self.encode(embedding)


class CTResidualBlock(nn.Module):
    """Convolutional residual block with FiLM and measurement conditioning."""

    def __init__(
        self,
        channels: int,
        *,
        cond_dim: int,
        krylov_steps: int,
    ) -> None:
        super().__init__()
        channels = int(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = FiLM(cond_dim, channels)
        self.measurement = CTMeasurementBlock(channels, krylov_steps)

    def forward(
        self,
        features: torch.Tensor,
        cond: torch.Tensor,
        sinogram: torch.Tensor,
        operator: ParallelBeamCT,
    ) -> torch.Tensor:
        value = F.relu(self.conv1(features), inplace=False)
        value = self.film(value, cond)
        residual = self.conv2(value)
        measured = self.measurement(value, sinogram, operator)
        return features + residual + measured


class CTStage(nn.Module):
    def __init__(
        self,
        channels: int,
        blocks: int,
        *,
        cond_dim: int,
        krylov_steps: int,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        if int(blocks) <= 0:
            raise ValueError("Each DriftReconCT stage needs at least one block.")
        self.blocks = nn.ModuleList(
            [
                CTResidualBlock(
                    channels,
                    cond_dim=cond_dim,
                    krylov_steps=krylov_steps,
                )
                for _ in range(int(blocks))
            ]
        )
        self.use_checkpoint = bool(use_checkpoint)

    def forward(
        self,
        features: torch.Tensor,
        cond: torch.Tensor,
        sinogram: torch.Tensor,
        operator: ParallelBeamCT,
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # The operator is captured because checkpoint takes tensors only.
                def run(x, c, y, current_block=block):
                    return current_block(x, c, y, operator)

                features = checkpoint(run, features, cond, sinogram, use_reentrant=False)
            else:
                features = block(features, cond, sinogram, operator)
        return features


class DriftReconCT(nn.Module):
    """DriftRecon generator for sparse-view CT.

    A four-level U-Net (default widths 64/128/256/512, four residual blocks
    per stage) whose every block is conditioned on the CFG scale and on
    Krylov features of the sparse-view sinogram.  Input and output are
    one-channel images ``(B, 1, H, W)`` with ``H, W`` divisible by 8.
    """

    def __init__(
        self,
        *,
        base_channels: int = 64,
        blocks_per_stage: int = 4,
        krylov_steps: int = 2,
        cond_dim: int = 256,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if int(base_channels) <= 0:
            raise ValueError("base_channels must be positive.")
        self.base_channels = int(base_channels)
        self.blocks_per_stage = int(blocks_per_stage)
        self.krylov_steps = int(krylov_steps)
        self.cond_dim = int(cond_dim)
        self.use_checkpoint = bool(use_checkpoint)
        widths = [self.base_channels * (2**index) for index in range(4)]
        self.condition_encoder = CFGConditionEncoder(cond_dim=self.cond_dim)
        self.head = nn.Conv2d(1, widths[0], 3, padding=1, bias=False)

        def stage(width: int) -> CTStage:
            return CTStage(
                width,
                self.blocks_per_stage,
                cond_dim=self.cond_dim,
                krylov_steps=self.krylov_steps,
                use_checkpoint=self.use_checkpoint,
            )

        self.encoder = nn.ModuleList([stage(width) for width in widths[:3]])
        self.body = stage(widths[3])
        self.decoder = nn.ModuleList([stage(width) for width in reversed(widths[:3])])
        self.down = nn.ModuleList(
            [
                nn.Conv2d(widths[index], widths[index + 1], 2, stride=2, bias=False)
                for index in range(3)
            ]
        )
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    widths[index + 1], widths[index], 2, stride=2, bias=False
                )
                for index in reversed(range(3))
            ]
        )
        self.tail = nn.Conv2d(widths[0], 1, 3, padding=1, bias=False)

    def forward(
        self,
        model_input: torch.Tensor,
        sinogram: torch.Tensor,
        operator: ParallelBeamCT,
        cfg_scale: torch.Tensor,
    ) -> torch.Tensor:
        if model_input.ndim != 4 or model_input.shape[1] != 1:
            raise ValueError(
                f"Expected CT input shaped (B, 1, H, W), got {tuple(model_input.shape)}."
            )
        if tuple(model_input.shape[-2:]) != operator.image_shape:
            raise ValueError("Model and CT operator image sizes differ.")
        if model_input.shape[-2] % 8 or model_input.shape[-1] % 8:
            raise ValueError("DriftReconCT image dimensions must be divisible by 8.")
        if sinogram.shape[0] != model_input.shape[0]:
            raise ValueError("Model input and sinogram batch sizes differ.")
        if cfg_scale.shape != (model_input.shape[0],):
            raise ValueError(
                f"Expected cfg_scale shape ({model_input.shape[0]},), got "
                f"{tuple(cfg_scale.shape)}."
            )

        cond = self.condition_encoder(cfg_scale)
        value = self.head(model_input)
        skips = []
        for index, stage in enumerate(self.encoder):
            value = stage(value, cond, sinogram, operator)
            skips.append(value)
            value = self.down[index](value)
        value = self.body(value, cond, sinogram, operator)
        for index, stage in enumerate(self.decoder):
            value = self.up[index](value)
            value = value + skips[-1 - index]
            value = stage(value, cond, sinogram, operator)
        return self.tail(value)

    def config_dict(self) -> dict:
        return {
            "base_channels": self.base_channels,
            "blocks_per_stage": self.blocks_per_stage,
            "krylov_steps": self.krylov_steps,
            "cond_dim": self.cond_dim,
            "use_checkpoint": self.use_checkpoint,
        }
