"""DriftReconMRI training engine and checkpoint I/O."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from ..core import TRAIN_NULL_NOISE_SIGMA
from ..core.cg import H1CGPseudoinverse
from ..core.drift import (
    driftrecon_objective,
    expand_generations,
    group_generations,
    sample_cfg_scales,
    scheduled_null_l1_weight,
)
from ..core.ema import EMA, load_model_weights
from .model import DriftReconMRI
from .physics import SenseOperator, expand_operator
from .reconstruction import build_model_input, magnitude_psnr, reconstruct

CHECKPOINT_FORMAT = "driftrecon_mri_v1"


@dataclass
class TrainConfig:
    max_steps: int = 20_000
    lr: float = 1e-4
    weight_decay: float = 0.0
    lr_decay_step: int = 10_000
    lr_decay_gamma: float = 0.5
    cfg_min: float = 1.0
    cfg_max: float = 6.0
    cfg_sampling_power: float = 3.0
    no_cfg_fraction: float = 0.5
    generations: int = 2
    cfg_temperature_beta: float = 0.25
    train_sigma: float = TRAIN_NULL_NOISE_SIGMA
    temperatures: tuple[float, ...] = (1.0,)
    drift_scale: float = 1.0
    lambda_drift: float = 1.0
    lambda_l1: float = 0.0
    lambda_mse: float = 0.01
    lambda_l1_null_start: float = 0.01
    lambda_l1_null_end: float = 0.0
    dwt_levels: int = 0
    drift_space: str = "null"
    max_grad_norm: Optional[float] = 1.0
    ema_rate: float = 0.999
    use_fp16: bool = True

    def __post_init__(self) -> None:
        if int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive.")
        if int(self.generations) < 2:
            raise ValueError("CFG drift requires generations >= 2.")
        if not 1.0 <= float(self.cfg_min) <= float(self.cfg_max):
            raise ValueError("Expected 1 <= cfg_min <= cfg_max.")
        if not math.isfinite(float(self.cfg_temperature_beta)) or float(self.cfg_temperature_beta) < 0:
            raise ValueError("cfg_temperature_beta must be finite and nonnegative.")
        if not self.temperatures or any(
            not math.isfinite(float(t)) or float(t) <= 0 for t in self.temperatures
        ):
            raise ValueError("temperatures must contain finite positive values.")
        if int(self.dwt_levels) < 0:
            raise ValueError("dwt_levels must be nonnegative.")
        if self.drift_space not in ("null", "image"):
            raise ValueError("drift_space must be 'null' or 'image'.")
        if int(self.lr_decay_step) <= 0 or not 0.0 < float(self.lr_decay_gamma) <= 1.0:
            raise ValueError("Invalid step learning-rate schedule.")
        if not 0.0 <= float(self.ema_rate) < 1.0:
            raise ValueError("ema_rate must be in [0, 1).")
        if not 0.0 <= float(self.lambda_l1_null_end) <= float(self.lambda_l1_null_start):
            raise ValueError("Null-space L1 weight must be nonnegative and non-increasing.")


class DriftReconMRITrainer:
    """CFG drift on soft null components + image fidelity, AMP and EMA."""

    def __init__(
        self,
        model: DriftReconMRI,
        p_inverse: H1CGPseudoinverse,
        config: TrainConfig,
        *,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.p_inverse = p_inverse.to(self.device)
        self.config = config
        self.step = 0
        groups = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
        ]
        inverse_parameters = [p for p in self.p_inverse.parameters() if p.requires_grad]
        if inverse_parameters:
            # Weight decay would bias the H1 weight; it is learned from the
            # loss alone.
            groups.append({"params": inverse_parameters, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(groups, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=int(config.lr_decay_step), gamma=float(config.lr_decay_gamma)
        )
        self.amp = bool(config.use_fp16 and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.ema = EMA(self.model, rate=config.ema_rate)

    def _autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def _operator_matched_negatives(
        self,
        x_fs: torch.Tensor,
        mask: torch.Tensor,
        coil_maps: torch.Tensor,
        spectral_scale: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, B, 2, H, W)`` tensor with entry ``[i, j] = P_{N_i} x_j``.

        Each condition's mask, coil maps and spectral scale define its soft
        null projector; one condition is processed at a time to bound memory.
        """
        batch_size = int(x_fs.shape[0])
        rows = []
        for index in range(batch_size):
            operator_i = SenseOperator(
                mask[index : index + 1].expand(batch_size, -1, -1),
                coil_maps[index : index + 1].expand(batch_size, -1, -1, -1),
                device=self.device,
            )
            rows.append(
                self.p_inverse.null_project(
                    operator_i, x_fs, spectral_scale=spectral_scale[index].expand(batch_size)
                )
            )
        return torch.stack(rows)

    def train_step(self, batch: dict) -> dict[str, float]:
        cfg = self.config
        x_fs = batch["x_fs"].to(self.device)
        x_us = batch["x_us"].to(self.device)
        kspace = batch["y_us"].to(self.device)
        mask = batch["mask"].to(self.device)
        coil_maps = batch["coil_maps"].to(self.device)
        labels = batch["contrast_id"].to(self.device, dtype=torch.long)
        batch_size = int(x_fs.shape[0])
        if batch_size < 2:
            raise ValueError("CFG drift requires a local batch of at least 2.")
        generations = int(cfg.generations)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        cfg_scale = sample_cfg_scales(
            batch_size,
            cfg.cfg_min,
            cfg.cfg_max,
            device=self.device,
            sampling_power=cfg.cfg_sampling_power,
            no_cfg_fraction=cfg.no_cfg_fraction,
        )
        lambda_l1_null, progress = scheduled_null_l1_weight(
            self.step, cfg.max_steps, start=cfg.lambda_l1_null_start, end=cfg.lambda_l1_null_end
        )
        operator = SenseOperator(mask, coil_maps, device=self.device)
        spectral_scale = self.p_inverse.estimate_spectral_scale(operator, x_us)
        with torch.no_grad():
            # The anchor is conditioning data; alpha is learned through the
            # null-space projections of the loss.
            anchor = self.p_inverse.reconstruct(
                operator, kspace, reference=x_us, spectral_scale=spectral_scale
            )
        operator_flat = expand_operator(operator, mask, coil_maps, generations)
        anchor_flat = expand_generations(anchor, generations)
        kspace_flat = expand_generations(kspace, generations)
        labels_flat = expand_generations(labels, generations)
        cfg_flat = expand_generations(cfg_scale, generations)
        scale_flat = expand_generations(spectral_scale, generations)
        with torch.no_grad():
            model_input = build_model_input(
                anchor_flat,
                torch.randn_like(anchor_flat),
                operator_flat,
                self.p_inverse,
                spectral_scale=scale_flat,
                sigma=cfg.train_sigma,
            )

        with self._autocast():
            output_flat = self.model(model_input, kspace_flat, operator_flat, cfg_flat, labels_flat)
            output = group_generations(output_flat, batch_size, generations)
            null_generated = group_generations(
                self.p_inverse.null_project(operator_flat, output_flat, spectral_scale=scale_flat),
                batch_size,
                generations,
            )
            with torch.no_grad():
                null_target = self.p_inverse.null_project(operator, x_fs, spectral_scale=spectral_scale)
                # Mismatched negatives of condition i are z_ij = P_{N_i} x_j
                # (image-domain drift uses the raw targets x_j instead).
                null_negatives = (
                    self._operator_matched_negatives(x_fs, mask, coil_maps, spectral_scale)
                    if cfg.drift_space == "null"
                    else None
                )
            losses = driftrecon_objective(
                output,
                x_fs,
                null_generated,
                null_target,
                cfg_scale,
                temperatures=cfg.temperatures,
                drift_scale=cfg.drift_scale,
                lambda_drift=cfg.lambda_drift,
                lambda_l1=cfg.lambda_l1,
                lambda_mse=cfg.lambda_mse,
                lambda_l1_null=lambda_l1_null,
                dwt_levels=cfg.dwt_levels,
                cfg_temperature_beta=cfg.cfg_temperature_beta,
                null_negatives=null_negatives,
                drift_space=cfg.drift_space,
            )

        self.scaler.scale(losses["loss"]).backward()
        if cfg.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.p_inverse.parameters()),
                float(cfg.max_grad_norm),
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update(self.model)

        with torch.no_grad():
            first = reconstruct(x_us, output[:, 0].float(), operator)
            psnr = magnitude_psnr(first, x_fs).mean()
        result = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
        result.update(
            {
                "step": float(self.step),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "psnr_reconstruction": float(psnr.cpu()),
                "lambda_l1_null": float(lambda_l1_null),
                "null_l1_schedule_progress": float(progress),
                "cfg_scale_mean": float(cfg_scale.mean().cpu()),
                "h1_alpha": float(self.p_inverse.h1_alpha.detach().cpu()),
            }
        )
        self.step += 1
        return result

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "p_inverse": self.p_inverse.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "model_config": self.model.config_dict(),
                "p_inverse_config": self.p_inverse.config_dict(),
                "train_config": asdict(self.config),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint.")
        saved = state.get("train_config", {})
        for key in ("cfg_temperature_beta", "dwt_levels", "generations", "drift_space"):
            if key in saved and saved[key] != getattr(self.config, key):
                raise ValueError(
                    f"Resume mismatch for {key}: checkpoint={saved[key]!r}, "
                    f"training={getattr(self.config, key)!r}."
                )
        self.model.load_state_dict(state["model"])
        self.p_inverse.load_state_dict(state["p_inverse"])
        if state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        for name in ("optimizer", "scheduler", "scaler"):
            if state.get(name) is not None:
                getattr(self, name).load_state_dict(state[name])
        self.step = int(state.get("step", 0))


def load_for_inference(
    path: str | Path,
    device: torch.device | str,
    *,
    use_ema: bool = False,
) -> tuple[DriftReconMRI, H1CGPseudoinverse, dict]:
    """Rebuild the model (optionally with EMA weights) and ``A^dagger``."""
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: expected a {CHECKPOINT_FORMAT} checkpoint, got "
            f"{state.get('format')!r} (convert older checkpoints with "
            "tools/convert_legacy_checkpoint.py)."
        )
    model_config = dict(state["model_config"])
    model_config["use_checkpoint"] = False
    model = DriftReconMRI(**model_config).to(device)
    p_inverse = H1CGPseudoinverse(**state["p_inverse_config"]).to(device)
    load_model_weights(model, state, use_ema=use_ema)
    p_inverse.load_state_dict(state["p_inverse"])
    model.eval()
    return model, p_inverse, state
