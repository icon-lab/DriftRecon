"""DriftReconCT training engine and checkpoint I/O."""

from __future__ import annotations

import math
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
from .model import DriftReconCT
from .physics import ParallelBeamCT
from .reconstruction import build_model_input, reconstruct

CHECKPOINT_FORMAT = "driftrecon_ct_v1"


@dataclass
class TrainConfig:
    max_steps: int = 100_000
    lr: float = 1e-4
    weight_decay: float = 0.0
    cfg_min: float = 1.0
    cfg_max: float = 4.0
    cfg_sampling_power: float = 3.0
    no_cfg_fraction: float = 0.5
    generations: int = 2
    cfg_temperature_beta: float = 0.5
    train_sigma: float = TRAIN_NULL_NOISE_SIGMA
    temperatures: tuple[float, ...] = (0.2, 1.0)
    drift_scale: float = 1.0
    lambda_drift: float = 1.0
    lambda_l1: float = 0.0
    lambda_mse: float = 1.0
    lambda_l1_null_start: float = 0.01
    lambda_l1_null_end: float = 0.0
    dwt_levels: int = 3
    max_grad_norm: Optional[float] = 1.0
    ema_rate: float = 0.999

    def __post_init__(self) -> None:
        if int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive.")
        if int(self.generations) < 2:
            raise ValueError("CFG drift requires generations >= 2.")
        if not math.isfinite(float(self.train_sigma)) or float(self.train_sigma) < 0:
            raise ValueError("train_sigma must be finite and nonnegative.")
        if not 1.0 <= float(self.cfg_min) <= float(self.cfg_max):
            raise ValueError("Expected 1 <= cfg_min <= cfg_max.")
        if not math.isfinite(float(self.cfg_temperature_beta)) or float(
            self.cfg_temperature_beta
        ) < 0.0:
            raise ValueError("cfg_temperature_beta must be finite and nonnegative.")
        if not self.temperatures or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.temperatures
        ):
            raise ValueError("temperatures must contain finite positive values.")
        if int(self.dwt_levels) < 0:
            raise ValueError("dwt_levels must be nonnegative.")
        if not 0.0 <= float(self.ema_rate) < 1.0:
            raise ValueError("ema_rate must be in [0, 1).")
        weights = (
            self.lambda_drift,
            self.lambda_l1,
            self.lambda_mse,
            self.lambda_l1_null_start,
            self.lambda_l1_null_end,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in weights):
            raise ValueError("Loss weights must be finite and nonnegative.")


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction.float() - target.float()).square())
    return float((-10.0 * torch.log10(mse.clamp_min(1e-12))).detach().cpu())


def validate_cfg_temperature_beta_resume(
    saved_train_config: dict,
    current_beta: float,
) -> float:
    """Reject resuming a checkpoint with a different bandwidth beta."""
    saved_beta = float(saved_train_config.get("cfg_temperature_beta", 0.0))
    current_beta = float(current_beta)
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (saved_beta, current_beta)
    ):
        raise ValueError(
            "Checkpoint and training cfg_temperature_beta values must be "
            "finite and nonnegative."
        )
    if not math.isclose(saved_beta, current_beta, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Checkpoint resume mismatch for cfg_temperature_beta: "
            f"checkpoint={saved_beta!r}, training={current_beta!r}."
        )
    return saved_beta


class DriftReconCTTrainer:
    """CFG drift on CG null components + image fidelity, with EMA tracking."""

    def __init__(
        self,
        model: DriftReconCT,
        operator: ParallelBeamCT,
        p_inverse: H1CGPseudoinverse,
        config: TrainConfig,
        *,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.operator = operator.to(self.device)
        self.p_inverse = p_inverse.to(self.device)
        self.config = config
        self.step = 0
        parameter_groups: list[dict] = [
            {
                "params": [
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ],
                "weight_decay": float(config.weight_decay),
            }
        ]
        inverse_parameters = [
            parameter
            for parameter in self.p_inverse.parameters()
            if parameter.requires_grad
        ]
        if inverse_parameters:
            # Weight decay would bias the H1 weight; it is learned from the
            # loss alone.
            parameter_groups.append(
                {"params": inverse_parameters, "weight_decay": 0.0}
            )
        self.optimizer = torch.optim.AdamW(parameter_groups, lr=float(config.lr))
        self.ema = EMA(self.model, rate=config.ema_rate)

    def _spectral_scale(self, images: torch.Tensor) -> torch.Tensor:
        # The geometry is shared by the whole batch, so L is estimated once.
        single = self.p_inverse.estimate_spectral_scale(self.operator, images[:1])
        return single.expand(images.shape[0])

    def train_step(self, images: torch.Tensor) -> dict[str, float]:
        images = images.to(self.device, dtype=torch.float32)
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Training images must have shape (B,1,H,W).")
        if images.shape[0] < 2:
            raise ValueError("CFG drift requires a local batch of at least 2.")
        if tuple(images.shape[-2:]) != self.operator.image_shape:
            raise ValueError("Training image and operator sizes differ.")

        batch_size = images.shape[0]
        generations = int(self.config.generations)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        sinogram = self.operator.forward(images)
        spectral_scale = self._spectral_scale(images)
        with torch.no_grad():
            anchor = self.p_inverse.reconstruct(
                self.operator,
                sinogram,
                reference=images,
                spectral_scale=spectral_scale,
            )
        cfg_scale = sample_cfg_scales(
            batch_size,
            self.config.cfg_min,
            self.config.cfg_max,
            device=self.device,
            sampling_power=self.config.cfg_sampling_power,
            no_cfg_fraction=self.config.no_cfg_fraction,
        )
        anchor_flat = expand_generations(anchor, generations)
        sinogram_flat = expand_generations(sinogram, generations)
        cfg_flat = expand_generations(cfg_scale, generations)
        scale_flat = expand_generations(spectral_scale, generations)
        noise = torch.randn_like(anchor_flat)
        with torch.no_grad():
            model_input = build_model_input(
                anchor_flat,
                noise,
                self.operator,
                self.p_inverse,
                spectral_scale=scale_flat,
                sigma=self.config.train_sigma,
            )
        model_output_flat = self.model(
            model_input, sinogram_flat, self.operator, cfg_flat
        )
        model_output = group_generations(model_output_flat, batch_size, generations)
        null_generated_flat = self.p_inverse.null_project(
            self.operator,
            model_output_flat,
            spectral_scale=scale_flat,
        )
        null_generated = group_generations(null_generated_flat, batch_size, generations)
        with torch.no_grad():
            null_target = self.p_inverse.null_project(
                self.operator,
                images,
                spectral_scale=spectral_scale,
            )
        lambda_l1_null, schedule_progress = scheduled_null_l1_weight(
            self.step,
            self.config.max_steps,
            start=self.config.lambda_l1_null_start,
            end=self.config.lambda_l1_null_end,
        )
        losses = driftrecon_objective(
            model_output,
            images,
            null_generated,
            null_target,
            cfg_scale,
            temperatures=self.config.temperatures,
            drift_scale=self.config.drift_scale,
            lambda_drift=self.config.lambda_drift,
            lambda_l1=self.config.lambda_l1,
            lambda_mse=self.config.lambda_mse,
            lambda_l1_null=lambda_l1_null,
            dwt_levels=self.config.dwt_levels,
            cfg_temperature_beta=self.config.cfg_temperature_beta,
        )
        losses["loss"].backward()
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.p_inverse.parameters()),
                float(self.config.max_grad_norm),
            )
        self.optimizer.step()
        self.ema.update(self.model)

        with torch.no_grad():
            first_reconstruction = reconstruct(
                anchor,
                model_output[:, 0],
                self.operator,
                self.p_inverse,
                spectral_scale=spectral_scale,
            )
            data_residual = torch.linalg.vector_norm(
                self.operator.forward(first_reconstruction) - sinogram
            ) / torch.linalg.vector_norm(sinogram).clamp_min(1e-12)
        result = {key: float(value.detach().cpu()) for key, value in losses.items()}
        temperature_factors = (
            1.0
            + float(self.config.cfg_temperature_beta)
            * (cfg_scale.detach().float() - 1.0)
        ).reciprocal()
        result.update(
            {
                "step": float(self.step),
                "psnr_model_output": psnr(model_output[:, 0], images),
                "psnr_reconstruction": psnr(first_reconstruction, images),
                "relative_data_residual": float(data_residual.cpu()),
                "cfg_scale_min": float(cfg_scale.min().cpu()),
                "cfg_scale_mean": float(cfg_scale.mean().cpu()),
                "cfg_scale_max": float(cfg_scale.max().cpu()),
                "lambda_l1_null": float(lambda_l1_null),
                "null_l1_schedule_progress": float(schedule_progress),
                "h1_alpha": float(self.p_inverse.h1_alpha.detach().cpu()),
                "cfg_temperature_factor_min": float(temperature_factors.min().cpu()),
                "cfg_temperature_factor_mean": float(temperature_factors.mean().cpu()),
                "cfg_temperature_factor_max": float(temperature_factors.max().cpu()),
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
                "model_config": self.model.config_dict(),
                "p_inverse_config": self.p_inverse.config_dict(),
                "operator_config": self.operator.geometry_dict(),
                "train_config": asdict(self.config),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint.")
        validate_cfg_temperature_beta_resume(
            state.get("train_config", {}), self.config.cfg_temperature_beta
        )
        saved_operator = state.get("operator_config", {})
        current_operator = self.operator.geometry_dict()
        for key in (
            "image_shape",
            "num_views",
            "num_detectors",
            "pixel_size",
            "detector_spacing",
        ):
            if tuple_or_value(saved_operator.get(key)) != tuple_or_value(current_operator.get(key)):
                raise ValueError(
                    f"Checkpoint CT geometry mismatch for {key}: "
                    f"saved={saved_operator.get(key)!r}, "
                    f"current={current_operator.get(key)!r}."
                )
        saved_angles = torch.as_tensor(saved_operator.get("angles", []))
        current_angles = torch.as_tensor(current_operator["angles"])
        if saved_angles.shape != current_angles.shape or not torch.allclose(
            saved_angles, current_angles, rtol=0.0, atol=1e-7
        ):
            raise ValueError("Checkpoint CT projection angles do not match.")
        saved_model = state.get("model_config", {})
        current_model = self.model.config_dict()
        for key in ("base_channels", "blocks_per_stage", "krylov_steps", "cond_dim"):
            if saved_model.get(key) != current_model.get(key):
                raise ValueError(f"Checkpoint model configuration mismatch for {key}.")
        saved_inverse = state.get("p_inverse_config", {})
        current_inverse = self.p_inverse.config_dict()
        for key in ("cg_steps", "h1_floor", "power_iterations", "learnable_alpha"):
            if saved_inverse.get(key) != current_inverse.get(key):
                raise ValueError(
                    f"Checkpoint pseudoinverse configuration mismatch for {key}."
                )
        self.model.load_state_dict(state["model"])
        self.p_inverse.load_state_dict(state["p_inverse"])
        if state.get("ema"):
            self.ema.load_state_dict(state["ema"])
        else:
            self.ema = EMA(self.model, rate=self.config.ema_rate)
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state.get("step", 0))


def tuple_or_value(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value


def load_for_inference(
    path: str | Path,
    device: torch.device | str,
    *,
    use_ema: bool = False,
) -> tuple[ParallelBeamCT, DriftReconCT, H1CGPseudoinverse, dict]:
    """Rebuild operator, model (optionally with EMA weights) and ``A^dagger``."""
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path}: expected a {CHECKPOINT_FORMAT} checkpoint, got "
            f"{state.get('format')!r} (convert older checkpoints with "
            "tools/convert_legacy_checkpoint.py)."
        )
    operator = ParallelBeamCT(**state["operator_config"]).to(device)
    model_config = dict(state["model_config"])
    model_config["use_checkpoint"] = False
    model = DriftReconCT(**model_config).to(device)
    p_inverse = H1CGPseudoinverse(**state["p_inverse_config"]).to(device)
    load_model_weights(model, state, use_ema=use_ema)
    p_inverse.load_state_dict(state["p_inverse"])
    model.eval()
    return operator, model, p_inverse, state
