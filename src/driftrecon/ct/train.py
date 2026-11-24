"""Train DriftReconCT: ``python -m driftrecon.ct.train``."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..core.cg import H1CGPseudoinverse
from ..core.cli import positive_float_tuple, resolve_device
from .data import CTSliceDataset, SyntheticPhantomDataset
from .model import DriftReconCT
from .physics import ParallelBeamCT
from .trainer import DriftReconCTTrainer, TrainConfig


def window_from_args(window_min, window_max):
    if window_min is None and window_max is None:
        return None
    if window_min is None or window_max is None:
        raise ValueError("--window-min and --window-max must be supplied together.")
    return (float(window_min), float(window_max))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DriftReconCT (sparse-view CT).")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory of 2-D slices; omitted = synthetic phantoms.")
    parser.add_argument("--array-key", default="image")
    parser.add_argument("--window-min", type=float, default=None)
    parser.add_argument("--window-max", type=float, default=None)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--views", type=int, default=58)
    parser.add_argument("--detectors", type=int, default=None,
                        help="Detector bins; omitted = image diagonal + 1 (LoDoPaB recipe: 498).")
    parser.add_argument("--angles-endpoint", action="store_true",
                        help="Angles uniformly spaced over [0, pi] with both end points (LoDoPaB "
                             "geometry); default [0, pi).")
    parser.add_argument("--augment", action="store_true",
                        help="Random horizontal/vertical flips of the training slices.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=120_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("none", "cosine", "step"), default="step",
                        help="Default: halve the lr every --lr-decay-step steps.")
    parser.add_argument("--lr-decay-step", type=int, default=20_000,
                        help="Step schedule: multiply the lr by --lr-decay-gamma every N steps.")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Cosine schedule end value.")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--blocks-per-stage", type=int, default=4)
    parser.add_argument("--krylov-steps", type=int, default=2)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--no-measurement-conditioning", action="store_true",
                        help="Ablation: remove the operator-aware (Krylov) measurement blocks.")
    parser.add_argument("--use-checkpoint", action="store_true",
                        help="Activation checkpointing (saves memory).")
    parser.add_argument("--cg-steps", type=int, default=15)
    parser.add_argument("--power-iterations", type=int, default=20)
    parser.add_argument("--h1-alpha", type=float, default=0.5)
    parser.add_argument("--h1-floor", type=float, default=1e-6)
    parser.add_argument("--fixed-h1-alpha", action="store_true")
    parser.add_argument("--dwt-levels", type=int, default=3)
    parser.add_argument("--drift-space", choices=("null", "image"), default="null",
                        help="Ablation: drift on null-space components (DriftRecon) or on raw "
                             "model outputs (image domain).")
    parser.add_argument("--temperatures", type=positive_float_tuple, default=(0.2, 1.0),
                        help="Comma-separated base drift bandwidths tau_0.")
    parser.add_argument("--lambda-drift", type=float, default=1.0)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-mse", type=float, default=1.0)
    parser.add_argument("--lambda-l1-null-start", type=float, default=0.01)
    parser.add_argument("--lambda-l1-null-end", type=float, default=0.0)
    parser.add_argument("--cfg-min", type=float, default=1.0)
    parser.add_argument("--cfg-max", type=float, default=6.0)
    parser.add_argument("--cfg-sampling-power", type=float, default=3.0)
    parser.add_argument("--no-cfg-fraction", type=float, default=0.5)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--cfg-temperature-beta", type=float, default=0.25,
                        help="tau(cfg)=tau_0/(1+beta*(cfg-1)); zero gives a fixed bandwidth.")
    parser.add_argument("--ema-rate", type=float, default=0.999)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    if args.image_size % 8:
        raise ValueError("--image-size must be divisible by 8.")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for CFG drift.")
    if args.log_every <= 0 or args.save_every <= 0:
        raise ValueError("--log-every and --save-every must be positive.")
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)
    image_shape = (int(args.image_size), int(args.image_size))
    if args.data_dir is None:
        dataset = SyntheticPhantomDataset(
            count=max(8, int(args.batch_size) * 2), image_size=args.image_size
        )
        data_description = "deterministic synthetic phantoms"
    else:
        dataset = CTSliceDataset(
            args.data_dir,
            image_size=image_shape,
            window=window_from_args(args.window_min, args.window_max),
            array_key=args.array_key,
            max_slices=args.max_slices,
            augment=args.augment,
        )
        data_description = str(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
    )
    if len(loader) == 0:
        raise ValueError("Dataset is too small for a full CFG batch.")

    angles = (
        torch.linspace(0.0, math.pi, int(args.views)).tolist() if args.angles_endpoint else None
    )
    operator = ParallelBeamCT(
        image_shape, num_views=args.views, num_detectors=args.detectors, angles=angles
    )
    model = DriftReconCT(
        base_channels=args.base_channels,
        blocks_per_stage=args.blocks_per_stage,
        krylov_steps=args.krylov_steps,
        cond_dim=args.cond_dim,
        use_checkpoint=args.use_checkpoint,
        measurement_conditioning=not args.no_measurement_conditioning,
    )
    p_inverse = H1CGPseudoinverse(
        cg_steps=args.cg_steps,
        h1_alpha_init=args.h1_alpha,
        h1_floor=args.h1_floor,
        power_iterations=args.power_iterations,
        learnable_alpha=not args.fixed_h1_alpha,
    )
    config = TrainConfig(
        max_steps=args.max_steps,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
        lr_min=args.lr_min,
        cfg_min=args.cfg_min,
        cfg_max=args.cfg_max,
        cfg_sampling_power=args.cfg_sampling_power,
        no_cfg_fraction=args.no_cfg_fraction,
        generations=args.generations,
        cfg_temperature_beta=args.cfg_temperature_beta,
        temperatures=args.temperatures,
        lambda_drift=args.lambda_drift,
        lambda_l1=args.lambda_l1,
        lambda_mse=args.lambda_mse,
        lambda_l1_null_start=args.lambda_l1_null_start,
        lambda_l1_null_end=args.lambda_l1_null_end,
        dwt_levels=args.dwt_levels,
        drift_space=args.drift_space,
        ema_rate=args.ema_rate,
    )
    trainer = DriftReconCTTrainer(model, operator, p_inverse, config, device=device)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "data": data_description,
                "augment": bool(args.augment),
                "samples": len(dataset),
                "operator": {k: v for k, v in operator.geometry_dict().items() if k != "angles"},
                "model": model.config_dict(),
                "p_inverse": p_inverse.config_dict(),
                "train_config": {
                    k: v for k, v in config.__dict__.items()
                },
            },
            indent=2,
            default=str,
        )
    )

    iterator = iter(loader)
    while trainer.step < int(args.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics = trainer.train_step(batch)
        completed_step = trainer.step
        if completed_step == 1 or completed_step % int(args.log_every) == 0:
            print(json.dumps(metrics, sort_keys=True))
        if completed_step % int(args.save_every) == 0:
            trainer.save_checkpoint(args.checkpoint_dir / f"model_{completed_step:06d}.pth")
    trainer.save_checkpoint(args.checkpoint_dir / "model_final.pth")


if __name__ == "__main__":
    main()
