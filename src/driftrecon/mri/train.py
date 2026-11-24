"""Train DriftReconMRI on M4Raw: ``python -m driftrecon.mri.train``."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..core.cg import H1CGPseudoinverse
from ..core.cli import positive_float_tuple, resolve_device
from .data import M4RawDataset, collate_mri_batch
from .model import DriftReconMRI
from .trainer import DriftReconMRITrainer, TrainConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DriftReconMRI (multi-coil MRI, M4Raw).")
    parser.add_argument("--m4raw-root", required=True,
                        help="Folder with multicoil_train/val/test subfolders.")
    parser.add_argument("--cache-dir", required=True, help="k-space / coil-map cache directory.")
    parser.add_argument("--contrast", default="all", help="all, T1, T2 or FLAIR.")
    parser.add_argument("--slices-per-volume", type=int, default=18)
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--train-r", default="4,8", help="Comma-separated training accelerations.")
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay-step", type=int, default=10_000)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-rate", type=float, default=0.999)
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--use-checkpoint", action="store_true",
                        help="Activation checkpointing (saves memory).")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--krylov-steps", type=int, default=2)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--no-measurement-conditioning", action="store_true",
                        help="Ablation: remove the operator-aware (Krylov) measurement blocks.")
    parser.add_argument("--cg-steps", type=int, default=15)
    parser.add_argument("--power-iterations", type=int, default=20)
    parser.add_argument("--h1-alpha", type=float, default=0.5)
    parser.add_argument("--h1-floor", type=float, default=1e-6)
    parser.add_argument("--fixed-h1-alpha", action="store_true")
    parser.add_argument("--dwt-levels", type=int, default=0,
                        help="Haar levels of the drift features; 0 = full-image drift (MRI default).")
    parser.add_argument("--drift-space", choices=("null", "image"), default="null",
                        help="Ablation: drift on null-space components (DriftRecon) or on raw "
                             "model outputs (image domain).")
    parser.add_argument("--temperatures", type=positive_float_tuple, default=(1.0,),
                        help="Comma-separated base drift bandwidths tau_0.")
    parser.add_argument("--lambda-drift", type=float, default=1.0)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-mse", type=float, default=0.01)
    parser.add_argument("--lambda-l1-null-start", type=float, default=0.01)
    parser.add_argument("--lambda-l1-null-end", type=float, default=0.0)
    parser.add_argument("--cfg-min", type=float, default=1.0)
    parser.add_argument("--cfg-max", type=float, default=6.0)
    parser.add_argument("--cfg-sampling-power", type=float, default=3.0)
    parser.add_argument("--no-cfg-fraction", type=float, default=0.5)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--cfg-temperature-beta", type=float, default=0.25,
                        help="tau(cfg)=tau_0/(1+beta*(cfg-1)); zero gives a fixed bandwidth.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for CFG drift.")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    model = DriftReconMRI(
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
        weight_decay=args.weight_decay,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
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
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        ema_rate=args.ema_rate,
        use_fp16=not args.no_fp16,
    )
    trainer = DriftReconMRITrainer(model, p_inverse, config, device=device)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    # Re-seed after initialization so the stochastic training stream does not
    # depend on how many random numbers model construction consumed.
    seed_everything(args.seed + trainer.step)

    dataset = M4RawDataset(
        data_root=args.m4raw_root,
        cache_dir=args.cache_dir,
        phase="train",
        contrast=args.contrast,
        slices_per_volume=args.slices_per_volume,
        max_volumes=args.max_volumes,
        train_r_values=args.train_r,
        mask_seed=args.mask_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_mri_batch,
    )
    if len(loader) == 0:
        raise ValueError("Dataset is too small for a full batch.")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(
        {
            "device": str(device),
            "train_samples": len(dataset),
            "train_r": dataset.r_values,
            "model": model.config_dict(),
            "p_inverse": p_inverse.config_dict(),
            "train_config": config.__dict__,
        },
        indent=2,
        default=str,
    ))

    iterator = iter(loader)
    while trainer.step < args.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics = trainer.train_step(batch)
        step = trainer.step
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(metrics, sort_keys=True), flush=True)
        if step % args.save_every == 0:
            trainer.save_checkpoint(args.checkpoint_dir / f"model_{step:06d}.pth")
    trainer.save_checkpoint(args.checkpoint_dir / "model_final.pth")


if __name__ == "__main__":
    main()
