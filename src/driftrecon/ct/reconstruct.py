"""Reconstruct one CT slice or measured sinogram: ``python -m driftrecon.ct.reconstruct``."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from ..core import TEST_NULL_NOISE_SIGMA
from .data import load_ct_sinogram, load_ct_slice
from .reconstruction import build_model_input, reconstruct
from ..core.cli import resolve_device
from .train import window_from_args
from .trainer import load_for_inference


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reconstruct one sparse-view CT slice with DriftReconCT.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path,
                        help="Reference slice; its sinogram is simulated with the saved geometry.")
    source.add_argument("--sinogram", type=Path, help="Measured (views, detectors) .npy/.npz sinogram.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-key", default="image")
    parser.add_argument("--sinogram-key", default="sinogram")
    parser.add_argument("--window-min", type=float, default=None)
    parser.add_argument("--window-max", type=float, default=None)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--input-noise-sigma", type=float, default=TEST_NULL_NOISE_SIGMA)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--samples", type=int, default=1, help="Number of averaged samples.")
    parser.add_argument("--no-final-dc", action="store_true",
                        help="Return the raw generator output (skip A^dagger y + P_N h).")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 1.0:
        raise ValueError("--cfg-scale must be finite and at least 1.")
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")

    device = resolve_device(args.device)
    operator, model, p_inverse, _ = load_for_inference(args.checkpoint, device, use_ema=args.use_ema)
    target = None
    if args.input is not None:
        target = load_ct_slice(
            args.input,
            array_key=args.array_key,
            window=window_from_args(args.window_min, args.window_max),
            image_size=operator.image_shape,
        ).unsqueeze(0).to(device)
        sinogram = operator.forward(target)
        reference = target
    else:
        sinogram = load_ct_sinogram(
            args.sinogram,
            array_key=args.sinogram_key,
            expected_shape=(operator.num_views, operator.num_detectors),
        ).to(device)
        reference = sinogram.new_zeros(1, 1, *operator.image_shape)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.noise_seed))
    with torch.no_grad():
        spectral_scale = p_inverse.estimate_spectral_scale(operator, reference)
        anchor = p_inverse.reconstruct(
            operator, sinogram, reference=reference, spectral_scale=spectral_scale
        )
        cfg_scale = torch.full((1,), float(args.cfg_scale), device=device)
        outputs, reconstructions = [], []
        for _ in range(args.samples):
            noise = torch.randn(anchor.shape, device=device, dtype=anchor.dtype, generator=generator)
            model_input = build_model_input(
                anchor, noise, operator, p_inverse,
                spectral_scale=spectral_scale, sigma=args.input_noise_sigma,
            )
            output = model(model_input, sinogram, operator, cfg_scale)
            outputs.append(output)
            reconstructions.append(
                output.float() if args.no_final_dc
                else reconstruct(anchor, output, operator, p_inverse, spectral_scale=spectral_scale)
            )
        output = torch.stack(outputs).mean(0)
        reconstruction = torch.stack(reconstructions).mean(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sinogram": sinogram.cpu().numpy(),
        "anchor": anchor.cpu().numpy(),
        "model_output": output.cpu().numpy(),
        "reconstruction": reconstruction.cpu().numpy(),
        "cfg_scale": np.float32(args.cfg_scale),
        "input_noise_sigma": np.float32(args.input_noise_sigma),
        "samples": np.int32(args.samples),
    }
    if target is not None:
        payload["target"] = target.cpu().numpy()
    np.savez_compressed(args.output, **payload)
    print(f"saved reconstruction to {args.output}")


if __name__ == "__main__":
    main()
