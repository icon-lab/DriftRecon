"""DriftReconCT inference on a folder of slices: ``python -m driftrecon.ct.test``.

For every slice the sparse-view sinogram ``y = A x`` is simulated with the
checkpoint geometry, the anchor ``A^dagger y`` is computed, and one
reconstruction ``A^dagger y + (I - A^dagger A) h`` is produced from a single
null-space input with noise level ``--input-noise-sigma``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..core import TEST_NULL_NOISE_SIGMA
from .data import CTSliceDataset
from .reconstruction import build_model_input, reconstruct
from ..core.cli import resolve_device
from .trainer import load_for_inference


@torch.no_grad()
def run(operator, model, p_inverse, loader, *, cfg_scale, sigma, seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    spectral = None
    predictions, psnrs = [], []
    for target in loader:
        target = target.to(device, dtype=torch.float32)
        batch = target.shape[0]
        sinogram = operator.forward(target)
        if spectral is None:
            spectral = p_inverse.estimate_spectral_scale(operator, target[:1])
        scale = spectral.expand(batch)
        anchor = p_inverse.reconstruct(operator, sinogram, reference=target, spectral_scale=scale)
        cfg = torch.full((batch,), float(cfg_scale), device=device)
        noise = torch.randn(anchor.shape, device=device, dtype=anchor.dtype, generator=generator)
        model_input = build_model_input(
            anchor, noise, operator, p_inverse, spectral_scale=scale, sigma=sigma
        )
        output = model(model_input, sinogram, operator, cfg)
        prediction = reconstruct(anchor, output, operator, p_inverse, spectral_scale=scale)
        prediction = prediction.clamp(0, 1.0)
        mse = ((prediction - target) ** 2).flatten(1).mean(1).clamp_min(1e-20)
        psnrs += (10 * torch.log10(1.0 / mse)).tolist()
        predictions.append(prediction[:, 0].float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32), float(np.mean(psnrs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DriftReconCT inference / checkpoint selection.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input-noise-sigma", type=float, default=TEST_NULL_NOISE_SIGMA,
                        help="Null-space input noise level at inference.")
    parser.add_argument("--use-ema", action="store_true", help="Use the EMA weights.")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.input_noise_sigma < 0:
        raise ValueError("--input-noise-sigma must be nonnegative.")

    device = resolve_device(args.device)
    checkpoints = sorted(p for p in args.checkpoint if p.suffix == ".pth")
    if not checkpoints:
        raise SystemExit("no checkpoint found")
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    image_shape = tuple(int(v) for v in first["operator_config"]["image_shape"])
    del first

    dataset = CTSliceDataset(args.data_dir, image_size=image_shape)
    names = [p.stem for p in dataset.paths]
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    args.out.mkdir(parents=True, exist_ok=True)
    weights = "ema" if args.use_ema else "model"
    print(
        f"{len(checkpoints)} checkpoint(s) x {len(names)} slices ({args.data_dir}), "
        f"sigma = {args.input_noise_sigma:g}, weights = {weights}, device = {device}"
    )

    rows = []
    for path in checkpoints:
        operator, model, p_inverse, state = load_for_inference(path, device, use_ema=args.use_ema)
        prediction, psnr = run(
            operator, model, p_inverse, loader,
            cfg_scale=args.cfg_scale, sigma=args.input_noise_sigma, seed=args.seed, device=device,
        )
        out_dir = args.out / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"pred_cfg{args.cfg_scale:g}.npy", prediction)
        (out_dir / "slice_names.txt").write_text("\n".join(names))
        step = int(state.get("step", -1))
        rows.append((path.stem, step, psnr))
        print(f"{path.stem}: PSNR[0,1] = {psnr:.3f} dB -> {out_dir}")

    with open(args.out / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "step", "psnr_01"])
        writer.writerows(rows)
    if len(rows) > 1:
        best = max(rows, key=lambda row: row[2])
        print(f"best: {best[0]} ({best[2]:.3f} dB)")


if __name__ == "__main__":
    main()
