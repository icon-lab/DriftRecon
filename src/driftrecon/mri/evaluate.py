"""DriftReconMRI evaluation / checkpoint selection: ``python -m driftrecon.mri.evaluate``.

For every checkpoint, acceleration ``R`` and contrast, each slice of the
split is reconstructed from null-space inputs with noise level
``--input-noise-sigma``; ``--avg-samples N`` averages N reconstructions
(default 1, as in the paper) and ``--no-final-dc`` keeps the raw generator
output (image-domain ablation).

The per-slice magnitude PSNR and SSIM (``data_range = max |x_fs|``) are
averaged over slices for every ``R``.  With several checkpoints the best one
is the highest PSNR at ``--select-r`` (default 4), ties broken by SSIM.
``--save-pred-dir`` exports magnitude predictions for ``tools/mri_metrics.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..core import TEST_NULL_NOISE_SIGMA
from ..core.cli import resolve_device
from .data import CONTRASTS, M4RawDataset, collate_mri_batch, parse_r_values
from .physics import SenseOperator
from .reconstruction import (
    build_model_input,
    magnitude,
    magnitude_psnr,
    magnitude_ssim,
    reconstruct,
)
from .trainer import load_for_inference


@torch.no_grad()
def evaluate_split(model, p_inverse, loader, *, sigma, cfg_scale, seed, device, use_fp16,
                   samples=1, final_dc=True):
    if int(samples) <= 0:
        raise ValueError("samples must be positive.")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    psnrs, ssims, predictions, references, ids = [], [], [], [], []
    amp = use_fp16 and device.type == "cuda"
    for batch in loader:
        x_fs = batch["x_fs"].to(device)
        x_us = batch["x_us"].to(device)
        kspace = batch["y_us"].to(device)
        labels = batch["contrast_id"].to(device, dtype=torch.long)
        operator = SenseOperator(batch["mask"].to(device), batch["coil_maps"].to(device), device=device)
        spectral_scale = p_inverse.estimate_spectral_scale(operator, x_us)
        anchor = p_inverse.reconstruct(operator, kspace, reference=x_us, spectral_scale=spectral_scale)
        cfg = torch.full((x_fs.shape[0],), float(cfg_scale), device=device)
        total = torch.zeros_like(x_fs, dtype=torch.float32)
        for _ in range(int(samples)):
            noise = torch.randn(anchor.shape, device=device, dtype=anchor.dtype, generator=generator)
            model_input = build_model_input(
                anchor, noise, operator, p_inverse, spectral_scale=spectral_scale, sigma=sigma
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                output = model(model_input, kspace, operator, cfg, labels)
            output = output.float()
            total += reconstruct(x_us, output, operator) if final_dc else output
        prediction = total / int(samples)
        psnrs.append(magnitude_psnr(prediction, x_fs).cpu())
        ssims.append(magnitude_ssim(prediction, x_fs).cpu())
        predictions.append(magnitude(prediction).cpu().numpy())
        references.append(magnitude(x_fs).cpu().numpy())
        ids.extend(batch["sample_id"])
    return (
        torch.cat(psnrs).numpy(),
        torch.cat(ssims).numpy(),
        np.concatenate(predictions).astype(np.float32),
        np.concatenate(references).astype(np.float32),
        ids,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--m4raw-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--phase", choices=("val", "test"), default="val")
    parser.add_argument("--R", default="4,8", help="Comma-separated accelerations.")
    parser.add_argument("--select-r", type=float, default=4.0,
                        help="Acceleration whose PSNR (ties: SSIM) selects the best checkpoint.")
    parser.add_argument("--contrast", default="all", help="all, T1, T2 or FLAIR.")
    parser.add_argument("--slices-per-volume", type=int, default=18)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--input-noise-sigma", type=float, default=TEST_NULL_NOISE_SIGMA,
                        help="Null-space input noise level at inference.")
    parser.add_argument("--use-ema", action="store_true", help="Use the EMA weights.")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--avg-samples", type=int, default=1, help="N: number of averaged samples.")
    parser.add_argument("--no-final-dc", action="store_true",
                        help="Evaluate the raw generator output (skip A^H y + (I - A^H A) h).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True, help="Summary / metrics directory.")
    parser.add_argument("--save-pred-dir", type=Path, default=None,
                        help="Export <dir>/<checkpoint>/R<R>/<contrast>.npz predictions.")
    args = parser.parse_args(argv)
    if args.input_noise_sigma < 0:
        raise ValueError("--input-noise-sigma must be nonnegative.")
    if args.avg_samples <= 0:
        raise ValueError("--avg-samples must be positive.")

    device = resolve_device(args.device)
    r_values = parse_r_values(args.R)
    select_r = float(args.select_r)
    if not any(abs(float(R) - select_r) < 1e-9 for R in r_values):
        raise ValueError(f"--select-r {select_r:g} is not among --R {list(r_values)}.")
    contrasts = CONTRASTS if args.contrast == "all" else (args.contrast,)
    loaders = {}
    for R in r_values:
        for contrast in contrasts:
            dataset = M4RawDataset(
                data_root=args.m4raw_root,
                cache_dir=args.cache_dir,
                phase=args.phase,
                contrast=contrast,
                slices_per_volume=args.slices_per_volume,
                R=R,
                mask_seed=args.mask_seed,
            )
            loaders[(R, contrast)] = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_mri_batch,
            )
    args.out.mkdir(parents=True, exist_ok=True)
    weights = "ema" if args.use_ema else "model"
    print(
        f"{len(args.checkpoint)} checkpoint(s), phase={args.phase}, R={list(r_values)}, "
        f"contrasts={list(contrasts)}, sigma={args.input_noise_sigma:g}, N={args.avg_samples}, "
        f"final DC={not args.no_final_dc}, weights={weights}"
    )

    rows = []
    for path in sorted(args.checkpoint):
        model, p_inverse, state = load_for_inference(path, device, use_ema=args.use_ema)
        per_r, per_r_ssim = {}, {}
        for R in r_values:
            psnr_r, ssim_r = [], []
            for contrast in contrasts:
                psnr, ssim, prediction, reference, ids = evaluate_split(
                    model, p_inverse, loaders[(R, contrast)],
                    sigma=args.input_noise_sigma, cfg_scale=args.cfg_scale,
                    seed=args.seed, device=device, use_fp16=not args.no_fp16,
                    samples=args.avg_samples, final_dc=not args.no_final_dc,
                )
                psnr_r.append(psnr)
                ssim_r.append(ssim)
                print(f"  {path.stem} R={R:g} {contrast}: {psnr.mean():.3f} dB, "
                      f"SSIM {ssim.mean():.4f} ({len(psnr)} slices)")
                if args.save_pred_dir is not None:
                    out_dir = args.save_pred_dir / path.stem / f"R{R:g}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        out_dir / f"{contrast}.npz",
                        pred=prediction, gt=reference, ids=np.array(ids), psnr=psnr, ssim=ssim,
                    )
            per_r[R] = float(np.concatenate(psnr_r).mean())
            per_r_ssim[R] = float(np.concatenate(ssim_r).mean())
        selected = next(R for R in r_values if abs(float(R) - select_r) < 1e-9)
        rows.append((path.stem, int(state.get("step", -1)), per_r, per_r_ssim, selected))
        print(f"{path.stem}: " + " ".join(
            f"(R{R:g} {per_r[R]:.3f} dB / {per_r_ssim[R]:.4f})" for R in r_values))

    with open(args.out / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "step",
                         *[f"psnr_R{R:g}" for R in r_values], *[f"ssim_R{R:g}" for R in r_values]])
        for name, step, per_r, per_r_ssim, _ in rows:
            writer.writerow([name, step, *[f"{per_r[R]:.4f}" for R in r_values],
                             *[f"{per_r_ssim[R]:.5f}" for R in r_values]])
    best = max(rows, key=lambda row: (round(row[2][row[4]], 4), row[3][row[4]]))
    selected = best[4]
    (args.out / "best.json").write_text(json.dumps({
        "checkpoint": best[0],
        "select_r": selected,
        "psnr": best[2][selected],
        "ssim": best[3][selected],
    }, indent=2))
    if len(rows) > 1:
        print(f"best (R{selected:g} PSNR, ties by SSIM): {best[0]} "
              f"({best[2][selected]:.3f} dB, SSIM {best[3][selected]:.4f})")


if __name__ == "__main__":
    main()
