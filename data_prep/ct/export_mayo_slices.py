#!/usr/bin/env python3
"""
Write the [0,1] patient volumes (prep_mayo.py convert output) as independent 2-D .npy slices with a
patient-wise train / val / test split.

    python export_mayo_slices.py ./mayo_npy ./mayo_slices --size 256

Output:
    mayo_slices/train/L067_0000.npy ...   (8 patients)
    mayo_slices/val/L333_0000.npy   ...   (1 patient)
    mayo_slices/test/L506_0000.npy  ...   (1 patient)

Values are in [0,1] (air = 0): do NOT pass --window-min/--window-max to driftrecon.ct.train. No circular mask by default
(the parallel-beam operator sees the whole square).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

SPLITS = dict(
    train=["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"],
    val=["L333"],
    test=["L506"],
)


def circle_mask(n):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    return ((xx**2 + yy**2) <= 1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="mayo_npy folder (prep_mayo.py convert output)")
    ap.add_argument("out", help="output folder; train/val/test are created inside")
    ap.add_argument("--size", type=int, default=256, help="slice size; area-averaged downsampling from 512")
    ap.add_argument("--mask", action="store_true", help="zero outside the inscribed circle")
    args = ap.parse_args()

    if args.size % 8:
        raise SystemExit("--size must be divisible by 8")
    m = circle_mask(args.size) if args.mask else None

    for split, pids in SPLITS.items():
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        n_total = 0
        for pid in pids:
            vol = np.load(os.path.join(args.root, f"{pid}.npy"), mmap_mode="r")
            for i in range(vol.shape[0]):
                x = torch.from_numpy(np.array(vol[i], dtype=np.float32))[None, None]
                if args.size != x.shape[-1]:
                    x = F.interpolate(x, size=(args.size, args.size), mode="area")
                if m is not None:
                    x = x * m
                np.save(os.path.join(d, f"{pid}_{i:04d}.npy"), x[0, 0].numpy().astype(np.float32))
            n_total += vol.shape[0]
            print(f"  {split}/{pid}: {vol.shape[0]} slices")
        print(f"{split}: {n_total} slices -> {d}")


if __name__ == "__main__":
    main()
