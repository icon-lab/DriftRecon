#!/usr/bin/env python3
"""Convert the LoDoPaB-CT reference images into one ``.npy`` file per slice.

Each split folder of the LoDoPaB-CT download holds ``image_fs.npz`` (key
``image_fs``, shape ``(N, 1, 352, 352)``, attenuation normalized to [0, 1]).
Slices are written in their original order as ``<out>/<split>/00000.npy``,
``00001.npy``, ...  Existing split folders that already contain slices are
left untouched.

    python data_prep/ct/prep_lodopab.py /path/to/lodopab-ct data/lodopab
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="LoDoPaB-CT folder with {train,val,test}/image_fs.npz.")
    parser.add_argument("target", type=Path, help="Output folder for {train,val,test}/*.npy.")
    parser.add_argument("--splits", default=",".join(SPLITS))
    args = parser.parse_args()
    for split in (s.strip() for s in args.splits.split(",") if s.strip()):
        out = args.target / split
        existing = sorted(out.glob("*.npy")) if out.exists() else []
        if existing:
            print(f"{split}: already exists ({len(existing)} slices)")
            continue
        source = args.source / split / "image_fs.npz"
        if not source.is_file():
            raise FileNotFoundError(f"missing {source}")
        out.mkdir(parents=True, exist_ok=True)
        images = np.load(source)["image_fs"][:, 0].astype(np.float32)
        for index, image in enumerate(images):
            np.save(out / f"{index:05d}.npy", image)
        print(f"{split}: {len(images)} slices, range [{images.min():.3f}, {images.max():.3f}] -> {out}")


if __name__ == "__main__":
    main()
