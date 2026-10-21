"""Pre-build the M4Raw k-space / ESPIRiT / scale caches: ``python -m driftrecon.mri.build_cache``.

Training reads the pooled ``all``-contrast cache; validation and test read one
cache per contrast.  Caches are otherwise built lazily on first use.
"""

from __future__ import annotations

import argparse

from .data import CONTRASTS
from .data.m4raw import ESPIRIT_CROP, load_or_build_cache


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4raw-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--phases", default="train,val,test")
    parser.add_argument("--slices-per-volume", type=int, default=18)
    parser.add_argument("--espirit-device", default="auto", help="auto, cpu, 0 or cuda:N.")
    parser.add_argument("--espirit-crop", type=float, default=ESPIRIT_CROP)
    parser.add_argument("--force", action="store_true", help="Rebuild existing caches.")
    args = parser.parse_args(argv)

    for phase in [p.strip() for p in args.phases.split(",") if p.strip()]:
        if phase not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported phase {phase!r}.")
        contrasts = ("all",) if phase == "train" else CONTRASTS
        for contrast in contrasts:
            kspace, _, _, slice_ids = load_or_build_cache(
                cache_dir=args.cache_dir,
                data_root=args.m4raw_root,
                phase=phase,
                contrast=contrast,
                slices_per_volume=args.slices_per_volume,
                espirit_device=args.espirit_device,
                espirit_crop=args.espirit_crop,
                force_rebuild=args.force,
            )
            print(f"{phase}/{contrast}: {len(slice_ids)} slices, k-space {tuple(kspace.shape)}")


if __name__ == "__main__":
    main()
