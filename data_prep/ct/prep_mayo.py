#!/usr/bin/env python3
"""
AAPM/Mayo 2016 Low Dose CT -- inventory the full-dose 1 mm .IMA files and convert each patient to a
[0,1] float32 .npy volume.

Usage:
  python prep_mayo.py inventory /path/to/full_1mm
  python prep_mayo.py convert   /path/to/full_1mm ./mayo_npy --every 1

Normalization (fixed for the whole dataset):
  HU = RescaleSlope * pixel + RescaleIntercept
  x  = clip((HU - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)     -> air = 0
Inverse: HU = x * (HU_MAX - HU_MIN) + HU_MIN
"""
import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import pydicom

HU_MIN, HU_MAX = -1000.0, 2000.0


# ----------------------------------------------------------------------------
# File collection / metadata
# ----------------------------------------------------------------------------
def find_files(patient_dir):
    files = []
    for root, _, fnames in os.walk(patient_dir):
        for f in fnames:
            if f.lower().endswith((".ima", ".dcm")):
                files.append(os.path.join(root, f))
    return sorted(files)


def read_meta(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    z = float(ipp[2]) if ipp is not None and len(ipp) == 3 else float("nan")
    spacing = getattr(ds, "PixelSpacing", None)
    spacing = tuple(float(v) for v in spacing) if spacing is not None else (float("nan"),) * 2
    return dict(
        path=path,
        sop=str(getattr(ds, "SOPInstanceUID", "")),
        series=str(getattr(ds, "SeriesInstanceUID", "")),
        desc=str(getattr(ds, "SeriesDescription", "")),
        kernel=str(getattr(ds, "ConvolutionKernel", "")),
        thick=float(getattr(ds, "SliceThickness", float("nan"))),
        inst=int(getattr(ds, "InstanceNumber", -1)),
        z=z,
        rows=int(getattr(ds, "Rows", -1)),
        cols=int(getattr(ds, "Columns", -1)),
        slope=float(getattr(ds, "RescaleSlope", 1.0)),
        intercept=float(getattr(ds, "RescaleIntercept", 0.0)),
        spacing=spacing,
    )


def collect(patient_dir):
    """Read all files and drop duplicates by SOPInstanceUID."""
    metas = [read_meta(p) for p in find_files(patient_dir)]
    seen, uniq, dups = set(), [], 0
    for m in metas:
        key = m["sop"] or m["path"]
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        uniq.append(m)
    return uniq, dups


def pick_series(metas):
    """If several series exist: keep 1 mm + B30 with the most slices."""
    by_series = defaultdict(list)
    for m in metas:
        by_series[m["series"]].append(m)
    if len(by_series) <= 1:
        return metas, None

    def score(items):
        d = (items[0]["desc"] + " " + items[0]["kernel"]).upper()
        is_1mm = ("1MM" in d.replace(" ", "")) or abs(items[0]["thick"] - 1.0) < 1e-3
        return (is_1mm, "B30" in d, len(items))

    best = max(by_series.values(), key=score)
    return best, {k[-12:]: len(v) for k, v in by_series.items()}


def sort_slices(metas):
    if metas and all(np.isfinite(m["z"]) for m in metas):
        return sorted(metas, key=lambda m: m["z"])
    return sorted(metas, key=lambda m: m["inst"])


def patient_dirs(root):
    return sorted(d for d in glob.glob(os.path.join(root, "L*")) if os.path.isdir(d))


# ----------------------------------------------------------------------------
# inventory
# ----------------------------------------------------------------------------
def inventory(root):
    total = 0
    for pdir in patient_dirs(root):
        pid = os.path.basename(pdir)
        metas, dups = collect(pdir)
        if not metas:
            print(f"{pid}: no files")
            continue
        chosen, series_info = pick_series(metas)
        chosen = sort_slices(chosen)
        zs = np.array([m["z"] for m in chosen], dtype=np.float64)
        dz = np.abs(np.diff(zs)) if len(zs) > 1 else np.array([np.nan])
        desc = Counter(m["desc"] for m in chosen).most_common(1)[0][0]
        kern = Counter(m["kernel"] for m in chosen).most_common(1)[0][0]
        c0 = chosen[0]
        print(
            f"{pid}: {len(metas) + dups} files | {dups} duplicates | {len(chosen)} selected slices | "
            f"desc={desc!r} kernel={kern!r} thick={c0['thick']} "
            f"spacing={c0['spacing'][0]:.3f}mm {c0['rows']}x{c0['cols']} | "
            f"z=[{np.nanmin(zs):.1f}, {np.nanmax(zs):.1f}] dz~{np.nanmedian(dz):.2f} | "
            f"slope/intercept={c0['slope']}/{c0['intercept']}"
        )
        if series_info:
            print(f"    !! several series found (seriesUID_last12: slices): {series_info}")
        if len(dz) and np.nanmax(dz) > 3 * np.nanmedian(dz):
            print(f"    !! gap along z (max dz={np.nanmax(dz):.2f})")
        total += len(chosen)
    print(f"TOTAL selected slices: {total}")


# ----------------------------------------------------------------------------
# convert
# ----------------------------------------------------------------------------
def to_norm(ds):
    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0)) + float(
        getattr(ds, "RescaleIntercept", 0.0)
    )
    x = (np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    return x.astype(np.float32)


def convert(root, out, every=1):
    os.makedirs(out, exist_ok=True)
    for pdir in patient_dirs(root):
        pid = os.path.basename(pdir)
        metas, _ = collect(pdir)
        if not metas:
            continue
        chosen, _ = pick_series(metas)
        chosen = sort_slices(chosen)[::every]
        vol = np.empty((len(chosen), chosen[0]["rows"], chosen[0]["cols"]), dtype=np.float32)
        for i, m in enumerate(chosen):
            ds = pydicom.dcmread(m["path"], force=True)
            vol[i] = to_norm(ds)
        np.save(os.path.join(out, f"{pid}.npy"), vol)
        np.save(os.path.join(out, f"{pid}_z.npy"), np.array([m["z"] for m in chosen], np.float32))
        print(
            f"{pid}: {vol.shape} -> {out}/{pid}.npy | min={vol.min():.3f} max={vol.max():.3f} "
            f"mean={vol.mean():.3f} | {vol.nbytes / 1e6:.0f} MB"
        )


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inventory")
    a.add_argument("root")
    b = sub.add_parser("convert")
    b.add_argument("root")
    b.add_argument("out")
    b.add_argument("--every", type=int, default=1, help="keep every k-th slice (1 = all)")
    args = ap.parse_args()
    if args.cmd == "inventory":
        inventory(args.root)
    else:
        convert(args.root, args.out, args.every)


if __name__ == "__main__":
    main()
