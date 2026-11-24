#!/usr/bin/env python3
"""Checkpoint weight averaging (post-hoc EMA) for DriftReconCT / DriftReconMRI.

Averages the weights stored under ``--keys`` (default: raw model, EMA and
H1-CG weights) over the checkpoints of the given steps.  Every other field
(format, configs, operator geometry, step) is taken from the last checkpoint;
optimizer, scheduler and AMP scaler states are dropped, so the result is for
evaluation only.  Keys that are missing or empty in the checkpoints are skipped.

    python tools/avg_ckpt.py --dir checkpoints/lodopab_R6 --steps 90000:110000:1000 \\
        --out checkpoints/lodopab_R6/model_avg.pth
    python tools/avg_ckpt.py --dir checkpoints/m4raw --steps 14000,15000,16000 \\
        --out checkpoints/m4raw/model_avg.pth

``--steps`` is ``a:b:s`` (a to b in steps of s, b included) or a comma list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

DROPPED = ("optimizer", "scheduler", "scaler")


def parse_steps(value: str) -> list[int]:
    if ":" in value:
        start, stop, stride = (int(v) for v in value.split(":"))
        if stride <= 0 or stop < start:
            raise ValueError("--steps a:b:s needs s > 0 and b >= a.")
        return list(range(start, stop + 1, stride))
    return [int(v) for v in value.split(",") if v.strip()]


def average_state_dicts(states: list[dict]) -> dict:
    keys = list(states[0].keys())
    for state in states[1:]:
        if list(state.keys()) != keys:
            raise ValueError("Checkpoints have different parameter names.")
    out = {}
    for key in keys:
        values = [state[key] for state in states]
        if torch.is_tensor(values[0]) and torch.is_floating_point(values[0]):
            stacked = torch.stack([value.double() for value in values])
            out[key] = stacked.mean(0).to(values[0].dtype)
        else:
            out[key] = values[-1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="Checkpoint folder (model_XXXXXX.pth).")
    parser.add_argument("--steps", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keys", default="model,ema,p_inverse",
                        help="Comma-separated state entries to average.")
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    if not steps:
        raise SystemExit("no steps given")
    paths = [args.dir / f"model_{step:06d}.pth" for step in steps]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    formats = {state.get("format") for state in states}
    if len(formats) != 1:
        raise SystemExit(f"checkpoints have different formats: {sorted(map(str, formats))}")

    result = {key: value for key, value in states[-1].items() if key not in DROPPED}
    averaged = []
    for key in (k.strip() for k in args.keys.split(",") if k.strip()):
        entries = [state.get(key) for state in states]
        if any(not entry for entry in entries):
            # Keeping the last checkpoint's copy would mix averaged and
            # unaveraged weights, so the entry is dropped.
            print(f"skip {key!r}: not present in every checkpoint (dropped from the output)")
            result.pop(key, None)
            continue
        result[key] = average_state_dicts(entries)
        averaged.append(key)
    result["averaged_steps"] = steps
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out)
    print(f"averaged {', '.join(averaged)} over {len(steps)} checkpoints "
          f"({steps[0]}..{steps[-1]}) -> {args.out}")


if __name__ == "__main__":
    main()
