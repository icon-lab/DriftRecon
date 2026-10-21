"""Small command-line helpers shared by the CT and MRI entry points."""

from __future__ import annotations

import argparse
import math

import torch


def resolve_device(value: str) -> torch.device:
    name = str(value).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def positive_float_tuple(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not parsed or any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be finite and positive")
    return parsed
