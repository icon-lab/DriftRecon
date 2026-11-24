#!/usr/bin/env python3
"""Convert checkpoints written by the pre-release CT/MRI code bases.

Model weights, H1-CG parameters and (MRI) EMA weights are remapped to the
DriftReconCT / DriftReconMRI module names so that existing trained models can
be evaluated with this repository.  Optimizer states are not carried over.

    python tools/convert_legacy_checkpoint.py ct  old.pth new.pth
    python tools/convert_legacy_checkpoint.py mri old.pth new.pth
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict

import torch

# ----------------------------------------------------------------- CT
_CT_RULES = [
    (r"^condition_encoder\.ct_domain$", "condition_encoder.domain_token"),
    (r"\.condition\.1\.(weight|bias)$", r".film.affine.1.\1"),
]


def convert_ct_key(key: str) -> str:
    for pattern, replacement in _CT_RULES:
        key = re.sub(pattern, replacement, key)
    return key


def convert_ct(state: dict) -> dict:
    model = OrderedDict((convert_ct_key(k), v) for k, v in state["model"].items())
    train_config = dict(state.get("train_config", {}))
    if "fixed_sigma" in train_config:
        train_config["train_sigma"] = train_config.pop("fixed_sigma")
    return {
        "format": "driftrecon_ct_v1",
        "step": state.get("step", 0),
        "model": model,
        "p_inverse": state["p_inverse"],
        "model_config": state["model_config"],
        "p_inverse_config": state["p_inverse_config"],
        "operator_config": state["operator_config"],
        "train_config": train_config,
    }


# ----------------------------------------------------------------- MRI
_MRI_STAGES = {
    "m_down1": "enc1", "m_down2": "enc2", "m_down3": "enc3", "m_body": "body",
    "m_up3": "dec3", "m_up2": "dec2", "m_up1": "dec1",
}
_MRI_BLOCK = {
    "conv1.weight": "conv1.weight",
    "conv2.weight": "conv2.weight",
    "gain": "gain",
    "cond_affine.1.weight": "film.affine.1.weight",
    "cond_affine.1.bias": "film.affine.1.bias",
    "PhysicsBlock.decoding_conv.tail0.convin.weight": "measurement.decode.weight",
    "PhysicsBlock.encoding_conv.head0.convin.weight": "measurement.encode.conv_in.weight",
    "PhysicsBlock.encoding_conv.head0.zero_conv_skip.weight": "measurement.encode.skip_in.weight",
    "PhysicsBlock.encoding_conv.head0.conv10.weight": "measurement.encode.conv_a.weight",
    "PhysicsBlock.encoding_conv.head0.conv20.weight": "measurement.encode.conv_b.weight",
    "PhysicsBlock.encoding_conv.head0.skipconv0.weight": "measurement.encode.skip_out.weight",
}
_MRI_TOP = {
    "m_head.conv0.weight": "head.weight",
    "m_tail.conv0.weight": "tail.weight",
    "pool1.weight": "down1.weight", "pool2.weight": "down2.weight", "pool3.weight": "down3.weight",
    "up1.weight": "up1.weight", "up2.weight": "up2.weight", "up3.weight": "up3.weight",
}


def convert_mri_key(key: str):
    """New parameter name, or ``None`` for parameters the forward pass never used."""
    if key.startswith("condition_encoder."):
        return key
    if not key.startswith("ram."):
        raise KeyError(f"Unexpected legacy MRI key {key!r}.")
    key = key[len("ram."):].replace(".block.", ".")
    if key in _MRI_TOP:
        return _MRI_TOP[key]
    match = re.match(r"^(m_down[123]|m_body|m_up[123])\.enc\.(\d+)\.(.+)$", key)
    if match:
        stage, index, rest = match.groups()
        if rest in _MRI_BLOCK:
            return f"{_MRI_STAGES[stage]}.blocks.{index}.{_MRI_BLOCK[rest]}"
        if rest.startswith("PhysicsBlock.gain") or rest.endswith("zero_conv_skip.weight"):
            return None
    if key == "fact_realign":
        return None
    raise KeyError(f"Unmapped legacy MRI key {key!r}.")


def convert_mri_state_dict(model_state: dict) -> "OrderedDict[str, torch.Tensor]":
    out = OrderedDict()
    for key, value in model_state.items():
        new_key = convert_mri_key(key)
        if new_key is not None:
            out[new_key] = value
    return out


def convert_mri(state: dict) -> dict:
    legacy_model = state["model"]
    model = convert_mri_state_dict(legacy_model)
    ema = None
    if state.get("ema"):
        # The legacy EMA is a list in parameter order, which matches the
        # (buffer-free) state-dict order.
        if len(state["ema"]) != len(legacy_model):
            raise ValueError("Legacy EMA length does not match the model state.")
        ema = OrderedDict()
        for key, value in zip(legacy_model.keys(), state["ema"]):
            new_key = convert_mri_key(key)
            if new_key is not None:
                ema[new_key] = value
    args = state.get("extra_state", {}) or {}
    krylov = args.get("krylov_n", 2)
    return {
        "format": "driftrecon_mri_v1",
        "step": state.get("step", 0),
        "model": model,
        "ema": ema,
        "p_inverse": state.get("null_projector"),
        "model_config": {
            "krylov_steps": int(krylov) if str(krylov) not in ("", "None") else 2,
            "blocks_per_stage": 4,
            "cond_dim": int(args.get("conditioning_dim", 256)),
            "num_classes": 3,
            "measurement_conditioning": str(args.get("meas_cond", True)).lower() not in ("false", "0"),
        },
        "p_inverse_config": {
            "cg_steps": int(args.get("null_cg_steps", 15)),
            "h1_alpha_init": float(args.get("null_cg_h1_alpha_init", 0.5)),
            "h1_floor": float(args.get("null_cg_h1_floor", 1e-6)),
            "power_iterations": int(args.get("null_cg_power_iters", 20)),
            "learnable_alpha": bool(args.get("null_cg_learnable_alpha", True)),
        },
        "train_config": {"legacy_args": {k: v for k, v in args.items() if isinstance(v, (int, float, str, bool))}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("domain", choices=("ct", "mri"))
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    state = torch.load(args.source, map_location="cpu", weights_only=False)
    converted = convert_ct(state) if args.domain == "ct" else convert_mri(state)
    torch.save(converted, args.target)
    print(f"{args.source} -> {args.target} ({converted['format']}, step {converted['step']})")


if __name__ == "__main__":
    main()
