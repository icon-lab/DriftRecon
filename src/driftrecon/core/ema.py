"""Exponential moving average of model parameters and EMA-aware loading."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


class EMA:
    """Shadow copy of the trainable parameters, keyed by parameter name."""

    def __init__(self, model: nn.Module, rate: float = 0.999) -> None:
        if not 0.0 <= float(rate) < 1.0:
            raise ValueError("EMA rate must be in [0, 1).")
        self.rate = float(rate)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            self.shadow[name].mul_(self.rate).add_(parameter.detach(), alpha=1.0 - self.rate)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        missing = set(self.shadow) - set(state)
        if missing:
            raise KeyError(f"EMA state is missing parameters: {sorted(missing)[:5]}")
        for name, value in self.shadow.items():
            value.copy_(state[name].to(device=value.device, dtype=value.dtype))


def load_model_weights(model: nn.Module, state: Mapping, *, use_ema: bool) -> None:
    """Load ``state["model"]`` and, with ``use_ema``, overwrite parameters by ``state["ema"]``."""
    model.load_state_dict(state["model"])
    if not use_ema:
        return
    ema = state.get("ema")
    if not ema:
        raise ValueError("--use-ema was requested but the checkpoint has no EMA weights.")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ema:
                raise KeyError(f"EMA weights are missing parameter {name!r}.")
            parameter.copy_(ema[name].to(device=parameter.device, dtype=parameter.dtype))
