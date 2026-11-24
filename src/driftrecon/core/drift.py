"""DriftRecon objective: grouped CFG null-space drift plus image fidelity.

For a local batch of ``B`` measurements with ``G`` generations each, every
group ``i`` is pulled toward its paired target and pushed away from its own
generations and, with weight ``(cfg_i - 1)(G - 1)/(B - 1)``, from the targets
of the other ``B - 1`` measurements.  The kernel bandwidth narrows with the
guidance scale as ``tau(cfg) = tau_0 / (1 + beta (cfg - 1))``.  The drift is
evaluated on the Haar wavelet bands of the null-space component.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


def sample_cfg_scales(
    batch_size: int,
    cfg_min: float = 1.0,
    cfg_max: float = 6.0,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    sampling_power: float = 3.0,
    no_cfg_fraction: float = 0.5,
) -> torch.Tensor:
    """Sample guidance scales from an inverse-power law on ``[cfg_min, cfg_max]``.

    A fraction ``no_cfg_fraction`` of the scales is set exactly to ``1``.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    cfg_min, cfg_max = float(cfg_min), float(cfg_max)
    sampling_power = float(sampling_power)
    no_cfg_fraction = float(no_cfg_fraction)
    if not 1.0 <= cfg_min <= cfg_max:
        raise ValueError("Expected 1 <= cfg_min <= cfg_max.")
    if not math.isfinite(sampling_power) or sampling_power <= 0:
        raise ValueError("sampling_power must be finite and positive.")
    if not 0.0 <= no_cfg_fraction <= 1.0:
        raise ValueError("no_cfg_fraction must be in [0, 1].")
    if cfg_min == cfg_max:
        sampled = torch.full((int(batch_size),), cfg_min, device=device)
    else:
        fraction = torch.rand(int(batch_size), device=device, generator=generator)
        inverse_power = 1.0 - sampling_power
        if abs(inverse_power) < 1e-6:
            sampled = torch.exp(
                math.log(cfg_min)
                + fraction * (math.log(cfg_max) - math.log(cfg_min))
            )
        else:
            sampled = (
                cfg_min**inverse_power
                + fraction * (cfg_max**inverse_power - cfg_min**inverse_power)
            ) ** (1.0 / inverse_power)
    unguided = torch.rand(int(batch_size), device=device, generator=generator)
    return torch.where(unguided < no_cfg_fraction, torch.ones_like(sampled), sampled)


def scheduled_null_l1_weight(
    step: int,
    max_steps: int,
    *,
    start: float = 0.01,
    end: float = 0.0,
) -> tuple[float, float]:
    """Cosine decay of the null-space L1 weight; returns ``(weight, progress)``."""
    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive.")
    if not 0.0 <= float(end) <= float(start):
        raise ValueError("Expected 0 <= end <= start for null-L1 decay.")
    progress = min(max(float(step) / float(max_steps), 0.0), 1.0)
    weight = float(end) + 0.5 * (float(start) - float(end)) * (
        1.0 + math.cos(math.pi * progress)
    )
    return weight, progress


def expand_generations(tensor: torch.Tensor | None, generations: int) -> torch.Tensor | None:
    """Repeat every batch entry ``generations`` times along dim 0."""
    if int(generations) < 1:
        raise ValueError("generations must be positive.")
    if tensor is None:
        return None
    return tensor.repeat_interleave(int(generations), dim=0)


def group_generations(
    tensor: torch.Tensor, batch_size: int, generations: int
) -> torch.Tensor:
    """Reshape a flat ``(B*G, ...)`` tensor to ``(B, G, ...)``."""
    if tensor.shape[0] != int(batch_size) * int(generations):
        raise ValueError("Flattened CFG batch has the wrong size.")
    return tensor.reshape(int(batch_size), int(generations), *tensor.shape[1:])


def cfg_temperature_factors(
    cfg_scale: torch.Tensor,
    cfg_temperature_beta: float,
) -> torch.Tensor:
    """Return ``tau(cfg) / tau_0`` for each CFG group."""
    beta = float(cfg_temperature_beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("cfg_temperature_beta must be finite and nonnegative.")
    if not torch.isfinite(cfg_scale).all() or torch.any(cfg_scale < 1.0):
        raise ValueError("cfg_scale values must be finite and at least 1.")
    scales = cfg_scale.detach().float()
    return (1.0 + beta * (scales - 1.0)).reciprocal()


def _targets_and_weights(
    generated: torch.Tensor,
    positives: torch.Tensor,
    negative_weight: torch.Tensor,
    index: int,
    negatives: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Targets ``[own generations, mismatched negatives, paired target]`` and weights.

    Mismatched negatives are ``negatives[index, j]`` for ``j != index`` when
    given, otherwise the other conditions' targets ``positives[j]``.
    """
    batch_size, generations = generated.shape[:2]
    source = positives if negatives is None else negatives[index]
    other_positives = torch.cat([source[:index], source[index + 1 :]], dim=0)
    targets = torch.cat(
        [generated[index], other_positives, positives[index : index + 1]], dim=0
    )
    weights = torch.cat(
        [
            torch.ones(generations, device=generated.device),
            negative_weight[index].expand(batch_size - 1),
            torch.ones(1, device=generated.device),
        ]
    )
    return targets, weights


def _cdist(left: torch.Tensor, right: torch.Tensor, eps: float) -> torch.Tensor:
    """Euclidean distances via a clipped squared-distance expansion."""
    dot = left @ right.transpose(0, 1)
    squared = (
        left.square().sum(-1, keepdim=True)
        + right.square().sum(-1).unsqueeze(0)
        - 2.0 * dot
    )
    return squared.clamp_min(float(eps)).sqrt()


def cfg_drift_from_features(
    generated: torch.Tensor,
    positives: torch.Tensor,
    cfg_scale: torch.Tensor,
    *,
    temperatures: Iterable[float] = (0.2, 1.0),
    drift_scale: float = 1.0,
    feature_eps: float = 1e-8,
    cfg_temperature_beta: float = 0.0,
    negatives: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped CFG drift loss on flattened features.

    ``generated`` is ``(B, G, ...)`` and ``positives`` is ``(B, ...)``.
    ``negatives`` is an optional ``(B, B, ...)`` tensor whose entry ``[i, j]``
    is the mismatched negative of condition ``i`` built from target ``j``
    (the diagonal is ignored). 
    Returns ``(loss, mean drift magnitude, mean negative weight)``.
    """
    if generated.ndim < 3 or positives.ndim != generated.ndim - 1:
        raise ValueError("Expected generated (B,G,...) and positives (B,...).")
    batch_size, generations = generated.shape[:2]
    if batch_size < 2 or generations < 2:
        raise ValueError("CFG drift requires batch_size >= 2 and generations >= 2.")
    if tuple(generated.shape[2:]) != tuple(positives.shape[1:]):
        raise ValueError("Generated and positive feature shapes differ.")
    if cfg_scale.shape != (batch_size,):
        raise ValueError("cfg_scale has the wrong shape.")
    if negatives is not None and tuple(negatives.shape) != (batch_size, *positives.shape):
        raise ValueError("negatives must be shaped (B, B, ...) like positives.")
    temperature_factors = cfg_temperature_factors(cfg_scale, cfg_temperature_beta)
    beta = float(cfg_temperature_beta)
    temperatures = tuple(float(value) for value in temperatures)
    if not temperatures or any(value <= 0 or not math.isfinite(value) for value in temperatures):
        raise ValueError("temperatures must be finite and positive.")

    generated_flat = generated.flatten(start_dim=2).float()
    positives_flat = positives.flatten(start_dim=1).float()
    old_generated = generated_flat.detach()
    fixed_positives = positives_flat.detach()
    fixed_negatives = (
        None if negatives is None else negatives.flatten(start_dim=2).float().detach()
    )
    negative_weight = (
        (cfg_scale.detach().float() - 1.0)
        * float(generations - 1)
        / float(batch_size - 1)
    )

    # Batch-level scale normalization: mean weighted distance per coordinate.
    weighted_distance_sum = old_generated.new_tensor(0.0)
    target_weight_sum = old_generated.new_tensor(0.0)
    distance_count = 0
    target_weight_count = 0
    for index in range(batch_size):
        targets, weights = _targets_and_weights(
            old_generated, fixed_positives, negative_weight, index, fixed_negatives
        )
        distances = _cdist(old_generated[index], targets, feature_eps)
        weighted_distance_sum = weighted_distance_sum + (
            distances * weights.unsqueeze(0)
        ).sum()
        target_weight_sum = target_weight_sum + weights.sum()
        distance_count += distances.numel()
        target_weight_count += weights.numel()
    mean_distance = weighted_distance_sum / float(distance_count)
    mean_weight = target_weight_sum / float(target_weight_count)
    normalization = mean_distance / mean_weight.clamp_min(feature_eps)
    input_scale = (normalization / math.sqrt(generated_flat.shape[-1])).clamp_min(1e-3)
    distance_scale = normalization.clamp_min(1e-3)
    old_scaled = old_generated / input_scale

    force = torch.zeros_like(old_scaled)
    for temperature in temperatures:
        group_forces = []
        for index in range(batch_size):
            targets, weights = _targets_and_weights(
                old_generated, fixed_positives, negative_weight, index, fixed_negatives
            )
            distances = _cdist(old_generated[index], targets, feature_eps) / distance_scale
            self_mask = F.pad(
                torch.eye(generations, device=generated.device, dtype=distances.dtype),
                (0, batch_size),
            )
            # Keep the scalar division at beta zero so the fixed-bandwidth
            # objective is reproduced bit for bit.
            effective_temperature = (
                temperature
                if beta == 0.0
                else temperature * temperature_factors[index]
            )
            logits = -(distances + 100.0 * self_mask) / effective_temperature
            affinity = (
                torch.softmax(logits, -1) * torch.softmax(logits, -2)
            ).clamp_min(1e-6).sqrt()
            affinity = affinity * weights.unsqueeze(0)
            negative_end = generations + batch_size - 1
            negative = affinity[:, :negative_end]
            positive = affinity[:, negative_end:]
            coefficients = torch.cat(
                [
                    -negative * positive.sum(-1, keepdim=True),
                    positive * negative.sum(-1, keepdim=True),
                ],
                dim=-1,
            )
            targets_scaled = targets / input_scale
            group_force = coefficients @ targets_scaled
            group_force = group_force - coefficients.sum(
                -1, keepdim=True
            ) * old_scaled[index]
            group_forces.append(group_force)
        temperature_force = torch.stack(group_forces)
        force = force + temperature_force / temperature_force.square().mean().clamp_min(
            1e-8
        ).sqrt()

    target = (old_scaled + float(drift_scale) * force).detach()
    loss = F.mse_loss(generated_flat / input_scale, target)
    drift_norm = (force * input_scale).abs().mean()
    return loss, drift_norm, negative_weight.mean()


def haar_bands(image: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Orthonormal multilevel 2-D Haar decomposition ``[LL, LH, HL, HH, ...]``.

    Applied independently to every channel of ``(N, C, H, W)``.
    """
    if int(levels) < 0:
        raise ValueError("dwt_levels must be nonnegative.")
    if int(levels) == 0:
        return [image]
    divisor = 2 ** int(levels)
    if image.shape[-2] % divisor or image.shape[-1] % divisor:
        raise ValueError(f"Image dimensions must be divisible by {divisor}.")
    current = image
    details = []
    for _ in range(int(levels)):
        a = current[..., 0::2, 0::2]
        b = current[..., 0::2, 1::2]
        c = current[..., 1::2, 0::2]
        d = current[..., 1::2, 1::2]
        ll = (a + b + c + d) * 0.5
        lh = (a - b + c - d) * 0.5
        hl = (a + b - c - d) * 0.5
        hh = (a - b - c + d) * 0.5
        details.extend([lh, hl, hh])
        current = ll
    return [current, *details]


def driftrecon_objective(
    generated: torch.Tensor,
    target: torch.Tensor,
    null_generated: torch.Tensor,
    null_target: torch.Tensor,
    cfg_scale: torch.Tensor,
    *,
    temperatures: Iterable[float] = (0.2, 1.0),
    drift_scale: float = 1.0,
    lambda_drift: float = 1.0,
    lambda_l1: float = 0.0,
    lambda_mse: float = 1.0,
    lambda_l1_null: float = 0.01,
    dwt_levels: int = 3,
    cfg_temperature_beta: float = 0.0,
    null_negatives: torch.Tensor | None = None,
    drift_space: str = "null",
) -> dict[str, torch.Tensor]:
    """Full DriftRecon loss.

    ``generated``/``null_generated`` are ``(B, G, C, H, W)`` model outputs and
    their soft null-space projections; ``target``/``null_target`` are the
    ``(B, C, H, W)`` references.  ``null_negatives`` optionally gives the
    ``(B, B, C, H, W)`` mismatched negatives ``P_{N_i} x_j`` of each condition
    ``i``. The drift term is averaged over the Haar bands of the null components; 
    the fidelity terms act on the raw outputs.  ``drift_space="image"`` is the
    image-domain ablation: the drift acts on ``generated``/``target`` directly
    (and ``null_negatives`` is ignored)::

        loss = lambda_drift * drift + lambda_mse * MSE + lambda_l1 * L1
               + lambda_l1_null * L1(null_generated, null_target)
    """
    if generated.ndim != 5 or target.ndim != 4:
        raise ValueError("Expected generated (B,G,C,H,W) and target (B,C,H,W).")
    if null_generated.shape != generated.shape or null_target.shape != target.shape:
        raise ValueError("Null-projected tensors must match generated/target shapes.")
    if drift_space not in ("null", "image"):
        raise ValueError("drift_space must be 'null' or 'image'.")
    cfg_temperature_factors(cfg_scale, cfg_temperature_beta)
    batch_size, generations = generated.shape[:2]
    if drift_space == "image":
        drift_generated, drift_target, null_negatives = generated, target, None
    else:
        drift_generated, drift_target = null_generated, null_target
    flat_drift = drift_generated.reshape(
        batch_size * generations, *drift_generated.shape[2:]
    )
    generated_bands = haar_bands(flat_drift, int(dwt_levels))
    target_bands = haar_bands(drift_target, int(dwt_levels))
    if null_negatives is None:
        negative_bands = [None] * len(target_bands)
    else:
        if tuple(null_negatives.shape) != (batch_size, *null_target.shape):
            raise ValueError("null_negatives must be shaped (B, B, C, H, W).")
        negative_bands = [
            band.reshape(batch_size, batch_size, *band.shape[1:])
            for band in haar_bands(
                null_negatives.reshape(batch_size * batch_size, *null_target.shape[1:]),
                int(dwt_levels),
            )
        ]
    drift_losses, drift_norms, negative_weights = [], [], []
    for generated_band, target_band, negative_band in zip(
        generated_bands, target_bands, negative_bands
    ):
        grouped_band = generated_band.reshape(
            batch_size, generations, *generated_band.shape[1:]
        )
        loss, norm, weight = cfg_drift_from_features(
            grouped_band,
            target_band,
            cfg_scale,
            temperatures=temperatures,
            drift_scale=drift_scale,
            cfg_temperature_beta=cfg_temperature_beta,
            negatives=negative_band,
        )
        drift_losses.append(loss)
        drift_norms.append(norm)
        negative_weights.append(weight)
    loss_drift = torch.stack(drift_losses).mean()
    drift_norm = torch.stack(drift_norms).mean()
    negative_weight = torch.stack(negative_weights).mean()
    expanded_target = target[:, None].expand_as(generated)
    expanded_null_target = null_target[:, None].expand_as(null_generated)
    loss_l1 = F.l1_loss(generated.float(), expanded_target.float())
    loss_mse = F.mse_loss(generated.float(), expanded_target.float())
    loss_l1_null = F.l1_loss(null_generated.float(), expanded_null_target.float())
    loss = (
        float(lambda_drift) * loss_drift
        + float(lambda_l1) * loss_l1
        + float(lambda_mse) * loss_mse
        + float(lambda_l1_null) * loss_l1_null
    )
    return {
        "loss": loss,
        "loss_drift": loss_drift,
        "loss_l1": loss_l1,
        "loss_mse": loss_mse,
        "loss_l1_null": loss_l1_null,
        "drift_norm": drift_norm,
        "cfg_negative_weight_mean": negative_weight,
    }
