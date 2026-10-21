import pytest
import torch

from driftrecon.core.conditioning import CFGConditionEncoder
from driftrecon.core.drift import (
    cfg_drift_from_features,
    cfg_temperature_factors,
    driftrecon_objective,
    haar_bands,
    sample_cfg_scales,
    scheduled_null_l1_weight,
)


def test_cfg_sampler_and_null_l1_schedule():
    scales = sample_cfg_scales(2048, device="cpu", generator=torch.Generator().manual_seed(3))
    assert torch.all((scales >= 1.0) & (scales <= 4.0))
    assert 0.4 < float((scales == 1.0).float().mean()) < 0.6
    assert scheduled_null_l1_weight(0, 100)[0] == 0.01
    assert scheduled_null_l1_weight(100, 100)[0] == pytest.approx(0.0)


def test_temperature_factors():
    torch.testing.assert_close(
        cfg_temperature_factors(torch.tensor([1.0, 2.0, 4.0]), 0.5),
        torch.tensor([1.0, 2.0 / 3.0, 0.4]),
    )
    for invalid in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            cfg_temperature_factors(torch.ones(2), invalid)


def test_haar_bands_are_orthonormal_and_per_channel():
    image = torch.randn(3, 2, 16, 8)
    bands = haar_bands(image, 3)
    assert len(bands) == 1 + 3 * 3
    energy = sum(band.square().sum() for band in bands)
    torch.testing.assert_close(energy, image.square().sum())
    assert all(band.shape[:2] == (3, 2) for band in bands)
    with pytest.raises(ValueError):
        haar_bands(torch.randn(1, 1, 12, 12), 3)


def test_beta_zero_is_bitwise_fixed_bandwidth():
    torch.manual_seed(11)
    first = torch.randn(3, 2, 1, 4, 4, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    positive = torch.randn(3, 1, 4, 4)
    cfg = torch.tensor([1.0, 2.0, 4.0])
    a = cfg_drift_from_features(first, positive, cfg, temperatures=(0.5, 1.0))
    b = cfg_drift_from_features(second, positive, cfg, temperatures=(0.5, 1.0), cfg_temperature_beta=0.0)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("channels", [1, 2])
def test_objective_with_wavelets_is_finite_and_differentiable(channels):
    torch.manual_seed(5)
    generated = torch.randn(2, 2, channels, 16, 16, requires_grad=True)
    target = torch.randn(2, channels, 16, 16)
    losses = driftrecon_objective(
        generated, target, generated * 0.7, target * 0.7, torch.tensor([1.0, 4.0]),
        dwt_levels=3, cfg_temperature_beta=0.5,
    )
    assert torch.isfinite(losses["loss"])
    assert float(losses["cfg_negative_weight_mean"]) == pytest.approx(1.5)
    losses["loss"].backward()
    assert torch.isfinite(generated.grad).all()


def test_wavelet_step_changes_drift_gradient():
    torch.manual_seed(12)
    base = torch.randn(3, 2, 2, 8, 8)
    target = torch.randn(3, 2, 8, 8)
    kwargs = dict(cfg_scale=torch.tensor([1.0, 2.0, 4.0]), temperatures=(1.0,), cfg_temperature_beta=0.5,
                  lambda_mse=0.0, lambda_l1_null=0.0)
    grads = []
    for levels in (0, 2):
        generated = base.clone().requires_grad_(True)
        driftrecon_objective(generated, target, generated, target, dwt_levels=levels, **kwargs)["loss"].backward()
        grads.append(generated.grad)
    assert not torch.allclose(grads[0], grads[1])


def test_condition_encoder_domain_and_class_modes():
    single = CFGConditionEncoder(8)
    assert single(torch.tensor([1.0, 2.0])).shape == (2, 8)
    classes = CFGConditionEncoder(8, num_classes=3)
    out = classes(torch.tensor([1.0, 1.0]), torch.tensor([0, 2]))
    assert not torch.equal(out[0], out[1])
    with pytest.raises(ValueError):
        classes(torch.tensor([1.0]))


def test_default_negatives_equal_tiled_targets_and_custom_negatives_change_drift():
    torch.manual_seed(13)
    generated = torch.randn(3, 2, 2, 8, 8)
    target = torch.randn(3, 2, 8, 8)
    kwargs = dict(cfg_scale=torch.tensor([1.5, 2.0, 4.0]), temperatures=(1.0,), dwt_levels=2,
                  cfg_temperature_beta=0.5, lambda_mse=0.0, lambda_l1_null=0.0)

    def drift_grad(negatives):
        g = generated.clone().requires_grad_(True)
        driftrecon_objective(g, target, g, target, null_negatives=negatives, **kwargs)["loss"].backward()
        return g.grad

    tiled = target[None].expand(3, -1, -1, -1, -1).contiguous()
    torch.testing.assert_close(drift_grad(None), drift_grad(tiled), rtol=0.0, atol=0.0)
    assert not torch.allclose(drift_grad(None), drift_grad(tiled + 0.5 * torch.randn_like(tiled)))
    with pytest.raises(ValueError):
        drift_grad(tiled[:, :2])
