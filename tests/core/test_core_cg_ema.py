import pytest
import torch
from torch import nn

from driftrecon.core.cg import H1CGPseudoinverse
from driftrecon.core.ema import EMA, load_model_weights
from driftrecon.ct.data import make_ellipse_phantom
from driftrecon.ct.physics import ParallelBeamCT


@pytest.fixture
def setup():
    torch.manual_seed(1)
    return ParallelBeamCT((16, 16), num_views=8), H1CGPseudoinverse(cg_steps=8, power_iterations=8)


def test_pseudoinverse_reduces_measurement_error(setup):
    operator, p_inverse = setup
    target = torch.stack([make_ellipse_phantom(0, 16), make_ellipse_phantom(1, 16)])
    measurement = operator(target)
    scale = p_inverse.estimate_spectral_scale(operator, target)
    reconstruction = p_inverse.reconstruct(operator, measurement, reference=target, spectral_scale=scale)
    error = torch.linalg.vector_norm(operator(reconstruction) - measurement) / torch.linalg.vector_norm(measurement)
    assert float(error.detach()) < 0.25


def test_null_projection_complements_range(setup):
    operator, p_inverse = setup
    image = torch.randn(2, 1, 16, 16)
    scale = p_inverse.estimate_spectral_scale(operator, image)
    null = p_inverse.null_project(operator, image, spectral_scale=scale)
    range_part = p_inverse.reconstruct(operator, operator(image), reference=image, spectral_scale=scale)
    torch.testing.assert_close(null + range_part, image, rtol=1e-5, atol=1e-5)


def test_unrolled_solve_has_image_and_alpha_gradients(setup):
    operator, p_inverse = setup
    image = torch.randn(1, 1, 16, 16, requires_grad=True)
    scale = p_inverse.estimate_spectral_scale(operator, image.detach())
    p_inverse.null_project(operator, image, spectral_scale=scale).square().mean().backward()
    assert torch.isfinite(image.grad).all()
    assert torch.isfinite(p_inverse.raw_alpha.grad)


def test_ema_tracking_and_flagged_loading():
    model = nn.Linear(3, 2)
    ema = EMA(model, rate=0.5)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    state = {"model": model.state_dict(), "ema": ema.state_dict()}
    raw, averaged = nn.Linear(3, 2), nn.Linear(3, 2)
    load_model_weights(raw, state, use_ema=False)
    load_model_weights(averaged, state, use_ema=True)
    torch.testing.assert_close(raw.weight, model.weight)
    torch.testing.assert_close(averaged.weight, model.weight - 0.5)
    with pytest.raises(ValueError, match="no EMA"):
        load_model_weights(nn.Linear(3, 2), {"model": model.state_dict()}, use_ema=True)
