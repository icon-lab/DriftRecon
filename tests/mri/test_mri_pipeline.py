import numpy as np
import pytest
import torch

pytest.importorskip("deepinv")

from driftrecon.core.cg import H1CGPseudoinverse
from driftrecon.mri.data.m4raw import contrast_id_from_slice_id, gaussian_mask
from driftrecon.mri.model import DriftReconMRI
from driftrecon.mri.physics import SenseOperator
from driftrecon.mri.reconstruction import build_model_input, reconstruct
from driftrecon.mri.trainer import DriftReconMRITrainer, TrainConfig, load_for_inference

B, N, H, W = 2, 3, 32, 32


def synthetic_batch(seed=0):
    generator = torch.Generator().manual_seed(seed)
    maps = torch.randn(B, N, H, W, dtype=torch.complex64, generator=generator)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    mask = torch.from_numpy(np.stack([gaussian_mask(H, W, 4.0, seed=i) for i in range(B)]))
    x_fs = torch.randn(B, 2, H, W, generator=generator)
    operator = SenseOperator(mask, maps, device="cpu")
    kspace = operator.forward(x_fs)
    return {
        "x_fs": x_fs, "x_us": operator.adjoint(kspace), "y_us": kspace, "mask": mask,
        "coil_maps": maps, "contrast_id": torch.tensor([0, 2]),
    }


def tiny_trainer():
    model = DriftReconMRI(blocks_per_stage=1, cond_dim=8)
    return DriftReconMRITrainer(
        model, H1CGPseudoinverse(cg_steps=3, power_iterations=3),
        TrainConfig(max_steps=2, ema_rate=0.5, use_fp16=False), device="cpu",
    )


def test_mask_is_deterministic_with_full_center():
    a, b = gaussian_mask(64, 64, 4.0, seed=3), gaussian_mask(64, 64, 4.0, seed=3)
    assert np.array_equal(a, b) and a[32, 32] == 1.0
    assert abs(a.mean() - 0.25) < 0.02
    assert contrast_id_from_slice_id("2022061003_FLAIR01.h5:slice3") == 2


def test_sense_adjoint_and_reconstruction_assembly():
    batch = synthetic_batch()
    operator = SenseOperator(batch["mask"], batch["coil_maps"], device="cpu")
    image, kspace = torch.randn(B, 2, H, W), torch.randn_like(batch["y_us"]) * batch["mask"][:, None, None]
    left = (operator.forward(image) * kspace).sum()
    right = (image * operator.adjoint(kspace)).sum()
    assert float((left - right).abs() / left.abs()) < 1e-4
    torch.testing.assert_close(reconstruct(batch["x_us"], torch.zeros(B, 2, H, W), operator), batch["x_us"])
    h = torch.randn(B, 2, H, W)
    torch.testing.assert_close(
        reconstruct(batch["x_us"], h, operator), batch["x_us"] + h - operator.adjoint(operator.forward(h))
    )


def test_train_step_with_wavelets_checkpoint_and_ema(tmp_path):
    torch.manual_seed(0)
    trainer = tiny_trainer()
    metrics = trainer.train_step(synthetic_batch())
    assert all(np.isfinite(v) for v in metrics.values())
    assert trainer.config.dwt_levels == 3
    path = tmp_path / "model.pth"
    trainer.save_checkpoint(path)
    restored = tiny_trainer()
    restored.load_checkpoint(path)
    assert restored.step == 1

    raw, p_inverse, _ = load_for_inference(path, "cpu", use_ema=False)
    averaged, _, _ = load_for_inference(path, "cpu", use_ema=True)
    assert not torch.equal(raw.tail.weight, averaged.tail.weight)

    batch = synthetic_batch(1)
    operator = SenseOperator(batch["mask"], batch["coil_maps"], device="cpu")
    with torch.no_grad():
        scale = p_inverse.estimate_spectral_scale(operator, batch["x_us"])
        anchor = p_inverse.reconstruct(operator, batch["y_us"], reference=batch["x_us"], spectral_scale=scale)
        noise = torch.randn_like(anchor)
        outs = [
            raw(build_model_input(anchor, noise, operator, p_inverse, spectral_scale=scale, sigma=s),
                batch["y_us"], operator, torch.ones(B), batch["contrast_id"])
            for s in (0.005, 0.05)
        ]
    assert outs[0].shape == (B, 2, H, W) and not torch.allclose(outs[0], outs[1])


def test_operator_matched_negatives_use_each_conditions_projector():
    torch.manual_seed(1)
    trainer = tiny_trainer()
    batch = synthetic_batch(2)
    op = SenseOperator(batch["mask"], batch["coil_maps"], device="cpu")
    scale = trainer.p_inverse.estimate_spectral_scale(op, batch["x_us"])
    negatives = trainer._operator_matched_negatives(batch["x_fs"], batch["mask"], batch["coil_maps"], scale)
    assert negatives.shape == (B, B, 2, H, W)
    own = trainer.p_inverse.null_project(op, batch["x_fs"], spectral_scale=scale)
    for i in range(B):
        torch.testing.assert_close(negatives[i, i], own[i], rtol=1e-4, atol=1e-5)
        op_i = SenseOperator(batch["mask"][i:i + 1], batch["coil_maps"][i:i + 1], device="cpu")
        for j in range(B):
            single = trainer.p_inverse.null_project(op_i, batch["x_fs"][j:j + 1], spectral_scale=scale[i:i + 1])
            torch.testing.assert_close(negatives[i, j], single[0], rtol=1e-4, atol=1e-5)
    # Different masks/coil maps per condition: P_{N_0} x_1 differs from P_{N_1} x_1.
    assert not torch.allclose(negatives[0, 1], negatives[1, 1], atol=1e-3)
