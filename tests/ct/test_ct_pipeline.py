import math

import numpy as np
import pytest
import torch

from driftrecon.core.cg import H1CGPseudoinverse
from driftrecon.ct import test as ct_test
from driftrecon.ct.data import CTSliceDataset, SyntheticPhantomDataset, load_ct_sinogram
from driftrecon.ct.model import DriftReconCT
from driftrecon.ct.physics import ParallelBeamCT
from driftrecon.ct.trainer import DriftReconCTTrainer, TrainConfig, load_for_inference


def tiny_model():
    return DriftReconCT(base_channels=2, blocks_per_stage=1, krylov_steps=1, cond_dim=8)


def make_trainer(views=5):
    return DriftReconCTTrainer(
        tiny_model(),
        ParallelBeamCT((16, 16), num_views=views),
        H1CGPseudoinverse(cg_steps=3, power_iterations=3),
        TrainConfig(max_steps=1, dwt_levels=2, ema_rate=0.5),
        device="cpu",
    )


def test_slice_dataset_windows_hu(tmp_path):
    np.save(tmp_path / "slice.npy", np.array([[-1000, 0], [500, 1000]], dtype=np.float32))
    image = CTSliceDataset(tmp_path, image_size=(8, 8), window=(-1000.0, 1000.0))[0]
    assert image.shape == (1, 8, 8) and 0.0 <= float(image.min()) <= float(image.max()) <= 1.0


def test_measured_sinogram_loader_validates_geometry(tmp_path):
    np.save(tmp_path / "m.npy", np.ones((6, 24), dtype=np.float32))
    assert load_ct_sinogram(tmp_path / "m.npy", expected_shape=(6, 24)).shape == (1, 1, 6, 24)
    with pytest.raises(ValueError):
        load_ct_sinogram(tmp_path / "m.npy", expected_shape=(5, 24))


def test_train_step_checkpoint_ema_and_resume(tmp_path):
    torch.manual_seed(6)
    images = torch.stack([SyntheticPhantomDataset(2, 16)[i] for i in range(2)])
    trainer = make_trainer()
    before = trainer.model.tail.weight.detach().clone()
    metrics = trainer.train_step(images)
    assert np.isfinite(list(metrics.values())).all()
    assert not torch.equal(before, trainer.model.tail.weight)
    path = tmp_path / "model.pth"
    trainer.save_checkpoint(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["format"] == "driftrecon_ct_v1" and "ema" in state

    restored = make_trainer()
    restored.load_checkpoint(path)
    assert restored.step == 1
    with pytest.raises(ValueError):
        make_trainer(views=6).load_checkpoint(path)

    _, raw, _, _ = load_for_inference(path, "cpu", use_ema=False)
    _, averaged, _, _ = load_for_inference(path, "cpu", use_ema=True)
    assert not torch.equal(raw.tail.weight, averaged.tail.weight)
    torch.testing.assert_close(averaged.tail.weight, 0.5 * (before + raw.tail.weight))


def test_test_entry_point_single_sample_sigma_and_ema(tmp_path):
    torch.manual_seed(7)
    trainer = make_trainer()
    images = torch.stack([SyntheticPhantomDataset(2, 16)[i] for i in range(2)])
    trainer.train_step(images)
    checkpoint = tmp_path / "model_000001.pth"
    trainer.save_checkpoint(checkpoint)
    data = tmp_path / "val"
    data.mkdir()
    for i in range(3):
        np.save(data / f"s{i}.npy", SyntheticPhantomDataset(3, 16)[i][0].numpy())

    outputs = {}
    for sigma, ema in ((0.005, False), (0.05, False), (0.005, True)):
        out = tmp_path / f"out_{sigma}_{ema}"
        argv = ["--checkpoint", str(checkpoint), "--data-dir", str(data), "--out", str(out),
                "--input-noise-sigma", str(sigma), "--device", "cpu"] + (["--use-ema"] if ema else [])
        ct_test.main(argv)
        outputs[(sigma, ema)] = np.load(out / "model_000001" / "pred_cfg1.npy")
        assert outputs[(sigma, ema)].shape == (3, 16, 16)
    assert not np.allclose(outputs[(0.005, False)], outputs[(0.05, False)])
    assert not np.allclose(outputs[(0.005, False)], outputs[(0.005, True)])

    def predict(*extra):
        out = tmp_path / ("x" + "_".join(extra))
        ct_test.main(["--checkpoint", str(checkpoint), "--data-dir", str(data), "--out", str(out),
                      "--input-noise-sigma", "0.05", "--device", "cpu", *extra])
        return np.load(out / "model_000001" / "pred_cfg1.npy")

    single = predict()
    averaged = predict("--samples", "3")
    batched = predict("--samples", "3", "--batched-samples", "--sample-chunk", "2")
    raw = predict("--no-final-dc")
    assert averaged.shape == batched.shape == raw.shape == (3, 16, 16)
    assert not np.allclose(single, averaged)
    assert not np.allclose(single, raw)
    with pytest.raises(ValueError):
        predict("--samples", "0")


def run_ct_model(model, images, operator):
    with torch.no_grad():
        return model(images, operator.forward(images), operator, torch.ones(images.shape[0]))


def test_krylov_cache_does_not_change_the_output(monkeypatch):
    from driftrecon.ct import model as ct_model

    torch.manual_seed(8)
    model = DriftReconCT(base_channels=2, blocks_per_stage=2, krylov_steps=2, cond_dim=8).eval()
    operator = ParallelBeamCT((16, 16), num_views=5)
    images = torch.rand(2, 1, 16, 16)
    monkeypatch.setattr(ct_model, "_KRYLOV_CACHE", True)
    cached = run_ct_model(model, images, operator)
    monkeypatch.setattr(ct_model, "_KRYLOV_CACHE", False)
    uncached = run_ct_model(model, images, operator)
    torch.testing.assert_close(cached, uncached, rtol=0.0, atol=0.0)
    assert all(block._shared_cache is None for block in model.modules()
               if isinstance(block, ct_model.CTMeasurementBlock))


def test_ablations_lr_schedule_and_endpoint_angles(tmp_path):
    torch.manual_seed(9)
    model = DriftReconCT(base_channels=2, blocks_per_stage=1, krylov_steps=1, cond_dim=8,
                         measurement_conditioning=False)
    from driftrecon.ct.model import CTMeasurementBlock

    assert not any(isinstance(m, CTMeasurementBlock) for m in model.modules())
    angles = torch.linspace(0.0, math.pi, 5).tolist()
    operator = ParallelBeamCT((16, 16), num_views=5, angles=angles)
    config = TrainConfig(max_steps=4, dwt_levels=1, drift_space="image", lr_schedule="step",
                         lr_decay_step=1, lr_decay_gamma=0.5)
    trainer = DriftReconCTTrainer(model, operator, H1CGPseudoinverse(cg_steps=3, power_iterations=3),
                                  config, device="cpu")
    images = torch.stack([SyntheticPhantomDataset(2, 16)[i] for i in range(2)])
    lrs = [trainer.train_step(images)["lr"] for _ in range(2)]
    assert lrs == pytest.approx([1e-4, 5e-5])
    path = tmp_path / "ablation.pth"
    trainer.save_checkpoint(path)
    loaded_operator, loaded_model, _, _ = load_for_inference(path, "cpu")
    assert not loaded_model.measurement_conditioning
    torch.testing.assert_close(loaded_operator.angles, operator.angles)
    assert float(loaded_operator.angles[-1]) == pytest.approx(math.pi)
    with pytest.raises(ValueError):
        DriftReconCTTrainer(tiny_model(), operator, H1CGPseudoinverse(cg_steps=3, power_iterations=3),
                            TrainConfig(max_steps=4, dwt_levels=1), device="cpu").load_checkpoint(path)


def test_flip_augmentation(tmp_path):
    image = np.arange(16, dtype=np.float32).reshape(4, 4) / 16.0
    np.save(tmp_path / "s.npy", image)
    dataset = CTSliceDataset(tmp_path, image_size=(4, 4), augment=True)
    torch.manual_seed(0)
    seen = {tuple(dataset[0].flatten().tolist()) for _ in range(40)}
    base = torch.from_numpy(image)
    expected = {tuple(t.flatten().tolist()) for t in (base, base.flip(-1), base.flip(-2), base.flip(-1).flip(-2))}
    assert seen == expected
