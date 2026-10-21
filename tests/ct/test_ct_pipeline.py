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
    with pytest.raises(SystemExit):
        ct_test.main(["--checkpoint", str(checkpoint), "--data-dir", str(data), "--out", str(tmp_path / "x"),
                      "--samples", "4"])
