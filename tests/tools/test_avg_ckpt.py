import subprocess
import sys
from pathlib import Path

import torch

TOOL = Path(__file__).resolve().parents[2] / "tools" / "avg_ckpt.py"


def test_averages_weights_and_drops_optimizer(tmp_path):
    for step, value in ((1000, 1.0), (2000, 2.0), (3000, 6.0)):
        torch.save(
            {
                "format": "driftrecon_ct_v1",
                "step": step,
                "model": {"w": torch.full((2,), value), "count": torch.tensor(step)},
                "ema": {"w": torch.full((2,), 10 * value)},
                "p_inverse": {"alpha": torch.tensor(value)},
                "optimizer": {"state": {}},
                "model_config": {"base_channels": 2},
            },
            tmp_path / f"model_{step:06d}.pth",
        )
    out = tmp_path / "avg.pth"
    subprocess.run(
        [sys.executable, str(TOOL), "--dir", str(tmp_path), "--steps", "1000:3000:1000", "--out", str(out)],
        check=True,
    )
    state = torch.load(out, weights_only=False)
    torch.testing.assert_close(state["model"]["w"], torch.full((2,), 3.0))
    assert int(state["model"]["count"]) == 3000
    torch.testing.assert_close(state["ema"]["w"], torch.full((2,), 30.0))
    torch.testing.assert_close(state["p_inverse"]["alpha"], torch.tensor(3.0))
    assert "optimizer" not in state and state["step"] == 3000
    assert state["averaged_steps"] == [1000, 2000, 3000]
