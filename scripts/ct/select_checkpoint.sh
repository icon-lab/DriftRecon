#!/usr/bin/env bash
# Step 3 -- select the checkpoint on the whole validation patient (L333, 305 slices).
#   VIEWS=60 bash scripts/ct/select_checkpoint.sh            # USE_EMA=1 to score the EMA weights
# Every 5k-step checkpoint, one sample, input null-noise sigma = SIGMA (default 0.005), CFG = 1;
# the highest validation PSNR wins (outputs/val_mayo${VIEWS}/summary.csv).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
VIEWS="${VIEWS:-60}"
DATA_DIR="${DATA_DIR:-data}"
SIGMA="${SIGMA:-0.005}"
EMA_FLAG=""; [[ "${USE_EMA:-0}" == "1" ]] && EMA_FLAG="--use-ema"
python -m driftrecon.ct.test --checkpoint checkpoints/mayo${VIEWS}/model_[0-9]*.pth \
  --data-dir "$DATA_DIR/mayo_slices/val" --input-noise-sigma "$SIGMA" $EMA_FLAG \
  --out "outputs/val_mayo${VIEWS}"
