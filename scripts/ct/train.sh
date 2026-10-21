#!/usr/bin/env bash
# Step 2 -- train DriftReconCT.
#   VIEWS=60 bash scripts/ct/train.sh      # R = 6
#   VIEWS=30 bash scripts/ct/train.sh      # R = 12
# Parallel beam, VIEWS projections over [0, pi), 256x256 images, 80k steps, batch 8.
# Package defaults: temperatures (0.2, 1.0), Haar dwt_levels 3, CFG in [1,4] (power 3, 50% at 1), G = 2,
# beta = 0.5, lambda_mse 1, null-L1 0.01 -> 0 (cosine), 15-step H1-CG (alpha 0.5, learnable),
# training null-noise sigma 0.05, constant lr 1e-4, EMA 0.999, seed 0.
# A checkpoint (with EMA weights) is written every 5000 steps.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
VIEWS="${VIEWS:-60}"
DATA_DIR="${DATA_DIR:-data}"
python -m driftrecon.ct.train \
  --data-dir "$DATA_DIR/mayo_slices/train" \
  --views "$VIEWS" \
  --batch-size 8 \
  --max-steps 80000 \
  --save-every 5000 \
  --checkpoint-dir "checkpoints/mayo${VIEWS}"
