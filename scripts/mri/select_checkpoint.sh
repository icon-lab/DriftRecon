#!/usr/bin/env bash
# Step 3 -- select the checkpoint on the whole M4Raw validation split at R = 4, 6, 8.
#   M4RAW_ROOT=/path/to/M4Raw bash scripts/mri/select_checkpoint.sh      # USE_EMA=1 to score the EMA weights
# Every 5k-step checkpoint, one sample, input null-noise sigma = SIGMA (default 0.005), CFG = 1;
# the highest PSNR averaged over R wins (outputs/mri_val/summary.csv, best.json).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${M4RAW_ROOT:?set M4RAW_ROOT}"
CACHE_DIR="${CACHE_DIR:-artifacts/m4raw_cache}"
CKPT_DIR="${CKPT_DIR:-checkpoints/m4raw}"
SIGMA="${SIGMA:-0.005}"
EMA_FLAG=""; [[ "${USE_EMA:-0}" == "1" ]] && EMA_FLAG="--use-ema"
python -m driftrecon.mri.evaluate --checkpoint "$CKPT_DIR"/model_[0-9]*.pth \
  --m4raw-root "$M4RAW_ROOT" --cache-dir "$CACHE_DIR" --phase val --R 4,6,8 \
  --input-noise-sigma "$SIGMA" $EMA_FLAG --out outputs/mri_val
