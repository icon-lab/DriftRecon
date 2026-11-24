#!/usr/bin/env bash
# Step 3 -- select the checkpoint on the M4Raw validation split.
#   M4RAW_ROOT=/path/to/M4Raw bash scripts/mri/select_checkpoint.sh      # USE_EMA=1 to score the EMA weights
# Every saved checkpoint, one sample, CFG alpha = 1, input null-noise sigma = SIGMA (default 0.001);
# the highest PSNR at R = 4 wins, ties broken by SSIM (outputs/mri_val/summary.csv, best.json).
# Weights around the best step can then be averaged with
#   python tools/avg_ckpt.py --dir checkpoints/m4raw --steps a:b:1000 --out checkpoints/m4raw/model_avg.pth
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${M4RAW_ROOT:?set M4RAW_ROOT}"
CACHE_DIR="${CACHE_DIR:-artifacts/m4raw_cache}"
CKPT_DIR="${CKPT_DIR:-checkpoints/m4raw}"
SIGMA="${SIGMA:-0.001}"
EMA_FLAG=""; [[ "${USE_EMA:-0}" == "1" ]] && EMA_FLAG="--use-ema"
python -m driftrecon.mri.evaluate --checkpoint "$CKPT_DIR"/model_[0-9]*.pth \
  --m4raw-root "$M4RAW_ROOT" --cache-dir "$CACHE_DIR" --phase val --R 4,8 --select-r 4 \
  --input-noise-sigma "$SIGMA" --avg-samples 1 --cfg-scale 1 $EMA_FLAG --out outputs/mri_val
