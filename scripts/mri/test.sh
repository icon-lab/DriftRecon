#!/usr/bin/env bash
# Step 4 -- M4Raw test split (R = 4, 6, 8; T1 / T2 / FLAIR), one sample at sigma = SIGMA, then the metrics.
#   M4RAW_ROOT=/path/to/M4Raw CKPT=checkpoints/m4raw/model_020000.pth bash scripts/mri/test.sh   # USE_EMA=1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${M4RAW_ROOT:?set M4RAW_ROOT}"
: "${CKPT:?set CKPT to the selected checkpoint}"
CACHE_DIR="${CACHE_DIR:-artifacts/m4raw_cache}"
SIGMA="${SIGMA:-0.005}"
EMA_FLAG=""; [[ "${USE_EMA:-0}" == "1" ]] && EMA_FLAG="--use-ema"
TAG=$(basename "$CKPT" .pth)
OUT=outputs/mri_test
python -m driftrecon.mri.evaluate --checkpoint "$CKPT" \
  --m4raw-root "$M4RAW_ROOT" --cache-dir "$CACHE_DIR" --phase test --R 4,6,8 \
  --input-noise-sigma "$SIGMA" $EMA_FLAG --out "$OUT" --save-pred-dir "$OUT/preds"
python tools/mri_metrics.py --R 4,6,8 --pred DriftRecon="$OUT/preds/$TAG" \
  --lpips alex --fid | tee "$OUT/table_${TAG}.txt"
