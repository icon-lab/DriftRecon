#!/usr/bin/env bash
# Step 4 -- M4Raw test split (R = 4, 8; T1 / T2 / FLAIR), single reconstruction (N = 1), CFG alpha = 1,
# input null-noise sigma = SIGMA (default 0.001), then the metrics.
#   M4RAW_ROOT=/path/to/M4Raw CKPT=checkpoints/m4raw/model_avg.pth bash scripts/mri/test.sh   # USE_EMA=1
# Image-domain ablation: EXTRA=--no-final-dc evaluates the raw generator output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${M4RAW_ROOT:?set M4RAW_ROOT}"
: "${CKPT:?set CKPT to the selected (or averaged) checkpoint}"
CACHE_DIR="${CACHE_DIR:-artifacts/m4raw_cache}"
SIGMA="${SIGMA:-0.001}"
EMA_FLAG=""; [[ "${USE_EMA:-0}" == "1" ]] && EMA_FLAG="--use-ema"
TAG=$(basename "$CKPT" .pth)
OUT=outputs/mri_test
python -m driftrecon.mri.evaluate --checkpoint "$CKPT" \
  --m4raw-root "$M4RAW_ROOT" --cache-dir "$CACHE_DIR" --phase test --R 4,8 --select-r 4 \
  --input-noise-sigma "$SIGMA" --avg-samples 1 --cfg-scale 1 $EMA_FLAG --out "$OUT" \
  --save-pred-dir "$OUT/preds" ${EXTRA:-}
python tools/mri_metrics.py --R 4,8 --pred DriftRecon="$OUT/preds/$TAG" \
  --lpips alex --fid | tee "$OUT/table_${TAG}.txt"
