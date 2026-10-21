#!/usr/bin/env bash
# Step 4 -- test patient (L506, 263 slices), one sample at input null-noise sigma = SIGMA, then the metrics.
#   VIEWS=60 CKPT=checkpoints/mayo60/model_080000.pth bash scripts/ct/test.sh   # USE_EMA=1 for EMA weights
# Sampling is seeded (--seed 0): a given checkpoint always produces the same predictions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
VIEWS="${VIEWS:-60}"
DATA_DIR="${DATA_DIR:-data}"
SIGMA="${SIGMA:-0.005}"
: "${CKPT:?set CKPT to the selected checkpoint}"
EMA_FLAG=""; [[ "${USE_EMA:-0}" == "1" ]] && EMA_FLAG="--use-ema"
TAG=$(basename "$CKPT" .pth)
OUT="outputs/mayo${VIEWS}_test"
python -m driftrecon.ct.test --checkpoint "$CKPT" --data-dir "$DATA_DIR/mayo_slices/test" \
  --input-noise-sigma "$SIGMA" $EMA_FLAG --seed 0 --out "$OUT"
python tools/ct_metrics.py --gt "$DATA_DIR/mayo_slices/test" \
  --pred DriftRecon="$OUT/$TAG" --lpips alex --fid | tee "outputs/table_mayo${VIEWS}_${TAG}.txt"
