#!/usr/bin/env bash
# Step 4 -- LoDoPaB-CT test split, single reconstruction (N = 1), CFG alpha = 1, input null-noise sigma = SIGMA
# (default 0.001), then the metrics.
#   R=6 CKPT=checkpoints/lodopab_R6/model_avg.pth bash scripts/ct/test.sh   # USE_EMA=1 for EMA weights
# Sampling is seeded (--seed 0): a given checkpoint always produces the same predictions.
# Image-domain ablation: EXTRA=--no-final-dc saves the raw generator output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R="${R:-6}"
DATA_DIR="${DATA_DIR:-data}"
SIGMA="${SIGMA:-0.001}"
: "${CKPT:?set CKPT to the selected (or averaged) checkpoint}"
EMA_FLAG=""; [[ "${USE_EMA:-0}" == "1" ]] && EMA_FLAG="--use-ema"
TAG=$(basename "$CKPT" .pth)
OUT="outputs/lodopab_R${R}_test"
python -m driftrecon.ct.test --checkpoint "$CKPT" --data-dir "$DATA_DIR/lodopab/test" \
  --input-noise-sigma "$SIGMA" --samples 1 --cfg-scale 1 $EMA_FLAG --seed 0 --out "$OUT" ${EXTRA:-}
python tools/ct_metrics.py --gt "$DATA_DIR/lodopab/test" \
  --pred DriftRecon="$OUT/$TAG" --lpips alex --fid | tee "outputs/table_lodopab_R${R}_${TAG}.txt"
