#!/usr/bin/env bash
# Step 3 -- select the checkpoint by validation PSNR on LoDoPaB-CT.
#   R=6 bash scripts/ct/select_checkpoint.sh            # USE_EMA=1 to score the EMA weights
# Every saved checkpoint (EVERY=10000 by default: model_*0000.pth), one sample, CFG alpha = 1,
# input null-noise sigma = SIGMA (default 0.001 = 0.02 x the training sigma); the highest validation PSNR
# wins (outputs/val_lodopab_R${R}/summary.csv). Weights around the best step can then be averaged with
# tools/avg_ckpt.py.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R="${R:-6}"
DATA_DIR="${DATA_DIR:-data}"
SIGMA="${SIGMA:-0.001}"
CKPT_DIR="${CKPT_DIR:-checkpoints/lodopab_R${R}}"
PATTERN="${PATTERN:-model_*0000.pth}"
EMA_FLAG=""; [[ "${USE_EMA:-0}" == "1" ]] && EMA_FLAG="--use-ema"
python -m driftrecon.ct.test --checkpoint "$CKPT_DIR"/$PATTERN \
  --data-dir "$DATA_DIR/lodopab/val" --input-noise-sigma "$SIGMA" --samples 1 --cfg-scale 1 $EMA_FLAG \
  --out "outputs/val_lodopab_R${R}"
