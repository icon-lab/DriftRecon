#!/usr/bin/env bash
# Step 2 -- train DriftReconCT on LoDoPaB-CT.
#   R=6 bash scripts/ct/train.sh          # R = 4 | 6 | 8 | 10  ->  88 | 58 | 44 | 35 views
# Parallel beam, 352x352 images, 498 detectors, angles uniformly spaced over [0, pi] (end points included),
# batch 4, horizontal/vertical flip augmentation, 120k steps (80k for R = 10), seed 1.
# Package defaults: CFG alpha in [1, 6] with p(alpha) ~ alpha^-3 and P(alpha = 1) = 0.5, G = 2, beta = 0.25,
# temperatures (0.2, 1.0), Haar dwt_levels 3, lambda_mse 1, null-L1 0.01 -> 0 (cosine), 15-step H1-CG
# (alpha 0.5, learnable), training null-noise sigma 0.05, lr 1e-4 halved every 20k (step schedule), EMA 0.999.
# A checkpoint is written every 1000 steps (for selection and weight averaging).
# Ablations: EXTRA="--drift-space image --no-measurement-conditioning" (image-domain drifting),
#   EXTRA="--no-measurement-conditioning", EXTRA="--lambda-mse 0 --lambda-l1-null-start 0 --lambda-l1-null-end 0",
#   EXTRA="--dwt-levels 0".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
R="${R:-6}"
DATA_DIR="${DATA_DIR:-data}"
case "$R" in
  4) VIEWS=88; STEPS=120000 ;;
  6) VIEWS=58; STEPS=120000 ;;
  8) VIEWS=44; STEPS=120000 ;;
  10) VIEWS=35; STEPS=80000 ;;
  *) echo "R must be 4, 6, 8 or 10" >&2; exit 1 ;;
esac
CKPT_DIR="${CKPT_DIR:-checkpoints/lodopab_R${R}}"
mkdir -p "$CKPT_DIR"
python -m driftrecon.ct.train \
  --data-dir "$DATA_DIR/lodopab/train" \
  --image-size 352 --views "$VIEWS" --detectors 498 --angles-endpoint \
  --batch-size 4 --max-steps "${STEPS_OVERRIDE:-$STEPS}" --augment --seed 1 \
  --save-every 1000 --checkpoint-dir "$CKPT_DIR" ${EXTRA:-} 2>&1 | tee -a "$CKPT_DIR/train.log"
