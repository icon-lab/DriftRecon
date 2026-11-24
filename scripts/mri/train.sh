#!/usr/bin/env bash
# Step 2 -- train DriftReconMRI on M4Raw (one model for R = 4 and 8, T1 / T2 / FLAIR).
#   M4RAW_ROOT=/path/to/M4Raw bash scripts/mri/train.sh
# 20k steps, batch 24, activation checkpointing, fp16. Package defaults: CFG alpha in [1, 6] with
# p(alpha) ~ alpha^-3 and P(alpha = 1) = 0.5, G = 2, beta = 0.25, temperature 1.0, full-image drift
# (dwt_levels 0) with operator-matched negatives, Krylov K = 2, lambda_mse 0.01, null-L1 0.01 -> 0 (cosine),
# 15-step H1-CG (alpha 0.5, learnable), training null-noise sigma 0.05, lr 1e-4 halved every 10k,
# EMA 0.999, seed 0. A checkpoint is written every 1000 steps (for selection and weight averaging).
# Ablations: EXTRA="--drift-space image --no-measurement-conditioning" (image-domain drifting),
#   EXTRA="--no-measurement-conditioning", EXTRA="--lambda-mse 0 --lambda-l1-null-start 0 --lambda-l1-null-end 0".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${M4RAW_ROOT:?set M4RAW_ROOT (folder containing multicoil_train, multicoil_val, multicoil_test)}"
CACHE_DIR="${CACHE_DIR:-artifacts/m4raw_cache}"
CKPT_DIR="${CKPT_DIR:-checkpoints/m4raw}"
mkdir -p "$CKPT_DIR"
python -m driftrecon.mri.train \
  --m4raw-root "$M4RAW_ROOT" --cache-dir "$CACHE_DIR" \
  --train-r 4,8 --batch-size "${BATCH_SIZE:-24}" --max-steps 20000 --save-every 1000 \
  --use-checkpoint --checkpoint-dir "$CKPT_DIR" ${EXTRA:-} 2>&1 | tee -a "$CKPT_DIR/train.log"
