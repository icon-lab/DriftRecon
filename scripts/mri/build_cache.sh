#!/usr/bin/env bash
# Step 1 (optional) -- pre-build the M4Raw caches (k-space, ESPIRiT coil maps, per-volume scales).
#   M4RAW_ROOT=/path/to/M4Raw CACHE_DIR=/fast/disk/m4raw_cache bash scripts/mri/build_cache.sh
# Caches are otherwise built on first use. ESPIRiT runs on the GPU when cupy is installed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${M4RAW_ROOT:?set M4RAW_ROOT (folder containing multicoil_train, multicoil_val, multicoil_test)}"
CACHE_DIR="${CACHE_DIR:-artifacts/m4raw_cache}"
python -m driftrecon.mri.build_cache --m4raw-root "$M4RAW_ROOT" --cache-dir "$CACHE_DIR" \
  --phases "${PHASES:-train,val,test}" --espirit-device "${ESPIRIT_DEVICE:-auto}"
