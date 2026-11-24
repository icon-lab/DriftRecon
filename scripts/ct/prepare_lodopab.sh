#!/usr/bin/env bash
# Step 1 -- LoDoPaB-CT reference images (image_fs.npz per split) -> one [0,1] 352x352 .npy per slice.
#   LODOPAB_ROOT=/path/to/lodopab-ct bash scripts/ct/prepare_lodopab.sh
# LODOPAB_ROOT holds train/, val/ and test/, each with image_fs.npz.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${LODOPAB_ROOT:?set LODOPAB_ROOT to the LoDoPaB-CT folder}"
DATA_DIR="${DATA_DIR:-data}"
python data_prep/ct/prep_lodopab.py "$LODOPAB_ROOT" "$DATA_DIR/lodopab"
for s in train val test; do echo "$s: $(ls "$DATA_DIR/lodopab/$s" | wc -l) slices"; done
