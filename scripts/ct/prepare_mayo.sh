#!/usr/bin/env bash
# Step 1 -- AAPM/Mayo 2016 full-dose 1 mm (.IMA) -> [0,1] 256x256 slices, patient-wise split.
#   MAYO_ROOT=/path/to/full_1mm bash scripts/ct/prepare_mayo.sh
# MAYO_ROOT holds one folder per patient (L067 L096 L109 L143 L192 L286 L291 L310 L333 L506).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MAYO_ROOT:?set MAYO_ROOT to the Mayo full_1mm folder}"
DATA_DIR="${DATA_DIR:-data}"

python data_prep/ct/prep_mayo.py inventory "$MAYO_ROOT"
# HU -> [0,1] with the fixed window [-1000, 2000] HU (air = 0), every slice, one volume per patient
python data_prep/ct/prep_mayo.py convert "$MAYO_ROOT" "$DATA_DIR/mayo_npy" --every 1
# 512 -> 256 (area averaging); train = 8 patients, val = L333, test = L506
python data_prep/ct/export_mayo_slices.py "$DATA_DIR/mayo_npy" "$DATA_DIR/mayo_slices" --size 256

for s in train val test; do echo "$s: $(ls "$DATA_DIR/mayo_slices/$s" | wc -l) slices"; done
echo "expected: train 2402, val 305, test 263"
