# Guidance-Coupled Nullspace Drifting for One-Step Medical Image Reconstruction

M. Tuna Alatan, A. Server Tapan, Saban Ozturk, M. Usama Mirza, Bilal Kabas, Ge Wang, and Tolga Çukur

[[arXiv]](https://arxiv.org/abs/2610.38953)

Official implementation of **DriftRecon** for sparse-view CT and accelerated multi-coil
MRI. DriftRecon is a generative method for one-step medical image reconstruction. It performs
drifting in the nullspace of the imaging operator, and its conditional guidance parameter uses a
coupled kernel temperature to steer the fidelity–realism trade-off at inference.

<p align="center">
  <img src="docs/figures/architecture.png" width="100%" alt="DriftRecon architecture">
</p>

*The input is the anchor `x† = A†y` plus nullspace-restricted noise `(I − A†A)ε`. The multi-scale generator
`g_θ` is conditioned at every stage on `(x†, y, A, α, c)`. Its output `h` is projected onto the nullspace
of `A`, and the final reconstruction is `x̂ = x† + P_N h`.*

## Method

<table>
  <tr>
    <td align="center" width="50%"><img src="docs/figures/nullspace.png" alt="Nullspace drifting"></td>
    <td align="center" width="50%"><img src="docs/figures/temperature.png" alt="Guidance-coupled temperature"></td>
  </tr>
  <tr>
    <td><em><b>Nullspace drifting.</b> Conventional drifting moves samples in the full image space and
    relies on a separate data-consistency projection. DriftRecon drifts only the nullspace component
    around the measurement-consistent set <code>x† + N(A)</code>.</em></td>
    <td><em><b>Guidance-coupled temperature.</b> With a fixed kernel temperature, stronger guidance
    (α &gt; 1) lets far-away negatives skew the drift direction. The coupled schedule
    <code>τ(α) = τ₀ / (1 + β(α − 1))</code> localizes interactions as guidance grows.</em></td>
  </tr>
</table>

Vector versions of the figures are in `docs/figures/*.pdf`.

## Layout

```
src/driftrecon/
  core/   drift.py (CFG drift, Haar bands, objective)   cg.py (H1-CG A_dagger / soft null projector)
          conditioning.py (CFG + domain/contrast condition, FiLM)   ema.py   cli.py
  ct/     physics.py (matched parallel-beam operator)   model.py (DriftReconCT)   data.py
          trainer.py   train.py   test.py   reconstruct.py
  mri/    physics.py (SENSE operator, multi-scale/padded wrappers)   model.py (DriftReconMRI)
          data/m4raw.py (cache, ESPIRiT, masks, dataset)   trainer.py   train.py   evaluate.py   build_cache.py
data_prep/ct/   Mayo .IMA -> [0,1] volumes -> 256x256 slices
scripts/ct/     prepare_mayo.sh -> train.sh -> select_checkpoint.sh -> test.sh
scripts/mri/    build_cache.sh  -> train.sh -> select_checkpoint.sh -> test.sh
tools/          ct_metrics.py  mri_metrics.py (paper metrics)   convert_legacy_checkpoint.py
tests/          core / ct / mri unit tests
docs/figures/   paper figures (PNG for this page, PDF vector originals)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'            # or .[mri], .[ct-data], .[eval] as needed
# optional GPU ESPIRiT: pip install cupy-cuda12x
pytest                             # CPU unit tests
```
Tested with Python 3.12 and PyTorch 2.x on a single NVIDIA RTX 5090 (32 GB).

## Sparse-view CT (AAPM/Mayo)

Data: AAPM 2016 Low Dose CT Grand Challenge, **full-dose, 1 mm, B30** `.IMA` files.

```bash
MAYO_ROOT=/path/to/full_1mm bash scripts/ct/prepare_mayo.sh   # train 2402 / val 305 / test 263 slices
VIEWS=60 bash scripts/ct/train.sh                              # checkpoints/mayo60/model_005000.pth ...
VIEWS=60 bash scripts/ct/select_checkpoint.sh                  # outputs/val_mayo60/summary.csv, prints best
VIEWS=60 CKPT=checkpoints/mayo60/model_XXXXXX.pth bash scripts/ct/test.sh
```
Add `USE_EMA=1` to the last two commands to use the EMA weights, and `SIGMA=...` to change the test noise
level. One slice or a measured sinogram can be reconstructed with `python -m driftrecon.ct.reconstruct`.

## Accelerated multi-coil MRI (M4Raw)

Data: M4Raw v1.6 (Zenodo [10.5281/zenodo.8056074](https://doi.org/10.5281/zenodo.8056074)). Extract the
archives so that `$M4RAW_ROOT/multicoil_{train,val,test}/*.h5` exist.

```bash
export M4RAW_ROOT=/path/to/M4Raw CACHE_DIR=/fast/disk/m4raw_cache
bash scripts/mri/build_cache.sh                                 # optional; otherwise built on first use
bash scripts/mri/train.sh                                       # checkpoints/m4raw/model_005000.pth ...
bash scripts/mri/select_checkpoint.sh                           # outputs/mri_val/summary.csv, best.json
CKPT=checkpoints/m4raw/model_XXXXXX.pth bash scripts/mri/test.sh
```
`USE_EMA=1` and `SIGMA=...` work as for CT.

## Citation

```bibtex
@article{alatan2026driftrecon,
  title   = {Guidance-Coupled Nullspace Drifting for One-Step Medical Image Reconstruction},
  author  = {Alatan, M. Tuna and Tapan, A. Server and Ozturk, Saban and Mirza, M. Usama and Kabas, Bilal and
             Wang, Ge and {\c{C}}ukur, Tolga},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

BSD 3-Clause (`LICENSE`). See `THIRD_PARTY_NOTICES.md`.
