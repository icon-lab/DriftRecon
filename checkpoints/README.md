# Checkpoints

Trained checkpoints are not stored in git. The training scripts write

| setting | directory |
|---|---|
| CT, LoDoPaB-CT, R = 4 / 6 / 8 / 10 (88 / 58 / 44 / 35 views) | `checkpoints/lodopab_R{4,6,8,10}/model_XXXXXX.pth` |
| MRI, M4Raw (joint R = 4, 8) | `checkpoints/m4raw/model_XXXXXX.pth` |

every 1000 steps. Each file holds the raw and EMA weights (`--use-ema` selects
the latter), the learned H1-CG weight and the full training configuration.

The reported models are selected by validation PSNR and their weights are
averaged over a window of checkpoints around the best step:

    python tools/avg_ckpt.py --dir checkpoints/lodopab_R6 --steps 90000:110000:1000 \
        --out checkpoints/lodopab_R6/model_avg.pth

Checkpoints produced by the pre-release code can be converted with
`python tools/convert_legacy_checkpoint.py {ct,mri} old.pth new.pth`.
