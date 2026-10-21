# Checkpoints

Trained checkpoints are not stored in git. The training scripts write

| setting | directory |
|---|---|
| CT, 60 views (R = 6)  | `checkpoints/mayo60/model_XXXXXX.pth` |
| CT, 30 views (R = 12) | `checkpoints/mayo30/model_XXXXXX.pth` |
| MRI, M4Raw (R = 4, 6, 8) | `checkpoints/m4raw/model_XXXXXX.pth` |

every 5000 steps. Each file holds the raw and EMA weights (`--use-ema` selects
the latter), the learned H1-CG weight and the full training configuration.

Checkpoints produced by the pre-release code can be converted with
`python tools/convert_legacy_checkpoint.py {ct,mri} old.pth new.pth`.
