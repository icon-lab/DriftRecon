#!/usr/bin/env python3
"""
DriftRecon CT -- paper metrics on the test set.

GT: LoDoPaB-CT test slices (data/lodopab/test/*.npy, sorted by name).
Predictions: ``python -m driftrecon.ct.test`` output folders (pred_cfg*.npy + slice_names.txt).

Per slice, mean +- std over the test slices:
  PSNR / SSIM (%) : each image divided by its own mean, data_range = max of the normalized reference
  LPIPS (AlexNet) : gray [0,1] image copied to 3 channels in [-1,1]
  FID             : Inception-v3 pool3 features over the test slices
Paired two-sided Wilcoxon signed-rank test against --ref (slice level).

  python tools/ct_metrics.py --gt data/lodopab/test \
      --pred DriftRecon=outputs/lodopab_R6_test/model_avg --lpips alex --fid
"""
import argparse
import math
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim


def metrics(gt, pred, mode):
    gt = np.asarray(gt, np.float64)
    pred = np.asarray(pred, np.float64)
    if mode == "paper":
        pred = np.clip(pred, 0, None)
        gt = np.abs(gt) / max(np.abs(gt).mean(), 1e-12)
        pred = np.abs(pred) / max(np.abs(pred).mean(), 1e-12)
        dr = gt.max()
        mse = max(np.mean((gt - pred) ** 2), 1e-20)
        return 10 * math.log10(dr ** 2 / mse), ssim(gt.astype(np.float32), pred.astype(np.float32), data_range=dr) * 100
    pred = np.clip(pred, 0, 1)
    mse = max(np.mean((gt - pred) ** 2), 1e-20)
    return 10 * math.log10(1.0 / mse), ssim(gt.astype(np.float32), pred.astype(np.float32), data_range=1.0)


def evaluate(gt, pred, label):
    assert pred.shape == gt.shape, f"{label}: {pred.shape} vs {gt.shape}"
    out = {"_pred": pred, "_slice": {}}
    for mode in ("unit", "paper"):
        vals = np.array([metrics(gt[i], pred[i], mode) for i in range(len(gt))])
        out[mode] = (vals[:, 0].mean(), vals[:, 0].std(), vals[:, 1].mean(), vals[:, 1].std())
        out["_slice"][mode] = vals            # (N,2): psnr, ssim (for the paired test)
    return out


class Perceptual:
    """LPIPS (per slice, gray [0,1] -> 3 channels [-1,1]) and FID (Inception-v3 pool3, per set)."""

    def __init__(self, lpips_net="alex", use_fid=True, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lp = None
        if lpips_net != "none":
            import lpips
            self.lp = lpips.LPIPS(net=lpips_net, verbose=False).to(self.device).eval()
        self.inc = None
        if use_fid:
            from pytorch_fid.inception import InceptionV3
            self.inc = InceptionV3([3], resize_input=True, normalize_input=True).to(self.device).eval()  # pool3, 2048-d
        self._gt_feats = None

    def _to3(self, x):  # (N,H,W) [0,1] -> (N,3,H,W)
        t = self.torch.from_numpy(np.clip(x, 0, 1)).float().unsqueeze(1).to(self.device)
        return t.repeat(1, 3, 1, 1)

    def lpips(self, gt, pred, bs=16):
        if self.lp is None:
            return (float("nan"), float("nan"))
        vals = []
        with self.torch.no_grad():
            for i in range(0, len(gt), bs):
                a = self._to3(gt[i:i + bs]) * 2 - 1
                b = self._to3(pred[i:i + bs]) * 2 - 1
                vals.append(self.lp(a, b).flatten().cpu().numpy())
        v = np.concatenate(vals)
        self.last_lpips = v
        return (float(v.mean()), float(v.std()))

    def _feats(self, x, bs=16):
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(x), bs):
                f = self.inc(self._to3(x[i:i + bs]))[0]          # (b,2048,1,1)
                outs.append(f.squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float64))
        return np.concatenate(outs)

    def fid(self, gt, pred):
        if self.inc is None:
            return float("nan")
        from scipy import linalg
        if self._gt_feats is None:
            self._gt_feats = self._feats(gt)
        f1, f2 = self._gt_feats, self._feats(pred)
        mu1, mu2 = f1.mean(0), f2.mean(0)
        s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
        covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
        if not np.isfinite(covmean).all():
            off = np.eye(s1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
        covmean = covmean.real
        return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def parse_named(items):
    res = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"expected name=path: {it}")
        name, path = it.rsplit("=", 1)      # the label may contain '='; the path follows the last '='
        res.append((name, Path(path)))
    return res



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True, help="test slices (data/lodopab/test)")
    ap.add_argument("--pred", nargs="+", required=True, help="name=prediction folder written by driftrecon.ct.test (contains pred_cfg*.npy)")
    ap.add_argument("--lpips", choices=("none", "alex", "vgg"), default="alex")
    ap.add_argument("--fid", action="store_true")
    ap.add_argument("--ref", default=None, help="row name used as the Wilcoxon reference (needs >= 2 --pred entries)")
    args = ap.parse_args()

    paths = sorted(args.gt.glob("*.npy"))
    if not paths:
        raise SystemExit(f"no .npy slices in {args.gt}")
    names_gt = [p.stem for p in paths]
    fs = np.stack([np.load(p).astype(np.float32) for p in paths])          # N,H,W in [0,1]
    print(f"GT: {fs.shape} from {args.gt}")

    rows = []
    for name, path in parse_named(args.pred):
        cands = sorted(path.glob("pred_cfg*.npy"))
        if not cands:
            raise SystemExit(f"{path}: no pred_cfg*.npy (run python -m driftrecon.ct.test first)")
        pred = np.asarray(np.load(cands[0]), np.float32)
        if pred.ndim == 4:
            pred = pred[:, 0]
        names_file = path / "slice_names.txt"
        if names_file.exists():
            names_o = names_file.read_text().split("\n")
            if names_o != names_gt:
                raise SystemExit(f"{path}: slice order differs from GT")
        rows.append((name, evaluate(fs, pred, name)))

    percep = None
    if args.lpips != "none" or args.fid:
        percep = Perceptual(lpips_net=args.lpips, use_fid=args.fid)
        for name, r in rows:
            r["lpips"] = percep.lpips(fs, r["_pred"])
            r["_slice"]["lpips"] = getattr(percep, "last_lpips", None)
            r["fid"] = percep.fid(fs, r["_pred"])
    extra = (f" | {'LPIPS(' + args.lpips + ')':>16}" if args.lpips != "none" else "") + (f" {'FID':>8}" if args.fid else "")
    print(f"\n{'method':>22} | {'PSNR (dB)':>14} {'SSIM (%)':>14}{extra}")
    for name, r in rows:
        b = r["paper"]
        line = f"{name:>22} | {b[0]:6.2f}+-{b[1]:4.2f}  {b[2]:6.2f}+-{b[3]:5.2f}"
        if percep is not None:
            if args.lpips != "none":
                line += f" | {r['lpips'][0]:.4f}+-{r['lpips'][1]:.4f}"
            if args.fid:
                line += f" {r['fid']:8.2f}"
        print(line)
    print("\nPSNR/SSIM: mean-normalized images, data_range = max of the reference; mean+-std over slices.")
    if args.ref:
        from scipy.stats import wilcoxon
        ref = next((r for name, r in rows if name == args.ref), None)
        if ref is None:
            raise SystemExit(f"--ref not found: {args.ref}")

        def ptxt(a, b):
            d = np.asarray(a) - np.asarray(b)
            if np.allclose(d, 0):
                return "   =    "
            pv = wilcoxon(d, alternative="two-sided", zero_method="zsplit").pvalue
            return f"{pv:8.1e}" + ("*" if pv < 0.05 else " ")
        print(f"\nPaired Wilcoxon (two-sided), reference = {args.ref}, N = {len(fs)} slices, '*' = p < 0.05")
        for name, r in rows:
            if r is ref:
                continue
            rp, op_ = ref["_slice"]["paper"], r["_slice"]["paper"]
            line = f"{name:>22} | PSNR {ptxt(rp[:, 0], op_[:, 0])} | SSIM {ptxt(rp[:, 1], op_[:, 1])}"
            if args.lpips != "none" and ref["_slice"].get("lpips") is not None:
                line += f" | LPIPS {ptxt(ref['_slice']['lpips'], r['_slice']['lpips'])}"
            print(line)


if __name__ == "__main__":
    main()
