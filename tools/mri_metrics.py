#!/usr/bin/env python3
"""
DriftRecon MRI -- paper metrics on the M4Raw test set.

Inputs are ``python -m driftrecon.mri.evaluate --save-pred-dir`` folders: <dir>/R<R>/{T1,T2,FLAIR}.npz with magnitude predictions
(pred), magnitude references (gt) and slice ids (ids). The reference images and ids are read from the first
--pred folder; every folder is aligned to them by slice id.

Per slice (magnitude images), mean +- std over slices:
  PSNR / SSIM (%) : each image divided by its own mean, data_range = max of the normalized reference
  LPIPS (AlexNet) : each image scaled to [0,1] by its own maximum, copied to 3 channels in [-1,1]
  FID             : Inception-v3 pool3 features, same [0,1] scaling, per acceleration rate
Paired two-sided Wilcoxon signed-rank test against --ref (slice level).

  python tools/mri_metrics.py --R 4,8 \\
      --pred DriftRecon=outputs/mri_test/preds/model_avg --lpips alex --fid
"""
import argparse
import os
from collections import OrderedDict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

CONTRASTS = ("T1", "T2", "FLAIR")


def paper_metrics(gt, pred):
    """PSNR / SSIM (%) on mean-normalized magnitude images, data_range = max of the normalized reference."""
    g = gt / max(gt.mean(), 1e-12)
    p = pred / max(pred.mean(), 1e-12)
    dr = float(g.max())
    return float(psnr_fn(g, p, data_range=dr)), 100.0 * float(ssim_fn(g, p, data_range=dr))


class Perceptual:
    """LPIPS / FID; every magnitude image is scaled to [0,1] by its own maximum."""

    def __init__(self, lpips_net="alex", use_fid=True):
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lp = None
        if lpips_net != "none":
            import lpips
            self.lp = lpips.LPIPS(net=lpips_net, verbose=False).to(self.device).eval()
        self.inc = None
        if use_fid:
            from pytorch_fid.inception import InceptionV3
            self.inc = InceptionV3([3], resize_input=True, normalize_input=True).to(self.device).eval()
        self._gt_feats = {}

    @staticmethod
    def _norm01(x):
        m = x.reshape(len(x), -1).max(1).clip(1e-12)[:, None, None]
        return np.clip(x / m, 0, 1)

    def _t3(self, x01):
        return self.torch.from_numpy(x01).float().unsqueeze(1).to(self.device).repeat(1, 3, 1, 1)

    def lpips(self, gt, pred, bs=32):
        vals = []
        with self.torch.no_grad():
            for i in range(0, len(gt), bs):
                g, p = self._norm01(gt[i:i + bs]), self._norm01(pred[i:i + bs])
                vals.append(self.lp(self._t3(g) * 2 - 1, self._t3(p) * 2 - 1).flatten().cpu().numpy())
        return np.concatenate(vals)

    def _feats(self, x, bs=32):
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(x), bs):
                outs.append(self.inc(self._t3(self._norm01(x[i:i + bs])))[0].squeeze(-1).squeeze(-1)
                            .cpu().numpy().astype(np.float64))
        return np.concatenate(outs)

    def fid(self, gt, pred, key):
        from scipy import linalg
        if key not in self._gt_feats:
            self._gt_feats[key] = self._feats(gt)
        f1, f2 = self._gt_feats[key], self._feats(pred)
        mu1, mu2 = f1.mean(0), f2.mean(0)
        s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
        covmean = linalg.sqrtm(s1.dot(s2))
        if not np.isfinite(covmean).all():
            off = np.eye(s1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
        return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def load_reference(pred_dir, R):
    gts, ids, contrast_of = [], [], []
    for c in CONTRASTS:
        f = os.path.join(pred_dir, f"R{R:g}", f"{c}.npz")
        if not os.path.exists(f):
            raise SystemExit(f"missing: {f}")
        d = np.load(f, allow_pickle=True)
        gts.append(d["gt"].astype(np.float32)); ids.extend(d["ids"].tolist()); contrast_of.extend([c] * len(d["gt"]))
    return np.concatenate(gts), ids, np.array(contrast_of)


def load_pred(pred_dir, R, ref_ids):
    preds = {}
    for c in CONTRASTS:
        d = np.load(os.path.join(pred_dir, f"R{R:g}", f"{c}.npz"), allow_pickle=True)
        for i, s in zip(d["ids"].tolist(), d["pred"]):
            preds[i] = s
    missing = [i for i in ref_ids if i not in preds]
    if missing:
        raise SystemExit(f"{pred_dir}: {len(missing)} reference ids missing (first: {missing[0]})")
    return np.stack([preds[i] for i in ref_ids]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", default="4,8")
    ap.add_argument("--pred", nargs="+", required=True, help="NAME=<prediction folder of one checkpoint>")
    ap.add_argument("--lpips", choices=("none", "alex", "vgg"), default="alex")
    ap.add_argument("--fid", action="store_true")
    ap.add_argument("--ref", default=None, help="row name used as Wilcoxon reference")
    args = ap.parse_args()

    named = OrderedDict(it.split("=", 1) for it in args.pred)
    ref_dir = next(iter(named.values()))
    Rs = [float(r) for r in args.R.split(",")]
    percep = Perceptual(args.lpips, args.fid) if (args.lpips != "none" or args.fid) else None
    results = {}

    for R in Rs:
        gt, ref_ids, contrast_of = load_reference(ref_dir, R)
        print(f"\n================ R = {R:g}  ({len(ref_ids)} slices: "
              + ", ".join(f"{c} {int((contrast_of == c).sum())}" for c in CONTRASTS) + ") ================")
        rows = OrderedDict()
        for name, d in named.items():
            pred = load_pred(d, R, ref_ids)
            vals = np.array([paper_metrics(gt[i], pred[i]) for i in range(len(gt))])
            r = {"psnr": vals[:, 0], "ssim": vals[:, 1]}
            if percep is not None:
                if percep.lp is not None:
                    r["lpips"] = percep.lpips(gt, pred)
                if percep.inc is not None:
                    r["fid"] = percep.fid(gt, pred, key=R)
            rows[name] = r
            results[(R, name)] = r

        hdr = f"{'method':>16} |" + "".join(f" {c:>6} PSNR {c:>3} SSIM |" for c in CONTRASTS) + f" {'ALL PSNR':>14} {'ALL SSIM':>14}"
        if percep is not None:
            hdr += (" |     LPIPS      " if percep.lp is not None else "") + ("     FID" if percep.inc is not None else "")
        print(hdr)
        for name, r in rows.items():
            line = f"{name:>16} |"
            for c in CONTRASTS:
                m = contrast_of == c
                line += f" {r['psnr'][m].mean():10.2f} {r['ssim'][m].mean():8.2f} |"
            line += f" {r['psnr'].mean():6.2f}+-{r['psnr'].std():4.2f}  {r['ssim'].mean():6.2f}+-{r['ssim'].std():4.2f}"
            if "lpips" in r:
                line += f" | {r['lpips'].mean():.4f}+-{r['lpips'].std():.4f}"
            if "fid" in r:
                line += f" {r['fid']:7.2f}"
            print(line)

        if args.ref and args.ref in rows and len(rows) > 1:
            from scipy.stats import wilcoxon
            ref = rows[args.ref]
            print(f"\n  Paired Wilcoxon (two-sided, slice level), reference = {args.ref}; '*' p < 0.05")
            for name, r in rows.items():
                if name == args.ref:
                    continue
                cells = []
                for key in ("psnr", "ssim", "lpips"):
                    if key not in r:
                        continue
                    d = ref[key] - r[key]
                    if np.allclose(d, 0):
                        cells.append(f"{key}: =")
                        continue
                    pv = wilcoxon(d, alternative="two-sided", zero_method="zsplit").pvalue
                    cells.append(f"{key}: {pv:.1e}{'*' if pv < 0.05 else ''}")
                print(f"    {name:>16}  " + "   ".join(cells))

    print(f"\n================ MEAN (R = {', '.join(f'{r:g}' for r in Rs)}) ================")
    print(f"{'method':>16} | {'PSNR':>7} {'SSIM':>7}" + ("  LPIPS (mean+-std)" if args.lpips != "none" else "") + ("     FID" if args.fid else ""))
    for n in named:
        rs = [results[(R, n)] for R in Rs]
        line = f"{n:>16} | {np.mean([r['psnr'].mean() for r in rs]):7.2f} {np.mean([r['ssim'].mean() for r in rs]):7.2f}"
        if args.lpips != "none":
            allv = np.concatenate([r["lpips"] for r in rs])
            line += f"  {allv.mean():.4f}+-{allv.std():.4f}"
        if args.fid:
            line += f" {np.mean([r['fid'] for r in rs]):7.2f}"
        print(line)


if __name__ == "__main__":
    main()
