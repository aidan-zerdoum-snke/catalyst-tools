#!/usr/bin/env python3
"""
Evaluate segmentation metrics (Dice, IoU, HD, HD95, ASSD) using MedPy across two folders.

Outputs:
  - CSV: one row per case × label with all metrics
  - JSON: per-label means (Dice, IoU, HD95, ASSD)

Usage (PowerShell):
  python evaluate_all_medpy.py `
    --gt   "C:\...\labelsTs" `
    --pred "C:\...\predictionsTs" `
    --out  "C:\...\predictionsTs" `
    --labels 1 2 3 `
    --keep-largest-cc
"""

import os, glob, csv, json, argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from medpy.metric.binary import dc, hd, hd95, assd
from scipy.ndimage import label as cc_label

def load_nii(path):
    nii = nib.load(path)
    arr = np.asanyarray(nii.get_fdata())
    if arr.dtype.kind == 'f':
        arr = np.round(arr).astype(np.int32)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    return arr, spacing

def ensure_same_shape(a, b, case_id):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for {case_id}: pred {a.shape} vs gt {b.shape}")

def labels_present(arr):
    labs = np.unique(arr)
    return [int(x) for x in labs if x != 0]

def match_gt_path(gt_dir, pred_filename):
    direct = os.path.join(gt_dir, os.path.basename(pred_filename))
    if os.path.exists(direct):
        return direct
    base = os.path.basename(pred_filename).split(".nii")[0]
    alt_name = base.split("_0000")[0] + ".nii.gz"
    alt = os.path.join(gt_dir, alt_name)
    return alt if os.path.exists(alt) else None

def largest_cc(mask):
    """Keep only the largest connected component in a binary mask."""
    if mask.max() == 0:
        return mask
    lbl, _ = cc_label(mask)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0  # background
    keep = sizes.argmax()
    return (lbl == keep).astype(mask.dtype)

def tp_fp_fn_tn(gt_bin, pred_bin):
    tp = np.sum((gt_bin == 1) & (pred_bin == 1))
    fp = np.sum((gt_bin == 0) & (pred_bin == 1))
    fn = np.sum((gt_bin == 1) & (pred_bin == 0))
    tn = np.sum((gt_bin == 0) & (pred_bin == 0))
    return tp, fp, fn, tn

def compute_metrics(pred_bin, gt_bin, spacing, keep_lcc=False):
    """
    Returns a dict with dice, iou, hd, hd95, assd, tp, fp, fn, tn.
    Handles empty masks robustly; distances in mm.
    """
    if keep_lcc:
        pred_bin = largest_cc(pred_bin)

    tp, fp, fn, tn = tp_fp_fn_tn(gt_bin, pred_bin)

    # True negative (both empty)
    if pred_bin.max() == 0 and gt_bin.max() == 0:
        return dict(dice=1.0, iou=1.0, hd=None, hd95=None, assd=None,
                    tp=tp, fp=fp, fn=fn, tn=tn)

    # Missed/extra: distance metrics are infinite; overlap is 0
    if gt_bin.max() == 0 and pred_bin.max() > 0:
        print("WARNING: no overlap found")
        return dict(dice=0.0, iou=0.0, hd=np.inf, hd95=np.inf, assd=np.inf,
                    tp=tp, fp=fp, fn=fn, tn=tn)
    if gt_bin.max() > 0 and pred_bin.max() == 0:
        print("WARNING: no overlap found")
        return dict(dice=0.0, iou=0.0, hd=np.inf, hd95=np.inf, assd=np.inf,
                    tp=tp, fp=fp, fn=fn, tn=tn)

    # Overlap metrics
    dice_val = float(dc(pred_bin, gt_bin))
    denom = (tp + fp + fn)
    iou_val = float(tp / denom) if denom > 0 else np.nan

    # Distance metrics (spacing in mm)
    hd_val   = float(hd(pred_bin,  gt_bin, voxelspacing=spacing))
    hd95_val = float(hd95(pred_bin, gt_bin, voxelspacing=spacing))
    assd_val = float(assd(pred_bin, gt_bin, voxelspacing=spacing))

    return dict(dice=dice_val, iou=iou_val, hd=hd_val, hd95=hd95_val, assd=assd_val,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

def evaluate_folder(gt_dir, pred_dir, label_indices=None, keep_lcc=False):
    pred_paths = sorted(glob.glob(os.path.join(pred_dir, "*.nii*")))
    rows = []
    for pp in tqdm(pred_paths, desc="Evaluating cases", unit="case"):
        case_id = os.path.basename(pp).split(".nii")[0]
        gt_path = match_gt_path(gt_dir, pp)
        if gt_path is None:
            print(f"[WARN] GT missing for {case_id}; skipping")
            continue

        pred, sp_pred = load_nii(pp)
        gt, sp_gt     = load_nii(gt_path)
        ensure_same_shape(pred, gt, case_id)
        spacing = sp_gt  # prefer GT spacing

        labs = label_indices if label_indices is not None \
               else sorted(set(labels_present(pred)) | set(labels_present(gt)))

        for lab in tqdm(labs, desc=f"Labels for {case_id}", leave=False, unit="label"):
            prb = (pred == lab).astype(np.uint8)
            gtb = (gt == lab).astype(np.uint8)
            met = compute_metrics(prb, gtb, spacing, keep_lcc=keep_lcc)
            rows.append(dict(
                case_id=case_id,
                label=lab,
                dice=met["dice"],
                iou=met["iou"],
                hd=met["hd"],
                hd95=met["hd95"],
                assd=met["assd"],
                tp=met["tp"], fp=met["fp"], fn=met["fn"], tn=met["tn"]
            ))
    return rows

def write_csv(rows, out_csv):
    if not rows:
        print("[WARN] No rows to write"); return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def summarize(rows):
    """
    Per-label means: Dice, IoU, HD95, ASSD.
    (We avoid averaging 'hd' because max HD is outlier-prone; use hd95 as robust default.)
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {"dice": [], "iou": [], "hd95": [], "assd": []})
    for r in rows:
        lab = int(r["label"])
        for k in ["dice", "iou", "hd95", "assd"]:
            v = r[k]
            if v is None: 
                continue
            if isinstance(v, (float, int)) and np.isfinite(v):
                agg[lab][k].append(float(v))

    summary = {}
    for lab, vals in agg.items():
        d = float(np.mean(vals["dice"])) if vals["dice"] else np.nan
        i = float(np.mean(vals["iou"]))  if vals["iou"]  else np.nan
        h = float(np.mean(vals["hd95"])) if vals["hd95"] else np.nan
        a = float(np.mean(vals["assd"])) if vals["assd"] else np.nan
        summary[lab] = {"dice_mean": d, "iou_mean": i, "hd95_mean_mm": h, "assd_mean_mm": a}
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",   required=True, help="Folder with ground-truth NIfTI labels")
    ap.add_argument("--pred", required=True, help="Folder with predicted NIfTI segmentations")
    ap.add_argument("--out",  required=True, help="Folder to write metrics CSV/JSON")
    ap.add_argument("--labels", nargs="+", type=int, default=None,
                    help="Optional explicit label indices (e.g., 1 2 3)")
    ap.add_argument("--keep-largest-cc", action="store_true",
                    help="Keep only largest connected component in predictions per label before metrics")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = evaluate_folder(args.gt, args.pred, label_indices=args.labels, keep_lcc=args.keep_largest_cc)

    csv_path = os.path.join(args.out, "all_metrics_per_case.csv")
    write_csv(rows, csv_path)
    summary = summarize(rows)
    with open(os.path.join(args.out, "all_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {os.path.join(args.out, 'all_metrics_summary.json')}")

if __name__ == "__main__":
    main()
