#!/usr/bin/env python3 
from __future__ import annotations

import re
import shutil
import gzip
import nibabel as nib
import numpy as np
from pathlib import Path

# --- Configure these ---
inputImageFolder = r"D:\Catalyst\SegThy\US_data\US_volunteer_dataset\ground_truth_data\US"
inputLabelFolder = r"D:\Catalyst\SegThy\US_data\US_volunteer_dataset\ground_truth_data\US_thyroid_label"
outputImageFolder = r"D:\Catalyst\ConvertedData\images"
outputLabelFolder = r"D:\Catalyst\ConvertedData\labels"

DRY_RUN = False        # True = print actions only
OVERWRITE = False      # True = overwrite existing outputs
# ----------------------

IMG_RE = re.compile(
    r"^(?P<id>\d{3})_(?P<p>P\d+)_(?P<num>\d+)_(?P<side>left|right)_US\.nii(?:\.gz)?$",
    re.IGNORECASE,
)

LBL_RE = re.compile(
    r"^(?P<id>\d{3})_(?P<p>P\d+)_(?P<num>\d+)_(?P<side>left|right)\.nii(?:\.gz)?$",
    re.IGNORECASE,
)


def collapse_labels(src: Path, dst: Path, label_map: dict[int, int]) -> None:
    """
    Simplest possible relabeler:
      - loads src (.nii or .nii.gz)
      - replaces old labels with new labels
      - writes dst as .nii.gz
    Respects DRY_RUN and OVERWRITE.
    """
    global DRY_RUN, OVERWRITE

    if dst.exists() and not OVERWRITE:
        raise FileExistsError(f"Destination exists: {dst}")

    if DRY_RUN:
        print(f"[DRY RUN] collapse_labels: {src} -> {dst}")
        return

    img = nib.load(str(src))
    data = img.get_fdata()

    # require integer data with NO rounding
    if not np.allclose(data, np.round(data)):
        raise ValueError(f"Non-integer labels detected in: {src}")

    data = data.astype(np.uint8)

    for old, new in label_map.items():
        data[data == old] = new

    out = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(out, str(dst))


def to_case_base(case_id: str, p: str, num: str, side: str) -> str:
    return f"CASE-{case_id}-{p.upper()}-{num}-{side.upper()}"


def safe_copy(src: Path, dst: Path) -> None:
    """
    ALWAYS writes .nii.gz outputs
    - If src is .nii.gz -> safe copy to dst
    - If src is .nii    -> compress to dst (.nii.gz)
    - Otherwise         -> warn and skip (no write)
    Respects DRY_RUN and OVERWRITE.
    """
    if dst.exists() and not OVERWRITE:
        raise FileExistsError(f"Destination exists (set OVERWRITE=True to replace): {dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Figure out source and destination extensions
    src_exts = "".join(src.suffixes).lower()
    dst_exts = "".join(dst.suffixes).lower()

    # Only support writing to .nii.gz (for nnU-Net)
    if dst_exts.endswith(".nii.gz"):
        if src_exts.endswith(".nii.gz"):
            if DRY_RUN:
                print(f"[DRY RUN] copy (.nii.gz -> .nii.gz): {src} -> {dst}")
                return
            shutil.copy2(src, dst)
            return
        elif src_exts.endswith(".nii"):
            if DRY_RUN:
                print(f"[DRY RUN] compress (.nii -> .nii.gz): {src} -> {dst}")
                return
            with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
            return
        else:
            print(f"[WARN] Unsupported input extension (skipping): {src.name}")
            return
    else:
        raise ValueError(f"Destination must end with .nii.gz: {dst}")


def convert_folder(in_dir: Path, out_dir: Path, regex: re.Pattern, is_image: bool) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    scanned = 0
    matched = 0
    skipped_samples = []

    for src in sorted(in_dir.rglob("*")):  # recurse through all subfolders
        if not src.is_file():
            continue
        scanned += 1

        m = regex.match(src.name)
        if not m:
            if len(skipped_samples) < 10:
                skipped_samples.append(src.name)
            continue

        case_base = to_case_base(m.group("id"), m.group("p"), m.group("num"), m.group("side"))

        # force nii.gz
        ext = "nii.gz"
        if is_image:
            dst = out_dir / f"{case_base}_0000.{ext}"
            safe_copy(src, dst)
        else:
            dst = out_dir / f"{case_base}.{ext}"
            collapse_labels(src, dst, label_map={4: 2, 5: 3})

        mapping[case_base] = dst
        matched += 1

    kind = "images" if is_image else "labels"
    print(f"[{kind}] scanned={scanned}, matched={matched}")

    if matched == 0:
        print(f"[{kind}] No matches. Example filenames seen (first 10):")
        for name in skipped_samples:
            print(f"  - {name}")

    return mapping


def main() -> None:
    in_img = Path(inputImageFolder)
    in_lbl = Path(inputLabelFolder)
    out_img = Path(outputImageFolder)
    out_lbl = Path(outputLabelFolder)

    if not in_img.exists():
        raise FileNotFoundError(f"inputImageFolder not found: {in_img}")
    if not in_lbl.exists():
        raise FileNotFoundError(f"inputLabelFolder not found: {in_lbl}")

    img_map = convert_folder(in_img, out_img, IMG_RE, is_image=True)
    lbl_map = convert_folder(in_lbl, out_lbl, LBL_RE, is_image=False)

    img_cases = set(img_map.keys())
    lbl_cases = set(lbl_map.keys())

    print()
    print(f"Images converted: {len(img_map)}")
    print(f"Labels converted: {len(lbl_map)}")
    print(f"Matched pairs:    {len(img_cases & lbl_cases)}")

    only_img = sorted(img_cases - lbl_cases)
    only_lbl = sorted(lbl_cases - img_cases)

    if only_img:
        print("\nWARNING: Images without matching labels (showing up to 20):")
        for c in only_img[:20]:
            print(f"  - {c}")
        if len(only_img) > 20:
            print(f"  ...and {len(only_img)-20} more")

    if only_lbl:
        print("\nWARNING: Labels without matching images (showing up to 20):")
        for c in only_lbl[:20]:
            print(f"  - {c}")
        if len(only_lbl) > 20:
            print(f"  ...and {len(only_lbl)-20} more")


if __name__ == "__main__":
    main()
