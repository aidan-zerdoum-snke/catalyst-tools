#!/usr/bin/env python3 
from __future__ import annotations

import re
import shutil
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
    r"^(?P<id>\d{3})_(?P<p>P\d+)_(?P<num>\d+)_(?P<side>left|right)_US\.nii$",
    re.IGNORECASE,
)

LBL_RE = re.compile(
    r"^(?P<id>\d{3})_(?P<p>P\d+)_(?P<num>\d+)_(?P<side>left|right)\.nii$",
    re.IGNORECASE,
)

def to_case_base(case_id: str, p: str, num: str, side: str) -> str:
    return f"CASE-{case_id}-{p.upper()}-{num}-{side.upper()}"

def safe_copy(src: Path, dst: Path) -> None:
    if dst.exists() and not OVERWRITE:
        raise FileExistsError(f"Destination exists (set OVERWRITE=True to replace): {dst}")
    if DRY_RUN:
        print(f"[DRY RUN] {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

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

        if is_image:
            dst = out_dir / f"{case_base}_0000.nii"
        else:
            dst = out_dir / f"{case_base}.nii"

        safe_copy(src, dst)
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
 