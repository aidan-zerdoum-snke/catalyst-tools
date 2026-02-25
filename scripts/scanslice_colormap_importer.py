"""
Creates a 3D volume from a series of slices output from the ultrasound simulator
To use:
- Ensure input folder contains separate folders for each model variant, named "IJV_FBX_Variant_XX"
- Each variant folder should have two child folders named "Scans" and "Segments"
- Open a single "Segment" PNG in slicer and use the data probe to determine "color_to_label" mapping (RGB)
- Ensure output spacing matches desired values

"""

import os
import re
import numpy as np
import SimpleITK as sitk
from PIL import Image


# Mapping of RGBA colors to integer labels for segmentation
# Colors not intended for segmentation (e.g., fat, muscle) are mapped to background (label 0)
color_to_label = {

    (255, 138, 0, 255): 1,      # thyroid
    (64, 0, 255, 255): 2,       # common carotid artery (cca)
    (11, 255, 0, 255): 3,       # internal jugular vein (ijv)
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def match_color_with_tolerance(rgba_array, target_color, tolerance=0):
    diff = np.abs(rgba_array.astype(np.int16) - np.array(target_color, dtype=np.int16))
    return np.all(diff <= tolerance, axis=-1)


# ----------------------------------------------------------------------------- 
# Process one variant folder
# -----------------------------------------------------------------------------
def process_variant(variant_dir, output_root, spacing, origin, tolerance=0):

    scans_dir = os.path.join(variant_dir, "Scans")
    seg_dir = os.path.join(variant_dir, "Segments")

    variant_name = os.path.basename(variant_dir)
    match = re.search(r"(\d+)", variant_name)
    variant_num = match.group(1).zfill(2) if match else "XX"

    output_scan = os.path.join(output_root, f"scan_{variant_num}.nii.gz")
    output_seg = os.path.join(output_root, f"seg_{variant_num}.nii.gz")

    print("\n==============================")
    print(f" Processing Variant {variant_num}")
    print("==============================")

    # ---------------- segmentation ----------------
    png_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(".png")],
                       key=natural_sort_key)
    seg_slices = []

    for filename in png_files:
        img = Image.open(os.path.join(seg_dir, filename)).convert("RGBA")
        arr = np.array(img)

        label_slice = np.zeros(arr.shape[:2], dtype=np.uint8)
        matched_mask = np.zeros(arr.shape[:2], dtype=bool)

        for rgba, label in color_to_label.items():
            mask = match_color_with_tolerance(arr, rgba, tolerance)
            label_slice[mask] = label
            matched_mask |= mask

        seg_slices.append(label_slice)

    seg_volume = sitk.GetImageFromArray(np.stack(seg_slices, axis=0))
    seg_volume.SetSpacing(spacing)
    seg_volume.SetOrigin(origin)
    sitk.WriteImage(seg_volume, output_seg)
    print(f"   ✔ Saved segmentation → {output_seg}")

    # ---------------- intensity volume ----------------
    scan_files = sorted([f for f in os.listdir(scans_dir) if f.lower().endswith(".png")],
                        key=natural_sort_key)
    scan_slices = []

    for filename in scan_files:
        img = Image.open(os.path.join(scans_dir, filename))
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img).astype(np.uint8)
        scan_slices.append(arr)

    scan_volume = sitk.GetImageFromArray(np.stack(scan_slices, axis=0))
    scan_volume.SetSpacing(spacing)
    scan_volume.SetOrigin(origin)
    sitk.WriteImage(scan_volume, output_scan)
    print(f"   ✔ Saved scan volume → {output_scan}")


# ----------------------------------------------------------------------------- 
# Batch runner for all variants
# -----------------------------------------------------------------------------
def run_batch(top_folder, output_folder, spacing=(0.12, 0.12, 1.0), origin=(0,0,0), tolerance=0):

    os.makedirs(output_folder, exist_ok=True)

    variant_folders = [
        os.path.join(top_folder, d)
        for d in os.listdir(top_folder)
        if os.path.isdir(os.path.join(top_folder, d)) and "Variant" in d
    ]

    variant_folders = sorted(variant_folders, key=natural_sort_key)

    print(f"\nFound {len(variant_folders)} variant folders.")

    for vf in variant_folders:
        process_variant(vf, output_folder, spacing, origin, tolerance)

    print("\n All variants processed successfully.")


# ----------------------------------------------------------------------------- 
# ENTRY POINT (EDIT THESE TWO PATHS)
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    TOP_VARIANTS_FOLDER = r"C:\Python\catalyst-tools\synthetic_data\Synthetic"   # contains IJV_FBX_Variant_01 .. _16
    OUTPUT_FOLDER = r"C:\Python\catalyst-tools\synthetic_data\Synthetic\output_volumes"        # where scan_XX and seg_XX will be saved

    run_batch(
        top_folder=TOP_VARIANTS_FOLDER,
        output_folder=OUTPUT_FOLDER,
        spacing=(0.048, 0.048, 0.4),
        origin=(0.0, 0.0, 0.0),
        tolerance=0,
    )
