"""
Batch IJV/CCA CSA + angle stats from paired nnU-Net style NIfTI images + labelmaps.

Filenames:
  Labels: CASE-001-P1-1-LEFT.nii.gz
  Images: CASE-001-P1-1-LEFT_0000.nii.gz

Pairing rule:
  image stem "CASE-001-P1-1-LEFT_0000" -> key "CASE-001-P1-1-LEFT" (strip trailing "_0000")
  label stem "CASE-001-P1-1-LEFT"      -> key "CASE-001-P1-1-LEFT"

Angle:
  Side inferred from key containing "LEFT" or "RIGHT" (case-insensitive)
  Left uses reference +X, Right uses reference -X in world coordinates (affine)
  CCW positive, CW negative (range ~[-180, 180])

Circularity per axial slice:
  C = 4*pi*A / P^2  (A in mm^2, P in mm)
  We compute circularity only on slices that contain IJV and filter out limiting cases where the calculation is bad (for example, very low voxel count)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
from scipy import ndimage


# ============================================================
# USER SETTINGS (edit these)
# ============================================================

## Seg Thy
#IMAGE_DIR = Path(r"C:\Users\shane.sicienski\OneDrive - Brainlab SE\Desktop\Dataset311_SegThyUS\imagesTr")   # contains *_0000.nii.gz
#LABEL_DIR = Path(r"C:\Users\shane.sicienski\OneDrive - Brainlab SE\Desktop\Dataset311_SegThyUS\labelsTr")   # contains *.nii.gz (no _0000)

# Synthetic
IMAGE_DIR = Path(r"C:\git\catalyst-tools\synthetic_data\Synthetic\output_volumes\imagesTr")   # contains *_0000.nii.gz
LABEL_DIR = Path(r"C:\git\catalyst-tools\synthetic_data\Synthetic\output_volumes\labelsTr")   # contains *.nii.gz (no _0000)

IJV_LABEL_ID = 3
CCA_LABEL_ID = 2

#OUTPUT_CSV = Path(r"C:\Users\shane.sicienski\OneDrive - Brainlab SE\Desktop\summary_SegThyUS.csv")
OUTPUT_CSV = Path(r"C:\Users\shane.sicienski\OneDrive - Brainlab SE\Desktop\summary_Synthetic.csv")

SKIP_MISSING_PAIRS = True
USE_LARGEST_COMPONENT = True
MAX_CASES: Optional[int] = None
# ============================================================


# -----------------------------
# Filename utilities
# -----------------------------
_NII_SUFFIX_RE = re.compile(r"\.nii(\.gz)?$", re.IGNORECASE)

def strip_nii_suffix(name: str) -> str:
    return _NII_SUFFIX_RE.sub("", name)

def key_from_image_stem(stem_no_nii: str) -> str:
    # "CASE-001-...-LEFT_0000" -> "CASE-001-...-LEFT"
    s = stem_no_nii
    if s.lower().endswith("_0000"):
        s = s[:-5]
    return s

def infer_side_from_key(key: str) -> str:
    k = key.lower()
    if "left" in k:
        return "Left"
    if "right" in k:
        return "Right"
    return "ERROR_NO_SIDE"

def list_nifti_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and _NII_SUFFIX_RE.search(p.name)])


# -----------------------------
# Core helpers
# -----------------------------
def largest_component_3d(mask: np.ndarray) -> np.ndarray:
    """Keep only largest connected component in a binary 3D mask."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    lbl, num = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num <= 1:
        return (lbl > 0).astype(np.uint8)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (lbl == keep).astype(np.uint8)

def compute_axial_csa_com_and_circularity(mask3d: np.ndarray, spacing_xyz: np.ndarray):
    """
    Treat array as (X, Y, Z) with Z axial slice axis.
    Returns:
      z_idx (P,) int
      areas_mm2 (P,) float
      com_vox (P,3) float  (x,y,z) voxel coords
      circ (P,) float      circularity per slice (NaN if perimeter degenerate)
    """
    spacing_xyz = np.asarray(spacing_xyz, dtype=float)
    sx, sy = float(spacing_xyz[0]), float(spacing_xyz[1])
    Z = int(mask3d.shape[2])

    z_idx, areas, coms, circs = [], [], [], []

    st2 = np.ones((3, 3), dtype=np.uint8)

    for z in range(Z):
        sl = (mask3d[:, :, z] > 0).astype(np.uint8)
        n = int(sl.sum())
        if n == 0:
            continue

        area_mm2 = float(n) * (sx * sy)

        cx, cy = ndimage.center_of_mass(sl)  # (x,y)
        if not np.all(np.isfinite([cx, cy])):
            continue

        # boundary pixels = sl - erode(sl)
        pix = int(sl.sum())
        if pix < 20:   # tune: 10-50 depending on your resolution
            circ = float("nan")
        else:
            er = ndimage.binary_erosion(sl, structure=st2, iterations=1, border_value=0)
            boundary = sl & (~er)
            bcount = int(boundary.sum())

            p_mm = float(bcount) * float(0.5 * (sx + sy))
            if p_mm <= 1e-9:
                circ = float("nan")
            else:
                circ = float(4.0 * np.pi * area_mm2 / (p_mm * p_mm))
                if np.isfinite(circ):
                    circ = float(max(0.0, min(1.0, circ)))  # clamp to physical max

        z_idx.append(int(z))
        areas.append(float(area_mm2))
        coms.append([float(cx), float(cy), float(z)])
        circs.append(float(circ))

    if not z_idx:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.empty((0, 3), dtype=float),
            np.array([], dtype=float),
        )

    return (
        np.asarray(z_idx, dtype=int),
        np.asarray(areas, dtype=float),
        np.asarray(coms, dtype=float),
        np.asarray(circs, dtype=float),
    )

def compute_volume_ml(mask3d: np.ndarray, spacing_xyz: np.ndarray) -> float:
    spacing_xyz = np.asarray(spacing_xyz, dtype=float)
    voxel_mm3 = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    return float(int(mask3d.sum()) * voxel_mm3 / 1000.0)

def signed_angle_deg_from_ref(v_xy: np.ndarray, ref_xy: np.ndarray) -> float:
    """Signed angle from ref -> v in XY plane, degrees in (-180, +180]; CCW positive, CW negative."""
    v = np.asarray(v_xy, dtype=float)
    r = np.asarray(ref_xy, dtype=float)

    nv = float(np.linalg.norm(v))
    nr = float(np.linalg.norm(r))
    if nv < 1e-9 or nr < 1e-9:
        return float("nan")

    v = v / nv
    r = r / nr

    dot = float(r[0] * v[0] + r[1] * v[1])
    cross = float(r[0] * v[1] - r[1] * v[0])

    ang = float(np.degrees(np.arctan2(cross, dot)))  # [-180, 180]
    if ang <= -180.0:
        ang += 360.0
    return ang

def stats(arr: np.ndarray) -> Tuple[float, float, float, float]:
    """min, max, mean, std over finite values; else all NaN."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan
    return float(a.min()), float(a.max()), float(a.mean()), float(a.std(ddof=0))


# -----------------------------
# Per-case computation
# -----------------------------
@dataclass
class CaseResult:
    # filenames
    volume_filename: str
    segment_filename: str

    # side
    side: str # Determined by whether "left" or "right" (case insensitive) is in filename

    # spacing
    spacing_x_mm: float
    spacing_y_mm: float
    spacing_z_mm: float

    # resolution (voxels)
    dim_x: int
    dim_y: int
    dim_z: int

    # extents (mm)
    extent_x_mm: float
    extent_y_mm: float
    extent_z_mm: float

    # slice counts
    num_slices_total: int  # Number of slices total (Z)
    num_slices_ijv: int    # Number of slices with IJV labels
    num_slices_angle: int  # Number of slices with both IJV and CCA labels

    # IJV CSA stats
    ijv_csa_min_mm2: float
    ijv_csa_max_mm2: float
    ijv_csa_mean_mm2: float
    ijv_csa_std_mm2: float

    # Angle stats (IJV->CCA)
    angle_min_deg: float
    angle_max_deg: float
    angle_mean_deg: float
    angle_std_deg: float

    # IJV volume
    ijv_volume_ml: float

    # circularity stats (per-slice IJV)
    ijv_circ_min: float
    ijv_circ_max: float
    ijv_circ_mean: float
    ijv_circ_std: float

    # extras
    error: str = ""


def process_case(img_path: Path, seg_path: Path) -> CaseResult:
    key = key_from_image_stem(strip_nii_suffix(img_path.name))
    side = infer_side_from_key(key)
    # Create error row if side cannot be inferred
    if side == "ERROR_NO_SIDE":
        nan = float("nan")
        return CaseResult(
            volume_filename=img_path.name,
            segment_filename=seg_path.name,
            side=side,

            spacing_x_mm=nan, spacing_y_mm=nan, spacing_z_mm=nan,
            dim_x=0, dim_y=0, dim_z=0,
            extent_x_mm=nan, extent_y_mm=nan, extent_z_mm=nan,

            num_slices_total=0,
            num_slices_ijv=0,
            num_slices_angle=0,

            ijv_csa_min_mm2=nan, ijv_csa_max_mm2=nan, ijv_csa_mean_mm2=nan, ijv_csa_std_mm2=nan,
            angle_min_deg=nan, angle_max_deg=nan, angle_mean_deg=nan, angle_std_deg=nan,
            ijv_volume_ml=nan,
            ijv_circ_min=nan, ijv_circ_max=nan, ijv_circ_mean=nan, ijv_circ_std=nan,

            error=f"Could not infer side from filename (expected LEFT/RIGHT in key): {key}",
        )

    try:
        seg_img = nib.load(str(seg_path))
        seg = seg_img.get_fdata(dtype=np.float32)
        seg = np.rint(seg).astype(np.int32, copy=False)

        affine = seg_img.affine
        zooms = seg_img.header.get_zooms()
        if len(zooms) < 3:
            raise ValueError(f"Seg header zooms has <3 dims: {zooms}")
        spacing_xyz = np.asarray(zooms[:3], dtype=float)

        # Masks (single label IDs)
        ijv_mask = (seg == int(IJV_LABEL_ID)).astype(np.uint8)
        cca_mask = (seg == int(CCA_LABEL_ID)).astype(np.uint8)

        if USE_LARGEST_COMPONENT:
            ijv_mask = largest_component_3d(ijv_mask)
            cca_mask = largest_component_3d(cca_mask)

        dim_x, dim_y, dim_z = map(int, seg.shape[:3])
        num_slices_total = dim_z
        extent_x_mm = float(dim_x * spacing_xyz[0])
        extent_y_mm = float(dim_y * spacing_xyz[1])
        extent_z_mm = float(dim_z * spacing_xyz[2])

        ijv_z, ijv_area, ijv_com_vox, ijv_circ = compute_axial_csa_com_and_circularity(ijv_mask, spacing_xyz)
        cca_z, _,        cca_com_vox, _        = compute_axial_csa_com_and_circularity(cca_mask, spacing_xyz)

        num_slices_ijv = int(len(ijv_z))

        ijv_min, ijv_max, ijv_mean, ijv_std = stats(ijv_area)
        ijv_volume_ml = compute_volume_ml(ijv_mask, spacing_xyz)

        # Angles computed on slices where BOTH exist (world coords)
        ijv_com_world = apply_affine(affine, ijv_com_vox)
        cca_com_world = apply_affine(affine, cca_com_vox)

        ijv_map = {int(z): ijv_com_world[i] for i, z in enumerate(ijv_z)}
        cca_map = {int(z): cca_com_world[i] for i, z in enumerate(cca_z)}
        common = sorted(set(ijv_map.keys()) & set(cca_map.keys()))

        if side == "Left":
            ref_xy = np.array([1.0, 0.0], dtype=float)   # +X
        elif side == "Right":
            ref_xy = np.array([-1.0, 0.0], dtype=float)  # -X
        else:
            ref_xy = np.array([1.0, 0.0], dtype=float)

        angles = []
        for z in common:
            v = cca_map[z] - ijv_map[z]
            ang = signed_angle_deg_from_ref(np.array([v[0], v[1]], dtype=float), ref_xy)
            if np.isfinite(ang):
                angles.append(float(ang))

        angles = np.asarray(angles, dtype=float)
        ang_min, ang_max, ang_mean, ang_std = stats(angles)

        circ_min, circ_max, circ_mean, circ_std = stats(ijv_circ)

        return CaseResult(
            volume_filename=img_path.name,
            segment_filename=seg_path.name,

            side=side,

            spacing_x_mm=float(spacing_xyz[0]),
            spacing_y_mm=float(spacing_xyz[1]),
            spacing_z_mm=float(spacing_xyz[2]),

            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,

            extent_x_mm=extent_x_mm,
            extent_y_mm=extent_y_mm,
            extent_z_mm=extent_z_mm,

            num_slices_total=num_slices_total,
            num_slices_ijv=num_slices_ijv,
            num_slices_angle=int(angles.size),

            ijv_csa_min_mm2=ijv_min,
            ijv_csa_max_mm2=ijv_max,
            ijv_csa_mean_mm2=ijv_mean,
            ijv_csa_std_mm2=ijv_std,

            angle_min_deg=ang_min,
            angle_max_deg=ang_max,
            angle_mean_deg=ang_mean,
            angle_std_deg=ang_std,

            ijv_volume_ml=float(ijv_volume_ml),

            ijv_circ_min=circ_min,
            ijv_circ_max=circ_max,
            ijv_circ_mean=circ_mean,
            ijv_circ_std=circ_std,

            error="",
        )

    except Exception as e:
        nan = float("nan")
        return CaseResult(
            volume_filename=img_path.name,
            segment_filename=seg_path.name,

            side=side,

            spacing_x_mm=nan,
            spacing_y_mm=nan,
            spacing_z_mm=nan,

            dim_x=0,
            dim_y=0,
            dim_z=0,

            extent_x_mm=nan,
            extent_y_mm=nan,
            extent_z_mm=nan,

            num_slices_total=0,
            num_slices_ijv=0,
            num_slices_angle=0,

            ijv_csa_min_mm2=nan,
            ijv_csa_max_mm2=nan,
            ijv_csa_mean_mm2=nan,
            ijv_csa_std_mm2=nan,

            angle_min_deg=nan,
            angle_max_deg=nan,
            angle_mean_deg=nan,
            angle_std_deg=nan,

            ijv_volume_ml=nan,

            ijv_circ_min=nan,
            ijv_circ_max=nan,
            ijv_circ_mean=nan,
            ijv_circ_std=nan,

            error=f"{type(e).__name__}: {e}",
        )


# -----------------------------
# Pairing logic
# -----------------------------
def build_label_index(label_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in list_nifti_files(label_dir):
        k = strip_nii_suffix(p.name)  # label stem is key
        idx[k] = p
    return idx

def match_image_to_label(img_path: Path, label_index: Dict[str, Path]) -> Tuple[str, Optional[Path]]:
    stem = strip_nii_suffix(img_path.name)
    key = key_from_image_stem(stem)
    return key, label_index.get(key, None)


# -----------------------------
# Main + logging
# -----------------------------
def main():
    t0 = time.perf_counter()

    if not IMAGE_DIR.exists():
        raise SystemExit(f"IMAGE_DIR not found: {IMAGE_DIR}")
    if not LABEL_DIR.exists():
        raise SystemExit(f"LABEL_DIR not found: {LABEL_DIR}")

    print("=== Setup ===", flush=True)
    print(f"IMAGE_DIR:     {IMAGE_DIR}", flush=True)
    print(f"LABEL_DIR:     {LABEL_DIR}", flush=True)
    print(f"IJV_LABEL_ID:  {IJV_LABEL_ID}", flush=True)
    print(f"CCA_LABEL_ID:  {CCA_LABEL_ID}", flush=True)
    print(f"OUTPUT_CSV:    {OUTPUT_CSV}", flush=True)
    print(f"USE_LARGEST_COMPONENT: {USE_LARGEST_COMPONENT}", flush=True)
    print(f"SKIP_MISSING_PAIRS:    {SKIP_MISSING_PAIRS}", flush=True)
    print("=============", flush=True)

    label_index = build_label_index(LABEL_DIR)
    imgs = list_nifti_files(IMAGE_DIR)
    if not imgs:
        raise SystemExit(f"No .nii/.nii.gz files found in IMAGE_DIR: {IMAGE_DIR}")
    if MAX_CASES is not None:
        imgs = imgs[: int(MAX_CASES)]

    print(f"Images found: {len(imgs)} | Labels indexed: {len(label_index)}", flush=True)

    results: List[CaseResult] = []
    times: List[float] = []

    for i, img_path in enumerate(imgs, start=1):
        key, seg_path = match_image_to_label(img_path, label_index)
        print(f"[{i}/{len(imgs)}] {img_path.name} -> key={key} label={'OK' if seg_path else 'MISSING'}", flush=True)

        t_case0 = time.perf_counter()

        if seg_path is None:
            if SKIP_MISSING_PAIRS:
                continue
            # create an error row with filenames
            results.append(CaseResult(
                volume_filename=img_path.name,
                segment_filename="",

                side=infer_side_from_key(key),

                spacing_x_mm=float("nan"),
                spacing_y_mm=float("nan"),
                spacing_z_mm=float("nan"),

                dim_x=0,
                dim_y=0,
                dim_z=0,

                extent_x_mm=float("nan"),
                extent_y_mm=float("nan"),
                extent_z_mm=float("nan"),

                num_slices_total=0,
                num_slices_ijv=0,
                num_slices_angle=0,

                ijv_csa_min_mm2=float("nan"),
                ijv_csa_max_mm2=float("nan"),
                ijv_csa_mean_mm2=float("nan"),
                ijv_csa_std_mm2=float("nan"),

                angle_min_deg=float("nan"),
                angle_max_deg=float("nan"),
                angle_mean_deg=float("nan"),
                angle_std_deg=float("nan"),

                ijv_volume_ml=float("nan"),

                ijv_circ_min=float("nan"),
                ijv_circ_max=float("nan"),
                ijv_circ_mean=float("nan"),
                ijv_circ_std=float("nan"),

                error="No matching label found",
            ))
            dt = time.perf_counter() - t_case0
            times.append(dt)
            print(f"    -> MISSING label (took {dt:.2f}s)", flush=True)
            continue

        res = process_case(img_path, seg_path)
        results.append(res)

        dt = time.perf_counter() - t_case0
        times.append(dt)

        if res.error:
            print(f"    -> ERROR: {res.error} (took {dt:.2f}s)", flush=True)
        else:
            print(
                f"    -> ok | side={res.side} | total_slices={res.num_slices_total} | ijv_slices={res.num_slices_ijv} | "
                f"angle_slices={res.num_slices_angle} | mean_angle={res.angle_mean_deg:.2f} | took {dt:.2f}s",
                flush=True
            )

        if len(times) >= 3:
            avg = sum(times) / len(times)
            remaining = avg * (len(imgs) - i)
            print(f"    -> avg/case {avg:.2f}s | ETA ~{remaining:.1f}s", flush=True)

    df = pd.DataFrame([r.__dict__ for r in results])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, na_rep="") # Write empty string, not nan

    total = time.perf_counter() - t0
    n_err = int((df["error"].astype(str) != "").sum())

    print("=== Done ===", flush=True)
    print(f"Wrote {len(df)} rows -> {OUTPUT_CSV}", flush=True)
    print(f"Total time: {total:.2f}s | Errors: {n_err}", flush=True)
    if n_err:
        print("First few errors:", flush=True)
        print(df.loc[df["error"].astype(str) != "", ["volume_filename", "segment_filename", "error"]].head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()