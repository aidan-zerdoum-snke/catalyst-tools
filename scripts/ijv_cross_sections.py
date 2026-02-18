
#!/usr/bin/env python3
"""
Volumetrics + CSA for SegThy labels (thyroid, CCA, IJV) from .nii/.nii.gz.
- Thyroid (label=1): volume only (CSA not computed by default).
- Vessels: CCA right/left (2/4), IJV right/left (3/5): volume + CSA along centerline.

Outputs:
- A tall-format CSV with one row per (file, structure).
- Optional per-slice CSVs for vessel CSA.

"""

# usages: 
# 
# 1. python ijv_cross_sections.py "D:\Catalyst\SegThy\US_data\US_volunteer_dataset\ground_truth_data\US_thyroid_label" --out_dir "D:\Catalyst\SegThy\CSA_output" --step_mm 2.0 --plane_size_mm 25.0 --res_mm 0.12
# 2. per slice output on MRI labels: python ijv_cross_sections.py "C:\Python\nnUNet_Catalyst\nnUNet_raw\Dataset303_SegThyMRI\labelsTr" --out_dir "C:\Python\nnUNet_Catalyst\nnUNet_raw\Dataset303_SegThyMRI\CSA_evaluation_output" --step_mm 2.0 --plane_size_mm 25.0 --res_mm 0.12 --per_slice
# 3. per slice output on OG segthy dataset with OG names: python ijv_cross_sections.py "D:\Catalyst\SegThy\US_data\US_volunteer_dataset\ground_truth_data\US_thyroid_label" --out_dir "D:\Catalyst\SegThy\CSA_output" --step_mm 2.0 --plane_size_mm 25.0 --res_mm 0.12 --per_slice
# 4. per slice output on synthetic dataset variants: python ijv_cross_sections.py "C:\Python\catalyst-tools\synthetic_data\Synthetic\output_volumes\labelsTr" --out_dir "C:\Python\catalyst-tools\synthetic_data\Synthetic\output_volumes\metrics" --step_mm 2.0 --plane_size_mm 25.0 --res_mm 0.12 --per_slice
#

import os
import glob
import math
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.orientations import aff2axcodes 
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from skimage.measure import label as cc_label

# ----------------------
# Label schema for SegThy
# ----------------------
SEGTHY_LABELS = {
    1: ("Thyroid", False),
    2: ("CCA_left", True),
    3: ("IJV_left", True),
    4: ("CCA_right", True),
    5: ("IJV_right", True),
}

# ----------------------
# Utilities
# ----------------------
def distances_from_cranial(pts_vox, spacing, vol_shape, axis_idx, affine=None, cranial_edge='auto'):
    """
    Return per-point distances (mm) from the cranial edge of the volume
    along the selected axis (0=x, 1=y, 2=z).
    """
    # Decide whether cranial lies at min or max index
    cranial_is_max = True
    if cranial_edge == 'min':
        cranial_is_max = False
    elif cranial_edge == 'max':
        cranial_is_max = True
    else:  # 'auto'
        if affine is not None:
            try:
                ax = aff2axcodes(affine)[axis_idx]  # e.g., ('R','A','S')
                if ax == 'S':  # increasing index -> Superior
                    cranial_is_max = True
                elif ax == 'I':  # increasing index -> Inferior
                    cranial_is_max = False
            except Exception:
                pass  # fall back to max

    cranial_index = (vol_shape[axis_idx] - 1) if cranial_is_max else 0
    delta_vox = np.abs(cranial_index - pts_vox[:, axis_idx])
    return delta_vox * spacing[axis_idx]

def get_voxel_spacing(img: nib.Nifti1Image, override=None):
    """
    Return voxel spacing (x,y,z) in mm.
    If override is a tuple/list of len 3, use that instead.
    """
    if override is not None and len(override) == 3:
        return np.array(override, dtype=float)
    zooms = img.header.get_zooms()
    spacing = np.array(zooms[:3], dtype=float)
    return spacing

def load_seg(path, override_spacing=None):
    img = nib.load(path)
    data = img.get_fdata().astype(np.int16)
    spacing = get_voxel_spacing(img, override=override_spacing)
    affine = img.affine 
    return data, spacing, affine


def binary_mask(seg, label_val):
    return (seg == label_val).astype(np.uint8)

def largest_component(mask):
    """Keep only the largest connected component of a binary mask."""
    if mask.sum() == 0:
        return mask
    lbl = cc_label(mask, connectivity=1)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0  # background
    largest = counts.argmax()
    return (lbl == largest).astype(np.uint8)

def compute_volume_ml(mask, spacing):
    """Volume in mL (1 mL = 1000 mm^3)."""
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    vol_mm3 = int(mask.sum()) * voxel_volume_mm3
    return vol_mm3 / 1000.0

def skeleton_centerline(mask):
    """
    Axial fallback centerline:
    - For each z-slice containing foreground, use 2D COM.
    - Stack COM points across z; prune large jumps.
    Returns Nx3 voxel coordinates ordered by z.
    """
    m = (mask > 0).astype(np.uint8)
    Z = m.shape[2]
    pts = []
    for z in range(Z):
        sl = m[:, :, z]
        if sl.sum() == 0:
            continue
        com = ndimage.center_of_mass(sl)
        if not np.isfinite(com[0]) or not np.isfinite(com[1]):
            continue
        x, y = com
        pts.append([x, y, float(z)])
    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)
    pts = np.array(pts, dtype=float)
    deltas = np.diff(pts, axis=0)
    d = np.linalg.norm(deltas, axis=1)
    keep = np.insert(d < 10.0, 0, True)  # allow up to ~10 voxel jump
    return pts[keep]

def resample_equal_arclength(points_vox, spacing, step_mm=2.0, smooth_window=9, smooth_poly=3):
    """
    Resample centerline to equal arc-length spacing (in mm).
    Returns: pts_vox_resampled (Mx3), t_mm (Mx3 unit tangents), s_mm (M,)
    """
    if len(points_vox) < 2:
        return points_vox, None, np.array([0.0] * len(points_vox))
    pts_mm = points_vox.astype(float) * spacing
    deltas_mm = np.diff(pts_mm, axis=0)
    seg_len_mm = np.linalg.norm(deltas_mm, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len_mm)])
    total_len = s[-1]
    if total_len < 1e-6:
        return points_vox[[0]], None, np.array([0.0])
    s_query = np.arange(0.0, total_len + 1e-6, step_mm)
    f_x = interp1d(s, pts_mm[:, 0], kind='linear')
    f_y = interp1d(s, pts_mm[:, 1], kind='linear')
    f_z = interp1d(s, pts_mm[:, 2], kind='linear')
    pts_mm_res = np.stack([f_x(s_query), f_y(s_query), f_z(s_query)], axis=1)
    pts_vox_res = pts_mm_res / spacing
    # smooth + gradient for tangents
    k = min(smooth_window, (len(pts_mm_res) // 2) * 2 - 1)  # odd, <= length
    if k >= 5:
        px = savgol_filter(pts_mm_res[:, 0], window_length=k, polyorder=smooth_poly)
        py = savgol_filter(pts_mm_res[:, 1], window_length=k, polyorder=smooth_poly)
        pz = savgol_filter(pts_mm_res[:, 2], window_length=k, polyorder=smooth_poly)
        pts_mm_res = np.stack([px, py, pz], axis=1)
    t_mm = np.gradient(pts_mm_res, s_query, axis=0)
    norms = np.linalg.norm(t_mm, axis=1) + 1e-12
    t_mm = t_mm / norms[:, None]
    return pts_vox_res, t_mm, s_query

def orthonormal_basis_perpendicular(t_mm):
    """Return two unit vectors (u,v) spanning plane ⟂ t."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, t_mm)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(t_mm, a)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(t_mm, u)
    v /= (np.linalg.norm(v) + 1e-12)
    return u, v

def sample_plane_area(mask, spacing, center_vox, t_mm,
                      plane_size_mm=25.0, res_mm=0.5, slab_thickness_mm=0.0):
    """
    Sample binary mask on plane orthogonal to t_mm through center_vox.
    Returns area_mm2 = (#foreground pixels) * res_mm^2.
    Allows slab thickness (OR of parallel planes).
    """
    if t_mm is None or np.linalg.norm(t_mm) < 1e-6:
        return 0.0
    u_mm, v_mm = orthonormal_basis_perpendicular(t_mm)
    half = plane_size_mm / 2.0
    xs = np.arange(-half, half + 1e-6, res_mm)
    ys = np.arange(-half, half + 1e-6, res_mm)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    H, W = X.shape
    c_vox = center_vox.astype(float)
    n_slabs = max(1, int(round(slab_thickness_mm / max(res_mm, 1e-6))))
    offsets_t = np.linspace(-0.5 * slab_thickness_mm, 0.5 * slab_thickness_mm, n_slabs)
    sampled = np.zeros((H, W), dtype=np.uint8)
    for dt in offsets_t:
        mm_offsets = (X[..., None] * u_mm[None, None, :] +
                      Y[..., None] * v_mm[None, None, :] +
                      dt * t_mm[None, None, :])
        vox_offsets = mm_offsets / spacing[None, None, :]
        coords_vox = c_vox[None, None, :] + vox_offsets
        coords = [coords_vox[..., i].ravel() for i in range(3)]
        vals = ndimage.map_coordinates(mask.astype(float), coords, order=0, mode='nearest')
        vals = (vals.reshape(H, W) > 0).astype(np.uint8)
        sampled = np.maximum(sampled, vals)
    area_mm2 = sampled.sum() * (res_mm ** 2)
    return float(area_mm2)

def compute_cross_sections(mask, spacing, step_mm=2.0,
                           plane_size_mm=25.0, res_mm=0.5, slab_thickness_mm=0.0):
    """
    For a vessel mask:
    - centerline points + tangents
    - per-plane area along arc length
    Returns: s_mm (M,), areas_mm2 (M,), pts_vox (Mx3)
    """
    cl_vox = skeleton_centerline(mask)
    if len(cl_vox) < 2:
        com = np.array(ndimage.center_of_mass(mask)) if mask.sum() > 0 else np.array([0, 0, 0], dtype=float)
        s_mm = np.array([0.0])
        areas = np.array([sample_plane_area(mask, spacing, com, np.array([0, 0, 1.0]),
                                            plane_size_mm, res_mm, slab_thickness_mm)])
        return s_mm, areas, np.array([com])
    pts_vox_res, t_mm, s_mm = resample_equal_arclength(cl_vox, spacing, step_mm=step_mm)
    areas = []
    for i in range(len(pts_vox_res)):
        t = t_mm[i] if t_mm is not None else np.array([0, 0, 1.0])
        a = sample_plane_area(mask, spacing, pts_vox_res[i], t,
                              plane_size_mm, res_mm, slab_thickness_mm)
        areas.append(a)
    return s_mm, np.array(areas), pts_vox_res

# ----------------------
# Processing per file/label
# ----------------------

def process_label(seg, spacing, affine, label_id, is_vessel, args, base_name):
    mask = binary_mask(seg, label_id)
    mask = largest_component(mask)
    volume_ml = compute_volume_ml(mask, spacing)

    mean_area = std_area = min_area = max_area = np.nan
    num_samples = 0
    per_slice_csv = ""

    if is_vessel and mask.sum() > 0:
        s_mm, areas_mm2, pts_vox = compute_cross_sections(
            mask, spacing,
            step_mm=args.step_mm,
            plane_size_mm=args.plane_size_mm,
            res_mm=args.res_mm,
            slab_thickness_mm=args.slab_mm
        )
        valid = areas_mm2[np.isfinite(areas_mm2) & (areas_mm2 > 0)]
        if valid.size:
            mean_area = float(valid.mean())
            std_area  = float(valid.std())
            min_area  = float(valid.min())
            max_area  = float(valid.max())
        num_samples = int(len(areas_mm2))

        if args.per_slice_out:
            per_slice_csv = os.path.join(
                args.out_dir, f"{base_name}_{SEGTHY_LABELS[label_id][0]}_slices.csv"
            )
            axis_map = {"x": 0, "y": 1, "z": 2}
            axis_idx = axis_map.get(args.distance_axis, 2)  # default z

            dists_mm = distances_from_cranial(
                pts_vox, spacing, seg.shape, axis_idx,
                affine=affine, cranial_edge=args.cranial_edge
            )

            df_sl = pd.DataFrame({
                "file": base_name,
                "structure": SEGTHY_LABELS[label_id][0],
                "label_id": label_id,
                "arc_length_mm": s_mm,
                "cross_section_mm2": areas_mm2,
                "distance_from_cranial_mm": dists_mm 
            })
            df_sl.to_csv(per_slice_csv, index=False)

    summary = {
        "file": base_name,
        "structure": SEGTHY_LABELS.get(label_id, (f"label_{label_id}", is_vessel))[0],
        "label_id": int(label_id),
        "is_vessel": bool(is_vessel),
        "volume_ml": float(volume_ml),
        "mean_area_mm2": mean_area,
        "std_area_mm2": std_area,
        "min_area_mm2": min_area,
        "max_area_mm2": max_area,
        "num_samples": int(num_samples),
        "spacing_x_mm": float(spacing[0]),
        "spacing_y_mm": float(spacing[1]),
        "spacing_z_mm": float(spacing[2]),
        "per_slice_csv": per_slice_csv
    }
    return summary

def detect_present_labels(seg):
    """Return the sorted list of non-zero labels found that match SEGTHY_LABELS."""
    found = sorted(set(int(l) for l in np.unique(seg) if l != 0))
    return [l for l in found if l in SEGTHY_LABELS]

def process_file(path, args):
    seg, spacing, affine = load_seg(path, override_spacing=args.override_spacing_mm)
    base = os.path.splitext(os.path.basename(path))[0]
    if base.endswith('.nii'):
        base = base[:-4]
    labels_present = detect_present_labels(seg)
    summaries = []
    if not labels_present:
        labels_present = [1] if (seg > 0).any() else []
    for lid in labels_present:
        name, is_vessel = SEGTHY_LABELS[lid]
        try:
            summary = process_label(seg, spacing, affine, lid, is_vessel, args, base)
            summaries.append(summary)

            if is_vessel:
                print(f"[OK] {base} | {name}: vol={summary['volume_ml']:.3f} mL, "
                      f"mean CSA={summary['mean_area_mm2'] if np.isfinite(summary['mean_area_mm2']) else np.nan:.1f} mm^2 "
                      f"(n={summary['num_samples']})")
            else:
                print(f"[OK] {base} | {name}: vol={summary['volume_ml']:.3f} mL")
        except Exception as e:
            print(f"[ERR] {base} | {name}: {e}")

    return summaries

# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute volumetrics + CSA for SegThy labels (thyroid, CCA, IJV) from .nii/.nii.gz.")
    parser.add_argument("in_dir", 
                        type=str, 
                        help="Input folder with .nii / .nii.gz files")
    parser.add_argument("--out_dir", 
                        type=str, 
                        default="segthy_results", 
                        help="Output folder for CSVs")
    parser.add_argument("--step_mm", 
                        type=float, 
                        default=2.0, 
                        help="Arc-length sampling interval (mm)")
    parser.add_argument("--plane_size_mm", 
                        type=float, 
                        default=25.0, 
                        help="Size of orthogonal sampling plane (mm)")
    parser.add_argument("--res_mm", 
                        type=float, 
                        default=0.5, 
                        help="In-plane sampling resolution (mm)")
    parser.add_argument("--slab_mm", 
                        type=float, 
                        default=0.0, 
                        help="Slab thickness along tangent (mm); 0 for single plane")
    parser.add_argument("--per_slice_out", 
                        action="store_true", 
                        help="Write per-slice CSA CSVs for vessel labels")
    parser.add_argument("--override_spacing_mm", 
                        type=float, 
                        nargs=3, 
                        default=None,
                        help="Override voxel spacing as three numbers (e.g., 0.12 0.12 0.12) if header is wrong")
    parser.add_argument("--distance_axis", type=str, 
                        default="z",
                        choices=["x", "y", "z"],
                        help="Axis used for cranial distance (default: z)")
    parser.add_argument("--cranial_edge", 
                        type=str, 
                        default="auto",
                        choices=["auto", "max", "min"],
                        help="Cranial extreme along the chosen axis: auto (use affine if available), max (highest index), min (lowest index)")


    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.nii"))) + sorted(glob.glob(os.path.join(args.in_dir, "*.nii.gz")))

    all_rows = []
    for f in files:
        rows = process_file(f, args)
        all_rows.extend(rows)

    out_csv = os.path.join(args.out_dir, "segthy_volumetrics_csa_summary.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nSummary written to: {out_csv}")

if __name__ == "__main__":
    main()