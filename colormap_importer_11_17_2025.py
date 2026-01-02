import os
import numpy as np
import SimpleITK as sitk
import re
from PIL import Image

# Mapping of RGBA colors to integer labels for segmentation
# Colors not intended for segmentation (e.g., fat, muscle) are mapped to background (label 0)
color_to_label = {
    (0, 0, 0, 255): 0,           # background
    (201, 255, 0, 255): 1,       # thyroid
    (255, 0, 156, 255): 2,       # common carotid artery (cca)
    (0, 207, 255, 255): 3,       # internal jugular vein (ijv)
    (255, 226, 0, 255): 0,       # thyroid
    (255, 0, 205, 255): 0,       # common carotid artery (cca)
    (0, 233, 255, 255): 0,       # internal jugular vein (ijv)
    (148, 113, 30, 255): 0,      # fat (ignored)
    (160, 131, 29, 255): 0,      # muscle (ignored)
    (255, 255, 255, 255): 0,     # white background (ignored)
    
    # Additional unmatched colors (ignored)
    (255, 0, 0, 255): 0,
    (0, 255, 200, 255): 0,
    (131, 255, 0, 255): 0,
    (194, 0, 255, 255): 0,
    (230, 255, 0, 255): 0,
    (0, 255, 246, 255): 0,
    (255, 0, 249, 255): 0,
    (0, 200, 255, 255): 0,
    (0, 147, 255, 255): 0,
    (0, 255, 147, 255): 0,
    (0, 255, 235, 255): 0,
    (58, 255, 0, 255): 0,
    (255, 0, 241, 255): 0
}

# Input directory containing PNG slices
input_dir = r"D:\Catalyst\Synthetic Data\Unreal_PNGS_11_17_2025\Slices"

# Output path for the final 3D segmentation volume
output_path = "20251203_Catalyst_segmented_volume.nii.gz"  # Can change to .nii.gz
# Tolerance for color matching (higher = more lenient)
tolerance = 0  # set to 0 to start

# Optional metadata for the volume (spacing in mm and origin in physical space)
spacing = (0.12, 0.12, 1.0)  # Pixel spacing: x, y, z
origin = (0.0, 0.0, 0.0)   # Origin of the volume

# Get a sorted list of all PNG files in the input directory
png_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])

# Initialize a list to hold each 2D label slice
volume_slices = []


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


png_files = sorted(
    [f for f in os.listdir(input_dir) if f.lower().endswith(".png")],
    key=natural_sort_key
)


def match_color_with_tolerance(rgba_array, target_color, tolerance=tolerance):
    """
    Returns a boolean mask where pixels match the target RGBA color within a given tolerance.
    This helps handle slight color variations due to compression or anti-aliasing.
    """
    diff = np.abs(rgba_array.astype(np.int16) - np.array(target_color, dtype=np.int16))
    return np.all(diff <= tolerance, axis=-1)


for filename in png_files:  # Process each PNG slice
    filepath = os.path.join(input_dir, filename)

    # Load image and convert to RGBA format
    try:
        img = Image.open(filepath).convert("RGBA")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue
    rgba_array = np.array(img)

    # Initialize an empty label slice
    label_slice = np.zeros(rgba_array.shape[:2], dtype=np.uint8)

    # Create a mask for each known color and assign the corresponding label
    matched_mask = np.zeros(rgba_array.shape[:2], dtype=bool)
    for rgba, label in color_to_label.items():
        mask = match_color_with_tolerance(rgba_array, rgba, tolerance)
        label_slice[mask] = label
        matched_mask |= mask  # Track which pixels were matched

    # Log any colors that were not matched to known labels
    unmatched_colors = rgba_array[~matched_mask]
    unique_unmatched = np.unique(unmatched_colors.reshape(-1, 4), axis=0)
    for color in unique_unmatched:
        print(f"Unmatched color {tuple(int(c) for c in color)} in {filename}")

    # Add the processed label slice to the volume
    volume_slices.append(label_slice)

# Stack all 2D slices into a 3D volume (z, y, x)
volume_array = np.stack(volume_slices, axis=0)

# Convert the NumPy array to a SimpleITK image
volume_image = sitk.GetImageFromArray(volume_array)

# Set spacing and origin metadata
volume_image.SetSpacing(spacing)
volume_image.SetOrigin(origin)

# Save the image to disk in NRRD format (or NIfTI if you change the extension)
sitk.WriteImage(volume_image, output_path)
print(f"Segmented volume saved to {output_path}")
