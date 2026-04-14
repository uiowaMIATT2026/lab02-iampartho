"""
generate_circles.py
====================
Generates two 2D circle images and saves them as .nrrd files.

NRRD (and NIfTI .nii.gz) are used instead of PNG because:
  - PNG has NO header fields for physical spacing, origin, or direction cosines.
  - NRRD/NIfTI store all three: space directions (spacing + direction cosine
    matrix), space origin, and space dimension name.
  - Medical image registration algorithms operate entirely in physical (mm)
    space, so this metadata is REQUIRED for correct results.

Image specifications (from assignment):
  img1.nrrd : 30 mm diameter circle, centered at (50 mm, 50 mm)
  img2.nrrd : 60 mm diameter circle, centered at (200 mm, 200 mm)

Coordinate convention:
  - Origin at (0, 0) mm
  - Spacing: 1.0 mm/pixel in both dimensions
  - Direction cosines: identity (standard anatomical axes)
  - Image size: 300 x 300 pixels  (covers 0..299 mm in each axis)
"""

import numpy as np
import SimpleITK as sitk


def generate_circle_image(
    size_pixels: int,
    spacing_mm: float,
    origin_mm: tuple,
    center_mm: tuple,
    diameter_mm: float,
) -> sitk.Image:
    """
    Create a 2D binary image with a filled circle.

    Parameters
    ----------
    size_pixels  : number of pixels along each axis (square image)
    spacing_mm   : physical size of one pixel in mm
    origin_mm    : (x, y) physical coordinate of the first pixel (0,0)
    center_mm    : (cx, cy) physical coordinate of the circle center in mm
    diameter_mm  : circle diameter in mm

    Returns
    -------
    SimpleITK Image (UInt8, values 0 or 255) with spacing/origin set.
    """
    radius_mm = diameter_mm / 2.0

    # Build a coordinate grid in PHYSICAL space (mm).
    # np.indices returns (row_indices, col_indices).
    # row index  → Y physical coordinate
    # col index  → X physical coordinate
    row_idx, col_idx = np.indices((size_pixels, size_pixels))

    x_phys = origin_mm[0] + col_idx * spacing_mm   # shape (N, N)
    y_phys = origin_mm[1] + row_idx * spacing_mm   # shape (N, N)

    # Euclidean distance from each pixel center to the circle center
    dist = np.sqrt((x_phys - center_mm[0])**2 + (y_phys - center_mm[1])**2)

    # Foreground (255) where distance ≤ radius, background (0) elsewhere
    arr = np.where(dist <= radius_mm, np.uint8(255), np.uint8(0))

    # ── Convert numpy array → SimpleITK image ─────────────────────────────
    # GetImageFromArray interprets the array as (rows, cols), which maps to
    # (Y, X) in SimpleITK's (X, Y) physical space.
    img = sitk.GetImageFromArray(arr)

    # SetSpacing takes (sx, sy) — the physical size of one pixel in each axis.
    img.SetSpacing([spacing_mm, spacing_mm])

    # SetOrigin takes (ox, oy) — physical coords of pixel index (0, 0).
    img.SetOrigin(list(origin_mm))

    # SetDirection takes a flattened row-major 2x2 matrix.
    # Identity means axis 0 of the array aligns with physical X,
    # and axis 1 aligns with physical Y — standard orientation.
    img.SetDirection([1.0, 0.0, 0.0, 1.0])

    return img


def main():
    size_for_circle_1    = 300          # pixels per axis (image region) circle 1
    spacing_for_circle_1 = 1.0          # mm per pixel (each image spacing per axis) for circle 1
    origin_for_circle_1  = (0.0, 0.0)  # physical origin for circle 1
    circle_1_center_physical = (200.0, 200.0) #Physical center (mm, mm) for the circle 1
    circle_1_diameter = 30.0 # circle 1 diameter in mm

    # ── Image 1: 30 mm circle centered at (50, 50) mm ─────────────────────
    img1 = generate_circle_image(
        size_pixels = size_for_circle_1,
        spacing_mm  = spacing_for_circle_1,
        origin_mm   = origin_for_circle_1,
        center_mm   = circle_1_center_physical,
        diameter_mm = circle_1_diameter,
    )
    sitk.WriteImage(img1, "img1.nrrd")
    print("Wrote img1.nrrd")
    print(f"  Size    : {img1.GetSize()} pixels")
    print(f"  Spacing : {img1.GetSpacing()} mm/pixel")
    print(f"  Origin  : {img1.GetOrigin()} mm")
    print(f"  Direction: {img1.GetDirection()}")


    size_for_circle_2 = 300 # pixels per axis (image region) circle 2
    spacing_for_circle_2 = 1.0 # mm per pixel (each image spacing per axis) for circle 2
    origin_for_circle_2 = (0.0, 0.0) # physical origin for circle 2
    circle_2_center_physical = (200.0, 200.0)  #Physical center (mm, mm) for the circle 2
    circle_2_diameter = 60.0 # circle 2 diameter in mm

    # ── Image 2: 60 mm circle centered at (200, 200) mm ───────────────────
    img2 = generate_circle_image(
        size_pixels = size_for_circle_2,
        spacing_mm  = spacing_for_circle_2,
        origin_mm   = origin_for_circle_2,
        center_mm   = circle_2_center_physical,
        diameter_mm = circle_2_diameter,
    )
    sitk.WriteImage(img2, "img2.nrrd")
    print("\nWrote img2.nrrd")
    print(f"  Size    : {img2.GetSize()} pixels")
    print(f"  Spacing : {img2.GetSpacing()} mm/pixel")
    print(f"  Origin  : {img2.GetOrigin()} mm")
    print(f"  Direction: {img2.GetDirection()}")

    # ── Verify NRRD round-trip: read back and check metadata ──────────────
    print("\n── Round-trip verification ──────────────────────────────────────")
    for fname in ("img1.nrrd", "img2.nrrd"):
        reloaded = sitk.ReadImage(fname)
        print(f"  {fname}: size={reloaded.GetSize()}  "
              f"spacing={reloaded.GetSpacing()}  "
              f"origin={reloaded.GetOrigin()}")


if __name__ == "__main__":
    main()
