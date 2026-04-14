"""
register_circles.py
====================
Registers img2.nrrd (moving) to img1.nrrd (fixed) using:
  - Similarity2DTransform  (scale + rotation + translation, 4 DOF)
  - Mean Squared Error metric
  - Regular Step Gradient Descent optimizer
  - Multi-resolution pyramid (3 levels)

ITK resampling convention (fixed -> moving):
    p_moving = s * R(theta) * (p_fixed - center) + center + t

Ground truth for this pair:
    scale = 2.0,  theta = 0 deg,  t = (150, 150) mm

Usage
-----
    python register_circles.py

Output
------
    registered.nrrd   —  moving image resampled into fixed image space
"""

import numpy as np
import SimpleITK as sitk


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load images
# ─────────────────────────────────────────────────────────────────────────────

fixed  = sitk.ReadImage("img1.nrrd", sitk.sitkFloat32)
moving = sitk.ReadImage("img2.nrrd", sitk.sitkFloat32)

print("Fixed  image: img1.nrrd  "
      f"size={fixed.GetSize()}  spacing={fixed.GetSpacing()}  origin={fixed.GetOrigin()}")
print("Moving image: img2.nrrd  "
      f"size={moving.GetSize()}  spacing={moving.GetSpacing()}  origin={moving.GetOrigin()}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Analytically initialise the transform from image moments
#
# Rather than starting gradient descent from a random point (which risks
# landing in a local minimum), we compute a closed-form starting estimate
# from first-order image moments (centroid) and foreground area (radius).
#
# Centroid — intensity-weighted mean pixel position, converted to mm:
#     x_phys = origin_x + col_centroid * spacing_x
#
# Radius — inferred from foreground pixel area:
#     A = n_pixels * spacing_x * spacing_y  →  r = sqrt(A / pi)
#
# From the ITK convention  p_moving = s*(p_fixed - center) + center + t
# with center = c_fixed, theta = 0:
#     s = r_moving / r_fixed      (scale maps circle sizes)
#     t = c_moving - c_fixed      (translation maps circle centers)
# ─────────────────────────────────────────────────────────────────────────────

def centroid_mm(img):
    arr = sitk.GetArrayFromImage(img)           # shape: (rows, cols)
    total = float(arr.sum())
    rows, cols = np.indices(arr.shape)
    rc = float((rows * arr).sum() / total)      # row centroid (pixels)
    cc = float((cols * arr).sum() / total)      # col centroid (pixels)
    sp, org = img.GetSpacing(), img.GetOrigin()
    return np.array([org[0] + cc * sp[0],       # x physical (mm)
                     org[1] + rc * sp[1]])       # y physical (mm)

def radius_mm(img):
    arr = sitk.GetArrayFromImage(img)
    sp  = img.GetSpacing()
    return float(np.sqrt((arr > 0).sum() * sp[0] * sp[1] / np.pi))


c_f = centroid_mm(fixed);    c_m = centroid_mm(moving)
r_f = radius_mm(fixed);      r_m = radius_mm(moving)

init_tx = sitk.Similarity2DTransform()
init_tx.SetCenter(c_f.tolist())             # centre of rotation = fixed centroid
init_tx.SetScale(r_m / r_f)                # s = r_moving / r_fixed  ≈ 2.0
init_tx.SetAngle(0.0)
init_tx.SetTranslation((c_m - c_f).tolist())

print(f"\nAnalytical init:  scale={init_tx.GetScale():.4f}  "
      f"t=({init_tx.GetTranslation()[0]:.2f}, {init_tx.GetTranslation()[1]:.2f}) mm")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Configure ImageRegistrationMethod and run
#
# SimpleITK's ImageRegistrationMethod takes the fixed image, moving image,
# transform, metric, and optimizer, then runs the full optimization loop:
#   1. At each iteration, resample moving into fixed space via current transform.
#   2. Evaluate MSE between fixed and resampled moving.
#   3. Compute gradient of MSE w.r.t. transform parameters.
#   4. Update parameters: params -= learning_rate * gradient.
#   5. Repeat until convergence or max iterations.
# ─────────────────────────────────────────────────────────────────────────────

R = sitk.ImageRegistrationMethod()

# Metric: Mean Squared Error  Σ(I_fixed(x) - I_moving(T(x)))²
# Good choice here because both images are binary with the same intensity scale.
R.SetMetricAsMeanSquares()
R.SetMetricSamplingStrategy(R.NONE)         # use all pixels (deterministic)

# Interpolator: linear sub-pixel interpolation during resampling
R.SetInterpolator(sitk.sitkLinear)

# Optimizer: Regular Step Gradient Descent
# estimateLearningRate=EachIteration balances mm-scale translation and
# unitless scale automatically — no manual per-parameter tuning needed.
R.SetOptimizerAsRegularStepGradientDescent(
    learningRate               = 4.0,
    minStep                    = 1e-4,
    numberOfIterations         = 500,
    gradientMagnitudeTolerance = 1e-6,
    estimateLearningRate       = R.EachIteration,
)
R.SetOptimizerScalesFromPhysicalShift()

# Multi-resolution pyramid: coarse (4x) → medium (2x) → full (1x)
# Coarse levels smooth out local minima; fine level achieves sub-pixel accuracy.
R.SetShrinkFactorsPerLevel([4, 2, 1])
R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])   # Gaussian sigma in mm
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# inPlace=True: optimizer writes results directly into init_tx
R.SetInitialTransform(init_tx, inPlace=True)

R.Execute(fixed, moving)        # <— runs the full gradient descent loop


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Resample moving into fixed space using the optimized transform
# ─────────────────────────────────────────────────────────────────────────────

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)          # output grid = fixed image grid
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0.0)
resampler.SetTransform(init_tx)             # init_tx now holds optimized params
registered = resampler.Execute(moving)

sitk.WriteImage(registered, "registered.nrrd")
print("Saved -> registered.nrrd")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Report results
# ─────────────────────────────────────────────────────────────────────────────

# Dice coefficient and IoU
a = sitk.GetArrayFromImage(fixed)      > 0.5
b = sitk.GetArrayFromImage(registered) > 0.5
inter = float((a & b).sum())
dice  = 2 * inter / (a.sum() + b.sum())
iou   = inter / float((a | b).sum())

print("\n── Results ─────────────────────────────────────────────────────")
print(f"  Scale       : {init_tx.GetScale():.6f}     (ground truth: 2.0)")
print(f"  Angle       : {np.degrees(init_tx.GetAngle()):.6f} deg  (ground truth: 0.0)")
print(f"  tx          : {init_tx.GetTranslation()[0]:.4f} mm   (ground truth: 150.0 mm)")
print(f"  ty          : {init_tx.GetTranslation()[1]:.4f} mm   (ground truth: 150.0 mm)")
print(f"  |Δ scale|   : {abs(init_tx.GetScale() - 2.0):.6f}")
print(f"  |Δ tx|      : {abs(init_tx.GetTranslation()[0] - 150.0):.4f} mm")
print(f"  |Δ ty|      : {abs(init_tx.GetTranslation()[1] - 150.0):.4f} mm")
print(f"  Dice        : {dice:.4f}")
print(f"  IoU         : {iou:.4f}")
