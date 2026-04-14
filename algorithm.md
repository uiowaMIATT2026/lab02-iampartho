# Registration Algorithm

## Inputs
- **F(x)** — Fixed image: `img1.nrrd` (30 mm circle at (50, 50) mm)
- **M(x)** — Moving image: `img2.nrrd` (60 mm circle at (200, 200) mm)

---

## Step 1 — Define the Transform T(p)

Use a **Similarity2DTransform** with parameters **p = (scale, angle, tx, ty)**.

This transform maps every point in the fixed image to a point in the moving image:

```
T(p) :  x_fixed  →  x_moving  =  scale · R(angle) · (x_fixed − center) + center + t
```

Set the **initial parameters p** analytically from the images:
- `scale  = radius_moving / radius_fixed`  ≈ 2.0
- `angle  = 0`
- `tx, ty = centroid_moving − centroid_fixed`  = (150, 150) mm

---

## Step 2 — Interpolator

For each point `T(p)` computed above, the transformed coordinate lands between
pixels in the moving image. Use **linear interpolation** to compute the moving
image intensity M(T(p)) at that sub-pixel location.

---

## Step 3 — Metric S(p | F, M, T)

Compare the fixed image intensity F(x) against the interpolated moving image
intensity M(T(p)) at every pixel x.

Use **Mean Squared Error**:

```
S(p) = (1/N) · Σ [ F(x) − M(T(p)) ]²
```

S = 0 means perfect alignment. The optimizer's goal is to minimize S.

---

## Step 4 — Optimizer

Use **Gradient Descent** to update the transform parameters p:

```
p  ←  p  −  learning_rate · ∇S(p)
```

Repeat Steps 2 → 3 → 4 until S stops decreasing (convergence).

The optimizer returns the final parameters **p\*** that minimize S.

---

## Step 5 — Resample and Save

Apply the final transform T(p\*) to resample the moving image into the fixed
image space. Save the result as `registered.nrrd`.
