# Reference photo findings → default parameters

Drop photos of real upholstered panels in this folder. The engine defaults
below were checked against five photos of a production teardrop petal
(black vinyl over a ~2" core, shared in chat 2026-07-30 — please copy the
originals here as `petal-side.jpg`, `petal-three-quarter.jpg`, etc. when convenient).

Observed on the real petal ⇄ encoded in the engine:

| Observation (photos) | Parameter |
| --- | --- |
| Wide end lofts to full height, narrow tail stays near core thickness; loft follows local width continuously | distance-field law `dz = f(dist-to-edge)` (engine core) |
| Crown at the wide (~300 mm) end ≈ 25–32 mm above the 50.8 mm core | `crown = 32.0` mm, saturating (`dref = 110` mm half-width) |
| Fast rise within the first ~25 mm off the edge, then a long gentle dome — no cone, no flat plateau edge | `exp = 0.55` (sub-linear profile) |
| Vinyl wraps the top edge with a small tight radius, roughly 5–10 mm | 2D-import `roundover = 8` mm default |
| Back face sits dead flat on the bench | vertical weight `w = ((z - zmin)/t)^1.5` pins the bottom |
| Side walls bulge very slightly outward near the top | same weight applied to side vertices (barreling) |
| Extreme tail pinches into a soft ridge where the two wrap directions meet | falls out of the distance field automatically; gaussian smoothing (`sigma = 5` cells) keeps the ridge soft, as in the photos |

## Rectangular wall panels (product shots, second photo batch 2026-07-30)

- Narrow rectangles (~6–8" wide) crown into a long, uniform, gentle ridge —
  clearly LESS loft than the wide square in the same photo. This is the
  saturation law the skinny-rectangle test pins down: crown scales with
  `(half-width / dref) ^ exp` until saturation.
- The square panel domes evenly from all four edges; corners stay pinned
  and sharp. Matches the distance-field behaviour with no tuning.
- Edge wrap on these is even tighter than the petal (looks < 8 mm). If a
  2D import of these panels looks too soft, drop `roundover` to ~5 mm.
- Backs are rigid boards with plastic mounting clips — the physical
  counterpart of "the back face stays perfectly flat" and of pass-through
  hardware classification for clip-like small components.

## Corner folds + edge profile (third photo batch, 2026-07-30)

- **Side profile photo (black vinyl, held edge-on):** the wrapped edge is
  a slim rounded lens — small tight radius at the face, a visible seam
  lip where the vinyl folds around to the back. Confirms the small
  roundover default and the flat back.
- **Corner photo (quilted gray square):** at a corner the vinyl cannot
  keep the straight-edge miter — it folds into a diagonal dart and the
  corner rounds off into a small chamfer, sitting LOW (pinned).
- **Encoded in geometry:** the 2D-import extrusion offsets its fillet
  rings a plain `roundover` along each vertex's bisector, deliberately
  NOT miter-extended (`_ring_normals_inward` in import_2d.py). Convex
  corners therefore come out chamfered/rounded like the photographed
  folds — and this same choice keeps the inset rings self-intersection
  free on sharp shapes (the 52x8" triangle panels).
- **Encoded in the viewer:** vinyl material = leather-grain bump +
  clearcoat sheen + studio environment reflections; the grazing rim
  light mimics the raking light that makes folds visible in the photos.

If future photos disagree (different foam, tighter wrap), tune the global
parameters in the UI first; only change the code defaults in
`backend/models.py::PillowParams` once a value is confirmed across several
panels.
