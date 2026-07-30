# Reference photo findings → default parameters

## Panel A 36×12 calibration set (fourth photo batch, 2026-07-30)

Five angles of the best-selling 36×12 panel — the calibration reference
for the 2D-import defaults. Measured/observed → encoded:

- Visible vertical edge ≈ 0.8–1″, then one continuous soft arc: defaults
  changed to **thickness 1.5″ / edge roll 0.5″** (was 2″/0.3″).
- No hard line anywhere from wall → roll → dome: the roll fades the
  crown in across its zone (smoothstep), the arc is curvature-blurred
  (~roll/3), and the crown knee uses the C1 smooth clamp.
- Side walls bow outward (never flat): `belly = 0.34·roll·sin(pi·t)`
  applied along the outward normal, zero at the base outline and at the
  welded top seam.
- Corner folds: convex outline corners (recorded at import) each carve a
  shallow diagonal dart along the inward bisector, scaled by roll radius
  and turn angle.
- Shading: the top field exports ANALYTIC vertex normals (from the
  height-field gradient) — averaged mesh normals showed streaks/beads on
  the glossy roll at grazing angles. The footprint raster also gets an
  opening pass (speckle-free boundary) and a wider margin (a 1-cell
  margin let binary_closing eat ~4 mm off every min-side edge — a bug
  inherited from the reference algorithm).

Still pending from the user: a straight-on edge photo with a tape
measure to confirm the 1.5″/0.5″ defaults against real dimensions.

### Third calibration round (viewport feedback: "lumpy, dented, dome
### not symmetrical", 2026-07-30)

Measured on the 36×12 at export settings: up to ~3 mm left/right height
asymmetry and crumpled corners. Root causes found and fixed:

- **Corner darts were drawn 6 mm off-position** (the dart grid used the
  old 1-cell raster margin instead of `MARGIN = 4`), shifting all four
  darts diagonally — deeper into the shoulder on two corners, half off
  the rim on the others. That was the "dented"/asymmetric look at the
  panel ends. Darts also softened (0.16·roll deep, was 0.38·roll — the
  photos show a faint tuck, not a groove) and only the dart field is
  blurred now, not the whole profile.
- **The EDT distance field quantized the outline to whole raster cells**:
  914.4 mm / res never divides evenly, so one edge sat on a cell boundary
  and the opposite edge mid-cell — the whole crown skewed up to half a
  cell per side. The raw distance field is now computed EXACTLY from the
  panel's true top-boundary segments (KDTree-accelerated point-segment
  distance); the raster only handles inside/outside decisions. Also kills
  stair-step scallops on diagonal/curved outlines.
- **Collar thinning made the shoulder wavy**: quantized-coordinate dedup
  kept collar points irregularly, so roll-zone chord sag varied with an
  irregular period (worst at preview res). Collar rows now keep full ring
  density — the "normal-pole beads" the thinning prevented died when the
  top switched to analytic normals.
- **Preview ≠ export**: cell-count sigmas (dist smoothing, knee blur)
  smoothed twice the millimetres at 4 mm preview res than at 2 mm export
  res. All cell-based sigmas are now rescaled by `REF_RES / res` so every
  resolution smooths the same physical distance.
- Membrane-tension upsample: bilinear → cubic + blur (gradient kinks from
  the coarse grid could ripple the shoulder); interior top grid centred
  in the footprint so symmetric outlines triangulate symmetrically.

After: left/right and front/back asymmetry ≤ 0.5 mm at export
(tessellation chords), preview silhouette matches export.

### Fourth calibration round (viewport feedback: bell-shaped sides,
### 2026-07-30)

User verdict on the third round: the dome top is right ("super soft,
rolls off very gently just like my photos") — do not touch it. But in a
lengthwise side view the shape belled: the dome rolled down to a waist
NARROWER than the panel, then the wall flared back out below it.

Cause: the side-wall belly weight was `sin(pi*t)` — pinned at the base
AND at the welded top seam, widest at mid-height. The dome therefore
landed on the pinch and the wall bulged out 4.3 mm beneath it.

Fix: belly weight is now `t^2` with amplitude `0.15*roll` — zero at the
base, monotone, maximal AT the seam. The widest point of the silhouette
is exactly where the dome's roll lands, and the wall tucks continuously
back under toward the base like real wrapped vinyl (photos: edges bulge
near the top, tuck at the bottom lip). A monotone weight cannot produce
a re-entrant waist at any parameter setting. The top height field is
untouched. Analytic wall normals updated for the new offset profile
(dr/dz = 2*amp*t/thickness).

That alone was NOT enough (user re-test still showed the bell). The
real driver was found by rendering the mesh with the user's own camera
(software rasterizer, `scratchpad` diagnostics) and isolating terms:

- **The crown fade made the shelf.** The smoothstep that zeroed the
  crown across the roll zone put a CONCAVE shelf directly above the
  CONVEX quarter-circle roll — concave-into-convex IS the bell, along
  every edge. The fade is gone: crown and roll drop are simply added.
  Both curves are concave-down, so the descent from dome to rim now has
  monotonically steepening slope — one continuous convex roll-over, at
  every parameter setting.
- **The crown is driven by the RAW (exact segment) distance.** Without
  the fade, the smoothed field's bleed past the boundary propped the
  rim up several mm (shallow-looking wrap). Raw distance is exactly 0
  at the outline, so the rim carries ~70% of the roll radius (the rest
  is knee/roll blur). The smoothed field still supplies gradient
  directions and near-edge falloffs.
- **The belly is one continuous field.** Displacing only the wall left
  the seam ring poking `belly_amp` past the undisplaced top surface
  just inside it — a hard lip that silhouetted as a waist ring. The
  same displacement (at t=1, fading inward over ~roll) now also moves
  the top vertices near the rim, tapering smoothly into the dome. All
  heights are sampled at original footprint positions before any
  horizontal displacement.
- **Rim pinholes closed**: boundary-band triangles (ring/collar only)
  are exempt from the centroid cull margin (cutout bridges still die on
  the exact-zero distance inside holes). Export is watertight: 0 open
  edges at both resolutions.

### Sixth round (user headboard render: NO lines may ever appear on
### the padded face, even soft ones)

Blurring the min-distance field (fifth round) widened the medial-axis /
corner-bisector creases into soft wide lines -- still visible on a
glossy face under studio lighting. Structural fix: the interior of the
crown's drive field is now the **Poisson/Spalding wall distance**
(`meshops.poisson_wall_distance`): solve the inflated-membrane equation
grad^2 h = -1 (h = 0 at every edge), then normalize with Spalding's
formula d = sqrt(|grad h|^2 + 2h) - |grad h|. This field:

- is smooth everywhere inside (a membrane has no fold lines) -- the
  face carries NO crease at any setting, by construction;
- equals the true wall distance exactly on a strip and to first order
  near every straight wall, so the calibrated near-edge rise, the roll,
  and the convex bell-free roll-over are unchanged (the raw/interior
  depth blend from round five still guards the rim);
- naturally lowers corner pockets and narrow tails (it IS the membrane
  physics the tension factor approximates).

Implementation notes: solved on a coarsened grid (odd pooling factor +
even padding so mirror symmetry survives; cubic upsample; red-black SOR
polish sweeps at full res). The raster lattice is now centred on the
panel (cell CENTRES mirror-symmetric, stamping by rounding), which
makes every grid-derived field exactly symmetric for symmetric
outlines. Engine v6.

### Seventh round (user's Pedal.DXF failed to import)

The file draws one outline as 2 LINEs + 2 ARCs -- separate OPEN
entities meeting end-to-end, which most CAD packages export. The
importer only accepted individually-closed entities (and skipped LINE
entirely). Added `_chain_fragments`: open entities are stitched into
closed loops by endpoint matching (1 mm tolerance); dead-end fragments
(dimension leaders, centerlines) are dropped as before. LINE joined
the accepted entity kinds; arc flattening tightened to 0.1 mm sagitta
(0.5 mm made ~20 mm facets that caught specular light on the roll).

Fixing that exposed a deep watertightness bug on straight/faceted
outlines: densified straight stretches are EXACTLY collinear, and
unconstrained Delaunay caps/tops handle the degenerate hull strip
unreliably (qhull-version/coordinate dependent), tearing the cap/wall
weld (26 open edges on the pedal slab; the plain rectangle only worked
by luck). Fix trio, applied identically to caps and the pillowed top:

- `_nudge_collinear`: exactly-collinear ring points get a deterministic
  0.01-0.02 mm inward dogleg (varied per point -- a constant offset
  just recreates a collinear row at the new depth);
- a dense collar row ~1.2 mm inside the boundary locks every rim
  segment in as a Delaunay edge (the pillow top already had collars);
- boundary-only triangles are culled by an EXACT inside test (shapely
  contains for the caps; winding-number against the true oriented
  outline segments for the top -- `meshops.winding_inside`), because
  the raster's bilinear distance cannot resolve which side of the
  outline a rim-hugging sliver sits on.

Result: rectangle, circle and pedal are watertight (0 open edges) at
preview and export; regression test builds a LINE+ARC pill DXF and
asserts chaining + watertightness end-to-end.

### Fifth round (user: shape is right, but peak/corner LINES on top)

Driving the crown with the raw min-distance printed the field's slope
creases -- the medial-axis "spine" and the corner bisectors -- onto the
top as visible lines (the crown never fully saturates on panel-sized
parts, so the creases never flatten out). Fix: the drive field is now a
DEPTH BLEND -- exact segment distance out to the roll zone (tight rim,
no bell, no bleed), cross-faded into the gaussian-smoothed field over
~2.5 blur radii (the interior look the user originally praised). The
blend weight is a smoothstep of the raw distance itself, so it is
shape-generic and cannot reintroduce a concave shelf; convexity, wall
monotonicity and watertightness re-verified after the change.

### Second calibration round (user viewport feedback on the above)

- Wall ribbing + rim sawtooth were shading artifacts: walls now get
  belly-aware ANALYTIC normals (offset-surface math) that agree with the
  top field's normals at the seam — no averaged-normal wobble anywhere
  on a 2D part.
- Plateau ripples came from the membrane-tension field's coarse-grid
  bilinear upsample; it is now smoothed on the coarse grid first.
- Deliverable focus: exported GLB is now glTF-standard (+Y up, meters)
  with planar UVs, analytic normals, no degenerate faces and verified
  winding — importable into rendering software with zero cleanup. STL
  stays mm/Z-up for CAD.

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

## Edge roll rework (user viewport feedback, 2026-07-30)

Feedback on the first black-leather renders: top edges nearly as hard as
the (wooden) bottom edges; corners malformed; crown shoulder still a hard
line. Root causes were all in the discrete fillet approach, so it was
replaced wholesale:

- The 2D extrusion is now an honest **sharp slab** (the board+foam core);
  the wrapped edge is generated by the pillow stage as a quarter-circle
  **roll height field** blended with the crown (`edge_roll` per part,
  from the import dialog's roundover). Edge + crown are one continuous
  C1 surface: no tangent break at the rim, and corners fold down
  naturally (no inset geometry left to malform).
- The crown's saturation `clip()` was replaced with a C1 smooth clamp
  (`meshops._smooth_clamp`), removing the hard ridge where the bulge
  flattens; deep centers still reach exactly the calibrated crown.

Calibration TODO: a straight-on edge-profile photo with a tape measure
would pin down the real default roll radius (currently 0.3 in).

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

## Tapering / irregular panels — tension (teardrop petal photos)

- On the teardrop petal the crown is tallest at the wide bulb and fades
  toward the narrow tail: less width = less room to loft = a tenser,
  lower crown. The nearest-edge distance field already reproduces the
  gross effect (tail distance is small), but it can't feel a *taper* or
  the all-sides tension of a rounded region.
- Added an inflated-membrane solve (grad^2 h = -1, pinned at every edge;
  `meshops.membrane_tension`). Its effective half-width equals the
  nearest-edge distance on a straight strip, so rectangles are unchanged,
  but on rounded/tapering shapes it is lower — the physical "tension"
  pull. Exposed as a **Tension** parameter (default 0.7); 0 recovers the
  original validated distance-field crown exactly.

## Viewer material (shape readability)

- Switched the pillow material to **black leather** (dark base + clearcoat
  sheen + grain bump under image-based studio lighting). A dark, slightly
  glossy surface reads curvature through the moving specular highlight far
  better than a matte light colour — matches the black-vinyl product.
- Smoother bulge: the crown profile is blurred past its saturation knee so
  the shoulder has no crease, and the retriangulated top seeds collar
  rings inside the boundary so the roll-over shades without serration.

If future photos disagree (different foam, tighter wrap), tune the global
parameters in the UI first; only change the code defaults in
`backend/models.py::PillowParams` once a value is confirmed across several
panels.
