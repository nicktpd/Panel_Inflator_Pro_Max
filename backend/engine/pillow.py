"""Core pillow displacement engine.

Port of the validated reference algorithm (see the build spec appendix; it
was proven against the production FlowerBoard geometry). Given one panel --
a connected mesh component with a flat top -- it:

1. rasterizes the xy footprint (sampling ON the top faces, because CAD
   tops have giant triangles with no interior vertices),
2. computes a smoothed interior distance-to-boundary field,
3. builds the crown profile dz = crown * clip(dist/dref, 0, 1)^exp,
4. deletes the flat-top faces and REtriangulates the top from the kept
   boundary ring plus a fresh interior grid (never subdivide: naive
   subdivide_to_size on CAD slivers explodes 4:1 forever and OOMs),
5. lifts the new top by the profile, displaces the kept side/fillet
   vertices by dz * w where w pins the bottom flat and lets the side
   walls barrel outward with height,
6. welds the new top's ring to the displaced old ring so the seam is
   exactly closed, and merges everything into one vertex/face set.

Phase 3 hook: a loft-multiplier mask grid is multiplied into the profile
before sampling, then lightly re-blurred so brush strokes never leave
hard creases (upholstery has no sharp interior lines).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay

from . import meshops

# Fixed seed: the footprint sampling is random, but previews/exports and
# cache hashes must be reproducible for identical inputs.
DEFAULT_SEED = 12345

# Interior points closer than this (mm) to the boundary are not gridded;
# the ring vertices already define the surface there.
GRID_INSET_FACTOR = 1.0  # inset = grid_step * factor

# Triangles whose centroid is closer than this (mm) to the outside are
# culled: they bridge concavities or holes in the outline.
CULL_DIST = 0.5

# Extra smoothing (in grid cells AT EXPORT RES) applied to the profile
# after a painted mask is multiplied in, so mask edges never crease the
# surface.
MASK_BLUR_SIGMA = 1.5

# Always-on light smoothing (grid cells AT EXPORT RES) of the finished
# profile. The crown formula has a slope discontinuity at its saturation
# knee (where dist/dref reaches 1 and the surface stops rising), which
# reads as a faint crease line around the shoulder of the bulge. A gentle
# blur of the profile after the nonlinearity rounds that knee into a
# smooth shoulder without moving the peak -- "upholstery has no sharp
# interior lines" (spec section 5).
KNEE_BLUR_SIGMA = 1.6

# Reference raster resolution the cell-based smoothing amounts are
# calibrated at (the export resolution). Sigmas given in cells are scaled
# by REF_RES / res before use, so they cover the same PHYSICAL distance
# at every resolution -- without this, the 4 mm preview smoothed twice as
# many millimetres as the 2 mm export and previewed a visibly softer,
# wider shoulder than the exported part ("preview doesn't match export").
REF_RES = 2.0


def pillow_panel(
    pv: np.ndarray,
    pf: np.ndarray,
    *,
    crown: float = 32.0,
    dref: float = 110.0,
    exp: float = 0.55,
    sigma: float = 5.0,
    w_exp: float = 1.5,
    tension: float = 0.7,
    edge_roll: float = 0.0,
    corners: list[list[float]] | None = None,
    res: float = 2.0,
    grid_step: float = 6.0,
    mask: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
    return_normals: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the pillow displacement to one panel.

    Parameters
    ----------
    pv, pf : the panel mesh (N x 3 float vertices, M x 3 int faces).
        Not modified; copies are made.
    crown, dref, exp, sigma, w_exp : see models.PillowParams.
    edge_roll : radius (mm) of the wrapped-edge roll for panels whose
        source geometry has SHARP top edges (2D-imported slabs). The roll
        is generated as part of the top height field (see
        meshops.edge_roll_drop), so edge + crown form one continuous
        surface. Also enables the outward side-wall belly. Leave 0 for
        STL parts whose CAD already models fillets.
    corners : optional [x, y, bisx, bisy, turn_deg] convex outline
        corners (from import_2d); each gets a subtle vinyl fold dart
        carved along its inward bisector, like the reference photos.
    res : raster resolution in mm/cell (2.0 export, 4.0 preview).
    grid_step : spacing in mm of the regenerated top grid (6 export,
        ~10 preview). Coarser = fewer triangles, faster, softer detail.
    mask : optional loft-multiplier field. Either a dict
        {"grid": 2d array, "xmin": float, "ymin": float, "res": float}
        (world-anchored -- painted at any resolution, resampled here by
        world position) or a bare 2d array assumed to span this panel's
        raster. 1.0 = normal loft, 0.0 = flat, >1 = extra loft.
    seed : RNG seed for footprint sampling (fixed for reproducibility).

    Returns
    -------
    (vertices, faces) of the pillowed panel, float64/int64. If the panel
    has no detectable flat top, returns copies of the input unchanged.
    """
    pv = np.asarray(pv, dtype=np.float64).copy()
    pf = np.asarray(pf, dtype=np.int64)
    rng = np.random.default_rng(seed)

    zmin, zmax = float(pv[:, 2].min()), float(pv[:, 2].max())
    thick = zmax - zmin
    if thick <= meshops.TOP_TOL or crown <= 0.0:
        if return_normals:
            return pv, pf.copy(), meshops.topology_vertex_normals(pv, pf)
        return pv, pf.copy()

    vtop = meshops.top_vertex_mask(pv, zmax)
    if not (vtop[pf].all(axis=1)).any():
        # No flat-top faces at all: nothing we can pillow safely.
        if return_normals:
            return pv, pf.copy(), meshops.topology_vertex_normals(pv, pf)
        return pv, pf.copy()

    # --- footprint raster + interior distance + crown profile ---
    # dist_raw: geometry truth (0 in holes/outside) for culling + inset.
    # dist: smoothed, drives the crown profile shape.
    # Cell-count sigmas are calibrated at export res; rescale so every
    # resolution smooths the same physical distance (preview == export).
    cell = REF_RES / res
    fp = meshops.rasterize_footprint(pv, pf, res, rng)
    segs = meshops.top_boundary_segments(pv, pf, zmax)
    dist_raw, dist = meshops.distance_field(fp, sigma * cell, segments=segs)

    # The crown's drive field: EXACT segment distance near the rim,
    # cross-faded into the gaussian-smoothed field with depth.
    #   * Near the edge the raw field is what makes the wrap right: it is
    #     exactly 0 at the outline, so the crown is truly pinned there
    #     and the edge roll's full drop survives (the smoothed field's
    #     bleed past the boundary propped the rim up several mm -- a
    #     visibly taller edge wall).
    #   * Deep inside, the raw min-distance field has slope CREASES along
    #     the medial axis (where the nearest edge switches -- the "spine"
    #     down a rectangle's middle) and along corner bisectors. Because
    #     the crown never fully saturates on panel-sized parts, driving
    #     the crown with raw distance printed those creases onto the top
    #     as visible peak/diagonal lines -- and BLURRING the field only
    #     widens a crease into a soft line (render feedback: even soft
    #     lines are wrong on the padded face). The interior field is
    #     therefore the Poisson/Spalding wall distance
    #     (meshops.poisson_wall_distance): the inflated-membrane solution
    #     normalized to match wall distance near every edge. It is smooth
    #     everywhere by construction -- a membrane has no fold lines --
    #     so the face carries NO crease at any blur setting, while the
    #     calibrated near-edge rise is preserved (the field is exact on a
    #     strip and first-order exact along every straight wall).
    # The blend starts past the roll zone and completes over ~2.5x the
    # blur radius, i.e. deep enough that the discrete solve's near-wall
    # error never touches the rim, so no concavity (bell) can be
    # reintroduced. Shape-generic: works for any outline, incl. holes.
    roll = min(edge_roll, thick * 0.6) if edge_roll > 0.0 else 0.0
    sigma_mm = sigma * cell * res  # transition scale in mm (res-invariant)
    d_int = meshops.poisson_wall_distance(fp)
    if d_int is None:
        d_int = dist  # degenerate domain: fall back to the smoothed EDT
    else:
        # Iron out the coarse solve's residual boundary-staircase wobble
        # (curved outlines quantize to ~6 mm blocks; the wobble showed as
        # wrinkles in the bulb shoulder highlight). ~6 mm physical blur,
        # res-invariant; the membrane field is smooth so this changes
        # nothing else, and the raw/interior blend means d_int is only
        # ever used well inside the rim where the blur has no edge-dip.
        d_int = ndimage.gaussian_filter(d_int, sigma=3.0 * cell)
    inner = max(roll, 3.0 * res)
    outer = inner + 2.5 * max(sigma_mm, res)

    # Tension: pull the crown down where the membrane is more constrained
    # than the nearest-edge distance implies (tapers, pinches, tips). The
    # factor equals 1 on straight panels, so calibration is preserved.
    tau_s = None
    if tension > 0.0:
        tau_s = meshops.membrane_tension(fp)
        if sigma > 0:
            tau_s = ndimage.gaussian_filter(tau_s, sigma=max(sigma * cell * 0.5, 1.0))

    mask_grid = _mask_on_raster(mask, fp)
    if mask_grid is not None:
        # Brush-mask smoothing so painted strokes never crease the
        # surface (upholstery has no hard interior lines).
        mask_grid = ndimage.gaussian_filter(
            mask_grid, sigma=(KNEE_BLUR_SIGMA + MASK_BLUR_SIGMA) * cell
        )

    # 1D blurred roll-drop table. The wrapped-edge roll is a pure
    # function of the exact edge distance, so it is precomputed as a 1D
    # curve: quarter circle, continued at -roll for d < 0 (the wall),
    # gaussian-blurred by ~roll/3 to round BOTH of its curvature
    # discontinuities (wall->arc and arc->crown). Physical units
    # throughout -- identical at preview and export resolution.
    drop_tab = None
    if roll > 0.0:
        sig_mm = max(roll / 3.0, 2.0)
        step = 0.05
        pad = 5.0 * sig_mm
        daxis = np.arange(-pad, roll + pad, step)
        curve = np.where(
            daxis < 0.0,
            -roll,
            np.sqrt(np.maximum(roll**2 - (roll - np.minimum(daxis, roll)) ** 2, 0.0))
            - roll,
        )
        curve = ndimage.gaussian_filter1d(curve, sigma=sig_mm / step)
        drop_tab = (daxis, curve)

    # Corner fold darts: real vinyl can't wrap a convex corner flat -- it
    # gathers into a short diagonal crease along the corner bisector
    # (reference photos). Kept as a raster field (smooth gaussians
    # bilinear-sample cleanly). Grid world coordinates use the raster's
    # MARGIN convention (an off-by-MARGIN here shifts every dart
    # diagonally -- asymmetric crumpled corners).
    darts_f = None
    if roll > 0.0 and corners:
        gy_w = fp.ymin + (np.arange(fp.grid.shape[0]) - meshops.MARGIN) * res
        gx_w = fp.xmin + (np.arange(fp.grid.shape[1]) - meshops.MARGIN) * res
        GX, GY = np.meshgrid(gx_w, gy_w)
        darts = np.zeros(fp.grid.shape)
        for cx, cy, bx, by, turn in corners:
            sharp = min(float(turn) / 90.0, 1.5)
            depth = 0.16 * roll * sharp
            length = 2.2 * roll
            width = 0.5 * roll
            along = (GX - cx) * bx + (GY - cy) * by
            across = -(GX - cx) * by + (GY - cy) * bx
            # Groove centred just past the roll zone so it reads on the
            # visible shoulder, not swallowed by the roll's own drop.
            dart = -depth * np.exp(
                -0.5 * ((along - 1.2 * roll) / (0.6 * length)) ** 2
                - 0.5 * (across / width) ** 2
            )
            dart[along < 0] = 0.0
            darts += dart
        darts_f = ndimage.gaussian_filter(darts, sigma=max(1.0 * cell, 0.5))

    def _H(xy):
        """The pillow surface: height above the slab top at world xy.

        Evaluated PER POINT from the EXACT segment distance -- never by
        sampling a rasterized profile. Bilinear sampling of the steep
        near-rim profile from the raster beat (moired) against curved
        outlines: the lattice-vs-curve alignment drifts along the rim, so
        the seam height wobbled ~0.2-0.5 mm quasi-periodically -- the
        "bumpy rim" on the petal bulb (straight edges, being
        lattice-aligned, never showed it). Raster fields only contribute
        where they are intrinsically smooth: the membrane interior field,
        tension, painted masks, darts. Roll drop and crown come from
        analytic/1D forms of the exact distance, so the rim is exact for
        every outline shape at every resolution.
        """
        if len(segs):
            d = meshops._dist_to_segments(xy, segs, res)
        else:
            d = fp.sample(dist_raw, xy)
        w = np.clip((d - inner) / max(outer - inner, 1e-9), 0.0, 1.0)
        # Quintic (C2) blend: the cubic smoothstep is only C1, and its
        # curvature jump at both band edges drew faint offset-contour
        # lines around the dome at grazing angles.
        w = w * w * w * (w * (6.0 * w - 15.0) + 10.0)
        d_c = d * (1.0 - w) + fp.sample(d_int, xy, order=3) * w
        if tau_s is not None:
            d_c = d_c * ((1.0 - tension) + tension * fp.sample(tau_s, xy, order=3))
        h = meshops.crown_profile(d_c, crown, dref, exp)
        if mask_grid is not None:
            h = h * fp.sample(mask_grid, xy, order=3)
        if drop_tab is not None:
            h = h + np.interp(d, drop_tab[0], drop_tab[1])
        if darts_f is not None:
            h = h + fp.sample(darts_f, xy, order=3)
        return h

    # --- delete flat top, keep sides/fillets/bottom ---
    keep = pf[~vtop[pf].all(axis=1)]
    # Boundary ring: top-plane vertices still referenced by kept faces.
    ring = np.unique(keep[vtop[keep].any(axis=1)])
    ring = ring[vtop[ring]]
    ring_xy = pv[ring, :2]

    # --- new domed top: ring + collar + interior grid, Delaunay, cull ---
    # The boundary ring is dense (~2 mm, so the side walls stay smooth),
    # but a coarse interior grid alone leaves long sliver triangles in the
    # band between them -- their alternating normals read as a serrated
    # line right where the crown rolls over the edge. Seed a couple of
    # concentric "collar" rings just inside the boundary (spaced like the
    # grid) so that band is filled with well-shaped triangles and the
    # shoulder shades smoothly.
    collar = _collar_points(ring_xy, fp, dist_raw, dist, grid_step)

    inset = grid_step * GRID_INSET_FACTOR
    # Centre the grid within the PANEL's own span (pv extremes, not the
    # raster origin -- fp.xmin is the first cell centre, offset from the
    # outline corner by the lattice-centring slack): anchoring off-centre
    # puts the last row/column a partial step from the far edge, so a
    # mirror-symmetric outline got an asymmetric triangulation (and
    # asymmetric chord error on the dome). Centred, symmetric shapes mesh
    # symmetrically.
    pxmin, pymin = pv[:, 0].min(), pv[:, 1].min()
    gx0 = pxmin + ((pv[:, 0].max() - pxmin) % grid_step) / 2.0
    gy0 = pymin + ((pv[:, 1].max() - pymin) % grid_step) / 2.0
    gx, gy = np.meshgrid(
        np.arange(gx0, pv[:, 0].max() + 1e-9, grid_step),
        np.arange(gy0, pv[:, 1].max() + 1e-9, grid_step),
    )
    gp = np.column_stack([gx.ravel(), gy.ravel()])
    gp = gp[fp.sample(dist_raw, gp) > inset]

    pts2d = np.vstack([ring_xy, collar, gp]) if len(collar) else np.vstack([ring_xy, gp])
    if len(pts2d) < 3:
        if return_normals:
            return pv, pf.copy(), meshops.topology_vertex_normals(pv, pf)
        return pv, pf.copy()
    tri = Delaunay(pts2d)
    cent = pts2d[tri.simplices].mean(axis=1)
    cd = fp.sample(dist_raw, cent)
    keep_tri = cd > CULL_DIST
    # Triangles made only of ring/collar vertices hug the boundary by
    # construction (tight convex corners), so the centroid margin doesn't
    # apply -- culling them left pinholes at the rim corners (open edges
    # in the export). Genuine hole-spanning bridges are still culled:
    # away from a cutout's rim the raw distance is exactly 0.
    # Boundary-band triangles (ring/collar only) hug the rim, where the
    # raster's bilinear distance cannot decide which side of the outline
    # a centroid is on: legitimate corner fans sit a few tenths of a mm
    # INSIDE, while the multi-level sliver "flaps" the collinearity nudge
    # leaves along the convex-hull chords sit just OUTSIDE (kept flaps
    # make top boundary edges the walls don't have = open seams). Decide
    # those by an exact winding-number test against the true outline
    # segments; everything else keeps the centroid-distance cull.
    boundary_band = (tri.simplices < len(ring_xy) + len(collar)).all(axis=1)
    if len(segs) and boundary_band.any():
        near_rim = boundary_band & ~keep_tri
        idx = np.nonzero(near_rim)[0]
        if len(idx):
            inside = meshops.winding_inside(cent[idx], segs)
            keep_tri[idx[inside]] = True
        # Flaps can also sneak past CULL_DIST via bilinear bleed: any
        # boundary-band triangle already kept must ALSO be truly inside.
        idx2 = np.nonzero(boundary_band & keep_tri)[0]
        if len(idx2):
            inside2 = meshops.winding_inside(cent[idx2], segs)
            keep_tri[idx2[~inside2]] = False
    else:
        keep_tri |= boundary_band & (cd > 1e-6)
    new_f = tri.simplices[keep_tri]
    # Delaunay orientation is not guaranteed; make the new top face up.
    new_f = meshops.ensure_up_normals(pts2d, new_f)
    newz = zmax + _H(pts2d)
    new_v = np.column_stack([pts2d, newz])

    # --- displace kept verts; bottom pinned, sides barrel with height ---
    # Side belly (2D parts only): the stuffed edge bows OUTWARD in plan.
    # The weight is t^2 -- zero (pinned) at the board's base outline and
    # MAXIMAL at the welded top seam, so the widest point of the side
    # silhouette is exactly where the dome's roll lands, and the wall
    # tucks monotonically back under toward the base like wrapped vinyl.
    # It was sin(pi*t) (widest at mid-height, pinned at BOTH ends), which
    # made the dome roll down to a waist at the seam and then flare back
    # out below it -- a bell profile the viewport feedback flagged; a
    # monotone weight cannot produce a re-entrant waist. Direction =
    # outward = negative distance gradient; amount fades with depth so
    # only the wall band moves. Reference: side views in
    # reference/NOTES.md -- edges bulge near the TOP, never at mid-air.
    # Heights are evaluated at the ORIGINAL footprint positions for every
    # vertex (kept and new) before any horizontal displacement, so wall
    # and top read the same surface function at the same place --
    # evaluating the ring after displacing it read the surface ~2 mm off
    # from where its neighbours read it, which creased the seam.
    dz = _H(pv[:, :2])
    w = np.clip((pv[:, 2] - zmin) / thick, 0.0, 1.0) ** w_exp
    pv[:, 2] += dz * w
    new_v[: len(ring), 2] = pv[ring, 2]  # weld the seam exactly

    wall = None  # stashed belly quantities, reused for analytic wall normals
    if roll > 0.0:
        # Gradient of the SMOOTHED distance: the raw field carries the
        # footprint's random-sampling speckle, and a jittery gradient
        # scallops the wall (visible as periodic lumps along straight
        # edges). The smoothed gradient is direction-stable.
        grow, gcol = np.gradient(dist)

        def _outward(pts_xy):
            vx = -fp.sample(gcol, pts_xy)
            vy = -fp.sample(grow, pts_xy)
            vn = np.hypot(vx, vy)
            ok = vn > 1e-6
            vx = np.where(ok, vx / np.maximum(vn, 1e-9), 0.0)
            vy = np.where(ok, vy / np.maximum(vn, 1e-9), 0.0)
            return vx, vy

        ox, oy = _outward(pv[:, :2])
        t = np.clip((pv[:, 2] - zmin) / thick, 0.0, 1.0)
        belly_amp = 0.15 * roll
        belly = belly_amp * t * t
        # Smoothed field here too: raw-field speckle would modulate the
        # belly amount along the wall (periodic lumps on the silhouette).
        near_edge = np.exp(-fp.sample(dist, pv[:, :2]) / max(roll, 1e-6))
        pv[:, 0] += ox * belly * near_edge
        pv[:, 1] += oy * belly * near_edge
        # The SAME displacement field, evaluated at t = 1, moves the new
        # top vertices near the rim (near_edge decays inward over ~roll).
        # Displacing only the wall left the seam ring poking ~belly_amp
        # past the undisplaced surface just inside it -- a hard lip right
        # where the dome lands, which silhouettes as a waist ring. As one
        # continuous field, wall belly and rim lean-out taper smoothly
        # into the untouched dome.
        tx, ty = _outward(new_v[:, :2])
        near_top = np.exp(-fp.sample(dist, new_v[:, :2]) / max(roll, 1e-6))
        new_v[:, 0] += tx * belly_amp * near_top
        new_v[:, 1] += ty * belly_amp * near_top
        wall = (ox, oy, t, near_edge, belly_amp)

    # --- merge old (kept) and new (top) into one vertex/face set ---
    offset = len(pv)
    nf = new_f.copy()
    is_ring = new_f < len(ring)
    nf[is_ring] = ring[new_f[is_ring]]
    nf[~is_ring] = new_f[~is_ring] - len(ring) + offset
    all_v = np.vstack([pv, new_v[len(ring) :]])
    all_f = np.vstack([keep, nf])

    if not return_normals:
        all_v, all_f = meshops.compact(all_v, all_f)
        return all_v, all_f

    # Analytic vertex normals for the whole top surface. Averaged
    # topology normals wobble wherever triangle sizes change (the graded
    # ring->collar->grid bands), which glossy vinyl shows as streaks at
    # grazing angles. The normal comes from a NUMERICAL gradient of the
    # same analytic surface function _H the vertex heights came from
    # (central differences, sub-mm step), evaluated at the ORIGINAL
    # (pre-belly) positions the heights were evaluated at -- geometry and
    # shading always agree, and the rim carries none of the raster moire
    # a sampled-gradient normal would pick up.
    topo = meshops.topology_vertex_normals(all_v, all_f)
    top_ids = np.concatenate([ring, np.arange(len(pv), len(all_v))])
    otxy = np.vstack([ring_xy, pts2d[len(ring):]])
    eps = 0.35
    hx = (_H(otxy + np.array([eps, 0.0])) - _H(otxy - np.array([eps, 0.0]))) / (2 * eps)
    hy = (_H(otxy + np.array([0.0, eps])) - _H(otxy - np.array([0.0, eps]))) / (2 * eps)
    ana = np.column_stack([-hx, -hy, np.ones(len(otxy))])
    ana /= np.linalg.norm(ana, axis=1, keepdims=True)
    # RING normals need special handling: the central-difference probes
    # of a ring vertex step OUTSIDE the outline, where the UNSIGNED exact
    # distance folds back (|d|) and the numerical gradient collapses --
    # per-vertex noise that reads as fine teeth along the seam highlight.
    # Instead, take the 1D profile slope dH/dd from probes safely INSIDE
    # (0.5 and 1.0 mm along the inward direction) and orient it along the
    # smoothed distance gradient, which is direction-stable on the rim.
    grow_n, gcol_n = np.gradient(dist)
    ivx = fp.sample(gcol_n, ring_xy)
    ivy = fp.sample(grow_n, ring_xy)
    ivn = np.hypot(ivx, ivy)
    ivx = np.where(ivn > 1e-6, ivx / np.maximum(ivn, 1e-9), 0.0)
    ivy = np.where(ivn > 1e-6, ivy / np.maximum(ivn, 1e-9), 0.0)
    inward = np.column_stack([ivx, ivy])
    h1 = _H(ring_xy + 0.5 * inward)
    h2 = _H(ring_xy + 1.0 * inward)
    slope = (h2 - h1) / 0.5  # dH/dd at the rim (positive: rises inward)
    ring_ana = np.column_stack([slope * -ivx, slope * -ivy, np.ones(len(ring_xy))])
    ring_ana /= np.linalg.norm(ring_ana, axis=1, keepdims=True)
    ana[: len(ring)] = ring_ana
    normals = topo.copy()
    normals[top_ids] = ana

    if wall is not None:
        # Analytic wall normals too (2D parts): the quad strip's averaged
        # normals wobble with the alternating triangle diagonals, which a
        # glossy material shows as vertical ribbing on the side edges.
        # The bellied wall is a surface offset r(z) = amp*t^2*near along
        # the outward direction, so its exact normal tilts by -dr/dz
        # against the horizontal outward vector. At the rim the roll
        # leaves the wall tangentially and the top's analytic normal is
        # nearly horizontal there, so wall and top normals agree at the
        # seam with no blending hacks.
        ox, oy, t, near_edge, belly_amp = wall
        drdz = belly_amp * near_edge * 2.0 * t / max(thick, 1e-9)
        n_wall = np.column_stack([ox, oy, -drdz])
        wn = np.linalg.norm(n_wall, axis=1, keepdims=True)
        n_wall = np.divide(n_wall, wn, out=np.zeros_like(n_wall), where=wn > 1e-9)
        # Wallness: full on the boundary band, fading inward; zero where
        # the outward direction was degenerate. The wall normal runs all
        # the way DOWN to the base ring: the old z-ramp (fading to
        # averaged topology normals over the bottom ~2 cells) made the
        # lowest wall band shade off the quad-strip diagonals -- a
        # serrated sawtooth line along the base of every wall. The cost
        # is that the bottom cap's rim ring loses its pure (0,0,-1), but
        # the bottom cap faces the bench and is never rendered.
        wallness = near_edge * (np.hypot(ox, oy) > 0.5)
        blended = (
            topo[: len(pv)] * (1.0 - wallness[:, None]) + n_wall * wallness[:, None]
        )
        bn = np.linalg.norm(blended, axis=1, keepdims=True)
        blended = np.divide(blended, bn, out=topo[: len(pv)].copy(), where=bn > 1e-9)
        normals[: len(pv)] = blended
        # Top field wins on the ring/top (assigned above); re-assert.
        normals[top_ids] = ana
        # Ring: average of the (agreeing) wall and top analytic normals.
        ring_mix = 0.5 * ana[: len(ring)] + 0.5 * n_wall[ring]
        rn = np.linalg.norm(ring_mix, axis=1, keepdims=True)
        normals[ring] = np.divide(
            ring_mix, rn, out=ana[: len(ring)].copy(), where=rn > 1e-9
        )
    else:
        # STL parts: keep the seam soft with a plain 50/50 blend.
        ring_n = 0.5 * ana[: len(ring)] + 0.5 * topo[ring]
        ring_n /= np.maximum(np.linalg.norm(ring_n, axis=1, keepdims=True), 1e-12)
        normals[ring] = ring_n

    all_v, all_f, normals = meshops.compact_with_normals(all_v, all_f, normals)
    return all_v, all_f, normals


def _collar_points(
    ring_xy: np.ndarray, fp, dist_raw: np.ndarray, dist_smooth: np.ndarray,
    grid_step: float
) -> np.ndarray:
    """Concentric offset rings just inside the boundary.

    Moves each (dense) boundary vertex inward along the SMOOTHED distance
    field's gradient (the raw field's random-sampling speckle jitters the
    direction, and jittered collar points sitting on the steep roll slope
    turn into height scallops along the shoulder) by fractions of the
    grid step, producing 1-2 collar rings that fill the boundary->interior
    band with well-shaped triangles. This removes the sliver-triangle
    serration where the crown rolls over the edge, giving a smooth
    shoulder. Points that don't land safely inside (sharp concavities,
    validated against the RAW field) are dropped; Delaunay + centroid
    culling downstream tolerate the rest.
    """
    if len(ring_xy) < 3:
        return np.empty((0, 2))
    grow, gcol = np.gradient(dist_smooth)  # d(dist)/drow, d(dist)/dcol
    vx = fp.sample(gcol, ring_xy)
    vy = fp.sample(grow, ring_xy)
    vec = np.column_stack([vx, vy])
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 1e-6)

    # Both collar rows keep the ring's full ~2 mm density. They used to be
    # thinned toward the grid spacing via quantized-coordinate dedup, but
    # that retention pattern is irregular along the edge, so the chord sag
    # of the triangles spanning the tightly-curved roll varied with an
    # irregular ~half-grid period -- visible as soft waves on the shoulder
    # silhouette (worst at preview resolution). Dense rows make the sag
    # uniform and tiny. The "normal pole beads" the thinning originally
    # prevented were an averaged-normal artifact; the top field now ships
    # analytic normals, so dense fans shade cleanly.
    out = []
    for frac in (0.4, 0.8):
        off = grid_step * frac
        cand = ring_xy + vec * off
        d = fp.sample(dist_raw, cand)
        cand = cand[d > off * 0.5]
        if len(cand):
            out.append(cand)
    if not out:
        return np.empty((0, 2))
    return np.vstack(out)


def _mask_on_raster(mask, fp) -> np.ndarray | None:
    """Resample a painted mask onto the panel's current raster.

    World-anchored masks (dicts with their own xmin/ymin/res) are sampled
    by world position, so a mask painted on the 4 mm preview raster lands
    exactly on the 2 mm export raster. Grid convention on both sides:
    cell (row, col) is centred at (xmin + (col-1)*res, ymin + (row-1)*res)
    -- the same 1-cell border used by FootprintRaster.
    """
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        # Bare array: assume it spans this panel's raster extent.
        scale = mask.shape[0] / fp.grid.shape[0] if mask.shape[0] else 1.0
        mask = {"grid": mask, "xmin": fp.xmin, "ymin": fp.ymin,
                "res": fp.res / max(scale, 1e-9)}
    grid = np.asarray(mask["grid"], dtype=np.float64)
    if not grid.size:
        return None
    ny, nx = fp.grid.shape
    rows = np.arange(ny, dtype=np.float64)
    cols = np.arange(nx, dtype=np.float64)
    # Raster cells use the MARGIN border; mask grids keep their own
    # 1-cell-border convention (that's how the brush paints them).
    world_y = fp.ymin + (rows - meshops.MARGIN) * fp.res
    world_x = fp.xmin + (cols - meshops.MARGIN) * fp.res
    mrow = (world_y - mask["ymin"]) / mask["res"] + 1
    mcol = (world_x - mask["xmin"]) / mask["res"] + 1
    mm, cc = np.meshgrid(mrow, mcol, indexing="ij")
    return ndimage.map_coordinates(grid, [mm, cc], order=1, mode="nearest")
