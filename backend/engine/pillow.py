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

# Extra smoothing (in grid cells) applied to the profile after a painted
# mask is multiplied in, so mask edges never crease the surface.
MASK_BLUR_SIGMA = 1.5


def pillow_panel(
    pv: np.ndarray,
    pf: np.ndarray,
    *,
    crown: float = 32.0,
    dref: float = 110.0,
    exp: float = 0.55,
    sigma: float = 5.0,
    w_exp: float = 1.5,
    res: float = 2.0,
    grid_step: float = 6.0,
    mask: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the pillow displacement to one panel.

    Parameters
    ----------
    pv, pf : the panel mesh (N x 3 float vertices, M x 3 int faces).
        Not modified; copies are made.
    crown, dref, exp, sigma, w_exp : see models.PillowParams.
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
        return pv, pf.copy()

    vtop = meshops.top_vertex_mask(pv, zmax)
    if not (vtop[pf].all(axis=1)).any():
        # No flat-top faces at all: nothing we can pillow safely.
        return pv, pf.copy()

    # --- footprint raster + interior distance + crown profile ---
    # dist_raw: geometry truth (0 in holes/outside) for culling + inset.
    # dist: smoothed, drives the crown profile shape.
    fp = meshops.rasterize_footprint(pv, pf, res, rng)
    dist_raw, dist = meshops.distance_field(fp, sigma)
    prof = meshops.crown_profile(dist, crown, dref, exp)

    mask_grid = _mask_on_raster(mask, fp)
    if mask_grid is not None:
        prof = prof * mask_grid
        # Upholstery never creases: soften whatever the brush did.
        prof = ndimage.gaussian_filter(prof, sigma=MASK_BLUR_SIGMA)

    # --- delete flat top, keep sides/fillets/bottom ---
    keep = pf[~vtop[pf].all(axis=1)]
    # Boundary ring: top-plane vertices still referenced by kept faces.
    ring = np.unique(keep[vtop[keep].any(axis=1)])
    ring = ring[vtop[ring]]
    ring_xy = pv[ring, :2]

    # --- new domed top: ring + interior grid, Delaunay, cull outside ---
    inset = grid_step * GRID_INSET_FACTOR
    gx, gy = np.meshgrid(
        np.arange(fp.xmin, pv[:, 0].max(), grid_step),
        np.arange(fp.ymin, pv[:, 1].max(), grid_step),
    )
    gp = np.column_stack([gx.ravel(), gy.ravel()])
    gp = gp[fp.sample(dist_raw, gp) > inset]

    pts2d = np.vstack([ring_xy, gp])
    if len(pts2d) < 3:
        return pv, pf.copy()
    tri = Delaunay(pts2d)
    cent = pts2d[tri.simplices].mean(axis=1)
    new_f = tri.simplices[fp.sample(dist_raw, cent) > CULL_DIST]
    # Delaunay orientation is not guaranteed; make the new top face up.
    new_f = meshops.ensure_up_normals(pts2d, new_f)
    newz = zmax + fp.sample(prof, pts2d)
    new_v = np.column_stack([pts2d, newz])

    # --- displace kept verts; bottom pinned, sides barrel with height ---
    dz = fp.sample(prof, pv[:, :2])
    w = np.clip((pv[:, 2] - zmin) / thick, 0.0, 1.0) ** w_exp
    pv[:, 2] += dz * w
    new_v[: len(ring), 2] = pv[ring, 2]  # weld the seam exactly

    # --- merge old (kept) and new (top) into one vertex/face set ---
    offset = len(pv)
    nf = new_f.copy()
    is_ring = new_f < len(ring)
    nf[is_ring] = ring[new_f[is_ring]]
    nf[~is_ring] = new_f[~is_ring] - len(ring) + offset
    all_v = np.vstack([pv, new_v[len(ring) :]])
    all_f = np.vstack([keep, nf])
    return meshops.compact(all_v, all_f)


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
    world_y = fp.ymin + (rows - 1) * fp.res
    world_x = fp.xmin + (cols - 1) * fp.res
    mrow = (world_y - mask["ymin"]) / mask["res"] + 1
    mcol = (world_x - mask["xmin"]) / mask["res"] + 1
    mm, cc = np.meshgrid(mrow, mcol, indexing="ij")
    return ndimage.map_coordinates(grid, [mm, cc], order=1, mode="nearest")
