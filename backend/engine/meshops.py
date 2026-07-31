"""Low-level mesh/grid operations shared by the pillow engine.

Everything here works on plain numpy arrays (``pv`` = N x 3 vertices,
``pf`` = M x 3 int faces) so the hot path never round-trips through
trimesh objects. Trimesh is only used at the import/export boundary.

Geometry conventions:
  * All coordinates are millimetres.
  * Panels lie flat: the mounting face is the minimum-z plane, the
    upholstered face is the maximum-z plane.
  * Rasters are (ny, nx) boolean/float grids with a 1-cell border margin;
    cell (row, col) covers world x = xmin + (col-1)*res .. etc. The +1
    offsets in stamp/sample keep that border consistent everywhere.
"""
from __future__ import annotations

import io

import numpy as np
import trimesh
from scipy import ndimage

# Vertices within this distance of zmax count as "on the flat top".
TOP_TOL = 1e-3

# Empty border cells around the footprint raster, each side. Must exceed
# the binary_closing iteration count (3): scipy's closing pads with zeros,
# so erosion EATS any occupied cells within `iterations` of the array
# edge. The original reference used a 1-cell margin, which silently
# shaved ~2 cells (4 mm at export res) off the min-x/min-y edges of every
# panel and zeroed the distance gradient there.
MARGIN = 4


class FootprintRaster:
    """A boolean occupancy grid of a panel footprint plus its transform.

    Bundles the grid with (xmin, ymin, res) so distance fields and profile
    grids derived from it can be sampled at arbitrary world xy points
    without re-passing the transform around.
    """

    def __init__(self, grid: np.ndarray, xmin: float, ymin: float, res: float):
        self.grid = grid
        self.xmin = xmin
        self.ymin = ymin
        self.res = res

    def sample(self, field: np.ndarray, pts_xy: np.ndarray, order: int = 1) -> np.ndarray:
        """Sample a grid-shaped field at world xy points.

        order=1 (bilinear) for fields with creases or hard zeros (the
        raw distance, occupancy-derived fields) -- cubic would overshoot
        them. order=3 (cubic, C2) for SMOOTH fields feeding the surface
        function: bilinear is only C0, and its cell-boundary curvature
        kinks set the floor on how line-free the dome can be.
        """
        return ndimage.map_coordinates(
            field,
            [
                (pts_xy[:, 1] - self.ymin) / self.res + MARGIN,
                (pts_xy[:, 0] - self.xmin) / self.res + MARGIN,
            ],
            order=order,
        )


def top_vertex_mask(pv: np.ndarray, zmax: float | None = None) -> np.ndarray:
    """Boolean mask of vertices on the flat top plane (z within TOP_TOL of zmax)."""
    if zmax is None:
        zmax = float(pv[:, 2].max())
    return np.abs(pv[:, 2] - zmax) < TOP_TOL


# Enclosed empty regions up to this area are treated as sampling speckle
# and filled; anything larger is a genuine cutout (handle hole, notch) and
# must stay open so the distance field dips the fabric toward its rim.
HOLE_FILL_MAX_MM2 = 200.0


def fill_small_holes(grid: np.ndarray, res: float, max_mm2: float = HOLE_FILL_MAX_MM2) -> np.ndarray:
    """Fill enclosed empty regions smaller than ``max_mm2``.

    Deliberate deviation from the reference algorithm's blanket
    ``binary_fill_holes``: that call existed to repair speckle left by the
    random top-face sampling, but a blanket fill would also erase real
    cutouts (e.g. a 100 mm handle hole), and the physics requires those to
    pull the surface down (spec section 2). Size-thresholding keeps both
    behaviours: speckle (a few cells) is filled, real holes survive.
    """
    filled = ndimage.binary_fill_holes(grid)
    holes = filled & ~grid
    if not holes.any():
        return grid
    labels, n = ndimage.label(holes)
    sizes = np.bincount(labels.ravel())  # index 0 is background
    max_cells = max_mm2 / (res * res)
    small = sizes <= max_cells
    small[0] = False
    return grid | (holes & small[labels])


def rasterize_footprint(
    pv: np.ndarray,
    pf: np.ndarray,
    res: float,
    rng: np.random.Generator,
) -> FootprintRaster:
    """Rasterize a panel's xy footprint at ``res`` mm/cell.

    Production lesson baked in: CAD flat tops are made of giant triangles
    with NO interior vertices, so stamping vertices alone leaves the grid
    interior empty and hole filling does nothing. We therefore scatter
    area-weighted random samples ON the top-face triangles (about one per
    2 mm^2), stamp those plus every vertex, then binary-close small gaps
    and fill remaining speckle (small enclosed voids only -- see
    fill_small_holes for why real cutouts are preserved).
    """
    zmax = float(pv[:, 2].max())
    xmin = float(pv[:, 0].min())
    ymin = float(pv[:, 1].min())
    span_x = float(pv[:, 0].max()) - xmin
    span_y = float(pv[:, 1].max()) - ymin
    # Centre the CELL CENTRES on the footprint: fields (exact distance,
    # Poisson membrane, tension) are evaluated at cell centres, so a
    # mirror-symmetric panel only produces mirror-symmetric fields if the
    # centre lattice is itself mirror-symmetric about the panel's middle.
    # Anchoring the first centre at the outline's min corner (the old
    # behaviour) skews every centre by up to res/2 relative to its mirror
    # partner whenever span/res is fractional -- measured as ~res/2 of
    # crown asymmetry. The first centre goes at half the leftover span so
    # first and last centres sit symmetrically inside the outline; points
    # are stamped by ROUNDING to the nearest centre (cell = centre +-
    # res/2), which keeps stamping consistent with that convention.
    nxc = max(int(np.floor(span_x / res + 0.5)) + 1, 1)
    nyc = max(int(np.floor(span_y / res + 0.5)) + 1, 1)
    xmin += (span_x - (nxc - 1) * res) / 2.0
    ymin += (span_y - (nyc - 1) * res) / 2.0
    nx = nxc + 2 * MARGIN
    ny = nyc + 2 * MARGIN

    vtop = top_vertex_mask(pv, zmax)
    ftop = pf[vtop[pf].all(axis=1)]  # flat-top triangles

    if len(ftop):
        tri0 = pv[ftop][:, :, :2].astype(np.float64)
        e1 = tri0[:, 1] - tri0[:, 0]
        e2 = tri0[:, 2] - tri0[:, 0]
        area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) * 0.5
        # ~1 sample per 2 mm^2, at least 1 per triangle
        nsamp = np.maximum((area / 2.0).astype(int), 1)
        reps = np.repeat(np.arange(len(tri0)), nsamp)
        r1 = np.sqrt(rng.random(len(reps)))[:, None]
        r2 = rng.random(len(reps))[:, None]
        sp = (1 - r1) * tri0[reps, 0] + r1 * (1 - r2) * tri0[reps, 1] + r1 * r2 * tri0[reps, 2]
        allp = np.vstack([pv[:, :2], sp])
    else:
        allp = pv[:, :2]

    grid = np.zeros((ny, nx), dtype=bool)
    grid[
        np.clip(np.floor((allp[:, 1] - ymin) / res + 0.5).astype(int) + MARGIN, 0, ny - 1),
        np.clip(np.floor((allp[:, 0] - xmin) / res + 0.5).astype(int) + MARGIN, 0, nx - 1),
    ] = True
    grid = ndimage.binary_closing(grid, np.ones((3, 3)), iterations=3)
    grid = fill_small_holes(grid, res)
    # Shave single-cell boundary raggedness left by the random sampling:
    # the EDT amplifies a ragged edge into periodic ridges that ripple
    # through every derived field (visible as beads along the edge roll).
    # One opening pass removes 1-cell protrusions; the closing above has
    # already guaranteed there are no 1-cell bays to widen.
    grid = ndimage.binary_opening(grid, np.ones((3, 3)), iterations=1)
    grid = fill_small_holes(grid, res)
    return FootprintRaster(grid, xmin, ymin, res)


def top_boundary_segments(pv: np.ndarray, pf: np.ndarray, zmax: float | None = None) -> np.ndarray:
    """XY segments (n, 2, 2) bounding the flat-top patch (outline + holes).

    Edges referenced by exactly one flat-top face are the border of the
    top region -- the panel's true outline (and hole rims) with no raster
    involved. Segments keep the FACE WINDING's direction (up-facing CCW
    faces walk the outline CCW and holes CW), so winding-number
    inside/outside tests work; plain distance queries don't care.
    """
    vtop = top_vertex_mask(pv, zmax)
    ftop = pf[vtop[pf].all(axis=1)]
    if not len(ftop):
        return np.empty((0, 2, 2))
    edges = np.concatenate([ftop[:, [0, 1]], ftop[:, [1, 2]], ftop[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    uniq, first, counts = np.unique(
        key, axis=0, return_index=True, return_counts=True
    )
    border = edges[first[counts == 1]]  # original direction, not sorted
    return pv[border][:, :, :2].astype(np.float64)


def nearest_segment_inward(pts_xy: np.ndarray, segs: np.ndarray, res: float) -> np.ndarray:
    """Unit inward normal of the ORIENTED boundary segment nearest each point.

    The material is left of travel (CCW outline, CW holes), so the left
    normal of the nearest segment always points into the panel -- even
    for points exactly ON the outline or at a convex corner, where
    gradient-of-distance directions degenerate. Used as the fallback
    direction for rim/wall normals at corners.
    """
    from scipy.spatial import cKDTree

    a, b = segs[:, 0], segs[:, 1]
    seg_len = np.linalg.norm(b - a, axis=1)
    nsub = np.maximum(np.ceil(seg_len / (2.0 * res)).astype(int), 1)
    pieces_a, pieces_b = [], []
    for n in np.unique(nsub):
        sel = nsub == n
        aa, bb = a[sel], b[sel]
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            pieces_a.append(aa + (bb - aa) * t0)
            pieces_b.append(aa + (bb - aa) * t1)
    pa = np.vstack(pieces_a)
    pb = np.vstack(pieces_b)
    mid = (pa + pb) * 0.5
    _, idx = cKDTree(mid).query(pts_xy, k=1)
    t = pb[idx] - pa[idx]
    tl = np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    t = t / tl
    return np.column_stack([-t[:, 1], t[:, 0]])  # left normal = inward


def winding_inside(pts_xy: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Exact inside test for points vs oriented boundary segments.

    Sums the signed angles each directed segment subtends at each point:
    ~+-2*pi inside a CCW outline (0 inside its CW holes), ~0 outside.
    Used for triangles hugging the rim, where the raster's bilinear
    distance can't resolve which side of the outline a centroid is on.
    Vectorized over chunks; cost is points x segments, so callers pass
    only near-rim candidates.
    """
    if not len(pts_xy):
        return np.zeros(0, dtype=bool)
    a = segs[:, 0]
    b = segs[:, 1]
    out = np.empty(len(pts_xy), dtype=bool)
    chunk = max(1, int(2_000_000 / max(len(segs), 1)))
    for s in range(0, len(pts_xy), chunk):
        p = pts_xy[s : s + chunk][:, None, :]
        ra = a[None, :, :] - p
        rb = b[None, :, :] - p
        ang = np.arctan2(
            ra[:, :, 0] * rb[:, :, 1] - ra[:, :, 1] * rb[:, :, 0],
            ra[:, :, 0] * rb[:, :, 0] + ra[:, :, 1] * rb[:, :, 1],
        )
        out[s : s + chunk] = np.abs(ang.sum(axis=1)) > np.pi
    return out


def _dist_to_segments(pts: np.ndarray, segs: np.ndarray, res: float) -> np.ndarray:
    """Exact distance from each xy point to the nearest of many segments.

    Long segments are subdivided to <= 2*res pieces, candidates are found
    with a KDTree on piece midpoints (k nearest), and the exact point-
    segment distance is evaluated only for those candidates. With pieces
    this short the k-candidate set always contains the true nearest
    segment, so the result is exact to float precision.
    """
    from scipy.spatial import cKDTree

    a, b = segs[:, 0], segs[:, 1]
    seg_len = np.linalg.norm(b - a, axis=1)
    pieces_a, pieces_b = [], []
    nsub = np.maximum(np.ceil(seg_len / (2.0 * res)).astype(int), 1)
    for n in np.unique(nsub):
        sel = nsub == n
        aa, bb = a[sel], b[sel]
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            pieces_a.append(aa + (bb - aa) * t0)
            pieces_b.append(aa + (bb - aa) * t1)
    pa = np.vstack(pieces_a)
    pb = np.vstack(pieces_b)
    mid = (pa + pb) * 0.5

    k = min(8, len(mid))
    _, idx = cKDTree(mid).query(pts, k=k)
    if k == 1:
        idx = idx[:, None]
    A = pa[idx]                      # (npts, k, 2)
    B = pb[idx]
    P = pts[:, None, :]
    AB = B - A
    denom = np.maximum((AB * AB).sum(axis=2), 1e-18)
    t = np.clip(((P - A) * AB).sum(axis=2) / denom, 0.0, 1.0)
    proj = A + AB * t[:, :, None]
    d = np.linalg.norm(P - proj, axis=2)
    return d.min(axis=1)


def distance_field(
    fp: FootprintRaster, sigma: float, segments: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Interior distance-to-boundary in mm: (raw, smoothed).

    With ``segments`` (the panel's true top-boundary segments from
    ``top_boundary_segments``) the raw field is computed EXACTLY at every
    occupied cell centre: sub-cell accurate, no raster stair-steps on
    diagonal edges, and -- critically -- symmetric for symmetric outlines.
    The cell-quantized EDT it replaces put one edge of a panel on a cell
    boundary and the opposite edge mid-cell whenever the span didn't
    divide evenly by the resolution, which skewed the whole crown by up
    to half a cell per side. Without segments (degenerate/non-manifold
    tops) it falls back to the EDT of the occupancy grid.

    The smoothed copy (gaussian, sigma in grid cells; callers rescale for
    resolution so preview and export smooth the same physical distance)
    drives the crown profile so it has no faceting from the raster.

    The RAW field must be used for geometric decisions (triangle culling,
    interior-grid inset): it is exactly 0 outside the footprint and inside
    cutouts. The smoothed field bleeds positive values ~sigma cells past
    the boundary, which at coarse preview resolution is enough to wrongly
    keep triangles spanning a real hole.
    """
    if segments is not None and len(segments):
        rows, cols = np.nonzero(fp.grid)
        cx = fp.xmin + (cols - MARGIN) * fp.res
        cy = fp.ymin + (rows - MARGIN) * fp.res
        d = _dist_to_segments(np.column_stack([cx, cy]), segments, fp.res)
        raw = np.zeros(fp.grid.shape, dtype=np.float64)
        raw[rows, cols] = d
    else:
        raw = ndimage.distance_transform_edt(fp.grid) * fp.res
    smooth = ndimage.gaussian_filter(raw, sigma=sigma) if sigma > 0 else raw
    return raw, smooth


# Half-width of the smooth saturation knee, in units of the normalized
# profile value p = (dist/dref)^exp. The hard clip() the reference used
# has a slope discontinuity exactly where the crown flattens -- it reads
# as a hard ridge around the top of the bulge. The C1 smooth clamp below
# blends slope to zero across p in [1-k, 1+k] instead. Values outside
# the knee are untouched, so the calibrated rise off the edge and the
# saturated center height are both preserved exactly.
KNEE_HALF_WIDTH = 0.35


def _smooth_clamp(p: np.ndarray, k: float = KNEE_HALF_WIDTH) -> np.ndarray:
    """C2 clamp of p to <= 1: identity below 1-k, exactly 1 above 1+k,
    and a quintic blend in between.

    C2 matters: the previous quadratic blend was only C1 -- CURVATURE
    jumped at both ends of the band, and glossy vinyl at grazing angles
    renders a curvature jump as a faint contour line ringing the dome
    where the crown saturates ("I can still see lines at just the right
    angle"). The quintic g(u) = u - u^3 + u^4/2 on the normalized band
    matches value, slope AND curvature at both ends (g'(u) =
    (1-u)^2(1+2u) >= 0, so it is also monotone); deep-interior points
    still reach exactly 1 and the rise below the band is untouched.
    """
    lo = 1.0 - k
    out = np.minimum(p, 1.0)
    band = (p > lo) & (p < 1.0 + k)
    u = (p[band] - lo) / (2.0 * k)
    g = u - u**3 + 0.5 * u**4
    out[band] = lo + 2.0 * k * g
    return out


def crown_profile(dist: np.ndarray, crown: float, dref: float, exp: float) -> np.ndarray:
    """The validated pillow shape with a softened saturation shoulder:
    dz = crown * smooth_clamp((dist/dref) ** exp).

    Edges are pinned (dist=0 -> dz=0); points deeper than dref from any
    edge saturate at the full crown. exp < 1 makes the surface rise fast
    off the edge then flatten into a gentle dome, like stretched vinyl.
    The smooth clamp rounds the transition into the flat center so the
    crown has no hard ridge where it stops rising (upholstery never
    creases); deep-interior points still reach exactly ``crown``.
    """
    p = np.power(np.maximum(dist, 0.0) / dref, exp)
    return crown * _smooth_clamp(p)


def edge_roll_drop(dist_raw: np.ndarray, roll: float, res: float = 0.0) -> np.ndarray:
    """Height drop (<= 0) of the wrapped-edge roll, vs distance to edge.

    Models the vinyl rolling over the arris of the core as a quarter
    circle of radius ``roll``: at the outline the surface sits ``roll``
    below the flat top and leaves the (vertical) side wall tangentially;
    by ``dist >= roll`` it has flattened out and the drop is 0. Added to
    the crown profile this makes edge roll + crown one continuous C1
    height field -- no discrete fillet geometry, no tangent break at the
    rim, and corners fold down naturally (dist is small on both sides of
    a corner, like the photographed folds).

    Uses the RAW distance so the roll radius is faithful; the caller may
    blur the result a touch to hide raster stair-steps. ``res`` shifts
    the curve one cell outward: boundary-adjacent cells read ~res from
    the EDT even though the outline itself is at distance 0, and without
    the shift the rim would only drop a fraction of the roll.
    """
    d = np.clip(dist_raw - res, 0.0, roll)
    return np.sqrt(np.maximum(roll * roll - (roll - d) ** 2, 0.0)) - roll


def _membrane_solve(occ: np.ndarray, res: float) -> np.ndarray | None:
    """Solve grad^2 h = -1 (h = 0 outside) over an occupancy grid.

    Returns h (mm^2) as a full grid (0 outside), or None if the domain is
    degenerate. 5-point Laplacian, direct sparse solve.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    ny, nx = occ.shape
    cells = np.argwhere(occ)
    if len(cells) < 9:
        return None
    idx = -np.ones((ny, nx), dtype=np.int64)
    idx[occ] = np.arange(len(cells))
    n = len(cells)
    ci, cj = cells[:, 0], cells[:, 1]
    k = np.arange(n)
    rows = [k]
    colz = [k]
    data = [np.full(n, 4.0)]
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = ci + di, cj + dj
        valid = (ni >= 0) & (ni < ny) & (nj >= 0) & (nj < nx)
        nn = -np.ones(n, dtype=np.int64)
        nn[valid] = idx[ni[valid], nj[valid]]
        has = nn >= 0
        rows.append(k[has])
        colz.append(nn[has])
        data.append(np.full(int(has.sum()), -1.0))
    a = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(colz))),
        shape=(n, n),
    ).tocsr()
    try:
        h = spsolve(a, np.full(n, res * res))
    except Exception:
        return None
    hg = np.zeros((ny, nx))
    hg[occ] = np.maximum(h, 0.0)
    return hg


def poisson_wall_distance(fp: "FootprintRaster", target_cells: int = 240) -> np.ndarray | None:
    """Crease-free distance-like field: Spalding/Poisson wall distance.

    The nearest-edge distance function ALWAYS has slope creases -- along
    the medial axis (the "spine" down the middle of a rectangle) and
    along corner bisectors -- and any crown driven by it prints those
    creases onto the upholstered face as visible lines, no matter how
    much the field is blurred (a blurred crease is a soft line, not no
    line). An inflated membrane has no such folds: the solution h of
    grad^2 h = -1, h = 0 on every edge, is smooth everywhere inside.

    Spalding's normalization turns h into a distance-LIKE field,

        d = sqrt(|grad h|^2 + 2 h) - |grad h|,

    which is EXACT on an infinite strip (everywhere, including the
    centerline) and first-order exact near every straight wall -- so the
    calibrated near-edge rise is preserved -- while rounding the medial
    axis and corner pockets with the physics of real stretched fabric.

    h is solved on a coarsened grid (<= ``target_cells`` per side; a
    direct full-res solve costs seconds) and upsampled with a cubic zoom
    -- h is C-infinity smooth, so the coarse solve loses nothing visible;
    its only inaccuracy is within a couple of coarse cells of the walls,
    which the caller's raw/interior blend never uses. The gradient for
    Spalding's formula is taken at full resolution AFTER upsampling.
    """
    grid = fp.grid
    ny, nx = grid.shape
    factor = max(1, int(np.ceil(max(ny, nx) / target_cells)))
    if factor > 1 and factor % 2 == 0:
        # An EVEN factor makes a symmetric pad impossible for odd grid
        # dims ((-ny) % 2 is then fixed at 1), so round up to odd.
        factor += 1
    if factor > 1:
        # Pad so the pooling blocks are CENTRED on the footprint: blocks
        # anchored at array index 0 chop a partial block off one side
        # only, which skews the coarse h and shows up as a millimetre-
        # scale asymmetry of the crown. The pad total is forced EVEN (an
        # odd pad can only go on one side, which is the same skew again);
        # with an odd factor, adding one more block flips parity, so an
        # even pad always exists.
        pad_y = (-ny) % factor
        if pad_y % 2:
            pad_y += factor
        pad_x = (-nx) % factor
        if pad_x % 2:
            pad_x += factor
        gp = np.pad(
            grid,
            ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
        )
        occ = (
            gp.reshape(gp.shape[0] // factor, factor, gp.shape[1] // factor, factor)
            .mean(axis=(1, 3))
            >= 0.5
        )
        h = _membrane_solve(occ, fp.res * factor)
        if h is None:
            return None
        h = ndimage.zoom(
            h, (gp.shape[0] / h.shape[0], gp.shape[1] / h.shape[1]),
            order=3, mode="nearest",
        )
        h = h[pad_y // 2 : pad_y // 2 + ny, pad_x // 2 : pad_x // 2 + nx]
        if h.shape != grid.shape:
            h = np.pad(
                h, ((0, ny - h.shape[0]), (0, nx - h.shape[1])), mode="edge"
            )
        h = np.where(grid, np.maximum(h, 0.0), 0.0)
        # Red-black SOR sweeps at FULL resolution, seeded with the coarse
        # solution: the coarse solve is only wrong within a couple of
        # coarse cells of the walls (high-frequency error), which
        # relaxation kills fast -- and the sweeps re-symmetrize whatever
        # block quantization the pooling left. Cost: a few np.rolls per
        # sweep, negligible next to the sparse solve.
        rhs = fp.res * fp.res
        checker = (np.add.outer(np.arange(ny), np.arange(nx)) % 2).astype(bool)
        omega = 1.7
        # Enough sweeps to also relax the coarse solve's boundary
        # STAIRCASE error on curved outlines (wavelength ~2 coarse cells
        # along the rim -- slower to die than pointwise error; visible as
        # wrinkles in the bulb shoulder of curved parts). Sweeps are a
        # few np.rolls each; even 60x3 costs ~tens of ms.
        for _ in range(60 * factor):
            for color in (checker, ~checker):
                nb = (
                    np.roll(h, 1, 0) + np.roll(h, -1, 0)
                    + np.roll(h, 1, 1) + np.roll(h, -1, 1)
                )
                gs = 0.25 * (nb + rhs)
                upd = color & grid
                h[upd] += omega * (gs[upd] - h[upd])
            h[~grid] = 0.0
        h = np.maximum(h, 0.0)
    else:
        h = _membrane_solve(grid, fp.res)
        if h is None:
            return None
    gy, gx = np.gradient(h, fp.res)
    g2 = gx * gx + gy * gy
    # Regularized Spalding: d = sqrt(g^2 + 2h + eps^2) - sqrt(g^2 + eps^2).
    # The textbook form subtracts |grad h|, whose CONICAL kink where the
    # gradient vanishes -- i.e. exactly along the dome's ridge line and
    # at its apex -- puts a curvature discontinuity down the middle of
    # the crown (a faint line at grazing angles). Adding eps^2 inside
    # both square roots keeps the formula smooth through the ridge; at
    # the walls (h = 0) the two terms cancel identically, so the field
    # stays exactly 0 there and first-order exact nearby. eps scales with
    # the membrane's own height so the rounding is proportionate on any
    # panel size.
    eps2 = (0.05 * np.sqrt(2.0 * h.max() + 1e-12)) ** 2
    d = np.sqrt(g2 + 2.0 * h + eps2) - np.sqrt(g2 + eps2)
    return np.where(grid, np.maximum(d, 0.0), 0.0)


def membrane_tension(fp: "FootprintRaster", target_cells: int = 160) -> np.ndarray:
    """Tension-suppression factor in [0, 1] from an inflated-membrane solve.

    The nearest-edge distance transform only knows the *closest* wall, so
    it can't tell a point squeezed between two converging edges (a
    tapering tail) from a point the same distance inside a broad panel.
    Real stretched vinyl is a membrane pinned at every edge: where the
    fabric is constrained from several sides it can't loft as high --
    that's the extra tension a narrowing region feels.

    We solve the small-deflection inflated-membrane equation grad^2 h = -1
    with h = 0 on every edge (and inside holes), on a coarsened copy of
    the footprint (the field is smooth and low-frequency, so a <=160-cell
    grid is plenty and keeps cost O(1) regardless of panel size/res).
    The membrane's effective half-width is d_mem = sqrt(2 h); for an
    infinite strip d_mem equals the nearest-edge distance exactly, so
    straight rectangles are unchanged. Where the shape tapers or pinches,
    d_mem < distance, and

        tension = clip(d_mem / distance, 0, 1)

    is < 1 -- the amount to pull the crown down. It is clipped at 1 so the
    factor can only *suppress* (never inflate) relative to the validated
    distance-field crown, which keeps every calibrated straight-panel case
    identical and only adds realism to irregular outlines.
    """
    from scipy import ndimage
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    grid = fp.grid
    ny, nx = grid.shape

    # Coarsen so the linear solve is bounded regardless of raster size.
    factor = max(1, int(np.ceil(max(ny, nx) / target_cells)))
    if factor > 1:
        cy, cx = (ny // factor) * factor, (nx // factor) * factor
        occ = (
            grid[:cy, :cx]
            .reshape(cy // factor, factor, cx // factor, factor)
            .mean(axis=(1, 3))
            >= 0.5
        )
        res_c = fp.res * factor
    else:
        occ = grid.copy()
        res_c = fp.res

    H, W = occ.shape
    cells = np.argwhere(occ)
    if len(cells) < 9:
        return np.ones_like(grid, dtype=np.float64)

    idx = -np.ones((H, W), dtype=np.int64)
    idx[occ] = np.arange(len(cells))
    n = len(cells)
    ci, cj = cells[:, 0], cells[:, 1]
    k = np.arange(n)

    rows = [k]
    colz = [k]
    data = [np.full(n, 4.0)]
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = ci + di, cj + dj
        valid = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
        nn = -np.ones(n, dtype=np.int64)
        nn[valid] = idx[ni[valid], nj[valid]]
        has = nn >= 0
        rows.append(k[has])
        colz.append(nn[has])
        data.append(np.full(int(has.sum()), -1.0))
    a = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(colz))),
        shape=(n, n),
    ).tocsr()
    h = spsolve(a, np.full(n, res_c * res_c))  # h in mm^2
    h = np.maximum(h, 0.0)

    d_mem = np.zeros((H, W))
    d_mem[occ] = np.sqrt(2.0 * h)  # mm
    d_dist = ndimage.distance_transform_edt(occ) * res_c
    # C2 soft clip instead of a hard clip at 1: the clip contour (where
    # membrane width first equals nearest-edge distance) is a closed
    # curve across the dome, and a slope discontinuity there reads as a
    # faint line at grazing angles. Small knee: suppression strength in
    # genuinely narrow regions is unchanged.
    ratio = np.where(occ, d_mem / np.maximum(d_dist, res_c), 1.0)
    tau = np.where(occ, np.maximum(_smooth_clamp(ratio, 0.15), 0.0), 1.0)
    # Smooth ON the coarse grid before upsampling: bilinear zoom of an
    # unsmoothed field leaves piecewise-linear creases every `factor`
    # cells, which surface as broad soft ripples on the crown plateau.
    tau = ndimage.gaussian_filter(tau, sigma=1.2)

    if factor > 1:
        # Cubic zoom: bilinear (order=1) leaves piecewise-linear gradient
        # kinks every `factor` cells, and because the crown profile is a
        # function of tau * dist those kinks surface as a wavy specular
        # band along the shoulder of an otherwise-clean panel. Cubic is
        # C2 across the coarse cells; a follow-up blur of ~the coarse cell
        # size removes any residual block structure.
        tau = ndimage.zoom(tau, (ny / H, nx / W), order=3, mode="nearest")
        tau = tau[:ny, :nx]
        if tau.shape != grid.shape:
            tau = np.pad(
                tau,
                ((0, ny - tau.shape[0]), (0, nx - tau.shape[1])),
                mode="edge",
            )
        tau = ndimage.gaussian_filter(tau, sigma=factor * 0.8)
        tau = np.clip(tau, 0.0, 1.0)
    return np.where(grid, tau, 1.0)


def ensure_up_normals(pts2d: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip 2D triangles to counter-clockwise so extruded normals point +z."""
    a = pts2d[faces[:, 0]]
    b = pts2d[faces[:, 1]]
    c = pts2d[faces[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flipped = faces.copy()
    cw = cross < 0
    flipped[cw] = flipped[cw][:, ::-1]
    return flipped


def topology_vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted average vertex normals (plain, fast, no trimesh)."""
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    fn = np.cross(e1, e2)  # length ~ 2*area: built-in area weighting
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], fn)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norm, out=np.zeros_like(out), where=norm > 1e-12)


def compact_with_normals(
    v: np.ndarray, f: np.ndarray, n: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """compact() that carries a per-vertex normal array along."""
    used, inv = np.unique(f, return_inverse=True)
    return v[used], inv.reshape(f.shape).astype(f.dtype), n[used]


def compact(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop vertices not referenced by any face, reindexing faces.

    Retriangulating the top orphans the old top's interior vertices (they
    were only referenced by the deleted flat-top faces). They are harmless
    geometrically but waste memory and GLB bytes, so previews/exports
    compact them away.
    """
    used, inv = np.unique(f, return_inverse=True)
    return v[used], inv.reshape(f.shape).astype(f.dtype)


def combine(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (vertices, faces) tuples into one vertex/face set."""
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.vstack(vs), np.vstack(fs)


def to_trimesh(v: np.ndarray, f: np.ndarray) -> trimesh.Trimesh:
    """Wrap arrays without any processing (no vertex merging, no repair)."""
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def export_stl_bytes(v: np.ndarray, f: np.ndarray) -> bytes:
    """Binary STL of a single combined mesh."""
    return to_trimesh(v, f).export(file_type="stl")


def export_glb_bytes(named_parts: list[tuple]) -> bytes:
    """GLB scene with one named node per part.

    Node/geometry names survive into Three.js' GLTFLoader, which is how the
    frontend maps a clicked mesh back to a part id for selection. Entries
    are (name, v, f) or (name, v, f, normals); explicit normals (the
    engine's analytic top-field normals) take precedence over trimesh's
    averaged ones.
    """
    scene = trimesh.Scene()
    for entry in named_parts:
        name, v, f = entry[0], entry[1], entry[2]
        normals = entry[3] if len(entry) > 3 else None
        mesh = to_trimesh(v.astype(np.float32), f)
        if normals is not None:
            mesh.vertex_normals = normals.astype(np.float32)
        else:
            # Force smooth vertex normals so curvature reads well.
            mesh.vertex_normals  # noqa: B018  (property access computes+caches)
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    if isinstance(data, str):
        data = data.encode()
    return data


def export_stl_stream(v: np.ndarray, f: np.ndarray) -> io.BytesIO:
    return io.BytesIO(export_stl_bytes(v, f))


def export_render_glb(named_parts: list[tuple]) -> bytes:
    """GLB for RENDERING SOFTWARE, conforming to the glTF conventions the
    engine's internal data does not: +Y up and METERS (internal data is
    +Z up millimetres -- Blender silently corrects that, most other
    renderers import it lying on its back and 1000x too big).

    Per part it also ships what a material workflow needs out of the box:
      * planar top-projection UVs (0..1 over the part's footprint) --
        panels are height fields, so this unwraps cleanly for grain/
        fabric textures without manual UV work; walls share the border
        texels, which vinyl grain tolerates,
      * the engine's analytic vertex normals (smooth-shaded roll/crown),
      * zero-area faces removed (black-speckle fodder for backface-culling
        renderers), and winding verified consistent.
    """
    scene = trimesh.Scene()
    for entry in named_parts:
        name, v, f = entry[0], np.asarray(entry[1], np.float64), np.asarray(entry[2])
        n = entry[3] if len(entry) > 3 and entry[3] is not None else None

        # Drop degenerate (zero-area) faces.
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
        f = f[area2 > 1e-8]

        # Planar top-projection UVs from footprint xy, before reorienting.
        minxy = v[:, :2].min(axis=0)
        span = np.maximum(v[:, :2].max(axis=0) - minxy, 1e-9)
        uv = (v[:, :2] - minxy) / span

        # Z-up mm -> Y-up m: (x, y, z) -> (x, z, -y), scaled to meters.
        v2 = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]) * 0.001
        mesh = trimesh.Trimesh(vertices=v2.astype(np.float32), faces=f, process=False)
        if n is not None:
            n = np.asarray(n, np.float64)
            mesh.vertex_normals = np.column_stack(
                [n[:, 0], n[:, 2], -n[:, 1]]
            ).astype(np.float32)
        else:
            mesh.vertex_normals  # noqa: B018 (compute+cache smooth normals)
        if not mesh.is_winding_consistent:
            # Should not happen by construction; repair and fall back to
            # recomputed normals rather than ship a broken mesh.
            mesh.fix_normals()
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    if isinstance(data, str):
        data = data.encode()
    return data
