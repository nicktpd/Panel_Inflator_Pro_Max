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

    def sample(self, field: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        """Bilinear-sample a grid-shaped field at world xy points."""
        return ndimage.map_coordinates(
            field,
            [
                (pts_xy[:, 1] - self.ymin) / self.res + 1,
                (pts_xy[:, 0] - self.xmin) / self.res + 1,
            ],
            order=1,
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
    nx = int(np.ceil((pv[:, 0].max() - xmin) / res)) + 3
    ny = int(np.ceil((pv[:, 1].max() - ymin) / res)) + 3

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
        np.clip(((allp[:, 1] - ymin) / res).astype(int) + 1, 0, ny - 1),
        np.clip(((allp[:, 0] - xmin) / res).astype(int) + 1, 0, nx - 1),
    ] = True
    grid = ndimage.binary_closing(grid, np.ones((3, 3)), iterations=3)
    grid = fill_small_holes(grid, res)
    return FootprintRaster(grid, xmin, ymin, res)


def distance_field(fp: FootprintRaster, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Interior distance-to-boundary in mm: (raw, smoothed).

    Euclidean distance transform of the occupancy grid scaled to mm; the
    smoothed copy (gaussian, sigma in grid cells so preview and export
    smooth by the same *relative* amount) drives the crown profile so it
    has no faceting from the raster.

    The RAW field must be used for geometric decisions (triangle culling,
    interior-grid inset): it is exactly 0 outside the footprint and inside
    cutouts. The smoothed field bleeds positive values ~sigma cells past
    the boundary, which at coarse preview resolution is enough to wrongly
    keep triangles spanning a real hole.
    """
    raw = ndimage.distance_transform_edt(fp.grid) * fp.res
    smooth = ndimage.gaussian_filter(raw, sigma=sigma) if sigma > 0 else raw
    return raw, smooth


def crown_profile(dist: np.ndarray, crown: float, dref: float, exp: float) -> np.ndarray:
    """The validated pillow shape: dz = crown * clip(dist/dref, 0, 1) ** exp.

    Edges are pinned (dist=0 -> dz=0); points deeper than dref from any
    edge saturate at the full crown. exp < 1 makes the surface rise fast
    off the edge then flatten into a gentle dome, like stretched vinyl.
    """
    return crown * np.power(np.clip(dist / dref, 0.0, 1.0), exp)


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
    tau = np.where(occ, np.clip(d_mem / np.maximum(d_dist, res_c), 0.0, 1.0), 1.0)

    if factor > 1:
        tau = ndimage.zoom(tau, (ny / H, nx / W), order=1, mode="nearest")
        tau = tau[:ny, :nx]
        if tau.shape != grid.shape:
            tau = np.pad(
                tau,
                ((0, ny - tau.shape[0]), (0, nx - tau.shape[1])),
                mode="edge",
            )
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


def export_glb_bytes(named_parts: list[tuple[str, np.ndarray, np.ndarray]]) -> bytes:
    """GLB scene with one named node per part.

    Node/geometry names survive into Three.js' GLTFLoader, which is how the
    frontend maps a clicked mesh back to a part id for selection.
    """
    scene = trimesh.Scene()
    for name, v, f in named_parts:
        mesh = to_trimesh(v.astype(np.float32), f)
        # Force smooth vertex normals so the crown curvature reads well.
        mesh.vertex_normals  # noqa: B018  (property access computes+caches)
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    if isinstance(data, str):
        data = data.encode()
    return data


def export_stl_stream(v: np.ndarray, f: np.ndarray) -> io.BytesIO:
    return io.BytesIO(export_stl_bytes(v, f))
