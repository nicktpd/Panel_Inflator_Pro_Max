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


def distance_field(fp: FootprintRaster, sigma: float) -> np.ndarray:
    """Smoothed interior distance-to-boundary in mm.

    Euclidean distance transform of the occupancy grid (cells to nearest
    empty cell) scaled to mm, then gaussian smoothed so the crown profile
    has no faceting from the raster. sigma is in grid cells, so preview
    (4 mm cells) and export (2 mm cells) smooth by the same *relative*
    amount the reference algorithm validated.
    """
    dist = ndimage.distance_transform_edt(fp.grid) * fp.res
    if sigma > 0:
        dist = ndimage.gaussian_filter(dist, sigma=sigma)
    return dist


def crown_profile(dist: np.ndarray, crown: float, dref: float, exp: float) -> np.ndarray:
    """The validated pillow shape: dz = crown * clip(dist/dref, 0, 1) ** exp.

    Edges are pinned (dist=0 -> dz=0); points deeper than dref from any
    edge saturate at the full crown. exp < 1 makes the surface rise fast
    off the edge then flatten into a gentle dome, like stretched vinyl.
    """
    return crown * np.power(np.clip(dist / dref, 0.0, 1.0), exp)


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
