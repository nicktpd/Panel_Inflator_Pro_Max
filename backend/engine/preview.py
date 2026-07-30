"""Per-part cached pillow computation and GLB/STL assembly.

Caching model: a part's pillowed mesh depends only on (its source
geometry, its effective parameters, the raster/grid resolution, and its
painted mask version). Source geometry is immutable after import, so the
cache key is a hash of the parameter dict + resolution + mask version.
Tweaking one part in an 11-panel board therefore recomputes exactly one
part; the other ten load from ``projects/<id>/cache/``.

Preview resolution is 4 mm raster / 10 mm top grid (fast, < 3 s for an
11-panel board); export is 2 mm / 6 mm (full quality).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import meshops, pillow

PREVIEW_RES = 4.0
PREVIEW_GRID = 10.0
EXPORT_RES = 2.0
EXPORT_GRID = 6.0

# Bump whenever the pillow engine's OUTPUT changes for identical inputs
# (algorithm fixes, not parameter changes), so stale caches from older
# engine versions are never served. 2 = exact-distance-field rework.
# 3 = monotone side-wall belly (no bell flare below the dome's landing).
# 4 = convex roll-over: crown added through the roll zone (no fade
#     shelf), raw-distance crown (full rim drop), continuous belly field
#     across seam, corner pinholes closed.
# 5 = depth-blended drive field: exact distance near the rim, smoothed
#     interior (no medial-axis/corner-bisector crease lines on top).
# 6 = membrane face: interior driven by the Poisson/Spalding wall
#     distance -- no fold lines on the padded face by construction;
#     raster lattice centred so all fields mirror symmetric shapes.
# 7 = chained-entity DXF import (LINE/ARC outlines), finer arc
#     flattening, collinearity nudge + exact inside culls: watertight
#     caps/top on every outline shape.
ENGINE_VERSION = 7


def params_key(params: dict, res: float, grid_step: float, mask_version: int) -> str:
    """Stable short hash identifying one pillow computation."""
    blob = json.dumps(
        {"p": params, "res": res, "grid": grid_step, "mask": mask_version,
         "engine": ENGINE_VERSION},
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def part_source_path(cache_dir: Path, part_id: int) -> Path:
    return cache_dir / f"src_part{part_id}.npz"


def save_part_source(cache_dir: Path, part_id: int, pv: np.ndarray, pf: np.ndarray) -> None:
    """Store a part's original geometry (float32) for fast reload.

    Lives in cache/ (regenerable from the source file) so cloud sync can
    skip it; ensure_part_sources() rebuilds it after a cache wipe.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        part_source_path(cache_dir, part_id),
        v=pv.astype(np.float32),
        f=pf.astype(np.int32),
    )


def load_part_source(cache_dir: Path, part_id: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(part_source_path(cache_dir, part_id)) as z:
        return z["v"], z["f"]


def compute_part_cached(
    cache_dir: Path,
    part_id: int,
    pv: np.ndarray,
    pf: np.ndarray,
    params: dict,
    res: float,
    grid_step: float,
    mask: np.ndarray | None = None,
    mask_version: int = 0,
    corners: list[list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Pillow one part, memoized on disk per (params, res, mask) hash.

    Returns (v, f, normals) where normals are the engine's analytic
    top-field vertex normals (None on old cache entries). corners is
    derived from the part's immutable source outline, so it doesn't need
    to be part of the cache key.
    """
    key = params_key(params, res, grid_step, mask_version)
    cpath = cache_dir / f"part{part_id}_{key}.npz"
    if cpath.exists():
        try:
            with np.load(cpath) as z:
                return z["v"], z["f"], (z["n"] if "n" in z.files else None)
        except Exception:
            cpath.unlink(missing_ok=True)  # corrupt cache: recompute

    v, f, n = pillow.pillow_panel(
        pv,
        pf,
        crown=params["crown"],
        dref=params["dref"],
        exp=params["exp"],
        sigma=params["sigma"],
        w_exp=params.get("w_exp", 1.5),
        tension=params.get("tension", 0.7),
        edge_roll=params.get("edge_roll", 0.0),
        corners=corners,
        res=res,
        grid_step=grid_step,
        mask=mask,
        return_normals=True,
    )
    v = v.astype(np.float32)
    f = f.astype(np.int32)
    n = n.astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cpath.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, v=v, f=f, n=n)
    tmp.replace(cpath)
    return v, f, n


def prune_part_cache(cache_dir: Path, keep_per_part: int = 4) -> None:
    """Keep the most recent N cached results per part; delete the rest."""
    if not cache_dir.exists():
        return
    by_part: dict[str, list[Path]] = {}
    for p in cache_dir.glob("part*_*.npz"):
        stem = p.name.split("_")[0]
        by_part.setdefault(stem, []).append(p)
    for paths in by_part.values():
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old in paths[keep_per_part:]:
            old.unlink(missing_ok=True)


def build_glb(named_parts: list[tuple[str, np.ndarray, np.ndarray]]) -> bytes:
    """GLB with one named node per part (names drive frontend selection)."""
    return meshops.export_glb_bytes(named_parts)


def build_stl(parts: list[tuple[np.ndarray, np.ndarray]]) -> bytes:
    """Single combined binary STL."""
    v, f = meshops.combine(parts)
    return meshops.export_stl_bytes(v, f)
