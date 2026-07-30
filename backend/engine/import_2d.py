"""2D outline import (SVG / DXF) -> extruded base solid -> pillow-ready parts.

Pipeline:
  1. Parse the file into closed 2D rings (polylines at ~0.5 mm chordal
     tolerance; SVG transforms are applied, y is flipped to y-up).
  2. Nest rings even-odd: outers become shells, odd-depth rings become
     holes of their immediate parent. Disjoint outers = separate parts.
  3. Extrude each polygon to the base thickness with a 3-ring quarter-
     circle roundover on the top perimeter edge, so the pre-pillow solid
     resembles a real CAD part (fillet included).
  4. The result feeds the exact Phase 1 pillow pipeline; holes in the
     polygon produce the fabric-dip behaviour automatically via the
     distance field.

Geometry notes:
  * The roundover is applied by insetting each ring along per-vertex
    angle-bisector normals (miter-limited). That keeps a 1:1 vertex
    correspondence between loft rings, which makes the side wall a clean
    quad strip. True polygon offsetting (shapely buffer) can change ring
    topology and would break the loft; for the small radii involved
    (default 8 mm) the bisector approximation is visually identical.
  * Caps are triangulated with mapbox-earcut via trimesh; if that wheel
    is unavailable, a pure-python fallback triangulates with scipy
    Delaunay culled to the polygon (same trick the pillow engine uses).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import shapely.geometry as sg
from shapely.geometry.polygon import orient

from . import import_stl

# Chordal tolerance for flattening curves, in mm.
FLATTEN_TOL = 0.5
# Sample spacing along segments when flattening (keeps chord error far
# below FLATTEN_TOL for typical panel-scale curvature).
SAMPLE_STEP = 2.0
# Rings with area below this are noise and dropped (mm^2).
MIN_RING_AREA = 4.0


# ---------------------------------------------------------------------------
# parsing: file -> list of closed rings ((n,2) float arrays, mm)
# ---------------------------------------------------------------------------


def _rings_from_svg(path: str, scale: float) -> list[np.ndarray]:
    """Closed rings from an SVG. Uses Document so group transforms apply.

    SVG y points down; we flip to y-up and later translate the whole
    drawing to the positive quadrant.
    """
    from svgpathtools import Document

    doc = Document(path)
    rings: list[np.ndarray] = []
    for spath in doc.paths():
        for sub in spath.continuous_subpaths():
            if len(sub) == 0:
                continue
            pts: list[complex] = []
            for seg in sub:
                try:
                    seg_len = seg.length(error=1e-3)
                except Exception:
                    seg_len = abs(seg.end - seg.start)
                n = max(int(math.ceil(seg_len / SAMPLE_STEP)), 1)
                for i in range(n):
                    pts.append(seg.point(i / n))
            if not pts:
                continue
            # Treat as closed if endpoints (nearly) coincide.
            end = sub[-1].end
            if abs(end - pts[0]) > FLATTEN_TOL * 4:
                continue  # open contour: not a panel outline
            arr = np.array([[p.real, -p.imag] for p in pts], dtype=np.float64) * scale
            rings.append(arr)
    return rings


def _rings_from_dxf(path: str, scale: float) -> list[np.ndarray]:
    """Closed rings from DXF LWPOLYLINE/POLYLINE/CIRCLE/ELLIPSE/ARC/SPLINE.

    Entities are flattened through ezdxf's path adaptor at the chordal
    tolerance. Open entities (lines, open arcs) are skipped: a panel
    outline must be a closed contour.
    """
    import ezdxf
    from ezdxf.path import make_path

    doc = ezdxf.readfile(path)
    rings: list[np.ndarray] = []
    for entity in doc.modelspace():
        if entity.dxftype() not in (
            "LWPOLYLINE", "POLYLINE", "CIRCLE", "ELLIPSE", "ARC", "SPLINE",
        ):
            continue
        try:
            epath = make_path(entity)
        except Exception:
            continue
        pts = np.array([(v.x, v.y) for v in epath.flattening(FLATTEN_TOL)], dtype=np.float64)
        if len(pts) < 3:
            continue
        closed = epath.is_closed or np.linalg.norm(pts[0] - pts[-1]) < FLATTEN_TOL * 4
        if not closed:
            continue
        rings.append(pts * scale)
    return rings


def _clean_ring(ring: np.ndarray) -> np.ndarray | None:
    """Drop repeated points (incl. closing duplicate); reject degenerates."""
    if len(ring) and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < 3:
        return None
    d = np.linalg.norm(np.diff(ring, axis=0, append=ring[:1]), axis=1)
    ring = ring[d > 1e-6]
    if len(ring) < 3:
        return None
    if abs(sg.Polygon(ring).area) < MIN_RING_AREA:
        return None
    return ring


# ---------------------------------------------------------------------------
# nesting: rings -> polygons with holes (even-odd rule)
# ---------------------------------------------------------------------------


def rings_to_polygons(rings: list[np.ndarray]) -> list[sg.Polygon]:
    """Nest rings even-odd: depth 0,2,4.. = shells; odd depth = holes.

    A hole is attached to its immediate (smallest-area containing) shell.
    Deeper even rings become independent shells again (island in a hole).
    """
    cleaned = [r for r in (_clean_ring(r) for r in rings) if r is not None]
    polys = [sg.Polygon(r) for r in cleaned]
    for i, p in enumerate(polys):
        if not p.is_valid:
            polys[i] = p.buffer(0)

    order = sorted(range(len(polys)), key=lambda i: -abs(polys[i].area))
    depth = {}
    parent = {}
    for idx, i in enumerate(order):
        pt = polys[i].representative_point()
        containers = [j for j in order[:idx] if polys[j].contains(pt)]
        depth[i] = len(containers)
        if containers:
            parent[i] = min(containers, key=lambda j: abs(polys[j].area))

    result = []
    for i in order:
        if depth[i] % 2 == 0:
            holes = [
                cleaned[h]
                for h in order
                if depth.get(h, 0) == depth[i] + 1 and parent.get(h) == i
            ]
            poly = orient(sg.Polygon(cleaned[i], holes), sign=1.0)  # CCW shell, CW holes
            if poly.is_valid and poly.area > MIN_RING_AREA:
                result.append(poly)
    return result


# ---------------------------------------------------------------------------
# triangulation (earcut with pure-python fallback)
# ---------------------------------------------------------------------------


def triangulate_cap(poly: sg.Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a polygon (with holes) into (verts2d, faces)."""
    import trimesh

    try:
        return trimesh.creation.triangulate_polygon(poly, engine="earcut")
    except BaseException:
        # Pure-python fallback: Delaunay over the boundary vertices,
        # culled to triangles whose centroid is inside the polygon.
        from scipy.spatial import Delaunay

        pts = np.array(poly.exterior.coords[:-1])
        for hole in poly.interiors:
            pts = np.vstack([pts, np.array(hole.coords[:-1])])
        tri = Delaunay(pts)
        cent = pts[tri.simplices].mean(axis=1)
        inside = np.array(
            [poly.contains(sg.Point(c)) for c in cent], dtype=bool
        )
        return pts, tri.simplices[inside]


# ---------------------------------------------------------------------------
# extrusion with top-edge roundover
# ---------------------------------------------------------------------------


def _ring_normals_inward(ring: np.ndarray) -> np.ndarray:
    """Per-vertex miter normals pointing INTO the material.

    Rings are oriented so material lies to the LEFT of travel (CCW shells,
    CW holes -- shapely orient() guarantees this), so the left normal of
    the angle bisector points inward for both. Miter length is limited so
    sharp corners don't spike.
    """
    prev = np.roll(ring, 1, axis=0)
    nxt = np.roll(ring, -1, axis=0)
    t1 = ring - prev
    t2 = nxt - ring
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 /= np.maximum(np.linalg.norm(t2, axis=1, keepdims=True), 1e-12)
    n1 = np.column_stack([-t1[:, 1], t1[:, 0]])  # left normals
    n2 = np.column_stack([-t2[:, 1], t2[:, 0]])
    bis = n1 + n2
    bn = np.linalg.norm(bis, axis=1, keepdims=True)
    # Degenerate (180 degree turn): fall back to one edge normal.
    flat = bn[:, 0] < 1e-9
    bis[flat] = n1[flat]
    bn[flat] = 1.0
    bis /= bn
    # Offset scale so EDGES move by d: 1/cos(half-turn); miter-limited.
    cos_half = np.sqrt(np.clip((1.0 + np.sum(t1 * t2, axis=1)) / 2.0, 0.1, 1.0))
    return bis / cos_half[:, None]


def extrude_with_roundover(
    poly: sg.Polygon, thickness: float, roundover: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a polygon to ``thickness`` with a top-edge fillet.

    The fillet is a quarter circle approximated by 3 loft rings (spec
    allows exactly this): at angle phi the ring is inset by r*(1-cos phi)
    and raised to z = (T - r) + r*sin phi. The top cap is the polygon
    inset by the full radius. All rings share vertex count, so walls are
    clean closed quad strips; caps are earcut-triangulated.
    """
    import trimesh

    thickness = float(thickness)
    r = float(min(roundover, thickness * 0.45))

    rings = [np.array(poly.exterior.coords[:-1], dtype=np.float64)]
    rings += [np.array(h.coords[:-1], dtype=np.float64) for h in poly.interiors]

    if r > 1e-6:
        phis = [0.0, math.radians(30), math.radians(60), math.radians(90)]
        levels = [(0.0, 0.0)] + [
            ((thickness - r) + r * math.sin(phi), r * (1.0 - math.cos(phi))) for phi in phis
        ]
    else:
        levels = [(0.0, 0.0), (thickness, 0.0)]

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    ring_level_start: list[list[int]] = []

    for ring in rings:
        normals = _ring_normals_inward(ring)
        starts = []
        for z, inset in levels:
            starts.append(sum(len(v) for v in verts))
            loop = ring + normals * inset
            verts.append(np.column_stack([loop, np.full(len(ring), z)]))
        ring_level_start.append(starts)
        n = len(ring)
        idx = np.arange(n)
        nxt = (idx + 1) % n
        for k in range(len(levels) - 1):
            a0 = starts[k] + idx
            a1 = starts[k] + nxt
            b0 = starts[k + 1] + idx
            b1 = starts[k + 1] + nxt
            # Material is left of travel, so outward is right: this
            # winding gives outward-facing wall normals for both shells
            # and holes.
            faces.append(np.column_stack([a0, a1, b1]))
            faces.append(np.column_stack([a0, b1, b0]))

    # Bottom cap (faces down) from the original polygon.
    v2d, f2d = triangulate_cap(poly)
    base = sum(len(v) for v in verts)
    verts.append(np.column_stack([v2d, np.zeros(len(v2d))]))
    faces.append(f2d[:, ::-1] + base)

    # Top cap (faces up) from the polygon inset by the full radius.
    top_inset = r if r > 1e-6 else 0.0
    if top_inset > 0.0:
        top_rings = []
        for ring in rings:
            top_rings.append(ring + _ring_normals_inward(ring) * top_inset)
        top_poly = sg.Polygon(top_rings[0], top_rings[1:])
        if not top_poly.is_valid:
            top_poly = top_poly.buffer(0)
            if top_poly.is_empty:
                raise ValueError(
                    "roundover radius too large for this outline; reduce it"
                )
            if top_poly.geom_type == "MultiPolygon":
                top_poly = max(top_poly.geoms, key=lambda g: g.area)
    else:
        top_poly = poly
    v2d, f2d = triangulate_cap(top_poly)
    base = sum(len(v) for v in verts)
    verts.append(np.column_stack([v2d, np.full(len(v2d), thickness)]))
    faces.append(f2d + base)

    v = np.vstack(verts)
    f = np.vstack(faces)
    # Merge coincident seam vertices (cap rims == top/bottom loft rings)
    # so the pillow engine sees one connected solid with a welded seam.
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    mesh.merge_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


# ---------------------------------------------------------------------------
# top-level entry
# ---------------------------------------------------------------------------


def load_2d_as_parts(
    path: str,
    scale: float = 1.0,
    thickness: float = 50.8,
    roundover: float = 8.0,
) -> list[dict]:
    """Parse an SVG/DXF and return pillow-ready classified parts.

    Every flat-topped extrusion is pillowable regardless of face count
    (2D extrusions are lean meshes; the 10k STL hardware threshold does
    not apply).
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".svg":
        rings = _rings_from_svg(path, scale)
    elif suffix == ".dxf":
        rings = _rings_from_dxf(path, scale)
    else:
        raise ValueError(f"unsupported 2D format: {suffix}")

    if not rings:
        raise ValueError("no closed outlines found in file")

    polys = rings_to_polygons(rings)
    if not polys:
        raise ValueError("outlines could not be assembled into polygons")

    # Normalize position: whole drawing translated to the positive quadrant.
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)

    part_arrays = []
    for poly in polys:
        from shapely.affinity import translate

        poly = translate(poly, xoff=-minx, yoff=-miny)
        v, f = extrude_with_roundover(poly, thickness, roundover)
        part_arrays.append((v, f))

    return import_stl.describe_parts_from_arrays(part_arrays, hardware_threshold=0)
