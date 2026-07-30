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
import re
from pathlib import Path

import numpy as np
import shapely
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
                # Sample so spacing is ~SAMPLE_STEP in mm AFTER scaling
                # (seg_len is in source units, e.g. px or inches).
                n = max(int(math.ceil(seg_len * scale / SAMPLE_STEP)), 1)
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


def _rings_from_dxf(path: str, scale: float) -> tuple[list[np.ndarray], list[dict]]:
    """Closed rings + text labels from a DXF.

    Outline geometry comes only from LWPOLYLINE/POLYLINE/CIRCLE/ELLIPSE/
    ARC/SPLINE, flattened through ezdxf's path adaptor at the chordal
    tolerance. Open entities are skipped: a panel outline must be closed.

    Everything else in the file is tolerated. Annotation (TEXT/MTEXT --
    panel keys, dimensions, quantities) is deliberately NOT geometry: it
    is collected as labels [{text, x, y}] so the importer can name parts
    after their panel key (e.g. a lone "H" inside an outline).
    """
    import ezdxf
    from ezdxf.path import make_path

    doc = ezdxf.readfile(path)
    rings: list[np.ndarray] = []
    labels: list[dict] = []
    for entity in doc.modelspace():
        kind = entity.dxftype()
        if kind in ("TEXT", "MTEXT"):
            try:
                text = entity.plain_text() if kind == "MTEXT" else entity.dxf.text
                ins = entity.dxf.insert
                labels.append({
                    "text": str(text).strip(),
                    "x": float(ins.x) * scale,
                    "y": float(ins.y) * scale,
                })
            except Exception:
                pass
            continue
        if kind not in (
            "LWPOLYLINE", "POLYLINE", "CIRCLE", "ELLIPSE", "ARC", "SPLINE",
        ):
            continue
        try:
            epath = make_path(entity)
        except Exception:
            continue
        # Flattening tolerance is in DRAWING units; divide by scale so it
        # equals FLATTEN_TOL in mm (an inch drawing flattened at 0.5 would
        # otherwise mean 12.7 mm chords -- visibly polygonal circles).
        tol = FLATTEN_TOL / scale
        pts = np.array([(v.x, v.y) for v in epath.flattening(tol)], dtype=np.float64)
        if len(pts) < 3:
            continue
        closed = epath.is_closed or np.linalg.norm(pts[0] - pts[-1]) < tol * 4
        if not closed:
            continue
        rings.append(pts * scale)
    return rings, labels


def _densify_ring(ring: np.ndarray, max_seg: float = SAMPLE_STEP) -> np.ndarray:
    """Insert points so no ring segment is longer than ``max_seg`` mm.

    DXF straight segments flatten to just their endpoints, so a 52-inch
    edge arrives as ONE segment. Everything downstream needs boundary
    density: loft rings that can follow the crown, a fine seam ring for
    the retriangulated top, and smooth viewer shading. (Production bug:
    a 52x8-inch right triangle rendered as a folded mess because its
    entire outline was 3 vertices.)
    """
    out: list[np.ndarray] = []
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        out.append(a)
        seg = float(np.linalg.norm(b - a))
        if seg > max_seg:
            k = int(math.ceil(seg / max_seg))
            for j in range(1, k):
                out.append(a + (b - a) * (j / k))
    return np.asarray(out, dtype=np.float64)


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
    cleaned = [
        _densify_ring(r) for r in (_clean_ring(r) for r in rings) if r is not None
    ]
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
    """Per-vertex UNIT bisector normals pointing INTO the material.

    Rings are oriented so material lies to the LEFT of travel (CCW shells,
    CW holes -- shapely orient() guarantees this), so the left normal of
    the angle bisector points inward for both.

    Deliberately NOT miter-scaled: extending a convex corner vertex to
    the miter point (d / cos(half-angle)) pushes it past its neighbours'
    offset line, and the inset ring then overlaps itself along the rim
    (production case: every corner of a plain square leaked open edges).
    A plain d-along-the-bisector offset under-cuts convex corners into a
    small chamfer instead -- which is exactly how wrapped vinyl behaves
    at a real corner fold (see the corner photos in reference/NOTES.md:
    the fabric rounds off, it never forms a sharp miter).
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
    return bis / bn


def _inset_ring_safe(
    poly: sg.Polygon, ring: np.ndarray, miter_normals: np.ndarray, d: float
) -> np.ndarray:
    """Inset ring vertices by ``d`` without crossing the outline.

    Near a sharp tail (production case: a 52x8-inch right triangle, ~9
    degree tip) the local width drops below 2*d and plain perpendicular
    offsets from the two converging edges cross each other, folding the
    loft. For each vertex we try the full miter offset and progressively
    smaller fractions, keeping the candidate (still inside the polygon)
    with the best clearance-to-boundary, capped at the requested d. Wide
    regions keep the exact offset; a narrowing tail pinches smoothly to
    its centreline, which is what wrapped vinyl does anyway.
    """
    if d <= 1e-9:
        return ring.copy()
    n = len(ring)
    fracs = np.array([1.0, 0.5, 0.25, 0.125])
    stack = np.stack([ring + miter_normals * (d * f) for f in fracs])  # (4, n, 2)
    pts = shapely.points(stack.reshape(-1, 2))
    inside = shapely.contains(poly, pts).reshape(len(fracs), n)
    clearance = shapely.distance(poly.boundary, pts).reshape(len(fracs), n)
    score = np.where(inside, np.minimum(clearance, d), -np.inf)
    score -= 1e-4 * (1.0 - fracs)[:, None]  # prefer the fuller offset on ties
    best = score.argmax(axis=0)
    out = stack[best, np.arange(n)]
    hopeless = ~np.isfinite(score[best, np.arange(n)])
    out[hopeless] = ring[hopeless]  # tighter than the tip itself: stay put
    return out


def _delaunay_cap(
    loops: list[np.ndarray], test_poly: sg.Polygon
) -> tuple[np.ndarray, np.ndarray]:
    """Flat cap triangulation that CONFORMS to a dense boundary.

    Earcut is the wrong tool once rings are densified: its fans create
    zero-area slivers on collinear boundary points, trimesh's processing
    deletes those, and the rim tears open (production case: 1225 open
    edges on a plain square). Delaunay over the exact loop vertices plus
    a coarse interior grid never produces degenerate boundary slivers,
    and with 2 mm boundary spacing the rim segments are always Delaunay
    edges — the cap welds to the walls by construction. Triangles are
    kept if their centroid is inside (or within 0.25 mm of) the polygon,
    which tolerates the locally self-touching loops of pinched tails.
    """
    from scipy.spatial import Delaunay

    from . import meshops

    pts = np.vstack(loops)
    minx, miny, maxx, maxy = test_poly.bounds
    step = 6.0
    gx, gy = np.meshgrid(
        np.arange(minx + step / 2, maxx, step),
        np.arange(miny + step / 2, maxy, step),
    )
    gp = np.column_stack([gx.ravel(), gy.ravel()])
    if len(gp):
        gpts = shapely.points(gp)
        inside = shapely.contains(test_poly, gpts)
        clear = shapely.distance(test_poly.boundary, gpts) > step / 2
        gp = gp[inside & clear]
    allpts = np.vstack([pts, gp]) if len(gp) else pts
    tri = Delaunay(allpts)
    cent = allpts[tri.simplices].mean(axis=1)
    keep = shapely.dwithin(test_poly, shapely.points(cent), 0.25)
    return allpts, meshops.ensure_up_normals(allpts, tri.simplices[keep])


def extrude_with_roundover(
    poly: sg.Polygon, thickness: float, roundover: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a polygon to ``thickness`` with a top-edge fillet.

    The fillet is a quarter circle approximated by 3 loft rings (spec
    allows exactly this): at angle phi the ring is inset by r*(1-cos phi)
    and raised to z = (T - r) + r*sin phi. The top cap reuses the last
    loft ring's exact vertices so the seam welds bit-identically. All
    rings share vertex count, so walls are clean closed quad strips;
    caps are earcut-triangulated. Insets are clearance-limited (see
    _inset_ring_safe) so sharp tails pinch instead of self-intersecting.
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
    top_loops: list[np.ndarray] = []  # last-level loop per ring, for the cap

    for ring in rings:
        normals = _ring_normals_inward(ring)
        starts = []
        last_loop = ring
        for z, inset in levels:
            starts.append(sum(len(v) for v in verts))
            loop = _inset_ring_safe(poly, ring, normals, inset)
            verts.append(np.column_stack([loop, np.full(len(ring), z)]))
            last_loop = loop
        top_loops.append(last_loop)
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

    # Bottom cap (faces down) on the original dense rings. Same conforming
    # triangulation as the top: earcut fans over densified (collinear)
    # boundary points create zero-area slivers that mesh processing then
    # deletes, tearing the rim open.
    rings0 = [np.array(poly.exterior.coords[:-1], dtype=np.float64)]
    rings0 += [np.array(h.coords[:-1], dtype=np.float64) for h in poly.interiors]
    v2d, f2d = _delaunay_cap(rings0, poly)
    base = sum(len(v) for v in verts)
    verts.append(np.column_stack([v2d, np.zeros(len(v2d))]))
    faces.append(f2d[:, ::-1] + base)

    # Top cap (faces up) built directly ON the last loft level's vertices
    # so the rim welds to the walls by construction. The loop polygon may
    # be locally self-touching at a pinched tail; a buffer(0)-repaired
    # COPY is used only as the containment test, never for the vertices.
    base = sum(len(v) for v in verts)
    if r > 1e-6:
        top_poly = sg.Polygon(top_loops[0], top_loops[1:])
        if not top_poly.is_valid:
            top_poly = top_poly.buffer(0)
            if top_poly.is_empty:
                raise ValueError(
                    "roundover radius too large for this outline; reduce it"
                )
        v2d, f2d = _delaunay_cap(top_loops, top_poly)
    else:
        v2d, f2d = _delaunay_cap(rings0, poly)
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
# part naming from annotation text
# ---------------------------------------------------------------------------

_QTY_RE = re.compile(r"^QTY\.?\s*[x×]?\s*(\d+)", re.IGNORECASE)
# A panel key is a short standalone identifier: "H", "F2", "A-1"...
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]{0,3}$")


def _name_hint_for(poly: sg.Polygon, labels: list[dict]) -> str | None:
    """Derive a display name for one outline from nearby annotation text.

    Labels inside the outline win; if none are inside (some drawings put
    the caption beside the part), the nearest label cluster within one
    bounding-box diagonal is used. The shortest key-looking string is the
    panel key; a "QTY n" line becomes a multiplier suffix.
    """
    if not labels:
        return None
    inside, nearby = [], []
    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    for lab in labels:
        if not lab["text"]:
            continue
        pt = sg.Point(lab["x"], lab["y"])
        if poly.contains(pt):
            inside.append(lab)
        elif poly.exterior.distance(pt) < diag:
            nearby.append(lab)
    pool = inside or nearby
    if not pool:
        return None
    key = None
    qty = None
    for lab in pool:
        text = lab["text"]
        m = _QTY_RE.match(text)
        if m:
            qty = int(m.group(1))
        elif _KEY_RE.match(text) and (key is None or len(text) < len(key)):
            key = text
    if key is None:
        return None
    name = f"Panel {key.upper()}"
    if qty and qty > 1:
        name += f" ×{qty}"
    return name


# ---------------------------------------------------------------------------
# top-level entry
# ---------------------------------------------------------------------------


def parse_polygons(path: str, scale: float = 1.0) -> tuple[list[sg.Polygon], list[str | None]]:
    """Parse an SVG/DXF into nested shapely polygons + per-part name hints.

    Shared front half of both the importer and the outline preview, so
    the shape the preview draws is byte-for-byte the shape the extruder
    will use (same parsing, densifying, even-odd nesting, and naming).
    """
    suffix = Path(path).suffix.lower()
    labels: list[dict] = []
    if suffix == ".svg":
        rings = _rings_from_svg(path, scale)
    elif suffix == ".dxf":
        rings, labels = _rings_from_dxf(path, scale)
    else:
        raise ValueError(f"unsupported 2D format: {suffix}")

    if not rings:
        raise ValueError("no closed outlines found in file")

    polys = rings_to_polygons(rings)
    if not polys:
        raise ValueError("outlines could not be assembled into polygons")

    # Name parts from annotation while text and outlines still share the
    # drawing's coordinate system.
    hints = [_name_hint_for(poly, labels) for poly in polys]
    return polys, hints


def outline_preview(path: str, scale: float = 1.0) -> list[dict]:
    """Extract outlines for the UI preview WITHOUT extruding or pillowing.

    Returns one dict per part with the exterior loop, hole loops, bounding
    box and (for DXF) the annotation-derived name. Coordinates are in the
    drawing's own units multiplied by ``scale`` (so pass the user's chosen
    unit factor to preview in millimetres). Fast: pure 2D parsing.
    """
    polys, hints = parse_polygons(path, scale)
    out = []
    for poly, hint in zip(polys, hints):
        minx, miny, maxx, maxy = poly.bounds
        out.append({
            "name": hint,
            "exterior": [[round(x, 4), round(y, 4)] for x, y in poly.exterior.coords],
            "holes": [
                [[round(x, 4), round(y, 4)] for x, y in ring.coords]
                for ring in poly.interiors
            ],
            "bbox": [minx, miny, maxx, maxy],
            "width": maxx - minx,
            "height": maxy - miny,
        })
    return out


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
    polys, hints = parse_polygons(path, scale)

    # Normalize position: whole drawing translated to the positive quadrant.
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)

    part_arrays = []
    for poly in polys:
        from shapely.affinity import translate

        poly = translate(poly, xoff=-minx, yoff=-miny)
        v, f = extrude_with_roundover(poly, thickness, roundover)
        part_arrays.append((v, f))

    return import_stl.describe_parts_from_arrays(
        part_arrays, hardware_threshold=0, name_hints=hints
    )
