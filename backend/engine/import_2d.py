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

# Chordal tolerance for flattening curves, in mm. The polygon IS the
# geometry the walls and every distance field follow, and glossy vinyl
# shading amplifies its facet joints: at 0.1 mm sagitta (~10 mm chords on
# a 125 mm arc) the roll direction wobbled per facet and the bulb
# shoulder of the petal read as wrinkled lumps. 0.01 mm (~3 mm chords)
# is geometrically indistinguishable from the true arc; the densify pass
# governs final point spacing, so the mesh doesn't grow.
FLATTEN_TOL = 0.01
# Endpoint tolerance (mm) for treating a single entity/subpath as closed
# -- deliberately independent of FLATTEN_TOL (it used to be derived from
# it, and tightening the flattening would have silently stopped slightly
# sloppy closed polylines from importing).
CLOSE_TOL = 2.0
# Sample spacing along segments when flattening (keeps chord error far
# below FLATTEN_TOL for typical panel-scale curvature).
SAMPLE_STEP = 2.0
# Rings with area below this are noise and dropped (mm^2).
MIN_RING_AREA = 4.0
# Endpoint-matching tolerance (mm) when chaining OPEN entities
# (LINE/ARC/open polylines) into closed loops. Real CAD exports commonly
# draw one outline as many separate entities that meet end-to-end;
# 1 mm absorbs sloppy drafting without gluing distinct parts together.
CHAIN_TOL = 1.0


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
            if abs(end - pts[0]) * scale > CLOSE_TOL:
                continue  # open contour: not a panel outline
            arr = np.array([[p.real, -p.imag] for p in pts], dtype=np.float64) * scale
            rings.append(arr)
    return rings


def _chain_fragments(frags: list[np.ndarray], tol: float = CHAIN_TOL) -> list[np.ndarray]:
    """Stitch open point-chains (mm) into closed rings by endpoint matching.

    Real CAD exports very often draw ONE outline as many separate open
    entities -- e.g. two LINEs and two ARCs meeting end-to-end (the
    "Pedal" file that motivated this). Greedy chaining: grow a chain by
    appending any fragment whose start or end lies within ``tol`` of the
    chain's end (reversing the fragment if needed); when neither end can
    grow, flip the chain once and keep trying; a chain whose two ends
    meet within ``tol`` becomes a ring. Dead-end chains (genuinely open
    geometry: dimension leaders, centerlines) are dropped, same as open
    entities always were.
    """
    frags = [np.asarray(f, dtype=np.float64) for f in frags if len(f) >= 2]
    used = [False] * len(frags)
    rings: list[np.ndarray] = []
    for i in range(len(frags)):
        if used[i]:
            continue
        used[i] = True
        chain = frags[i]
        flipped_once = False
        while True:
            if len(chain) >= 3 and np.linalg.norm(chain[0] - chain[-1]) < tol:
                rings.append(chain)
                break
            grew = False
            end = chain[-1]
            for j in range(len(frags)):
                if used[j]:
                    continue
                fj = frags[j]
                if np.linalg.norm(fj[0] - end) < tol:
                    chain = np.vstack([chain, fj[1:]])
                    used[j] = True
                    grew = True
                    break
                if np.linalg.norm(fj[-1] - end) < tol:
                    chain = np.vstack([chain, fj[::-1][1:]])
                    used[j] = True
                    grew = True
                    break
            if grew:
                flipped_once = False
                continue
            if not flipped_once:
                chain = chain[::-1]
                flipped_once = True
                continue
            break  # dead end on both sides: not a closed outline
    return rings


def _rings_from_dxf(path: str, scale: float) -> tuple[list[np.ndarray], list[dict]]:
    """Closed rings + text labels from a DXF.

    Outline geometry comes from LINE/LWPOLYLINE/POLYLINE/CIRCLE/ELLIPSE/
    ARC/SPLINE, flattened through ezdxf's path adaptor at the chordal
    tolerance. Entities that are closed by themselves become rings
    directly; OPEN entities (lines, arcs, unclosed polylines) are
    collected and chained end-to-end into rings (_chain_fragments) --
    many CAD exports draw one outline as separate segments. Fragments
    that never close (dimension leaders, centerlines) are dropped.

    Everything else in the file is tolerated. Annotation (TEXT/MTEXT --
    panel keys, dimensions, quantities) is deliberately NOT geometry: it
    is collected as labels [{text, x, y}] so the importer can name parts
    after their panel key (e.g. a lone "H" inside an outline).
    """
    import ezdxf
    from ezdxf.path import make_path

    doc = ezdxf.readfile(path)
    rings: list[np.ndarray] = []
    fragments: list[np.ndarray] = []
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
            "LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ELLIPSE", "ARC",
            "SPLINE",
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
        if len(pts) < 2:
            continue
        closed = (
            epath.is_closed
            or np.linalg.norm(pts[0] - pts[-1]) * scale < CLOSE_TOL
        )
        if closed and len(pts) >= 3:
            rings.append(pts * scale)
        else:
            fragments.append(pts * scale)
    rings.extend(_chain_fragments(fragments))
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


def _nudge_collinear(ring: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Nudge exactly-collinear ring points ~``eps`` mm toward the material.

    Densifying a straight outline stretch produces EXACTLY collinear
    points. A boundary that is locally straight makes the cap/top
    Delaunay triangulations fragile there: the strip between the convex
    hull's long chord and the boundary degenerates, and the resulting
    sliver handling tears the cap/wall weld open along straight and
    coarsely-flattened edges. A ~0.01 mm inward dogleg is far below
    anything visible (or cuttable), and because walls, caps and the
    pillow top all reuse the same ring arrays, everything stays welded.

    The amplitude VARIES per point (deterministically, from the point
    index): with a constant nudge the whole displaced row is collinear
    again at its new offset, and the hull-chord strip collapses into a
    few giant slivers that survive area-based culls. Varied depths break
    the strip into sub-0.05 mm^2 pieces the caller's micro-sliver cull
    removes cleanly. Inward = left of travel: the exterior is CCW and
    holes are CW, so the material side is the left side for both.
    """
    n = len(ring)
    if n < 3:
        return ring
    prev = np.roll(ring, 1, axis=0)
    nxt = np.roll(ring, -1, axis=0)
    v1 = ring - prev
    v2 = nxt - ring
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    seg = nxt - prev
    seglen = np.linalg.norm(seg, axis=1)
    collinear = (np.abs(cross) < 1e-9 * np.maximum(seglen, 1.0) ** 2) & (seglen > 1e-9)
    if not collinear.any():
        return ring
    left = np.column_stack([-seg[:, 1], seg[:, 0]]) / np.maximum(seglen, 1e-12)[:, None]
    # Deterministic per-index variation in [1, 2): reproducible meshes
    # for identical inputs (cache keys hash the result).
    amp = eps * (1.0 + ((np.arange(n) * 2654435761) % 97) / 97.0)
    out = ring.copy()
    out[collinear] += left[collinear] * amp[collinear, None]
    return out


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


def _ring_inward_offset(ring: np.ndarray, off: float) -> np.ndarray:
    """Each ring point moved ``off`` mm along its inward (left) normal."""
    seg = np.roll(ring, -1, axis=0) - np.roll(ring, 1, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    left = np.column_stack([-seg[:, 1], seg[:, 0]]) / np.maximum(seglen, 1e-12)[:, None]
    return ring + left * off


def _delaunay_cap(
    loops: list[np.ndarray], test_poly: sg.Polygon
) -> tuple[np.ndarray, np.ndarray]:
    """Flat cap triangulation that CONFORMS to a dense boundary.

    Earcut is the wrong tool once rings are densified: its fans create
    zero-area slivers on collinear boundary points, trimesh's processing
    deletes those, and the rim tears open (production case: 1225 open
    edges on a plain square). Delaunay over the exact loop vertices plus
    a coarse interior grid never produces degenerate boundary slivers --
    and, like the pillow top's collar rings, a dense COLLAR row ~1.2 mm
    inside the boundary locks every rim segment in as a Delaunay edge
    (without it, a chord skimming a nearly-straight boundary stretch has
    an empty circumcircle and the cap boundary can skip ring points --
    the interior grid keeps 3 mm clear of the boundary, so nothing else
    forbids such chords). The cap therefore welds to the walls by
    construction. Triangles are kept if their centroid is inside (or
    within 0.25 mm of) the polygon, which tolerates the locally
    self-touching loops of pinched tails; micro-slivers from the
    collinearity nudge are culled by area at the end.
    """
    from scipy.spatial import Delaunay

    from . import meshops

    pts = np.vstack(loops)
    collar = np.vstack([_ring_inward_offset(lp, 1.2) for lp in loops])
    cpts = shapely.points(collar)
    ok = shapely.contains(test_poly, cpts) & (
        shapely.distance(test_poly.boundary, cpts) > 0.5
    )
    collar = collar[ok]
    if len(collar):
        pts = np.vstack([pts, collar])
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
    cpts = shapely.points(cent)
    # Boundary-only triangles (every vertex on a loop) hug the rim; the
    # collinearity nudge (_nudge_collinear) leaves multi-level sliver
    # "flaps" between the convex-hull chords and the nudged boundary row,
    # ALL of whose vertices are loop points and whose centroids sit just
    # OUTSIDE the polygon. Any of them that survive make cap boundary
    # edges the walls don't have (open seams), so they are culled by a
    # strict inside test. Mixed triangles (touching collar/grid points)
    # keep the 0.25 mm slack that tolerates the locally self-touching
    # loops of pinched tails.
    nloop = sum(len(lp) for lp in loops)
    boundary_only = (tri.simplices < nloop).all(axis=1)
    inside = shapely.contains(test_poly, cpts)
    keep = np.where(
        boundary_only, inside, shapely.dwithin(test_poly, cpts, 0.25)
    )
    return allpts, meshops.ensure_up_normals(allpts, tri.simplices[keep])


def extrude_with_roundover(
    poly: sg.Polygon, thickness: float, roundover: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a polygon into a straight, sharp-edged slab.

    The slab is the honest pre-pillow core (board + foam, square arris).
    The wrapped-edge roundover is NOT modelled as fillet geometry any
    more: it is applied by the pillow stage as part of one continuous
    edge-roll + crown height field (``pillow_panel(edge_roll=...)``).
    The old discrete 3-ring fillet loft put a tangent break at the rim
    (a visibly hard top edge) and its corner insets were a chronic source
    of malformed geometry; a height field has neither problem. The
    ``roundover`` argument is kept for signature compatibility and is
    recorded by the caller as the part's edge_roll.

    Walls are quad strips over the densified rings; caps use the
    boundary-conforming Delaunay triangulation (earcut fans over
    densified collinear boundary points create zero-area slivers that
    mesh processing deletes, tearing the rim open).
    """
    import trimesh

    thickness = float(thickness)

    # Densify here too (the import path already does): direct callers may
    # pass polygons with bare corner vertices (e.g. shapely boxes), and
    # everything downstream depends on ~2 mm boundary density.
    rings = [_densify_ring(np.array(poly.exterior.coords[:-1], dtype=np.float64))]
    rings += [
        _densify_ring(np.array(h.coords[:-1], dtype=np.float64))
        for h in poly.interiors
    ]
    # Break exact collinearity so the cap/top Delaunay keeps every
    # boundary point (see _nudge_collinear); must happen before the
    # walls are built so walls, caps and the pillow top share vertices.
    rings = [_nudge_collinear(r) for r in rings]
    poly = sg.Polygon(rings[0], rings[1:])

    # Several wall rings between bottom and top: the pillow stage bows
    # the side walls outward (belly) and barrels them with the crown,
    # which needs mid-height vertices to act on. ~4 mm vertical spacing
    # (6 mm rows showed as flat bands in flat-shading model viewers).
    nz = max(3, int(math.ceil(thickness / 4.0)) + 1)
    zs = np.linspace(0.0, thickness, nz)

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for ring in rings:
        start = sum(len(v) for v in verts)
        n = len(ring)
        for z in zs:
            verts.append(np.column_stack([ring, np.full(n, z)]))
        idx = np.arange(n)
        nxt = (idx + 1) % n
        for k in range(nz - 1):
            a0, a1 = start + k * n + idx, start + k * n + nxt
            b0, b1 = start + (k + 1) * n + idx, start + (k + 1) * n + nxt
            # Material is left of travel, so outward is right: this winding
            # gives outward-facing wall normals for both shells and holes.
            faces.append(np.column_stack([a0, a1, b1]))
            faces.append(np.column_stack([a0, b1, b0]))

    v2d, f2d = _delaunay_cap(rings, poly)
    base = sum(len(v) for v in verts)
    verts.append(np.column_stack([v2d, np.zeros(len(v2d))]))
    faces.append(f2d[:, ::-1] + base)  # bottom cap faces down
    base = sum(len(v) for v in verts)
    verts.append(np.column_stack([v2d, np.full(len(v2d), thickness)]))
    faces.append(f2d + base)  # top cap faces up

    v = np.vstack(verts)
    f = np.vstack(faces)
    # Merge coincident seam vertices (cap rims == wall rings) so the
    # pillow engine sees one connected solid with a welded seam.
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    mesh.merge_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


# ---------------------------------------------------------------------------
# part naming from annotation text
# ---------------------------------------------------------------------------

_QTY_RE = re.compile(r"^QTY\.?\s*[x×]?\s*(\d+)", re.IGNORECASE)
# A panel key is a short standalone identifier: "H", "F2", "A-1"...
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]{0,3}$")


def _pool_to_name(pool: list[dict]) -> str | None:
    """Panel name from a label pool: shortest key-looking string is the
    panel key; a "QTY n" line becomes a multiplier suffix."""
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


def _name_hints(polys: list[sg.Polygon], labels: list[dict]) -> list[str | None]:
    """Per-part display names from annotation text.

    Two passes over ALL parts: labels INSIDE an outline name that part
    and are thereby claimed; only then do unnamed parts fall back to the
    nearest UNCLAIMED label cluster within one bounding-box diagonal
    (some drawings put the caption beside the part). Without the
    claiming, an unlabeled part sitting next to a labeled one stole its
    neighbour's caption and two parts showed the same name.
    """
    hints: list[str | None] = [None] * len(polys)
    if not labels:
        return hints
    labs = [lab for lab in labels if lab["text"]]
    claimed: set[int] = set()
    # Pass 1: labels contained in an outline.
    for i, poly in enumerate(polys):
        pool = []
        for j, lab in enumerate(labs):
            if poly.contains(sg.Point(lab["x"], lab["y"])):
                pool.append(lab)
                claimed.add(j)
        if pool:
            hints[i] = _pool_to_name(pool)
    # Pass 2: nearest unclaimed labels for still-unnamed parts.
    for i, poly in enumerate(polys):
        if hints[i] is not None:
            continue
        minx, miny, maxx, maxy = poly.bounds
        diag = math.hypot(maxx - minx, maxy - miny)
        pool = [
            lab
            for j, lab in enumerate(labs)
            if j not in claimed
            and poly.exterior.distance(sg.Point(lab["x"], lab["y"])) < diag
        ]
        if pool:
            hints[i] = _pool_to_name(pool)
    return hints


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
    hints = _name_hints(polys, labels)
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


def _convex_corners(poly: sg.Polygon, min_turn_deg: float = 35.0) -> list[list[float]]:
    """Convex outline corners as [x, y, bisector_x, bisector_y, turn_deg].

    Real vinyl forms a fold dart at every convex corner (see the corner
    photos in reference/NOTES.md); the pillow stage carves those darts
    along the inward bisector recorded here. Uses the simplified outline
    (before densification would drown corners in collinear points):
    consecutive near-collinear vertices are skipped via a coarse
    resimplification, and only convex turns count -- concave corners
    don't fold, the fabric bridges them.
    """
    out: list[list[float]] = []
    ext = orient(poly, sign=1.0).exterior.simplify(1.0)
    pts = np.array(ext.coords[:-1], dtype=np.float64)
    n = len(pts)
    if n < 3:
        return out
    for i in range(n):
        p = pts[i]
        v1 = p - pts[(i - 1) % n]
        v2 = pts[(i + 1) % n] - p
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        v1 /= l1
        v2 /= l2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = float(np.clip(v1 @ v2, -1.0, 1.0))
        turn = math.degrees(math.atan2(cross, dot))
        if turn < min_turn_deg:  # concave (negative) or too gentle
            continue
        # Inward bisector: mean of the two left normals (material is left
        # of travel on a CCW exterior).
        n1 = np.array([-v1[1], v1[0]])
        n2 = np.array([-v2[1], v2[0]])
        bis = n1 + n2
        bn = np.linalg.norm(bis)
        if bn < 1e-9:
            continue
        bis /= bn
        out.append([float(p[0]), float(p[1]), float(bis[0]), float(bis[1]), float(turn)])
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
    corners_per_part = []
    for poly in polys:
        from shapely.affinity import translate

        poly = translate(poly, xoff=-minx, yoff=-miny)
        v, f = extrude_with_roundover(poly, thickness, roundover)
        part_arrays.append((v, f))
        corners_per_part.append(_convex_corners(poly))

    parts = import_stl.describe_parts_from_arrays(
        part_arrays, hardware_threshold=0, name_hints=hints
    )
    # The roundover is applied by the pillow stage as a continuous
    # edge-roll height field; record the radius (and the fold-dart
    # corners) on every pillowable part. Parts were sorted by face count,
    # so match corners back to parts by vertex identity via bbox.
    by_key = {
        (round(v[:, 0].min(), 3), round(v[:, 1].min(), 3), len(f)): c
        for (v, f), c in zip(part_arrays, corners_per_part)
    }
    for part in parts:
        if part["classification"] == "pillow":
            part["edge_roll"] = float(roundover)
            key = (
                round(float(part["bbox_min"][0]), 3),
                round(float(part["bbox_min"][1]), 3),
                part["face_count"],
            )
            part["corners"] = by_key.get(key, [])
    return parts
