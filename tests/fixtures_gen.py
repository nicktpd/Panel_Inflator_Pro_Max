"""Procedural test fixtures (binaries are generated, never committed).

Every helper writes into tests/fixtures/ and returns the path. Meshes are
built with trimesh primitives; boxes are subdivided so the flat tops have
realistic vertex/face density like real CAD exports.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

FIXTURES = Path(__file__).parent / "fixtures"


def _box(extents, translate=(0, 0, 0), subdivisions=0) -> trimesh.Trimesh:
    """Axis-aligned box with min corner at ``translate``."""
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(np.asarray(extents) / 2.0 + np.asarray(translate, dtype=float))
    for _ in range(subdivisions):
        m = m.subdivide()
    return m


def square_panel_stl(subdivisions: int = 6) -> Path:
    """400 x 400 x 50 mm flat panel, ~49k faces (12 * 4^subdivisions)."""
    FIXTURES.mkdir(exist_ok=True)
    path = FIXTURES / f"square_panel_s{subdivisions}.stl"
    if not path.exists():
        _box([400, 400, 50], subdivisions=subdivisions).export(path)
    return path


def skinny_rect_stl(subdivisions: int = 6) -> Path:
    """600 x 80 x 50 mm skinny panel: narrow parts must loft less."""
    FIXTURES.mkdir(exist_ok=True)
    path = FIXTURES / f"skinny_rect_s{subdivisions}.stl"
    if not path.exists():
        _box([600, 80, 50], subdivisions=subdivisions).export(path)
    return path


def multipart_stl() -> Path:
    """Two >=10k-face boxes sharing one coincident corner vertex + a
    12-face sliver. Edge-adjacency splitting must find exactly 3 parts."""
    FIXTURES.mkdir(exist_ok=True)
    path = FIXTURES / "multipart.stl"
    if not path.exists():
        a = _box([200, 200, 50], translate=(0, 0, 0), subdivisions=5)  # 12288 faces
        # b's min corner exactly at a's max corner in xy, same z base:
        # they touch at the single vertex (200, 200, 0) and (200,200,50).
        b = _box([200, 200, 50], translate=(200, 200, 0), subdivisions=5)
        sliver = _box([4, 4, 4], translate=(320, 20, 0), subdivisions=0)  # 12 faces
        trimesh.util.concatenate([a, b, sliver]).export(path)
    return path


def big_panel_arrays(subdivisions: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """~196k-face panel for the memory guard test (returned, not saved)."""
    m = _box([500, 400, 50], subdivisions=subdivisions)
    return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.int32)


def donut_svg(outer: float = 300.0, hole_d: float = 100.0) -> Path:
    """Square with a centered circular hole, as an SVG (Phase 2 path)."""
    FIXTURES.mkdir(exist_ok=True)
    path = FIXTURES / "donut.svg"
    r = hole_d / 2.0
    cx = cy = outer / 2.0
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{outer}mm" height="{outer}mm"
     viewBox="0 0 {outer} {outer}">
  <path d="M 0 0 L {outer} 0 L {outer} {outer} L 0 {outer} Z" fill="black"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="white"/>
</svg>
"""
    path.write_text(svg)
    return path
