"""Phase 2 tests: SVG/DXF outline -> extruded solid -> pillow pipeline.

Covers spec acceptance test 3 (donut: fabric dips toward the hole rim).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engine import import_2d, pillow  # noqa: E402
from tests import fixtures_gen  # noqa: E402

OUTER = 300.0
HOLE_D = 100.0


def _donut_parts():
    svg = fixtures_gen.donut_svg(OUTER, HOLE_D)
    return import_2d.load_2d_as_parts(str(svg), scale=1.0, thickness=50.8, roundover=8.0)


def test_donut_svg_import():
    parts = _donut_parts()
    assert len(parts) == 1
    part = parts[0]
    assert part["classification"] == "pillow"
    assert part["has_flat_top"]
    v = part["vertices"]
    # ~300 x 300 x 50.8 with the hole punched through
    assert v[:, 0].max() - v[:, 0].min() == pytest.approx(OUTER, abs=1.0)
    assert v[:, 2].max() == pytest.approx(50.8, abs=1e-3)
    # No vertices deep inside the hole footprint
    center = np.array([OUTER / 2, OUTER / 2])
    rad = np.linalg.norm(v[:, :2] - center, axis=1)
    assert (rad > HOLE_D / 2 - 1.0).all(), "vertices found inside the hole"


def test_donut_dips_toward_hole_rim():
    """Spec test 3: height 5 mm from the rim < height 60 mm from the rim."""
    part = _donut_parts()[0]
    v, f = pillow.pillow_panel(part["vertices"], part["faces"])
    assert np.isfinite(v).all()

    center = np.array([OUTER / 2, OUTER / 2])
    rad = np.linalg.norm(v[:, :2] - center, axis=1)
    top = v[:, 2] > 40.0  # ignore bottom/lower wall verts

    r_rim = HOLE_D / 2
    near = top & (np.abs(rad - (r_rim + 5)) < 2.5)
    far = top & (np.abs(rad - (r_rim + 60)) < 2.5)
    assert near.any() and far.any()
    z_near = v[near, 2].max()
    z_far = v[far, 2].max()
    assert z_near + 3.0 < z_far, (
        f"surface must dip toward the hole rim (near={z_near:.1f}, far={z_far:.1f})"
    )


@pytest.mark.parametrize("res,grid", [(2.0, 6.0), (4.0, 10.0)])
def test_donut_hole_stays_open(res, grid):
    """Regression: triangle culling must use the RAW distance field.

    With the smoothed field, preview resolution (4 mm cells, sigma 5
    cells = 20 mm bleed) kept triangles spanning the 100 mm hole and the
    preview rendered with the hole covered.
    """
    part = _donut_parts()[0]
    v, f = pillow.pillow_panel(part["vertices"], part["faces"], res=res, grid_step=grid)
    center = np.array([OUTER / 2, OUTER / 2])
    tri_cent = v[f].mean(axis=1)
    rad = np.linalg.norm(tri_cent[:, :2] - center, axis=1)
    top = tri_cent[:, 2] > 45.0
    deep_inside = top & (rad < HOLE_D / 2 - 15.0)
    assert not deep_inside.any(), (
        f"{deep_inside.sum()} top faces span the hole at res={res}"
    )


def test_dxf_import():
    """Rectangle + circular hole via ezdxf round trip."""
    import ezdxf

    fixtures_gen.FIXTURES.mkdir(exist_ok=True)
    path = fixtures_gen.FIXTURES / "rect_hole.dxf"
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (400, 0), (400, 200), (0, 200)], format="xy", close=True
    )
    msp.add_circle((100, 100), 30)
    doc.saveas(path)

    parts = import_2d.load_2d_as_parts(str(path), scale=1.0, thickness=50.8, roundover=8.0)
    assert len(parts) == 1
    v = parts[0]["vertices"]
    assert v[:, 0].max() - v[:, 0].min() == pytest.approx(400, abs=1.0)
    rad = np.linalg.norm(v[:, :2] - np.array([100.0, 100.0]), axis=1)
    assert (rad > 29.0).all()


def test_dxf_text_tolerated_and_used_for_names():
    """Annotation text (panel key, dimensions, QTY) must never become
    geometry, and the panel-key text names the part."""
    import ezdxf

    fixtures_gen.FIXTURES.mkdir(exist_ok=True)
    path = fixtures_gen.FIXTURES / "annotated.dxf"
    doc = ezdxf.new()
    msp = doc.modelspace()
    # Two panels side by side, drawn in inches like the production files.
    msp.add_lwpolyline([(0, 0), (24, 0), (24, 24), (0, 24)], format="xy", close=True)
    msp.add_lwpolyline([(30, 0), (82, 0), (30, 8)], format="xy", close=True)
    # Annotation inside each outline + one label that must NOT match.
    msp.add_text("F", height=2, dxfattribs={"insert": (12, 14)})
    msp.add_text('24.00\"x24.00\"', height=1, dxfattribs={"insert": (12, 11)})
    msp.add_text("QTY 7", height=1, dxfattribs={"insert": (12, 9)})
    msp.add_text("H", height=1.2, dxfattribs={"insert": (40, 2)})
    msp.add_text('RIGHT TRIANGLE 52\" X 8\"', height=0.5, dxfattribs={"insert": (40, 1)})
    doc.saveas(path)

    parts = import_2d.load_2d_as_parts(str(path), scale=25.4, thickness=50.8, roundover=8.0)
    assert len(parts) == 2, "text must not create extra parts"
    names = {p.get("name_hint") for p in parts}
    assert names == {"Panel F ×7", "Panel H"}
    # Text never becomes geometry: bounds match the outlines only.
    for p in parts:
        assert p["bbox_max"][2] == pytest.approx(50.8, abs=1e-3)


def test_outline_preview_matches_import():
    """The preview must report the same shape the importer builds:
    outer/holes present, dimensions correct, name carried through."""
    svg = fixtures_gen.donut_svg(OUTER, HOLE_D)
    preview = import_2d.outline_preview(str(svg), scale=1.0)
    assert len(preview) == 1
    o = preview[0]
    assert o["width"] == pytest.approx(OUTER, abs=1.0)
    assert o["height"] == pytest.approx(OUTER, abs=1.0)
    assert len(o["holes"]) == 1
    # Exterior/holes are closed rings of [x, y] pairs.
    assert o["exterior"][0] == o["exterior"][-1]
    assert all(len(p) == 2 for p in o["exterior"])

    # Same part count as the real import path.
    parts = import_2d.load_2d_as_parts(str(svg), scale=1.0, thickness=50.8, roundover=8.0)
    assert len(parts) == len(preview)


def test_outline_preview_dxf_names_and_scale():
    """DXF preview carries the panel-key name and scales coordinates."""
    import ezdxf

    fixtures_gen.FIXTURES.mkdir(exist_ok=True)
    path = fixtures_gen.FIXTURES / "preview_named.dxf"
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (52, 0), (52, 8)], format="xy", close=True)
    msp.add_text("J", height=1.2, dxfattribs={"insert": (34, 3)})
    doc.saveas(path)

    preview = import_2d.outline_preview(str(path), scale=25.4)
    assert len(preview) == 1
    assert preview[0]["name"] == "Panel J"
    assert preview[0]["width"] == pytest.approx(52 * 25.4, abs=1.0)


def test_two_disjoint_outlines_two_parts():
    """Multiple disjoint outlines become multiple parts."""
    fixtures_gen.FIXTURES.mkdir(exist_ok=True)
    path = fixtures_gen.FIXTURES / "two_squares.svg"
    path.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 200">
        <rect x="0" y="0" width="200" height="200"/>
        <rect x="300" y="0" width="200" height="200"/>
        </svg>"""
    )
    parts = import_2d.load_2d_as_parts(str(path), scale=1.0, thickness=50.8, roundover=8.0)
    assert len(parts) == 2
    assert all(p["classification"] == "pillow" for p in parts)


def test_extrusion_watertight_enough():
    """The extruded solid must have a flat bottom at 0, top at thickness,
    and pillow cleanly (seam welded, no NaNs)."""
    part = _donut_parts()[0]
    pv, pf = part["vertices"], part["faces"]
    assert pv[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    v, f = pillow.pillow_panel(pv, pf)
    assert np.isfinite(v).all()
    assert v[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    # Crown actually applied
    assert v[:, 2].max() > 50.8 + 10.0
