"""Render-software readiness of exported files.

The exported GLB must import correctly into rendering software with NO
manual fixes: glTF-standard orientation/units, UVs present, smooth
finite normals, no degenerate faces, consistent winding.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import fixtures_gen  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.main as main

    monkeypatch.setattr(main, "PROJECTS_DIR", tmp_path / "projects")
    main.PROJECTS_DIR.mkdir()
    return TestClient(main.app)


def _wait_job(client, job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "done":
            return job
        if job["status"] == "error":
            raise AssertionError(f"job failed: {job['error']}")
        time.sleep(0.2)
    raise AssertionError("job timed out")


def _make_36x12_dxf():
    import ezdxf

    fixtures_gen.FIXTURES.mkdir(exist_ok=True)
    path = fixtures_gen.FIXTURES / "export_36x12.dxf"
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (36, 0), (36, 12), (0, 12)], format="xy", close=True)
    doc.saveas(path)
    return path


def test_render_glb_is_render_ready(client):
    dxf = _make_36x12_dxf()
    with dxf.open("rb") as fh:
        r = client.post(
            "/api/projects",
            files={"file": ("panel.dxf", fh, "application/dxf")},
            data={"options": '{"scale": 25.4, "thickness": 38.1, "roundover": 12.7}'},
        )
    job = _wait_job(client, r.json()["job_id"])
    pid = job["result"]["project"]["id"]

    r = client.post(f"/api/projects/{pid}/export", params={"fmt": "glb", "res": 2.0})
    job = _wait_job(client, r.json()["job_id"])
    glb = client.get(job["result"]["url"]).content
    assert glb[:4] == b"glTF"

    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    meshes = list(scene.geometry.values())
    assert len(meshes) == 1
    m = meshes[0]

    # glTF convention: +Y up, METERS. The 36x12in panel is ~0.914 x 0.305 m
    # in plan (X/Z after reorientation) and ~0.07 m tall in Y.
    ext = m.bounding_box.extents  # trimesh loads glTF as-is (Y up)
    dims = sorted(ext)
    assert dims[2] == pytest.approx(0.9144, rel=0.02), "36in length in meters"
    assert dims[1] == pytest.approx(0.3048, rel=0.05), "12in width in meters"
    assert 0.045 < dims[0] < 0.085, "height (thickness+crown) in meters, Y-up"

    # UVs present and normalized.
    assert m.visual.uv is not None and len(m.visual.uv) == len(m.vertices)
    assert m.visual.uv.min() >= -1e-6 and m.visual.uv.max() <= 1 + 1e-6

    # Smooth finite normals, unit length.
    n = np.asarray(m.vertex_normals)
    assert np.isfinite(n).all()
    ln = np.linalg.norm(n, axis=1)
    assert np.all(ln > 0.9) and np.all(ln < 1.1)

    # No degenerate faces; consistent winding.
    assert float(m.area_faces.min()) > 0.0
    assert m.is_winding_consistent


def test_stl_export_stays_mm_zup(client):
    """STL keeps the CAD convention: millimetres, Z-up."""
    dxf = _make_36x12_dxf()
    with dxf.open("rb") as fh:
        r = client.post(
            "/api/projects",
            files={"file": ("panel.dxf", fh, "application/dxf")},
            data={"options": '{"scale": 25.4, "thickness": 38.1, "roundover": 12.7}'},
        )
    job = _wait_job(client, r.json()["job_id"])
    pid = job["result"]["project"]["id"]
    r = client.post(f"/api/projects/{pid}/export", params={"fmt": "stl", "res": 2.0})
    job = _wait_job(client, r.json()["job_id"])
    m = trimesh.load(io.BytesIO(client.get(job["result"]["url"]).content), file_type="stl")
    assert m.vertices[:, 0].max() - m.vertices[:, 0].min() == pytest.approx(914.4, rel=0.02)
    assert m.vertices[:, 2].max() == pytest.approx(38.1 + 32.0, abs=3.0)