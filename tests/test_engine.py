"""Engine + API acceptance tests (spec section 7).

Run with:  .venv/bin/python -m pytest tests/ -v
Fixtures are generated procedurally into tests/fixtures/ (gitignored).
"""
from __future__ import annotations

import io
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engine import import_stl, meshops, pillow  # noqa: E402
from tests import fixtures_gen  # noqa: E402

CROWN, DREF, EXP = 32.0, 110.0, 0.55


def _pillow_file(path, **kw):
    mesh = trimesh.load(str(path), force="mesh")
    pv = np.asarray(mesh.vertices)
    pf = np.asarray(mesh.faces)
    v, f = pillow.pillow_panel(pv, pf, **kw)
    return pv, pf, v, f


# ---------------------------------------------------------------------------
# 1. Square panel 400x400x50: center ~= 50 + CROWN, corners low, flat bottom
# ---------------------------------------------------------------------------


def test_square_panel():
    path = fixtures_gen.square_panel_stl()
    pv, pf, v, f = _pillow_file(path)

    assert np.isfinite(v).all(), "mesh contains NaNs/inf"

    # Bottom face stays exactly at zmin (it mounts to a wall).
    assert v[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    n_bottom = int((v[:, 2] < 1e-6).sum())
    assert n_bottom > 100, "bottom face vertices disappeared"

    # Center lofts to thickness + CROWN (dist=200 >> DREF saturates).
    center = v[(np.abs(v[:, 0] - 200) < 10) & (np.abs(v[:, 1] - 200) < 10)]
    assert len(center) > 0
    assert center[:, 2].max() == pytest.approx(50 + CROWN, abs=2.0)

    # Corners stay pinned low.
    corner = v[(v[:, 0] < 15) & (v[:, 1] < 15) & (v[:, 2] > 25)]
    assert corner[:, 2].max() < 50 + 8.0, "corners should stay near base thickness"

    # Face count must not balloon (no naive subdivision).
    assert len(f) < len(pf) * 1.30


# ---------------------------------------------------------------------------
# 2. Skinny rectangle 600x80x50: narrow parts loft less (saturation law)
# ---------------------------------------------------------------------------


def test_skinny_rectangle_lofts_less():
    path = fixtures_gen.skinny_rect_stl()
    _, _, v, f = _pillow_file(path)

    max_crown = v[:, 2].max() - 50.0
    expected = CROWN * (40.0 / DREF) ** EXP  # half-width 40 mm
    # Gaussian smoothing of the distance field rounds the ridge peak a
    # little, so allow a few mm below the closed-form value.
    assert max_crown == pytest.approx(expected, abs=4.0)
    assert max_crown < CROWN * 0.75, "narrow panel must not reach full crown"


# ---------------------------------------------------------------------------
# 4. Multi-part STL: edge-adjacency split, sliver passthrough
# ---------------------------------------------------------------------------


def test_multipart_split_and_classification():
    path = fixtures_gen.multipart_stl()
    parts = import_stl.load_split_classify(str(path))

    assert len(parts) == 3, (
        "edge-adjacency split must find 3 components even though the two "
        "boxes share a coincident corner vertex"
    )
    classes = [p["classification"] for p in parts]
    assert classes.count("pillow") == 2
    assert classes.count("passthrough") == 1

    sliver = parts[-1]
    assert sliver["face_count"] == 12
    # Pillow the two panels independently; sliver passes through untouched.
    for p in parts[:2]:
        v, f = pillow.pillow_panel(p["vertices"], p["faces"])
        assert np.isfinite(v).all()
        top_gain = v[:, 2].max() - p["vertices"][:, 2].max()
        assert top_gain > 10.0, "panels should visibly loft"


# ---------------------------------------------------------------------------
# 5. Memory guard: 150k+ face panel stays under 1.5 GB RSS
# ---------------------------------------------------------------------------


def test_memory_guard():
    code = textwrap.dedent(
        """
        import resource, sys
        sys.path.insert(0, %r)
        from tests import fixtures_gen
        from backend.engine import pillow
        pv, pf = fixtures_gen.big_panel_arrays()   # ~196k faces
        v, f = pillow.pillow_panel(pv, pf, res=2.0)
        assert len(f) > 100
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("PEAK_KB", peak_kb)
        assert peak_kb < 1.5 * 1024 * 1024, f"peak RSS {peak_kb/1048576:.2f} GB"
        """
        % str(Path(__file__).resolve().parent.parent)
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 6. API smoke test: import -> patch -> preview -> export round trip
# ---------------------------------------------------------------------------


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


def test_api_round_trip(client):
    # subdivisions=5 -> 12288 faces: above the 10k hardware threshold so
    # the part classifies as pillowable, while staying fast to process.
    stl = fixtures_gen.square_panel_stl(subdivisions=5)
    with stl.open("rb") as fh:
        r = client.post(
            "/api/projects", files={"file": ("square.stl", fh, "model/stl")}
        )
    assert r.status_code == 200, r.text
    job = _wait_job(client, r.json()["job_id"])
    project = job["result"]["project"]
    pid = project["id"]
    assert len(project["parts"]) == 1
    part_id = project["parts"][0]["id"]

    # Patch per-part params (lower crown on this one part).
    r = client.patch(
        f"/api/projects/{pid}/parts/{part_id}",
        json={"params": {"crown": 15.0, "dref": 110.0, "exp": 0.55, "sigma": 5.0, "w_exp": 1.5}},
    )
    assert r.status_code == 200
    assert r.json()["parts"][0]["params"]["crown"] == 15.0

    # Preview -> GLB served.
    r = client.post(f"/api/projects/{pid}/preview")
    job = _wait_job(client, r.json()["job_id"])
    glb = client.get(job["result"]["url"])
    assert glb.status_code == 200
    assert glb.content[:4] == b"glTF"

    # Export -> loadable binary STL with the patched (15 mm) crown.
    r = client.post(f"/api/projects/{pid}/export", params={"fmt": "stl", "res": 2.0})
    job = _wait_job(client, r.json()["job_id"])
    dl = client.get(job["result"]["url"])
    assert dl.status_code == 200
    out = trimesh.load(io.BytesIO(dl.content), file_type="stl")
    assert len(out.faces) > 100
    assert out.vertices[:, 2].max() == pytest.approx(50 + 15.0, abs=2.0)
