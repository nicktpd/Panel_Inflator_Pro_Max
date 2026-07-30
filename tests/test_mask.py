"""Phase 3 tests: loft-multiplier masks (selective loft painting)."""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engine import pillow  # noqa: E402
from tests import fixtures_gen  # noqa: E402


def _square_part():
    mesh = trimesh.load(str(fixtures_gen.square_panel_stl(subdivisions=5)), force="mesh")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _mask_for_bbox(xmin, ymin, xmax, ymax, res=4.0):
    nx = int(np.ceil((xmax - xmin) / res)) + 3
    ny = int(np.ceil((ymax - ymin) / res)) + 3
    return np.ones((ny, nx), dtype=np.float32), xmin, ymin, res


def test_mask_suppresses_loft_locally():
    """Left half painted to 0.2 -> left crown much lower than right."""
    pv, pf = _square_part()
    grid, xmin, ymin, res = _mask_for_bbox(0, 0, 400, 400)
    cols = np.arange(grid.shape[1])
    world_x = xmin + (cols - 1) * res
    grid[:, world_x < 200] = 0.2
    mask = {"grid": grid, "xmin": xmin, "ymin": ymin, "res": res}

    v, f = pillow.pillow_panel(pv, pf, mask=mask)
    assert np.isfinite(v).all()

    left = v[(v[:, 0] > 60) & (v[:, 0] < 140) & (np.abs(v[:, 1] - 200) < 60)]
    right = v[(v[:, 0] > 260) & (v[:, 0] < 340) & (np.abs(v[:, 1] - 200) < 60)]
    crown_left = left[:, 2].max() - 50
    crown_right = right[:, 2].max() - 50
    assert crown_right > 20, "unpainted side should loft normally"
    assert crown_left < crown_right * 0.55, (
        f"painted side must loft much less (left={crown_left:.1f}, right={crown_right:.1f})"
    )


def test_mask_resolution_independent():
    """A mask painted at preview res must land in the same world place at
    export res (world-anchored resampling, not index stretching)."""
    pv, pf = _square_part()
    grid, xmin, ymin, res = _mask_for_bbox(0, 0, 400, 400)
    cols = np.arange(grid.shape[1])
    world_x = xmin + (cols - 1) * res
    grid[:, world_x < 200] = 0.0
    mask = {"grid": grid, "xmin": xmin, "ymin": ymin, "res": res}

    for out_res, gs in [(4.0, 10.0), (2.0, 6.0)]:
        v, _ = pillow.pillow_panel(pv, pf, mask=mask, res=out_res, grid_step=gs)
        left = v[(v[:, 0] > 60) & (v[:, 0] < 140)]
        # Fully suppressed + post-mask blur: essentially flat far from the split.
        assert left[:, 2].max() < 50 + 6.0, f"res={out_res}: masked side should stay flat"


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


def test_mask_api_round_trip(client):
    stl = fixtures_gen.square_panel_stl(subdivisions=5)
    with stl.open("rb") as fh:
        r = client.post("/api/projects", files={"file": ("square.stl", fh, "model/stl")})
    job = _wait_job(client, r.json()["job_id"])
    pid = job["result"]["project"]["id"]
    part_id = job["result"]["project"]["parts"][0]["id"]

    # Blank mask meta first.
    meta = client.get(f"/api/projects/{pid}/parts/{part_id}/mask").json()
    assert meta["data_b64"] is None
    w, h = meta["w"], meta["h"]

    # Paint: kill the loft on the left half, PUT, verify GET round trip.
    grid = np.ones((h, w), dtype="<f4")
    cols = np.arange(w)
    world_x = meta["xmin"] + (cols - 1) * meta["res"]
    grid[:, world_x < 200] = 0.0
    payload = dict(meta, data_b64=base64.b64encode(grid.tobytes()).decode())
    payload.pop("data_b64", None)
    payload["data_b64"] = base64.b64encode(grid.tobytes()).decode()
    r = client.put(f"/api/projects/{pid}/parts/{part_id}/mask", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["mask_version"] == 1

    back = client.get(f"/api/projects/{pid}/parts/{part_id}/mask").json()
    got = np.frombuffer(base64.b64decode(back["data_b64"]), dtype="<f4").reshape(h, w)
    assert np.allclose(got, grid)

    # Export honors the mask: left flat-ish, right lofted.
    r = client.post(f"/api/projects/{pid}/export", params={"fmt": "stl", "res": 2.0})
    job = _wait_job(client, r.json()["job_id"])
    import io

    dl = client.get(job["result"]["url"])
    out = trimesh.load(io.BytesIO(dl.content), file_type="stl")
    v = np.asarray(out.vertices)
    left = v[(v[:, 0] > 60) & (v[:, 0] < 140)]
    right = v[(v[:, 0] > 260) & (v[:, 0] < 340)]
    assert right[:, 2].max() > 50 + 20
    assert left[:, 2].max() < 50 + 6

    # DELETE clears back to normal loft.
    r = client.delete(f"/api/projects/{pid}/parts/{part_id}/mask")
    assert r.json()["mask_version"] == 2
    meta = client.get(f"/api/projects/{pid}/parts/{part_id}/mask").json()
    assert meta["data_b64"] is None
