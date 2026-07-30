"""Panel Inflator Pro Max — FastAPI backend.

Serves the static frontend and a small JSON API. Long operations (import,
preview, export) run in worker threads and are polled through
``GET /api/jobs/{job_id}`` so the browser never hangs on a slow request.

Run with:  python -m backend.main   (the run.bat / run.sh daily drivers)
Binds to 127.0.0.1 only — this is a local tool, never expose it.
"""
from __future__ import annotations

import base64
import json
import shutil
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import import_stl, preview
from .models import (
    Import2DOptions,
    PartInfo,
    PartPatch,
    PillowParams,
    Project,
    ProjectPatch,
)

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"
FRONTEND_DIR = ROOT / "frontend"

HOST = "127.0.0.1"
PORT = 8177

app = FastAPI(title="Panel Inflator Pro Max", docs_url="/api/docs")

# --------------------------------------------------------------------------
# Job registry: tiny in-process background task system with polling.
# --------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS = 50


def _job_create(kind: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "progress": 0.0,
            "message": "starting",
            "result": None,
            "error": None,
            "created": time.time(),
        }
        # Trim old finished jobs so the dict never grows unbounded.
        if len(_jobs) > _MAX_JOBS:
            done = sorted(
                (j for j in _jobs.values() if j["status"] != "running"),
                key=lambda j: j["created"],
            )
            for j in done[: len(_jobs) - _MAX_JOBS]:
                _jobs.pop(j["id"], None)
    return job_id


def _job_update(job_id: str, progress: float | None = None, message: str | None = None) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        if progress is not None:
            job["progress"] = round(float(progress), 3)
        if message is not None:
            job["message"] = message


def _job_finish(job_id: str, result: dict | None = None, error: str | None = None) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "error" if error else "done"
        job["progress"] = 1.0
        job["result"] = result
        job["error"] = error


def _run_job(kind: str, fn) -> str:
    """Start fn(job_id) in a daemon thread; return the job id."""
    job_id = _job_create(kind)

    def worker():
        try:
            result = fn(job_id)
            _job_finish(job_id, result=result)
        except Exception as exc:  # surface the message to the UI
            import traceback

            traceback.print_exc()
            _job_finish(job_id, error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=worker, daemon=True).start()
    return job_id


_VERSION_CACHE: dict | None = None


def _git(args: list[str]) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@app.get("/api/version")
def get_version():
    """Report the running build so the user can confirm they're current.

    Reads the checked-out git commit (short hash, date, subject). A ZIP
    install with no git returns install='zip' and hash='unknown', which is
    itself the answer to 'why isn't my app updating' — ZIP installs can't
    self-update. Cached after the first call; commit only changes on a
    pull, which restarts the server anyway.
    """
    global _VERSION_CACHE
    if _VERSION_CACHE is not None:
        return _VERSION_CACHE
    is_git = (ROOT / ".git").exists() and _git(["rev-parse", "HEAD"]) is not None
    if is_git:
        info = {
            "install": "git",
            "hash": _git(["rev-parse", "--short", "HEAD"]) or "unknown",
            "date": _git(["log", "-1", "--format=%cd", "--date=short"]) or "",
            "subject": _git(["log", "-1", "--format=%s"]) or "",
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "",
        }
    else:
        info = {"install": "zip", "hash": "unknown", "date": "", "subject": "", "branch": ""}
    _VERSION_CACHE = info
    return info


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return dict(job)


# --------------------------------------------------------------------------
# Project persistence
# --------------------------------------------------------------------------


def _project_dir(project_id: str) -> Path:
    d = PROJECTS_DIR / project_id
    if not d.exists():
        raise HTTPException(404, "project not found")
    return d


def _load_project(project_id: str) -> Project:
    path = _project_dir(project_id) / "project.json"
    if not path.exists():
        raise HTTPException(404, "project not found")
    return Project.model_validate_json(path.read_text())


def _save_project(project: Project) -> None:
    d = PROJECTS_DIR / project.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "project.json").write_text(project.model_dump_json(indent=2))


def _cache_dir(project_id: str) -> Path:
    d = PROJECTS_DIR / project_id / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_part_sources(project: Project) -> Path:
    """Guarantee per-part source npz files exist (cache may have been wiped).

    The cache directory is excluded from cloud sync and git; when it is
    missing we re-split the persisted source file, which is always synced.
    """
    cache = _cache_dir(project.id)
    missing = [p.id for p in project.parts if not preview.part_source_path(cache, p.id).exists()]
    if not missing:
        return cache
    source = PROJECTS_DIR / project.id / project.source_file
    if project.source_type == "stl":
        parts = import_stl.load_split_classify(str(source))
    else:
        from .engine import import_2d

        opts = project.import_2d or Import2DOptions()
        parts = import_2d.load_2d_as_parts(
            str(source), scale=opts.scale, thickness=opts.thickness, roundover=opts.roundover
        )
    if len(parts) != len(project.parts):
        raise RuntimeError("source file changed since import; re-import the project")
    for info, raw in zip(sorted(project.parts, key=lambda p: p.id), parts):
        preview.save_part_source(cache, info.id, raw["vertices"], raw["faces"])
    return cache


def _effective_params(project: Project, part: PartInfo) -> dict:
    p = part.params if part.params is not None else project.global_params
    return p.model_dump()


def _load_mask(project: Project, part: PartInfo) -> dict | None:
    """Load a painted loft mask as a world-anchored dict for the engine."""
    if not part.mask_file:
        return None
    mpath = PROJECTS_DIR / project.id / "masks" / part.mask_file
    if not mpath.exists():
        return None
    with np.load(mpath) as z:
        return {
            "grid": z["mask"].astype(np.float64),
            "xmin": float(z["xmin"]),
            "ymin": float(z["ymin"]),
            "res": float(z["res"]),
        }


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------


def _part_display_name(raw: dict, index: int) -> str:
    if raw["classification"] == "pillow":
        return f"Panel {index + 1}"
    return f"Hardware {index + 1}"


def _do_import(job_id: str, project_id: str, filename: str, options_2d: Import2DOptions | None):
    pdir = PROJECTS_DIR / project_id
    source = pdir / filename
    suffix = source.suffix.lower()
    _job_update(job_id, 0.05, "reading file")

    if suffix == ".stl":
        source_type = "stl"
        parts_raw = import_stl.load_split_classify(str(source))
    elif suffix in (".svg", ".dxf"):
        source_type = "2d"
        from .engine import import_2d

        opts = options_2d or Import2DOptions()
        parts_raw = import_2d.load_2d_as_parts(
            str(source), scale=opts.scale, thickness=opts.thickness, roundover=opts.roundover
        )
    else:
        raise ValueError(f"unsupported file type: {suffix}")

    _job_update(job_id, 0.5, f"classifying {len(parts_raw)} parts")

    cache = pdir / "cache"
    parts_info: list[PartInfo] = []
    panel_i = 0
    hw_i = 0
    for i, raw in enumerate(parts_raw):
        if raw.get("name_hint"):
            name = raw["name_hint"]  # e.g. "Panel H" from DXF annotation
        elif raw["classification"] == "pillow":
            name = f"Panel {panel_i + 1}"
            panel_i += 1
        else:
            name = f"Hardware {hw_i + 1}"
            hw_i += 1
        parts_info.append(
            PartInfo(
                id=i,
                name=name,
                face_count=raw["face_count"],
                classification=raw["classification"],
                auto_classification=raw["classification"],
                bbox_min=raw["bbox_min"],
                bbox_max=raw["bbox_max"],
            )
        )
        preview.save_part_source(cache, i, raw["vertices"], raw["faces"])
        _job_update(job_id, 0.5 + 0.3 * (i + 1) / len(parts_raw), f"caching part {i + 1}")

    project = Project(
        id=project_id,
        name=Path(filename).stem,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_file=filename,
        source_type=source_type,
        import_2d=options_2d,
        parts=parts_info,
    )
    _save_project(project)

    # Build the "before" GLB once: the flat original, same node names as
    # the pillowed preview so viewport selection works in both states.
    _job_update(job_id, 0.85, "building original preview")
    named = []
    for info in parts_info:
        pv, pf = preview.load_part_source(cache, info.id)
        named.append((f"part_{info.id}", pv, pf))
    (cache / "original.glb").write_bytes(preview.build_glb(named))
    del named

    return {"project": json.loads(project.model_dump_json())}


@app.post("/api/inspect2d")
async def inspect_2d(file: UploadFile = File(...)):
    """Extract the outline(s) from an uploaded SVG/DXF for the import
    preview, WITHOUT creating a project. Fast, synchronous 2D parse.

    Coordinates are returned in the drawing's own units (scale=1); the
    frontend applies the chosen unit factor for display, so changing the
    unit dropdown never needs a re-parse.
    """
    import tempfile

    from .engine import import_2d

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".svg", ".dxf"):
        raise HTTPException(400, "outline preview is only for SVG/DXF files")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        outlines = import_2d.outline_preview(tmp_path, scale=1.0)
    except Exception as exc:
        raise HTTPException(422, f"could not read outline: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return {"outlines": outlines, "source_type": suffix.lstrip(".")}


@app.post("/api/projects")
async def create_project(
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
):
    """Create a project from an uploaded STL/SVG/DXF. Returns a job id."""
    filename = Path(file.filename or "upload.stl").name
    suffix = Path(filename).suffix.lower()
    if suffix not in (".stl", ".svg", ".dxf"):
        raise HTTPException(400, f"unsupported file type: {suffix or '(none)'}")

    options_2d: Import2DOptions | None = None
    if options:
        options_2d = Import2DOptions.model_validate_json(options)

    project_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    pdir = PROJECTS_DIR / project_id
    pdir.mkdir(parents=True, exist_ok=True)
    dest = pdir / filename
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)

    job_id = _run_job(
        "import", lambda jid: _do_import(jid, project_id, filename, options_2d)
    )
    return {"job_id": job_id, "project_id": project_id}


# --------------------------------------------------------------------------
# Project CRUD
# --------------------------------------------------------------------------


@app.get("/api/projects")
def list_projects():
    out = []
    if PROJECTS_DIR.exists():
        for d in sorted(PROJECTS_DIR.iterdir(), reverse=True):
            pj = d / "project.json"
            if pj.exists():
                try:
                    p = Project.model_validate_json(pj.read_text())
                except Exception:
                    continue
                out.append(
                    {
                        "id": p.id,
                        "name": p.name,
                        "created": p.created,
                        "source_type": p.source_type,
                        "n_parts": len(p.parts),
                    }
                )
    return out


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return json.loads(_load_project(project_id).model_dump_json())


@app.patch("/api/projects/{project_id}")
def patch_project(project_id: str, patch: ProjectPatch):
    project = _load_project(project_id)
    if patch.name is not None:
        project.name = patch.name
    if patch.units_display is not None:
        project.units_display = patch.units_display
    if patch.global_params is not None:
        project.global_params = patch.global_params
    _save_project(project)
    return json.loads(project.model_dump_json())


@app.patch("/api/projects/{project_id}/parts/{part_id}")
def patch_part(project_id: str, part_id: int, patch: PartPatch):
    project = _load_project(project_id)
    part = next((p for p in project.parts if p.id == part_id), None)
    if part is None:
        raise HTTPException(404, "part not found")
    if patch.classification is not None:
        part.classification = patch.classification
    if patch.reset_params:
        part.params = None
    elif patch.params is not None:
        part.params = patch.params
    _save_project(project)
    return json.loads(project.model_dump_json())


# --------------------------------------------------------------------------
# Preview + export
# --------------------------------------------------------------------------


def _assemble(
    job_id: str,
    project: Project,
    res: float,
    grid_step: float,
    progress_lo: float,
    progress_hi: float,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Compute (cached) pillowed geometry for every part of a project."""
    cache = _ensure_part_sources(project)
    named: list[tuple[str, np.ndarray, np.ndarray]] = []
    n = max(len(project.parts), 1)
    for k, info in enumerate(project.parts):
        pv, pf = preview.load_part_source(cache, info.id)
        if info.classification == "pillow":
            _job_update(
                job_id,
                progress_lo + (progress_hi - progress_lo) * k / n,
                f"pillowing {info.name}",
            )
            mask = _load_mask(project, info)
            v, f = preview.compute_part_cached(
                cache,
                info.id,
                pv,
                pf,
                _effective_params(project, info),
                res,
                grid_step,
                mask=mask,
                mask_version=info.mask_version,
            )
        else:
            v, f = pv, pf
        named.append((f"part_{info.id}", v, f))
    return named


def _do_preview(job_id: str, project_id: str):
    project = _load_project(project_id)
    named = _assemble(job_id, project, preview.PREVIEW_RES, preview.PREVIEW_GRID, 0.0, 0.9)
    _job_update(job_id, 0.9, "building GLB")
    cache = _cache_dir(project_id)
    data = preview.build_glb(named)
    (cache / "preview.glb").write_bytes(data)
    preview.prune_part_cache(cache)
    return {"url": f"/api/projects/{project_id}/files/preview.glb?v={uuid.uuid4().hex[:8]}"}


@app.post("/api/projects/{project_id}/preview")
def start_preview(project_id: str):
    _load_project(project_id)  # 404 early
    return {"job_id": _run_job("preview", lambda jid: _do_preview(jid, project_id))}


def _do_export(job_id: str, project_id: str, fmt: str, res: float):
    project = _load_project(project_id)
    grid_step = preview.EXPORT_GRID * (res / preview.EXPORT_RES)
    named = _assemble(job_id, project, res, grid_step, 0.0, 0.85)
    _job_update(job_id, 0.85, f"writing {fmt.upper()}")
    cache = _cache_dir(project_id)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project.name) or "export"
    fname = f"{safe_name}-pillowed-{stamp}.{fmt}"
    if fmt == "stl":
        data = preview.build_stl([(v, f) for _, v, f in named])
    else:
        data = preview.build_glb(named)
    (cache / fname).write_bytes(data)
    preview.prune_part_cache(cache)
    return {"url": f"/api/projects/{project_id}/download/{fname}", "filename": fname}


@app.post("/api/projects/{project_id}/export")
def start_export(project_id: str, fmt: str = "stl", res: float = 2.0):
    if fmt not in ("stl", "glb"):
        raise HTTPException(400, "fmt must be stl or glb")
    if not (0.5 <= res <= 10.0):
        raise HTTPException(400, "res must be between 0.5 and 10 mm")
    _load_project(project_id)
    return {"job_id": _run_job("export", lambda jid: _do_export(jid, project_id, fmt, res))}


@app.get("/api/projects/{project_id}/files/{name}")
def get_cache_file(project_id: str, name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad filename")
    path = _project_dir(project_id) / "cache" / name
    if not path.exists():
        raise HTTPException(404, "file not found (run a preview first)")
    media = "model/gltf-binary" if name.endswith(".glb") else "application/octet-stream"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "no-store"})


@app.get("/api/projects/{project_id}/download/{name}")
def download_file(project_id: str, name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad filename")
    path = _project_dir(project_id) / "cache" / name
    if not path.exists():
        raise HTTPException(404, "export not found (it may have been pruned; export again)")
    return FileResponse(path, filename=name, media_type="application/octet-stream")


# --------------------------------------------------------------------------
# Phase 3: loft-multiplier masks (paint-to-loft brushing)
# --------------------------------------------------------------------------


class MaskPayload(BaseModel):
    w: int
    h: int
    xmin: float
    ymin: float
    res: float
    data_b64: str  # little-endian float32, h*w values, row-major


def _blank_mask_meta(part: PartInfo) -> dict:
    """Grid metadata for a not-yet-painted part, matching the preview
    raster geometry (same +3 cell / 1-cell-border convention)."""
    res = preview.PREVIEW_RES
    xmin, ymin = float(part.bbox_min[0]), float(part.bbox_min[1])
    nx = int(np.ceil((part.bbox_max[0] - xmin) / res)) + 3
    ny = int(np.ceil((part.bbox_max[1] - ymin) / res)) + 3
    return {"w": nx, "h": ny, "xmin": xmin, "ymin": ymin, "res": res, "data_b64": None}


@app.get("/api/projects/{project_id}/parts/{part_id}/mask")
def get_mask(project_id: str, part_id: int):
    project = _load_project(project_id)
    part = next((p for p in project.parts if p.id == part_id), None)
    if part is None:
        raise HTTPException(404, "part not found")
    mask = _load_mask(project, part)
    if mask is None:
        return _blank_mask_meta(part)
    grid = mask["grid"].astype("<f4")
    return {
        "w": grid.shape[1],
        "h": grid.shape[0],
        "xmin": mask["xmin"],
        "ymin": mask["ymin"],
        "res": mask["res"],
        "data_b64": base64.b64encode(grid.tobytes()).decode(),
    }


@app.put("/api/projects/{project_id}/parts/{part_id}/mask")
def put_mask(project_id: str, part_id: int, payload: MaskPayload):
    project = _load_project(project_id)
    part = next((p for p in project.parts if p.id == part_id), None)
    if part is None:
        raise HTTPException(404, "part not found")
    if payload.w < 1 or payload.h < 1 or payload.w * payload.h > 4_000_000:
        raise HTTPException(400, "mask dimensions out of range")
    try:
        data = np.frombuffer(base64.b64decode(payload.data_b64), dtype="<f4")
    except Exception:
        raise HTTPException(400, "bad mask data")
    if data.size != payload.w * payload.h:
        raise HTTPException(400, f"expected {payload.w * payload.h} values, got {data.size}")
    grid = np.clip(data.reshape(payload.h, payload.w), 0.0, 4.0)

    mdir = PROJECTS_DIR / project_id / "masks"
    mdir.mkdir(parents=True, exist_ok=True)
    fname = f"part{part_id}.npz"
    np.savez_compressed(
        mdir / fname,
        mask=grid.astype(np.float32),
        xmin=payload.xmin,
        ymin=payload.ymin,
        res=payload.res,
    )
    part.mask_file = fname
    part.mask_version += 1
    _save_project(project)
    return {"ok": True, "mask_version": part.mask_version}


@app.delete("/api/projects/{project_id}/parts/{part_id}/mask")
def delete_mask(project_id: str, part_id: int):
    project = _load_project(project_id)
    part = next((p for p in project.parts if p.id == part_id), None)
    if part is None:
        raise HTTPException(404, "part not found")
    if part.mask_file:
        (PROJECTS_DIR / project_id / "masks" / part.mask_file).unlink(missing_ok=True)
        part.mask_file = None
        part.mask_version += 1
        _save_project(project)
    return {"ok": True, "mask_version": part.mask_version}


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    d = _project_dir(project_id)
    shutil.rmtree(d)
    return JSONResponse({"deleted": project_id})


# --------------------------------------------------------------------------
# Static frontend (mounted last so /api keeps priority)
# --------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


def main() -> None:
    import uvicorn

    PROJECTS_DIR.mkdir(exist_ok=True)
    threading.Timer(1.2, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
