"""STL import: load, split into parts, classify panel vs hardware.

Memory discipline matters here: the production validation file is 1.6M
faces / 82 MB and target machines may have 3-4 GB free RAM. Vertices are
stored float32, parts are extracted one at a time, and the full mesh is
freed as soon as the split is done.
"""
from __future__ import annotations

import gc

import numpy as np
import trimesh

# Components with fewer faces than this are hardware (mounting pins,
# dowels, slivers) and pass through the pipeline untouched.
HARDWARE_FACE_THRESHOLD = 10_000


def load_split_classify(path: str) -> list[dict]:
    """Load an STL and split it into classified connected components.

    The split uses EDGE-based face adjacency (two faces are connected only
    if they share a full edge). Production warning baked in: do NOT use
    vertex connectivity -- separate panels in real CAD exports share
    coincident vertices at junctions (the FlowerBoard petals all touch the
    hub) and vertex connectivity merges them into one blob. Sharing a
    single vertex does not create a face-adjacency edge, so edge-based
    labelling keeps them separate.

    Returns a list of dicts sorted by face count (descending):
      {vertices: float32 (n,3), faces: int32 (m,3),
       classification: 'pillow'|'passthrough',
       bbox_min, bbox_max, face_count, has_flat_top: bool}
    """
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("File contains no triangle geometry")

    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces)
    )
    n_parts = int(labels.max()) + 1 if len(labels) else 0

    all_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    all_faces = np.asarray(mesh.faces, dtype=np.int64)
    del mesh
    gc.collect()

    parts: list[dict] = []
    for lab in range(n_parts):
        fmask = labels == lab
        pf_global = all_faces[fmask]
        vids, inv = np.unique(pf_global, return_inverse=True)
        pv = all_vertices[vids].copy()
        pf = inv.reshape(-1, 3).astype(np.int32)
        parts.append(_describe_part(pv, pf))
    del all_vertices, all_faces, labels
    gc.collect()

    parts.sort(key=lambda p: p["face_count"], reverse=True)
    return parts


def _describe_part(
    pv: np.ndarray, pf: np.ndarray, hardware_threshold: int = HARDWARE_FACE_THRESHOLD
) -> dict:
    """Classify one component and record its metadata."""
    face_count = int(len(pf))
    zmax = float(pv[:, 2].max())
    vtop = np.abs(pv[:, 2] - zmax) < 1e-3
    has_flat_top = bool((vtop[pf].all(axis=1)).any())
    if face_count < hardware_threshold or not has_flat_top:
        classification = "passthrough"
    else:
        classification = "pillow"
    return {
        "vertices": pv,
        "faces": pf,
        "classification": classification,
        "face_count": face_count,
        "bbox_min": pv.min(axis=0).tolist(),
        "bbox_max": pv.max(axis=0).tolist(),
        "has_flat_top": has_flat_top,
    }


def describe_parts_from_arrays(
    part_arrays: list[tuple[np.ndarray, np.ndarray]],
    hardware_threshold: int = 0,
    name_hints: list[str | None] | None = None,
) -> list[dict]:
    """Same classification/metadata for parts built elsewhere (Phase 2).

    Extruded 2D outlines are lean meshes (hundreds of faces, not the dense
    CAD tessellations the 10k STL threshold assumes), so by default every
    flat-topped 2D part is pillowable regardless of face count.

    name_hints (parallel to part_arrays) carries display names derived
    from drawing annotation, e.g. "Panel H" from a DXF panel-key label.
    """
    parts = []
    for i, (pv, pf) in enumerate(part_arrays):
        part = _describe_part(pv.astype(np.float32), pf.astype(np.int32), hardware_threshold)
        if name_hints is not None and name_hints[i]:
            part["name_hint"] = name_hints[i]
        parts.append(part)
    parts.sort(key=lambda p: p["face_count"], reverse=True)
    return parts
