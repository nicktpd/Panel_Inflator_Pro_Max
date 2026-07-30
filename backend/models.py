"""Pydantic schemas for Panel Inflator Pro Max projects.

A Project is persisted as ``projects/<id>/project.json`` with the uploaded
source file copied alongside, so a project survives machine switches even
when the (gitignored, regenerable) ``cache/`` directory does not.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PillowParams(BaseModel):
    """Tunable parameters of the pillow displacement.

    crown : max crown height in mm added at deep-interior points.
    dref  : distance-to-edge (mm) at which the crown saturates. Roughly the
            half-width of a panel that just reaches full crown.
    exp   : profile exponent. <1 gives a fast rise off the edge and a gentle
            dome in the middle (validated 0.55 against real product photos).
    sigma : gaussian smoothing of the distance field, in grid cells.
    w_exp : vertical weighting exponent for side-wall barreling; the bottom
            of a part is pinned (w=0) and the top moves fully (w=1).
    """

    crown: float = Field(default=32.0, ge=0.0, le=500.0)
    dref: float = Field(default=110.0, gt=0.0, le=5000.0)
    exp: float = Field(default=0.55, gt=0.0, le=4.0)
    sigma: float = Field(default=5.0, ge=0.0, le=50.0)
    w_exp: float = Field(default=1.5, ge=0.1, le=10.0)


class PartInfo(BaseModel):
    """One connected component of the imported geometry."""

    id: int
    name: str
    face_count: int
    # 'pillow' parts run through the displacement engine; 'passthrough'
    # parts (hardware: pins, dowels, slivers) are exported untouched.
    classification: Literal["pillow", "passthrough"]
    # What the importer decided before any user override (for UI display).
    auto_classification: Literal["pillow", "passthrough"]
    bbox_min: list[float]
    bbox_max: list[float]
    # Per-part parameter override; None means "use the project globals".
    params: Optional[PillowParams] = None
    # Phase 3: filename of the painted loft-multiplier mask, if any,
    # relative to the project's masks/ directory.
    mask_file: Optional[str] = None
    # Bumped every time the mask is painted, so caches invalidate.
    mask_version: int = 0


class Import2DOptions(BaseModel):
    """Options the user confirms when importing an SVG/DXF outline."""

    scale: float = Field(default=1.0, gt=0.0)  # source unit -> mm multiplier
    thickness: float = Field(default=50.8, gt=0.0)  # rigid core, mm (2")
    roundover: float = Field(default=8.0, ge=0.0)  # top edge fillet, mm


class Project(BaseModel):
    id: str
    name: str
    created: str
    source_file: str  # filename inside the project directory
    source_type: Literal["stl", "2d"]
    units_display: Literal["mm", "inch"] = "mm"  # display only, data is mm
    global_params: PillowParams = PillowParams()
    import_2d: Optional[Import2DOptions] = None
    parts: list[PartInfo] = []


class PartPatch(BaseModel):
    """PATCH body for /api/projects/{id}/parts/{part_id}."""

    classification: Optional[Literal["pillow", "passthrough"]] = None
    # Explicit null in JSON clears the override back to globals, therefore
    # we need a sentinel to distinguish "not sent" from "sent as null".
    params: Optional[PillowParams] = None
    reset_params: bool = False


class ProjectPatch(BaseModel):
    """PATCH body for /api/projects/{id}."""

    name: Optional[str] = None
    units_display: Optional[Literal["mm", "inch"]] = None
    global_params: Optional[PillowParams] = None
