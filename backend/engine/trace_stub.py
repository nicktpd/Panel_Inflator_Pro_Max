"""PHASE 4 STUB — PNG/JPG sketch tracing to outline. NOT IMPLEMENTED.

Planned design (do not build yet, per the project spec):

  1. Load a photo/sketch of a panel outline (PNG/JPG).
  2. Grayscale -> adaptive threshold -> morphological cleanup.
  3. ``cv2.findContours`` (RETR_CCOMP for outer contours + holes),
     ``cv2.approxPolyDP`` at ~0.5 mm tolerance after the user sets a
     pixels-per-mm scale (probably by clicking two points a known
     distance apart).
  4. Feed the resulting rings into ``import_2d.rings_to_polygons`` and
     the existing extrude + pillow pipeline unchanged.

Dependency note: this will need ``opencv-python-headless`` — keep it an
optional extra so the core app never requires it.

The API endpoint and UI button exist but are disabled; calling the
function raises so nothing can silently pretend to trace.
"""
from __future__ import annotations


def trace_image_to_rings(path: str, px_per_mm: float) -> list:
    """Trace a raster sketch into closed outline rings. NOT IMPLEMENTED."""
    raise NotImplementedError(
        "Sketch tracing is a Phase 4 feature and has not been built yet."
    )
