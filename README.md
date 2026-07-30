# Panel Inflator Pro Max

Turn flat panel geometry (flat 3D STLs from CAD, or 2D vector outlines) into
realistic upholstered / padded 3D models. It applies a distance-field
"pillow" displacement: edges stay pinned low where the vinyl wraps, deep
interior points loft high, holes and notches pull the fabric down toward
their rims — the same physics as a real stretched-vinyl-over-foam panel.

Everything runs **locally**: a Python (FastAPI) backend does the geometry,
a vanilla-JS + Three.js frontend gives you a 3D viewer with per-part
parameters. No Node, no npm, no build step — a machine with only Python
installed can run this.

## Quickstart

**One-time per machine** (needs Python 3.11+ installed):

- Windows: double-click `setup.bat`
- macOS/Linux: `./setup.sh`

**Every day after that:**

- Windows: double-click `run.bat`
- macOS/Linux: `./run.sh`

The server starts on `http://127.0.0.1:8177` and your browser opens
automatically. Drag an STL (or SVG/DXF outline) into the window.

> **Cloud sync users:** exclude `.venv/` from sync — see `SYNC-IGNORE.txt`.

## What it does

1. **Import** — an STL is split into connected parts (edge-adjacency, so
   panels that share coincident vertices stay separate). Small components
   (mounting pins, dowels) are classified as pass-through hardware; big
   ones become pillowable panels. 2D outlines (SVG/DXF) are extruded into
   a base solid first.
2. **Pillow** — for each panel, the flat top is rasterized, an interior
   distance field is computed, and the crown profile
   `dz = CROWN * clip(dist/DREF, 0, 1) ^ EXP` is applied. The top is
   retriangulated (never subdivided), side walls barrel outward with
   height, and the bottom stays perfectly flat for wall mounting.
3. **Review & export** — orbit the result in 3D, toggle before/after,
   tweak crown/DREF/exponent per part with live preview, then export a
   full-resolution binary STL or GLB.

## Keyboard shortcuts

| Key | Action |
| --- | ------ |
| `R` | Reset camera view |
| `B` | Before/after toggle |

## Screenshots

_(coming soon — drop your own renders here)_

## Project layout

```
backend/            FastAPI app + geometry engine
frontend/           static UI (vanilla JS + vendored Three.js)
projects/           saved projects (JSON + source files; caches are regenerated)
reference/          drop photos of real upholstered panels here
tests/              engine + API tests (fixtures generated procedurally)
```

## Troubleshooting

- **`setup.bat` says Python not found** — install Python 3.11+ from
  python.org and tick "Add python.exe to PATH", then re-run.
- **Browser doesn't open** — go to `http://127.0.0.1:8177` manually.
- **Port already in use** — another copy is running; close it or reboot.
- **Broken venv after switching machines** — you synced `.venv/`. Delete
  it, exclude it from sync (`SYNC-IGNORE.txt`), run setup again.
- **Huge STL is slow to import** — normal; the 3D preview uses a coarser
  raster (4 mm) than export (2 mm). Watch the progress bar; the browser
  never blocks on long operations.
