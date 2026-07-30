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

## Staying up to date

Install via `git clone` (not ZIP) and updates are automatic: every
launch, `run.bat` / `run.sh` prints an update banner, does a
fast-forward pull from GitHub, and reinstalls dependencies only if they
changed. Offline or ZIP installs say so and start normally.

**Which version am I running?** The top bar shows a build badge like
`build 95dbd1a · 2026-07-30`. Click it to open the GitHub commit list
and compare. A ZIP install shows a red `ZIP build (no auto-update)`
badge instead — that's the usual reason an app "won't update".

**First update after installing an older copy:** if your app predates
the auto-update feature, its `run.bat` doesn't know to pull yet. Do a
one-time `git pull` in the project folder (or run `update.bat`); every
launch after that updates on its own.

- Force an update / see errors: `update.bat` (or `./update.sh`).
- Launch without checking: set the environment variable `PIPM_NO_UPDATE=1`.
- Your saved projects are never touched by updates (`projects/` is
  ignored by git).

So the improvement loop is: ask Claude for a change → Claude pushes to
`main` → double-click `run.bat` and you're on the new version.

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

## Using it

1. **Import** — drag in an `.stl`, `.svg`, or `.dxf`. 2D files first show
   an **outline preview**: the exact shape extracted from the file (holes
   punched, dimensions and panel-key name shown) so you can confirm it's
   what you expect before committing. The same dialog sets units, base
   thickness (default 50.8 mm = 2"), and edge roundover (default 8 mm) —
   the dimension readout updates live as you change the unit. Big STLs
   show a progress bar; nothing blocks.
2. **Review parts** — the sidebar lists every connected component with a
   `PILLOW` / `PASS` chip (click the chip to reclassify). Click a part in
   the list or in the 3D view to select it.
3. **Tune** — global sliders set the defaults; a selected part can
   override them (crown, saturation width, exponent, smoothing) or be
   reset to global. The preview recomputes live (~400 ms debounce), and
   only changed parts are recomputed thanks to per-part caching.
4. **Paint** (optional) — with a part selected, open **Loft brush** and
   enter paint mode. Drag on the surface: blue regions loft less, orange
   more. Paint / Smooth / Erase modes, adjustable radius, strength and
   target multiplier. Strokes are blurred slightly on the backend so the
   vinyl never creases. `Esc` exits paint mode.
5. **Export** — STL or GLB, at full (2 mm) or preview (4 mm) resolution.

## Keyboard shortcuts

| Key | Action |
| --- | ------ |
| `R` | Reset camera view |
| `B` | Before/after toggle |
| `Esc` | Deselect / exit paint mode |

## Performance (measured)

A synthetic 11-petal FlowerBoard analog — 1.79M faces, 90 MB STL, petals
sharing hub vertices, plus mounting pins — on a modest 2-core container:

| Operation | Time |
| --- | --- |
| Import + split + classify | ~13 s |
| First full preview | ~5 s |
| Preview after tweaking one petal | ~3 s (10 of 11 parts cached) |
| Full-res (2 mm) STL export | ~6 s |

Peak backend memory: **1.7 GB** — safe on a 3–4 GB machine.

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
- **A panel imported as `PASS`** — components under 10k faces are assumed
  to be hardware, and a part with no flat top can't be pillowed. Click
  the chip in the part list to force it to `PILLOW` if it's a real panel.
- **A hole in my outline got filled in** — enclosed voids smaller than
  ~200 mm² are treated as mesh noise and filled. Real cutouts (25 mm+
  across) always survive and pull the fabric down toward their rims.
- **SVG imports at the wrong size** — pick the right unit in the import
  dialog; "SVG px at 96 dpi" is what Inkscape exports by default.

## Roadmap

Phase 4 (not built yet): trace a photo/sketch (PNG/JPG) into an outline —
see the disabled "Trace sketch" button and `backend/engine/trace_stub.py`.
