// Phase 3: selective loft painting.
//
// A per-part loft-multiplier grid (1.0 = normal loft) is painted directly
// on the pillowed 3D surface: raycast hit -> part-local xy -> grid cells.
// While painting, the mask is shown as a vertex-color overlay (blue =
// suppressed loft, orange = boosted). On stroke end the mask is PUT to the
// backend, which re-pillows with `dz * w * mask` and re-blurs slightly so
// brush edges never crease the surface.

import * as THREE from 'three';

const BASE = new THREE.Color(0.72, 0.67, 0.60);   // mask == 1
const LOW = new THREE.Color(0.20, 0.38, 0.85);    // mask == 0
const HIGH = new THREE.Color(0.95, 0.45, 0.10);   // mask == 2

export function initBrush(viewer, ctx) {
  // ctx: { api, getProject, schedulePreview, setStatus }
  const section = document.getElementById('brush-section');
  const controls = document.getElementById('brush-controls');

  const state = {
    partId: null,
    active: false,
    painting: false,
    dirty: false,
    mask: null,        // { w, h, xmin, ymin, res, data: Float32Array }
    radius: 40,        // mm
    strength: 0.35,
    target: 1.5,
    mode: 'paint',     // 'paint' | 'smooth' | 'erase'
    savedMaterials: new Map(),
  };

  // Brush cursor: a ring hovering over the surface.
  const cursor = new THREE.LineLoop(
    circleGeometry(64),
    new THREE.LineBasicMaterial({ color: 0xe8b84b, transparent: true, opacity: 0.9 })
  );
  cursor.visible = false;
  viewer.scene.add(cursor);

  buildControls();
  section.hidden = true;

  // ---- UI -----------------------------------------------------------------

  function buildControls() {
    controls.innerHTML = '';

    const toggle = document.createElement('button');
    toggle.className = 'btn';
    toggle.id = 'brush-toggle';
    toggle.textContent = 'Enter paint mode';
    toggle.addEventListener('click', () => setActive(!state.active));
    controls.append(toggle);

    const modeRow = document.createElement('div');
    modeRow.className = 'brush-mode-row';
    modeRow.style.marginTop = '8px';
    for (const [mode, label, tip] of [
      ['paint', 'Paint', 'paint toward the target multiplier'],
      ['smooth', 'Smooth', 'blur the mask locally'],
      ['erase', 'Erase', 'paint back toward 1.0 (no effect)'],
    ]) {
      const b = document.createElement('button');
      b.className = 'btn';
      b.dataset.mode = mode;
      b.textContent = label;
      b.title = tip;
      b.addEventListener('click', () => {
        state.mode = mode;
        for (const x of modeRow.children) x.classList.toggle('active', x === b);
      });
      if (mode === state.mode) b.classList.add('active');
      modeRow.append(b);
    }
    controls.append(modeRow);

    controls.append(
      sliderRow('Radius', 5, 150, 1, state.radius, 'mm', (v) => { state.radius = v; }),
      sliderRow('Strength', 0.05, 1, 0.05, state.strength, '', (v) => { state.strength = v; }),
      sliderRow('Paint toward', 0, 2, 0.05, state.target, '×', (v) => { state.target = v; }),
    );

    const actions = document.createElement('div');
    actions.className = 'param-actions';
    const clear = document.createElement('button');
    clear.className = 'btn';
    clear.textContent = 'Clear mask';
    clear.title = 'reset the whole part to multiplier 1.0';
    clear.addEventListener('click', clearMask);
    actions.append(clear);
    controls.append(actions);

    const hint = document.createElement('div');
    hint.style.cssText = 'font-size:10px;color:var(--text-dim);margin-top:6px;';
    hint.textContent = 'Drag on the selected part to paint. Blue = flat, orange = boosted.';
    controls.append(hint);
  }

  function sliderRow(label, min, max, step, value, unit, onInput) {
    const row = document.createElement('div');
    row.className = 'param-row';
    row.style.marginTop = '8px';
    const lab = document.createElement('label');
    const name = document.createElement('span');
    name.textContent = label;
    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = `${value}${unit ? ' ' + unit : ''}`;
    lab.append(name, val);
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = min; slider.max = max; slider.step = step; slider.value = value;
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      val.textContent = `${v}${unit ? ' ' + unit : ''}`;
      onInput(v);
    });
    row.append(lab, slider);
    return row;
  }

  // ---- mask data ------------------------------------------------------------

  async function fetchMask(partId) {
    const project = ctx.getProject();
    const meta = await ctx.api(`/api/projects/${project.id}/parts/${partId}/mask`);
    const n = meta.w * meta.h;
    let data;
    if (meta.data_b64) {
      const raw = atob(meta.data_b64);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      data = new Float32Array(bytes.buffer);
    } else {
      data = new Float32Array(n).fill(1.0);
    }
    state.mask = { w: meta.w, h: meta.h, xmin: meta.xmin, ymin: meta.ymin, res: meta.res, data };
  }

  async function commitMask() {
    if (!state.mask || !state.dirty) return;
    state.dirty = false;
    const project = ctx.getProject();
    const bytes = new Uint8Array(state.mask.data.buffer.slice(0));
    let bin = '';
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    await ctx.api(`/api/projects/${project.id}/parts/${state.partId}/mask`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        w: state.mask.w, h: state.mask.h,
        xmin: state.mask.xmin, ymin: state.mask.ymin, res: state.mask.res,
        data_b64: btoa(bin),
      }),
    });
    ctx.schedulePreview();
  }

  async function clearMask() {
    if (state.partId === null) return;
    const project = ctx.getProject();
    await ctx.api(`/api/projects/${project.id}/parts/${state.partId}/mask`, { method: 'DELETE' });
    await fetchMask(state.partId);
    applyOverlay();
    ctx.schedulePreview();
  }

  // ---- painting ------------------------------------------------------------

  function sampleMask(x, y) {
    const m = state.mask;
    const fc = (x - m.xmin) / m.res + 1;
    const fr = (y - m.ymin) / m.res + 1;
    const c0 = Math.max(0, Math.min(m.w - 1, Math.floor(fc)));
    const r0 = Math.max(0, Math.min(m.h - 1, Math.floor(fr)));
    const c1 = Math.min(m.w - 1, c0 + 1);
    const r1 = Math.min(m.h - 1, r0 + 1);
    const tc = Math.max(0, Math.min(1, fc - c0));
    const tr = Math.max(0, Math.min(1, fr - r0));
    const d = m.data;
    const v00 = d[r0 * m.w + c0], v01 = d[r0 * m.w + c1];
    const v10 = d[r1 * m.w + c0], v11 = d[r1 * m.w + c1];
    return (v00 * (1 - tc) + v01 * tc) * (1 - tr) + (v10 * (1 - tc) + v11 * tc) * tr;
  }

  function paintAt(x, y) {
    const m = state.mask;
    const rCells = state.radius / m.res;
    const cc = (x - m.xmin) / m.res + 1;
    const cr = (y - m.ymin) / m.res + 1;
    const c0 = Math.max(0, Math.floor(cc - rCells));
    const c1 = Math.min(m.w - 1, Math.ceil(cc + rCells));
    const r0 = Math.max(0, Math.floor(cr - rCells));
    const r1 = Math.min(m.h - 1, Math.ceil(cr + rCells));
    const target = state.mode === 'erase' ? 1.0 : state.target;
    // Scaled so a slow drag converges without instantly saturating.
    const k = state.strength * 0.28;

    if (state.mode === 'smooth') {
      const src = state.mask.data.slice();
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const dx = c - cc, dy = r - cr;
          const d2 = (dx * dx + dy * dy) / (rCells * rCells);
          if (d2 > 1) continue;
          const f = (1 - d2) * (1 - d2);
          const rm = Math.max(0, r - 1), rp = Math.min(m.h - 1, r + 1);
          const cm = Math.max(0, c - 1), cp = Math.min(m.w - 1, c + 1);
          const mean = (src[rm * m.w + c] + src[rp * m.w + c] +
                        src[r * m.w + cm] + src[r * m.w + cp] +
                        src[r * m.w + c]) / 5;
          const i = r * m.w + c;
          m.data[i] += (mean - m.data[i]) * Math.min(1, k * 3) * f;
        }
      }
    } else {
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const dx = c - cc, dy = r - cr;
          const d2 = (dx * dx + dy * dy) / (rCells * rCells);
          if (d2 > 1) continue;
          const f = (1 - d2) * (1 - d2); // smooth falloff to the rim
          const i = r * m.w + c;
          m.data[i] += (target - m.data[i]) * k * f;
        }
      }
    }
    state.dirty = true;
  }

  // ---- overlay -------------------------------------------------------------

  function maskColor(v, out) {
    if (v <= 1) out.copy(LOW).lerp(BASE, Math.max(0, v));
    else out.copy(BASE).lerp(HIGH, Math.min(1, v - 1));
    return out;
  }

  function applyOverlay() {
    if (!state.active || state.partId === null || !state.mask) return;
    const tmp = new THREE.Color();
    for (const mesh of viewer.partMeshes(state.partId)) {
      if (!state.savedMaterials.has(mesh)) {
        state.savedMaterials.set(mesh, mesh.material);
        mesh.material = mesh.material.clone();
        mesh.material.vertexColors = true;
        mesh.material.color.set(0xffffff);
      }
      const pos = mesh.geometry.attributes.position;
      let colors = mesh.geometry.attributes.color;
      if (!colors || colors.count !== pos.count) {
        colors = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3);
        mesh.geometry.setAttribute('color', colors);
      }
      for (let i = 0; i < pos.count; i++) {
        maskColor(sampleMask(pos.getX(i), pos.getY(i)), tmp);
        colors.setXYZ(i, tmp.r, tmp.g, tmp.b);
      }
      colors.needsUpdate = true;
    }
  }

  function removeOverlay() {
    for (const [mesh, mat] of state.savedMaterials) {
      if (mesh.material) mesh.material.dispose();
      mesh.material = mat;
    }
    state.savedMaterials.clear();
  }

  // ---- mode switching --------------------------------------------------------

  async function setActive(on) {
    if (on && state.partId === null) return;
    state.active = on;
    const toggle = document.getElementById('brush-toggle');
    if (toggle) {
      toggle.textContent = on ? 'Exit paint mode' : 'Enter paint mode';
      toggle.classList.toggle('accent', on);
    }
    viewer.pickEnabled = !on;
    viewer.setRotateEnabled(!on);
    if (on) {
      if (!state.mask) await fetchMask(state.partId);
      // Painting happens on the pillowed surface; make sure it's shown.
      const morph = document.getElementById('morph-slider');
      morph.value = 1;
      viewer.setMorph(1);
      applyOverlay();
      ctx.setStatus('paint mode: drag to paint, blue = flat, orange = boosted');
    } else {
      cursor.visible = false;
      removeOverlay();
      commitMask().catch((e) => ctx.setStatus(`mask save failed: ${e.message}`, { error: true }));
    }
  }

  // ---- pointer handling -------------------------------------------------------

  const canvas = viewer.canvas;

  canvas.addEventListener('pointerdown', (e) => {
    if (!state.active || e.button !== 0) return;
    const hit = viewer.raycastAt(e.clientX, e.clientY, state.partId);
    if (!hit) return;
    state.painting = true;
    canvas.setPointerCapture(e.pointerId);
    paintAt(hit.point.x, hit.point.y);
    applyOverlay();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!state.active) return;
    const hit = viewer.raycastAt(e.clientX, e.clientY, state.partId);
    if (hit) {
      cursor.visible = true;
      cursor.position.set(hit.point.x, hit.point.y, hit.point.z + 1.5);
      cursor.scale.setScalar(state.radius);
    } else {
      cursor.visible = false;
    }
    if (state.painting && hit) {
      paintAt(hit.point.x, hit.point.y);
      applyOverlay();
    }
  });

  const endStroke = (e) => {
    if (!state.painting) return;
    state.painting = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* ok */ }
    commitMask().catch((err) => ctx.setStatus(`mask save failed: ${err.message}`, { error: true }));
  };
  canvas.addEventListener('pointerup', endStroke);
  canvas.addEventListener('pointercancel', endStroke);

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.active) setActive(false);
  });

  // Re-apply the overlay when the preview mesh is swapped in.
  viewer.onAfterLoaded = () => {
    state.savedMaterials.clear(); // old meshes disposed with the old group
    if (state.active) applyOverlay();
  };

  // ---- public API ---------------------------------------------------------

  return {
    async setPart(partId) {
      if (state.active) await setActive(false);
      state.partId = partId;
      state.mask = null;
      const project = ctx.getProject();
      const part = project?.parts.find((p) => p.id === partId);
      const paintable = part && part.classification === 'pillow';
      section.hidden = !paintable;
    },
    setEnabled(v) { if (!v) setActive(false); },
    dispose() {},
  };
}

function circleGeometry(segments) {
  const pts = [];
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0));
  }
  return new THREE.BufferGeometry().setFromPoints(pts);
}
