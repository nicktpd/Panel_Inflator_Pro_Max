// App state + API glue: import flow with job polling, part selection,
// debounced live preview, before/after, exports, keyboard shortcuts.

import { Viewer } from './viewer.js';
import { renderParamSliders, fmtDims, DEFAULT_PARAMS } from './panels.js';
import { initBrush } from './brush.js';

const $ = (id) => document.getElementById(id);

const MM_PER_INCH = 25.4;

const state = {
  project: null,
  selectedPartId: null,
  units: 'inch',
  previewJobRunning: false,
  previewDirty: false,
};

const viewer = new Viewer($('canvas3d'));
const brush = initBrush(viewer, {
  api,
  getProject: () => state.project,
  schedulePreview: () => schedulePreview(),
  setStatus: (m, o) => setStatus(m, o),
});

// ---------------------------------------------------------------------------
// status bar
// ---------------------------------------------------------------------------

function setStatus(msg, { progress = null, error = false } = {}) {
  const el = $('status-text');
  el.textContent = msg;
  el.classList.toggle('err', error);
  $('progress-fill').style.width = progress === null ? '0%' : `${Math.round(progress * 100)}%`;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail);
  }
  return r.json();
}

function pollJob(jobId, label) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      let job;
      try { job = await api(`/api/jobs/${jobId}`); }
      catch (e) { return reject(e); }
      if (job.status === 'running') {
        setStatus(`${label}: ${job.message}`, { progress: job.progress });
        setTimeout(tick, 250);
      } else if (job.status === 'done') {
        resolve(job.result);
      } else {
        reject(new Error(job.error || 'job failed'));
      }
    };
    tick();
  });
}

// ---------------------------------------------------------------------------
// import flow
// ---------------------------------------------------------------------------

async function importFile(file) {
  let options = null;
  const ext = file.name.toLowerCase().split('.').pop();
  if (ext === 'svg' || ext === 'dxf') {
    // Kick off the outline extraction so the dialog can show the shape
    // we actually parsed from the file before the user commits.
    const previewPromise = (async () => {
      const fd = new FormData();
      fd.append('file', file);
      return api('/api/inspect2d', { method: 'POST', body: fd });
    })();
    options = await ask2DOptions(previewPromise);
    if (options === null) return; // cancelled
  }

  try {
    setStatus(`uploading ${file.name}…`, { progress: 0 });
    const fd = new FormData();
    fd.append('file', file);
    if (options) fd.append('options', JSON.stringify(options));
    const { job_id } = await api('/api/projects', { method: 'POST', body: fd });
    const result = await pollJob(job_id, 'importing');
    setStatus(`imported ${file.name} — ${result.project.parts.length} parts`);
    await openProject(result.project); // ends with 'preview updated' status
  } catch (e) {
    setStatus(`import failed: ${e.message}`, { error: true });
  }
}

function currentScale() {
  const unitsSel = $('d2-units');
  return unitsSel.value === 'custom'
    ? parseFloat($('d2-custom').value) || 1.0
    : parseFloat(unitsSel.value);
}

function ask2DOptions(previewPromise) {
  const dlg = $('dialog-2d');
  const unitsSel = $('d2-units');
  const customWrap = $('d2-custom-wrap');
  const statusEl = $('d2-preview-status');

  // State: outlines arrive async; the canvas re-renders whenever the
  // unit factor changes so the dimension readout stays correct.
  let outlines = null;
  const rerender = () => {
    if (outlines) drawOutlinePreview(outlines, currentScale());
    renderPartDims(outlines, currentScale());
  };

  unitsSel.onchange = () => {
    customWrap.hidden = unitsSel.value !== 'custom';
    rerender();
  };
  $('d2-custom').oninput = rerender;

  // Reset preview UI.
  statusEl.textContent = 'reading outline…';
  statusEl.classList.remove('err', 'hidden');
  $('d2-parts').innerHTML = '';
  const ctx = $('d2-preview').getContext('2d');
  ctx.clearRect(0, 0, $('d2-preview').width, $('d2-preview').height);

  previewPromise.then((res) => {
    outlines = res.outlines;
    statusEl.classList.add('hidden');
    rerender();
  }).catch((e) => {
    statusEl.textContent = `couldn't read outline: ${e.message}`;
    statusEl.classList.add('err');
    statusEl.classList.remove('hidden');
  });

  return new Promise((resolve) => {
    dlg.onclose = () => {
      if (dlg.returnValue !== 'ok') return resolve(null);
      // Dialog inputs are in inches; the engine works in mm.
      resolve({
        scale: currentScale(),
        thickness: (parseFloat($('d2-thickness').value) || 2) * MM_PER_INCH,
        roundover: (parseFloat($('d2-roundover').value) || 0) * MM_PER_INCH,
      });
    };
    dlg.showModal();
  });
}

const PREVIEW_COLORS = ['#c9a86a', '#8fb6d9', '#a3c98f', '#d99a8f', '#bfa3d9', '#d9c98f'];

// Draw the extracted outline(s) on the dialog canvas: exterior filled
// with holes punched (even-odd), auto-fit with a small margin, y flipped
// so it matches how the panel sits (y up).
function drawOutlinePreview(outlines, scale) {
  const canvas = $('d2-preview');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, pad = 16;
  ctx.clearRect(0, 0, W, H);
  if (!outlines || !outlines.length) return;

  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const o of outlines) {
    minx = Math.min(minx, o.bbox[0]); miny = Math.min(miny, o.bbox[1]);
    maxx = Math.max(maxx, o.bbox[2]); maxy = Math.max(maxy, o.bbox[3]);
  }
  const w = maxx - minx || 1, h = maxy - miny || 1;
  const k = Math.min((W - 2 * pad) / w, (H - 2 * pad) / h);
  const offx = (W - w * k) / 2, offy = (H - h * k) / 2;
  const tx = (x) => offx + (x - minx) * k;
  const ty = (y) => H - (offy + (y - miny) * k); // flip y

  outlines.forEach((o, i) => {
    const color = PREVIEW_COLORS[i % PREVIEW_COLORS.length];
    const path = new Path2D();
    const addLoop = (loop) => {
      loop.forEach((p, j) => {
        const X = tx(p[0]), Y = ty(p[1]);
        if (j === 0) path.moveTo(X, Y); else path.lineTo(X, Y);
      });
      path.closePath();
    };
    addLoop(o.exterior);
    o.holes.forEach(addLoop);
    ctx.fillStyle = color + '33';
    ctx.fill(path, 'evenodd');
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke(path);
  });
}

function fmtDim(mm) {
  if (state.units === 'inch') return (mm / 25.4).toFixed(2) + '″';
  return Math.round(mm) + ' mm';
}

function renderPartDims(outlines, scale) {
  const ul = $('d2-parts');
  ul.innerHTML = '';
  if (!outlines || !outlines.length) { ul.classList.add('empty'); return; }
  ul.classList.remove('empty');
  outlines.forEach((o, i) => {
    const li = document.createElement('li');
    const name = document.createElement('span');
    name.textContent = o.name || `Outline ${i + 1}`;
    name.style.color = PREVIEW_COLORS[i % PREVIEW_COLORS.length];
    const dim = document.createElement('span');
    dim.className = 'pdim';
    const wmm = o.width * scale, hmm = o.height * scale;
    const holes = o.holes.length ? ` · ${o.holes.length} hole${o.holes.length > 1 ? 's' : ''}` : '';
    dim.textContent = `${fmtDim(wmm)} × ${fmtDim(hmm)}${holes}`;
    li.append(name, dim);
    ul.append(li);
  });
}

async function openProject(project) {
  state.project = project;
  state.selectedPartId = null;
  $('project-name').textContent = project.name;
  $('empty-hint').style.display = 'none';
  $('morph-wrap').hidden = false;

  viewer.clear();
  viewer.setPartClasses(Object.fromEntries(project.parts.map((p) => [p.id, p.classification])));
  renderPartList();
  renderGlobalPanel();
  renderPartPanel();

  await viewer.setBefore(`/api/projects/${project.id}/files/original.glb?v=${Date.now()}`);
  viewer.fit();
  await refreshPreview();
}

// ---------------------------------------------------------------------------
// preview (debounced, job-polled)
// ---------------------------------------------------------------------------

let previewTimer = null;

function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPreview, 400);
}

async function refreshPreview() {
  if (!state.project) return;
  if (state.previewJobRunning) { state.previewDirty = true; return; }
  state.previewJobRunning = true;
  try {
    const { job_id } = await api(`/api/projects/${state.project.id}/preview`, { method: 'POST' });
    const result = await pollJob(job_id, 'pillowing');
    await viewer.setAfter(result.url);
    setStatus('preview updated');
  } catch (e) {
    setStatus(`preview failed: ${e.message}`, { error: true });
  } finally {
    state.previewJobRunning = false;
    if (state.previewDirty) {
      state.previewDirty = false;
      schedulePreview();
    }
  }
}

// ---------------------------------------------------------------------------
// part list + panels
// ---------------------------------------------------------------------------

const PART_DOT_COLORS = ['#c9a86a', '#8fb6d9', '#a3c98f', '#d99a8f', '#bfa3d9',
  '#d9c98f', '#8fd9c3', '#d98fb6', '#9aa7d9', '#c3d98f', '#d9b68f'];

function renderPartList() {
  const ul = $('part-list');
  ul.innerHTML = '';
  for (const part of state.project.parts) {
    const li = document.createElement('li');
    li.classList.toggle('selected', part.id === state.selectedPartId);

    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = PART_DOT_COLORS[part.id % PART_DOT_COLORS.length];

    const name = document.createElement('span');
    name.className = 'pname';
    name.textContent = part.name + (part.params ? ' •' : '');
    name.title = fmtDims(part.bbox_min, part.bbox_max, state.units);

    const faces = document.createElement('span');
    faces.className = 'pfaces';
    faces.textContent = part.face_count < 1000
      ? String(part.face_count)
      : `${(part.face_count / 1000).toFixed(part.face_count > 9999 ? 0 : 1)}k`;

    const chip = document.createElement('span');
    chip.className = `chip ${part.classification}`;
    chip.textContent = part.classification === 'pillow' ? 'pillow' : 'pass';
    chip.title = 'click to toggle pillow / pass-through';
    chip.addEventListener('click', async (e) => {
      e.stopPropagation();
      await toggleClassification(part);
    });

    li.append(dot, name, faces, chip);
    li.addEventListener('click', () => selectPart(part.id));
    ul.append(li);
  }
}

function selectPart(partId) {
  state.selectedPartId = partId;
  viewer.select(partId);
  renderPartList();
  renderPartPanel();
  brush.setPart(partId);
}

viewer.onSelect = (partId) => selectPart(partId);

async function toggleClassification(part) {
  const next = part.classification === 'pillow' ? 'passthrough' : 'pillow';
  try {
    const project = await api(
      `/api/projects/${state.project.id}/parts/${part.id}`,
      { method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ classification: next }) }
    );
    state.project = project;
    viewer.setPartClasses(Object.fromEntries(project.parts.map((p) => [p.id, p.classification])));
    renderPartList();
    renderPartPanel();
    schedulePreview();
  } catch (e) {
    setStatus(`update failed: ${e.message}`, { error: true });
  }
}

function currentPart() {
  return state.project?.parts.find((p) => p.id === state.selectedPartId) || null;
}

function effectiveParams(part) {
  return { ...DEFAULT_PARAMS, ...(part.params || state.project.global_params) };
}

// -- global panel ------------------------------------------------------------

function renderGlobalPanel() {
  const container = $('global-params');
  if (!state.project) { container.innerHTML = ''; return; }
  renderParamSliders(
    container,
    { ...DEFAULT_PARAMS, ...state.project.global_params },
    state.units,
    (key, value, committed) => {
      state.project.global_params[key] = value;
      if (committed) pushGlobalParams();
    }
  );
}

let globalPatchTimer = null;
function pushGlobalParams() {
  clearTimeout(globalPatchTimer);
  globalPatchTimer = setTimeout(async () => {
    try {
      state.project = await api(`/api/projects/${state.project.id}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ global_params: state.project.global_params }),
      });
      renderPartList();
      schedulePreview();
    } catch (e) {
      setStatus(`update failed: ${e.message}`, { error: true });
    }
  }, 150);
}

$('btn-reset-global').addEventListener('click', () => {
  if (!state.project) return;
  state.project.global_params = { ...DEFAULT_PARAMS };
  renderGlobalPanel();
  pushGlobalParams();
});

// -- per-part panel ------------------------------------------------------------

function renderPartPanel() {
  const section = $('part-panel-section');
  const part = currentPart();
  if (!part) { section.hidden = true; return; }
  section.hidden = false;
  $('part-panel-title').textContent = `${part.name} — ${fmtDims(part.bbox_min, part.bbox_max, state.units)}`;

  const container = $('part-params');
  container.innerHTML = '';

  const checkRow = document.createElement('div');
  checkRow.className = 'check-row';
  const check = document.createElement('input');
  check.type = 'checkbox';
  check.id = 'part-passthrough';
  check.checked = part.classification === 'passthrough';
  const checkLabel = document.createElement('label');
  checkLabel.htmlFor = 'part-passthrough';
  checkLabel.textContent = 'Pass through untouched (hardware)';
  checkRow.append(check, checkLabel);
  check.addEventListener('change', () => toggleClassification(part));
  container.append(checkRow);

  if (part.classification === 'pillow') {
    const sliders = document.createElement('div');
    container.append(sliders);
    renderParamSliders(sliders, effectiveParams(part), state.units, (key, value, committed) => {
      const params = { ...effectiveParams(part), [key]: value };
      part.params = params;
      if (committed) pushPartParams(part, params);
    });

    const actions = document.createElement('div');
    actions.className = 'param-actions';
    const resetBtn = document.createElement('button');
    resetBtn.className = 'btn';
    resetBtn.textContent = 'Reset to global';
    resetBtn.disabled = !part.params;
    resetBtn.addEventListener('click', async () => {
      try {
        state.project = await api(
          `/api/projects/${state.project.id}/parts/${part.id}`,
          { method: 'PATCH', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reset_params: true }) }
        );
        renderPartList();
        renderPartPanel();
        schedulePreview();
      } catch (e) {
        setStatus(`update failed: ${e.message}`, { error: true });
      }
    });
    actions.append(resetBtn);
    container.append(actions);
  }
}

let partPatchTimer = null;
function pushPartParams(part, params) {
  clearTimeout(partPatchTimer);
  partPatchTimer = setTimeout(async () => {
    try {
      state.project = await api(
        `/api/projects/${state.project.id}/parts/${part.id}`,
        { method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ params }) }
      );
      renderPartList();
      renderPartPanel();
      schedulePreview();
    } catch (e) {
      setStatus(`update failed: ${e.message}`, { error: true });
    }
  }, 150);
}

// ---------------------------------------------------------------------------
// units toggle
// ---------------------------------------------------------------------------

$('units-toggle').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-u]');
  if (!btn) return;
  state.units = btn.dataset.u;
  for (const b of $('units-toggle').children) b.classList.toggle('on', b === btn);
  if (state.project) {
    renderPartList();
    renderGlobalPanel();
    renderPartPanel();
    api(`/api/projects/${state.project.id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ units_display: state.units }),
    }).catch(() => {});
  }
});

// ---------------------------------------------------------------------------
// before / after
// ---------------------------------------------------------------------------

const morphSlider = $('morph-slider');

morphSlider.addEventListener('input', () => {
  viewer.setMorph(parseFloat(morphSlider.value));
});

function toggleBeforeAfter() {
  const t = viewer.morphT >= 0.5 ? 0 : 1;
  morphSlider.value = t;
  viewer.setMorph(t);
}

$('btn-before-after').addEventListener('click', toggleBeforeAfter);

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

async function doExport(fmt) {
  if (!state.project) return;
  const res = parseFloat($('export-res').value);
  try {
    const { job_id } = await api(
      `/api/projects/${state.project.id}/export?fmt=${fmt}&res=${res}`,
      { method: 'POST' }
    );
    const result = await pollJob(job_id, `exporting ${fmt.toUpperCase()}`);
    setStatus(`export ready: ${result.filename}`);
    const a = document.createElement('a');
    a.href = result.url;
    a.download = result.filename;
    a.click();
  } catch (e) {
    setStatus(`export failed: ${e.message}`, { error: true });
  }
}

$('btn-export-stl').addEventListener('click', () => doExport('stl'));
$('btn-export-glb').addEventListener('click', () => doExport('glb'));

// ---------------------------------------------------------------------------
// file input + drag & drop
// ---------------------------------------------------------------------------

$('btn-import').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', (e) => {
  if (e.target.files.length) importFile(e.target.files[0]);
  e.target.value = '';
});

let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth++;
  $('dropzone').classList.remove('hidden');
});
window.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) { dragDepth = 0; $('dropzone').classList.add('hidden'); }
});
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  $('dropzone').classList.add('hidden');
  if (e.dataTransfer.files.length) importFile(e.dataTransfer.files[0]);
});

// ---------------------------------------------------------------------------
// saved projects dialog
// ---------------------------------------------------------------------------

$('btn-projects').addEventListener('click', async () => {
  const dlg = $('dialog-projects');
  const ul = $('project-list');
  ul.innerHTML = '<li>loading…</li>';
  dlg.showModal();
  try {
    const projects = await api('/api/projects');
    ul.innerHTML = projects.length ? '' : '<li>no saved projects yet</li>';
    for (const p of projects) {
      const li = document.createElement('li');
      const name = document.createElement('span');
      name.textContent = p.name;
      const meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = `${p.n_parts} parts · ${p.source_type} · ${p.created.slice(0, 10)}`;
      const del = document.createElement('button');
      del.className = 'mini del';
      del.textContent = 'delete';
      del.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!confirm(`Delete project "${p.name}"? The saved source copy is removed too.`)) return;
        await api(`/api/projects/${p.id}`, { method: 'DELETE' });
        li.remove();
      });
      li.append(name, meta, del);
      li.addEventListener('click', async () => {
        dlg.close();
        try {
          const project = await api(`/api/projects/${p.id}`);
          setStatus(`opening ${project.name}…`);
          await openProject(project);
          setStatus(`opened ${project.name}`);
        } catch (e2) {
          setStatus(`open failed: ${e2.message}`, { error: true });
        }
      });
      ul.append(li);
    }
  } catch (e) {
    ul.innerHTML = `<li>failed to load: ${e.message}</li>`;
  }
});

// ---------------------------------------------------------------------------
// keyboard shortcuts
// ---------------------------------------------------------------------------

window.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.key === 'r' || e.key === 'R') viewer.fit();
  if (e.key === 'b' || e.key === 'B') toggleBeforeAfter();
  if (e.key === 'Escape') selectPart(null);
});

// ---------------------------------------------------------------------------
// build/version badge — lets the user confirm their copy is up to date
// ---------------------------------------------------------------------------

(async function showBuild() {
  const badge = $('build-badge');
  try {
    const v = await api('/api/version');
    if (v.install === 'zip') {
      badge.textContent = 'ZIP build (no auto-update)';
      badge.classList.add('zip');
      badge.title = 'This copy was installed from a ZIP, so it cannot self-update. '
        + 'Re-download the latest ZIP, or install with git clone for automatic updates.';
    } else {
      badge.textContent = `build ${v.hash} · ${v.date}`;
      badge.title = `${v.subject}\n(${v.branch} @ ${v.hash}, ${v.date})\n`
        + 'Click to check GitHub for a newer version.';
    }
    badge.dataset.hash = v.hash;
    badge.onclick = () => {
      // Point the user at the latest commit list so they can compare.
      window.open('https://github.com/nicktpd/Panel_Inflator_Pro_Max/commits/main', '_blank');
    };
    console.log('[Panel Inflator] running build:', v);
  } catch {
    badge.textContent = 'build ?';
  }
})();

setStatus('ready — drop a file to begin');
