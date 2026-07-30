// App state + API glue: import flow with job polling, part selection,
// debounced live preview, before/after, exports, keyboard shortcuts.

import { Viewer } from './viewer.js';
import { renderParamSliders, fmtDims, DEFAULT_PARAMS } from './panels.js';
import { initBrush } from './brush.js';

const $ = (id) => document.getElementById(id);

const state = {
  project: null,
  selectedPartId: null,
  units: 'mm',
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
    options = await ask2DOptions();
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

function ask2DOptions() {
  const dlg = $('dialog-2d');
  const unitsSel = $('d2-units');
  const customWrap = $('d2-custom-wrap');
  unitsSel.onchange = () => { customWrap.hidden = unitsSel.value !== 'custom'; };
  return new Promise((resolve) => {
    dlg.onclose = () => {
      if (dlg.returnValue !== 'ok') return resolve(null);
      const scale = unitsSel.value === 'custom'
        ? parseFloat($('d2-custom').value) || 1.0
        : parseFloat(unitsSel.value);
      resolve({
        scale,
        thickness: parseFloat($('d2-thickness').value) || 50.8,
        roundover: parseFloat($('d2-roundover').value) || 0,
      });
    };
    dlg.showModal();
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

setStatus('ready — drop a file to begin');
