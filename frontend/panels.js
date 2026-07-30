// Parameter panel UI: slider rows for the pillow parameters, used for
// both the global panel and the selected-part override panel.

export const PARAM_DEFS = [
  { key: 'crown', label: 'Crown height', min: 0, max: 100, step: 0.5, unit: 'mm',
    hint: 'max loft at deep interior points' },
  { key: 'dref', label: 'Saturation width', min: 20, max: 400, step: 1, unit: 'mm',
    hint: 'edge distance where crown maxes out (DREF)' },
  { key: 'exp', label: 'Profile exponent', min: 0.2, max: 1.5, step: 0.01, unit: '',
    hint: '<1 = fast rise, gentle dome' },
  { key: 'sigma', label: 'Smoothing', min: 0, max: 15, step: 0.5, unit: 'cells',
    hint: 'gaussian blur of the distance field' },
  { key: 'tension', label: 'Tension', min: 0, max: 1, step: 0.05, unit: '',
    hint: 'membrane pull: lowers the crown in narrow/tapering areas (0 = off, straight panels unaffected)' },
];

export const DEFAULT_PARAMS = { crown: 32.0, dref: 110.0, exp: 0.55, sigma: 5.0, w_exp: 1.5, tension: 0.7 };

const MM_PER_INCH = 25.4;

function fmtVal(def, v, units) {
  if (def.unit === 'mm' && units === 'inch') {
    return (v / MM_PER_INCH).toFixed(2) + '″';
  }
  const s = def.step < 0.1 ? v.toFixed(2) : (def.step < 1 ? v.toFixed(1) : String(Math.round(v)));
  return def.unit ? `${s} ${def.unit}` : s;
}

/**
 * Render slider rows into `container`.
 * params: current values (mutated copy handled by caller via onChange).
 * onChange(key, value, committed) — committed=true on release.
 */
export function renderParamSliders(container, params, units, onChange) {
  container.innerHTML = '';
  for (const def of PARAM_DEFS) {
    const row = document.createElement('div');
    row.className = 'param-row';
    const label = document.createElement('label');
    label.title = def.hint;
    const name = document.createElement('span');
    name.textContent = def.label;
    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = fmtVal(def, params[def.key], units);
    label.append(name, val);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = def.min; slider.max = def.max; slider.step = def.step;
    slider.value = params[def.key];
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      val.textContent = fmtVal(def, v, units);
      onChange(def.key, v, false);
    });
    slider.addEventListener('change', () => {
      onChange(def.key, parseFloat(slider.value), true);
    });

    row.append(label, slider);
    container.append(row);
  }
}

/** Dimensions string for the part list (display units only). */
export function fmtDims(bmin, bmax, units) {
  const d = bmax.map((v, i) => v - bmin[i]);
  if (units === 'inch') {
    return d.map((v) => (v / MM_PER_INCH).toFixed(1)).join('″ × ') + '″';
  }
  return d.map((v) => Math.round(v)).join(' × ') + ' mm';
}
