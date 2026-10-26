/* LIVAGX PMF explorer - vanilla JS + inline SVG, no external dependencies.
 *
 * Opened straight from disk (file://), so nothing is fetched: the data come
 * from assets/pmf_data.js (`window.PMF_DATA = {...}`), written by
 * overall/scripts/export_explorer_data.py.
 *
 * Curve encoding in PMF_DATA (see the export script):
 *   {r0, dr, y[, b]}  uniform grid    or    {r, y[, b]}  explicit grid
 *   y = PMF (kcal/mol, null = empty bin), b = |PMF1-PMF2|/2 band (mean curves)
 */
(function () {
  'use strict';

  var D = window.PMF_DATA;
  var app = document.getElementById('px-app');
  if (!D || !D.pmf || !D.systems || !D.systems.length) {
    if (app) {
      app.innerHTML = '<p class="px-error" style="margin-top:24px">Could not load <code>assets/pmf_data.js</code>. ' +
        'Re-create it from the <code>overall/</code> folder with ' +
        '<code>python scripts/export_explorer_data.py</code>.</p>';
    }
    return;
  }

  // ------------------------------------------------------------ constants
  const SVGNS = 'http://www.w3.org/2000/svg';
  const MINUS = '−';
  const NDASH = '–';
  const SYS = D.systems;
  const SYS_I = {};
  SYS.forEach((s, i) => { SYS_I[s.key] = i; });
  const BY_KEY = {};
  SYS.forEach((s) => { BY_KEY[s.key] = s; });
  const CONV = D.conventions || {};
  const NORM = CONV.normalization_nm || 1.8;
  const FIRST = D.pmf[SYS[0].key];
  const WNAMES = D.window_names || ['LIV', 'IVA', 'VAG', 'AGX'];
  const STAMP = String(D.data_as_of || D.generated || '?').replace('T', ' ');

  const FAMS = {
    pairs: {
      name: 'Residue pairs', noun: 'residue pair', ids: Object.keys(FIRST.pairs || {}),
      bin: CONV.pair_bin_nm || 0.01, x: [0.3, NORM], kinds: ['mean', 'full', 'halves'],
    },
    windows: {
      name: '3-residue windows', noun: 'window pair', ids: Object.keys(FIRST.windows || {}),
      bin: CONV.window_bin_nm || 0.02, x: [0.2, NORM], kinds: ['mean', 'halves'],
    },
    fragment: {
      name: 'LIVA fragment', noun: 'fragment', ids: ['LIVA'],
      bin: CONV.window_bin_nm || 0.02, x: [0.2, NORM], kinds: ['mean', 'halves'],
    },
  };
  const FAM_ORDER = ['pairs', 'windows', 'fragment'];
  const KINDS = {
    mean: { name: 'Mean ± band', short: 'mean of halves' },
    full: { name: 'Full trajectory', short: 'whole trajectory' },
    halves: { name: 'Halves 1 vs 2', short: 'half 1 vs half 2' },
  };
  const VIEWS = { overlay: 'Overlay', multiples: 'Small multiples', table: 'Table' };
  const BANDS = { on: 'Show', off: 'Hide' };
  const YMODES = { fit: 'Fit', shared: 'Shared', full: 'Full' };
  const YMODE_TITLES = {
    fit: 'Zoom on the wells of the curves shown',
    shared: 'One range for every PMF of this family (compare depths between cells)',
    full: 'Include the repulsive wall at short distance',
  };
  const PRESETS = [
    { name: 'All', keys: SYS.map((s) => s.key), title: 'Show all eight systems' },
    { name: 'None', keys: [], title: 'Hide all systems' },
    { name: 'Glu states', keys: ['livage_cc', 'livage_cn', 'livage_nn'], title: 'LIVAGE with 2, 1 or 0 charged Glu' },
    { name: 'Cationic X', keys: ['livagh', 'livagk', 'livagr'], title: 'X = HIP (+1), LYS (+1), ARG (+1)' },
    { name: 'Neutral X', keys: ['livage_nn', 'livagg', 'livagq'], title: 'X = GLH/GLH, GLY, GLN' },
  ].map((p) => Object.assign({}, p, { keys: p.keys.filter((k) => k in SYS_I) }));

  // colour ramps (style.py: CMAP_DIV / SEQ_BLUE); dark variants keep "more
  // magnitude = more contrast against the surface"
  const DIV_LIGHT = ['#104281', '#3987e5', '#9ec5f4', '#f0efec', '#f4a6a5', '#e34948', '#a32525'];
  const DIV_DARK = ['#b7d3f6', '#5598e7', '#1c5cab', '#2c2c2a', '#a32525', '#e34948', '#f4a6a5'];
  const SEQ_LIGHT = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
    '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
  const SEQ_DARK = SEQ_LIGHT.slice().reverse();
  const DIV_LIMIT = 1.5; // kcal/mol, diverging ramp clamps beyond +-1.5
  const Y_TICKS = 7; // target number of PMF-axis ticks in the overlay chart

  // ------------------------------------------------------------ state
  const state = {
    fam: 'pairs',
    sel: { pairs: FAMS.pairs.ids[0] || '1-1', windows: FAMS.windows.ids[0] || '1-1', fragment: 'LIVA' },
    kind: 'mean',
    view: 'overlay',
    y: 'fit',
    band: 'on',
    shade: 'avg',
    on: new Set(SYS.map((s) => s.key)),
  };
  let hoverR = null;

  // ------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const floorTo = (v, s) => Math.floor(v / s + 1e-9) * s;
  const ceilTo = (v, s) => Math.ceil(v / s - 1e-9) * s;
  const curId = () => state.sel[state.fam];
  const shownSystems = () => SYS.filter((s) => state.on.has(s.key));
  const color = (i) => 'var(--s' + i + ')';

  function fmt(v, d) {
    if (d == null) d = 2;
    if (v == null || !isFinite(v)) return NDASH;
    const s = Math.abs(v).toFixed(d);
    return (v < 0 && Number(s) !== 0 ? MINUS : '') + s;
  }
  function fmtSigned(v, d) {
    const s = fmt(v, d);
    return (v > 0 && Number(Math.abs(v).toFixed(d)) !== 0) ? '+' + s : s;
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function decimalsFor(step) {
    let d = 0;
    while (d < 4 && Math.abs(Math.round(step * Math.pow(10, d)) - step * Math.pow(10, d)) > 1e-7) d++;
    return d;
  }
  function el(tag, attrs, parent) {
    const n = document.createElementNS(SVGNS, tag);
    if (attrs) {
      for (const k in attrs) if (attrs[k] != null && attrs[k] !== false) n.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(n);
    return n;
  }
  function h(tag, attrs, html) {
    const n = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (attrs[k] == null || attrs[k] === false) continue;
        if (k === 'class') n.className = attrs[k];
        else if (k === 'text') n.textContent = attrs[k];
        else n.setAttribute(k, attrs[k]);
      }
    }
    if (html != null) n.innerHTML = html;
    return n;
  }
  function parseId(id) {
    const p = String(id).split('-');
    return [Number(p[0]), Number(p[1])];
  }
  function seqOf(sys) { return sys ? sys.sequence : 'LIVAGX'; }
  function idLabel(fam, id, sys) {
    const s = seqOf(sys);
    if (fam === 'pairs') {
      const [i, j] = parseId(id);
      return s[i - 1] + i + NDASH + s[j - 1] + j;
    }
    if (fam === 'windows') {
      const [a, b] = parseId(id);
      return s.slice(a - 1, a + 2) + NDASH + s.slice(b - 1, b + 2);
    }
    return 'LIVA' + NDASH + 'LIVA';
  }
  // systems whose residue 6 differs between the peptides (LIVAGE E-/E0: GLU on
  // peptide 1, GLH on peptide 2), so peptide-1/peptide-2 order matters for X
  function asymmetricShown() {
    return shownSystems().filter((s) => s.x_by_peptide && s.x_by_peptide[0] !== s.x_by_peptide[1]);
  }
  function asymNote(fam, id) {
    const asym = asymmetricShown();
    if (!asym.length) return '';
    const names = asym.map((s) => s.label).join(' and ');
    const what = asym.map((s) => s.x_by_peptide[0] + ' on peptide 1 but ' + s.x_by_peptide[1] + ' on peptide 2').join('; ');
    if (fam === 'pairs') {
      const [i, j] = parseId(id);
      if (i !== 6 && j !== 6) return '';
      const mirror = idLabel(fam, j + '-' + i);
      return ' In ' + names + ' X6 is ' + what + (i === j ? '.' : ', so ' + idLabel(fam, id) + ' is not equivalent to its mirror pair ' + mirror + ' there.');
    }
    if (fam === 'windows') {
      const [a, b] = parseId(id);
      if (a !== 4 && b !== 4) return '';
      return ' In ' + names + ' X6 is ' + what + (a !== b ? ', and this cell pools both orderings, i.e. both residue-6 variants.' : '.');
    }
    return '';
  }
  function idDescription(fam, id) {
    if (fam === 'pairs') {
      const [i, j] = parseId(id);
      const g = 'LIVAGX';
      return 'Centre-of-mass distance between ' + g[i - 1] + i + ' of peptide 1 (residue :' + (i + 1) +
        ') and ' + g[j - 1] + j + ' of peptide 2 (residue :' + (j + 9) + '). X = residue 6.' + asymNote(fam, id);
    }
    if (fam === 'windows') {
      const [a, b] = parseId(id);
      const pooled = a === b ? '' : ' Both orderings (peptide 1/peptide 2 and the reverse) are pooled.';
      return 'Centre-of-mass distance between the ' + WNAMES[a - 1] + ' window of one peptide and the ' +
        WNAMES[b - 1] + ' window of the other.' + pooled + asymNote(fam, id);
    }
    return 'Centre-of-mass distance between the LIVA fragments of the two peptides (residues :2' + NDASH +
      '5 vs :10' + NDASH + '13).';
  }

  // ------------------------------------------------------------ data access
  function block(key, fam, id) {
    const p = D.pmf[key];
    if (!p) return null;
    if (fam === 'fragment') return p.fragment || null;
    return (p[fam] || {})[id] || null;
  }
  const decCache = new WeakMap();
  function dec(e) {
    if (!e || !e.y) return null;
    let c = decCache.get(e);
    if (c) return c;
    const n = e.y.length;
    const r = new Array(n);
    for (let k = 0; k < n; k++) r[k] = e.r ? e.r[k] : e.r0 + k * e.dr;
    c = { r: r, y: e.y, b: e.b || null };
    decCache.set(e, c);
    return c;
  }
  // traces of one system for one PMF: [{sys, i, color, part, c, well, band, dash}]
  function tracesFor(sys, fam, id, kind) {
    const b = block(sys.key, fam, id);
    if (!b) return [];
    const i = SYS_I[sys.key];
    const base = { sys: sys, i: i, color: color(i) };
    const wells = b.well || {};
    if (kind === 'full') {
      return b.full ? [Object.assign({}, base, { part: 'full', c: dec(b.full), well: wells.full })] : [];
    }
    if (kind === 'halves') {
      const out = [];
      if (b.h1) out.push(Object.assign({}, base, { part: 'h1', c: dec(b.h1), well: wells.h1 }));
      if (b.h2) out.push(Object.assign({}, base, { part: 'h2', dash: true, c: dec(b.h2), well: wells.h2 }));
      return out;
    }
    return b.mean ? [Object.assign({}, base, { part: 'mean', band: true, c: dec(b.mean), well: wells.mean })] : [];
  }
  // PMF (or band) of a decoded curve at distance x, linear between bins
  function valueAt(c, x, arr) {
    if (!c) return null;
    arr = arr || c.y;
    const r = c.r;
    const n = r.length;
    if (!n || !arr) return null;
    const half = n > 1 ? (r[1] - r[0]) / 2 : 0.005;
    if (x < r[0] - half || x > r[n - 1] + half) return null;
    if (n === 1 || x <= r[0]) return arr[0];
    if (x >= r[n - 1]) return arr[n - 1];
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (r[mid] <= x) lo = mid; else hi = mid;
    }
    const y0 = arr[lo];
    const y1 = arr[hi];
    if (y0 == null || y1 == null) return (x - r[lo] <= r[hi] - x) ? y0 : y1;
    return y0 + (x - r[lo]) / (r[hi] - r[lo]) * (y1 - y0);
  }
  function extent(traces, xd) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const t of traces) {
      if (!t.c) continue;
      const r = t.c.r;
      const y = t.c.y;
      const b = t.band ? t.c.b : null;
      for (let k = 0; k < r.length; k++) {
        if (y[k] == null || r[k] < xd[0] || r[k] > xd[1]) continue;
        const bb = b && b[k] != null ? b[k] : 0;
        if (y[k] - bb < lo) lo = y[k] - bb;
        if (y[k] + bb > hi) hi = y[k] + bb;
      }
    }
    return [lo, hi];
  }
  const sharedCache = {};
  function sharedExtent(fam) {
    if (!sharedCache[fam]) {
      let all = [];
      for (const s of SYS) {
        for (const id of FAMS[fam].ids) {
          for (const k of FAMS[fam].kinds) all = all.concat(tracesFor(s, fam, id, k));
        }
      }
      sharedCache[fam] = extent(all, FAMS[fam].x);
    }
    return sharedCache[fam];
  }
  // PMF-axis range; both ends land on a tick so the frame is labelled
  // returns [lo, hi, tick step]
  function yDomain(traces, nt) {
    const snap = (lo, hi) => {
      const st = niceStep(hi - lo, nt || Y_TICKS);
      return [floorTo(lo, st), ceilTo(hi, st), st];
    };
    const xd = FAMS[state.fam].x;
    if (state.y === 'shared') {
      const g = sharedExtent(state.fam);
      return snap(Math.min(-0.5, g[0] - 0.05), 1.5);
    }
    const e = extent(traces, xd);
    if (!isFinite(e[0])) return [-1, 1, 0.5];
    const lo = Math.min(-0.25, e[0] - 0.03);
    if (state.y === 'full') return snap(lo, Math.max(0.5, e[1]));
    return snap(lo, Math.max(0.5, Math.min(1.5, ceilTo(-lo, 0.25))));
  }
  function niceStep(span, target) {
    const raw = span / Math.max(1, target);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw - 1e-12) return m * p;
    return 10 * p;
  }
  function ticks(lo, hi, target, forced) {
    const st = forced || niceStep(hi - lo, target);
    const out = [];
    const k0 = Math.ceil(lo / st - 1e-9);
    const k1 = Math.floor(hi / st + 1e-9);
    for (let k = k0; k <= k1; k++) out.push(Number((k * st).toFixed(10)));
    return { values: out, step: st };
  }
  function wellText(w, withBand) {
    if (!w || w[0] == null) return 'no data';
    if (w[0] > -0.02) return 'no attractive well';
    let s = 'min ' + fmt(w[0]);
    if (withBand && w[2] != null) s += ' ± ' + fmt(w[2]);
    return s + ' at ' + fmt(w[1]) + ' nm';
  }

  // ------------------------------------------------------------ colour ramps
  function isDark() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t === 'dark') return true;
    if (t === 'light') return false;
    try { return window.matchMedia('(prefers-color-scheme: dark)').matches; } catch (e) { return false; }
  }
  function hexToRgb(hx) {
    const n = parseInt(hx.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function lerpStops(stops, t) {
    t = clamp(t, 0, 1);
    const x = t * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(x));
    const f = x - k;
    const a = hexToRgb(stops[k]);
    const b = hexToRgb(stops[k + 1]);
    const c = a.map((v, q) => Math.round(v + (b[q] - v) * f));
    return '#' + c.map((v) => v.toString(16).padStart(2, '0')).join('');
  }
  function luminance(hx) {
    const c = hexToRgb(hx).map((v) => {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }

  // ------------------------------------------------------------ series tokens
  (function injectSeriesTokens() {
    const light = SYS.map((s, i) => '--s' + i + ':' + s.color_light + ';').join('');
    const dark = SYS.map((s, i) => '--s' + i + ':' + (s.color_dark || s.color_light) + ';').join('');
    const css = ':root{' + light + '}' +
      '@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){' + dark + '}}' +
      ':root[data-theme="dark"]{' + dark + '}';
    const st = document.createElement('style');
    st.id = 'px-series-tokens';
    st.textContent = css;
    document.head.appendChild(st);
  })();

  // ------------------------------------------------------------ URL state
  function readHash() {
    let q;
    try { q = new URLSearchParams(location.hash.replace(/^#/, '')); } catch (e) { return; }
    const f = q.get('f');
    if (f && FAMS[f]) state.fam = f;
    const p = q.get('p');
    if (p && FAMS[state.fam].ids.indexOf(p) >= 0) state.sel[state.fam] = p;
    if (KINDS[q.get('k')]) state.kind = q.get('k');
    if (VIEWS[q.get('v')]) state.view = q.get('v');
    if (YMODES[q.get('y')]) state.y = q.get('y');
    if (BANDS[q.get('b')]) state.band = q.get('b');
    const sh = q.get('sh');
    if (sh && (sh in SYS_I || sh === 'none' || sh === 'avg' || sh === 'range')) state.shade = sh;
    if (q.has('s')) {
      const s = q.get('s');
      state.on = new Set(s === 'all' ? SYS.map((x) => x.key) : s.split(',').filter((k) => k in SYS_I));
    }
    fixKind();
  }
  function writeHash() {
    const q = new URLSearchParams();
    q.set('f', state.fam);
    q.set('p', curId());
    q.set('k', state.kind);
    q.set('v', state.view);
    q.set('y', state.y);
    q.set('b', state.band);
    q.set('sh', state.shade);
    q.set('s', state.on.size === SYS.length ? 'all' : shownSystems().map((s) => s.key).join(','));
    const hs = '#' + q.toString();
    if (hs === location.hash) return;
    try { history.replaceState(null, '', hs); } catch (e) { /* some browsers refuse on file:// */ }
  }
  function fixKind() {
    if (FAMS[state.fam].kinds.indexOf(state.kind) < 0) state.kind = 'mean';
  }

  // ------------------------------------------------------------ controls
  const segButtons = {};
  function buildSeg(boxId, items, onPick) {
    const box = $(boxId);
    box.innerHTML = '';
    segButtons[boxId] = items.map((it) => {
      const b = h('button', { type: 'button', 'data-v': it[0], title: it[2] || null, text: it[1] });
      b.dataset.title0 = it[2] || '';
      b.addEventListener('click', () => {
        if (b.getAttribute('aria-disabled') === 'true') return;
        onPick(it[0]);
      });
      box.appendChild(b);
      return b;
    });
  }
  function syncSeg(boxId, cur, disabled) {
    for (const b of segButtons[boxId]) {
      const v = b.getAttribute('data-v');
      b.setAttribute('aria-pressed', String(v === cur));
      const dis = disabled && disabled[v];
      if (dis) {
        b.setAttribute('aria-disabled', 'true');
        b.title = dis;
      } else {
        b.removeAttribute('aria-disabled');
        if (b.dataset.title0) b.title = b.dataset.title0; else b.removeAttribute('title');
      }
    }
  }
  const chipEls = {};
  function buildControls() {
    buildSeg('px-fam', FAM_ORDER.map((f) => [f, FAMS[f].name]), (v) => {
      if (v === state.fam) return;
      state.fam = v;
      fixKind();
      hoverR = null;
      renderAll();
    });
    buildSeg('px-kind', Object.keys(KINDS).map((k) => [k, KINDS[k].name]), (v) => {
      state.kind = v;
      renderAll();
    });
    buildSeg('px-view', Object.keys(VIEWS).map((k) => [k, VIEWS[k]]), (v) => {
      state.view = v;
      renderAll();
    });
    buildSeg('px-y', Object.keys(YMODES).map((k) => [k, YMODES[k], YMODE_TITLES[k]]), (v) => {
      state.y = v;
      renderAll();
    });

    const box = $('px-systems');
    box.innerHTML = '';
    SYS.forEach((s, i) => {
      const b = h('button', {
        type: 'button', class: 'px-chip', 'data-key': s.key,
        title: s.sequence + ' · residue 6 = ' +
          (s.x_by_peptide && s.x_by_peptide[0] !== s.x_by_peptide[1]
            ? s.x_by_peptide[0] + ' on peptide 1, ' + s.x_by_peptide[1] + ' on peptide 2'
            : s.x_resname) +
          ' (' + s.note + '). Double-click to show only this system.',
      });
      b.style.setProperty('--c', color(i));
      b.appendChild(h('span', { class: 'px-sw', 'aria-hidden': 'true' }));
      b.appendChild(document.createTextNode(s.label));
      b.addEventListener('click', () => {
        if (state.on.has(s.key)) state.on.delete(s.key); else state.on.add(s.key);
        renderAll();
      });
      b.addEventListener('dblclick', (e) => {
        e.preventDefault();
        state.on = new Set([s.key]);
        renderAll();
      });
      chipEls[s.key] = b;
      box.appendChild(b);
    });
    const pre = h('span', { class: 'px-presets', role: 'group', 'aria-label': 'System presets' });
    for (const p of PRESETS) {
      const b = h('button', { type: 'button', class: 'px-btn', title: p.title, text: p.name });
      b.addEventListener('click', () => {
        state.on = new Set(p.keys);
        renderAll();
      });
      pre.appendChild(b);
    }
    box.appendChild(pre);

    const sel = $('px-shade');
    sel.innerHTML = '';
    const opts = [['none', 'No shading'], ['avg', 'Mean of shown systems'], ['range', 'Spread across shown (max − min)']]
      .concat(SYS.map((s) => [s.key, s.label]));
    for (const o of opts) sel.appendChild(h('option', { value: o[0], text: o[1] }));
    sel.addEventListener('change', () => {
      state.shade = sel.value;
      renderAll();
    });
  }
  function syncControls() {
    syncSeg('px-fam', state.fam);
    const kindDis = {};
    for (const k of Object.keys(KINDS)) {
      if (FAMS[state.fam].kinds.indexOf(k) < 0) kindDis[k] = 'Whole-trajectory PMFs exist only for residue pairs';
    }
    syncSeg('px-kind', state.kind, kindDis);
    syncSeg('px-view', state.view);
    syncSeg('px-y', state.y);
    for (const s of SYS) chipEls[s.key].setAttribute('aria-pressed', String(state.on.has(s.key)));
    $('px-shade').value = state.shade;
  }

  // ------------------------------------------------------------ matrix
  function wellOf(sys, fam, id) {
    const b = block(sys.key, fam, id);
    if (!b || !b.well) return null;
    const w = (state.kind === 'full' && b.well.full) ? b.well.full : b.well.mean;
    return w && w[0] != null ? w[0] : null;
  }
  function shadeValues() {
    const out = {};
    const fam = state.fam;
    const shown = shownSystems();
    for (const id of FAMS[fam].ids) {
      let v = null;
      if (state.shade === 'avg' || state.shade === 'range') {
        const vs = shown.map((s) => wellOf(s, fam, id)).filter((x) => x != null);
        if (vs.length) {
          v = state.shade === 'avg' ? vs.reduce((a, b) => a + b, 0) / vs.length
            : Math.max.apply(null, vs) - Math.min.apply(null, vs);
        }
      } else if (state.shade in BY_KEY) {
        v = wellOf(BY_KEY[state.shade], fam, id);
      }
      out[id] = v;
    }
    return out;
  }
  function rampSpec(vals) {
    if (state.shade === 'none') return null;
    if (state.shade === 'range') {
      let mx = 0;
      for (const k in vals) if (vals[k] != null) mx = Math.max(mx, vals[k]);
      const hi = Math.max(0.25, ceilTo(mx, 0.25));
      return { kind: 'seq', stops: isDark() ? SEQ_DARK : SEQ_LIGHT, lo: 0, hi: hi };
    }
    return { kind: 'div', stops: isDark() ? DIV_DARK : DIV_LIGHT, lo: -DIV_LIMIT, hi: DIV_LIMIT };
  }
  function shadeLabel() {
    const kind = state.kind === 'full' ? 'whole-trajectory' : 'mean-of-halves';
    if (state.shade === 'avg') return 'mean ' + kind + ' well depth across shown systems';
    if (state.shade === 'range') return 'spread (max − min) of the ' + kind + ' well depth across shown systems';
    if (state.shade in BY_KEY) return BY_KEY[state.shade].label + ' ' + kind + ' well depth';
    return '';
  }
  function renderMatrix(focusSel) {
    const box = $('px-matrix');
    box.innerHTML = '';
    const fam = state.fam;
    const vals = shadeValues();
    const ramp = rampSpec(vals);
    const cap = $('px-axis-cap');
    const cells = [];

    function cell(id, label) {
      const v = vals[id];
      const b = h('button', { type: 'button', class: 'px-cell', 'data-id': id });
      const selected = id === curId();
      b.setAttribute('aria-pressed', String(selected));
      b.tabIndex = selected ? 0 : -1;
      let lab = label;
      if (ramp && v != null) {
        const t = (v - ramp.lo) / (ramp.hi - ramp.lo);
        const bg = lerpStops(ramp.stops, t);
        b.style.background = bg;
        b.style.borderColor = 'transparent';
        b.style.color = luminance(bg) > 0.33 ? '#0b0b0b' : '#fcfcfb';
        b.textContent = fmt(v, 2);
        lab += ', ' + shadeLabel() + ' ' + fmt(v, 2) + ' kcal/mol';
      } else if (fam === 'fragment') {
        b.textContent = label;
      }
      b.setAttribute('aria-label', lab);
      b.title = lab;
      b.addEventListener('click', () => select(id));
      cells.push(b);
      return b;
    }

    if (fam === 'pairs') {
      box.style.gridTemplateColumns = '30px repeat(6, minmax(0, 1fr))';
      box.appendChild(h('span', { class: 'px-mh', 'aria-hidden': 'true' }));
      for (let j = 1; j <= 6; j++) box.appendChild(h('span', { class: 'px-mh', 'aria-hidden': 'true', text: 'LIVAGX'[j - 1] + j }));
      for (let i = 1; i <= 6; i++) {
        box.appendChild(h('span', { class: 'px-mh px-row', 'aria-hidden': 'true', text: 'LIVAGX'[i - 1] + i }));
        for (let j = 1; j <= 6; j++) {
          const id = i + '-' + j;
          box.appendChild(cell(id, idLabel(fam, id) + ' (peptide 1 ' + 'LIVAGX'[i - 1] + i + ', peptide 2 ' + 'LIVAGX'[j - 1] + j + ')'));
        }
      }
      cap.textContent = 'Rows: position on peptide 1 · columns: position on peptide 2 · X = residue 6.';
    } else if (fam === 'windows') {
      box.style.gridTemplateColumns = '34px repeat(4, minmax(0, 1fr))';
      box.appendChild(h('span', { class: 'px-mh', 'aria-hidden': 'true' }));
      for (let b = 1; b <= 4; b++) box.appendChild(h('span', { class: 'px-mh', 'aria-hidden': 'true', text: WNAMES[b - 1] }));
      for (let a = 1; a <= 4; a++) {
        box.appendChild(h('span', { class: 'px-mh px-row', 'aria-hidden': 'true', text: WNAMES[a - 1] }));
        for (let b = 1; b <= 4; b++) {
          const id = a + '-' + b;
          if (FAMS.windows.ids.indexOf(id) >= 0) box.appendChild(cell(id, idLabel(fam, id)));
          else box.appendChild(h('span', { class: 'px-cell-empty', 'aria-hidden': 'true' }));
        }
      }
      cap.textContent = 'Upper triangle: 10 unordered window pairs (both orderings pooled).';
    } else {
      box.style.gridTemplateColumns = 'minmax(0, 1fr)';
      const b = cell('LIVA', 'LIVA' + NDASH + 'LIVA');
      b.classList.add('px-frag-cell');
      if (ramp && vals.LIVA != null) b.textContent = 'LIVA' + NDASH + 'LIVA  ·  ' + fmt(vals.LIVA, 2) + ' kcal/mol';
      box.appendChild(b);
      cap.textContent = 'One PMF: residues :2' + NDASH + '5 vs :10' + NDASH + '13.';
    }
    renderRamp(ramp);
    if (focusSel) {
      const s = cells.find((c) => c.getAttribute('data-id') === curId());
      if (s) s.focus();
    }
  }
  function renderRamp(ramp) {
    const box = $('px-ramp-box');
    box.innerHTML = '';
    if (!ramp) return;
    const n = 12;
    const cols = [];
    for (let k = 0; k <= n; k++) cols.push(lerpStops(ramp.stops, k / n));
    const bar = h('div', { class: 'px-ramp', role: 'img', 'aria-label': 'Colour scale from ' + fmt(ramp.lo, 2) + ' to ' + fmt(ramp.hi, 2) + ' kcal/mol' });
    bar.style.background = 'linear-gradient(to right, ' + cols.join(', ') + ')';
    box.appendChild(bar);
    const tk = h('div', { class: 'px-ramp-ticks', 'aria-hidden': 'true' });
    if (ramp.kind === 'div') {
      tk.innerHTML = '<span>≤ ' + fmt(ramp.lo, 1) + '</span><span>0</span><span>≥ +' + fmt(ramp.hi, 1) + '</span>';
    } else {
      tk.innerHTML = '<span>0</span><span>' + fmt(ramp.hi, 2) + '</span>';
    }
    box.appendChild(tk);
    const cap = ramp.kind === 'div'
      ? 'Well depth = minimum of the PMF (kcal/mol); blue = attractive. ' + capFirst(shadeLabel()) + '.'
      : capFirst(shadeLabel()) + ' (kcal/mol); ' + (isDark() ? 'brighter' : 'darker') + ' = systems disagree more.';
    box.appendChild(h('p', { class: 'px-ramp-cap', text: cap }));
  }
  function capFirst(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
  function select(id) {
    if (state.sel[state.fam] === id) return;
    state.sel[state.fam] = id;
    renderAll(true);
  }
  function onMatrixKey(e) {
    const t = e.target;
    if (!t || !t.classList || !t.classList.contains('px-cell')) return;
    if (state.fam === 'fragment') return;
    const id = t.getAttribute('data-id');
    let [a, b] = parseId(id);
    const n = state.fam === 'pairs' ? 6 : 4;
    const k = e.key;
    if (k === 'ArrowLeft') b -= 1;
    else if (k === 'ArrowRight') b += 1;
    else if (k === 'ArrowUp') a -= 1;
    else if (k === 'ArrowDown') a += 1;
    else if (k === 'Home') b = state.fam === 'pairs' ? 1 : a;
    else if (k === 'End') b = n;
    else return;
    e.preventDefault();
    a = clamp(a, 1, n);
    b = clamp(b, 1, n);
    const nid = a + '-' + b;
    if (FAMS[state.fam].ids.indexOf(nid) < 0) return;
    state.sel[state.fam] = nid;
    renderAll(true);
  }

  // ------------------------------------------------------------ SVG chart
  let uid = 0;
  function linePath(c, sx, sy) {
    let d = '';
    let pen = false;
    for (let k = 0; k < c.r.length; k++) {
      const v = c.y[k];
      if (v == null) { pen = false; continue; }
      d += (pen ? 'L' : 'M') + sx(c.r[k]).toFixed(1) + ' ' + sy(v).toFixed(1);
      pen = true;
    }
    return d;
  }
  function bandPath(c, sx, sy) {
    if (!c.b) return '';
    let d = '';
    let seg = [];
    const flush = () => {
      if (seg.length > 1) {
        d += 'M' + seg.map((k) => sx(c.r[k]).toFixed(1) + ' ' + sy(c.y[k] + c.b[k]).toFixed(1)).join('L');
        d += 'L' + seg.slice().reverse().map((k) => sx(c.r[k]).toFixed(1) + ' ' + sy(c.y[k] - c.b[k]).toFixed(1)).join('L') + 'Z';
      }
      seg = [];
    };
    for (let k = 0; k < c.r.length; k++) {
      if (c.y[k] == null || c.b[k] == null) flush(); else seg.push(k);
    }
    flush();
    return d;
  }
  function lastPoint(c, xd) {
    for (let k = c.r.length - 1; k >= 0; k--) {
      if (c.y[k] != null && c.r[k] <= xd[1] + 1e-9) return [c.r[k], c.y[k]];
    }
    return null;
  }
  function dodge(items, lo, hi, gap) {
    items.sort((a, b) => a.y - b.y);
    for (let k = 0; k < items.length; k++) {
      items[k].ly = Math.max(items[k].y, k ? items[k - 1].ly + gap : lo);
    }
    const over = items.length ? items[items.length - 1].ly - hi : 0;
    if (over > 0) {
      for (let k = items.length - 1; k >= 0; k--) {
        const cap = k === items.length - 1 ? hi : items[k + 1].ly - gap;
        items[k].ly = Math.max(lo, Math.min(items[k].ly, cap));
      }
    }
    return items;
  }

  /* cfg: {w, h, m, xd, yd, traces, context, lineW, markers, bandOpacity,
           xTicks, yTicks, xTitle, yTitle, directLabels, focusable, aria,
           step, onHover(r, evt)} */
  function drawChart(host, cfg) {
    const w = cfg.w;
    const hh = cfg.h;
    const m = cfg.m;
    const xd = cfg.xd;
    const yd = cfg.yd;
    const pw = Math.max(10, w - m.l - m.r);
    const ph = Math.max(10, hh - m.t - m.b);
    const sx = (r) => m.l + (r - xd[0]) / (xd[1] - xd[0]) * pw;
    const sy = (v) => m.t + (yd[1] - v) / (yd[1] - yd[0]) * ph;
    const inv = (px) => xd[0] + (px - m.l) / pw * (xd[1] - xd[0]);

    const svg = el('svg', {
      class: 'px-svg', viewBox: '0 0 ' + w + ' ' + hh, width: w, height: hh,
      role: 'img', 'aria-label': cfg.aria || 'PMF chart', tabindex: cfg.focusable ? '0' : null,
    });
    const fs = cfg.compact ? 10.5 : 11.5;

    // grid (solid hairlines), zero line, axes
    const gGrid = el('g', { 'shape-rendering': 'crispEdges' }, svg);
    const yt = ticks(yd[0], yd[1], cfg.yTicks || 6, yd[2]);
    const xt = ticks(xd[0], xd[1], cfg.xTicks || 8);
    const ydp = decimalsFor(yt.step);
    const xdp = decimalsFor(xt.step);
    for (const v of yt.values) {
      el('line', { x1: m.l, x2: m.l + pw, y1: Math.round(sy(v)) + 0.5, y2: Math.round(sy(v)) + 0.5, style: 'stroke:var(--grid)', 'stroke-width': 1 }, gGrid);
    }
    for (const v of xt.values) {
      el('line', { x1: Math.round(sx(v)) + 0.5, x2: Math.round(sx(v)) + 0.5, y1: m.t, y2: m.t + ph, style: 'stroke:var(--grid)', 'stroke-width': 1 }, gGrid);
    }
    if (yd[0] < 0 && yd[1] > 0) {
      el('line', { x1: m.l, x2: m.l + pw, y1: Math.round(sy(0)) + 0.5, y2: Math.round(sy(0)) + 0.5, style: 'stroke:var(--muted)', 'stroke-width': 1 }, gGrid);
    }
    el('line', { x1: m.l, x2: m.l + pw, y1: m.t + ph + 0.5, y2: m.t + ph + 0.5, style: 'stroke:var(--axis)', 'stroke-width': 1 }, gGrid);
    el('line', { x1: m.l - 0.5, x2: m.l - 0.5, y1: m.t, y2: m.t + ph, style: 'stroke:var(--axis)', 'stroke-width': 1 }, gGrid);

    const gTxt = el('g', { 'font-size': fs, style: 'fill:var(--ink-2)' }, svg);
    for (const v of yt.values) {
      const t = el('text', { x: m.l - 6, y: sy(v), 'text-anchor': 'end', 'dominant-baseline': 'middle' }, gTxt);
      t.textContent = fmt(v, ydp);
    }
    for (const v of xt.values) {
      const t = el('text', { x: sx(v), y: m.t + ph + fs + 5, 'text-anchor': 'middle' }, gTxt);
      t.textContent = v.toFixed(xdp);
    }
    if (cfg.xTitle) {
      const t = el('text', { x: m.l + pw / 2, y: hh - 4, 'text-anchor': 'middle', 'font-size': fs + 0.5 }, gTxt);
      t.textContent = cfg.xTitle;
    }
    if (cfg.yTitle) {
      const t = el('text', {
        x: 0, y: 0, 'text-anchor': 'middle', 'font-size': fs + 0.5,
        transform: 'translate(' + (fs + 1) + ' ' + (m.t + ph / 2) + ') rotate(-90)',
      }, gTxt);
      t.textContent = cfg.yTitle;
    }

    // data, clipped to the plot area
    const cid = 'px-clip-' + (++uid);
    const defs = el('defs', null, svg);
    const cp = el('clipPath', { id: cid }, defs);
    el('rect', { x: m.l, y: m.t - 1, width: pw, height: ph + 2 }, cp);
    const gData = el('g', { 'clip-path': 'url(#' + cid + ')' }, svg);
    for (const t of cfg.context || []) {
      if (!t.c) continue;
      el('path', { d: linePath(t.c, sx, sy), fill: 'none', style: 'stroke:var(--context)', 'stroke-width': 1, 'stroke-linejoin': 'round' }, gData);
    }
    for (const t of cfg.traces) {
      if (state.band === 'off' || !t.band || !t.c || !t.c.b) continue;
      el('path', { d: bandPath(t.c, sx, sy), style: 'fill:' + t.color, 'fill-opacity': cfg.bandOpacity || 0.18, stroke: 'none' }, gData);
    }
    for (const t of cfg.traces) {
      if (!t.c) continue;
      el('path', {
        d: linePath(t.c, sx, sy), fill: 'none', style: 'stroke:' + t.color, 'stroke-width': cfg.lineW || 2,
        'stroke-dasharray': t.dash ? '5 4' : null, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
      }, gData);
    }
    if (cfg.markers) {
      for (const t of cfg.traces) {
        const wl = t.well;
        if (!wl || wl[0] == null || wl[0] > -0.02 || wl[1] < xd[0] || wl[0] < yd[0]) continue;
        el('circle', { cx: sx(wl[1]), cy: sy(wl[0]), r: cfg.compact ? 2.6 : 3.2, style: 'fill:' + t.color + ';stroke:var(--surface)', 'stroke-width': 1.2 }, gData);
      }
    }

    // direct labels at the line ends, dodged in the right margin
    if (cfg.directLabels && cfg.directLabels.length) {
      const gL = el('g', { 'font-size': fs }, svg);
      const items = [];
      for (const d of cfg.directLabels) {
        const p = d.c ? lastPoint(d.c, xd) : null;
        if (!p) continue;
        const y = clamp(sy(p[1]), m.t + 4, m.t + ph - 4);
        items.push({ x: sx(p[0]), y: y, text: d.text, color: d.color });
      }
      dodge(items, m.t + 6, m.t + ph - 6, fs + 5);
      const x0 = m.l + pw + 8;
      for (const it of items) {
        el('polyline', {
          points: (it.x + 2) + ',' + it.y + ' ' + (x0 + 2) + ',' + it.ly + ' ' + (x0 + 6) + ',' + it.ly,
          fill: 'none', style: 'stroke:var(--axis)', 'stroke-width': 1,
        }, gL);
        el('line', { x1: x0 + 8, x2: x0 + 20, y1: it.ly, y2: it.ly, style: 'stroke:' + it.color, 'stroke-width': 2.5, 'stroke-linecap': 'round' }, gL);
        const t = el('text', { x: x0 + 25, y: it.ly, 'dominant-baseline': 'middle', style: 'fill:var(--ink)' }, gL);
        t.textContent = it.text;
      }
    }

    // crosshair
    const gX = el('g', { 'pointer-events': 'none' }, svg);
    gX.style.display = 'none';
    const xl = el('line', { y1: m.t, y2: m.t + ph, style: 'stroke:var(--ink-2)', 'stroke-width': 1 }, gX);
    const dots = cfg.traces.map((t) => el('circle', { r: cfg.compact ? 3 : 3.6, style: 'fill:' + t.color + ';stroke:var(--surface)', 'stroke-width': 1.5 }, gX));
    function setCrosshair(r) {
      if (r == null) { gX.style.display = 'none'; return; }
      gX.style.display = '';
      const x = sx(r);
      xl.setAttribute('x1', x);
      xl.setAttribute('x2', x);
      cfg.traces.forEach((t, k) => {
        const v = valueAt(t.c, r);
        if (v == null || v < yd[0] || v > yd[1]) { dots[k].style.display = 'none'; return; }
        dots[k].style.display = '';
        dots[k].setAttribute('cx', x);
        dots[k].setAttribute('cy', sy(v));
      });
    }

    // pointer + keyboard
    const hit = el('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent', style: 'cursor:crosshair' }, svg);
    const toR = (e) => {
      const rect = svg.getBoundingClientRect();
      const px = (e.clientX - rect.left) * (w / (rect.width || w));
      return clamp(inv(px), xd[0], xd[1]);
    };
    if (cfg.onHover) {
      hit.addEventListener('pointermove', (e) => cfg.onHover(toR(e), e));
      hit.addEventListener('pointerdown', (e) => cfg.onHover(toR(e), e));
      hit.addEventListener('pointerleave', (e) => { if (e.pointerType !== 'touch') cfg.onHover(null, e); });
      if (cfg.focusable) {
        const step = cfg.step || 0.01;
        svg.addEventListener('keydown', (e) => {
          let r = hoverR == null ? (xd[0] + xd[1]) / 2 : hoverR;
          if (e.key === 'ArrowLeft') r -= step * (e.shiftKey ? 5 : 1);
          else if (e.key === 'ArrowRight') r += step * (e.shiftKey ? 5 : 1);
          else if (e.key === 'Home') r = xd[0];
          else if (e.key === 'End') r = xd[1];
          else if (e.key === 'Escape') { cfg.onHover(null, null); return; }
          else return;
          e.preventDefault();
          cfg.onHover(clamp(r, xd[0], xd[1]), null);
        });
        svg.addEventListener('focus', () => {
          if (hoverR == null) {
            const t0 = cfg.traces[0];
            const r0 = t0 && t0.well && t0.well[1] != null ? t0.well[1] : (xd[0] + xd[1]) / 2;
            cfg.onHover(clamp(r0, xd[0], xd[1]), null);
          }
        });
        svg.addEventListener('blur', () => cfg.onHover(null, null));
      }
    }
    host.appendChild(svg);
    return { svg: svg, setCrosshair: setCrosshair, sx: sx, sy: sy, m: m, pw: pw, ph: ph, w: w, h: hh };
  }

  // ------------------------------------------------------------ chart views
  function ariaSummary(shown, traces, idLab) {
    const parts = shown.map((s) => {
      const t = traces.find((x) => x.sys === s);
      return s.label + ' ' + (t ? wellText(t.well, false) : 'no data');
    });
    return 'PMF (kcal/mol) versus distance (nm) for ' + idLab + ', ' + KINDS[state.kind].short + '. ' + parts.join('; ') + '.';
  }
  function tipRows(shown, r) {
    const fam = state.fam;
    const id = curId();
    let rows = '';
    for (const s of shown) {
      const tr = tracesFor(s, fam, id, state.kind);
      let val;
      if (state.kind === 'halves') {
        const a = tr.find((t) => t.part === 'h1');
        const b = tr.find((t) => t.part === 'h2');
        val = fmt(valueAt(a && a.c, r)) + ' <span class="px-u">/</span> ' + fmt(valueAt(b && b.c, r));
      } else if (state.kind === 'mean') {
        const t = tr[0];
        const v = valueAt(t && t.c, r);
        const bb = t && t.c && t.c.b ? valueAt(t.c, r, t.c.b) : null;
        val = fmt(v) + (v != null && bb != null ? ' <span class="px-u">± ' + fmt(bb) + '</span>' : '');
      } else {
        const t = tr[0];
        val = fmt(valueAt(t && t.c, r));
      }
      rows += '<tr><td><span class="px-sw-dot" style="--c:' + color(SYS_I[s.key]) + '"></span></td>' +
        '<td class="px-l">' + esc(s.label) + '</td><td class="px-v">' + val + '</td></tr>';
    }
    const what = state.kind === 'halves' ? 'half 1 / half 2' : state.kind === 'mean' ? 'mean ± band' : 'full trajectory';
    return '<div class="px-tip-h">r = ' + r.toFixed(3) + ' nm · ' + what + ' (kcal/mol)</div><table>' + rows + '</table>';
  }

  function renderOverlay(host) {
    const fam = state.fam;
    const id = curId();
    const shown = shownSystems();
    const traces = [];
    for (const s of shown) for (const t of tracesFor(s, fam, id, state.kind)) traces.push(t);
    if (!traces.length) {
      host.appendChild(h('div', { class: 'px-empty', text: 'No systems selected — pick at least one system above.' }));
      return;
    }
    const W = Math.max(280, Math.floor(host.clientWidth || 640));
    const direct = W >= 560 && shown.length <= 4;
    const H = Math.round(clamp(W * 0.56, 270, 470));
    const m = { t: 12, r: direct ? 136 : 14, b: 46, l: 56 };
    const xd = FAMS[fam].x;
    const yd = yDomain(traces);
    const tip = h('div', { class: 'px-tip', role: 'status' });
    tip.hidden = true;
    let ctl = null;
    const onHover = (r, evt) => {
      hoverR = r;
      ctl.setCrosshair(r);
      if (r == null) { tip.hidden = true; return; }
      tip.innerHTML = tipRows(shown, r);
      tip.hidden = false;
      const scale = (ctl.svg.getBoundingClientRect().width || W) / W;
      const hostRect = host.getBoundingClientRect();
      const x = ctl.sx(r) * scale;
      const tw = tip.offsetWidth;
      const th = tip.offsetHeight;
      let left = x + 14;
      if (left + tw > hostRect.width - 2) left = x - 14 - tw;
      left = clamp(left, 0, Math.max(0, hostRect.width - tw));
      let top = evt && evt.clientY != null ? evt.clientY - hostRect.top - th / 2 : ctl.m.t * scale + 6;
      top = clamp(top, 0, Math.max(0, hostRect.height - th));
      tip.style.left = left + 'px';
      tip.style.top = top + 'px';
    };
    const firstOf = (s) => traces.find((t) => t.sys === s);
    ctl = drawChart(host, {
      w: W, h: H, m: m, xd: xd, yd: yd, traces: traces, lineW: 2,
      bandOpacity: shown.length <= 4 ? 0.18 : 0.1,
      markers: state.kind !== 'halves',
      xTicks: W < 480 ? 4 : 8, yTicks: Y_TICKS,
      xTitle: 'distance (nm)', yTitle: 'PMF (kcal/mol)',
      directLabels: direct ? shown.map((s) => { const t = firstOf(s); return t ? { c: t.c, text: s.label, color: t.color } : null; }).filter(Boolean) : null,
      focusable: true, step: FAMS[fam].bin,
      aria: ariaSummary(shown, traces, idLabel(fam, id)),
      onHover: onHover,
    });
    host.appendChild(tip);
    if (hoverR != null) onHover(hoverR, null);
  }

  function readout(tr, r) {
    if (r == null) {
      const t = tr[0];
      if (!t) return 'no data';
      if (state.kind === 'halves') {
        const b = tr[1];
        const f = (w) => (w && w[0] != null && w[0] <= -0.02 ? fmt(w[0]) : NDASH);
        return 'min ' + f(t.well) + ' / ' + f(b && b.well) + ' kcal/mol';
      }
      return wellText(t.well, state.kind === 'mean');
    }
    if (state.kind === 'halves') {
      return 'r ' + r.toFixed(2) + ' nm: ' + fmt(valueAt(tr[0] && tr[0].c, r)) + ' / ' + fmt(valueAt(tr[1] && tr[1].c, r)) + ' kcal/mol';
    }
    const t = tr[0];
    const v = valueAt(t && t.c, r);
    const bb = state.kind === 'mean' && t && t.c && t.c.b ? valueAt(t.c, r, t.c.b) : null;
    return 'r ' + r.toFixed(2) + ' nm: ' + fmt(v) + (bb != null && v != null ? ' ± ' + fmt(bb) : '') + ' kcal/mol';
  }

  function renderMultiples(host) {
    const fam = state.fam;
    const id = curId();
    const shown = shownSystems();
    if (!shown.length) {
      host.appendChild(h('div', { class: 'px-empty', text: 'No systems selected — pick at least one system above.' }));
      return;
    }
    const all = shown.map((s) => ({ s: s, tr: tracesFor(s, fam, id, state.kind) }));
    const allTraces = [];
    for (const a of all) for (const t of a.tr) allTraces.push(t);
    const yd = yDomain(allTraces);
    const xd = FAMS[fam].x;
    const ctxKind = state.kind === 'full' ? 'full' : 'mean';
    const grid = h('div', { class: 'px-mult' });
    host.appendChild(grid);
    const panels = all.map((a) => {
      const p = h('div', { class: 'px-panel' });
      const head = h('div', { class: 'px-panel-h' });
      const sw = h('span', { class: 'px-sw-dot', 'aria-hidden': 'true' });
      sw.style.setProperty('--c', color(SYS_I[a.s.key]));
      head.appendChild(sw);
      head.appendChild(h('span', { class: 'px-pl', text: a.s.label }));
      head.appendChild(h('span', { class: 'px-pid', text: idLabel(fam, id, a.s) }));
      const ro = h('span', { class: 'px-ro', text: readout(a.tr, null) });
      head.appendChild(ro);
      p.appendChild(head);
      const box = h('div');
      p.appendChild(box);
      grid.appendChild(p);
      return { a: a, box: box, ro: ro };
    });
    const charts = [];
    const sync = (r) => {
      hoverR = r;
      charts.forEach((c, k) => {
        c.setCrosshair(r);
        panels[k].ro.textContent = readout(panels[k].a.tr, r);
      });
    };
    for (const p of panels) {
      const W = Math.max(200, Math.floor(p.box.clientWidth || 260));
      const context = [];
      for (const o of shown) {
        if (o === p.a.s) continue;
        for (const t of tracesFor(o, fam, id, ctxKind)) context.push(t);
      }
      charts.push(drawChart(p.box, {
        w: W, h: Math.round(clamp(W * 0.68, 170, 230)), m: { t: 8, r: 10, b: 38, l: 46 },
        xd: xd, yd: yd, traces: p.a.tr, context: context, lineW: 1.75, bandOpacity: 0.2,
        markers: state.kind !== 'halves', compact: true, xTicks: 4, yTicks: 4,
        xTitle: 'distance (nm)', yTitle: 'PMF (kcal/mol)',
        focusable: true, step: FAMS[fam].bin,
        aria: ariaSummary([p.a.s], p.a.tr, idLabel(fam, id, p.a.s)),
        onHover: sync,
      }));
    }
    if (hoverR != null) sync(hoverR);
  }

  function tableColumns(shown) {
    const fam = state.fam;
    const id = curId();
    const cols = [];
    for (const s of shown) {
      const tr = tracesFor(s, fam, id, state.kind);
      if (state.kind === 'halves') {
        for (const t of tr) cols.push({ head: s.label + (t.part === 'h1' ? ' half 1' : ' half 2'), key: s.key + '_' + t.part, t: t });
      } else if (state.kind === 'mean') {
        if (tr[0]) {
          cols.push({ head: s.label + ' mean', key: s.key + '_mean', t: tr[0] });
          cols.push({ head: '± band', key: s.key + '_band', t: tr[0], band: true });
        }
      } else if (tr[0]) {
        cols.push({ head: s.label + ' full', key: s.key + '_full', t: tr[0] });
      }
    }
    return cols;
  }
  function tableGrid(cols) {
    const step = FAMS[state.fam].bin;
    const xd = FAMS[state.fam].x;
    let lo = Infinity;
    let hi = -Infinity;
    for (const c of cols) {
      const r = c.t.c.r;
      if (r.length) { lo = Math.min(lo, r[0]); hi = Math.max(hi, r[r.length - 1]); }
    }
    if (!isFinite(lo)) return [];
    const k0 = Math.ceil(Math.max(lo, xd[0]) / step - 1e-6);
    const k1 = Math.floor(Math.min(hi, xd[1]) / step + 1e-6);
    const out = [];
    for (let k = k0; k <= k1; k++) out.push(Number((k * step).toFixed(4)));
    return out;
  }
  function renderTable(host) {
    const fam = state.fam;
    const id = curId();
    const shown = shownSystems();
    if (!shown.length) {
      host.appendChild(h('div', { class: 'px-empty', text: 'No systems selected — pick at least one system above.' }));
      return;
    }
    const cols = tableColumns(shown);
    const grid = tableGrid(cols);
    const dpR = decimalsFor(FAMS[fam].bin);

    const bar = h('div', { class: 'px-tablebar' });
    const dl = h('button', { type: 'button', class: 'px-btn', text: 'Download CSV' });
    dl.addEventListener('click', () => downloadCSV(cols, grid, dpR));
    bar.appendChild(dl);
    bar.appendChild(h('span', { class: 'px-note', text: 'Values linearly interpolated onto a common ' + FAMS[fam].bin + ' nm grid (each system has its own bin centres).' }));
    host.appendChild(bar);

    // well summary
    const wrap1 = h('div', { class: 'px-tscroll' });
    const t1 = h('table', { class: 'px-table' });
    t1.appendChild(h('caption', {
      text: 'Well depth (global minimum of the PMF, kcal/mol) and its position, ' + idLabel(fam, id) +
        '. Negative = attractive; a minimum above ' + MINUS + '0.02 kcal/mol next to 1.8 nm means there is no attractive well.',
    }));
    const hasFull = fam === 'pairs';
    let head = '<tr><th scope="col">System</th><th scope="col">X (Amber)</th><th scope="col">mean depth (kcal/mol)</th>' +
      '<th scope="col">r<sub>min</sub> (nm)</th><th scope="col">± band at r<sub>min</sub></th>' +
      '<th scope="col">half 1 depth</th><th scope="col">half 2 depth</th>' + (hasFull ? '<th scope="col">full depth</th><th scope="col">full r<sub>min</sub></th>' : '') + '</tr>';
    let body = '';
    for (const s of shown) {
      const b = block(s.key, fam, id);
      const w = (b && b.well) || {};
      const td = (v, d) => '<td' + (v == null ? ' class="px-na"' : '') + '>' + fmt(v, d) + '</td>';
      const g = (k, q) => (w[k] ? w[k][q] : null);
      body += '<tr><th scope="row" style="text-align:left;font-weight:500;color:var(--ink)"><span class="px-sw-dot" style="--c:' + color(SYS_I[s.key]) + '"></span> ' + esc(s.label) + '</th>' +
        '<td>' + esc(s.x_resname) + '</td>' + td(g('mean', 0), 3) + td(g('mean', 1), 3) + td(g('mean', 2), 3) +
        td(g('h1', 0), 3) + td(g('h2', 0), 3) + (hasFull ? td(g('full', 0), 3) + td(g('full', 1), 3) : '') + '</tr>';
    }
    t1.insertAdjacentHTML('beforeend', '<thead>' + head + '</thead><tbody>' + body + '</tbody>');
    wrap1.appendChild(t1);
    host.appendChild(wrap1);

    // values
    const wrap2 = h('div', { class: 'px-tscroll' });
    const t2 = h('table', { class: 'px-table' });
    t2.appendChild(h('caption', { text: 'PMF (kcal/mol) vs distance, ' + idLabel(fam, id) + ', ' + KINDS[state.kind].short + '.' }));
    head = '<tr><th scope="col">r (nm)</th>' + cols.map((c) => '<th scope="col">' + esc(c.head) + '</th>').join('') + '</tr>';
    const rows = [];
    for (const r of grid) {
      let tr = '<tr><td>' + r.toFixed(dpR) + '</td>';
      for (const c of cols) {
        const v = c.band ? valueAt(c.t.c, r, c.t.c.b) : valueAt(c.t.c, r);
        tr += v == null ? '<td class="px-na">' + NDASH + '</td>' : '<td>' + fmt(v, 3) + '</td>';
      }
      rows.push(tr + '</tr>');
    }
    t2.insertAdjacentHTML('beforeend', '<thead>' + head + '</thead><tbody>' + rows.join('') + '</tbody>');
    wrap2.appendChild(t2);
    host.appendChild(wrap2);
  }
  function downloadCSV(cols, grid, dpR) {
    const lines = ['distance_nm,' + cols.map((c) => c.key).join(',')];
    for (const r of grid) {
      const vals = cols.map((c) => {
        const v = c.band ? valueAt(c.t.c, r, c.t.c.b) : valueAt(c.t.c, r);
        return v == null ? '' : v.toFixed(4);
      });
      lines.push(r.toFixed(dpR) + ',' + vals.join(','));
    }
    const name = 'pmf_' + state.fam + '_' + idLabel(state.fam, curId()).replace(/[^A-Za-z0-9]+/g, '-') + '_' + state.kind + '.csv';
    try {
      const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
    } catch (e) {
      window.alert('Download is not available in this browser.');
    }
  }

  function renderLegend() {
    const ul = $('px-legend');
    const key = $('px-key');
    ul.innerHTML = '';
    key.innerHTML = '';
    if (state.view === 'table') return;
    const fam = state.fam;
    const id = curId();
    const shown = shownSystems();
    if (state.view === 'overlay') {
      for (const s of shown) {
        const tr = tracesFor(s, fam, id, state.kind);
        let wt;
        if (state.kind === 'halves') {
          const f = (t) => (t && t.well && t.well[0] != null && t.well[0] <= -0.02 ? fmt(t.well[0]) : NDASH);
          wt = 'min ' + f(tr[0]) + ' / ' + f(tr[1]);
        } else {
          wt = tr[0] ? wellText(tr[0].well, state.kind === 'mean') : 'no data';
        }
        const li = h('li', null, '<span class="px-sw-line" style="--c:' + color(SYS_I[s.key]) + '"></span>' +
          '<span>' + esc(s.label) + '</span><span class="px-lw">' + esc(wt) + '</span>');
        ul.appendChild(li);
      }
    }
    const parts = [];
    if (state.kind === 'mean') {
      // the band legend entry doubles as the show/hide toggle for the band
      const on = state.band !== 'off';
      parts.push('<button type="button" class="px-band-btn" id="px-band-btn" aria-pressed="' + on + '" title="' +
        (on ? 'Hide' : 'Show') + ' the |PMF1 \u2212 PMF2|/2 band"><span class="px-band-sw"></span><span>band = |PMF<sub>1</sub> \u2212 PMF<sub>2</sub>|/2' +
        (on ? '' : ' (hidden)') + '</span></button>');
    }
    if (state.kind === 'halves') {
      parts.push('<span><span class="px-sw-line" style="--c:var(--ink-2)"></span>half 1</span>');
      parts.push('<span><span class="px-sw-line px-dash" style="--c:var(--ink-2)"></span>half 2</span>');
    } else {
      parts.push('<span><span class="px-sw-dot" style="--c:var(--ink-2)"></span>curve minimum (shown when below −0.02)</span>');
    }
    if (state.view === 'multiples') {
      parts.push('<span><span class="px-sw-line" style="--c:var(--context)"></span>other shown systems (' + (state.kind === 'full' ? 'full' : 'mean') + ')</span>');
    }
    parts.push('<span>units: distance nm, PMF kcal/mol; 0 at 1.8 nm</span>');
    key.innerHTML = parts.join('');
    const bb = $('px-band-btn');
    if (bb) {
      bb.addEventListener('click', () => {
        state.band = state.band === 'off' ? 'on' : 'off';
        renderChartArea();
        writeHash();
        const nb = $('px-band-btn');
        if (nb) nb.focus();
      });
    }
  }

  function captionText() {
    const fam = state.fam;
    const kind = state.kind;
    const bin = FAMS[fam].bin;
    let s = '';
    if (kind === 'mean') {
      s = '<strong>Lines</strong>: mean of the two trajectory halves (each with the first 5% of all frames dropped), ' +
        'interpolated on a common ' + bin + ' nm grid; <strong>band</strong>' + (state.band === 'off' ? ' (hidden)' : '') +
        ': |PMF<sub>1</sub> − PMF<sub>2</sub>|/2, a quick convergence check (wide band = halves disagree). ';
    } else if (kind === 'full') {
      s = '<strong>Lines</strong>: PMF from the whole trajectory after dropping the first 5% of frames (no halves split, ' + bin + ' nm bins). ';
    } else {
      s = '<strong>Solid</strong>: first half, <strong>dashed</strong>: second half of the trajectory (each with the first 5% of all frames dropped; ' + bin + ' nm bins). Where the two disagree the PMF is not converged. ';
    }
    s += 'PMF = −<i>kT</i> ln <i>P</i>(<i>r</i>) of the raw distance histogram (no <i>r</i>² volume correction, as in <code>pmf_from_cpptraj.py</code>), ' +
      'shifted to 0 at the populated bin nearest 1.8 nm and cut there; negative = more probable than at 1.8 nm (attractive). ' +
      'Colours are fixed per system across the whole documentation. ';
    if (state.view === 'multiples') s += 'Small multiples share both axes; gray lines are the other shown systems for context. ';
    if (state.y === 'shared') s += 'PMF axis: one range for every ' + FAMS[fam].noun + ' of this family. ';
    s += 'Source: <code>overall/data/&lt;system&gt;/&lt;SEQ&gt;_PMF/</code> (from <code>pmf_from_cpptraj.py</code>); newest PMF file ' + esc(STAMP) + '.';
    return s;
  }

  // ------------------------------------------------------------ render
  let lastWidth = 0;
  function renderChartArea() {
    const fam = state.fam;
    const id = curId();
    $('px-title').textContent = idLabel(fam, id) + ' · ' + FAMS[fam].noun;
    $('px-sub').textContent = idDescription(fam, id) + ' Curve: ' + KINDS[state.kind].short + '.';
    const host = $('px-chart');
    host.innerHTML = '';
    lastWidth = host.clientWidth;
    if (state.view === 'multiples') renderMultiples(host);
    else if (state.view === 'table') renderTable(host);
    else renderOverlay(host);
    renderLegend();
    $('px-caption').innerHTML = captionText();
  }
  function announce() {
    const n = state.on.size;
    $('px-live').textContent = 'Showing ' + idLabel(state.fam, curId()) + ', ' + KINDS[state.kind].short + ', ' +
      VIEWS[state.view].toLowerCase() + ', ' + n + ' system' + (n === 1 ? '' : 's') + '.';
  }
  // focusSel: move keyboard focus to the selected matrix cell after re-render
  function renderAll(focusSel) {
    syncControls();
    renderMatrix(!!focusSel);
    renderChartArea();
    writeHash();
    announce();
  }

  // ------------------------------------------------------------ theme
  const THEMES = ['auto', 'light', 'dark'];
  function currentTheme() {
    const t = document.documentElement.getAttribute('data-theme');
    return t === 'light' || t === 'dark' ? t : 'auto';
  }
  function setTheme(t) {
    if (t === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
    try {
      if (t === 'auto') localStorage.removeItem('livagx-docs-theme');
      else localStorage.setItem('livagx-docs-theme', t);
    } catch (e) { /* storage blocked */ }
    $('px-theme').textContent = 'Theme: ' + t;
    renderMatrix(false);
  }

  // ------------------------------------------------------------ boot
  function boot() {
    readHash();
    buildControls();
    const meta = $('px-meta');
    if (meta) {
      meta.textContent = SYS.length + ' systems · 36 residue pairs, 10 window pairs and the LIVA fragment each · newest PMF file ' +
        STAMP + ' · exported by overall/scripts/export_explorer_data.py';
    }
    $('px-matrix').addEventListener('keydown', onMatrixKey);
    const tb = $('px-theme');
    if (tb) {
      tb.textContent = 'Theme: ' + currentTheme();
      tb.addEventListener('click', () => {
        const next = THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length];
        setTheme(next);
      });
    }
    try {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => renderMatrix(false));
    } catch (e) { /* old browsers */ }
    window.addEventListener('hashchange', () => {
      readHash();
      renderAll();
    });
    renderAll();
    // re-draw the SVG when the chart column changes width
    const host = $('px-chart');
    let raf = 0;
    const onResize = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        raf = 0;
        if (Math.abs(host.clientWidth - lastWidth) > 2) renderChartArea();
      });
    };
    if (window.ResizeObserver) new ResizeObserver(onResize).observe(host);
    else window.addEventListener('resize', onResize);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
