// Story page: one lesson told step by step, at a logical level (wires and symbols) and a physical level (3x3 patches).
// openStory builds the whole page inside root; everything inside is scoped by classes, so two stories never share ids.
import { SCRIPTS, PLAQUETTES, phaseText, phaseMore } from './storyText.js';
import { buildPhysical, PHYS_LEGEND } from './storyPhysical.js';
import { ringCut } from './patches.js';

const NS = 'http://www.w3.org/2000/svg';
const DWELL = 7;             // logical level: seconds per step at 1x
const LAYER_SECONDS = 2.5;   // physical level: seconds per layer at 1x
const W = 1100;              // picture width in user units
const TABLES = {             // syndrome (recipe order) -> local qubit the decoder flips
  Z: { '0010': 0, '1010': 1, '1000': 2, '0100': 3, '1100': 4, '0101': 7, '0001': 8 },
  X: { '1000': 0, '0010': 2, '1001': 3, '1100': 4, '0110': 5, '0001': 6, '0100': 7 },
};
let instances = 0;

const pct = (x) => (x === null || x === undefined ? 'n/a' : `${(100 * x).toFixed(1)}%`);
const gradeCol = (d) => (d === null ? 'var(--muted)' : d < 2 ? 'var(--good)' : d < 5 ? 'var(--x)' : 'var(--bad)');
const setAttrs = (e, a) => { for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); };

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, s, attrs = {}) {
  const t = el('text', { x, y, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--fg)', ...attrs }, parent);
  t.textContent = s;
  return t;
}
// SVG label parts from a script string: "_L" becomes a subscript L ('|0_L⟩' -> '|0', {sub: 'L'}, '⟩').
const parts = (s) => s.split('_L').flatMap((p, i) => (i ? [{ sub: 'L' }, p] : [p])).filter((p) => p !== '');
function setParts(t, ps) {
  const key = JSON.stringify(ps);
  if (t.dataset.parts === key) return;
  t.dataset.parts = key;
  t.replaceChildren();
  for (const p of ps) {
    if (typeof p === 'string') t.append(p);
    else el('tspan', { 'baseline-shift': 'sub', 'font-size': '80%' }, t).textContent = p.sub;
  }
}
// Readable on any fill: a card-coloured halo behind the glyphs.
const HALO = { stroke: 'var(--card)', 'stroke-width': 3, 'paint-order': 'stroke' };

function injection(lesson, layout, settings) {
  if (lesson !== 1) return null;
  const L = layout.inject, S = settings && settings.inject;
  if (!L && !S) return null;
  const pauli = (L ? L.pauli : S.pauli).toUpperCase();
  const q = L ? (L.local ?? L.q % 9) : S.qubit;
  const block = L ? (L.block || 0) : 0;
  const basis = (layout.bases || ['Z'])[block];
  const flips = basis === 'Z' ? pauli !== 'Z' : pauli !== 'X';
  const own = (PLAQUETTES[basis] || []).map((p, k) => (flips && p.includes(q) ? k : -1)).filter((k) => k >= 0);
  const caught = L && Array.isArray(L.caught) && L.caught.every((k) => typeof k === 'number') ? L.caught : own;
  const syn = [0, 1, 2, 3].map((k) => (caught.includes(k) ? 1 : 0)).join('');
  const fromTable = (TABLES[basis] || {})[syn];
  const cq = L && typeof L.corrected_qubit === 'number' ? L.corrected_qubit % 9 : (fromTable ?? null);
  return { pauli, q, block, basis, caught, cq: caught.length ? cq : null };
}

export function openStory(root, { lesson, rows, layout, settings, onClose }) {
  const script = SCRIPTS[lesson];
  if (!script) throw new Error(`no story for lesson ${lesson}`);
  const ids = `story${++instances}`;
  const enc = rows.find((r) => r.kind === 'enc' && (!layout.name || r.name === layout.name)) || rows.find((r) => r.kind === 'enc') || rows[0];
  const bare = rows.find((r) => r.kind === 'bare' && r.label === enc.label) || rows.find((r) => r.kind === 'bare') || null;
  const inj = injection(lesson, layout, settings);
  const rings = (enc.blocks || []).some((b) => (b.corrections || []).some((c) => c >= ringCut(b.corrections || [])));
  const ctx = { rows, enc, bare, layout, settings, inj, rings, nq: 9 * layout.blocks.length };
  const val = (x) => (typeof x === 'function' ? x(ctx) : x);
  const steps = script.steps.map((st) => Object.fromEntries(Object.entries(st).map(([k, v]) => [k, val(v)])));
  const N = steps.length;
  const TITLES = steps.map((s) => s.title);
  const wires = script.wires;
  const NW = wires.length;
  const HUE = wires.map((w) => `var(--${w.hue})`);
  const symbols = val(script.symbols);
  const labels = val(script.labels);
  const intro = val(script.intro);
  // Lesson 4 is told at the physical level only; the ideal logical circuit sits below it as a static drawing.
  const PHYS_ONLY = lesson === 4;

  // ---------- shell ----------
  root.classList.add('story');
  root.replaceChildren();
  root.innerHTML = `<div class="story-in">
  <header class="story-head"><h1>Lesson ${lesson}: ${script.title}</h1>
    <button type="button" class="story-back">Back to lessons</button></header>
  <svg class="stage" viewBox="0 0 ${W} 400" role="figure"></svg>
  ${PHYS_ONLY ? '<figure class="ideal"><figcaption>Logical circuit (ideal)</figcaption><svg role="img"></svg></figure>' : ''}
  <div class="narr">
    <p class="qlegend">Legend: wire = one logical qubit through time · gold line = now</p>
    <p class="stepline"></p>
    <div class="capbox">
      <p class="caption" aria-live="polite"></p>
      <p class="layer-line" hidden></p>
      <details class="why" hidden><summary></summary><p></p></details>
    </div>
  </div>
  <section class="state-panel" aria-live="polite">
    <h2>State now</h2>
    <p class="hdr"></p>
    <div class="state-body"></div>
  </section>
  <div class="controls">
    <button class="play" type="button">Play</button>
    <button class="back" type="button" aria-label="Back one step">◀ Back</button>
    <button class="next" type="button" aria-label="Next step">Next ▶</button>
    <div class="strip">
      <input class="scrub sr-only" type="range" min="0" max="1000" step="1" value="0" aria-label="Timeline">
      <div class="track"><div class="rail"></div><div class="fill"></div></div>
    </div>
    <span class="seg" role="group" aria-label="Level of detail">
      <button class="lv-logical" type="button" aria-pressed="true">Logical</button>
      <button class="lv-physical" type="button" aria-pressed="false">Physical</button>
    </span>
    <label>speed
      <select class="speed"><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select>
    </label>
  </div>
</div>`;
  const $ = (sel) => root.querySelector(sel);
  const svg = $('.stage');
  svg.setAttribute('aria-label', `Animated diagram: ${script.title}, on ${ctx.nq} physical qubits`);
  const defs = el('defs', {}, svg);
  for (const [id, col] of [['ah-x', 'var(--x)'], ['ah-w', 'var(--fg)']]) {
    const m = el('marker', { id: `${ids}-${id}`, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 9, markerHeight: 9, markerUnits: 'userSpaceOnUse', orient: 'auto' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: col }, m);
  }
  const glow = el('filter', { id: `${ids}-glow`, x: '-60%', y: '-60%', width: '220%', height: '220%' }, defs);
  el('feGaussianBlur', { stdDeviation: 3.5 }, glow);
  // Entanglement ribbons: vertical gradient from one wire's hue to the other's, soft at both wires.
  function ribbonGrad(i, j) {
    const id = `${ids}-rib-${i}${j}`;
    const g = el('linearGradient', { id, x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    for (const [o, w, a] of [[0, i, 0], [0.25, i, 1], [0.75, j, 1], [1, j, 0]]) el('stop', { offset: o, style: `stop-color: ${HUE[w]}; stop-opacity: ${a}` }, g);
    return `url(#${id})`;
  }

  // ---------- geometry of the logical level ----------
  const WIRE_Y = { 1: [200], 2: [120, 260], 3: [90, 200, 310] }[NW];
  const lastY = WIRE_Y[NW - 1];
  const COL = Array.from({ length: N }, (_, k) => 250 + k * (560 / (N - 1)));
  const CARDS_Y = lastY + 42;
  const LOG_H = CARDS_Y + 52;
  const PH_Y = [18, lastY + 34];
  const kMeter = symbols.findIndex((g) => g.some((s) => s.t === 'meter'));
  const post = symbols.flat().find((s) => s.t === 'post') || null;
  const kPost = post ? symbols.findIndex((g) => g.includes(post)) : -1;
  const goesOn = (w) => (post ? post.on === w : true);   // wires that continue as classical bits after their meter
  const meterX = COL[kMeter];
  const WIRE_END = meterX - 20;
  const tokX = post ? COL[kPost] + 55 + 24 : COL[N - 1];

  // ---------- result cards (shared by both levels) ----------
  const cards = script.cards(rows, ctx).map(([big, lbl, tone, v]) => {
    const col = tone === 'grade' ? gradeCol(v === null || v === undefined ? null : Math.abs(v - enc.ideal_p1) * 100)
      : tone === 'z' ? 'var(--z)' : tone === 'x' ? 'var(--x)' : 'var(--muted)';
    return { big, lbl, col, w: Math.max(big.length * 11.5, lbl.length * 6.5) + 32 };
  });
  function resultsPanel(parent, y) {
    const g = el('g', { class: 'fade res', opacity: 0 }, parent);
    el('title', {}, g).textContent = 'Results from the actual run';
    const GAP = 14;
    const total = cards.reduce((a, c) => a + c.w, 0) + GAP * (cards.length - 1);
    const k = Math.min(1, (W - 16) / total);
    let x = (W - total * k) / 2;
    for (const c of cards) {
      const w = c.w * k;
      el('rect', { x, y, width: w, height: 46, rx: 10, fill: 'var(--bg)', stroke: c.col, 'stroke-width': 1.5 }, g);
      text(g, x + 14 * k, y + 22, c.big, { 'text-anchor': 'start', 'font-size': 19 * Math.max(k, 0.85), 'font-weight': 700, fill: c.col });
      text(g, x + 14 * k, y + 39, c.lbl, { 'text-anchor': 'start', 'font-size': 12.5 * Math.max(k, 0.85), fill: 'var(--muted)' });
      x += w + GAP * k;
    }
    return g;
  }

  // ---------- logical level ----------
  const logical = el('g', { class: 'lvl lvl-logical' }, svg);
  const ribbons = [];
  const addRibbon = (i, j, x0, x1, opacity) => {
    const r = el('rect', { class: 'rib', x: x0, y: WIRE_Y[i], width: 0, height: WIRE_Y[j] - WIRE_Y[i], fill: ribbonGrad(i, j) }, logical);
    ribbons.push({ r, x0, x1, opacity });
  };
  // opacity(s) of each ribbon; width grows with the playhead from x0 up to x1.
  if (lesson === 2) addRibbon(0, 1, COL[1], meterX, (s) => (s >= 4 ? 0.05 : s >= 1 ? 0.14 : 0));
  if (lesson === 4) {
    addRibbon(1, 2, COL[1], meterX, (s) => (s >= 5 ? 0.05 : 0.14));
    addRibbon(0, 1, COL[2], meterX, (s) => (s >= 5 ? 0.05 : s >= 4 ? 0.14 : 0));
    ribbons[1].full = true;
  }

  function patchIcon(parent, cx, cy, hue) {
    const g = el('g', { transform: `translate(${cx},${cy})`, 'aria-hidden': 'true' }, parent);
    for (const [x, y, col] of [[-18, -18, 'x'], [0, -18, 'z'], [-18, 0, 'z'], [0, 0, 'x']]) {
      el('rect', { x, y, width: 18, height: 18, fill: `var(--${col})`, 'fill-opacity': 0.3, stroke: 'var(--line)', 'stroke-width': 0.8 }, g);
    }
    el('rect', { x: -18, y: -18, width: 36, height: 36, fill: 'none', stroke: hue, 'stroke-width': 1.5 }, g);
    for (let i = 0; i < 9; i++) el('circle', { cx: -18 + (i % 3) * 18, cy: -18 + Math.floor(i / 3) * 18, r: 2.6, fill: 'var(--card)', stroke: 'var(--fg)', 'stroke-width': 1 }, g);
  }
  const heads = el('g', { class: 'head' }, logical);
  WIRE_Y.forEach((y, i) => {
    el('line', { x1: 150, y1: y, x2: WIRE_END, y2: y, stroke: 'var(--muted)', 'stroke-width': 2 }, logical);
    el('line', { x1: 150, y1: y + 2.25, x2: WIRE_END, y2: y + 2.25, stroke: HUE[i], 'stroke-width': 1.5, 'stroke-opacity': 0.85 }, logical);
    patchIcon(heads, 40, y, HUE[i]);
    const t = text(heads, 68, y + 5, '', { 'text-anchor': 'start', 'font-size': 16 });
    el('tspan', { 'font-weight': 700, fill: HUE[i] }, t).textContent = wires[i].name;
    setParts(el('tspan', { dx: 10, fill: 'var(--muted)', 'font-size': 14 }, t), parts(wires[i].ket));
    if (goesOn(i)) for (const dy of [-2.5, 2.5]) el('line', { x1: meterX + 20, y1: y + dy, x2: 1040, y2: y + dy, stroke: 'var(--muted)', 'stroke-width': 1.2 }, logical);
  });

  // Symbol group k holds one .inst group per drawn symbol, so each scales about its own centre when revealed.
  const sym = [];
  for (let k = 0; k < N; k++) sym[k] = el('g', { class: 'sym' }, logical);
  const inst = (k) => el('g', { class: 'inst' }, sym[k]);
  function cnot(g, x, yc, yt) {
    const sgn = yt > yc ? 1 : -1;
    el('line', { x1: x, y1: yc, x2: x, y2: yt + 14 * sgn, stroke: 'var(--fg)', 'stroke-width': 2 }, g);
    el('circle', { cx: x, cy: yc, r: 6, fill: 'var(--fg)' }, g);
    el('circle', { cx: x, cy: yt, r: 14, fill: 'var(--card)', stroke: 'var(--fg)', 'stroke-width': 2 }, g);
    el('line', { x1: x - 14, y1: yt, x2: x + 14, y2: yt, stroke: 'var(--fg)', 'stroke-width': 2 }, g);
    el('line', { x1: x, y1: yt - 14, x2: x, y2: yt + 14, stroke: 'var(--fg)', 'stroke-width': 2 }, g);
  }
  const CORNER = { checks: [42, 24], meter: [20, 16] };
  symbols.forEach((group, k) => {
    const x = COL[k];
    for (const s of group) {
      const y = WIRE_Y[s.w], g = inst(k);
      if (s.t === 'box') {
        const bw = Math.max(72, s.label.length * 7.5 + 18);
        el('rect', { x: x - bw / 2, y: y - 16, width: bw, height: 32, rx: 6, fill: 'var(--card)', stroke: 'var(--x)', 'stroke-width': 1.5 }, g);
        text(g, x, y + 5, s.label, { 'font-size': 13 });
      } else if (s.t === 'cnot') {
        cnot(g, x, WIRE_Y[s.c], WIRE_Y[s.g]);
        text(g, x, Math.max(WIRE_Y[s.c], WIRE_Y[s.g]) + 34, s.label, { 'font-size': 11, fill: 'var(--muted)' });
      } else if (s.t === 'error') {
        el('path', { d: `M${x - 9},${y - 20} L${x + 5},${y - 5} L${x - 5},${y + 4} L${x + 9},${y + 20}`, fill: 'none', stroke: 'var(--bad)', 'stroke-width': 3.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, g);
        text(g, x, y - 28, s.label, { 'font-size': 12, 'font-weight': 700, fill: 'var(--bad)', ...HALO });
      } else if (s.t === 'note') {
        text(g, x, y - 12, s.label, { 'font-size': 12, fill: 'var(--muted)', ...HALO });
      } else if (s.t === 'checks') {
        const basis = (layout.bases || enc.bases)[s.w];
        el('rect', { x: x - 42, y: y - 24, width: 84, height: 48, rx: 10, fill: 'var(--card)', stroke: 'var(--z)', 'stroke-width': 1.5 }, g);
        text(g, x, y - 3, 'checks', { 'font-size': 14, 'font-weight': 600 });
        text(g, x, y + 15, s.sub || (basis === 'Y' ? 'none in Y' : `${basis} plaquettes`), { 'font-size': 11, fill: basis === 'X' ? 'var(--x)' : basis === 'Z' ? 'var(--z)' : 'var(--muted)' });
      } else if (s.t === 'meter') {
        el('rect', { x: x - 20, y: y - 16, width: 40, height: 32, rx: 4, fill: 'var(--card)', stroke: 'var(--fg)', 'stroke-width': 1.5 }, g);
        el('path', { d: `M${x - 12},${y + 8} A13,13 0 0 1 ${x + 12},${y + 8}`, fill: 'none', stroke: 'var(--fg)', 'stroke-width': 1.5 }, g);
        el('line', { x1: x, y1: y + 8, x2: x + 9, y2: y - 8, stroke: 'var(--fg)', 'stroke-width': 1.5 }, g);
        text(g, x, y - 22, s.basis, { 'font-size': 13, 'font-weight': 700, fill: s.basis === 'X' ? 'var(--x)' : s.basis === 'Z' ? 'var(--z)' : 'var(--fg)' });
      } else if (s.t === 'badge') {
        const [dx, dy] = CORNER[s.on];
        const bx = (s.on === 'checks' ? COL[symbols.findIndex((gr) => gr.some((o) => o.t === 'checks'))] : meterX) + dx;
        const bw = s.text.length * 7 + 12;
        el('rect', { x: bx - bw / 2, y: y - dy - 10, width: bw, height: 18, rx: 9, fill: s.tone === 'bad' ? 'var(--bad)' : 'var(--good)' }, g);
        text(g, bx, y - dy + 3.5, s.text, { 'font-size': 11, 'font-weight': 700, fill: 'var(--bg)' });
      } else if (s.t === 'post') {
        const yo = WIRE_Y[s.on];
        s.from.forEach((w, j) => {
          const yf = WIRE_Y[w], top = yf < yo ? yo - 18 : yo + 18;
          const fx = x - 50 + 25 * j;
          el('path', { d: `M${meterX + 20},${yf} H${fx} V${top}`, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.2, 'stroke-dasharray': '4 4' }, g);
        });
        el('rect', { x: x - 55, y: yo - 18, width: 110, height: 36, rx: 10, fill: 'var(--card)', stroke: 'var(--focus)', 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }, g);
        text(g, x, yo + 5, s.label, { 'font-size': 13 });
        text(g, x, yo + 33, s.sub, { 'font-size': 11, fill: 'var(--muted)' });
      }
    }
  });
  // Symbol group k is revealed once the playhead reaches its column.
  const symLit = (k, u) => u >= k / N - 1e-9;
  const stateLabels = WIRE_Y.map((y, i) => text(logical, 160, y - 26, '', { 'text-anchor': 'start', 'font-size': 13, fill: HUE[i], ...HALO }));

  // ---------- tokens (lesson 4) ----------
  // A token rides ahead of the playhead and changes form at fixed steps. It never slides while hidden or across
  // the picture: it jumps, then opacity or the pop does the showing.
  function makeToken(hue, label) {
    const wrap = el('g', { class: 'tok-wrap' }, logical);
    const tok = el('g', { class: 'tok', 'pointer-events': 'none', 'aria-hidden': 'true' }, wrap);
    const tin = el('g', { class: 'tok-in' }, tok);
    const halo = el('rect', { y: -17, height: 34, rx: 17, fill: hue, opacity: 0.55, filter: `url(#${ids}-glow)` }, tin);
    const body = el('rect', { y: -13, height: 26, rx: 13, fill: hue }, tin);
    const t = text(tin, 0, 5.5, '', { 'font-size': 16, 'font-weight': 700, fill: '#1a1408' });
    const lbl = text(tok, -16, 4, label || '', { class: 'tok-lbl', 'text-anchor': 'end', 'font-size': 12, 'font-weight': 600, fill: hue, ...HALO });
    lbl.style.opacity = 0;
    let mode = null, shown = '';
    function setText(s) {
      if (s === shown) return;
      shown = s;
      setParts(t, parts(s));
      const w = Math.max(26, [...s.replace('_L', '')].length * 9 + 10);
      for (const [r, pad] of [[halo, 4], [body, 0]]) { r.setAttribute('x', -w / 2 - pad); r.setAttribute('width', w + 2 * pad); }
    }
    function place(m, x, y, opts = {}) {
      const changed = m !== mode;
      const jump = changed && (mode === null || mode === 'gone' || opts.jump);
      if (jump) tok.style.transition = 'none';
      tok.style.transform = `translate(${x}px, ${y}px)`;
      if (changed) {
        tok.style.opacity = m === 'gone' ? 0 : 1;
        lbl.style.opacity = opts.label ? 1 : 0;
        tin.classList.toggle('small', !!opts.small);
        tin.classList.remove('pop');
        if (opts.pop && mode !== null) { void tin.getBBox(); tin.classList.add('pop'); }
      }
      if (jump) { void tok.getBoundingClientRect(); tok.style.transition = ''; }
      mode = m;
    }
    return { place, setText, el: tin };
  }
  const tokens = [];
  if (lesson === 4) {
    const tk = makeToken(HUE[0], 'T ⊗ Bell');
    tk.setText('T');
    const MID = (WIRE_Y[0] + WIRE_Y[1]) / 2;
    tokens.push((s, f, ph) => {
      if (s < 2 || (s === 2 && f < 0.3)) tk.place('a', Math.min(ph + 26, COL[2]), WIRE_Y[0]);
      else if (s === 2) tk.place('ab', COL[2], MID, { label: true, small: true });
      else if (s === 3) tk.place('ab', Math.min(Math.max(ph + 26, COL[2]), meterX), MID, { label: true, small: true });
      else if (s === 4) tk.place('gone', meterX, MID);
      else tk.place('c', tokX, WIRE_Y[2], { jump: true, pop: true });
    });
  }

  // Static copy of the logical level, every symbol shown, without tokens or ribbons (built before the playhead and cards).
  if (PHYS_ONLY) {
    const fig = $('.ideal svg');
    const y0 = WIRE_Y[0] - 40;
    setAttrs(fig, { viewBox: `0 ${y0} ${W} ${lastY + 50 - y0}`,
      'aria-label': 'Logical circuit (ideal): encode A, B and C; CNOT B to C; CNOT A to B; read A in X and B in Z; Pauli frame on C from both readouts; read C in X or Y.' });
    const copy = logical.cloneNode(true);
    copy.querySelectorAll('.tok-wrap, .rib').forEach((n) => n.remove());
    fig.append(copy);
  }

  const logPlayhead = el('line', { y1: PH_Y[0], y2: PH_Y[1], stroke: 'var(--focus)', 'stroke-width': 2, 'stroke-opacity': 0.6, 'pointer-events': 'none' }, logical);
  const logResults = resultsPanel(logical, CARDS_Y);
  // Playhead x is piecewise linear: step k starts on its own column and ends on the next one.
  function playheadX(v) {
    const pts = COL.map((x, k) => [k / N, x]).concat([[1, 1040]]);
    for (let i = 1; i < pts.length; i++) {
      if (v <= pts[i][0]) { const [u0, x0] = pts[i - 1], [u1, x1] = pts[i]; return x0 + (x1 - x0) * (v - u0) / (u1 - u0); }
    }
    return 1040;
  }

  // ---------- physical level ----------
  const phys = buildPhysical({
    parent: svg, layout, row: enc, steps, lesson, inj, el, text, setParts, HALO, hues: HUE, ids,
    kets: wires.map((w) => parts(w.ket)),
  });
  const physical = phys.g;
  const phyResults = resultsPanel(physical, phys.resY);
  const CLOSE_PHY_Y = phys.resY + 46 + 10;

  // ---------- state panel (HTML, both levels) ----------
  $('.state-panel .hdr').innerHTML = script.stateHeader || '';
  const stateBody = $('.state-body');
  const pl = (str) => str.replace(/\b[ABC]\b/g, (ch) => `<span class="q${ch.toLowerCase()}">${ch}</span>`);
  const names = layout.blocks.map((b) => b.name);
  const resultPatch = post ? wires[post.on].name : wires[0].name;
  function actingPatches(phase) {
    if (phase === null) return resultPatch;
    let xs = phase === 'measure' || phase === 'logical readout' ? names : [...new Set(phase.match(/\b[ABC]\b/g) || [])];
    if (!xs.length) xs = phase.startsWith('inject') && inj ? [names[inj.block]] : names.slice(0, 1);
    return xs.length <= 1 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;
  }
  let lastState = '';
  function renderState(s, phase) {
    const where = level === 'physical' ? actingPatches(phase) : '';
    const key = `${s}|${level}|${where}`;
    if (key === lastState) return;
    lastState = key;
    const st = steps[s];
    let h = '';
    if (st.pre) h += `<p class="pre">${st.pre}</p>`;
    h += `<p class="expr">${st.state}</p>`;
    if (st.table) {
      h += `<table><tr>${st.table.head.map((x) => `<th>${x}</th>`).join('')}</tr>`
        + st.table.rows.map((r) => `<tr>${r.map((x) => `<td>${x}</td>`).join('')}</tr>`).join('') + '</table>';
    }
    if (st.stateNote) h += `<p class="line">${st.stateNote}</p>`;
    if (level === 'physical') h += `<p class="where">Patch holding the action now: ${pl(where)}</p>`;
    stateBody.innerHTML = h;
  }

  // ---------- legend (HTML list with 24x14 SVG swatches drawn with the main picture's strokes) ----------
  const sw = (inner) => `<svg width="24" height="14" viewBox="0 0 24 14" aria-hidden="true">${inner}</svg>`;
  const SW = {
    op: sw('<rect x="2" y="1.5" width="20" height="11" rx="3" fill="var(--card)" stroke="var(--x)" stroke-width="1.5"/>'),
    cnot: sw('<line x1="4" y1="7" x2="18" y2="7" stroke="var(--fg)" stroke-width="2"/><circle cx="4" cy="7" r="3" fill="var(--fg)"/>'
      + '<circle cx="17.5" cy="7" r="5.5" fill="var(--card)" stroke="var(--fg)" stroke-width="1.5"/>'
      + '<line x1="12" y1="7" x2="23" y2="7" stroke="var(--fg)" stroke-width="1.5"/><line x1="17.5" y1="1.5" x2="17.5" y2="12.5" stroke="var(--fg)" stroke-width="1.5"/>'),
    error: sw('<path d="M8,1 L14,6 L10,8 L16,13" fill="none" stroke="var(--bad)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'),
    checks: sw('<rect x="2" y="1" width="20" height="12" rx="3.5" fill="var(--card)" stroke="var(--z)" stroke-width="1.5"/>'),
    bad: sw('<rect x="3" y="2" width="18" height="10" rx="5" fill="var(--bad)"/>'),
    meter: sw('<rect x="4" y="0.75" width="16" height="12.5" rx="2" fill="var(--card)" stroke="var(--fg)" stroke-width="1.5"/>'
      + '<path d="M7,10 A5.5,5.5 0 0 1 17,10" fill="none" stroke="var(--fg)" stroke-width="1.5"/><line x1="12" y1="10" x2="15.5" y2="4" stroke="var(--fg)" stroke-width="1.5"/>'),
    bits: sw('<line x1="0" y1="4.5" x2="24" y2="4.5" stroke="var(--muted)" stroke-width="1.2"/><line x1="0" y1="9.5" x2="24" y2="9.5" stroke="var(--muted)" stroke-width="1.2"/>'),
    good: sw('<rect x="3" y="2" width="18" height="10" rx="5" fill="var(--good)"/>'),
    feed: sw('<line x1="0" y1="7" x2="24" y2="7" stroke="var(--muted)" stroke-width="1.2" stroke-dasharray="4 4"/>'),
    frame: sw('<rect x="1.5" y="1.5" width="21" height="11" rx="3" fill="var(--card)" stroke="var(--focus)" stroke-width="1.5" stroke-dasharray="6 4"/>'),
  };
  const LOGICAL_LEGEND = [
    ['op', (s) => s.t === 'box', 'a logical operation'],
    ['cnot', (s) => s.t === 'cnot', 'CNOT: filled dot is the control, circle-plus the target'],
    ['error', (s) => s.t === 'error', 'red zigzag: an error strikes the patch'],
    ['checks', (s) => s.t === 'checks', "checks box: the patch's plaquette checks, read from its measurement"],
    ['bad', (s) => s.t === 'badge' && s.tone === 'bad', 'red badge: a check reads −1'],
    ['meter', (s) => s.t === 'meter', 'meter with a letter: measurement in that basis'],
    ['bits', (s) => s.t === 'meter', 'double line: classical bits after measurement'],
    ['good', (s) => s.t === 'badge' && s.tone === 'good', 'green badge: the decoder fixed it'],
    ['feed', (s) => s.t === 'post' && s.from.length > 0, 'thin dashed line: a measurement outcome carried into post-processing'],
    ['frame', (s) => s.t === 'post', 'dashed box: done on the recorded bits, not on the hardware'],
  ];
  // What the overlay legend lists at a position: only the changing symbols currently in the picture, in a fixed order.
  function legendItems(lv, p) {
    if (lv === 'physical') return phys.legendItems(phys.layerOf(p.s, p.f));
    const u = (p.s + p.f) / N;
    const shown = symbols.filter((_, k) => symLit(k, u)).flat();
    return LOGICAL_LEGEND.filter(([, test]) => shown.some(test)).map(([key, , label]) => ({ key, icon: SW[key], label }));
  }
  const LEG = { logical: { x: 802, y: 8, w: 290, h: Math.max(160, PH_Y[1] - 40) }, physical: PHYS_LEGEND };
  const legendFO = el('foreignObject', { x: LEG.logical.x, y: LEG.logical.y, width: LEG.logical.w, height: LEG.logical.h }, svg);
  const legendBox = document.createElement('div');
  legendBox.className = 'ov ov-tr legend';
  legendBox.setAttribute('role', 'region'); legendBox.setAttribute('aria-label', 'Legend');
  const legendBody = document.createElement('div'); legendBody.className = 'legend-body';
  legendBox.append(legendBody); legendFO.append(legendBox);
  let lastLegend = '', legendBottom = LEG.logical.y;
  function renderLegend(show) {
    const items = show ? legendItems(level, pos) : [];
    legendFO.setAttribute('display', items.length ? 'inline' : 'none');
    if (!items.length) return;
    const key = level + '|' + items.map((it) => it.key).join(',');
    if (key === lastLegend) return;
    lastLegend = key;
    setAttrs(legendFO, { x: LEG[level].x, y: LEG[level].y, width: LEG[level].w, height: LEG[level].h });
    legendBox.classList.toggle('stack', level === 'physical');
    legendBody.innerHTML = `<ul>${items.map((it) => `<li data-key="${it.key}">${it.icon}<span>${it.label}</span></li>`).join('')}</ul>`;
    legendBottom = LEG[level].y + legendBox.offsetHeight;   // picture units: the foreignObject lays out in user space
  }

  // ---------- title card and closing card (inside the picture) ----------
  function cardFO(cls, html) {
    const fo = el('foreignObject', { class: cls, display: 'none' }, svg);
    const wrap = document.createElement('div');
    const box = document.createElement('div'); box.className = 'ovcard'; box.innerHTML = html;
    wrap.append(box); fo.append(wrap);
    return { fo, wrap, box };
  }
  const introCard = cardFO('intro', `<h2>${intro.title}</h2><p class="sub">${intro.subtitle}</p><p>${intro.line}</p><p class="go">Press Play</p>`);
  introCard.fo.setAttribute('pointer-events', 'none');
  Object.assign(introCard.wrap.style, { height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' });
  const closeCard = cardFO('closing', `<h3>What you saw</h3><ul>${script.closing(rows, ctx).map((l) => `<li>${l}</li>`).join('')}</ul>`);
  const CLOSE_AT = { logical: { x: 160, y: 20, width: 580, height: CARDS_Y - 30 }, physical: { x: 24, y: CLOSE_PHY_Y, width: 1052, height: 200 } };

  // ---------- timeline ----------
  // Master position: step s in 0..N-1 and the fraction f in [0,1) of the way through it.
  let pos = { s: 0, f: 0 }, level = 'logical', playing = false, speed = 1, lastTs = null, lastNarr = '', raf = 0, closed = false;
  // introOn: the title card shows at {0, 0} until Play, Next, a dot or the timeline is used. ended: Play reached the end.
  let introOn = true, ended = false, lastVB = '';
  const uOf = () => (pos.s + pos.f) / N;
  function setFromU(v) {
    v = Math.min(1, Math.max(0, v));
    const s = Math.min(N - 1, Math.floor(v * N + 1e-9));
    pos = { s, f: Math.min(0.999999, Math.max(0, v * N - s)) };
  }
  const scrub = $('.scrub'), playBtn = $('.play'), caption = $('.caption'), stepline = $('.stepline');
  const why = $('.why'), whySum = $('.why summary'), whyText = $('.why p'), qlegend = $('.qlegend'), capbox = $('.capbox');
  const layerLine = $('.layer-line');
  const track = $('.track'), stripFill = $('.track .fill');
  // Step strip: one real button per step (aria-current on the current one); the hidden range keeps the keyboard and screen-reader path.
  const dots = TITLES.map((title, k) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'sdot'; b.title = `Step ${k + 1}: ${title}`;
    b.setAttribute('aria-label', `Step ${k + 1} of ${N}: ${title}`);
    b.style.left = `${(100 * k) / (N - 1)}%`;
    b.innerHTML = `<span class="pip"></span><span class="sd-lbl">${title}</span>`;
    b.addEventListener('click', () => jumpTo(k));
    track.append(b);
    return b;
  });
  function renderStrip() {
    stripFill.style.width = `${(100 * Math.min(N - 1, pos.s + pos.f)) / (N - 1)}%`;
    dots.forEach((d, k) => {
      d.classList.toggle('past', k < pos.s);
      if (k === pos.s) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current');
    });
    scrub.setAttribute('aria-valuetext', `Step ${pos.s + 1} of ${N}, ${TITLES[pos.s]}`);
  }

  // Step line (small, muted): "Step k of N · Title", plus the phase name at the physical level. Caption below it.
  // more = [summary, text] fills the closed disclosure under the caption.
  function setNarr(s, phase, body, more) {
    const key = `${s}|${phase}|${body}|${level}|${more}`;
    if (key === lastNarr) return;
    lastNarr = key;
    const b = document.createElement('b'); b.textContent = TITLES[s];
    stepline.replaceChildren(`Step ${s + 1} of ${N} · `, b);
    if (phase) stepline.append(` · ${phase}`);
    caption.innerHTML = body;
    capbox.classList.toggle('phys', !!more);
    why.hidden = !more;
    const [sum, txt] = more || ['', ''];
    if (whyText.textContent !== txt) { why.open = false; whySum.textContent = sum; whyText.textContent = txt; }
  }

  // Physical level: "Layer k of n in <phase>: <caption>" for a real layer; nothing for a virtual one or the logical level.
  function setLayerLine(phase, t) {
    const p = t === null || t === undefined ? null : layout.phases.find((x) => x.name === phase);
    const cap = p && layout.layer_text ? layout.layer_text[String(t)] : '';
    const str = cap ? `Layer ${t - p.t0 + 1} of ${p.t1 - p.t0 + 1} in ${phase}: ${cap}` : '';
    if (layerLine.textContent !== str) layerLine.textContent = str;
    layerLine.hidden = !str;
  }

  function render() {
    if (closed) return;
    const { s, f } = pos, u = uOf();
    if (document.activeElement !== scrub) scrub.value = Math.round(u * 1000);
    renderStrip();
    const showIntro = introOn && s === 0 && f === 0;
    const closing = s === N - 1 && (f >= 0.6 || ended);
    logical.classList.toggle('faint', showIntro);
    physical.classList.toggle('faint', showIntro);
    introCard.fo.setAttribute('display', showIntro ? 'inline' : 'none');
    closeCard.fo.setAttribute('display', closing ? 'inline' : 'none');
    logical.classList.toggle('closing', closing);
    if (closing) setAttrs(closeCard.fo, CLOSE_AT[level]);
    // Picture height: fixed on the logical level; on the physical level it ends under the bars, and grows at the result step.
    let H = LOG_H;
    if (level === 'physical') H = s < N - 1 ? phys.height : closing ? CLOSE_PHY_Y + closeCard.box.offsetHeight + 12 : phys.resY + 46 + 12;
    const vb = `0 0 ${W} ${H}`;
    if (vb !== lastVB) { lastVB = vb; svg.setAttribute('viewBox', vb); }
    // The card is centred right of the wire heads (which end near x 175), so the names and kets stay readable.
    if (showIntro) setAttrs(introCard.fo, { x: 200, y: 0, width: 880, height: H });
    renderLegend(!showIntro);
    qlegend.hidden = level !== 'logical';
    if (level === 'logical') {
      const x = playheadX(u);
      logPlayhead.setAttribute('x1', x); logPlayhead.setAttribute('x2', x);
      // In the legend's strip the playhead starts below the legend, so it never runs through the text.
      const legShown = legendFO.getAttribute('display') !== 'none';
      logPlayhead.setAttribute('y1', legShown && x >= LEG.logical.x - 6 ? legendBottom + 8 : PH_Y[0]);
      // Reveal: class on the group, opacity on the leaves (a dimmed group leaves a faint compositing box in Chromium).
      sym.forEach((g, k) => g.classList.toggle('on', symLit(k, u)));
      for (const rb of ribbons) {
        const end = rb.full ? rb.x1 : Math.min(Math.max(x, rb.x0), rb.x1);
        rb.r.setAttribute('width', end - rb.x0);
        rb.r.style.opacity = rb.opacity(s);
      }
      for (const t of tokens) t(s, f, x);
      (labels[s] || []).forEach((str, i) => setParts(stateLabels[i], parts(str)));
      logResults.setAttribute('opacity', s === N - 1 ? 1 : 0);
      setNarr(s, '', steps[s].caption, steps[s].more || null);
      setLayerLine(null, null);
      renderState(s, null);
    } else {
      const { phase, t } = phys.render(s, f);
      phyResults.setAttribute('opacity', s === N - 1 ? 1 : 0);
      setNarr(s, phase || '', phase ? phaseText(lesson, phase, ctx) : steps[s].caption, phase ? phaseMore(lesson, phase) : null);
      setLayerLine(phase, t);
      renderState(s, phase);
    }
  }

  function setLevel(lv) {
    level = lv === 'physical' || PHYS_ONLY ? 'physical' : 'logical';
    logical.setAttribute('display', level === 'logical' ? 'inline' : 'none');
    physical.setAttribute('display', level === 'physical' ? 'inline' : 'none');
    $('.lv-logical').setAttribute('aria-pressed', level === 'logical');
    $('.lv-physical').setAttribute('aria-pressed', level === 'physical');
    phys.invalidate();
    render();
  }
  function setPlaying(p) {
    playing = p;
    if (playing) { introOn = false; ended = false; }
    if (playing && pos.s >= N - 1) pos = { s: 0, f: 0 };
    playBtn.textContent = playing ? 'Pause' : 'Play';
    lastTs = null;
    cancelAnimationFrame(raf);
    if (playing) raf = requestAnimationFrame(tick);
    render();
  }
  function tick(ts) {
    if (!playing || closed) return;
    if (lastTs !== null) {
      // Logical: every step dwells the same time. Physical: every layer gets the same time.
      const L = phys.stepLayers[pos.s];
      const stepSeconds = level === 'logical' ? DWELL : (L ? L.n : 1) * LAYER_SECONDS;
      let f = pos.f + ((ts - lastTs) / 1000) * speed / stepSeconds, s = pos.s;
      if (f >= 1) { s += 1; f = 0; }
      pos = { s, f };
      if (s >= N - 1) { playing = false; ended = true; playBtn.textContent = 'Play'; }   // the result step holds, with the closing card
    }
    lastTs = ts;
    render();
    if (playing) raf = requestAnimationFrame(tick);
  }
  // Back goes to the start of the current step, or the previous one when already at a start.
  function step(dir) {
    const { s, f } = pos;
    introOn = false; ended = false;
    if (dir > 0) pos = { s: Math.min(N - 1, s + 1), f: 0 };
    else pos = f > 0.02 ? { s, f: 0 } : { s: Math.max(0, s - 1), f: 0 };
    render();
  }
  // A dot jumps to the start of its step; a click or drag on the line scrubs (the far end shows the closing card).
  function jumpTo(k) {
    if (playing) setPlaying(false);
    introOn = false; ended = false; pos = { s: k, f: 0 };
    render();
  }
  function seekX(clientX) {
    const r = track.getBoundingClientRect(), q = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
    if (playing) setPlaying(false);
    introOn = false; ended = false;
    setFromU(q >= 0.995 ? 0.99999 : (q * (N - 1)) / N);
    render();
  }
  track.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.sdot')) return;
    track.setPointerCapture(e.pointerId); seekX(e.clientX);
  });
  track.addEventListener('pointermove', (e) => { if (track.hasPointerCapture(e.pointerId)) seekX(e.clientX); });
  playBtn.addEventListener('click', () => setPlaying(!playing));
  $('.back').addEventListener('click', () => step(-1));
  $('.next').addEventListener('click', () => step(1));
  $('.lv-logical').addEventListener('click', () => setLevel('logical'));
  $('.lv-physical').addEventListener('click', () => setLevel('physical'));
  $('.speed').addEventListener('change', (e) => { speed = Number(e.target.value); });
  scrub.addEventListener('input', () => { if (playing) setPlaying(false); introOn = false; ended = false; setFromU(Number(scrub.value) / 1000); render(); });
  $('.story-back').addEventListener('click', () => leave());

  function onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'Escape') { e.preventDefault(); leave(); return; }
    const tag = e.target.tagName;
    if (tag === 'SELECT' || tag === 'TEXTAREA' || (tag === 'INPUT' && e.target !== scrub) || e.target.isContentEditable) return;
    if (e.key === ' ' && tag === 'SUMMARY') return;   // Space opens the caption's disclosure, not Play
    if (e.key === ' ') { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    else if (e.key === 'l' || e.key === 'L') setLevel('logical');
    else if (e.key === 'p' || e.key === 'P') setLevel('physical');
  }
  // A focused button would also "click" on Space keyup; the keydown handler already toggled play.
  function onKeyUp(e) { if (e.key === ' ' && e.target.tagName === 'BUTTON' && root.contains(e.target)) e.preventDefault(); }
  document.addEventListener('keydown', onKey);
  document.addEventListener('keyup', onKeyUp);

  function close() {
    if (closed) return;
    closed = true; playing = false;
    cancelAnimationFrame(raf);
    document.removeEventListener('keydown', onKey);
    document.removeEventListener('keyup', onKeyUp);
    root.replaceChildren();
    if (window.__story === api) delete window.__story;
  }
  // Back to lessons and Escape tear the story down, then hand control back to the page.
  function leave() {
    close();
    if (onClose) onClose();
  }

  const api = {
    close,
    setPos: (s, f = 0) => { ended = false; pos = { s: Math.max(0, Math.min(N - 1, s)), f: Math.max(0, Math.min(0.999999, f)) }; render(); },
    setLevel,
    // Test hooks so a screenshot script can place the playhead exactly.
    setU: (v) => { ended = false; setFromU(v); render(); },
    showIntro: (on) => { introOn = !!on; render(); },
    get u() { return uOf(); },
    get pos() { return { ...pos, step: pos.s }; },
    get level() { return level; },
    get steps() { return N; },
    legendItems: (lv, s, f) => legendItems(lv, { s, f }).map((it) => it.label),
  };
  window.__story = api;
  // The story opens on the title card and waits for Play.
  $('.seg').hidden = PHYS_ONLY;
  setLevel(PHYS_ONLY ? 'physical' : 'logical');
  playBtn.focus({ preventScroll: true });
  return api;
}
