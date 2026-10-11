// Physical level of the story page: one 3x3 patch per logical qubit, animated layer by layer from the spacetime layout.
// story.js owns the timeline; this module draws the patches, the step header row, the phase bar and the dynamic legend items.
import { PLAQUETTES } from './storyText.js';
import { ringCut } from './patches.js';

const SP = 44;                                  // dot spacing inside a patch
// Fewer patches draw larger: one patch (with its caps) fills about 70% of the height above the legend strip.
const SP_OF = { 1: 60, 2: 52 };
const CXS = { 1: [470], 2: [300, 640], 3: [190, 515, 840] };
const CY = 124;                                 // centre row of every patch
export const PHYS = { nameY: CY + 122 };
const STRIP = { x: 8, y: PHYS.nameY + 12, w: 1084, h: 38 };
const HDR = { y: STRIP.y + STRIP.h + 3, h: 18 };
const BAR = { x0: 60, x1: 1040, y: HDR.y + HDR.h + 4, h: 30 };
export const PHYS_LEGEND = { x: 972, y: 6, w: 122, h: STRIP.y - 12 };
const GATE = { h: 'H', ry: 'Ry', t: 'T', tdg: 'T†', sdg: 'S†', s: 'S', x: 'X', y: 'Y', z: 'Z' };
// Bulk plaquettes are squares; boundary ones are half-discs on the named side.
const SHAPE = { '0,1': 'top', '7,8': 'bottom', '2,5': 'right', '3,6': 'left' };

const sw = (inner) => `<svg width="24" height="14" viewBox="0 0 24 14" aria-hidden="true">${inner}</svg>`;
const head = (x, y, col) => `<path d="M${x - 6},${y - 3.5} L${x},${y} L${x - 6},${y + 3.5} Z" fill="${col}"/>`;
const arrowSw = (col, w) => sw(`<path d="M1,11 Q10,1 18,7" fill="none" stroke="${col}" stroke-width="${w}" stroke-linecap="round"/>${head(23, 8.5, col)}`);
const SW = {
  gold: arrowSw('var(--x)', 2.5),
  white: arrowSw('var(--fg)', 3),
  ring: sw('<circle cx="12" cy="7" r="5.5" fill="none" stroke="var(--focus)" stroke-width="2.5"/>'),
  burst: sw('<circle cx="12" cy="7" r="6" fill="var(--bad)" fill-opacity="0.35" stroke="var(--bad)" stroke-width="2"/>'),
  dataM: sw('<circle cx="6" cy="7" r="5" fill="var(--card)" stroke="var(--fg)" stroke-width="2"/><circle cx="18" cy="7" r="5" fill="var(--muted)" stroke="var(--fg)" stroke-width="2"/>'),
  dashed: sw('<rect x="5" y="1.5" width="14" height="11" fill="none" stroke="var(--muted)" stroke-dasharray="3 2"/>'),
  fix: sw('<circle cx="12" cy="7" r="3" fill="var(--muted)"/><circle cx="12" cy="7" r="6" fill="none" stroke="var(--bad)" stroke-width="2"/>'),
  logic: sw('<line x1="2" y1="7" x2="22" y2="7" stroke="var(--focus)" stroke-width="4" stroke-dasharray="5 3"/>'),
};

const arrowKind = (role) => (role === 'transversal' ? 'white' : 'gold');

// Short bar labels for phases too narrow for their names, derived from the ops they hold.
function shortLabel(name, ops) {
  if (name === 'measure') return 'M';
  if (name === 'logical readout') return 'L';
  if (name === 'no error') return '·';
  if (name.startsWith('transversal CNOT ')) return name.slice(17);
  if (name.startsWith('inject ')) return name.split(' ')[1];
  const gates = [];
  for (const op of ops) if (op.phase === name && !gates.includes(GATE[op.gate] || op.gate)) gates.push(GATE[op.gate] || op.gate);
  return gates.join(' ');
}

// Display sequence: real layout phases grouped into steps by name prefix, plus one virtual layer for a step that
// has none of its own (for example "logical readout", or "no error" when nothing was injected).
function buildSequence(layout, steps) {
  const segs = [];
  const used = new Set();
  steps.forEach((st, k) => {
    const prefixes = st.phases || [];
    const mine = layout.phases.filter((p) => !used.has(p.name) && prefixes.some((x) => p.name.startsWith(x)));
    mine.forEach((p) => { used.add(p.name); segs.push({ name: p.name, t0: p.t0, t1: p.t1, step: k, real: true }); });
    if (!mine.length && st.virtual) segs.push({ name: st.virtual, step: k, real: false });
  });
  // A layout phase no step claims joins the step of the phase before it, so no layer is ever dropped.
  layout.phases.forEach((p) => {
    if (used.has(p.name)) return;
    const prev = segs.filter((s) => s.real && s.t1 < p.t0).pop();
    const at = prev ? segs.indexOf(prev) + 1 : 0;
    segs.splice(at, 0, { name: p.name, t0: p.t0, t1: p.t1, step: prev ? prev.step : 0, real: true });
    used.add(p.name);
    console.warn(`story: phase "${p.name}" not named by any step; shown with step ${prev ? prev.step + 1 : 1}`);
  });
  const seq = [];
  for (const s of segs) {
    s.d0 = seq.length;
    if (s.real) for (let t = s.t0; t <= s.t1; t++) seq.push({ t, seg: s });
    else seq.push({ t: null, seg: s });
    s.n = seq.length - s.d0;
  }
  const stepLayers = steps.map((_, k) => {
    const mine = segs.filter((s) => s.step === k);
    if (!mine.length) return null;
    return { d0: mine[0].d0, n: mine.reduce((a, s) => a + s.n, 0) };
  });
  return { segs, seq, stepLayers };
}

export function buildPhysical(o) {
  const { parent, layout, row, steps, lesson, inj, el, text, setParts, HALO, hues, ids } = o;
  const nb = layout.blocks.length;
  const CX = CXS[nb] || CXS[3];
  const sp = SP_OF[nb] || SP;
  const bases = layout.bases || row.bases;
  const { segs, seq, stepLayers } = buildSequence(layout, steps);
  const RESULT = seq.length;
  const dMeasure = seq.findIndex((x) => x.seg.name === 'measure');
  const dRead = seq.findIndex((x) => x.seg.name === 'logical readout');
  const dInject = inj && layout.inject ? seq.findIndex((x) => x.t === layout.inject.t) : -1;
  const lit = (d) => dMeasure >= 0 && d >= dMeasure;

  const g = el('g', { class: 'lvl lvl-physical', display: 'none' }, parent);
  const QPOS = {}, qDot = {}, qNum = {}, qOf = {};
  for (const q of layout.qubits) qOf[q.id] = q;
  const patches = [];
  layout.blocks.forEach((b, bi) => {
    const cx = CX[bi];
    const P = (i) => [cx + ((i % 3) - 1) * sp, CY + (Math.floor(i / 3) - 1) * sp];
    const pg = el('g', {}, g);
    const blk = (row.blocks && row.blocks[bi]) || { checks: [], corrections: [] };
    const basis = bases[bi];
    const plaqs = [];
    for (const type of ['X', 'Z']) {
      PLAQUETTES[type].forEach((qs, k) => {
        const col = type === 'X' ? 'var(--x)' : 'var(--z)';
        const shape = SHAPE[qs.join(',')] || 'sq';
        let e, lx, ly, anchor = 'middle', closed = null, arc = null;
        if (shape === 'sq') {
          const [x0, y0] = P(qs[0]);
          e = el('rect', { x: x0, y: y0, width: sp, height: sp, fill: col, 'fill-opacity': 0.12, stroke: 'var(--line)' }, pg);
          lx = x0 + sp / 2; ly = y0 + sp / 2 + 4;
        } else {
          const [xa, ya] = P(qs[0]), [xb, yb] = P(qs[1]), r = sp / 2;
          const sweep = shape === 'top' || shape === 'right' ? 1 : 0;
          arc = `M${xa},${ya} A${r},${r} 0 0 ${sweep} ${xb},${yb}`;
          closed = `${arc} Z`;
          e = el('path', { d: closed, fill: col, 'fill-opacity': 0.12, stroke: 'var(--line)' }, pg);
          const mx = (xa + xb) / 2, my = (ya + yb) / 2;
          if (shape === 'top') { lx = mx; ly = ya - r - 5; }
          else if (shape === 'bottom') { lx = mx; ly = ya + r + 13; }
          else if (shape === 'right') { lx = xa + r + 4; ly = my + 4; anchor = 'start'; }
          else { lx = xa - r - 4; ly = my + 4; anchor = 'end'; }
        }
        const readable = type === basis;
        const rate = readable ? (blk.checks[k] || 0) : 0;
        const tt = el('title', {}, e);
        tt.textContent = readable
          ? `${type}(${qs.join(',')}) check of ${b.name}, fired in ${(100 * rate).toFixed(1)}% of shots`
          : `${type}(${qs.join(',')}) check of ${b.name}, not readable when ${b.name} is read in ${basis}`;
        const lbl = text(pg, lx, ly, readable ? `${(100 * rate).toFixed(1)}%` : '', { 'text-anchor': anchor, 'font-size': 10.5, display: 'none', 'pointer-events': 'none', ...HALO });
        plaqs.push({ e, type, k, shape, closed, arc, col, readable, rate, lbl, caught: inj && bi === inj.block && readable && inj.caught.includes(k) });
      });
    }
    const [x0, y0] = P(0), [x8, y8] = P(8);
    el('rect', { x: x0, y: y0, width: x8 - x0, height: y8 - y0, fill: 'none', stroke: hues[bi], 'stroke-width': 1.5, 'pointer-events': 'none' }, pg);
    const zl = el('line', { x1: x0, y1: y0, x2: x0, y2: y8, stroke: 'var(--fg)', 'stroke-width': 4, 'stroke-dasharray': '6 4', 'stroke-opacity': 0.7 }, pg);
    el('title', {}, zl).textContent = 'Logical Z = Z0 Z3 Z6 (left column)';
    const xl = el('line', { x1: x0, y1: y0, x2: x8, y2: y0, stroke: 'var(--fg)', 'stroke-width': 4, 'stroke-dasharray': '2 4', 'stroke-opacity': 0.7 }, pg);
    el('title', {}, xl).textContent = 'Logical X = X0 X1 X2 (top row)';
    const rings = el('g', { display: 'none', 'pointer-events': 'none' }, pg);
    const cut = ringCut(blk.corrections || []);
    b.data.forEach((qid, i) => {
      const [x, y] = P(i);
      QPOS[qid] = [x, y, 9];
      const c = (blk.corrections || [])[i] || 0;
      if (c >= cut) {
        const s = Math.sqrt(Math.min(1, c));
        const ring = el('circle', { cx: x, cy: y, r: 12 + 9 * s, fill: 'none', stroke: 'var(--bad)', 'stroke-width': 2, 'stroke-opacity': Math.min(1, 0.25 + 0.75 * c / 0.25) }, rings);
        el('title', {}, ring).textContent = `q${i} corrected by the decoder in ${(100 * c).toFixed(1)}% of shots`;
      }
      qDot[qid] = el('circle', { cx: x, cy: y, r: 9, fill: 'var(--card)', stroke: 'var(--fg)', 'stroke-width': 2 }, pg);
      el('title', {}, qDot[qid]).textContent = `${b.name} data qubit q${i}`;
      qNum[qid] = text(pg, x, y + 3.2, `${i}`, { 'font-size': 9, fill: 'var(--muted)', 'pointer-events': 'none' });
    });
    const nm = text(pg, cx, PHYS.nameY, '', { 'font-size': 13 });
    el('tspan', { 'font-weight': 700, fill: hues[bi] }, nm).textContent = b.name;
    setParts(el('tspan', { dx: 10, fill: 'var(--muted)' }, nm), o.kets[bi]);
    // Lesson 1 with an injected error: caught checks outlined red and a "corrected" tag at the measure layer.
    let tag = null;
    if (inj && bi === inj.block && inj.cq !== null && inj.caught.length) {
      tag = el('g', { display: 'none', 'pointer-events': 'none' }, pg);
      // The leader leaves the qubit down and to the right, through the gap between rows, to a tag right of the patch.
      const [qx, qy] = P(inj.cq), tx = cx + sp + 64, ty = qy + sp / 2 + 4;
      el('path', { d: `M${qx + 9},${qy + 9} L${qx + sp / 2},${qy + sp / 2} H${tx - 6}`, fill: 'none', stroke: 'var(--bad)', 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, tag);
      text(tag, tx, ty + 4, `corrected ↺ q${inj.cq}`, { 'text-anchor': 'start', 'font-size': 12, 'font-weight': 700, fill: 'var(--bad)', ...HALO });
    }
    patches.push({ plaqs, rings, zl, xl, basis, tag });
  });
  const dyn = el('g', { 'pointer-events': 'none' }, g);

  // Quiet strip (constant legend), header row of steps, fine phase bar.
  {
    const fo = el('foreignObject', { x: STRIP.x, y: STRIP.y, width: STRIP.w, height: STRIP.h }, g);
    const dv = document.createElement('div');
    dv.className = 'ov ov-strip';
    dv.innerHTML = '<p>Legend: circle = data qubit · gold = X plaquette · blue = Z plaquette'
      + ' · dashed left column = logical Z · dotted top row = logical X</p>'
      + '<p>Codewords: |0<sub>L</sub>⟩: 16 terms, |1<sub>L</sub>⟩ = X<sub>L</sub>|0<sub>L</sub>⟩</p>';
    fo.append(dv);
  }
  const pxPerLayer = (BAR.x1 - BAR.x0) / seq.length;
  const hdrSegs = steps.map((st, k) => {
    const L = stepLayers[k];
    const x = L ? BAR.x0 + L.d0 * pxPerLayer : BAR.x1 + 4, w = L ? L.n * pxPerLayer : 1096 - x;
    const hg = el('g', {}, g);
    el('title', {}, hg).textContent = `Step ${k + 1}: ${st.title}`;
    const r = el('rect', { x: x + 1, y: HDR.y, width: w - 2, height: HDR.h, rx: 4, fill: 'var(--bg)', stroke: 'var(--line)' }, hg);
    // Full title when it fits, else the step's short label, cut with an ellipsis when even that is too wide.
    const room = (str) => str.length * 5.6 + 6 <= w - 2;
    const fits = room(st.title);
    let lbl = fits ? st.title : st.short || st.title;
    if (!room(lbl)) lbl = `${lbl.slice(0, Math.max(1, Math.floor((w - 8) / 5.6) - 1))}…`;
    const t = text(hg, x + w / 2, HDR.y + HDR.h / 2 + 3.5, lbl, { 'font-size': 10, fill: 'var(--muted)' });
    return { r, t, fits, cx: x + w / 2, title: st.title };
  });
  const barSegs = segs.map((s) => {
    const x = BAR.x0 + s.d0 * pxPerLayer, w = s.n * pxPerLayer;
    const bg = el('g', {}, g);
    el('title', {}, bg).textContent = s.real ? s.name : `${s.name} (no gates)`;
    const r = el('rect', { x: x + 1, y: BAR.y, width: w - 2, height: BAR.h, rx: 6, fill: 'var(--bg)', stroke: 'var(--line)', ...(s.real ? {} : { 'stroke-dasharray': '3 3' }) }, bg);
    const fitsW = (str) => str.length * 5.4 + 4 <= w - 2;
    const sh = shortLabel(s.name, layout.ops);
    const lbl = fitsW(s.name) ? s.name : (sh && fitsW(sh) ? sh : '');
    if (lbl) text(bg, x + w / 2, BAR.y + BAR.h / 2 + 3.5, lbl, { 'font-size': 10, fill: 'var(--muted)' });
    return { s, r };
  });
  const pill = el('g', { display: 'none', 'pointer-events': 'none' }, g);
  const pillRect = el('rect', { y: HDR.y - 1, height: HDR.h + 2, rx: 5, fill: 'var(--card)', stroke: 'var(--focus)', 'stroke-width': 1.5 }, pill);
  const pillText = text(pill, 0, HDR.y + HDR.h / 2 + 3.5, '', { 'font-size': 10, 'font-weight': 600 });
  const barHead = el('line', { y1: HDR.y - 3, y2: BAR.y + BAR.h + 6, stroke: 'var(--focus)', 'stroke-width': 2.5, 'pointer-events': 'none' }, g);

  // ---------- one display layer ----------
  function layerMarks(d) {
    const cur = seq[d];
    if (!cur || cur.t === null) return [];
    const prev = seq[d - 1];
    const marks = [];
    if (prev && prev.t === cur.t - 1) for (const op of layout.ops) if (op.gate === 'cx' && op.t === prev.t) marks.push({ type: 'cx', op, opacity: 0.3 });
    for (const op of layout.ops) {
      if (op.t !== cur.t || op.gate === 'measure' || op.gate === 'barrier') continue;
      marks.push(op.gate === 'cx' ? { type: 'cx', op, opacity: 1 } : { type: op.role === 'inject' ? 'inject' : 'gate', op });
    }
    return marks;
  }
  function onSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy || 1;
    const u = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L2));
    return Math.hypot(ax + u * dx - px, ay + u * dy - py);
  }
  function arrow(a, b, role, opacity) {
    const [x0, y0, r0] = QPOS[a], [x1, y1, r1] = QPOS[b];
    let cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, look;
    if (arrowKind(role) === 'white') {
      // Matching qubits of two patches: row 0 bows over the top, row 2 under the bottom, row 1 rises gently between rows.
      const loc = qOf[a].local, rowi = Math.floor(loc / 3), col = loc % 3;
      if (rowi === 0) cy -= 120 - 18 * col;
      else if (rowi === 2) cy += 120 - 18 * col;
      else cy -= 64;
      look = ['var(--fg)', 3, `${ids}-ah-w`];
    } else {
      // Inside a patch: a slight bow, or a wider one when the straight path would cross another dot.
      const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1;
      const blocked = Object.entries(QPOS).some(([q, [px, py]]) => +q !== a && +q !== b && qOf[q].block === qOf[a].block
        && onSegment(px, py, x0, y0, x1, y1) < 14);
      const bow = blocked ? 46 : 14;
      cx += (-dy / L) * bow; cy += (dx / L) * bow;
      look = ['var(--x)', 2.5, `${ids}-ah-x`];
    }
    const u0 = [cx - x0, cy - y0], l0 = Math.hypot(...u0) || 1;
    const u1 = [x1 - cx, y1 - cy], l1 = Math.hypot(...u1) || 1;
    const sx = x0 + (u0[0] / l0) * (r0 + 1), sy = y0 + (u0[1] / l0) * (r0 + 1);
    const ex = x1 - (u1[0] / l1) * (r1 + 2), ey = y1 - (u1[1] / l1) * (r1 + 2);
    el('path', { d: `M${sx},${sy} Q${cx},${cy} ${ex},${ey}`, fill: 'none', stroke: look[0], 'stroke-width': look[1], 'marker-end': `url(#${look[2]})`, opacity, 'stroke-linecap': 'round', pathLength: 1, ...(opacity === 1 ? { class: 'draw' } : {}) }, dyn);
  }
  let lastD = null;
  function drawLayer(d) {
    if (d === lastD) return;
    lastD = d;
    dyn.replaceChildren();
    for (const { type, op, opacity } of layerMarks(d)) {
      if (type === 'cx') { arrow(op.q[0], op.q[1], op.role, opacity); continue; }
      const [x, y] = QPOS[op.q[0]];
      if (type === 'inject') {
        el('circle', { cx: x, cy: y, r: 18, fill: 'var(--bad)', 'fill-opacity': 0.35, stroke: 'var(--bad)', 'stroke-width': 3, class: 'flash' }, dyn);
        text(dyn, x, y - 24, op.gate.toUpperCase(), { 'font-size': 12, 'font-weight': 700, fill: 'var(--bad)', ...HALO });
        continue;
      }
      el('circle', { cx: x, cy: y, r: 15, fill: 'none', stroke: 'var(--focus)', 'stroke-width': 2.5, class: 'flash' }, dyn);
      text(dyn, x, y - 20, GATE[op.gate] || op.gate, { 'font-size': 12, 'font-weight': 700, fill: 'var(--focus)', ...HALO });
    }
    const on = lit(d), read = dRead >= 0 && d >= dRead;
    for (const [qid, dot] of Object.entries(qDot)) {
      dot.setAttribute('fill', on ? 'var(--muted)' : 'var(--card)');
      qNum[qid].setAttribute('fill', on ? 'var(--bg)' : 'var(--muted)');
      const hit = dInject >= 0 && d >= dInject && +qid === layout.inject.q;
      dot.setAttribute('stroke', hit ? 'var(--bad)' : 'var(--fg)');
      dot.setAttribute('stroke-width', hit ? 3.5 : 2);
    }
    patches.forEach((p) => {
      for (const q of p.plaqs) {
        const dashed = on && !q.readable;
        if (q.arc) q.e.setAttribute('d', dashed ? q.arc : q.closed);   // open arc, so the dashes do not double a logical edge
        q.e.setAttribute('fill', dashed ? 'none' : q.col);
        q.e.setAttribute('fill-opacity', on && q.readable ? 0.12 + 0.88 * q.rate : 0.12);
        const red = on && q.caught;
        q.e.setAttribute('stroke', red ? 'var(--bad)' : dashed ? 'var(--muted)' : 'var(--line)');
        q.e.setAttribute('stroke-width', red ? 2.5 : 1);
        if (dashed) q.e.setAttribute('stroke-dasharray', '4 3'); else q.e.removeAttribute('stroke-dasharray');
        q.lbl.setAttribute('display', on && q.readable ? 'inline' : 'none');
      }
      p.rings.setAttribute('display', on ? 'inline' : 'none');
      if (p.tag) p.tag.setAttribute('display', on ? 'inline' : 'none');
      const hiZ = read && p.basis !== 'X', hiX = read && p.basis !== 'Z';
      for (const [ln, hi] of [[p.zl, hiZ], [p.xl, hiX]]) {
        ln.setAttribute('stroke', hi ? 'var(--focus)' : 'var(--fg)');
        ln.setAttribute('stroke-opacity', hi ? 1 : 0.7);
        ln.setAttribute('stroke-width', hi ? 5 : 4);
      }
    });
  }

  // Changing symbols in the picture at layer d, in a fixed order (the constant ones live in the quiet strip).
  function legendItems(d) {
    const items = [];
    const add = (cond, key, label) => { if (cond) items.push({ key, icon: SW[key], label }); };
    const marks = layerMarks(d);
    const kinds = new Set(marks.filter((m) => m.type === 'cx').map((m) => arrowKind(m.op.role)));
    add(kinds.has('gold'), 'gold', 'gold arrow: CNOT inside a patch, control to target');
    add(kinds.has('white'), 'white', 'arc: one CNOT of the transversal gate');
    add(marks.some((m) => m.type === 'gate'), 'ring', 'yellow ring: single-qubit gate (letter says which)');
    add(marks.some((m) => m.type === 'inject'), 'burst', 'red burst: injected error');
    add(lit(d), 'dataM', 'grey: measured');
    add(lit(d), 'dashed', 'dashed: not readable in this basis');
    add(lit(d) && (row.blocks || []).some((b) => (b.corrections || []).some((c) => c >= ringCut(b.corrections))), 'fix', 'red ring: corrected by the decoder');
    add(dRead >= 0 && d >= dRead, 'logic', 'gold line: read as the logical value');
    return items;
  }

  // Header row and bar for step s at fraction f; returns the display layer and its phase name (null at the result).
  function render(s, f) {
    const L = stepLayers[s];
    const d = L ? L.d0 + Math.min(L.n - 1, Math.floor(f * L.n)) : RESULT;
    drawLayer(d);
    const tf = L ? L.d0 + f * L.n : seq.length;
    const hx = BAR.x0 + Math.min(tf, seq.length) * pxPerLayer;
    barHead.setAttribute('x1', hx); barHead.setAttribute('x2', hx);
    const cur = seq[d] ? seq[d].seg : null;
    for (const b of barSegs) {
      const on = b.s === cur;
      b.r.setAttribute('stroke', on ? 'var(--focus)' : 'var(--line)');
      b.r.setAttribute('stroke-width', on ? 2 : 1);
      b.r.setAttribute('fill', on ? 'var(--card)' : 'var(--bg)');
    }
    hdrSegs.forEach((h, k) => {
      const on = k === s;
      h.r.setAttribute('stroke', on ? 'var(--focus)' : 'var(--line)');
      h.r.setAttribute('stroke-width', on ? 2 : 1);
      h.r.setAttribute('fill', on ? 'var(--card)' : 'var(--bg)');
      h.t.setAttribute('fill', on ? 'var(--fg)' : 'var(--muted)');
      h.t.setAttribute('font-weight', on ? 600 : 400);
    });
    const hc = hdrSegs[s];
    pill.setAttribute('display', hc.fits ? 'none' : 'inline');
    if (!hc.fits) {
      pillText.textContent = hc.title;
      const w = pillText.getComputedTextLength() + 16;
      const x = Math.min(1096 - w / 2, Math.max(4 + w / 2, hc.cx));
      pillText.setAttribute('x', x);
      pillRect.setAttribute('x', x - w / 2); pillRect.setAttribute('width', w);
    }
    return { d, phase: cur ? cur.name : null, t: seq[d] ? seq[d].t : null };
  }

  return {
    g, render, legendItems, stepLayers,
    layerOf: (s, f) => { const L = stepLayers[s]; return L ? L.d0 + Math.min(L.n - 1, Math.floor(f * L.n)) : RESULT; },
    height: BAR.y + BAR.h + 10, resY: BAR.y + BAR.h + 8,
    invalidate: () => { lastD = null; },
    lesson,
  };
}
