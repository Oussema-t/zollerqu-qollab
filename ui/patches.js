// Draws one SVG patch per logical qubit for the current lesson, lit by check fire rates and decoder corrections.
const NS = 'http://www.w3.org/2000/svg';
const BLOCK_NAMES = { 1: ['A'], 2: ['A', 'B'], 3: ['A', 'B', 'C'], 4: ['A', 'B'] };
const ROLE = {
  1: { A: 'the logical qubit' },
  2: { A: 'control, |+>', B: 'target, |0>' },
  3: { A: 'input ψ', B: 'half of Bell pair', C: 'output' },
  4: { A: 'data ψ', B: 'magic state T|+>' },
};
const LINKS = { 2: [['A', 'B']], 3: [['B', 'C'], ['A', 'B']], 4: [['B', 'A']] };
// Data qubit i sits at (col, row) = (i % 3, floor(i / 3)), spacing S, q0 top-left.
// Grid spans 0..120 inside the group; caps reach R beyond each edge; name and role sit below the bottom cap.
const S = 60, R = 30;
const W = 210, H = 240, X0 = 45, Y0 = 40;
const pos = (i) => [S * (i % 3), S * Math.floor(i / 3)];
// Plaquettes; idx is the position in the check list of the matching readout basis.
// shape 'sq' = bulk square with its top-left corner at q[0]; 'top'/'bottom'/'left'/'right' = boundary semicircle.
const PLAQ = [
  { type: 'X', idx: 0, q: [0, 1, 3, 4], shape: 'sq' },
  { type: 'X', idx: 1, q: [4, 5, 7, 8], shape: 'sq' },
  { type: 'X', idx: 2, q: [2, 5], shape: 'right' },
  { type: 'X', idx: 3, q: [3, 6], shape: 'left' },
  { type: 'Z', idx: 0, q: [1, 2, 4, 5], shape: 'sq' },
  { type: 'Z', idx: 1, q: [3, 4, 6, 7], shape: 'sq' },
  { type: 'Z', idx: 2, q: [0, 1], shape: 'top' },
  { type: 'Z', idx: 3, q: [7, 8], shape: 'bottom' },
];
// Qubit label offsets [dx, dy, anchor], chosen to clear the caps and the correction rings.
const QLABEL = [
  [-16, 4, 'end'], [10, -12, 'start'], [0, -17, 'middle'],
  [-16, -10, 'end'], [12, 19, 'start'], [15, 12, 'start'],
  [0, 24, 'middle'], [-10, 20, 'end'], [16, 4, 'start'],
];

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function pct(x) { return `${(100 * x).toFixed(1)}%`; }

// Rate or note drawn on a face; a halo keeps it readable on a fully lit face, and it never steals the hover.
function label(g, x, y, text, size = 12) {
  el('text', {
    x, y, 'text-anchor': 'middle', 'font-size': size, fill: 'var(--fg)', stroke: 'var(--card)',
    'stroke-width': 3, 'paint-order': 'stroke', 'pointer-events': 'none',
  }, g).textContent = text;
}

// Outline path, rate label position and type letter position of one plaquette.
function geometry(p) {
  const [x, y] = pos(p.q[0]);
  if (p.shape === 'sq') {
    return { d: `M${x},${y} h${S} v${S} h${-S} Z`, lab: [x + S / 2, y + S / 2 + 4], tag: [x + 46, y + 19] };
  }
  const [bx, by] = pos(p.q[1]);
  const arc = (sweep) => `M${x},${y} A${R},${R} 0 0 ${sweep} ${bx},${by} Z`;
  if (p.shape === 'top') return { d: arc(1), lab: [x + R, y - 6], tag: [x + R, y - 20] };
  if (p.shape === 'bottom') return { d: arc(0), lab: [x + R, y + 13], tag: [x + R, y + 27] };
  if (p.shape === 'right') return { d: arc(1), lab: [x + 16, y + R + 11], tag: [x + 15, y + R - 6] };
  return { d: arc(0), lab: [x - 18, y + R + 11], tag: [x - 15, y + R - 6] };
}

export function renderPatches(root, lesson, rows, settings) {
  root.replaceChildren();
  const enc = rows.find((r) => r.kind === 'enc') || rows[0];
  const names = BLOCK_NAMES[lesson];
  const blocks = enc.blocks || [];
  // three patches share the lesson grid column, so they are drawn a little smaller
  const scale = names.length >= 3 ? 0.82 : 1;
  const fired = [];
  names.forEach((name, bi) => {
    const blk = blocks[bi] || {};
    const basis = blk.basis || enc.bases[bi];
    const checks = blk.checks || [];
    const corr = blk.corrections || [];
    const svg = el('svg', {
      viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-label': `Patch ${name}, ${ROLE[lesson][name]}, read in ${basis}`,
      width: Math.round(W * scale), height: Math.round(H * scale),
    });
    el('title', {}, svg).textContent = `Patch ${name} of the [[9,1,3]] code with check fire rates and decoder corrections, read in ${basis}`;
    const g = el('g', { transform: `translate(${X0}, ${Y0})` }, svg);
    // plaquettes: lit by fire rate when this readout can read them, dashed outline otherwise
    PLAQ.forEach((p) => {
      const geo = geometry(p);
      const qs = p.q.map((i) => `q${i}`).join(' ');
      const read = basis === p.type;
      const rate = read ? (checks[p.idx] || 0) : 0;
      const look = read
        ? { fill: p.type === 'X' ? 'var(--x)' : 'var(--z)', 'fill-opacity': 0.12 + 0.88 * rate, stroke: 'var(--line)' }
        : { fill: 'none', stroke: 'var(--muted)', 'stroke-dasharray': '4 3' };
      const face = el('path', { d: geo.d, ...look, class: 'face' }, g);
      el('title', {}, face).textContent = read
        ? `${p.type} check on ${qs}, fired in ${pct(rate)} of shots`
        : `${p.type} check on ${qs}, not readable in a ${basis} readout`;
      if (read) label(g, geo.lab[0], geo.lab[1], pct(rate), 11);
      el('text', {
        x: geo.tag[0], y: geo.tag[1], 'text-anchor': 'middle', 'font-size': 9, fill: 'var(--muted)',
        stroke: 'var(--card)', 'stroke-width': 2, 'paint-order': 'stroke', 'pointer-events': 'none',
      }, g).textContent = p.type;
    });
    // logical operators as edges with hover titles
    const zl = el('line', { x1: 0, y1: 0, x2: 0, y2: 2 * S, stroke: 'var(--fg)', 'stroke-width': 4, 'stroke-dasharray': '6 4', tabindex: 0, role: 'img', 'aria-label': 'Logical Z = Z0 Z3 Z6 (left column)' }, g);
    el('title', {}, zl).textContent = 'Logical Z = Z0 Z3 Z6 (left column)';
    const xl = el('line', { x1: 0, y1: 0, x2: 2 * S, y2: 0, stroke: 'var(--fg)', 'stroke-width': 4, 'stroke-dasharray': '2 4', tabindex: 0, role: 'img', 'aria-label': 'Logical X = X0 X1 X2 (top row)' }, g);
    el('title', {}, xl).textContent = 'Logical X = X0 X1 X2 (top row)';
    // data qubits, with a red ring where the decoder applied a correction
    for (let i = 0; i < 9; i++) {
      const [x, y] = pos(i);
      const c = corr[i] || 0;
      if (c > 0) {
        const ring = el('circle', {
          cx: x, cy: y, r: 14, fill: 'none', stroke: 'var(--bad)', 'stroke-width': 3,
          'stroke-opacity': 0.25 + 0.75 * Math.min(1, c * 4),
        }, g);
        el('title', {}, ring).textContent = `decoder corrected this qubit in ${pct(c)} of shots`;
      }
      const hit = settings && settings.inject && lesson === 1 && settings.inject.qubit === i;
      const dot = el('circle', { cx: x, cy: y, r: 9, fill: hit ? 'var(--bad)' : 'var(--card)', stroke: 'var(--fg)', 'stroke-width': 2 }, g);
      el('title', {}, dot).textContent = hit ? `data qubit ${i}, injected ${settings.inject.pauli} error` : `data qubit ${i}`;
      const [dx, dy, anchor] = QLABEL[i];
      el('text', { x: x + dx, y: y + dy, 'text-anchor': anchor, 'font-size': 11, fill: 'var(--muted)', 'pointer-events': 'none' }, g).textContent = `q${i}`;
    }
    el('text', { x: S, y: 2 * S + R + 26, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--fg)', 'font-weight': 600 }, g).textContent = `Patch ${name}`;
    el('text', { x: S, y: 2 * S + R + 42, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }, g).textContent = ROLE[lesson][name];
    root.appendChild(svg);
    const list = PLAQ.filter((p) => p.type === basis && (checks[p.idx] || 0) > 0)
      .map((p) => `${p.type}(${p.q.join(',')}) ${pct(checks[p.idx])}`);
    const fixes = corr.map((c, i) => [i, c]).filter(([, c]) => c > 0).map(([i, c]) => `q${i} ${pct(c)}`);
    let text = `Patch ${name}: ${list.length ? list.join(', ') : 'no check fired'}`;
    if (fixes.length) text += `; corrections: ${fixes.join(', ')}`;
    text += basis === 'Y' ? ' (no checks readable in basis Y)' : ` (${basis === 'Z' ? 'X' : 'Z'} checks not readable in basis ${basis})`;
    fired.push(text);
  });
  // transversal CNOT links between matching data qubits, as one line of text under the patches
  const links = (LINKS[lesson] || []).map(([a, b]) => `transversal CNOT ${a} → ${b}, qubit to qubit, 9 CNOTs`);
  if (links.length) {
    const l = document.createElement('p');
    l.className = 'links';
    l.textContent = links.join('; ');
    l.title = 'Each logical CNOT is nine physical CNOTs, one between each pair of matching data qubits';
    root.appendChild(l);
  }
  const p = document.createElement('p');
  p.className = 'fired-list';
  p.textContent = `Rates from ${enc.name}. Checks fired (share of all shots). ` + fired.join(' | ');
  root.appendChild(p);
}
