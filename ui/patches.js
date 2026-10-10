// Draws one SVG patch per logical qubit for the current lesson, lit by syndrome fire rates.
const NS = 'http://www.w3.org/2000/svg';
const BLOCK_NAMES = { 1: ['A'], 2: ['A', 'B'], 3: ['A', 'B', 'C'], 4: ['A', 'B'] };
const ROLE = {
  1: { A: 'the logical qubit' },
  2: { A: 'control, |+>', B: 'target, |0>' },
  3: { A: 'input psi', B: 'half of Bell pair', C: 'output' },
  4: { A: 'data psi', B: 'magic state T|+>' },
};
const LINKS = { 2: [['A', 'B']], 3: [['B', 'C'], ['A', 'B']], 4: [['B', 'A']] };
// Square at y 0..120 inside the group, caps reach y -60 and 180, labels sit below the bottom cap.
const W = 170, H = 300, X0 = 25, Y0 = 70;

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

export function renderPatches(root, lesson, rows, settings) {
  root.replaceChildren();
  const enc = rows.find((r) => r.kind === 'enc') || rows[0];
  const names = BLOCK_NAMES[lesson];
  const blocks = enc.blocks || [];
  const fired = [];
  names.forEach((name, bi) => {
    const f = blocks[bi] || { x_face: 0, z_top: 0, z_bottom: 0, zzzz_anc: 0 };
    const basis = enc.bases[bi];
    const zRead = basis === 'Z';   // Z0Z1 and Z2Z3 are only readable from a Z readout
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'group', 'aria-label': `Patch ${name}, ${ROLE[lesson][name]}, read in ${basis}`, width: W, height: H });
    el('title', {}, svg).textContent = `Patch ${name} of the [[4,1,2]] code with syndrome fire rates, read in ${basis}`;
    const g = el('g', { transform: `translate(${X0}, ${Y0})` }, svg);
    const P = { q0: [0, 0], q1: [120, 0], q2: [0, 120], q3: [120, 120] };
    // face: XXXX
    const face = el('rect', { x: 0, y: 0, width: 120, height: 120, fill: 'var(--x)', 'fill-opacity': 0.12 + 0.88 * f.x_face, stroke: 'var(--line)', class: 'face' }, g);
    el('title', {}, face).textContent = basis === 'Y'
      ? `XXXX check from the ancilla, fired in ${pct(f.x_face)} of shots`
      : `XXXX check, fired in ${pct(f.x_face)} of shots`;
    label(g, 60, 64, pct(f.x_face));
    // top and bottom semicircles: Z0Z1, Z2Z3; dashed muted outline when this readout cannot measure them
    const zLook = (rate) => (zRead
      ? { fill: 'var(--z)', 'fill-opacity': 0.12 + 0.88 * rate, stroke: 'var(--line)' }
      : { fill: 'none', stroke: 'var(--muted)', 'stroke-dasharray': '4 3' });
    const top = el('path', { d: 'M0,0 A60,60 0 0 1 120,0 Z', ...zLook(f.z_top), class: 'face' }, g);
    el('title', {}, top).textContent = zRead
      ? `Z0Z1 check, fired in ${pct(f.z_top)} of shots`
      : 'Z0Z1 check, not measured in this readout';
    const bot = el('path', { d: 'M0,120 A60,60 0 0 0 120,120 Z', ...zLook(f.z_bottom), class: 'face' }, g);
    el('title', {}, bot).textContent = zRead
      ? `Z2Z3 check, fired in ${pct(f.z_bottom)} of shots`
      : 'Z2Z3 check, not measured in this readout';
    if (zRead) {
      label(g, 60, -24, pct(f.z_top));
      label(g, 60, 154, pct(f.z_bottom));
    } else {
      label(g, 60, -24, 'not measured', 10);
      label(g, 60, 154, 'not measured', 10);
    }
    // logical operators as edges with hover titles
    const zl = el('line', { x1: 0, y1: 0, x2: 0, y2: 120, stroke: 'var(--fg)', 'stroke-width': 4, 'stroke-dasharray': '6 4', tabindex: 0, role: 'img', 'aria-label': 'Logical Z = Z0 Z2 (left edge)' }, g);
    el('title', {}, zl).textContent = 'Logical Z = Z0 Z2 (left edge)';
    const xl = el('line', { x1: 0, y1: 0, x2: 120, y2: 0, stroke: 'var(--fg)', 'stroke-width': 4, 'stroke-dasharray': '2 4', tabindex: 0, role: 'img', 'aria-label': 'Logical X = X0 X1 (top edge)' }, g);
    el('title', {}, xl).textContent = 'Logical X = X0 X1 (top edge)';
    // data qubits
    Object.entries(P).forEach(([q, [x, y]], i) => {
      const hit = settings && settings.inject && lesson === 1 && settings.inject.qubit === i;
      const dot = el('circle', { cx: x, cy: y, r: 10, fill: hit ? 'var(--bad)' : 'var(--card)', stroke: 'var(--fg)', 'stroke-width': 2 }, g);
      el('title', {}, dot).textContent = hit ? `data qubit ${i}, injected ${settings.inject.pauli} error` : `data qubit ${i}`;
      el('text', { x: x + (x ? 14 : -14), y: y + (y ? 24 : -14), 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' }, g).textContent = `q${i}`;
    });
    el('text', { x: 60, y: 205, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--fg)', 'font-weight': 600 }, g).textContent = `Patch ${name}`;
    el('text', { x: 60, y: 221, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }, g).textContent = ROLE[lesson][name];
    root.appendChild(svg);
    const list = [['XXXX', f.x_face], ['Z0Z1', f.z_top], ['Z2Z3', f.z_bottom], ['ZZZZ ancilla', f.zzzz_anc]]
      .filter(([, v]) => v > 0).map(([k, v]) => `${k} ${pct(v)}`);
    const unread = zRead ? '' : ` (Z edges not measured, basis ${basis})`;
    fired.push(`Patch ${name}: ${list.length ? list.join(', ') : 'no check fired'}${unread}`);
  });
  // transversal CNOT links between corresponding corners, as one line of text under the patches
  const links = (LINKS[lesson] || []).map(([a, b]) => `transversal CNOT ${a} → ${b}, corner to corner`);
  if (links.length) {
    const l = document.createElement('p');
    l.className = 'links';
    l.textContent = links.join('; ');
    l.title = 'Each logical CNOT is four physical CNOTs, one between each pair of matching corners';
    root.appendChild(l);
  }
  const p = document.createElement('p');
  p.className = 'fired-list';
  p.textContent = `Rates from ${enc.name}. Checks fired (share of all shots). ` + fired.join(' | ');
  root.appendChild(p);
}
