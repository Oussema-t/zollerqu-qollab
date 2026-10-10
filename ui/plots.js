// ui/plots.js: the theory-match sweep plot and the Durr-Hoyer threshold trajectory, hand-drawn SVG.
const NS = 'http://www.w3.org/2000/svg';
export const SERIES_COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)'];

function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function text(parent, x, y, s, attrs = {}) {
  el('text', { x, y, ...attrs }, parent).textContent = s;
}

function frame(svgEl, W, H, label) {
  svgEl.replaceChildren();
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svgEl.setAttribute('role', 'img');
  svgEl.setAttribute('aria-label', label);
}

// series: [{M, T, points: [{j, measured, theory}], max_dev}] from sweep_analyze_json; N = 2^m.
export function renderSweep(svgEl, series, N) {
  const W = 560, H = 320, L = 48, R = 16, T = 40, B = 44;
  frame(svgEl, W, H, 'Measured Grover success probability against the theory curve for ' +
    series.map((s) => `M = ${s.M}`).join(', ') + ` of N = ${N}.`);
  const jMax = Math.max(...series.flatMap((s) => s.points.map((p) => p.j)));
  const x = (j) => L + (j / jMax) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);

  const axes = el('g', { class: 'axes' }, svgEl);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v === 0 ? 'axis' : 'gridline' }, axes);
    text(axes, L - 6, y(v) + 4, v.toFixed(2), { 'text-anchor': 'end' });
  }
  for (let j = 0; j <= jMax; j++) text(axes, x(j), H - B + 16, String(j), { 'text-anchor': 'middle' });
  text(axes, (L + W - R) / 2, H - 8, 'Grover iterations j', { 'text-anchor': 'middle' });
  text(axes, 12, (T + H - B) / 2, 'P(success)', { 'text-anchor': 'middle', transform: `rotate(-90 12 ${(T + H - B) / 2})` });

  series.forEach((s, k) => {
    const color = SERIES_COLORS[k % SERIES_COLORS.length];
    const theta = Math.asin(Math.sqrt(s.M / N));
    const last = s.points[s.points.length - 1].j;
    let d = '';
    for (let i = 0; i <= 200; i++) {
      const j = (i / 200) * last;
      d += (i ? 'L' : 'M') + x(j).toFixed(1) + ',' + y(Math.sin((2 * j + 1) * theta) ** 2).toFixed(1);
    }
    el('path', { d, class: 'theory', stroke: color }, svgEl);
    const g = el('g', { fill: color }, svgEl);
    for (const p of s.points) {
      const c = el('circle', { cx: x(p.j), cy: y(p.measured), r: 4, class: 'meas' }, g);
      el('title', {}, c).textContent = `M = ${s.M}, j = ${p.j}: measured ${p.measured.toFixed(3)}, theory ${p.theory.toFixed(3)}`;
    }
  });

  const lg = el('g', { class: 'legend' }, svgEl);
  let lx = L;
  series.forEach((s, k) => {
    const color = SERIES_COLORS[k % SERIES_COLORS.length];
    el('line', { x1: lx, x2: lx + 18, y1: 16, y2: 16, stroke: color, class: 'theory' }, lg);
    el('circle', { cx: lx + 9, cy: 16, r: 4, fill: color }, lg);
    text(lg, lx + 24, 20, `M = ${s.M} of ${N}`);
    lx += 130;
  });
}

// start: threshold before round 1; rounds: dh_feed round records; exact: brute-force minimum.
export function renderTrajectory(svgEl, start, rounds, exact) {
  const W = 360, H = 200, L = 48, R = 12, T = 14, B = 34;
  const best = [start];
  for (const r of rounds) best.push(r.improved ? r.best_sample_cost : best[best.length - 1]);
  frame(svgEl, W, H, `Threshold per round: started at ${start.toFixed(2)}, now ${best[best.length - 1].toFixed(2)}; exact minimum ${exact.toFixed(2)}.`);
  const n = Math.max(1, best.length - 1);
  const lo = Math.min(exact, ...best), hi = Math.max(exact, ...best);
  const span = hi - lo || 1;
  const x = (i) => L + (i / n) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / span) * (H - T - B);

  const axes = el('g', { class: 'axes' }, svgEl);
  el('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, class: 'axis' }, axes);
  text(axes, L - 6, y(hi) + 4, hi.toFixed(1), { 'text-anchor': 'end' });
  text(axes, L - 6, y(lo) + 4, lo.toFixed(1), { 'text-anchor': 'end' });
  text(axes, (L + W - R) / 2, H - 8, `round (${rounds.length} so far)`, { 'text-anchor': 'middle' });
  el('line', { x1: L, x2: W - R, y1: y(exact), y2: y(exact), class: 'exact' }, svgEl);
  text(svgEl, W - R, y(exact) - 4, 'exact', { 'text-anchor': 'end', class: 'muted' });

  let d = '';
  best.forEach((v, i) => {
    d += i ? `H${x(i).toFixed(1)}V${y(v).toFixed(1)}` : `M${x(0).toFixed(1)},${y(v).toFixed(1)}`;
  });
  el('path', { d, class: 'traj' }, svgEl);
  best.forEach((v, i) => {
    const on = i === 0 || rounds[i - 1].improved;
    const c = el('circle', { cx: x(i), cy: y(v), r: on ? 3.5 : 2, class: on ? 'imp' : 'flat' }, svgEl);
    el('title', {}, c).textContent = i ? `round ${i}: j = ${rounds[i - 1].j}, threshold ${v.toFixed(2)}` : `start ${v.toFixed(2)}`;
  });
}
