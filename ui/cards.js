// ui/cards.js: the classical comparison table, the result cards and the Durr-Hoyer counters card.
import { fitSqrtN } from './plots.js';

// Costs are multiples of 1/8: shown exactly (8.625, 62.5), never rounded to an arbitrary decimal.
const fc = (v) => (v == null ? '-' : String(Number(v)));
const same = (a, b) => a != null && b != null && Math.abs(a - b) < 1e-9;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// cl: classical(p) JSON; N: the C(m,k) placements (n_feasible); quantum: {sites, cost, oracle_calls} or null.
// Every count comes from the bridge. Rows carry a button that shows the placement on the map.
export function renderClassical(tableEl, cl, { N, m, k, travelConst, quantum }, onShow) {
  const bf = cl.brute_force, gr = cl.greedy, ls = cl.local_search;
  const rows = [
    ['Brute force (exact)', bf.sites, fc(bf.cost), `${bf.evaluations} ${bf.unit}`],
    ['Greedy', gr.sites, fc(gr.cost), `${gr.evaluations} ${gr.unit} (different unit)`, 'muted'],
    ['Local search (20 restarts)', ls.sites, fc(ls.best_cost),
      `${ls.evaluations} ${ls.unit} (repeats included), ${ls.local_minima} local minima`],
  ];
  if (quantum) rows.push(['Quantum minimum finding', quantum.sites, fc(quantum.cost),
    `${quantum.oracle_calls} placements scored (oracle calls: 1 start + Grover iterations + one measurement per round)`]);
  tableEl.innerHTML = `<caption>Best placement per method over the C(${m},${k}) = ${N} placements of k = ${k} stores opened among m = ${m} candidate locations. `
    + 'Cost: the sum of each open site\'s cost and each open pair\'s penalty, in multiples of 1/8 (travel, staff, crowding, warehouse); '
    + `the ${fc(travelConst)} travel minutes every placement shares are left out.</caption>`
    + '<thead><tr><th scope="col">Method</th><th scope="col">Sites</th><th scope="col">Cost</th><th scope="col">Work</th></tr></thead><tbody>'
    + rows.map(([name, sites, cost, work, cls], i) => `<tr><th scope="row">${esc(name)}</th><td>${sites
      ? `<button class="link" data-row="${i}" aria-label="Show ${esc(name)} placement, sites ${sites.join(', ')}, on the map">${sites.join(', ')}</button>`
      : '-'}</td><td>${cost}</td><td class="work${cls ? ' ' + cls : ''}">${esc(work)}</td></tr>`).join('') + '</tbody>';
  tableEl.querySelectorAll('button[data-row]').forEach((b) => b.addEventListener('click', () => onShow(rows[Number(b.dataset.row)][1])));
}

function card(big, lbl, cls = '', lblCls = '') {
  return `<div class="card ${cls}"><div class="big">${big}</div><div class="lbl ${lblCls}">${lbl}</div></div>`;
}

// s: {exact, qBest, oracleCalls, N, m, k, maxDev, localMinima, study}; any may be null before a run (study: from a loaded take).
export function renderCards(el, s) {
  const eq = same(s.qBest, s.exact);
  const top = s.study ? s.study.reduce((a, r) => (r.m > a.m ? r : a)) : null;
  el.innerHTML = [
    card(s.qBest == null ? '-' : `${fc(s.qBest)} ${eq ? '=' : '&ne;'} ${fc(s.exact)}`,
      s.qBest == null ? `Best cost found by quantum search vs exact ${fc(s.exact)}`
        : eq ? 'Best cost: quantum vs exact (equal)'
          : 'Best cost: quantum vs exact; not the exact minimum this time (paper guarantee: at least 1/2 within its budget); see the study for the success fraction',
      s.qBest == null ? '' : eq ? 'good' : 'amber', s.qBest == null || eq ? '' : 'amber'),
    card(s.oracleCalls == null ? '-' : `${s.oracleCalls} / ${s.N}`, `placements scored: quantum loop vs brute force over all C(${s.m},${s.k}) placements`),
    card(s.maxDev == null ? '-' : `${(100 * s.maxDev).toFixed(1)} pp`, 'Max deviation of measured from theory'),
    card(s.localMinima == null ? '-' : String(s.localMinima), 'Local minima seen by local search'),
    top ? card(`c = ${fitSqrtN(s.study).toFixed(1)}`, `Placements scored by the loop grow like c &radic;C(m,4) (study fit, k = 4 stores opened); exact minimum at m = ${top.m}: `
      + `${Math.round(top.exact_fraction * top.seeds)} of ${top.seeds} seeds`) : '',
  ].join('');
}

// s: {round, calls, N, m, k, best, exact, done, note}
export function renderCounters(el, s) {
  const eq = same(s.best, s.exact);
  el.dataset.done = String(!!s.done);
  el.innerHTML = `<h3>Minimum finding ${s.done ? '(done)' : `(round ${s.round})`}</h3>`
    + `<dl><dt>Placements scored (1 start + Grover iterations + one per round)</dt><dd><b>${s.calls}</b> of C(${s.m},${s.k}) = ${s.N}</dd>`
    + `<dt>Best cost</dt><dd><b class="${s.done ? (eq ? 'good' : 'amber') : ''}">${fc(s.best)}</b> vs exact ${fc(s.exact)}</dd></dl>`
    + (s.note ? `<p class="note">${esc(s.note)}</p>` : '');
}
