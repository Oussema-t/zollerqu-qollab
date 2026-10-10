// ui/cards.js: the classical comparison table, the result cards and the Durr-Hoyer counters card.
import { fitSqrtN } from './plots.js';

const f2 = (v) => (v == null ? '-' : Number(v).toFixed(2));
const same = (a, b) => a != null && b != null && Math.abs(a - b) < 1e-9;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// cl: classical(p) JSON; quantum: {sites, cost, oracle_calls} or null. Rows carry a button that shows the placement on the map.
export function renderClassical(tableEl, cl, { k, m, quantum }, onShow) {
  const N = 2 ** m;
  const rows = [
    ['Brute force (exact)', cl.brute_force.sites, f2(cl.brute_force.cost), `${cl.brute_force.evaluations} cost evaluations`],
    // Greedy evaluates every unopened site once per step (qopt/classical.py greedy): k*m - k(k-1)/2.
    ['Greedy', cl.greedy.sites, f2(cl.greedy.cost), `${k * m - (k * (k - 1)) / 2} cost evaluations`],
    ['Local search (20 restarts)', cl.local_search.sites, f2(cl.local_search.best_cost),
      `${cl.local_search.evaluations} cost evaluations, ${cl.local_search.local_minima} local minima`],
    cl.milp
      ? ['p-median MILP (travel term only)', cl.milp.sites, `${f2(cl.milp.travel)} min travel`, 'not the full cost']
      : ['p-median MILP (travel term only)', null, '-', 'not available here'],
  ];
  if (quantum) rows.push(['Quantum minimum finding', quantum.sites, f2(quantum.cost), `${quantum.oracle_calls} oracle calls (N = ${N})`]);
  tableEl.innerHTML = '<caption>Best placement per method, cost in the model\'s units (minutes plus staff, crowding and warehouse terms)</caption>'
    + '<thead><tr><th scope="col">Method</th><th scope="col">Sites</th><th scope="col">Cost</th><th scope="col">Work</th></tr></thead><tbody>'
    + rows.map(([name, sites, cost, work], i) => `<tr><th scope="row">${esc(name)}</th><td>${sites
      ? `<button class="link" data-row="${i}" aria-label="Show ${esc(name)} placement, sites ${sites.join(', ')}, on the map">${sites.join(', ')}</button>`
      : '-'}</td><td>${cost}</td><td>${esc(work)}</td></tr>`).join('') + '</tbody>';
  tableEl.querySelectorAll('button[data-row]').forEach((b) => b.addEventListener('click', () => onShow(rows[Number(b.dataset.row)][1])));
}

function card(big, lbl, cls = '', lblCls = '') {
  return `<div class="card ${cls}"><div class="big">${big}</div><div class="lbl ${lblCls}">${lbl}</div></div>`;
}

// s: {exact, qBest, oracleCalls, N, maxDev, localMinima, study}; any may be null before a run (study: from a loaded take).
export function renderCards(el, s) {
  const eq = same(s.qBest, s.exact);
  const top = s.study ? s.study.reduce((a, r) => (r.m > a.m ? r : a)) : null;
  el.innerHTML = [
    card(s.qBest == null ? '-' : `${f2(s.qBest)} ${eq ? '=' : '&ne;'} ${f2(s.exact)}`,
      s.qBest == null ? `Best cost found by quantum search vs exact ${f2(s.exact)}`
        : eq ? 'Best cost: quantum vs exact (equal)'
          : 'Best cost: quantum vs exact; not the exact minimum this time (paper guarantee: at least 1/2 within its budget); see the study for the success fraction',
      s.qBest == null ? '' : eq ? 'good' : 'amber', s.qBest == null || eq ? '' : 'amber'),
    card(s.oracleCalls == null ? '-' : `${s.oracleCalls} / ${s.N}`, 'Oracle calls vs N classical evaluations'),
    card(s.maxDev == null ? '-' : `${(100 * s.maxDev).toFixed(1)} pp`, 'Max deviation of measured from theory'),
    card(s.localMinima == null ? '-' : String(s.localMinima), 'Local minima seen by local search'),
    top ? card(`c = ${fitSqrtN(s.study).toFixed(1)}`, `Calls grow like c &radic;N (study fit); exact minimum at m = ${top.m}: `
      + `${Math.round(top.exact_fraction * top.seeds)} of ${top.seeds} seeds`) : '',
  ].join('');
}

// s: {round, calls, N, best, exact, done, note}
export function renderCounters(el, s) {
  const eq = same(s.best, s.exact);
  el.dataset.done = String(!!s.done);
  el.innerHTML = `<h3>Minimum finding ${s.done ? '(done)' : `(round ${s.round})`}</h3>`
    + `<dl><dt>Oracle calls (Grover iterations + one per round)</dt><dd><b>${s.calls}</b> of N = ${s.N}</dd>`
    + `<dt>Best cost</dt><dd><b class="${s.done ? (eq ? 'good' : 'amber') : ''}">${f2(s.best)}</b> vs exact ${f2(s.exact)}</dd></dl>`
    + (s.note ? `<p class="note">${esc(s.note)}</p>` : '');
}
