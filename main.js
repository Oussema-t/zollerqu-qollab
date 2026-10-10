import * as bridge from 'qopt.bridge';
import { renderMap, servingWarehouse } from './ui/map.js';
import { renderSweep, renderTrajectory, renderScaling } from './ui/plots.js';
import { renderClassical, renderCards, renderCounters } from './ui/cards.js';

const $ = (id) => document.getElementById(id);
const PARAMS = ['k', 'lam', 'tau', 'c', 'wh_weight', 'staff_cost', 'm'];
const DEFAULTS = { k: 4, lam: 10, tau: 10, c: 15000, wh_weight: 0.1, staff_cost: 3, m: 10 };
const SWEEP_SHOTS = 1000, DH_SHOTS = 1;   // the loop takes one measurement per round
const IONQ_DH = 'Durr-Hoyer loop shown from the recorded take: IonQ backends accept one job per run';
const IONQ_AGAIN = 'IonQ backends allow one job per run: press Stop, then Run, to submit again.';
// view: what the map, table and cards show; it always belongs to the params the classical table was computed for.
// last: the latest live run, kept for export.
// study: the scaling study of the loaded take; it stays on screen after live runs.
const state = { data: null, backendName: '', isLocal: true, busy: false, locked: false, seq: 0,
  view: null, last: null, study: null };
const toPlain = (x) => {
  if (!(x && typeof x.toJs === 'function')) return x;
  const v = x.toJs({ dict_converter: Object.fromEntries });
  return v instanceof Map ? Object.fromEntries(v) : v;
};
const errText = (e) => (e && e.message ? e.message : String(e));

// Works for the browser setTimeout(cb, ms), an injected setTimeout(cb, ms) returning a promise, and setTimeout(ms) returning a promise.
async function sleep(ms) {
  const start = Date.now();
  await new Promise((resolve) => {
    let r;
    try { r = setTimeout(resolve, ms); } catch { resolve(); return; }
    if (r && typeof r.then === 'function') r.then(resolve, resolve);
  });
  if (Date.now() - start < ms / 2) {
    try { const p = setTimeout(ms); if (p && typeof p.then === 'function') await p; } catch { /* ignore */ }
  }
  while (Date.now() - start < ms - 50) {
    await new Promise((r) => requestAnimationFrame(r));
  }
}

function backendLabel() {
  try { return String(backend.name && typeof backend.name === 'function' ? backend.name() : backend.name); }
  catch { return 'unknown backend'; }
}

function setBanner(text, cls = '') { const b = $('banner'); b.textContent = text; b.className = 'banner ' + cls; }
function setStatus(text) { $('status').textContent = text; }
function showBackendBanner() {
  setBanner(state.isLocal
    ? `Backend: ${state.backendName}. Local simulator: the sweep job and the minimum-finding loop (one job and one measurement per round) both run live.`
    : `Backend: ${state.backendName}. IonQ backends accept one job per run, so only the theory-match sweep runs live; the minimum-finding loop is shown from the recorded take.`,
  state.isLocal ? '' : 'warn');
}

function setBusy(on) {
  state.busy = on;
  $('controls').disabled = on;
  $('seed').disabled = on;
  $('run').disabled = on || state.locked;
  $('load-recorded').disabled = on;
  $('export').disabled = on;
}

function readParams() {
  const p = {};
  for (const key of PARAMS) p[key] = Number($(key).value);
  return p;
}

function writeParams(p) {
  for (const key of PARAMS) {
    $(key).value = String(key in p ? p[key] : DEFAULTS[key]);
    if ($(key + '-out')) $(key + '-out').value = $(key).value;
  }
}

function showPlacement(sites, what) {
  const v = state.view;
  renderMap($('map'), state.data, { m: v.m, chosen: sites, highlightWarehouse: sites.length ? servingWarehouse(state.data, sites) : null });
  $('map-what').textContent = what;
}

function drawTable() {
  const v = state.view;
  renderClassical($('classical'), v.classical, { k: v.k, m: v.m, quantum: v.quantum },
    (sites) => showPlacement(sites, `Showing sites ${sites.join(', ')}.`));
}

function drawCards() {
  const v = state.view;
  renderCards($('cards'), { exact: v.exact, qBest: v.quantum ? v.quantum.cost : null, oracleCalls: v.quantum ? v.quantum.oracle_calls : null,
    N: 2 ** v.m, maxDev: v.maxDev, localMinima: v.classical.local_search.local_minima, study: state.study });
}

// e.g. "294 oracle calls (214 Grover iterations over 80 rounds), best cost 62.73 (exact 62.73)"
function dhSummary(dh) {
  const r = dh.result;
  return `${r.oracle_calls} oracle calls (${r.grover_iterations} Grover iterations over ${dh.rounds.length} rounds), `
    + `best cost ${r.best_cost.toFixed(2)} (exact ${state.view.exact.toFixed(2)})`;
}

// The recorded take: fetched like champaign.json, validated by the bridge. Shows its scaling study when it has one.
async function loadTake() {
  const r = await fetch('data/recorded_local.json');
  if (!r.ok) throw new Error(`data/recorded_local.json: HTTP ${r.status}`);
  const take = JSON.parse(await bridge.load_recorded.callPromising(await r.text()));
  if (take.study && take.study.length) {
    state.study = take.study;
    const c = renderScaling($('scaling'), take.study);
    $('scaling-cap').textContent = 'Mean oracle calls of the one-measurement loop over the study seeds against sqrt N; '
      + `dashed: N evaluations of brute force; fitted c = ${c.toFixed(1)}.`;
    $('scaling-fig').hidden = false;
  }
  return take;
}

function clearQuantum() {
  $('sweep').replaceChildren();
  $('sweep').setAttribute('aria-label', 'No sweep yet');
  $('traj').replaceChildren();
  $('traj').setAttribute('aria-label', 'No minimum-finding run yet');
  const c = $('dh-counters');
  c.dataset.done = 'false';
  c.innerHTML = '<h3>Minimum finding</h3><p class="lbl">Oracle calls and best cost appear here during a run.</p>';
}

// Recompute the cost table and the classical answers for p; the quantum panel is cleared because it no longer matches.
async function refreshClassical(p) {
  const seq = ++state.seq;
  const pj = JSON.stringify(p);
  setStatus('Computing the cost table and the classical answers...');
  try {
    const table = JSON.parse(await bridge.cost_table_json.callPromising(pj));
    const classical = JSON.parse(await bridge.classical.callPromising(pj));
    if (seq !== state.seq) return false;   // a newer control change superseded this one
    state.view = { pj, m: p.m, k: table.k, exact: table.best_cost, bestSites: table.best_sites,
      classical, quantum: null, maxDev: null };
    showPlacement(table.best_sites, `Showing the exact optimum, sites ${table.best_sites.join(', ')}.`);
    drawTable();
    drawCards();
    clearQuantum();
    setStatus(`${table.n_feasible} feasible placements of ${2 ** p.m} bitstrings; exact minimum ${table.best_cost.toFixed(2)}.`);
    return true;
  } catch (e) {
    if (seq === state.seq) { setBanner('Classical step failed: ' + errText(e), 'err'); setStatus(''); }
    return false;
  }
}

// One backend job; returns the counts of its first n circuits as plain objects.
async function runJob(circuits, shots, n, label) {
  const job = await backend.run(circuits, { shots });
  // Proxies to the same Python enum member are not === in JS, so compare by name.
  // Up to 5 failed status calls in a row are tolerated so a network blip cannot lose the one IonQ job.
  let statusFailures = 0;
  const statusName = async () => {
    let s;
    try { s = await job.status(); } catch (e) {
      statusFailures += 1;
      if (statusFailures > 5) throw e;
      return 'retrying';
    }
    statusFailures = 0;
    try { return String(s.name ?? s); } catch { return String(s); }
  };
  const terminal = (st) => ['done', 'completed', 'error', 'failed', 'cancel'].some((k) => st.toLowerCase().includes(k));
  let status = await statusName();
  const t0 = Date.now();
  let shownAt = t0;
  while (!terminal(status)) {
    if (Date.now() - t0 > 3600 * 1000) throw new Error('Timed out waiting for the job');
    await sleep(500);
    status = await statusName();
    if (label && Date.now() - shownAt >= 5000) {   // at most one status update every 5 seconds
      shownAt = Date.now();
      setStatus(`${label}: job ${status}... ${Math.round((shownAt - t0) / 1000)}s.`);
    }
  }
  if (!['done', 'completed'].some((k) => status.toLowerCase().includes(k))) throw new Error('Job ended with status ' + status);
  const result = await job.result();
  const counts = [];
  for (let i = 0; i < n; i++) counts.push(toPlain(await result.get_counts(i)));
  return counts;
}

async function showSweep(pj, planJson, countsJson, m) {
  const series = JSON.parse(await bridge.sweep_analyze_json.callPromising(pj, planJson, countsJson));
  renderSweep($('sweep'), series, 2 ** m);
  state.view.maxDev = Math.max(...series.map((s) => s.max_dev));
  drawCards();
}

// Draw one state of the Durr-Hoyer loop: trajectory, counters, and (when known) the current best placement.
function showDh(start, rounds, { result = null, sites = null, note = '' } = {}) {
  const v = state.view;
  const last = rounds[rounds.length - 1];
  const best = result ? result.best_cost : last ? (last.improved ? last.best_sample_cost : last.T) : start;
  renderTrajectory($('traj'), start, rounds, v.exact);
  renderCounters($('dh-counters'), { round: rounds.length, calls: last ? last.oracle_calls_cum : 0, N: 2 ** v.m,
    best, exact: v.exact, done: !!result, note });
  if (sites) showPlacement(sites, `Showing the minimum-finding best placement, sites ${sites.join(', ')}.`);
  if (result) {
    v.quantum = { sites: sites || [], cost: result.best_cost, oracle_calls: result.oracle_calls };
    drawTable();
    drawCards();
  }
}

async function showDhFromTake(dh, note) {
  const sites = JSON.parse(await bridge.placement_sites.callPromising(dh.result.best_index));
  showDh(dh.rounds.length ? dh.rounds[0].T : dh.result.best_cost, dh.rounds, { result: dh.result, sites, note });
}

async function liveDh(pj, seed) {
  const start = JSON.parse(await bridge.dh_start.callPromising(pj, seed));
  const rounds = [];
  showDh(start.T, rounds, { sites: start.best_sites });
  setStatus(`Minimum finding: random start, threshold ${start.T.toFixed(2)}.`);
  for (let i = 0; i < 200; i++) {
    const circ = await bridge.dh_next_circuit.callPromising();
    const [counts] = await runJob([circ], DH_SHOTS, 1, `Minimum finding round ${i + 1}`);
    const fb = JSON.parse(await bridge.dh_feed.callPromising(JSON.stringify(counts)));
    rounds.push(fb.round);
    const r = fb.round;
    if (fb.done) {
      const sites = JSON.parse(await bridge.placement_sites.callPromising(fb.result.best_index));
      const { rounds: _all, ...result } = fb.result;   // the take format keeps rounds beside the result
      showDh(start.T, rounds, { result, sites });
      return { seed, rounds, result };
    }
    // The round record carries the new threshold but not its placement; the map keeps the last known one.
    const sites = 'best_index' in r ? JSON.parse(await bridge.placement_sites.callPromising(r.best_index)) : null;
    showDh(start.T, rounds, { sites });
    setStatus(`Round ${rounds.length}: j = ${r.j}, best sample ${r.best_sample_cost == null ? '-' : r.best_sample_cost.toFixed(2)}`
      + `${r.improved ? ' (new threshold)' : ''}, ${r.oracle_calls_cum} oracle calls so far.`);
  }
  throw new Error('Minimum finding did not stop within 200 rounds');
}

async function runQuantum() {
  const p = readParams();
  const pj = JSON.stringify(p);
  let submitted = false;
  setBusy(true);
  try {
    if (!state.view || state.view.pj !== pj) {
      clearTimeout(state.debounce);
      if (!(await refreshClassical(p))) return;
    } else {
      Object.assign(state.view, { quantum: null, maxDev: null });
      drawTable();
      drawCards();
      clearQuantum();
    }
    setStatus('Building the theory-match sweep...');
    const planJson = await bridge.sweep.callPromising(pj);
    const plan = JSON.parse(planJson);
    const n = plan.reduce((a, s) => a + s.js.length, 0);
    const circuits = await bridge.sweep_circuits_py.callPromising(pj);
    setStatus(`Submitting one job with ${n} circuits, ${SWEEP_SHOTS} shots each, to ${state.backendName}...`);
    submitted = true;
    const counts = await runJob(circuits, SWEEP_SHOTS, n, 'Theory-match sweep');
    const countsJson = JSON.stringify(counts);
    state.last = { backend: state.backendName, pj, planJson, countsJson, dh: null };
    $('export').hidden = false;   // exportable even if a later step fails
    await showSweep(pj, planJson, countsJson, p.m);
    if (state.isLocal) {
      const seed = Math.max(0, parseInt($('seed').value, 10) || 0);
      state.last.dh = await liveDh(pj, seed);
      setStatus(`Done: ${dhSummary(state.last.dh)}.`);
    } else {
      state.locked = true;
      const take = await loadTake();
      drawCards();
      const tp = JSON.stringify({ ...DEFAULTS, ...take.params });
      const note = IONQ_DH + (tp === pj ? '.' : ` (take parameters: ${tp}).`);
      await showDhFromTake(take.dh, note);
      setStatus(`Sweep done on ${state.backendName}. ${note} ${IONQ_AGAIN}`);
    }
  } catch (e) {
    setBanner('Run failed: ' + errText(e), 'err');
    if (state.last) console.log('Counts of the sweep, so the job is not lost:', state.last.countsJson);
    state.locked = !state.isLocal && submitted;
    setStatus(state.locked ? IONQ_AGAIN : '');
  } finally {
    setBusy(false);
  }
}

async function loadRecorded() {
  setBusy(true);
  setStatus('Loading recorded take...');
  try {
    const take = await loadTake();
    const p = { ...DEFAULTS, ...take.params };
    writeParams(p);
    clearTimeout(state.debounce);
    if (!(await refreshClassical(p))) return;
    $('export').hidden = true;   // export is for live runs, not the recorded take
    showBackendBanner();
    await showSweep(JSON.stringify(take.params), JSON.stringify(take.plan), JSON.stringify(take.sweep_counts), p.m);
    if (take.dh) await showDhFromTake(take.dh, `Recorded take, seed ${take.dh.seed}, one measurement per round.`);
    setStatus(`Showing the recorded take: ${take.backend}, ${take.date} (${take.source})`
      + (take.dh ? `. ${dhSummary(take.dh)}.` : '.'));
  } catch (e) {
    setBanner('Recorded take not available: ' + errText(e), 'err');
    setStatus('');
  } finally {
    setBusy(false);
  }
}

async function exportRun() {
  const run = state.last;
  try {
    const text = await bridge.export_take.callPromising(run.backend, run.pj, run.planJson, run.countsJson,
      JSON.stringify(run.dh), new Date().toISOString().slice(0, 10));
    console.log('Exported run JSON:', text.slice(0, 200) + '...');
    setStatus('Exported: the JSON file is offered as recorded_take.json in the Qollab console.');
  } catch (e) {
    setBanner('Export failed: ' + errText(e), 'err');
  }
}

async function init() {
  state.backendName = backendLabel();
  state.isLocal = !/ionq|aria|forte/i.test(state.backendName);
  showBackendBanner();
  for (const key of PARAMS) {
    $(key).addEventListener('input', () => {
      if ($(key + '-out')) $(key + '-out').value = $(key).value;
      clearTimeout(state.debounce);
      state.debounce = setTimeout(() => refreshClassical(readParams()), 150);
    });
  }
  $('run').addEventListener('click', runQuantum);
  $('load-recorded').addEventListener('click', loadRecorded);
  $('export').addEventListener('click', exportRun);
  clearQuantum();
  setBusy(true);
  setStatus('Loading Champaign County data...');
  try {
    const text = await (await fetch('data/champaign.json')).text();
    state.data = JSON.parse(text);
    await bridge.load_data.callPromising(text);
    await refreshClassical(readParams());
  } catch (e) {
    setBanner('Could not load the data: ' + errText(e), 'err');
    setStatus('');
  } finally {
    setBusy(false);
  }
}

init();
