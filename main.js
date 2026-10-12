import * as bridge from 'qopt.bridge';
import { renderMap, servingWarehouse } from './ui/map.js';
import { renderSweep, renderTrajectory, renderScaling } from './ui/plots.js';
import { renderClassical, renderCards, renderCounters } from './ui/cards.js';

const $ = (id) => document.getElementById(id);
const PARAMS = ['k', 'lam', 'tau', 'c', 'wh_weight', 'staff_cost', 'm'];
const DEFAULTS = { k: 2, lam: 10, tau: 10, c: 15000, wh_weight: 0.1, staff_cost: 3, m: 6 };   // 16 qubits: fast live runs
const SWEEP_SHOTS = 1000;
const IONQ_AGAIN = 'IonQ backends allow one job per run: press Stop, then Run, to submit again.';
// view: what the map, table and cards show; it always belongs to the params the classical table was computed for.
// last: the latest live run, kept for export.
// study: the scaling study of the loaded take; it stays on screen after live runs.
const state = { data: null, backendName: '', isLocal: true, busy: false, locked: false, seq: 0,
  view: null, last: null, study: null };
const errText = (e) => (e && e.message ? e.message : String(e));
// Costs are multiples of 1/8 (the model's integers over 8): shown exactly, never rounded to an arbitrary decimal.
const fc = (v) => String(Number(v));

function backendLabel() {
  try { return String(backend.name && typeof backend.name === 'function' ? backend.name() : backend.name); }
  catch { return 'unknown backend'; }
}

function setBanner(text, cls = '') { const b = $('banner'); b.textContent = text; b.className = 'banner ' + cls; }
function setStatus(text) { $('status').textContent = text; }
function showBackendBanner() {
  setBanner(state.isLocal
    ? `Backend: ${state.backendName}. Run quantum search runs one circuit live: the last search of the loop, which finds the cheapest placement. Load recorded take shows the full loop and sweep.`
    : `Backend: ${state.backendName}. IonQ backends accept one single-circuit job per run: Run quantum search runs the last search of the loop, which finds the cheapest placement. Load recorded take shows the full loop and sweep.`,
  state.isLocal ? '' : 'warn');
}

function setBusy(on) {
  state.busy = on;
  $('controls').disabled = on;
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
  renderClassical($('classical'), v.classical, { N: v.N, m: v.m, k: v.k, travelConst: v.travelConst, quantum: v.quantum },
    (sites) => showPlacement(sites, `Showing sites ${sites.join(', ')}.`));
}

function drawCards() {
  const v = state.view;
  renderCards($('cards'), { exact: v.exact, qBest: v.quantum ? v.quantum.cost : null, oracleCalls: v.quantum ? v.quantum.oracle_calls : null,
    N: v.N, m: v.m, k: v.k, maxDev: v.maxDev, localMinima: v.classical.local_search.local_minima, study: state.study });
}

// e.g. "96 placements scored (1 start + 71 Grover iterations + 24 rounds), best cost 8.625 (exact 8.625)".
// The start term is what the bridge's total leaves after Grover iterations and rounds (1 by the counting rule).
function dhSummary(dh, exact) {
  const r = dh.result, n = dh.rounds.length;
  return `${r.oracle_calls} placements scored (${r.oracle_calls - r.grover_iterations - n} start + ${r.grover_iterations} Grover iterations + ${n} rounds), `
    + `best cost ${fc(r.best_cost)} (exact ${fc(exact)})`;
}

// The m select names the C(m,k) placements; the selected option carries the bridge's count.
function labelM(k, nFeasible) {
  for (const o of $('m').options) {
    const slow = Number(o.value) <= 6 ? ', fast' : ', slow';
    o.textContent = `${o.value} candidate locations (C(${o.value},${k})${o.selected ? ` = ${nFeasible}` : ''} placements${slow})`;
  }
}

const sameParams = (a, b) => PARAMS.every((key) => Number(a[key]) === Number(b[key]));

// The recorded take: read by the bridge from data/recorded_local.json; validated there. Shows its scaling study when it has one.
async function loadTake() {
  const take = JSON.parse(await bridge.load_recorded.callPromising());
  if (take.study && take.study.length) {
    state.study = take.study;
    const c = renderScaling($('scaling'), take.study);
    $('scaling-cap').textContent = 'Mean placements scored by the loop over the study seeds against the square root of C(m,4), '
      + 'the placements of k = 4 stores opened among m candidate locations; '
      + `dashed: C(m,4), the brute-force count; fitted c = ${c.toFixed(1)}.`;
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
  c.innerHTML = '<h3>Minimum finding</h3><p class="lbl">Placements scored and best cost appear here during a run.</p>';
}

// Recompute the classical answers for p; the quantum panel is cleared because it no longer matches.
async function refreshClassical(p) {
  const seq = ++state.seq;
  const pj = JSON.stringify(p);
  setStatus('Computing the classical answers...');
  try {
    const table = JSON.parse(await bridge.cost_table_json.callPromising(pj));
    const classical = JSON.parse(await bridge.classical.callPromising(pj));
    if (seq !== state.seq) return false;   // a newer control change superseded this one
    state.view = { pj, p, m: p.m, k: table.k, N: table.n_feasible, travelConst: table.travel_constant, exact: table.best_cost, bestSites: table.best_sites,
      classical, quantum: null, maxDev: null };
    labelM(table.k, table.n_feasible);
    showPlacement(table.best_sites, `Showing the exact optimum, sites ${table.best_sites.join(', ')}.`);
    drawTable();
    drawCards();
    clearQuantum();
    setStatus(`k = ${table.k} stores opened among m = ${table.m} candidate locations: C(${table.m},${table.k}) = ${table.n_feasible} placements; exact minimum ${fc(table.best_cost)}.`);
    return true;
  } catch (e) {
    if (seq === state.seq) { setBanner('Classical step failed: ' + errText(e), 'err'); setStatus(''); }
    return false;
  }
}

async function showSweep(pj, planJson, countsJson) {
  const series = JSON.parse(await bridge.sweep_analyze_json.callPromising(pj, planJson, countsJson));
  renderSweep($('sweep'), series, state.view.N);
  state.view.maxDev = Math.max(...series.map((s) => s.max_dev));
  drawCards();
}

// Draw one state of the Durr-Hoyer loop: trajectory, counters, and (when known) the current best placement.
// own: {N, m, k, exact} of a recorded take at other parameters; it then goes to the counters panel only, not the map, table or cards.
function showDh(start, rounds, { result = null, sites = null, note = '', own = null } = {}) {
  const v = state.view;
  const ref = own || v;
  const last = rounds[rounds.length - 1];
  const best = result ? result.best_cost : last ? (last.improved ? last.best_sample_cost : last.T) : start;
  renderTrajectory($('traj'), start, rounds, ref.exact);
  renderCounters($('dh-counters'), { round: rounds.length, calls: result ? result.oracle_calls : last ? last.oracle_calls_cum : 0,
    N: ref.N, m: ref.m, k: ref.k, best, exact: ref.exact, done: !!result, note });
  if (own) return;
  if (sites) showPlacement(sites, `Showing the minimum-finding best placement, sites ${sites.join(', ')}.`);
  if (result) {
    v.quantum = { sites: sites || [], cost: result.best_cost, oracle_calls: result.oracle_calls };
    drawTable();
    drawCards();
  }
}

// The take's loop enters the table and cards only when its params (after defaults) equal the current ones;
// otherwise only the counters panel shows it, against its own placement count and exact minimum. Returns that exact minimum.
async function showDhFromTake(take, note) {
  const dh = take.dh;
  const tp = { ...DEFAULTS, ...take.params };
  let own = null;
  if (!sameParams(tp, state.view.p)) {
    const t = JSON.parse(await bridge.cost_table_json.callPromising(JSON.stringify(tp)));
    own = { N: t.n_feasible, m: t.m, k: t.k, exact: t.best_cost };
    note = `${note} This is the recorded take at its own parameters (m = ${tp.m}, k = ${tp.k}).`;
  }
  const sites = JSON.parse(await bridge.placement_sites.callPromising(dh.result.best_index));
  showDh(dh.rounds.length ? dh.rounds[0].T : dh.result.best_cost, dh.rounds, { result: dh.result, sites, note, own });
  return own ? own.exact : state.view.exact;
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
    setStatus('Building the last search of the minimum-finding loop...');
    const planJson = await bridge.sweep.callPromising(pj, true);
    const j = JSON.parse(planJson)[0].js[0];
    setStatus(`Running one circuit (${j} Grover iterations, ${SWEEP_SHOTS} shots) on ${state.backendName}: the threshold marks only the cheapest placement...`);
    submitted = true;
    const countsJson = await bridge.run_sweep.callPromising(backend, pj, SWEEP_SHOTS, true);
    state.last = { backend: state.backendName, pj, planJson, countsJson, dh: null };
    $('export').hidden = false;   // exportable even if a later step fails
    await showSweep(pj, planJson, countsJson);
    state.locked = !state.isLocal;
    const r = JSON.parse(await bridge.live_result.callPromising(pj, countsJson));
    showPlacement(r.sites, `Measured most often on ${state.backendName}: sites ${r.sites.join(', ')}.`);
    state.view.quantum = { sites: r.sites, cost: r.cost, oracle_calls: j };
    drawTable();
    drawCards();
    const pct = (x) => `${(100 * x).toFixed(1)}%`;
    setStatus(`Done on ${state.backendName}: measured sites ${r.sites.join(', ')}, cost ${fc(r.cost)}, `
      + `${r.exact ? 'the exact minimum' : `exact minimum ${fc(state.view.exact)}`}, in ${pct(r.share)} of shots (theory ${pct(r.theory)}).`
      + `${state.isLocal ? '' : ' ' + IONQ_AGAIN}`);
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
    await showSweep(JSON.stringify(take.params), JSON.stringify(take.plan), JSON.stringify(take.sweep_counts));
    const exact = take.dh ? await showDhFromTake(take, `Recorded take, seed ${take.dh.seed}, one measurement per round.`) : null;
    setStatus(`Showing the recorded take: ${take.backend}, ${take.date} (${take.source})`
      + (take.dh ? `. ${dhSummary(take.dh, exact)}.` : '.'));
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
    const text = await bridge.load_data.callPromising();
    state.data = JSON.parse(text);
    await refreshClassical(readParams());
  } catch (e) {
    setBanner('Could not load the data: ' + errText(e), 'err');
    setStatus('');
  } finally {
    setBusy(false);
  }
}

init();

// Keep this run alive. Qollab routes remote backend requests (IonQ simulators and hardware) only while main.js is
// still running; once it returns, the run is "Done" and remote backends are unreachable. Press Stop in the Qollab
// toolbar to end the session.
await new Promise(() => {});
