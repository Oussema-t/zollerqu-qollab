import * as bridge from 'qec.bridge';
import { LESSONS } from './ui/lessons.js';
import { renderPatches } from './ui/patches.js';
import { renderCards, renderCharts } from './ui/charts.js';
import { renderSpacetime } from './ui/spacetime.js';

const $ = (id) => document.getElementById(id);
// settings and results always change together (in explore); lastRun is the latest live job, kept for export.
const state = { settings: null, results: null, source: '', lesson: 1, backendName: '', isIonq: false, submitLocked: false, lastRun: null, spacetime: null, spacetimeToken: 0 };
const IONQ_AGAIN = 'IonQ backends allow one job per run: press Stop, then Run, to submit again.';
const toPlain = (x) => {
  if (!(x && typeof x.toJs === 'function')) return x;
  const v = x.toJs();
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

function setBusy(on) {
  $('submit').disabled = on || state.submitLocked;
  $('load-recorded').disabled = on;
  $('export').disabled = on;
}

function backendLabel() {
  try { return String(backend.name && typeof backend.name === 'function' ? backend.name() : backend.name); }
  catch { return 'unknown backend'; }
}

function setBanner(text, cls = '') { const b = $('banner'); b.textContent = text; b.className = 'banner ' + cls; }
function setStatus(text) { $('status').textContent = text; }
function showBackendBanner() {
  setBanner(state.isIonq
    ? `Backend: ${state.backendName}. IonQ backends accept one job per run, so all 16 circuits go in one submission.`
    : `Backend: ${state.backendName}. Local simulator, you can resubmit as often as you like.`, state.isIonq ? 'warn' : '');
}

function readSettings() {
  const shots = Math.min(2000, Math.max(100, parseInt($('shots').value, 10) || 500));
  const pauli = $('inject-pauli').value;
  const inject = pauli ? { pauli, qubit: parseInt($('inject-qubit').value, 10) } : null;
  return { shots, inject };
}

async function submit() {
  const settings = readSettings();
  let submitted = false;
  let run = null;   // this submission's settings, metadata and counts once collected
  setBusy(true);
  setStatus('Building 16 circuits...');
  try {
    // Kept local until the run succeeds, so a failed run leaves the last good results intact.
    const metaJson = await bridge.build.callPromising(JSON.stringify(settings));
    const meta = JSON.parse(metaJson);
    const circuits = await bridge.circuits.callPromising();
    setStatus(`Submitting one job with 16 circuits, ${settings.shots} shots each, to ${state.backendName}...`);
    submitted = true;
    const job = await backend.run(circuits, { shots: settings.shots });
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
    let ticks = 0;
    const t0 = Date.now();
    let shownAt = t0;
    while (!terminal(status)) {
      if (ticks >= 3600) throw new Error('Timed out waiting for the job');
      await sleep(1000);
      status = await statusName();
      ticks += 1;
      if (Date.now() - shownAt >= 5000) {   // at most one status update every 5 seconds
        shownAt = Date.now();
        setStatus(`Job ${status}... ${Math.round((shownAt - t0) / 1000)}s. Noise-model simulators run one shot at a time, so this scales with shots.`);
      }
    }
    if (!['done', 'completed'].some((k) => status.toLowerCase().includes(k))) throw new Error('Job ended with status ' + status);
    const result = await job.result();
    const counts = [];
    for (let i = 0; i < meta.length; i++) {
      counts.push(toPlain(await result.get_counts(i)));
    }
    run = { settings, metaJson, countsJson: JSON.stringify(counts) };
    state.lastRun = run;
    $('export').hidden = false;   // exportable even if the analysis below fails
    showBackendBanner();
    await explore(settings, metaJson, run.countsJson, `live run on ${state.backendName}`);
    state.submitLocked = state.isIonq;
    setBusy(false);
    if (state.isIonq) setStatus('Done. ' + IONQ_AGAIN);
  } catch (e) {
    setBanner('Run failed: ' + errText(e), 'err');
    if (run) console.log('Counts of the failed run, so the job is not lost:', run.countsJson);
    state.submitLocked = state.isIonq && submitted;
    setBusy(false);
    setStatus(state.submitLocked ? IONQ_AGAIN : '');
  }
}

async function loadRecorded() {
  setBusy(true);
  setStatus('Loading recorded take...');
  try {
    const take = JSON.parse(await bridge.load_recorded.callPromising());
    $('export').hidden = true;   // export is for live runs, not the recorded take
    await bridge.build.callPromising(JSON.stringify(take.settings));   // so circuit drawings match
    showBackendBanner();
    await explore(take.settings, JSON.stringify(take.meta), JSON.stringify(take.counts),
      `recorded take: ${take.backend}, ${take.date} (${take.source})`);
  } catch (e) {
    setBanner('Recorded take not available: ' + errText(e), 'err');
    setStatus('');
  } finally {
    setBusy(false);
  }
}

async function explore(settings, metaJson, countsJson, source) {
  setStatus('Analysing...');
  const results = JSON.parse(await bridge.analyze.callPromising(metaJson, countsJson));
  Object.assign(state, { settings, results, source });   // together, so a tab click never mixes old results with new settings
  $('tabs').hidden = false;
  $('panel').hidden = false;
  showLesson(state.lesson);
  setStatus(`Showing ${state.source}. Shots per circuit: ${state.settings.shots}.`);
  $('setup').open = false;
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  $('panel').scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
}

function showLesson(n) {
  state.lesson = n;
  document.querySelectorAll('[role="tab"]').forEach((t) => {
    const on = Number(t.dataset.lesson) === n;
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
  });
  $('panel').setAttribute('aria-labelledby', 'tab-' + n);
  const rows = state.results.filter((r) => r.lesson === n);
  $('lesson-text').innerHTML = `<h2>${LESSONS[n].title}</h2>` + LESSONS[n].html;
  renderPatches($('patches'), n, rows, state.settings);
  renderCards($('cards'), rows);
  renderCharts($('charts'), rows);
  $('circuit-img').innerHTML = '';
  $('circuit-details').open = false;
  mountSpacetime(n, rows);
}

async function mountSpacetime(lesson, rows) {
  const token = ++state.spacetimeToken;
  if (state.spacetime) { state.spacetime.dispose(); state.spacetime = null; }
  const root = $('spacetime');
  const row = rows.find((r) => r.kind === 'enc');
  if (!row) { root.replaceChildren(); return; }
  root.textContent = 'Building the spacetime view...';
  try {
    const L = JSON.parse(await bridge.spacetime.callPromising(row.index));
    if (token !== state.spacetimeToken) return;   // the tab changed meanwhile
    state.spacetime = await renderSpacetime(root, L, row);
    if (token !== state.spacetimeToken) { state.spacetime.dispose(); state.spacetime = null; }
  } catch (e) {
    if (token === state.spacetimeToken) root.textContent = 'Spacetime view unavailable: ' + errText(e);
  }
}

async function drawCircuit() {
  if (!$('circuit-details').open || $('circuit-img').childElementCount) return;
  const lesson = state.lesson;
  const row = state.results.find((r) => r.lesson === lesson && r.kind === 'enc');
  $('circuit-img').textContent = 'Drawing...';
  try {
    const fig = await bridge.draw.callPromising(row.index);
    if (state.lesson !== lesson) return;   // the tab changed while drawing
    const img = document.createElement('img');
    img.alt = `Physical circuit for lesson ${lesson}, ${row.name}`;
    img.src = URL.createObjectURL(fig);
    $('circuit-img').replaceChildren(img);
  } catch (e) {
    if (state.lesson !== lesson) return;
    const msg = 'Could not draw the circuit on this backend: ' + errText(e);
    $('circuit-img').textContent = /visualization|matplotlib/i.test(msg)
      ? msg + ' Enable Qiskit Visualizations in the Libraries list.'
      : msg;
  }
}

async function exportRun() {
  const run = state.lastRun;
  try {
    const text = await bridge.export_take.callPromising(state.backendName, JSON.stringify(run.settings), run.metaJson, run.countsJson, new Date().toISOString().slice(0, 10));
    console.log('Exported run JSON (also offered as a file in the console):');
    console.log(text.slice(0, 200) + '...');
    setStatus('Exported: the JSON file is in the Qollab console (recorded_take.json).');
  } catch (e) {
    setBanner('Export failed: ' + errText(e), 'err');
  }
}

function wire() {
  state.backendName = backendLabel();
  state.isIonq = /ionq|aria|forte/i.test(state.backendName);
  showBackendBanner();
  $('submit').addEventListener('click', submit);
  $('load-recorded').addEventListener('click', loadRecorded);
  $('export').addEventListener('click', exportRun);
  $('circuit-details').addEventListener('toggle', drawCircuit);
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => showLesson(Number(t.dataset.lesson)));
    t.addEventListener('keydown', (e) => {
      const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!dir) return;
      e.preventDefault();
      const next = tabs[(i + dir + tabs.length) % tabs.length];
      next.focus();
      showLesson(Number(next.dataset.lesson));
    });
  });
  setStatus('Choose shots, optionally inject an error for lesson 1, then submit. Or load the recorded take.');
}

wire();
