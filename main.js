import * as bridge from 'qec.bridge';
import { LESSONS } from './ui/lessons.js';
import { renderPatches } from './ui/patches.js';
import { renderCards, renderCharts } from './ui/charts.js';
import { openStory } from './ui/story.js';

const $ = (id) => document.getElementById(id);
// settings, meta, results and sources always change together (in explore); lastRun is the latest live job, kept for export.
// sources: lesson -> 'live' | 'live+recorded' | 'recorded' | 'missing' (a lesson too big for this backend is filled from the recorded take,
// or left empty when the take is not available); takeSettings: the settings of that take.
const state = { settings: null, meta: null, results: null, source: '', sources: {}, takeSettings: null, lesson: 1, backendName: '', maxQubits: 24, isIonq: false, submitLocked: false, busy: false, lastRun: null, story: null };
const IONQ_AGAIN = 'IonQ backends allow one job per run: press Stop, then Run, to submit again.';
const IONQ_LIVE = 'On IonQ only the teleportation circuit (lesson 4, X readout) runs live: one circuit per job, one job per run';
const errText = (e) => (e && e.message ? e.message : String(e));

function setBusy(on) {
  state.busy = on;
  $('submit').disabled = on || state.submitLocked;
  $('load-recorded').disabled = on;
  $('export').disabled = on;
}

// Qubit ceilings of the Qollab backends (ceilings, not guarantees).
function maxQubitsFor(name) {
  return /forte|harness|aer/i.test(name) ? 29 : /aria|braket/i.test(name) ? 25 : 24;
}

function backendLabel() {
  try { return String(backend.name && typeof backend.name === 'function' ? backend.name() : backend.name); }
  catch { return 'unknown backend'; }
}

function setBanner(text, cls = '') { const b = $('banner'); b.textContent = text; b.className = 'banner ' + cls; }
function setStatus(text) { $('status').textContent = text; }
function showBackendBanner() {
  setBanner(state.isIonq
    ? (state.maxQubits < 27
      ? `Backend: ${state.backendName}. On IonQ only the 27-qubit teleportation circuit runs live, and it does not fit here: pick the Forte simulator, or press Load recorded take.`
      : `Backend: ${state.backendName}. ${IONQ_LIVE}. Everything else comes from the recorded take.`)
    : `Backend: ${state.backendName}. Local simulator, you can resubmit as often as you like.`
      + (state.maxQubits < 27 ? ' Lesson 4 needs 27 qubits, so on this backend it is shown from the recorded take; pick a Forte simulator to run it live.' : ''),
  state.isIonq ? 'warn' : '');
}

// Status line for the lesson on screen: what is shown, and why a lesson comes from the recorded take.
function lessonStatus() {
  let s = `Showing ${state.source}. Shots per circuit: ${state.settings.shots}.`;
  const src = state.sources[state.lesson];
  const takeNote = () => { const t = state.takeSettings, inj = t.inject; return `${t.shots} shots, ${inj ? `${inj.pauli} injected on q${inj.qubit}` : 'no injection'}`; };
  if (state.isIonq && (src === 'recorded' || src === 'missing')) {
    s += ` ${IONQ_LIVE}; lesson ${state.lesson} ` + (src === 'missing' ? 'is empty: the recorded take could not be loaded.' : `shows the recorded take (${takeNote()}).`);
  } else if (state.isIonq && state.lesson === 4) {
    s += ` ${IONQ_LIVE}; ` + (src === 'live+recorded' ? `the other lesson 4 rows come from the recorded take (${takeNote()}).` : 'the other lesson 4 rows are empty: the recorded take could not be loaded.');
  } else if (src === 'recorded' || src === 'missing') {
    const need = Math.max(...state.meta.filter((m) => m.lesson === state.lesson).map((m) => m.nqubits));
    s += ` Lesson ${state.lesson} needs ${need} qubits; this backend offers ${state.maxQubits}`;
    if (src === 'missing') s += ', and the recorded take could not be loaded.';
    else {
      s += `. Showing the recorded take for this lesson (${takeNote()}).`;
    }
  }
  if (state.submitLocked) s += ' ' + IONQ_AGAIN;
  return s;
}

function readSettings() {
  const shots = Math.min(2000, Math.max(100, parseInt($('shots').value, 10) || 500));
  const pauli = $('inject-pauli').value;
  const inject = pauli ? { pauli, qubit: parseInt($('inject-qubit').value, 10) } : null;
  return { shots, inject };
}

async function submit() {
  closeStory();
  const settings = readSettings();
  let submitted = false;
  let run = null;   // this submission's settings, metadata and counts once collected
  setBusy(true);
  setStatus('Building 16 circuits...');
  try {
    // Kept local until the run succeeds, so a failed run leaves the last good results intact.
    const metaJson = await bridge.build.callPromising(JSON.stringify(settings));
    const meta = JSON.parse(metaJson);
    // A lesson with any circuit wider than the backend stays out of the job whole, so its cards never mix
    // live and recorded rows; its counts come from the recorded take.
    const tooBig = new Set(meta.filter((m) => m.nqubits > state.maxQubits).map((m) => m.lesson));
    const included = [], excluded = [];
    const live = (m) => (state.isIonq ? m.lesson === 4 && m.name === 'enc_X' && m.nqubits <= state.maxQubits : !tooBig.has(m.lesson));
    meta.forEach((m, i) => (live(m) ? included : excluded).push(i));
    let take = null;
    if (excluded.length) {
      try { take = JSON.parse(await bridge.load_recorded.callPromising()); } catch (e) { console.log('Recorded take not available:', errText(e)); }
    }
    if (!included.length) throw new Error('The 27-qubit teleportation circuit does not fit this backend: pick the Forte simulator, or press Load recorded take.');
    setStatus(`Running one job with ${included.length} of ${meta.length} circuits, ${settings.shots} shots each, on ${state.backendName}. Noise-model simulators run one shot at a time, so this can take minutes.`);
    submitted = true;
    const sub = JSON.parse(await bridge.run_subset.callPromising(backend, JSON.stringify(included), settings.shots));
    const counts = meta.map(() => ({}));
    for (let k = 0; k < included.length; k++) counts[included[k]] = sub[k];
    const sources = {};
    meta.forEach((m) => { sources[m.lesson] = 'live'; });
    // The circuits not run live are filled from the take only when all of a lesson's ones have a matching entry; else they stay empty.
    // A lesson with no live circuit is marked recorded (or missing); lesson 4 on IonQ stays live, its other rows recorded.
    const matches = (i) => { const t = take && take.meta && take.meta[i]; return !!t && t.name === meta[i].name && t.lesson === meta[i].lesson; };
    for (const lesson of new Set(excluded.map((i) => meta[i].lesson))) {
      const mine = excluded.filter((i) => meta[i].lesson === lesson);
      const ok = mine.every(matches);
      if (ok) for (const i of mine) counts[i] = take.counts[i];
      else console.log(`No recorded counts for every circuit of lesson ${lesson}; its circuits are left empty.`);
      if (!included.some((i) => meta[i].lesson === lesson)) sources[lesson] = ok ? 'recorded' : 'missing';
      else if (ok) sources[lesson] = 'live+recorded';
    }
    run = { settings, metaJson, countsJson: JSON.stringify(counts) };
    state.lastRun = run;
    $('export').hidden = false;   // exportable even if the analysis below fails
    showBackendBanner();
    await explore(settings, metaJson, run.countsJson, `live run on ${state.backendName}`, sources, take && take.settings);
    state.submitLocked = state.isIonq;
    setBusy(false);
    if (state.isIonq) setStatus('Done. ' + lessonStatus());
  } catch (e) {
    setBanner('Run failed: ' + errText(e), 'err');
    if (run) console.log('Counts of the failed run, so the job is not lost:', run.countsJson);
    state.submitLocked = state.isIonq && submitted;
    setBusy(false);
    setStatus(state.submitLocked ? IONQ_AGAIN : '');
  }
}

async function loadRecorded() {
  closeStory();
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

async function explore(settings, metaJson, countsJson, source, sources = {}, takeSettings = null) {
  setStatus('Analysing...');
  const results = JSON.parse(await bridge.analyze.callPromising(metaJson, countsJson));
  const meta = JSON.parse(metaJson);
  Object.assign(state, { settings, meta, results, source, sources, takeSettings });   // together, so a tab click never mixes old results with new settings
  $('tabs').hidden = false;
  $('panel').hidden = false;
  showLesson(state.lesson);
  setStatus(lessonStatus());
  $('setup').open = false;
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  $('panel').scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
}

function showLesson(n) {
  closeStory();   // a story never shows a stale lesson
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
}

function storyClosed(refocus = true) {
  $('story').hidden = true;
  document.body.style.overflow = '';
  state.story = null;
  if (refocus) $('story-open').focus();
}

function closeStory() {
  if (!state.story) return;
  const hadFocus = $('story').contains(document.activeElement);
  state.story.close();   // empties the story root; the story itself calls onClose only from its Back button and Escape
  storyClosed(hadFocus);
}

async function watchStory() {
  closeStory();
  const lesson = state.lesson;
  const row = state.results.find((r) => r.lesson === lesson && r.kind === 'enc');
  const btn = $('story-open');
  btn.disabled = true;
  setStatus(`Building the story for lesson ${lesson}...`);
  try {
    const layout = JSON.parse(await bridge.spacetime.callPromising(row.index));
    if (state.lesson !== lesson) return;   // the tab changed while loading
    state.story = openStory($('story'), {
      lesson, rows: state.results.filter((r) => r.lesson === lesson), layout, settings: state.settings, onClose: storyClosed,
    });
    $('story').hidden = false;
    document.body.style.overflow = 'hidden';
  } catch (e) {
    setBanner('Story unavailable: ' + errText(e), 'err');
  } finally {
    btn.disabled = false;
    if (!state.busy) setStatus(lessonStatus());
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
  state.maxQubits = maxQubitsFor(state.backendName);
  state.isIonq = /ionq|aria|forte/i.test(state.backendName);
  showBackendBanner();
  $('submit').addEventListener('click', submit);
  $('load-recorded').addEventListener('click', loadRecorded);
  $('export').addEventListener('click', exportRun);
  $('circuit-details').addEventListener('toggle', drawCircuit);
  $('story-open').addEventListener('click', watchStory);
  const pick = (n) => { showLesson(n); if (!state.busy) setStatus(lessonStatus()); };
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => pick(Number(t.dataset.lesson)));
    t.addEventListener('keydown', (e) => {
      const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!dir) return;
      e.preventDefault();
      const next = tabs[(i + dir + tabs.length) % tabs.length];
      next.focus();
      pick(Number(next.dataset.lesson));
    });
  });
  setStatus('Choose shots, optionally inject an error for lesson 1, then submit. Or load the recorded take.');
}

wire();
// Keep this run alive. Qollab routes remote backend requests (IonQ simulators and hardware) only while main.js is
// still running; once it returns, the run is "Done" and remote backends are unreachable. Press Stop in the Qollab
// toolbar to end the session.
await new Promise(() => {});
