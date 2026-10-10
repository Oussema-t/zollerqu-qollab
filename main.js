import * as bridge from 'qec.bridge';
import { LESSONS } from './ui/lessons.js';
import { renderPatches } from './ui/patches.js';
import { renderCards, renderCharts } from './ui/charts.js';

const $ = (id) => document.getElementById(id);
const state = { settings: null, metaJson: null, meta: null, countsJson: null, results: null, lesson: 1, backendName: '', isIonq: false, source: '' };

function backendLabel() {
  try { return String(backend.name && typeof backend.name === 'function' ? backend.name() : backend.name); }
  catch { return 'unknown backend'; }
}

function setBanner(text, cls = '') { const b = $('banner'); b.textContent = text; b.className = 'banner ' + cls; }
function setStatus(text) { $('status').textContent = text; }

function readSettings() {
  const shots = Math.min(2000, Math.max(100, parseInt($('shots').value, 10) || 500));
  const pauli = $('inject-pauli').value;
  const inject = pauli ? { pauli, qubit: parseInt($('inject-qubit').value, 10) } : null;
  return { shots, inject };
}

async function submit() {
  const settings = readSettings();
  state.settings = settings;
  $('submit').disabled = true;
  setStatus('Building 16 circuits...');
  try {
    state.metaJson = await bridge.build.callPromising(JSON.stringify(settings));
    state.meta = JSON.parse(state.metaJson);
    const circuits = await bridge.circuits.callPromising();
    setStatus(`Submitting one job with 16 circuits, ${settings.shots} shots each, to ${state.backendName}...`);
    const job = await backend.run(circuits, { shots: settings.shots });
    // Proxies to the same Python enum member are not === in JS, so compare by name.
    const statusName = async () => { const s = await job.status(); try { return String(s.name ?? s); } catch { return String(s); } };
    let status = await statusName();
    let ticks = 0;
    while (!['DONE', 'ERROR', 'CANCELLED'].some((k) => status.includes(k))) {
      await setTimeout(1000);
      status = await statusName();
      ticks += 1;
      setStatus(`Job ${status}... ${ticks}s. Noise-model simulators run one shot at a time, so this scales with shots.`);
    }
    if (!status.includes('DONE')) throw new Error('Job ended with status ' + status);
    const result = await job.result();
    const counts = [];
    for (let i = 0; i < state.meta.length; i++) {
      counts.push(Object.fromEntries(result.get_counts(i).toJs()));
    }
    state.countsJson = JSON.stringify(counts);
    state.source = `live run on ${state.backendName}`;
    await explore();
    $('export').hidden = false;
    if (!state.isIonq) $('submit').disabled = false;
    else setStatus('Done. IonQ backends allow one job per run: press Stop, then Run, to submit again.');
  } catch (e) {
    setBanner('Run failed: ' + (e && e.message ? e.message : String(e)), 'err');
    setStatus('');
    $('submit').disabled = false;
  }
}

async function loadRecorded() {
  setStatus('Loading recorded take...');
  const take = JSON.parse(await bridge.load_recorded.callPromising());
  state.settings = take.settings;
  state.metaJson = JSON.stringify(take.meta);
  state.meta = take.meta;
  state.countsJson = JSON.stringify(take.counts);
  state.source = `recorded take: ${take.backend}, ${take.date} (${take.source})`;
  await bridge.build.callPromising(JSON.stringify(take.settings));   // so circuit drawings match
  await explore();
}

async function explore() {
  setStatus('Analysing...');
  state.results = JSON.parse(await bridge.analyze.callPromising(state.metaJson, state.countsJson));
  $('tabs').hidden = false;
  $('panel').hidden = false;
  showLesson(state.lesson);
  setStatus(`Showing ${state.source}. Shots per circuit: ${state.settings.shots}.`);
  $('panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
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
}

async function drawCircuit() {
  if (!$('circuit-details').open || $('circuit-img').childElementCount) return;
  const row = state.results.find((r) => r.lesson === state.lesson && r.kind === 'enc');
  $('circuit-img').textContent = 'Drawing...';
  try {
    const fig = await bridge.draw.callPromising(row.index);
    const img = document.createElement('img');
    img.alt = `Physical circuit for lesson ${state.lesson}, ${row.name}`;
    img.src = URL.createObjectURL(fig);
    $('circuit-img').replaceChildren(img);
  } catch (e) {
    $('circuit-img').textContent = 'Drawing needs the Qiskit Visualizations library enabled in the Libraries list. ' + e;
  }
}

async function exportRun() {
  const text = await bridge.export_take.callPromising(state.backendName, JSON.stringify(state.settings), state.metaJson, state.countsJson, new Date().toISOString().slice(0, 10));
  console.log('Exported run JSON (also offered as a file in the console):');
  console.log(text.slice(0, 200) + '...');
}

function wire() {
  state.backendName = backendLabel();
  state.isIonq = /ionq|aria|forte/i.test(state.backendName);
  setBanner(state.isIonq
    ? `Backend: ${state.backendName}. IonQ backends accept one job per run, so all 16 circuits go in one submission.`
    : `Backend: ${state.backendName}. Local simulator, you can resubmit as often as you like.`, state.isIonq ? 'warn' : '');
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
