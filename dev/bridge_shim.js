// dev/bridge_shim.js
// Stand-in for the Qollab module 'qopt.bridge' and the injected 'backend' global. Every call goes to dev/serve.py.
async function call(fn, args, as = 'text') {
  const r = await fetch(`/api/${fn}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ args }) });
  if (!r.ok) throw new Error(await r.text());
  return as === 'blob' ? r.blob() : as === 'json' ? r.json() : r.text();
}
// Each export is callable directly (bridge.x(...), the synchronous form on Qollab) and as bridge.x.callPromising(...);
// both return a Promise here, which main.js awaits.
const fn = (name, as) => {
  const f = (...args) => call(name, args, as);
  f.callPromising = f;
  return f;
};

export const load_data = fn('load_data');
export const cost_table_json = fn('cost_table_json');
export const classical = fn('classical');
export const sweep = fn('sweep');
export const sweep_analyze_json = fn('sweep_analyze_json');
export const live_result = fn('live_result');
// The backend argument is not JSON-serialisable; the server runs on its own harness backend, so it is dropped here.
export const run_sweep = (_backend, ...a) => call('run_sweep', a);
run_sweep.callPromising = run_sweep;
export const dh_run_round = (_backend, ...a) => call('dh_run_round', a);
dh_run_round.callPromising = dh_run_round;
export const dh_start = fn('dh_start');
export const dh_feed = fn('dh_feed');
export const placement_sites = fn('placement_sites');
export const load_recorded = fn('load_recorded');
export const export_take = fn('export_take');

globalThis.backend = { name: 'local dev harness (Aer statevector)' };   // main.js reads its name and passes it to the bridge
