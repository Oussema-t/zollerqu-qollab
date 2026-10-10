// dev/bridge_shim.js
// Stand-in for the Qollab module 'qopt.bridge' and the injected 'backend' global. Every call goes to dev/serve.py.
async function call(fn, args, as = 'text') {
  const r = await fetch(`/api/${fn}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ args }) });
  if (!r.ok) throw new Error(await r.text());
  return as === 'blob' ? r.blob() : as === 'json' ? r.json() : r.text();
}
const fn = (name, as) => ({ callPromising: (...args) => call(name, args, as) });

export const load_data = fn('load_data');
export const cost_table_json = fn('cost_table_json');
export const classical = fn('classical');
export const sweep = fn('sweep');
export const sweep_circuits_py = fn('sweep_circuits_py', 'json');   // number of circuits built
export const sweep_analyze_json = fn('sweep_analyze_json');
export const dh_start = fn('dh_start');
export const dh_next_circuit = fn('dh_next_circuit', 'json');       // 1
export const dh_feed = fn('dh_feed');
export const placement_sites = fn('placement_sites');
export const load_recorded = fn('load_recorded');
export const export_take = fn('export_take');

globalThis.backend = {
  name: 'local dev harness (Aer statevector)',
  async run(_circuits, { shots }) {
    let counts = null, error = null;
    const done = call('run', [shots], 'json').then((c) => { counts = c; }, (e) => { error = e; });
    return {
      async status() { return { name: error ? 'ERROR' : counts ? 'DONE' : 'RUNNING' }; },
      async result() { await done; if (error) throw error; return { get_counts: async (i) => counts[i] }; },
    };
  },
};
