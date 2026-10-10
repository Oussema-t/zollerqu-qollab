// dev/bridge_shim.js
// Stand-in for the Qollab module 'qec.bridge' and the injected 'backend' global. Every call goes to dev/serve.py.
async function call(fn, args, as = 'text') {
  const r = await fetch(`/api/${fn}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ args }) });
  if (!r.ok) throw new Error(await r.text());
  return as === 'blob' ? r.blob() : as === 'json' ? r.json() : r.text();
}
const fn = (name, as) => ({ callPromising: (...args) => call(name, args, as) });

export const build = fn('build');
export const circuits = fn('circuits', 'json');
export const circuits_subset = fn('circuits_subset', 'json');
export const analyze = fn('analyze');
export const draw = fn('draw', 'blob');          // Blob of SVG, so URL.createObjectURL works as on Qollab
export const load_recorded = fn('load_recorded');
export const export_take = fn('export_take');
export const spacetime = fn('spacetime');

globalThis.backend = {
  name: 'local dev harness (Aer, depolarizing stand-in)',
  async run(_circuits, { shots }) {
    let counts = null, error = null;
    const done = call('run', [shots], 'json').then((c) => { counts = c; }, (e) => { error = e; });
    return {
      async status() { return { name: error ? 'ERROR' : counts ? 'DONE' : 'RUNNING' }; },
      async result() { await done; if (error) throw error; return { get_counts: async (i) => counts[i] }; },
    };
  },
};
