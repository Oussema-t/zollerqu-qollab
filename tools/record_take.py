"""Record a local Aer take for the page. Run from the repo root: python tools/record_take.py"""
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qopt import bridge
from qopt.aer_run import aer_run

OUT = Path("data/recorded_local.json")
PARAMS = {"m": 10}
DH_SEED = 7


def main():
    bridge.load_data(Path("data/champaign.json").read_text(encoding="utf-8"))
    params = json.dumps(PARAMS)
    plan_json = bridge.sweep(params)
    circuits = bridge.sweep_circuits_py(params)
    t0 = time.time()
    counts = aer_run(circuits, 1000, seed=1234)
    print(f"sweep: {len(circuits)} circuits in {time.time() - t0:.1f}s")
    for s in json.loads(bridge.sweep_analyze_json(params, plan_json, json.dumps(counts))):
        print(f"  M={s['M']}: max deviation {s['max_dev']:.4f}")

    bridge.dh_start(params, DH_SEED)
    t0, done, n = time.time(), False, 0
    while not done:
        out = json.loads(bridge.dh_feed(json.dumps(aer_run([bridge.dh_next_circuit()], 100, seed=n)[0])))
        n += 1
        done = out["done"]
    res = out["result"]
    print(f"durr-hoyer: {n} rounds, {res['oracle_calls']} oracle calls in {time.time() - t0:.1f}s, best {res['best_cost']:.4f}")
    dh = {"seed": DH_SEED, "rounds": res["rounds"], "result": {key: v for key, v in res.items() if key != "rounds"}}
    take = {"source": "local-aer", "backend": "Aer statevector (local stand-in, Forte run pending)",
            "date": date.today().isoformat(), "params": PARAMS, "plan": json.loads(plan_json),
            "sweep_counts": counts, "dh": dh}
    OUT.write_text(json.dumps(take), encoding="utf-8")


if __name__ == "__main__":
    main()
