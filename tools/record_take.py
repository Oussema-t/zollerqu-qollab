"""Record a local Aer take for the page. Run from the repo root: python tools/record_take.py"""
import itertools
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qopt import bridge
from qopt.aer_run import aer_run
from qopt.classical import brute_force
from qopt.cost import model, model_table
from qopt.search import feasible_count, find_min

OUT = Path("data/recorded_local.json")
PARAMS = {"m": 8}
DH_SEED = 7
DH_SHOTS = 1
# Seeds per m (candidate locations). The in-circuit oracle makes a Grover iteration cost about 1 s on Aer at m = 8
# (20 qubits) and about 4x that at m = 10 (22 qubits), so the study keeps the recorder near 20 minutes: 20 seeds at
# m = 6, 5 at m = 8, 2 at m = 10, none at m = 12 (24 qubits).
STUDY_SEEDS = {6: 20, 8: 5, 10: 2}


def aer_seed(seed, round_index):
    """The one simulator seeding rule: loop seed s, round r -> Aer seed s * 1000 + r."""
    return seed * 1000 + round_index


def study_entry(data, m, seeds):
    mdl = model(data, {}, m)
    cost = model_table(mdl)
    exact = brute_force(cost)[1]
    runs = []
    for seed in seeds:
        rounds = itertools.count()
        res = find_min(mdl, lambda c, sh: aer_run(c, sh, seed=aer_seed(seed, next(rounds))),
                       np.random.default_rng(seed), shots=1)
        runs.append(res)
    calls = [r["oracle_calls"] for r in runs]
    return {"m": m, "N": feasible_count(cost), "seeds": len(runs), "calls_mean": float(np.mean(calls)),
            "calls_min": int(min(calls)), "calls_max": int(max(calls)),
            "grover_mean": float(np.mean([r["grover_iterations"] for r in runs])),
            "rounds_mean": float(np.mean([len(r["rounds"]) for r in runs])),
            "exact_fraction": float(np.mean([abs(r["best_cost"] - exact) <= 1e-9 for r in runs]))}


def scaling_study(data):
    study = []
    for m, seeds in STUDY_SEEDS.items():
        t0 = time.time()
        study.append(study_entry(data, m, range(seeds)))
        print(f"study m={m}: {seeds} seeds in {time.time() - t0:.1f}s", flush=True)
    print("study (one shot per round):")
    print("   m      N  seeds  calls_mean  min   max  grover_mean  rounds_mean  exact")
    for e in study:
        print(f"  {e['m']:2d}  {e['N']:5d}  {e['seeds']:5d}  {e['calls_mean']:10.1f}  {e['calls_min']:3d}  {e['calls_max']:4d}"
              f"  {e['grover_mean']:11.1f}  {e['rounds_mean']:11.1f}  {e['exact_fraction']:.2f}")
    return study


def main():
    t_start = time.time()
    data_text = Path("data/champaign.json").read_text(encoding="utf-8")
    bridge.load_data(data_text)
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
        dh_counts = aer_run([bridge.dh_next_circuit()], DH_SHOTS, seed=aer_seed(DH_SEED, n))[0]
        out = json.loads(bridge.dh_feed(json.dumps(dh_counts)))
        n += 1
        done = out["done"]
    res = out["result"]
    print(f"durr-hoyer: {n} rounds, {res['grover_iterations']} Grover iterations, {res['oracle_calls']} oracle calls"
          f" in {time.time() - t0:.1f}s, best {res['best_cost']:.4f}")
    dh = {"seed": DH_SEED, "dh_shots": DH_SHOTS, "rounds": res["rounds"],
          "result": {key: v for key, v in res.items() if key != "rounds"}}
    study = scaling_study(json.loads(data_text))
    take = {"source": "local-aer", "backend": "Aer statevector (local stand-in, Forte run pending)",
            "date": date.today().isoformat(), "params": PARAMS, "plan": json.loads(plan_json),
            "sweep_counts": counts, "dh": dh, "study": study}
    OUT.write_text(json.dumps(take), encoding="utf-8")
    print(f"recorder wall time {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
