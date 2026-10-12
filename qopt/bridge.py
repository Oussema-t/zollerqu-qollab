"""JSON-string entry points for the browser page (Pyodide). State lives in _STATE.

params_json: {"m": 10, "k": 4, "c": ..., "staff_cost": ..., "tau": ..., "lam": ..., "wh_weight": ...};
every key is optional (m defaults to 8, the rest to qopt.cost.DEFAULTS:
k=4, c=15000, staff_cost=3.0, tau=10.0, lam=10.0, wh_weight=0.1).

load_data(json_text=None) parses json_text if given, else reads data/champaign.json, keeps the data in _STATE and returns the data text;
load_recorded(text=None) returns text unchanged (after JSON validation) if given, else the text of data/recorded_local.json; export_take(...) returns (and, inside Qollab,
downloads) a JSON take {source, backend, date, params, plan, sweep_counts, dh}.
run_sweep(backend, params_json, shots, live) and dh_run_round(backend, shots) run the circuits on the backend
object the page passes in and return the counts as JSON, so the page never handles a circuit.

Costs are the quadratic model of qopt.cost (multiples of 1/8). The circuits are built from its per-site and
per-pair integers; the cost of every placement is computed only for the classical baselines and the checks.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

from qopt import classical as cl
from qopt import search
from qopt.cost import DEFAULTS, model, model_table

_STATE = {}
DEFAULT_M = 8    # one Grover iteration at m = 10 is far too slow on the built-in simulator (tools/time_grover.py)
DATA = "data/champaign.json"
RECORDED = "data/recorded_local.json"
LOCAL_SEARCH_RESTARTS = 20


def _sites(index):
    return [i for i in range(index.bit_length()) if index >> i & 1]


def _clean(x):
    """JSON-safe: +inf becomes null (JSON has no Infinity)."""
    return None if isinstance(x, float) and not math.isfinite(x) else x


def _setup(params_json):
    """m, cost params, the coefficient model (for the circuits) and the cost of every placement (classical side)."""
    params = json.loads(params_json)
    m = params["m"] if "m" in params else DEFAULT_M
    cost_params = {key: v for key, v in params.items() if key != "m"}
    mdl = model(_STATE["data"], cost_params, m)
    return m, cost_params, mdl, model_table(mdl)


def load_data(json_text=None):
    if json_text is None:
        path = Path(DATA)
        if not path.exists():
            path = Path(__file__).resolve().parents[1] / DATA
        with open(path, encoding="utf-8") as f:
            json_text = f.read()
    _STATE["data"] = json.loads(json_text)
    return json_text


def cost_table_json(params_json):
    """Exact classical answer; travel_constant is the model's constant travel term (the same for every placement,
    not searched over), to the nearest 1/8."""
    m, params, mdl, cost = _setup(params_json)
    idx, best, _ = cl.brute_force(cost)
    n = search.feasible_count(cost)
    return json.dumps({"m": m, "k": {**DEFAULTS, **params}["k"], "N": n, "n_feasible": n,
                       "best_index": idx, "best_cost": best, "best_sites": _sites(idx),
                       "travel_constant": round(mdl["const"] * mdl["scale"]) / mdl["scale"]})


def classical(params_json):
    m, params, _, cost = _setup(params_json)
    k = {**DEFAULTS, **params}["k"]
    idx, best, evals = cl.brute_force(cost)
    g_sites, g_cost, g_evals = cl.greedy(_STATE["data"], params, m)
    ls = cl.local_search(cost, m, k, LOCAL_SEARCH_RESTARTS, np.random.default_rng(0))
    ls["sites"] = _sites(ls["best_index"])
    ls["unit"] = "placements scored"
    return json.dumps({
        "brute_force": {"index": idx, "cost": best, "sites": _sites(idx), "evaluations": evals,
                        "unit": "placements scored"},
        "greedy": {"sites": g_sites, "cost": g_cost, "index": sum(1 << i for i in g_sites),
                   "evaluations": g_evals, "unit": "partial placements scored"},
        "local_search": ls,
    })


def _plan(cost, live):
    """Full sweep; live: one circuit, the loop's last search. Its threshold marks only the optimum (M = 1) and j
    maximises the chance of measuring it, so the run ends on the exact answer in one single-circuit job (as IonQ requires)."""
    if not live:
        return search.sweep_plan(cost)
    s = search.sweep_plan(cost, M_targets=(1,), max_circuits=1)[0]
    theta = math.asin(math.sqrt(s["M"] / s["N"]))
    x = math.pi / (4 * theta) - 0.5
    # the better of the two integers around x; on a tie the smaller, which is the shorter circuit
    j = max({max(0, math.floor(x)), max(0, math.ceil(x))}, key=lambda j: (round(search.success_theory(s["M"], s["N"], j), 9), -j))
    return [{**s, "js": [j]}]


def sweep(params_json, live=False):
    _, _, _, cost = _setup(params_json)
    return json.dumps(_plan(cost, live))


def sweep_circuits_py(params_json, live=False):
    """The plan's thresholds come from the classical side (they fix M for the theory curve); the circuits from the model."""
    _, _, mdl, cost = _setup(params_json)
    return search.to_basis(search.sweep_circuits(mdl, _plan(cost, live)))


def live_result(params_json, counts_json):
    """The live circuit's answer: the placement measured most often, its cost, whether it is the exact optimum,
    its share of the shots and the theory's chance sin^2((2j+1) theta) of measuring a marked placement."""
    _, _, _, cost = _setup(params_json)
    s = _plan(cost, True)[0]
    counts = json.loads(counts_json)[0]
    key = max(counts, key=counts.get)
    idx = search.index_from_key(key)
    j = s["js"][0]
    return json.dumps({"sites": _sites(idx), "cost": _clean(float(cost[idx])), "exact": bool(cost[idx] <= s["T"]),
                       "share": counts[key] / sum(counts.values()), "j": j,
                       "theory": search.success_theory(s["M"], s["N"], j)})


def _is_ionq(backend):
    return bool(re.search("ionq|aria|forte", str(getattr(backend, "name", "")), re.I))


def run_sweep(backend, params_json, shots, live=False):
    """Run the sweep circuits as one job on backend (a single circuit bare, as IonQ requires); counts per circuit in plan order.
    IonQ gets high-level gates, the other backends the rz, sx, x, cx unrolling."""
    if _is_ionq(backend):
        _, _, mdl, cost = _setup(params_json)
        circuits = search.to_ionq(search.sweep_circuits(mdl, _plan(cost, live)))
    else:
        circuits = sweep_circuits_py(params_json, live)
    result = backend.run(circuits[0] if len(circuits) == 1 else circuits, shots=int(shots)).result()
    return json.dumps([result.get_counts(i) for i in range(len(circuits))])


def sweep_analyze_json(params_json, plan_json, counts_json):
    _, _, _, cost = _setup(params_json)
    return json.dumps(search.sweep_analyze(cost, json.loads(plan_json), json.loads(counts_json)))


def dh_start(params_json, seed):
    _, _, mdl, _ = _setup(params_json)
    rng = np.random.default_rng(seed)
    _STATE["dh"] = {"model": mdl, "rng": rng, "state": search.dh_init(mdl, rng)}
    s = _STATE["dh"]["state"]
    return json.dumps({"N": s["N"], "T": s["T"], "best_index": s["best_index"],
                       "best_sites": _sites(s["best_index"])})


def dh_next_circuit():
    dh = _STATE["dh"]
    return search.to_basis([search.dh_next(dh["model"], dh["state"], dh["rng"])])[0]


def dh_run_round(backend, shots):
    """Run the next loop circuit on backend; its counts as one dict, ready for dh_feed."""
    result = backend.run([dh_next_circuit()], shots=int(shots)).result()
    return json.dumps(result.get_counts(0))


def dh_feed(counts_json):
    dh = _STATE["dh"]
    record = search.dh_feed(dh["model"], dh["state"], json.loads(counts_json))
    record = {key: _clean(v) for key, v in record.items()}
    done = search.dh_done(dh["state"])
    out = {"round": record, "done": done, "result": None}
    if done:
        res = search.dh_result(dh["model"], dh["state"])
        res["rounds"] = [{key: _clean(v) for key, v in r.items()} for r in res["rounds"]]
        res["best_sites"] = _sites(res["best_index"])
        out["result"] = res
    return json.dumps(out)


def placement_sites(index):
    return json.dumps(_sites(int(index)))


def load_recorded(text=None):
    if text is not None:
        json.loads(text)
        return text
    path = Path(RECORDED)
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / RECORDED
    with open(path, encoding="utf-8") as f:
        return f.read()


def export_take(backend_name, params_json, plan_json, sweep_counts_json, dh_json, date):
    """Hand a JSON file to the Qollab console so a real run can be saved as data/recorded_<backend>.json."""
    take = {
        "source": "qollab-run", "backend": backend_name, "date": date, "params": json.loads(params_json),
        "plan": json.loads(plan_json), "sweep_counts": json.loads(sweep_counts_json), "dh": json.loads(dh_json),
    }
    text = json.dumps(take)
    try:
        from js import sendFile  # only exists inside the Qollab runtime
        sendFile("recorded_take.json", "application/json", text)
    except ImportError:
        pass
    return text
