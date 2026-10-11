"""JSON-string entry points for the browser page (Pyodide). State lives in _STATE.

params_json: {"m": 10, "k": 4, "c": ..., "staff_cost": ..., "tau": ..., "lam": ..., "wh_weight": ...};
every key is optional (m defaults to 10, the rest to qopt.cost.DEFAULTS:
k=4, c=15000, staff_cost=3.0, tau=10.0, lam=10.0, wh_weight=0.1).

load_data(json_text=None) parses json_text if given, else reads data/champaign.json, keeps the data in _STATE and returns the data text;
load_recorded(text=None) returns text unchanged (after JSON validation) if given, else the text of data/recorded_local.json; export_take(...) returns (and, inside Qollab,
downloads) a JSON take {source, backend, date, params, plan, sweep_counts, dh}.
"""
import json
import math
from pathlib import Path

import numpy as np

from qopt import classical as cl
from qopt import search
from qopt.cost import DEFAULTS, cost_table

_STATE = {}
DEFAULT_M = 10
DATA = "data/champaign.json"
RECORDED = "data/recorded_local.json"
LOCAL_SEARCH_RESTARTS = 20


def _sites(index):
    return [i for i in range(index.bit_length()) if index >> i & 1]


def _clean(x):
    """JSON-safe: +inf becomes null (JSON has no Infinity)."""
    return None if isinstance(x, float) and not math.isfinite(x) else x


def _setup(params_json):
    params = json.loads(params_json)
    m = params["m"] if "m" in params else DEFAULT_M
    cost_params = {key: v for key, v in params.items() if key != "m"}
    return m, cost_params, cost_table(_STATE["data"], cost_params, m)


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
    m, params, cost = _setup(params_json)
    idx, best, _ = cl.brute_force(cost)
    n = search.feasible_count(cost)
    return json.dumps({"m": m, "k": {**DEFAULTS, **params}["k"], "N": n, "n_feasible": n,
                       "best_index": idx, "best_cost": best, "best_sites": _sites(idx)})


def classical(params_json):
    m, params, cost = _setup(params_json)
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


def sweep(params_json):
    _, _, cost = _setup(params_json)
    return json.dumps(search.sweep_plan(cost))


def sweep_circuits_py(params_json):
    _, _, cost = _setup(params_json)
    return search.sweep_circuits(cost, search.sweep_plan(cost))


def sweep_analyze_json(params_json, plan_json, counts_json):
    _, _, cost = _setup(params_json)
    return json.dumps(search.sweep_analyze(cost, json.loads(plan_json), json.loads(counts_json)))


def dh_start(params_json, seed):
    _, _, cost = _setup(params_json)
    rng = np.random.default_rng(seed)
    _STATE["dh"] = {"cost": cost, "rng": rng, "state": search.dh_init(cost, rng)}
    s = _STATE["dh"]["state"]
    return json.dumps({"N": s["N"], "T": s["T"], "best_index": s["best_index"],
                       "best_sites": _sites(s["best_index"])})


def dh_next_circuit():
    dh = _STATE["dh"]
    return search.dh_next(dh["cost"], dh["state"], dh["rng"])


def dh_feed(counts_json):
    dh = _STATE["dh"]
    record = search.dh_feed(dh["cost"], dh["state"], json.loads(counts_json))
    record = {key: _clean(v) for key, v in record.items()}
    done = search.dh_done(dh["state"])
    out = {"round": record, "done": done, "result": None}
    if done:
        res = search.dh_result(dh["cost"], dh["state"])
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
