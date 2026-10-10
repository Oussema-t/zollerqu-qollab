"""JSON-string entry points for the browser page (Pyodide). State lives in _STATE.

params_json: {"m": 10, "k": 3, "c": ..., "staff_cost": ..., "tau": ..., "lam": ..., "wh_weight": ...};
every key is optional (m defaults to 10, the rest to qopt.cost.DEFAULTS).
"""
import json
import math

import numpy as np

from qopt import classical as cl
from qopt import search
from qopt.cost import DEFAULTS, cost_table

_STATE = {}
DEFAULT_M = 10
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


def load_data(json_text):
    data = json.loads(json_text)
    _STATE["data"] = data
    return json.dumps({"blocks": len(data["blocks"]), "sites": len(data["sites"])})


def cost_table_json(params_json):
    m, params, cost = _setup(params_json)
    idx, best, _ = cl.brute_force(cost)
    return json.dumps({"m": m, "k": {**DEFAULTS, **params}["k"], "n_feasible": int(np.isfinite(cost).sum()),
                       "best_index": idx, "best_cost": best, "best_sites": _sites(idx)})


def classical(params_json):
    m, params, cost = _setup(params_json)
    k = {**DEFAULTS, **params}["k"]
    idx, best, evals = cl.brute_force(cost)
    g_sites, g_cost = cl.greedy(_STATE["data"], params, m)
    ls = cl.local_search(cost, m, k, LOCAL_SEARCH_RESTARTS, np.random.default_rng(0))
    ls["sites"] = _sites(ls["best_index"])
    return json.dumps({
        "brute_force": {"index": idx, "cost": best, "sites": _sites(idx), "evaluations": evals},
        "greedy": {"sites": g_sites, "cost": g_cost, "index": sum(1 << i for i in g_sites)},
        "local_search": ls,
        "milp": cl.milp(_STATE["data"], params, m),
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
