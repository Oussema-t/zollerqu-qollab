"""Store-placement cost model. Site i of a placement is bit i of its index."""
from itertools import combinations

import numpy as np

DEFAULTS = {"k": 4, "c": 15000, "staff_cost": 3.0, "tau": 10.0, "lam": 10.0, "wh_weight": 0.1}


def _terms(data, params, subsets):
    """Cost terms for an (F, s) array of site-index rows, each row sorted ascending."""
    p = {**DEFAULTS, **params}
    pop = np.array([b["pop"] for b in data["blocks"]], dtype=float)
    tb = np.asarray(data["t_block_site"], dtype=float)
    ts = np.asarray(data["t_site_site"], dtype=float)
    tw = np.asarray(data["t_wh_site"], dtype=float)
    s = subsets.shape[1]

    times = tb[:, subsets]                       # (B, F, s)
    nearest = times.argmin(axis=2)               # first minimum = lowest site index
    travel = pop @ times.min(axis=2) / pop.sum()

    assigned = np.stack([pop @ (nearest == slot) for slot in range(s)], axis=1)  # (F, s)
    staff = np.ceil(assigned / p["c"]).sum(axis=1) * p["staff_cost"]

    iu, ju = np.triu_indices(s, k=1)
    close = ts[subsets[:, iu], subsets[:, ju]] < p["tau"]
    crowd = p["lam"] * close.sum(axis=1)

    warehouse = p["wh_weight"] * tw[:, subsets].sum(axis=2).min(axis=0)
    return {"travel": travel, "staff": staff, "crowd": crowd, "warehouse": warehouse,
            "total": travel + staff + crowd + warehouse}


def cost_terms(data, params, sites_list):
    subsets = np.array([sorted(sites_list)], dtype=int)
    return {name: float(v[0]) for name, v in _terms(data, params, subsets).items()}


def feasible_mask(m, k):
    idx = np.arange(2 ** m)
    popcount = np.array([bin(i).count("1") for i in idx])
    return popcount == k


def cost_table(data, params, m):
    k = {**DEFAULTS, **params}["k"]
    subsets = np.array(list(combinations(range(m), k)), dtype=int)
    table = np.full(2 ** m, np.inf)
    index = (1 << subsets).sum(axis=1)
    table[index] = _terms(data, params, subsets)["total"]
    return table
