"""Store-placement cost model in quadratic form. Site i of a placement is bit i of its index.

cost(S) = sum_{i in S} a_i + sum_{i<j in S} b_ij over the k stores opened, S a subset of the m candidate locations.
Every coefficient is rounded to the nearest 1/SCALE, so A_i = round(a_i * SCALE) and B_ij = round(b_ij * SCALE)
are integers, every cost is a multiple of 1/8, and the classical baselines and the quantum oracle add the
same integers.

- travel: second-order inclusion-exclusion of the minutes saved against the farthest candidate,
  D_b = max_i t(b,i), g_i(b) = (D_b - t(b,i)) pop_b / sum pop, G_i = sum_b g_i(b),
  H_ij = sum_b min(g_i(b), g_j(b)); travel ~ const - sum G_i + sum H_ij (exact for one or two open sites).
- staff, per site: ceil(pop of the blocks whose nearest of the m candidates is i / c) * staff_cost.
- crowding, per pair: lam if the two sites are closer than tau minutes.
- warehouse, per site: wh_weight * minutes to its nearest warehouse.
a_i = -G_i + staff_i + warehouse_i, b_ij = H_ij + crowd_ij. const (the travel constant) is reported, not searched over.
"""
from itertools import combinations

import numpy as np

DEFAULTS = {"k": 4, "c": 15000, "staff_cost": 3.0, "tau": 10.0, "lam": 10.0, "wh_weight": 0.1}
SCALE = 8


def model(data, params, m):
    """Per-site and per-pair coefficients for the first m candidate sites: {m, k, a, b, A, B, scale, const}.

    B holds only the pairs with a nonzero integer coefficient, keyed (i, j) with i < j.
    """
    p = {**DEFAULTS, **params}
    pop = np.array([b["pop"] for b in data["blocks"]], dtype=float)
    tb = np.asarray(data["t_block_site"], dtype=float)[:, :m]
    ts = np.asarray(data["t_site_site"], dtype=float)
    tw = np.asarray(data["t_wh_site"], dtype=float)

    D = tb.max(axis=1)
    g = (D[:, None] - tb) * pop[:, None] / pop.sum()       # (blocks, m)
    G = g.sum(axis=0)
    nearest = tb.argmin(axis=1)                            # first minimum = lowest site index
    staff = np.array([np.ceil(pop[nearest == i].sum() / p["c"]) for i in range(m)]) * p["staff_cost"]
    warehouse = p["wh_weight"] * tw[:, :m].min(axis=0)
    a = -G + staff + warehouse
    b = {}
    for i, j in combinations(range(m), 2):
        b[(i, j)] = float(np.minimum(g[:, i], g[:, j]).sum()) + (p["lam"] if ts[i][j] < p["tau"] else 0.0)
    A = [int(round(v * SCALE)) for v in a]
    B = {key: int(round(v * SCALE)) for key, v in b.items() if round(v * SCALE) != 0}
    return {"m": m, "k": int(p["k"]), "a": a.tolist(), "b": b, "A": A, "B": B, "scale": SCALE,
            "const": float(pop @ D / pop.sum())}


def model_cost_int(mdl, sites):
    """The integer cost (in 1/SCALE units) of one placement: sum of A over its sites plus B over its pairs."""
    sites = sorted(sites)
    return sum(mdl["A"][i] for i in sites) + sum(mdl["B"].get(pair, 0) for pair in combinations(sites, 2))


def model_cost(mdl, sites):
    return model_cost_int(mdl, sites) / mdl["scale"]


def model_table(mdl):
    """Cost of every feasible placement, +inf elsewhere. For the classical baselines and the checks only;
    the quantum circuits are built from the coefficients."""
    m, k = mdl["m"], mdl["k"]
    table = np.full(2 ** m, np.inf)
    for sites in combinations(range(m), k):
        table[sum(1 << i for i in sites)] = model_cost(mdl, sites)
    return table


def feasible_mask(m, k):
    idx = np.arange(2 ** m)
    popcount = np.array([bin(i).count("1") for i in idx])
    return popcount == k


def cost_table(data, params, m):
    return model_table(model(data, params, m))
