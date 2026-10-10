"""Classical baselines on the same cost model."""
import numpy as np

from qopt.cost import DEFAULTS, cost_terms


def brute_force(cost):
    idx = int(np.argmin(cost))
    return idx, float(cost[idx]), len(cost)


def greedy(data, params, m):
    k = {**DEFAULTS, **params}["k"]
    sites = []
    for _ in range(k):
        best = min((i for i in range(m) if i not in sites),
                   key=lambda i: cost_terms(data, params, sites + [i])["total"])
        sites.append(best)
    sites.sort()
    return sites, cost_terms(data, params, sites)["total"]


def local_search(cost, m, k, restarts, rng):
    """Steepest descent in the swap neighbourhood from random feasible starts."""
    evaluations = 0
    minima = set()
    for _ in range(restarts):
        x = int(sum(1 << int(i) for i in rng.choice(m, k, replace=False)))
        evaluations += 1
        while True:
            opened = [i for i in range(m) if x >> i & 1]
            closed = [i for i in range(m) if not x >> i & 1]
            neighbours = [x ^ (1 << i) ^ (1 << j) for i in opened for j in closed]
            evaluations += len(neighbours)
            best = min(neighbours, key=lambda y: cost[y])
            if cost[best] < cost[x]:
                x = best
            else:
                break
        minima.add(x)
    best_index = min(minima, key=lambda y: cost[y])
    return {"best_index": best_index, "best_cost": float(cost[best_index]),
            "local_minima": len(minima), "evaluations": evaluations}


def milp(data, params, m):
    """p-median MIP on the travel term only (scipy HiGHS). None if milp is unavailable or fails."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp as scipy_milp
    except ImportError:
        return None
    try:
        k = {**DEFAULTS, **params}["k"]
        pop = np.array([b["pop"] for b in data["blocks"]], dtype=float)
        tb = np.asarray(data["t_block_site"], dtype=float)[:, :m]
        nb = len(pop)
        # variables: y_0..y_{m-1}, then x_{b,i} row-major
        c = np.concatenate([np.zeros(m), (pop[:, None] * tb).ravel() / pop.sum()])
        n = m + nb * m
        assign = np.zeros((nb, n))
        link = np.zeros((nb * m, n))
        for b in range(nb):
            assign[b, m + b * m: m + (b + 1) * m] = 1
            for i in range(m):
                link[b * m + i, m + b * m + i] = 1
                link[b * m + i, i] = -1
        count = np.concatenate([np.ones(m), np.zeros(nb * m)])[None, :]
        constraints = [LinearConstraint(assign, 1, 1), LinearConstraint(link, -np.inf, 0),
                       LinearConstraint(count, k, k)]
        integrality = np.concatenate([np.ones(m), np.zeros(nb * m)])
        res = scipy_milp(c, constraints=constraints, integrality=integrality, bounds=Bounds(0, 1))
        if not res.success:
            return None
        sites = [i for i in range(m) if res.x[i] > 0.5]
        return {"sites": sites, "index": sum(1 << i for i in sites), "travel": float(res.fun),
                "scope": "travel part only"}
    except Exception:
        return None
