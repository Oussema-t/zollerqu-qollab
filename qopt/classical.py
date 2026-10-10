"""Classical baselines on the same cost model."""
import numpy as np

from qopt.cost import DEFAULTS, cost_terms


def brute_force(cost):
    """Exhaustive scan of the feasible placements; evaluations = how many there are."""
    idx = int(np.argmin(cost))
    return idx, float(cost[idx]), int(np.isfinite(cost).sum())


def greedy(data, params, m):
    """Add the site that lowers the cost most, k times. Evaluations = cost_terms calls (partial placements)."""
    k = {**DEFAULTS, **params}["k"]
    sites, evaluations = [], 0
    for _ in range(k):
        scores = {i: cost_terms(data, params, sites + [i])["total"] for i in range(m) if i not in sites}
        evaluations += len(scores)
        best = min(scores, key=scores.get)
        sites.append(best)
        total = scores[best]
    sites.sort()
    return sites, total, evaluations


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

