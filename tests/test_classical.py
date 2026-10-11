import numpy as np

from qopt import classical
from qopt.classical import brute_force, greedy, local_search
from qopt.cost import cost_table, model, model_cost
from tests.conftest import TOY_PARAMS, random_data


def test_brute_force_is_table_min(toy):
    table = cost_table(toy, TOY_PARAMS, 3)
    idx, best, evals = brute_force(table)
    assert best == min(table) and table[idx] == best and evals == 3  # C(3,2) feasible entries, not 2^3
    assert isinstance(idx, int) and isinstance(best, float)


def test_greedy_returns_k_sites_with_their_cost():
    data = random_data(40, 8, seed=2)
    sites, cost, evals = greedy(data, {"k": 3}, 8)
    assert len(sites) == 3 and len(set(sites)) == 3 and all(0 <= s < 8 for s in sites)
    assert cost == model_cost(model(data, {"k": 3}, 8), sites)  # the same quadratic cost the oracle adds
    assert evals == 8 + 7 + 6


def test_greedy_evaluations_count_real_cost_calls(monkeypatch):
    calls = []

    def counting(*args):
        calls.append(args)
        return model_cost(*args)

    monkeypatch.setattr(classical, "model_cost", counting)
    _, _, evals = greedy(random_data(40, 8, seed=2), {"k": 3}, 8)
    assert evals == len(calls)


def test_local_search_on_toy_is_exact(toy):
    table = cost_table(toy, TOY_PARAMS, 3)
    res = local_search(table, 3, 2, restarts=5, rng=np.random.default_rng(0))
    assert res["best_cost"] == brute_force(table)[1]
    assert res["local_minima"] >= 1 and res["evaluations"] > 0


def test_local_search_never_beats_brute_force():
    data = random_data(40, 8, seed=3)
    table = cost_table(data, {"k": 3}, 8)
    res = local_search(table, 8, 3, restarts=10, rng=np.random.default_rng(1))
    assert res["best_cost"] >= brute_force(table)[1]
    assert table[res["best_index"]] == res["best_cost"]
    assert bin(res["best_index"]).count("1") == 3
    assert res["local_minima"] >= 1
