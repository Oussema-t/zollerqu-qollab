import numpy as np

from qopt.classical import brute_force, greedy, local_search, milp
from qopt.cost import cost_table, cost_terms
from tests.conftest import TOY_PARAMS, random_data


def test_brute_force_is_table_min(toy):
    table = cost_table(toy, TOY_PARAMS, 3)
    idx, best, evals = brute_force(table)
    assert best == min(table) and table[idx] == best and evals == 8
    assert isinstance(idx, int) and isinstance(best, float)


def test_greedy_returns_k_sites_with_their_cost():
    data = random_data(40, 8, seed=2)
    sites, cost = greedy(data, {"k": 3}, 8)
    assert len(sites) == 3 and len(set(sites)) == 3 and all(0 <= s < 8 for s in sites)
    assert np.isclose(cost, cost_terms(data, {"k": 3}, sites)["total"])


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


def test_milp_travel_part_matches_brute_force_on_travel():
    data = random_data(25, 6, seed=4)
    res = milp(data, {"k": 2}, 6)
    if res is None:
        return
    best_travel = min(cost_terms(data, {}, [i, j])["travel"] for i in range(6) for j in range(i + 1, 6))
    assert np.isclose(res["travel"], best_travel)
    assert len(res["sites"]) == 2 and res["scope"] == "travel part only"


def test_milp_returns_none_when_solver_raises(monkeypatch):
    import scipy.optimize

    def boom(*args, **kwargs):
        raise RuntimeError("HiGHS failed")

    monkeypatch.setattr(scipy.optimize, "milp", boom)
    assert milp(random_data(25, 6, seed=4), {"k": 2}, 6) is None
