from math import comb

import numpy as np

from qopt.cost import DEFAULTS, cost_table, cost_terms, feasible_mask
from tests.conftest import TOY_PARAMS


def test_terms_by_hand(toy):
    # S = {0, 2}. Nearest open site per block: b0->0 (2), b1->2 (4), b2->2 (1), b3->0 (3)
    t = cost_terms(toy, TOY_PARAMS, [0, 2])
    assert t["travel"] == (100 * 2 + 200 * 4 + 300 * 1 + 400 * 3) / 1000  # 2.5
    # site 0 serves 100+400=500, site 2 serves 200+300=500; ceil(500/300)=2 each, 1.5 per employee
    assert t["staff"] == (2 + 2) * 1.5
    # pair (0,2) is 4 min apart < tau=6
    assert t["crowd"] == 10.0
    # warehouse 0: 2*(3+2)=10, warehouse 1: 2*(1+10)=22
    assert t["warehouse"] == 10.0
    assert t["total"] == 2.5 + 6.0 + 10.0 + 10.0


def test_nearest_site_ties_go_to_lowest_index(toy):
    toy["t_block_site"] = [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]]
    # all 1000 people go to site 0: ceil(1000/300)=4, site 2 gets 0 people -> 0 employees
    assert cost_terms(toy, TOY_PARAMS, [0, 2])["staff"] == 4 * 1.5


def test_defaults_used_when_params_missing(toy):
    t = cost_terms(toy, {}, [0, 1, 2])
    assert t["crowd"] == 10.0 * 3  # pairs (0,1)=5, (0,2)=4 and (1,2)=8 are all < 10
    # b0->0, b1->1, b2->2, b3->0: every site serves some people, all < 15000 -> 1 employee each
    assert t["staff"] == 3.0 * 3


def test_defaults_values():
    assert DEFAULTS == {"k": 4, "c": 15000, "staff_cost": 3.0, "tau": 10.0, "lam": 10.0, "wh_weight": 0.1}


def test_feasible_mask():
    mask = feasible_mask(5, 2)
    assert mask.shape == (32,)
    assert mask.sum() == comb(5, 2)
    assert mask[0b00011] and not mask[0b00111]


def test_table_infeasible_is_inf_and_count(toy):
    table = cost_table(toy, TOY_PARAMS, 3)
    assert table.shape == (8,) and table.dtype == np.float64
    assert np.isfinite(table).sum() == comb(3, 2)
    for idx in (0b000, 0b001, 0b111):
        assert table[idx] == np.inf


def test_table_matches_loop_over_cost_terms(toy):
    table = cost_table(toy, TOY_PARAMS, 3)
    for idx in range(8):
        sites = [i for i in range(3) if idx >> i & 1]
        if len(sites) == 2:
            assert np.isclose(table[idx], cost_terms(toy, TOY_PARAMS, sites)["total"])
    assert table[0b101] == 28.5


def test_table_uses_first_m_sites():
    from tests.conftest import random_data
    data = random_data(30, 8, seed=1)
    table = cost_table(data, {"k": 3}, 6)
    assert len(table) == 64 and np.isfinite(table).sum() == comb(6, 3)
    assert np.isclose(table[0b100101], cost_terms(data, {"k": 3}, [0, 2, 5])["total"])
