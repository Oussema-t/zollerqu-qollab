from itertools import combinations
from math import comb

import numpy as np
import pytest

from qopt.cost import DEFAULTS, SCALE, cost_table, feasible_mask, model, model_cost, model_cost_int
from tests.conftest import TOY_PARAMS


def test_scale_is_eight():
    assert SCALE == 8


def test_coefficients_by_hand(toy):
    # pop 100, 200, 300, 400 (total 1000). D_b = max over the 3 sites: 9, 6, 8, 9.
    # g_i(b) = (D_b - t_bi) pop_b / 1000:
    #   b0 0.7 0.4 0 | b1 0 0.6 0.4 | b2 0 0.3 2.1 | b3 2.4 0 1.2
    # G = 3.1, 1.3, 3.7; H01 = 0.4, H02 = 1.2, H12 = 0.7; constant = 8.1 minutes.
    # Catchments among the 3 sites: site 0 = b0 + b3 (500), site 1 = b1 (200), site 2 = b2 (300);
    # ceil(pop / 300) * 1.5 = 3.0, 1.5, 1.5. Warehouse: 2 * min_w t = 2, 2, 4.
    # a = -G + staff + warehouse = 1.9, 2.2, 1.8.
    # Crowding (tau 6, lam 10): pairs (0,1) at 5 and (0,2) at 4 are close, (1,2) at 8 is not.
    # b = H + crowd = 10.4, 11.2, 0.7.
    mdl = model(toy, TOY_PARAMS, 3)
    assert mdl["m"] == 3 and mdl["k"] == 2 and mdl["scale"] == 8
    assert np.allclose(mdl["a"], [1.9, 2.2, 1.8])
    assert set(mdl["b"]) == {(0, 1), (0, 2), (1, 2)}
    assert np.allclose([mdl["b"][p] for p in [(0, 1), (0, 2), (1, 2)]], [10.4, 11.2, 0.7])
    assert np.isclose(mdl["const"], 8.1)
    # nearest eighth: 15.2 -> 15, 17.6 -> 18, 14.4 -> 14; 83.2 -> 83, 89.6 -> 90, 5.6 -> 6
    assert mdl["A"] == [15, 18, 14]
    assert mdl["B"] == {(0, 1): 83, (0, 2): 90, (1, 2): 6}
    assert all(isinstance(v, int) for v in mdl["A"] + list(mdl["B"].values()))


def test_travel_part_is_exact_for_two_sites(toy):
    # S = {0, 2}: exact mean travel to the nearest open site is (100*2 + 200*4 + 300*1 + 400*3) / 1000 = 2.5,
    # and constant - G0 - G2 + H02 = 8.1 - 3.1 - 3.7 + 1.2 = 2.5
    mdl = model(toy, TOY_PARAMS, 3)
    assert np.isclose(mdl["const"] - 3.1 - 3.7 + 1.2, 2.5)


def test_table_is_integer_sum_over_scale(toy):
    mdl = model(toy, TOY_PARAMS, 3)
    table = cost_table(toy, TOY_PARAMS, 3)
    for sites in combinations(range(3), 2):
        idx = sum(1 << i for i in sites)
        n = sum(mdl["A"][i] for i in sites) + sum(mdl["B"][p] for p in combinations(sites, 2))
        assert model_cost_int(mdl, sites) == n
        assert table[idx] == n / 8 == model_cost(mdl, sites)
    assert table[0b011] == 116 / 8 and table[0b101] == 119 / 8 and table[0b110] == 38 / 8


def test_every_cost_is_a_multiple_of_one_eighth():
    from tests.conftest import random_data
    table = cost_table(random_data(30, 8, seed=1), {"k": 3}, 8)
    finite = table[np.isfinite(table)]
    assert np.array_equal(finite * 8, np.round(finite * 8))


def test_crowding_uses_params(toy):
    mdl = model(toy, {**TOY_PARAMS, "lam": 0.0}, 3)
    assert np.allclose([mdl["b"][p] for p in [(0, 1), (0, 2), (1, 2)]], [0.4, 1.2, 0.7])


def test_zero_pairs_are_left_out(toy):
    # sites 1 and 2 never both beat the farthest site for the same block, and are not close: b_12 = 0
    toy["t_block_site"] = [[2, 5, 9], [6, 9, 4], [8, 7, 9], [3, 9, 6]]
    mdl = model(toy, {**TOY_PARAMS, "lam": 0.0}, 3)
    assert (1, 2) not in mdl["B"] and all(v != 0 for v in mdl["B"].values())


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


def test_table_uses_first_m_sites():
    from tests.conftest import random_data
    data = random_data(30, 8, seed=1)
    table = cost_table(data, {"k": 3}, 6)
    assert len(table) == 64 and np.isfinite(table).sum() == comb(6, 3)
    assert table[0b100101] == model_cost(model(data, {"k": 3}, 6), [0, 2, 5])
