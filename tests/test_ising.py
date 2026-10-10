"""Task 3 tests: Ising mapping of the two portfolio modes, thinning by |rho|, balance penalty."""
import math
import unittest

import numpy as np

from stocks import classical_solvers as solvers
from stocks import ising, objective, task3_report

MU = np.array([0.01, -0.02, 0.005])
COV = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, -0.02], [0.0, -0.02, 0.01]])
V_REF = 0.14 / 3
SIGMA_REF = math.sqrt(V_REF)


def random_states(n, count, seed):
    return (1 - 2 * np.random.default_rng(seed).integers(0, 2, size=(count, n))).astype(np.int8)


def setUpModule():
    global CTX, SETUP12, SETUP29
    CTX = task3_report.load_context()
    SETUP12 = task3_report.size_setup(CTX, 12)
    SETUP29 = task3_report.size_setup(CTX, 29)


class FormulaTests(unittest.TestCase):
    def test_mode_a_terms(self):
        m = ising.build_ising(MU, COV, "long_short", 0.5)
        self.assertAlmostEqual(m.couplings[0, 1], 2 * 0.01 / (9 * V_REF), places=15)
        self.assertAlmostEqual(m.couplings[1, 2], 2 * -0.02 / (9 * V_REF), places=15)
        self.assertEqual(m.couplings[0, 0], 0.0)
        np.testing.assert_allclose(m.fields, -0.5 * MU / (3 * SIGMA_REF), rtol=1e-14)
        self.assertAlmostEqual(m.const, 1 / 3, places=15)  # trace(C) / (n^2 V_ref) = 1/n

    def test_mode_b_terms(self):
        m = ising.build_ising(MU, COV, "buy_or_skip", 0.5)
        self.assertAlmostEqual(m.couplings[0, 1], 0.01 / (2 * 9 * V_REF), places=15)
        row_sums = COV.sum(axis=1)  # includes the diagonal
        np.testing.assert_allclose(m.fields, row_sums / (2 * 9 * V_REF) - 0.5 * MU / (2 * 3 * SIGMA_REF), rtol=1e-14)
        expected_const = (COV.sum() + 0.14) / (4 * 9 * V_REF) - 0.5 * MU.sum() / (2 * 3 * SIGMA_REF)
        self.assertAlmostEqual(m.const, expected_const, places=15)

    def test_mode_a_equals_task2_score_small_and_real(self):
        for mu, cov, n in ((MU, COV, 3), (SETUP12["stats"].mu, SETUP12["stats"].cov, 12)):
            for eta in ising.ETA_GRID:
                model = ising.build_ising(mu, cov, "long_short", eta)
                for s in random_states(n, 100, seed=1):
                    self.assertAlmostEqual(model.energy(s), objective.risk_return_score(s, mu, cov, eta)["score"], places=13)

    def test_mode_b_equals_direct_x_objective(self):
        for mu, cov, n in ((MU, COV, 3), (SETUP12["stats"].mu, SETUP12["stats"].cov, 12)):
            v_ref = np.mean(np.diag(cov))
            for eta in ising.ETA_GRID:
                model = ising.build_ising(mu, cov, "buy_or_skip", eta)
                for s in random_states(n, 100, seed=2):
                    x = (1 + s) / 2
                    direct = x @ cov @ x / (n ** 2 * v_ref) - eta * (mu @ x) / (n * math.sqrt(v_ref))
                    self.assertAlmostEqual(model.energy(s), direct, places=13)

    def test_lower_is_better_same_sign(self):
        # a higher mean return on a long asset must lower both the dense score and the Ising energy
        s = np.array([1, -1, 1])
        better = MU + np.array([0.01, 0.0, 0.0])
        for mode in ising.MODES:
            self.assertLess(ising.dense_scores(s, better, COV, mode, 1.0), ising.dense_scores(s, MU, COV, mode, 1.0))
            self.assertLess(ising.build_ising(better, COV, mode, 1.0).energy(s), ising.build_ising(MU, COV, mode, 1.0).energy(s))

    def test_batch_energy_and_conversions(self):
        model = ising.build_ising(MU, COV, "long_short", 1.0)
        states = random_states(3, 5, seed=3)
        np.testing.assert_allclose(model.energy(states), [model.energy(s) for s in states])
        np.testing.assert_array_equal(ising.spins_to_x(np.array([1, -1, 1])), [1, 0, 1])
        np.testing.assert_array_equal(ising.x_to_spins([1, 0, 1]), [1, -1, 1])
        self.assertEqual(ising.spins_to_bitstring([-1, 1, 1]), "001")  # qubit 0 is the rightmost character
        np.testing.assert_array_equal(ising.bitstring_to_spins("001"), [-1, 1, 1])
        self.assertEqual(ising.describe([1, -1, 1], ["A", "B", "C"], "buy_or_skip"), {"buy": ["A", "C"], "skip": ["B"]})
        self.assertEqual(ising.describe([1, -1, 1], ["A", "B", "C"], "long_short"), {"long": ["A", "C"], "short": ["B"]})

    def test_invalid_inputs(self):
        bad_cov = COV.copy()
        bad_cov[0, 1] = 0.5
        nan_mu = MU.copy()
        nan_mu[0] = np.nan
        cases = [
            dict(mu=MU, cov=COV, mode="hedge", eta=1.0), dict(mu=MU, cov=COV, mode="long_short", eta=-1.0),
            dict(mu=MU, cov=COV, mode="long_short", eta=float("inf")), dict(mu=MU[:2], cov=COV, mode="long_short", eta=1.0),
            dict(mu=nan_mu, cov=COV, mode="long_short", eta=1.0), dict(mu=MU, cov=bad_cov, mode="long_short", eta=1.0),
            dict(mu=MU, cov=np.zeros((3, 3)), mode="long_short", eta=1.0),
            dict(mu=MU, cov=COV, mode="long_short", eta=1.0, keep_mask=np.ones((3, 3))),          # not boolean
            dict(mu=MU, cov=COV, mode="long_short", eta=1.0, keep_mask=np.triu(np.ones((3, 3), bool), 1)),  # not symmetric
            dict(mu=MU, cov=COV, mode="long_short", eta=1.0, target_buys=1),                      # needs balance
            dict(mu=MU, cov=COV, mode="long_short", eta=1.0, balance=True, target_buys=1),        # mode B only
            dict(mu=MU, cov=COV, mode="buy_or_skip", eta=1.0, balance=True, target_buys=4),
            dict(mu=MU, cov=COV, mode="buy_or_skip", eta=1.0, balance=True, balance_method="hard_constraint", penalty_lambda=1.0),
        ]
        for kwargs in cases:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                ising.build_ising(**kwargs)
        model = ising.build_ising(MU, COV, "long_short", 1.0)
        for bad in ([1, 0, 1], [1, -1], [1, 2, -1]):
            with self.assertRaises(ValueError):
                model.energy(bad)


class ThinningTests(unittest.TestCase):
    def test_default_fractions_on_real_data(self):
        self.assertEqual(SETUP12["mask_report"]["kept_edges"], 66)
        self.assertEqual(SETUP12["mask_report"]["dropped_edges"], 0)
        self.assertEqual(SETUP29["mask_report"]["kept_edges"], math.ceil(0.60 * 406))  # 244
        self.assertEqual(int(SETUP29["mask"].sum()) // 2, 244)
        self.assertEqual(ising.keep_fraction_for(12), 1.0)
        self.assertEqual(ising.keep_fraction_for(29), 0.60)

    def test_ranks_by_absolute_correlation_with_pair_index_ties(self):
        rho = np.array([[1.0, 0.2, -0.9, 0.5],
                        [0.2, 1.0, 0.5, -0.5],
                        [-0.9, 0.5, 1.0, 0.1],
                        [0.5, -0.5, 0.1, 1.0]])
        mask, report = ising.correlation_mask(rho, 0.5)  # 6 pairs -> keep 3
        # |-0.9| first (a strong NEGATIVE link is kept), then the tie at 0.5: pairs (0,3) and (1,2) come before (1,3)
        self.assertEqual(sorted(zip(*np.nonzero(np.triu(mask)))), [(0, 2), (0, 3), (1, 2)])
        self.assertTrue(report["tie_at_threshold"])
        self.assertEqual(report["kept_negative_edges"], 1)
        self.assertEqual([(i, j) for i, j, _ in report["dropped_pairs"]], [(1, 3), (0, 1), (2, 3)])
        self.assertTrue(np.array_equal(mask, mask.T) and not mask.diagonal().any())

    def test_sparse_model_equals_sparse_surrogate_not_dense_score(self):
        stats, mask = SETUP29["stats"], SETUP29["mask"]
        differs = 0
        for mode in ising.MODES:
            model = ising.build_ising(stats.mu, stats.cov, mode, 1.0, keep_mask=mask)
            for s in random_states(29, 30, seed=4):
                surrogate = ising.sparse_surrogate_energy(s, stats.mu, stats.cov, mode, 1.0, mask)
                self.assertAlmostEqual(model.energy(s), surrogate, places=13)
                differs += abs(model.energy(s) - ising.dense_scores(s, stats.mu, stats.cov, mode, 1.0)) > 1e-6
        self.assertGreater(differs, 0)  # the surrogate is NOT the dense score

    def test_fields_and_constant_stay_dense(self):
        stats, mask = SETUP29["stats"], SETUP29["mask"]
        for mode in ising.MODES:
            dense = ising.build_ising(stats.mu, stats.cov, mode, 1.0)
            sparse = ising.build_ising(stats.mu, stats.cov, mode, 1.0, keep_mask=mask)
            np.testing.assert_array_equal(sparse.fields, dense.fields)
            self.assertEqual(sparse.const, dense.const)
            self.assertEqual(len(sparse.pair_terms()), 244)
            self.assertEqual(len(dense.pair_terms()), 406)
            np.testing.assert_array_equal(sparse.couplings[mask], dense.couplings[mask])
            self.assertTrue(np.all(sparse.couplings[~mask] == 0))


class BalanceTests(unittest.TestCase):
    def test_feasibility_rules(self):
        even = ising.build_ising(np.zeros(4), np.eye(4), "long_short", 1.0, balance=True, penalty_lambda=1.0)
        self.assertTrue(even.is_feasible([1, 1, -1, -1]))
        self.assertFalse(even.is_feasible([1, 1, 1, -1]))
        odd = ising.build_ising(np.zeros(3), np.eye(3), "long_short", 1.0, balance=True, balance_method="hard_constraint")
        self.assertTrue(odd.is_feasible([1, 1, -1]) and odd.is_feasible([1, -1, -1]))  # |sum| = 1 is the nearest split
        self.assertFalse(odd.is_feasible([1, 1, 1]))
        buy = ising.build_ising(np.zeros(4), np.eye(4), "buy_or_skip", 1.0, balance=True, target_buys=3, penalty_lambda=1.0)
        self.assertEqual((buy.target_buys, buy.target_sum), (3, 2))
        self.assertTrue(buy.is_feasible([1, 1, 1, -1]))
        self.assertFalse(buy.is_feasible([1, 1, -1, -1]))
        default = ising.build_ising(np.zeros(5), np.eye(5), "buy_or_skip", 1.0, balance=True, balance_method="hard_constraint")
        self.assertEqual(default.target_buys, 2)  # n // 2

    def test_penalty_coefficients(self):
        lam = 0.37
        for mode, kwargs, m in (("long_short", {}, 0), ("buy_or_skip", {"target_buys": 2}, 1)):
            base = ising.build_ising(MU, COV, mode, 1.0)
            pen = ising.build_ising(MU, COV, mode, 1.0, balance=True, penalty_lambda=lam, **kwargs)
            off = ~np.eye(3, dtype=bool)
            np.testing.assert_allclose((pen.couplings - base.couplings)[off], 2 * lam)
            np.testing.assert_allclose(pen.fields - base.fields, -2 * lam * m)
            self.assertAlmostEqual(pen.const - base.const, lam * (3 + m ** 2), places=14)
            for s in random_states(3, 8, seed=5):
                self.assertAlmostEqual(pen.energy(s) - base.energy(s), lam * (s.sum() - m) ** 2, places=13)

    def test_penalty_verified_exhaustively_on_real_data(self):
        stats, mask = SETUP12["stats"], SETUP12["mask"]
        for mode in ising.MODES:
            for eta in ising.ETA_GRID:
                model = ising.build_ising(stats.mu, stats.cov, mode, eta, keep_mask=mask, balance=True)
                exact = solvers.exact_solve(model)
                self.assertEqual(exact.n_states, 4096)
                self.assertEqual(exact.search_space, math.comb(12, 6))
                check = solvers.verify_penalty(exact)
                self.assertTrue(check["ok"], (mode, eta, check))
                self.assertLess(exact.e_min, exact.energies[~exact.feasible].min())
                # the global minimum of the penalised model is a balanced state
                self.assertTrue(model.is_feasible(ising.all_states(12)[np.argmin(exact.energies)]))

    def test_hard_constraint_adds_no_penalty_terms(self):
        stats, mask = SETUP29["stats"], SETUP29["mask"]
        for mode in ising.MODES:
            plain = ising.build_ising(stats.mu, stats.cov, mode, 1.0, keep_mask=mask)
            hard = task3_report.build_model(SETUP29, mode, 1.0, True)
            self.assertEqual(hard.balance_method, ising.BALANCE_HARD)
            self.assertEqual(hard.penalty_lambda, 0.0)
            np.testing.assert_array_equal(hard.couplings, plain.couplings)
            np.testing.assert_array_equal(hard.fields, plain.fields)
            self.assertEqual(len(hard.pair_terms()), 244)
        self.assertEqual(task3_report.build_model(SETUP12, "long_short", 1.0, True).balance_method, ising.BALANCE_PENALTY)


if __name__ == "__main__":
    unittest.main()
