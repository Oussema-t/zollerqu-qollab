"""Task 3 tests: exact enumeration, classical baselines and held-out evaluation."""
import itertools
import unittest

import numpy as np

from stocks import classical_evaluation as evaluation
from stocks import classical_solvers as solvers
from stocks import ising, returns, task3_report


def small_model(n=6, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(40, n))
    return ising.build_ising(rng.normal(scale=0.01, size=n), np.cov(a, rowvar=False), kwargs.pop("mode", "long_short"),
                             1.0, **kwargs)


def setUpModule():
    global CTX, SETUP12, SETUP29
    CTX = task3_report.load_context()
    SETUP12 = task3_report.size_setup(CTX, 12)
    SETUP29 = task3_report.size_setup(CTX, 29)


class ExactTests(unittest.TestCase):
    def test_matches_naive_enumeration_without_fixing_a_spin(self):
        model = small_model()
        exact = solvers.exact_solve(model)
        naive = {s: model.energy(np.array(s)) for s in itertools.product((1, -1), repeat=6)}
        best = min(naive, key=naive.get)
        self.assertEqual(exact.n_states, 64)
        self.assertAlmostEqual(exact.e_min, naive[best], places=14)
        self.assertAlmostEqual(exact.e_max, max(naive.values()), places=14)
        self.assertEqual(tuple(exact.optimal_states[0]), best)
        # fields break the spin-flip symmetry: the mirrored state is NOT an optimum
        self.assertGreater(model.energy(-np.array(best)), exact.e_min + exact.tolerance)
        self.assertEqual(exact.n_optimal, 1)

    def test_real_12_assets_enumerates_4096_states(self):
        model = task3_report.build_model(SETUP12, "long_short", 1.0, False)
        exact = solvers.exact_solve(model)
        self.assertEqual((exact.n_states, exact.search_space), (4096, 4096))
        self.assertEqual(len(np.unique(ising.all_states(12), axis=0)), 4096)

    def test_ties_and_zero_spread(self):
        symmetric = ising.build_ising(np.zeros(4), np.eye(4) + 0.3, "long_short", 0.0)  # no fields: s and -s tie
        exact = solvers.exact_solve(symmetric)
        self.assertEqual(exact.n_optimal % 2, 0)
        self.assertGreaterEqual(exact.n_optimal, 2)
        flat = ising.build_ising(np.zeros(3), np.eye(3), "long_short", 0.0)  # every state has the same energy
        exact = solvers.exact_solve(flat)
        self.assertEqual(exact.n_optimal, 8)
        self.assertEqual(solvers.normalized_quality(exact.e_min, exact.e_min, exact.e_max), 1.0)
        self.assertEqual(solvers.normalized_quality(1.0, 1.0, 3.0), 1.0)
        self.assertEqual(solvers.normalized_quality(3.0, 1.0, 3.0), 0.0)


class SolverTests(unittest.TestCase):
    def test_deterministic_for_fixed_seed(self):
        for balance in (False, True):
            model = task3_report.build_model(SETUP29, "buy_or_skip", 1.0, balance)
            for solve in (lambda rng: solvers.random_search(model, 200, rng),
                          lambda rng: solvers.greedy_restarts(model, 5, rng),
                          lambda rng: solvers.simulated_annealing(model, 5, 20, rng)):
                a, b = solve(np.random.default_rng(7)), solve(np.random.default_rng(7))
                np.testing.assert_array_equal(a.best_spins, b.best_spins)
                np.testing.assert_array_equal(a.energies, b.energies)

    def test_balanced_solvers_stay_feasible(self):
        cases = [task3_report.build_model(SETUP29, "long_short", 1.0, True),    # odd n: |sum| = 1
                 task3_report.build_model(SETUP29, "buy_or_skip", 1.0, True),   # exactly 14 buys
                 task3_report.build_model(SETUP12, "long_short", 1.0, True),    # penalty model, 6 / 6
                 ising.build_ising(SETUP12["stats"].mu, SETUP12["stats"].cov, "buy_or_skip", 1.0, balance=True,
                                   balance_method="hard_constraint", target_buys=3)]
        for model in cases:
            rng = np.random.default_rng(3)
            states = solvers.random_spins(300, model, rng)
            self.assertTrue(np.all(model.is_feasible(states)))
            if model.mode == "buy_or_skip":
                self.assertTrue(np.all((states == 1).sum(axis=1) == model.target_buys))
            self.assertTrue(model.is_feasible(solvers.greedy_descent(model, states[0])))
            self.assertTrue(model.is_feasible(solvers.greedy_restarts(model, 5, rng).best_spins))
            self.assertTrue(model.is_feasible(solvers.simulated_annealing(model, 5, 20, rng).best_spins))

    def test_greedy_reaches_a_local_minimum(self):
        model = small_model(n=8, seed=2)
        s = solvers.greedy_descent(model, np.ones(8, dtype=int))
        base = model.energy(s)
        for k in range(8):
            flipped = s.copy()
            flipped[k] *= -1
            self.assertGreaterEqual(model.energy(flipped), base - 1e-12)

    def test_balanced_greedy_reaches_a_swap_minimum(self):
        # checked with real energy differences, independently of the solver's own swap formula
        for model in (task3_report.build_model(SETUP29, "long_short", 1.0, True),
                      task3_report.build_model(SETUP12, "buy_or_skip", 1.0, True)):
            start = solvers.random_spins(1, model, np.random.default_rng(11))[0]
            s = solvers.greedy_descent(model, start)
            base = model.energy(s)
            self.assertLessEqual(base, model.energy(start) + 1e-15)
            for i in np.flatnonzero(s > 0):
                for j in np.flatnonzero(s < 0):
                    swapped = s.copy()
                    swapped[i], swapped[j] = -1, 1
                    self.assertGreaterEqual(model.energy(swapped), base - 1e-12)

    def test_solvers_find_the_exact_optimum_on_12_assets(self):
        for mode in ising.MODES:
            for balance in (False, True):
                model = task3_report.build_model(SETUP12, mode, 1.0, balance)
                exact = solvers.exact_solve(model)
                sa = solvers.simulated_annealing(model, 30, 200, np.random.default_rng(2026))
                self.assertAlmostEqual(sa.best_energy, exact.e_min, delta=exact.tolerance + 1e-15)
                self.assertGreaterEqual(sa.best_energy, exact.e_min - 1e-15)  # nothing can beat the exact optimum

    def test_random_states_are_uniform_enough(self):
        model = small_model(n=4)
        states = solvers.random_spins(4000, model, np.random.default_rng(1))
        self.assertTrue(np.all(np.abs(states.mean(axis=0)) < 0.1))
        self.assertEqual(len(np.unique(states, axis=0)), 16)


class HeldOutTests(unittest.TestCase):
    def test_manual_example(self):
        prices = np.array([[100.0, 50.0], [110.0, 45.0], [99.0, 54.0]])
        r = evaluation.simple_returns(prices)
        np.testing.assert_allclose(r, [[0.10, -0.10], [-0.10, 0.20]])
        benchmark = np.array([0.02, -0.01])
        a = evaluation.heldout_metrics([1, -1], "long_short", r, benchmark)   # weekly: (0.10 + 0.10)/2, (-0.10 - 0.20)/2
        self.assertAlmostEqual(a["mean"], (0.10 - 0.15) / 2)
        self.assertAlmostEqual(a["worst_week"], -0.15)
        self.assertAlmostEqual(a["volatility"], np.std([0.10, -0.15], ddof=1))
        self.assertAlmostEqual(a["exposure"], 0.0)
        self.assertAlmostEqual(a["corr_benchmark"], 1.0)
        b = evaluation.heldout_metrics([1, -1], "buy_or_skip", r, benchmark)  # only asset 1, half the money in cash
        self.assertAlmostEqual(b["mean"], (0.05 - 0.05) / 2)
        self.assertAlmostEqual(b["exposure"], 0.5)
        nothing = evaluation.heldout_metrics([-1, -1], "buy_or_skip", r, benchmark)
        self.assertEqual((nothing["mean"], nothing["volatility"], nothing["exposure"]), (0.0, 0.0, 0.0))
        self.assertIsNone(nothing["corr_benchmark"])  # undefined for a constant series
        with self.assertRaises(ValueError):
            evaluation.simple_returns(np.array([1.0, 0.0]))

    def test_alignment_with_task2_split(self):
        simple, log_r = CTX["simple_r"], CTX["log_r"]
        self.assertEqual(simple.shape, log_r.shape)
        np.testing.assert_allclose(np.log1p(simple), log_r, atol=1e-12)  # same rows, no shift
        split = CTX["split"]
        self.assertEqual(len(SETUP12["test_assets"]), 180)
        self.assertEqual(str(CTX["dates"][split.test][0]), "2014-06-06")
        prices = CTX["weekly"].prices
        k = split.split_index  # first test return uses prices k and k + 1
        np.testing.assert_allclose(SETUP29["test_benchmark"][0], prices[k + 1, -1] / prices[k, -1] - 1)
        self.assertEqual(SETUP12["stats"].n_obs, 418)  # statistics use training rows only

    def test_random_portfolios_follow_mode_and_balance(self):
        for mode, balance in (("long_short", True), ("buy_or_skip", True), ("buy_or_skip", False)):
            model = task3_report.build_model(SETUP29, mode, 1.0, balance)
            a = evaluation.random_portfolio_metrics(model, SETUP29["test_assets"], 200, np.random.default_rng(5))
            b = evaluation.random_portfolio_metrics(model, SETUP29["test_assets"], 200, np.random.default_rng(5))
            np.testing.assert_array_equal(a["mean"], b["mean"])
            self.assertEqual(a["volatility"].shape, (200,))
        w = evaluation.weights(np.array([[1, -1, 1, -1]]), "buy_or_skip")
        np.testing.assert_allclose(w, [[0.25, 0.0, 0.25, 0.0]])  # fixed 1/n per bought asset, rest in cash
        self.assertEqual(evaluation.percentile_rank(2.0, [1.0, 2.0, 3.0, 4.0]), 50.0)


class GridTests(unittest.TestCase):
    def test_one_configuration_end_to_end(self):
        r = task3_report.run_configuration(SETUP12, "buy_or_skip", 1.0, True)
        self.assertEqual(r["reference"], "exact optimum")
        self.assertEqual((r["n_states"], r["n_plus"], r["target_buys"]), (4096, 6, 6))
        self.assertTrue(r["feasible"] and r["penalty_check"]["ok"])
        self.assertAlmostEqual(r["model_energy"], r["dense_score"], places=12)  # full mask, zero penalty when balanced
        self.assertEqual(sorted(r["assignment"]), ["buy", "skip"])
        big = task3_report.run_configuration(SETUP29, "long_short", 1.0, True)
        self.assertIn("not proven optimal", big["reference"])
        self.assertEqual(big["balance_method"], ising.BALANCE_HARD)
        self.assertIsNone(big["n_optimal"])
        self.assertEqual(abs(sum(big["spins"])), 1)
        self.assertNotAlmostEqual(big["model_energy"], big["dense_score"], places=6)  # surrogate vs dense score


if __name__ == "__main__":
    unittest.main()
