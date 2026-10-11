"""Same-budget comparison: the budgeted solvers never exceed their budget and find small optima."""
import unittest

import numpy as np

from stocks import fair_budget, ising

MU = np.array([0.01, -0.02, 0.005, 0.0])
COV = np.array([[0.04, 0.01, 0.0, 0.005], [0.01, 0.09, -0.02, 0.0], [0.0, -0.02, 0.01, 0.002], [0.005, 0.0, 0.002, 0.03]])


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.model = ising.build_ising(MU, COV, "long_short", 1.0)
        self.e_min = float(self.model.energy(ising.all_states(4)).min())

    def test_greedy_respects_budget_and_finds_small_optimum(self):
        for budget in (4, 37, 200):
            best, used, restarts = fair_budget.greedy_within_budget(self.model, budget, np.random.default_rng(0))
            self.assertLessEqual(used, budget)
            self.assertGreaterEqual(restarts, 1)
            if budget >= 200:
                self.assertAlmostEqual(best, self.e_min, places=12)
        with self.assertRaises(ValueError):
            fair_budget.greedy_within_budget(self.model, 3, np.random.default_rng(0))  # less than one step

    def test_annealing_uses_exactly_the_budget(self):
        best, used = fair_budget.annealing_within_budget(self.model, 150, np.random.default_rng(1))
        self.assertEqual(used, 150)
        self.assertGreaterEqual(best, self.e_min - 1e-12)
        self.assertAlmostEqual(best, self.e_min, places=12)  # 150 proposals on 16 states

    def test_random_and_quantum_budgets(self):
        best, used = fair_budget.random_within_budget(self.model, 100, np.random.default_rng(2))
        self.assertEqual(used, 100)
        probabilities = np.full(16, 1 / 16)
        energies = self.model.energy(ising.all_states(4))
        best, used = fair_budget.quantum_within_budget(probabilities, energies, 100, np.random.default_rng(3))
        self.assertEqual(used, 100)
        self.assertAlmostEqual(best, self.e_min, places=12)

    def test_summary(self):
        s = fair_budget.summarize([0.0, 0.0, 1.0, 2.0], 0.0, 2.0, 1e-12)
        self.assertEqual(s["hit_rate"], 0.5)
        self.assertAlmostEqual(s["mean_best_quality"], (1 + 1 + 0.5 + 0) / 4)
        self.assertEqual(s["repetitions"], 4)


if __name__ == "__main__":
    unittest.main()
