"""The numba simulator must give the same distribution and metrics as the reference simulator (needs numba)."""
import unittest

import numpy as np

try:
    from stocks import fast_sim
except ImportError:  # numba is only installed on the simulation servers
    fast_sim = None

from stocks import ising, large_sim, qaoa_sim, task3_report

ANGLES = (([0.5], [-0.3]), ([0.2, 0.4, 0.7], [-0.4, -0.3, -0.1]), ([0.3, 0.6], [0.2, 0.1]))


@unittest.skipIf(fast_sim is None, "numba is not installed")
class FastSimTests(unittest.TestCase):
    def models(self):
        setup = task3_report.size_setup(task3_report.load_context(), 12)
        yield task3_report.build_model(setup, "buy_or_skip", 1.0, False)
        yield task3_report.build_model(setup, "long_short", 1.0, False)
        rng = np.random.default_rng(1)
        for n in (13, 15):   # odd sizes: one single-qubit pass after the pairs
            yield ising.build_ising(rng.normal(scale=0.01, size=n), np.cov(rng.normal(size=(60, n)), rowvar=False), "long_short", 1.0)

    def test_probabilities_match_reference_for_every_kernel_split(self):
        for model in self.models():
            reference = qaoa_sim.QaoaSimulator(model)
            energies = large_sim.energies_all_states(model)
            for low in (0, 1, 3, 6, 14):   # 0: every qubit through the pair kernel; 14: every qubit in the chunk kernel
                fast = fast_sim.FastQaoa(model, energies, low=low)
                for gammas, betas in ANGLES:
                    a, b = reference.probabilities(gammas, betas), fast.probabilities(gammas, betas)
                    self.assertLess(np.abs(a - b).max(), 2e-7, (model.n, low))

    def test_metrics_match_the_full_distribution(self):
        for model in self.models():
            reference = qaoa_sim.QaoaSimulator(model)
            energies64 = reference.energies
            fast = fast_sim.FastQaoa(model, large_sim.energies_all_states(model), low=5)
            spread = fast.e_max - fast.e_min
            for gammas, betas in ANGLES:
                p = reference.probabilities(gammas, betas)
                d = fast.evaluate(gammas, betas)
                self.assertAlmostEqual(d.norm, 1.0, places=5)
                self.assertAlmostEqual(d.expectation, float(p @ energies64), delta=2e-6 * spread)
                self.assertAlmostEqual(d.p_optimum, float(p[energies64 <= energies64.min() + 1e-6 * spread].sum()), delta=1e-7)
                for alpha in (0.1, 0.25):
                    self.assertAlmostEqual(d.cvar(alpha), qaoa_sim.cvar(p, energies64, alpha), delta=2 * spread / fast.nbins)
                for budget in (10, 1000):
                    self.assertAlmostEqual(d.expected_best(budget), qaoa_sim.expected_best_of_shots(p, energies64, budget), delta=2 * spread / fast.nbins)
                    self.assertAlmostEqual(d.hit(budget), 1.0 - (1.0 - d.p_optimum) ** budget, places=9)

    def test_thinned_circuit_is_graded_on_the_dense_energies(self):
        ctx = task3_report.load_context()
        dense = task3_report.build_model(task3_report.size_setup(ctx, 14, 1.0), "buy_or_skip", 1.0, False)
        thin = task3_report.build_model(task3_report.size_setup(ctx, 14, 0.6), "buy_or_skip", 1.0, False)
        e_dense, e_thin = large_sim.energies_all_states(dense), large_sim.energies_all_states(thin)
        lean = large_sim.LargeQaoa(thin, e_thin)
        fast = fast_sim.FastQaoa(thin, e_thin, e_dense, low=4)
        p = lean.probabilities([0.4, 0.8], [-0.5, -0.2]).astype(np.float64)
        d = fast.evaluate([0.4, 0.8], [-0.5, -0.2])
        self.assertAlmostEqual(d.expectation, float(p @ e_dense.astype(np.float64)) / p.sum(), delta=3e-6 * (fast.e_max - fast.e_min))
        self.assertAlmostEqual(d.p_optimum, float(p[int(np.argmin(e_dense))]) / p.sum(), delta=1e-7)

    def test_sample_mean_is_unbiased(self):
        model = next(self.models())
        d = fast_sim.FastQaoa(model, large_sim.energies_all_states(model)).evaluate([0.5], [-0.3])
        rng = np.random.default_rng(3)
        means = [d.sample_mean(1000, rng) for _ in range(300)]
        self.assertAlmostEqual(float(np.mean(means)), d.expectation, delta=4 * float(np.std(means)) / np.sqrt(300))


if __name__ == "__main__":
    unittest.main()
