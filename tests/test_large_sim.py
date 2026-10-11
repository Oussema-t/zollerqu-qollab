"""Memory-lean large-n simulator: energies and probabilities must equal the reference implementation."""
import unittest

import numpy as np

from stocks import ising, large_sim, qaoa_sim, task3_report


class LargeSimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx = task3_report.load_context()
        cls.setup = task3_report.size_setup(ctx, 12)

    def test_energies_match_bit_by_bit_construction(self):
        for mode in ising.MODES:
            model = task3_report.build_model(self.setup, mode, 1.0, False)
            reference = model.energy(ising.all_states(12))
            energies = large_sim.energies_all_states(model, dtype=np.float64)
            np.testing.assert_allclose(energies, reference, rtol=0, atol=1e-14)
            self.assertEqual(energies.dtype, np.float64)
            self.assertEqual(large_sim.energies_all_states(model).dtype, np.float32)

    def test_probabilities_match_reference_simulator(self):
        for mode in ising.MODES:
            model = task3_report.build_model(self.setup, mode, 1.0, False)
            reference, lean = qaoa_sim.QaoaSimulator(model), large_sim.LargeQaoa(model)
            for gammas, betas in (([0.5], [-0.3]), ([0.2, 0.4, 0.7], [-0.4, -0.3, -0.1]), ([0.3, 0.6], [0.2, 0.1])):
                a, b = reference.probabilities(gammas, betas), lean.probabilities(gammas, betas)
                self.assertLess(np.abs(a - b).max(), 1e-7)
                self.assertAlmostEqual(float(b.sum(dtype=np.float64)), 1.0, places=5)
            self.assertEqual(lean.evaluations, 3)

    def test_spins_of_index_and_metrics(self):
        np.testing.assert_array_equal(large_sim.spins_of_index(5, 4), [-1, 1, -1, 1])   # bits 1010 -> qubits 0 and 2 are |1>
        energies = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        probabilities = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        m = large_sim.metrics(probabilities, energies, 0.0, 3.0, 1e-6, shots=10)
        self.assertAlmostEqual(m["expectation"], 2.0, places=6)
        self.assertAlmostEqual(m["p_optimum"], 0.1, places=6)
        self.assertEqual(m["most_probable_state"], 3)
        self.assertAlmostEqual(large_sim.dot64(probabilities, energies), 2.0, places=6)
        hits = large_sim.sampled_hits(probabilities, energies, 0.0, 1e-6, 50, 20, np.random.default_rng(0))
        self.assertGreater(hits, 0.9)  # 50 shots at p = 0.1 find state 0 almost always

    def test_group_boundary_sizes(self):
        """n not divisible by GROUP (12 = 4+4+4, 13 = 4+4+4+1, 14 = 4+4+4+2) must all work."""
        rng = np.random.default_rng(1)
        for n in (13, 14):
            a = rng.normal(size=(60, n))
            model = ising.build_ising(rng.normal(scale=0.01, size=n), np.cov(a, rowvar=False), "long_short", 1.0)
            ref, lean = qaoa_sim.QaoaSimulator(model), large_sim.LargeQaoa(model)
            self.assertLess(np.abs(ref.probabilities([0.4], [-0.2]) - lean.probabilities([0.4], [-0.2])).max(), 1e-7)


if __name__ == "__main__":
    unittest.main()
