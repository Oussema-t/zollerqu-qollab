"""Task 2 tests: training statistics, negative links, sparsification diagnostics, risk-return score."""
import math
import unittest

import numpy as np

from stocks import graph, objective


def stats_from(cov, tickers):
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.diag(cov))
    return graph.TrainingStats(list(tickers), 100, np.zeros(len(tickers)), cov, cov / np.outer(sd, sd))


class TrainingStatsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.r = rng.normal(size=(50, 4)) @ np.array([[1, .5, 0, 0], [0, 1, -.3, 0], [0, 0, 1, .2], [0, 0, 0, 1]])
        self.tickers = ["A", "B", "C", "D"]

    def test_sample_covariance_and_correlation(self):
        stats = graph.training_stats(self.r, self.tickers)
        centred = self.r - self.r.mean(axis=0)
        np.testing.assert_allclose(stats.cov, centred.T @ centred / (len(self.r) - 1), rtol=1e-12)  # ddof = 1
        np.testing.assert_allclose(stats.corr, np.corrcoef(self.r, rowvar=False), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(stats.mu, self.r.mean(axis=0))
        self.assertEqual(stats.n_obs, 50)
        checks = graph.matrix_checks(stats)
        self.assertTrue(all(v for k, v in checks.items() if k != "corr_diag_max_deviation"))
        self.assertLess(checks["corr_diag_max_deviation"], 1e-12)

    def test_constant_series_is_reported(self):
        self.r[:, 2] = 0.01
        with self.assertRaisesRegex(ValueError, r"zero variance.*'C'"):
            graph.training_stats(self.r, self.tickers)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            graph.training_stats(self.r, ["A", "B"])
        self.r[3, 1] = np.nan
        with self.assertRaises(ValueError):
            graph.training_stats(self.r, self.tickers)


class NegativeLinkTests(unittest.TestCase):
    def test_counts_and_group_pairs(self):
        cov = [[1.0, -0.5, 0.2], [-0.5, 1.0, -0.1], [0.2, -0.1, 1.0]]
        stats = stats_from(cov, ["G1", "B1", "T1"])
        groups = ["Gold", "Bonds", "Tech"]
        neg = graph.negative_links(stats, groups, ["Tech", "Gold", "Bonds"])
        self.assertEqual((neg["n_pairs"], neg["n_negative_cov"], neg["n_negative_corr"]), (3, 2, 2))
        self.assertEqual(neg["sign_disagreements"], [])
        self.assertEqual([(p["a"], p["b"]) for p in neg["pairs"]], [("G1", "B1"), ("B1", "T1")])  # most negative first
        self.assertEqual(dict(neg["by_group_pair"]), {"Gold – Bonds": 1, "Tech – Bonds": 1})
        most_negative, strongest = graph.extreme_correlations(stats, groups, k=1)
        self.assertEqual((most_negative[0]["a"], strongest[0]["b"]), ("G1", "T1"))


class SparsifyTests(unittest.TestCase):
    def test_keep_count_ties_share_isolated(self):
        cov = np.array([[4.0, 3.0, -1.0, 0.0],
                        [3.0, 4.0, 1.0, 0.5],
                        [-1.0, 1.0, 4.0, 0.0],
                        [0.0, 0.5, 0.0, 4.0]])
        original = cov.copy()
        g = graph.sparsify(cov, keep_fraction=0.30)
        # 6 pairs, keep ceil(1.8) = 2: (0,1)=3 first, then the tie |-1| = |1| goes to pair (0,2) by pair index
        self.assertEqual(g.edges, [(0, 1), (0, 2)])
        self.assertEqual(g.report["kept_edges"], 2)
        self.assertEqual(g.report["dropped_edges"], 4)
        self.assertTrue(g.report["tie_at_threshold"])
        self.assertAlmostEqual(g.report["dropped_abs_share"], (1.0 + 0.5) / (3 + 1 + 1 + 0.5))
        self.assertEqual(g.report["kept_negative_edges"], 1)
        self.assertEqual(g.report["isolated_nodes"], [3])
        self.assertTrue(np.array_equal(g.mask, g.mask.T) and not g.mask.diagonal().any())
        np.testing.assert_array_equal(cov, original)  # dense matrix untouched

    def test_ceil_is_robust_and_deterministic(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(30, 5))
        cov = np.cov(a, rowvar=False)
        g1, g2 = graph.sparsify(cov), graph.sparsify(cov)
        self.assertEqual(g1.report["kept_edges"], 3)  # 0.3 * 10 = 3.0000000000000004 -> 3
        self.assertEqual(g1.edges, g2.edges)

    def test_zero_matrix_and_invalid(self):
        g = graph.sparsify(np.zeros((3, 3)))
        self.assertEqual(g.report["dropped_abs_share"], 0.0)
        self.assertTrue(g.report["zero_total_weight"])
        with self.assertRaises(ValueError):
            graph.sparsify(np.eye(3), keep_fraction=0)
        with self.assertRaises(ValueError):
            graph.sparsify(np.ones((2, 3)))


class RiskReturnScoreTests(unittest.TestCase):
    MU = np.array([0.01, -0.02, 0.005])
    COV = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, -0.02], [0.0, -0.02, 0.01]])

    def test_hand_calculation(self):
        s = objective.risk_return_score([1, -1, 1], self.MU, self.COV, eta=0.5)
        # E_p = (0.01 + 0.02 + 0.005) / 3; s^T C s = 0.14 + 2 * (-0.01 + 0 + 0.02) = 0.16; V_p = 0.16 / 9
        expected_e, expected_v, v_ref = 0.035 / 3, 0.16 / 9, 0.14 / 3
        self.assertAlmostEqual(s["expected_return"], expected_e, places=15)
        self.assertAlmostEqual(s["variance"], expected_v, places=15)
        self.assertAlmostEqual(s["v_ref"], v_ref, places=15)
        self.assertAlmostEqual(s["risk_term"], expected_v / v_ref, places=12)
        self.assertAlmostEqual(s["return_term"], 0.5 * expected_e / math.sqrt(v_ref), places=12)
        self.assertAlmostEqual(s["score"], expected_v / v_ref - 0.5 * expected_e / math.sqrt(v_ref), places=12)

    def test_invalid_input(self):
        cases = [([1, 0, 1], self.MU, self.COV, 1.0), ([1, -1], self.MU, self.COV, 1.0),
                 ([1, -1, 1], self.MU, self.COV, float("nan")), ([1, -1, 1], self.MU, self.COV, -1.0),
                 ([1, -1, 1], self.MU, np.zeros((3, 3)), 1.0)]
        for spins, mu, cov, eta in cases:
            with self.assertRaises(ValueError):
                objective.risk_return_score(spins, mu, cov, eta)

    def test_example_spins_are_fixed(self):
        a, b = objective.example_spins(12), objective.example_spins(12)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
            self.assertTrue(np.all(np.isin(a[key], (-1, 1))))


if __name__ == "__main__":
    unittest.main()
