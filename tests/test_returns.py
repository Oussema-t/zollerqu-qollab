"""Task 2 tests: weekly log returns and the chronological 70/30 split."""
import math
import unittest

import numpy as np

from stocks import kaggle_data, returns


class LogReturnTests(unittest.TestCase):
    def test_known_series(self):
        prices = np.array([1.0, math.e, math.e ** 3, math.e ** 2])
        np.testing.assert_allclose(returns.log_returns(prices), [1.0, 2.0, -1.0], rtol=0, atol=1e-12)

    def test_two_dimensional(self):
        prices = np.array([[100.0, 10.0], [110.0, 5.0]])
        np.testing.assert_allclose(returns.log_returns(prices), [[math.log(1.1), math.log(0.5)]])

    def test_rejects_bad_prices(self):
        for bad in ([1.0, 0.0], [1.0, -2.0], [1.0, float("nan")], [1.0, float("inf")], [5.0]):
            with self.assertRaises(ValueError):
                returns.log_returns(np.array(bad))


class SplitTests(unittest.TestCase):
    def test_counts_and_blocks(self):
        split = returns.chronological_split(598, 0.70)
        self.assertEqual(split.split_index, 418)  # floor(0.7 * 598) = floor(418.6)
        rows = np.arange(598)
        train, test = rows[split.train], rows[split.test]
        self.assertEqual((len(train), len(test)), (418, 180))
        self.assertEqual(len(set(train) & set(test)), 0)               # disjoint
        np.testing.assert_array_equal(np.r_[train, test], rows)        # complete, in order, no shuffle
        self.assertEqual(train[-1] + 1, test[0])                       # no gap

    def test_float_rounding(self):
        self.assertEqual(returns.chronological_split(10, 0.7).split_index, 7)  # 0.7 * 10 = 7.000000000000001

    def test_invalid(self):
        for n, fraction in ((598, 0.0), (598, 1.0), (598, 1.2), (1, 0.7), (2, 0.4)):
            with self.assertRaises(ValueError):
                returns.chronological_split(n, fraction)

    def test_window_rows(self):
        dates = np.array(["2008-09-12", "2008-09-19", "2009-03-06", "2009-03-13"], dtype="datetime64[D]")
        np.testing.assert_array_equal(returns.window_rows(dates), [1, 2])


class RealDataSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weekly = kaggle_data.load_csv()
        cls.weekly = weekly
        cls.r = returns.log_returns(weekly.prices)
        cls.dates = weekly.dates[1:]
        cls.split = returns.chronological_split(len(cls.r))

    def test_counts_and_dates(self):
        self.assertEqual(self.r.shape, (598, 30))
        self.assertEqual(self.split.split_index, 418)
        self.assertEqual(str(self.dates[self.split.train][0]), "2006-06-02")
        self.assertEqual(str(self.dates[self.split.train][-1]), "2014-05-30")
        self.assertEqual(str(self.dates[self.split.test][0]), "2014-06-06")
        self.assertEqual(str(self.dates[self.split.test][-1]), "2017-11-10")
        self.assertTrue(np.all(np.diff(self.dates).astype(int) > 0))

    def test_crash_in_training(self):
        self.assertIn(returns.CRASH_START, self.weekly.dates)
        self.assertIn(returns.CRASH_END, self.weekly.dates)
        crash = returns.window_rows(self.dates)
        self.assertEqual(len(crash), 25)
        self.assertTrue(np.all(crash < self.split.split_index))


if __name__ == "__main__":
    unittest.main()
