"""Task 1 tests: OR-Library parser, normalised CSV and error handling.

Run from the repo root:  python -m unittest discover -s tests -t . -v
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stocks import data
from stocks.data import DataFormatError


class RealSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market, cls.report = data.parse_orlib(data.RAW_PATH)

    def test_source_parses(self):
        self.assertEqual(self.report.header, (31, 290))
        self.assertEqual(self.report.layout, "(N+1)*(T+1)")
        self.assertEqual(self.report.n_values, 32 * 291)

    def test_series_counts(self):
        self.assertEqual(self.market.prices.shape, (291, 31))
        self.assertEqual(self.market.index.shape, (291,))
        self.assertEqual(list(self.market.weeks), list(range(291)))

    def test_first_index_near_8749(self):
        self.assertAlmostEqual(self.market.index[0], 8749, delta=1.0)

    def test_prices_finite_and_positive(self):
        for series in (self.market.index, self.market.prices):
            self.assertTrue(np.all(np.isfinite(series)))
            self.assertTrue(np.all(series > 0))

    def test_source_order_preserved(self):
        tokens = data.RAW_PATH.read_text().split()[2:]
        self.assertEqual(self.market.index[0], float(tokens[0]))          # index block first
        self.assertEqual(self.market.prices[0, 0], float(tokens[291]))    # then stock 1
        self.assertEqual(self.market.prices[-1, -1], float(tokens[-1]))   # last stock, last week


class CsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market, _ = data.parse_orlib(data.RAW_PATH)
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_committed_csv_columns(self):
        header = data.CSV_PATH.read_text().splitlines()[0].split(",")
        self.assertEqual(header, ["week", "index"] + [f"s{i:02d}" for i in range(1, 32)])

    def test_committed_csv_matches_source(self):
        loaded = data.load_csv(data.CSV_PATH)
        self.assertEqual(loaded.prices.shape, (291, 31))
        np.testing.assert_array_equal(loaded.index, self.market.index)
        np.testing.assert_array_equal(loaded.prices, self.market.prices)

    def test_roundtrip_keeps_order_and_values(self):
        path = self.tmp / "roundtrip.csv"
        data.write_csv(self.market, path)
        loaded = data.load_csv(path)
        self.assertEqual(loaded.names, self.market.names)
        np.testing.assert_array_equal(loaded.weeks, self.market.weeks)
        np.testing.assert_array_equal(loaded.index, self.market.index)
        np.testing.assert_array_equal(loaded.prices, self.market.prices)

    def test_csv_with_missing_cell_fails(self):
        path = self.tmp / "missing.csv"
        lines = data.CSV_PATH.read_text().splitlines()
        cells = lines[5].split(",")
        cells[3] = ""
        lines[5] = ",".join(cells)
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(DataFormatError, "non-empty cells"):
            data.load_csv(path)

    def test_csv_with_wrong_columns_fails(self):
        path = self.tmp / "columns.csv"
        lines = data.CSV_PATH.read_text().splitlines()
        lines[0] = lines[0].replace("s01,s02", "s02,s01")
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(DataFormatError, "in order"):
            data.load_csv(path)


def write_source(path, header, series, per_line=4):
    """Write a small OR-Library-style file: each series on its own lines, per_line values per line."""
    lines = [header]
    for values in series:
        for start in range(0, len(values), per_line):
            lines.append(" ".join(str(v) for v in values[start:start + per_line]))
    Path(path).write_text("\n".join(lines) + "\n")


class InvalidInputTests(unittest.TestCase):
    """Small synthetic files: 2 stocks, T = 3. The indtrack1-specific checks are switched off."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.path = self.tmp / "src.txt"
        self.series = [[100.0, 101.0, 102.0, 103.0], [1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3]]

    def parse(self, **kwargs):
        options = dict(expected_header=None, expected_first_index=None)
        options.update(kwargs)
        return data.parse_orlib(self.path, **options)

    def test_valid_layout_t_plus_1(self):
        write_source(self.path, "2 3", self.series)
        market, report = self.parse()
        self.assertEqual(report.layout, "(N+1)*(T+1)")
        self.assertEqual(market.prices.shape, (4, 2))
        self.assertEqual(market.prices[2, 1], 2.2)

    def test_valid_layout_t(self):
        write_source(self.path, "2 4", self.series)  # 4 observations = T
        _, report = self.parse()
        self.assertEqual(report.layout, "(N+1)*T")

    def test_non_integer_header(self):
        write_source(self.path, "2 three", self.series)
        with self.assertRaisesRegex(DataFormatError, "two integers"):
            self.parse()

    def test_header_with_one_value(self):
        write_source(self.path, "2", self.series)
        with self.assertRaisesRegex(DataFormatError, "two integers"):
            self.parse()

    def test_unexpected_header(self):
        write_source(self.path, "2 3", self.series)
        with self.assertRaisesRegex(DataFormatError, "expected '31 290'"):
            self.parse(expected_header=(31, 290))

    def test_missing_value(self):
        self.series[1] = self.series[1][:-1]
        write_source(self.path, "2 3", self.series)
        with self.assertRaisesRegex(DataFormatError, "match no supported layout"):
            self.parse()

    def test_unsupported_layout_extra_values(self):
        self.series[2] = self.series[2] + [2.4, 2.5]
        write_source(self.path, "2 3", self.series)
        with self.assertRaisesRegex(DataFormatError, "match no supported layout"):
            self.parse()

    def test_count_fits_but_series_not_on_new_lines(self):
        flat = [v for s in self.series for v in s]
        self.path.write_text("2 3\n" + " ".join(map(str, flat[:5])) + "\n" + " ".join(map(str, flat[5:])) + "\n")
        with self.assertRaisesRegex(DataFormatError, "do not each start on a new line"):
            self.parse()

    def test_non_numeric_token(self):
        self.series[1][2] = "abc"
        write_source(self.path, "2 3", self.series)
        with self.assertRaisesRegex(DataFormatError, "'abc' is not a number"):
            self.parse()

    def test_non_positive_price(self):
        self.series[2][1] = 0.0
        write_source(self.path, "2 3", self.series)
        with self.assertRaisesRegex(DataFormatError, "not finite and positive"):
            self.parse()

    def test_nan_price(self):
        self.series[0][3] = "nan"
        write_source(self.path, "2 3", self.series)
        with self.assertRaisesRegex(DataFormatError, "not finite and positive"):
            self.parse()

    def test_index_not_first(self):
        write_source(self.path, "2 3", [self.series[1], self.series[0], self.series[2]])
        with self.assertRaisesRegex(DataFormatError, "is the index block first"):
            self.parse(expected_first_index=100.0)

    def test_empty_file(self):
        self.path.write_text("")
        with self.assertRaisesRegex(DataFormatError, "empty file"):
            self.parse()


if __name__ == "__main__":
    unittest.main()
