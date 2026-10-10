"""Task 2 tests: frozen universe, asset selection, archive reading, weekly aggregation, shipped CSV.

Unit tests need no network and no archive; ArchiveIntegrationTests run only when the local archive exists.
"""
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from stocks import kaggle_data as kd
from stocks.kaggle_data import KaggleDataError

FROZEN = {
    "Tech": ["AAPL", "MSFT", "INTC", "CSCO", "XLK"],
    "Gold & silver": ["GLD", "SLV", "GDX", "NEM"],
    "Energy (oil)": ["XOM", "CVX", "SLB", "USO"],
    "Finance (banks)": ["JPM", "BAC", "GS", "XLF"],
    "Bonds (safe haven)": ["TLT", "IEF", "SHY"],
    "Healthcare": ["JNJ", "PFE", "XLV"],
    "Consumer staples": ["PG", "KO", "WMT"],
    "Utilities": ["XLU"],
    "Real estate": ["VNQ"],
    "Materials (copper)": ["FCX"],
}
SELECT_12 = ["AAPL", "GLD", "XOM", "JPM", "TLT", "JNJ", "PG", "XLU", "VNQ", "FCX", "MSFT", "SLV"]
SELECT_29 = SELECT_12 + ["CVX", "BAC", "IEF", "PFE", "KO", "INTC", "GDX", "SLB", "GS", "SHY", "XLV", "WMT",
                         "CSCO", "NEM", "USO", "XLF", "XLK"]


class UniverseTests(unittest.TestCase):
    def test_frozen_universe_and_group_order(self):
        self.assertEqual(list(kd.GROUPS), list(FROZEN))
        for group, tickers in FROZEN.items():
            self.assertEqual([t for t, _, g in kd.UNIVERSE if g == group], tickers)
        self.assertEqual(len(kd.TICKERS), 29)
        self.assertEqual(kd.ALL_TICKERS[-1], "SPY")
        self.assertNotIn("SPY", kd.TICKERS)

    def test_select_12_exact(self):
        self.assertEqual(kd.select_assets(12), SELECT_12)

    def test_select_29_exact_and_complete(self):
        picked = kd.select_assets(29)
        self.assertEqual(picked, SELECT_29)
        self.assertEqual(sorted(picked), sorted(kd.TICKERS))

    def test_smaller_selections_are_prefixes(self):
        full = kd.select_assets(29)
        for n in range(2, 30):
            self.assertEqual(kd.select_assets(n), full[:n])

    def test_round_robin_one_per_group_first(self):
        first_round = kd.select_assets(len(kd.GROUPS))
        self.assertEqual([kd.GROUP_OF[t] for t in first_round], list(kd.GROUPS))
        for n in range(2, 30):
            counts = [sum(kd.GROUP_OF[t] == g for t in kd.select_assets(n)) for g in kd.GROUPS]
            # a group can only be ahead of another by one round, unless the other ran out of assets
            for g, c in zip(kd.GROUPS, counts):
                for h, d in zip(kd.GROUPS, counts):
                    if d < len(FROZEN[h]):
                        self.assertLessEqual(c - d, 1, (n, g, h))

    def test_spy_never_selected(self):
        for n in range(2, 30):
            self.assertNotIn("SPY", kd.select_assets(n))

    def test_invalid_sizes(self):
        for bad in (0, 1, 30, -3, 2.5, True, "12", None):
            with self.assertRaises(ValueError):
                kd.select_assets(bad)

    def test_shipped_metadata(self):
        rows = kd.load_metadata()
        self.assertEqual([r["ticker"] for r in rows], list(kd.ALL_TICKERS))
        self.assertEqual((rows[-1]["role"], rows[-1]["is_qubit"]), ("benchmark", "false"))
        self.assertTrue(all(r["is_qubit"] == "true" for r in rows[:-1]))


def daily_file(rows, header="Date,Open,High,Low,Close,Volume,OpenInt"):
    return "\n".join([header] + [f"{d},1,1,1,{p},100,0" for d, p in rows]) + "\n"


class ArchiveMemberTests(unittest.TestCase):
    NAMES = ["Data/ETFs/gld.us.txt", "ETFs/gld.us.txt", "Stocks/AAPL.US.TXT", "ETFs/dup.us.txt",
             "Stocks/dup.us.txt", "Other/xyz.us.txt", "ETFs/sub/abc.us.txt"]

    def test_canonical_folder_only(self):
        self.assertEqual(kd.resolve_member(self.NAMES, "GLD"), "ETFs/gld.us.txt")

    def test_case_insensitive(self):
        self.assertEqual(kd.resolve_member(self.NAMES, "aapl"), "Stocks/AAPL.US.TXT")

    def test_missing_or_outside_canonical_folders(self):
        for ticker in ("XYZ", "ABC", "NOPE"):
            with self.assertRaisesRegex(KaggleDataError, "no file"):
                kd.resolve_member(self.NAMES, ticker)

    def test_ambiguous(self):
        with self.assertRaisesRegex(KaggleDataError, "ambiguous"):
            kd.resolve_member(self.NAMES, "DUP")


class ParseDailyTests(unittest.TestCase):
    def test_valid(self):
        dates, prices = kd.parse_daily(daily_file([("2024-01-02", 10.5), ("2024-01-03", 11.0)]), "x")
        self.assertEqual(list(dates.astype(str)), ["2024-01-02", "2024-01-03"])
        self.assertEqual(list(prices), [10.5, 11.0])

    def test_errors(self):
        cases = {
            "duplicate date": [("2024-01-02", 1), ("2024-01-02", 2)],
            "out of order": [("2024-01-03", 1), ("2024-01-02", 2)],
            "not finite and positive": [("2024-01-02", 1), ("2024-01-03", 0)],
            "not YYYY-MM-DD": [("2024-1-2", 1)],
        }
        for message, rows in cases.items():
            with self.assertRaisesRegex(KaggleDataError, message):
                kd.parse_daily(daily_file(rows), "x")
        with self.assertRaisesRegex(KaggleDataError, "header must contain"):
            kd.parse_daily(daily_file([("2024-01-02", 1)], header="Date,Open,High,Low,Last,Volume,OpenInt"), "x")
        with self.assertRaisesRegex(KaggleDataError, "no price rows"):
            kd.parse_daily("Date,Close\n", "x")


class WeeklyAggregationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_week_ending_is_friday_saturday_starts_next_week(self):
        days = np.array(["2024-01-01", "2024-01-05", "2024-01-06", "2024-01-07"], dtype="datetime64[D]")
        self.assertEqual(list(kd.week_ending(days).astype(str)), ["2024-01-05", "2024-01-05", "2024-01-12", "2024-01-12"])

    def test_last_price_of_week_and_empty_week(self):
        dates = np.array(["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-15"], dtype="datetime64[D]")
        labels = np.array(["2024-01-05", "2024-01-12", "2024-01-19"], dtype="datetime64[D]")
        out = kd.weekly_last(dates, np.array([1.0, 2.0, 3.0, 4.0]), labels)
        self.assertEqual(out[0], 3.0)        # Friday missing: Thursday is the last price
        self.assertTrue(np.isnan(out[1]))    # no trading that week
        self.assertEqual(out[2], 4.0)

    def make_archive(self, files):
        path = self.tmp / "archive.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, text in files.items():
                archive.writestr(name, text)
        return path

    def test_build_weekly_common_window(self):
        aaa = [("2024-01-01", 1.0), ("2024-01-05", 2.0), ("2024-01-08", 3.0), ("2024-01-12", 4.0),
               ("2024-01-16", 4.5), ("2024-01-19", 5.0)]
        bbb = [("2024-01-03", 10.0), ("2024-01-10", 11.0), ("2024-01-16", 12.0)]
        path = self.make_archive({"Stocks/aaa.us.txt": daily_file(aaa), "ETFs/bbb.us.txt": daily_file(bbb),
                                  "Data/Stocks/aaa.us.txt": daily_file([("2024-01-01", 99.0)])})
        weekly = kd.build_weekly(path, tickers=("AAA", "BBB"))
        # common window: latest start (BBB, 01-03) to earliest end (BBB, 01-16)
        self.assertEqual(weekly.daily_window, (np.datetime64("2024-01-03"), np.datetime64("2024-01-16")))
        self.assertEqual(list(weekly.dates.astype(str)), ["2024-01-05", "2024-01-12", "2024-01-19"])
        # AAA: 01-01 and 01-19 fall outside the window; the Data/ copy (99.0) is never read
        np.testing.assert_array_equal(weekly.prices, [[2.0, 10.0], [4.0, 11.0], [4.5, 12.0]])

    def test_week_without_price_fails(self):
        aaa = [("2024-01-01", 1.0), ("2024-01-19", 2.0)]
        bbb = [("2024-01-01", 1.0), ("2024-01-10", 1.5), ("2024-01-19", 2.0)]
        path = self.make_archive({"Stocks/aaa.us.txt": daily_file(aaa), "Stocks/bbb.us.txt": daily_file(bbb)})
        with self.assertRaisesRegex(KaggleDataError, "weeks have no price"):
            kd.build_weekly(path, tickers=("AAA", "BBB"))

    def test_missing_archive_fails_clearly(self):
        with self.assertRaisesRegex(FileNotFoundError, "curl -L"):
            kd.build_weekly(self.tmp / "absent.zip")

    def test_csv_roundtrip(self):
        weekly = kd.WeeklyPrices(np.array(["2024-01-05", "2024-01-12"], dtype="datetime64[D]"), ["AAA", "BBB"],
                                 np.array([[1.0 / 3, 2.5], [1e-3, 123456.789]]))
        path = self.tmp / "w.csv"
        kd.write_csv(weekly, path)
        loaded = kd.load_csv(path, expected_tickers=("AAA", "BBB"))
        self.assertEqual(loaded.tickers, ["AAA", "BBB"])
        np.testing.assert_array_equal(loaded.dates, weekly.dates)
        np.testing.assert_array_equal(loaded.prices, weekly.prices)

    def test_csv_rejects_gaps_and_wrong_columns(self):
        path = self.tmp / "bad.csv"
        path.write_text("date,AAA\n2024-01-05,1\n2024-01-19,2\n")
        with self.assertRaisesRegex(KaggleDataError, "consecutive weeks"):
            kd.load_csv(path, expected_tickers=("AAA",))
        with self.assertRaisesRegex(KaggleDataError, "columns must be"):
            kd.load_csv(path, expected_tickers=("BBB",))


class ShippedCsvTests(unittest.TestCase):
    def test_shape_order_and_values(self):
        weekly = kd.load_csv()
        self.assertEqual(weekly.prices.shape, (599, 30))
        self.assertEqual(weekly.tickers, list(kd.ALL_TICKERS))
        self.assertEqual((str(weekly.dates[0]), str(weekly.dates[-1])), ("2006-05-26", "2017-11-10"))
        self.assertTrue(np.all(np.isfinite(weekly.prices) & (weekly.prices > 0)))
        self.assertLess(kd.CSV_PATH.stat().st_size, 2_000_000)

    def test_licence_note(self):
        ok, message = kd.check_licence()
        self.assertTrue(ok, message)


@unittest.skipUnless(kd.ARCHIVE_PATH.is_file(), "local Kaggle archive not present (integration test)")
class ArchiveIntegrationTests(unittest.TestCase):
    def test_rebuild_matches_shipped_csv(self):
        built = kd.build_weekly()
        shipped = kd.load_csv()
        self.assertEqual(built.tickers, shipped.tickers)
        np.testing.assert_array_equal(built.dates, shipped.dates)
        np.testing.assert_array_equal(built.prices, shipped.prices)


if __name__ == "__main__":
    unittest.main()
