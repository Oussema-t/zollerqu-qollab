"""Task 1: rebuild data/hangseng_weekly.csv from the OR-Library source and validate it.

Run from the repo root:   python -m stocks.validate_data
Local tool: reads data/raw/indtrack1.txt and never downloads anything.
"""
import csv
import hashlib
import io
import math
import sys
import unittest

import numpy as np

from stocks import data

ROOT = data.DATA_DIR.parent
# Sentences of the OR-Library notice (compared with whitespace normalised); all must be present.
LICENCE_SENTENCES = (
    "All of the material available from OR-Library is freely available for use. "
    "As such the MIT licence, reproduced below, applies:",
    "Copyright (c) (2010) (J E Beasley} Permission is hereby granted, free of charge,",
    "The above copyright notice and this permission notice shall be included in all copies "
    "or substantial portions of the Software.",
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,',
)


def check_licence(path=data.LICENSE_PATH):
    """(ok, message) for the local notice file. It cannot re-check the website itself."""
    if not path.is_file() or path.stat().st_size == 0:
        return False, f"NEEDS REVIEW ({path.name} missing or empty)"
    text = " ".join(path.read_text().split())
    missing = sum(sentence not in text for sentence in LICENCE_SENTENCES)
    if missing:
        return False, f"NEEDS REVIEW ({missing} notice sentence(s) missing from {path.name})"
    return True, (f"verified ({path.relative_to(ROOT)} holds the full MIT notice, "
                  f"copied verbatim from OR-Library legal.html)")


def csv_cell_counts(path=data.CSV_PATH):
    """Rows, columns and price cells that are empty, non-numeric, non-finite or <= 0."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    bad = 0
    for row in rows[1:]:
        for cell in row[1:]:
            try:
                value = float(cell)
            except ValueError:
                bad += 1
                continue
            bad += not (math.isfinite(value) and value > 0)
    return len(rows) - 1, len(rows[0]), bad


def run_tests():
    """Run tests/test_stock_data.py. Returns (passed, failed, log)."""
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), pattern="test_stock_data.py", top_level_dir=str(ROOT))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    failed = len(result.failures) + len(result.errors)
    return result.testsRun - failed - len(result.skipped), failed, log.getvalue()


def main():
    market, report = data.parse_orlib(data.RAW_PATH)
    data.write_csv(market, data.CSV_PATH)
    reloaded = data.load_csv(data.CSV_PATH)
    identical = (reloaded.names == market.names
                 and np.array_equal(reloaded.index, market.index)
                 and np.array_equal(reloaded.prices, market.prices))
    n_rows, n_cols, bad_cells = csv_cell_counts(data.CSV_PATH)
    licence_ok, licence_msg = check_licence()
    passed, failed, log = run_tests()
    sha = hashlib.sha256(data.RAW_PATH.read_bytes()).hexdigest()

    print("[TASK 1] OR-Library dataset validation")
    print(f"- Source: original OR-Library file {data.RAW_PATH.relative_to(ROOT)} (sha256 {sha[:16]}...)")
    print(f"- Header: {report.header[0]} {report.header[1]}")
    print(f"- Layout: {report.layout} = {report.n_series} series x {report.observations} "
          f"= {report.n_values} prices, each series starting on a new line")
    print(f"- Stock series: {market.prices.shape[1]}")
    print(f"- Index series: {market.index.ndim}")
    print(f"- Price observations per series: {market.prices.shape[0]} "
          f"(index: {market.index.shape[0]})")
    print(f"- First index price: {float(market.index[0])!r}")
    print(f"- Invalid or missing prices: {bad_cells} (counted in the written CSV)")
    print(f"- CSV output: {data.CSV_PATH.relative_to(ROOT)} ({n_rows} rows x {n_cols} columns, "
          f"reload identical to source: {'yes' if identical else 'NO'})")
    print(f"- Licence notice: {licence_msg}")
    print(f"- Tests: {passed} passed, {failed} failed")
    if failed:
        print(log)
    ok = identical and bad_cells == 0 and licence_ok and failed == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
