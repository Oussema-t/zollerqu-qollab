"""OR-Library index-tracking data (indtrack1, Hang Seng): parse, validate, write and load the CSV.

Source format (OR-Library indtrackinfo.html):
    N T
    index value at t = 0..T
    for each stock i = 1..N: stock value at t = 0..T
Observed in indtrack1.txt: header "31 290", 9312 prices = 32 series x 291 observations,
each series starting on a new line. See data/README.md.

No network access here: the source file is downloaded once during development.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "raw" / "indtrack1.txt"
CSV_PATH = DATA_DIR / "hangseng_weekly.csv"
LICENSE_PATH = DATA_DIR / "LICENSE_ORLIB.txt"

EXPECTED_HEADER = (31, 290)
# Sanity check that the first block is the index: it starts at 8749.3, while every
# stock price in indtrack1 is below 500. A 1 % tolerance cannot confuse the two.
EXPECTED_FIRST_INDEX = 8749.0
FIRST_INDEX_REL_TOL = 0.01


class DataFormatError(ValueError):
    """The file does not match a supported OR-Library layout."""


@dataclass
class MarketData:
    weeks: np.ndarray   # (obs,) observation number 0..obs-1 (the source has no dates)
    index: np.ndarray   # (obs,) index level
    prices: np.ndarray  # (obs, N) stock prices; column k is names[k]
    names: list         # ["s01", ..., "sNN"] in source order


@dataclass
class ParseReport:
    header: tuple       # (N, T) as written in line 1
    n_values: int       # prices after the header
    layout: str         # "(N+1)*(T+1)" or "(N+1)*T"
    n_series: int       # index + stocks
    observations: int   # per series


def stock_names(n_stocks):
    width = max(2, len(str(n_stocks)))
    return [f"s{i:0{width}d}" for i in range(1, n_stocks + 1)]


def parse_orlib(path=RAW_PATH, expected_header=EXPECTED_HEADER,
                expected_first_index=EXPECTED_FIRST_INDEX, rel_tol=FIRST_INDEX_REL_TOL):
    """Parse an OR-Library index-tracking file. Returns (MarketData, ParseReport).

    A layout is accepted only if the price count matches it AND every series starts on a
    new line in the file. Anything else raises DataFormatError; nothing is dropped or filled.
    Pass expected_header=None / expected_first_index=None to skip those indtrack1-specific checks.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines or not lines[0].strip():
        raise DataFormatError(f"{path}: empty file or empty first line (expected 'N T')")

    head = lines[0].split()
    if len(head) != 2:
        raise DataFormatError(f"{path}:1: header must be two integers 'N T', got {lines[0].strip()!r}")
    try:
        n_stocks, n_periods = int(head[0]), int(head[1])
    except ValueError:
        raise DataFormatError(f"{path}:1: header must be two integers 'N T', got {lines[0].strip()!r}") from None
    if n_stocks < 1 or n_periods < 1:
        raise DataFormatError(f"{path}:1: header values must be positive, got {n_stocks} {n_periods}")
    if expected_header is not None and (n_stocks, n_periods) != tuple(expected_header):
        raise DataFormatError(f"{path}:1: header is '{n_stocks} {n_periods}', expected "
                              f"'{expected_header[0]} {expected_header[1]}'")

    values, line_starts = [], set()
    for lineno, line in enumerate(lines[1:], start=2):
        tokens = line.split()
        if tokens:
            line_starts.add(len(values))
        for token in tokens:
            try:
                values.append(float(token))
            except ValueError:
                raise DataFormatError(f"{path}:{lineno}: {token!r} is not a number") from None

    n_series = n_stocks + 1
    layouts = {"(N+1)*(T+1)": n_periods + 1, "(N+1)*T": n_periods}
    by_count = [(name, obs) for name, obs in layouts.items() if n_series * obs == len(values)]
    if not by_count:
        raise DataFormatError(
            f"{path}: {len(values)} prices after the header match no supported layout: "
            f"(N+1)*(T+1) = {n_series * (n_periods + 1)} or (N+1)*T = {n_series * n_periods} "
            f"(N={n_stocks}, T={n_periods}). Missing or extra values?")
    supported = [(name, obs) for name, obs in by_count
                 if all(k * obs in line_starts for k in range(n_series))]
    if len(supported) != 1:
        names = ", ".join(name for name, _ in by_count)
        raise DataFormatError(
            f"{path}: the price count fits {names}, but the series do not each start on a new "
            f"line, so the layout is not confirmed by the file format")
    layout, obs = supported[0]

    blocks = np.array(values).reshape(n_series, obs)
    bad = ~np.isfinite(blocks) | (blocks <= 0)
    if bad.any():
        series, t = map(int, np.argwhere(bad)[0])
        label = "index" if series == 0 else f"stock {series}"
        raise DataFormatError(f"{path}: {int(bad.sum())} prices are not finite and positive "
                              f"(first: {label}, observation {t}, value {blocks[series, t]})")

    first = float(blocks[0, 0])
    if expected_first_index is not None and abs(first / expected_first_index - 1) > rel_tol:
        raise DataFormatError(f"{path}: first index price is {first}, expected near "
                              f"{expected_first_index} (+/- {rel_tol:.0%}); is the index block first?")

    market = MarketData(np.arange(obs), blocks[0].copy(), blocks[1:].T.copy(), stock_names(n_stocks))
    report = ParseReport((n_stocks, n_periods), len(values), layout, n_series, obs)
    return market, report


def write_csv(market, path=CSV_PATH):
    """One row per observation: week,index,s01,... Floats use repr, which round-trips exactly."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["week", "index", *market.names])
        for t in range(len(market.weeks)):
            writer.writerow([int(market.weeks[t]), repr(float(market.index[t])),
                             *(repr(float(p)) for p in market.prices[t])])


def load_csv(path=CSV_PATH):
    """Load the CSV written by write_csv, checking columns, row lengths, week order and prices."""
    path = Path(path)
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise DataFormatError(f"{path}: empty CSV")
    header, body = rows[0], rows[1:]
    if header[:2] != ["week", "index"] or len(header) < 3:
        raise DataFormatError(f"{path}: header must start with 'week,index' and list stocks, got {header[:3]}")
    names = header[2:]
    if names != stock_names(len(names)):
        raise DataFormatError(f"{path}: stock columns must be {stock_names(len(names))[:2]}... in order, got {names[:3]}...")
    for lineno, row in enumerate(body, start=2):
        if len(row) != len(header) or any(cell.strip() == "" for cell in row):
            raise DataFormatError(f"{path}:{lineno}: expected {len(header)} non-empty cells, got {row}")
    try:
        table = np.array(body, dtype=float)
    except ValueError as err:
        raise DataFormatError(f"{path}: non-numeric cell ({err})") from None
    if table.size == 0:
        raise DataFormatError(f"{path}: no observations")
    weeks = table[:, 0]
    if not np.array_equal(weeks, np.arange(len(weeks))):
        raise DataFormatError(f"{path}: 'week' must be 0, 1, 2, ... in order")
    prices = table[:, 1:]
    if not np.all(np.isfinite(prices) & (prices > 0)):
        raise DataFormatError(f"{path}: prices must be finite and positive")
    return MarketData(weeks.astype(int), table[:, 1].copy(), table[:, 2:].copy(), names)
