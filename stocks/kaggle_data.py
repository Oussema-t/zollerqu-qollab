"""Kaggle "Huge Stock Market Dataset" (borismarjanovic): 29 named assets + SPY as weekly closes.

Build the shipped CSV from the local archive (no network access):
    python -m stocks.kaggle_data
Runtime code only reads data/kaggle_weekly_29.csv and data/kaggle_asset_metadata.csv
(standard library + NumPy, no pandas, so it also runs on Qollab).

Data treatment, identical to notebook section K.2 (pandas resample("W-FRI").last()):
- price field "Close"; the dataset description says prices are adjusted for dividends and splits;
- common window: latest first trading day to earliest last trading day over the 30 series;
- weekly price = last available close in each week ending on Friday (weeks run Saturday..Friday).
"""
import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARCHIVE_PATH = DATA_DIR / "raw" / "kaggle" / "archive.zip"
CSV_PATH = DATA_DIR / "kaggle_weekly_29.csv"
METADATA_PATH = DATA_DIR / "kaggle_asset_metadata.csv"
LICENSE_PATH = DATA_DIR / "LICENSE_KAGGLE.txt"
DATASET_URL = "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/price-volume-data-for-all-us-stocks-etfs"

CANONICAL_FOLDERS = ("ETFs", "Stocks")  # the archive also holds identical copies under Data/
PRICE_FIELD = "Close"
FRIDAY = 4  # weekday number, Monday = 0

# Frozen universe (notebook K.2). The order fixes the group order and select_assets().
UNIVERSE = (
    ("AAPL", "Apple", "Tech"),
    ("MSFT", "Microsoft", "Tech"),
    ("INTC", "Intel", "Tech"),
    ("CSCO", "Cisco Systems", "Tech"),
    ("XLK", "Technology Select Sector SPDR fund", "Tech"),
    ("GLD", "SPDR Gold Shares (gold)", "Gold & silver"),
    ("SLV", "iShares Silver Trust (silver)", "Gold & silver"),
    ("GDX", "VanEck Gold Miners fund", "Gold & silver"),
    ("NEM", "Newmont (gold miner)", "Gold & silver"),
    ("XOM", "Exxon Mobil", "Energy (oil)"),
    ("CVX", "Chevron", "Energy (oil)"),
    ("SLB", "Schlumberger (oilfield services)", "Energy (oil)"),
    ("USO", "United States Oil Fund (oil futures)", "Energy (oil)"),
    ("JPM", "JPMorgan Chase", "Finance (banks)"),
    ("BAC", "Bank of America", "Finance (banks)"),
    ("GS", "Goldman Sachs", "Finance (banks)"),
    ("XLF", "Financial Select Sector SPDR fund", "Finance (banks)"),
    ("TLT", "iShares 20+ Year Treasury Bond fund", "Bonds (safe haven)"),
    ("IEF", "iShares 7-10 Year Treasury Bond fund", "Bonds (safe haven)"),
    ("SHY", "iShares 1-3 Year Treasury Bond fund", "Bonds (safe haven)"),
    ("JNJ", "Johnson & Johnson", "Healthcare"),
    ("PFE", "Pfizer", "Healthcare"),
    ("XLV", "Health Care Select Sector SPDR fund", "Healthcare"),
    ("PG", "Procter & Gamble", "Consumer staples"),
    ("KO", "Coca-Cola", "Consumer staples"),
    ("WMT", "Walmart", "Consumer staples"),
    ("XLU", "Utilities Select Sector SPDR fund", "Utilities"),
    ("VNQ", "Vanguard Real Estate fund", "Real estate"),
    ("FCX", "Freeport-McMoRan (copper miner)", "Materials (copper)"),
)
BENCHMARK = ("SPY", "SPDR S&P 500 fund (whole market)", "Market reference (not a qubit)")
TICKERS = tuple(t for t, _, _ in UNIVERSE)
ALL_TICKERS = TICKERS + (BENCHMARK[0],)
GROUPS = tuple(dict.fromkeys(g for _, _, g in UNIVERSE))
GROUP_OF = {t: g for t, _, g in UNIVERSE + (BENCHMARK,)}
NAME_OF = {t: name for t, name, _ in UNIVERSE + (BENCHMARK,)}
MIN_ASSETS = 2
MAX_QUBITS = len(UNIVERSE)  # 29 = IonQ Forte 1 noise-model simulator limit on Qollab

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class KaggleDataError(ValueError):
    """The archive or CSV does not match the expected content."""


def select_assets(n):
    """First n tickers, taking one asset from each group in turn (frozen order). Never includes SPY."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not MIN_ASSETS <= n <= MAX_QUBITS:
        raise ValueError(f"n = {n!r}: choose an integer from {MIN_ASSETS} to {MAX_QUBITS} (one asset per qubit)")
    queues = [[t for t, _, g in UNIVERSE if g == group] for group in GROUPS]
    picked = []
    while len(picked) < n:
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop(0))
    return picked


def resolve_member(names, ticker):
    """Archive member for a ticker under ETFs/ or Stocks/ (case-insensitive); fails if missing or ambiguous."""
    target = f"{ticker.lower()}.us.txt"
    matches = []
    for name in names:
        folder, _, filename = name.partition("/")
        if folder in CANONICAL_FOLDERS and "/" not in filename and filename.lower() == target:
            matches.append(name)
    if not matches:
        raise KaggleDataError(f"{ticker}: no file '{target}' under {' or '.join(f + '/' for f in CANONICAL_FOLDERS)}")
    if len(matches) > 1:
        raise KaggleDataError(f"{ticker}: ambiguous, several candidate files {matches}")
    return matches[0]


def parse_daily(text, label):
    """Dates (datetime64[D]) and PRICE_FIELD values of one daily file. Strict: nothing is dropped."""
    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 2:
        raise KaggleDataError(f"{label}: no price rows")
    header = rows[0]
    if "Date" not in header or PRICE_FIELD not in header:
        raise KaggleDataError(f"{label}: header must contain 'Date' and '{PRICE_FIELD}', got {header}")
    date_col, price_col = header.index("Date"), header.index(PRICE_FIELD)
    dates, prices = [], []
    for lineno, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise KaggleDataError(f"{label}:{lineno}: expected {len(header)} fields, got {len(row)}")
        if not _ISO_DATE.fullmatch(row[date_col]):
            raise KaggleDataError(f"{label}:{lineno}: date {row[date_col]!r} is not YYYY-MM-DD")
        try:
            dates.append(np.datetime64(row[date_col], "D"))
            prices.append(float(row[price_col]))
        except ValueError:
            raise KaggleDataError(f"{label}:{lineno}: bad date or price {row[date_col]!r}, {row[price_col]!r}") from None
    dates = np.array(dates, dtype="datetime64[D]")
    prices = np.array(prices)
    steps = np.diff(dates).astype(int)
    if np.any(steps == 0):
        raise KaggleDataError(f"{label}: duplicate date {dates[1:][steps == 0][0]}")
    if np.any(steps < 0):
        raise KaggleDataError(f"{label}: dates out of order at {dates[1:][steps < 0][0]}")
    bad = ~np.isfinite(prices) | (prices <= 0)
    if bad.any():
        raise KaggleDataError(f"{label}: {int(bad.sum())} prices not finite and positive (first on {dates[bad][0]})")
    return dates, prices


def week_ending(dates):
    """The Friday that closes each date's week (weeks run Saturday..Friday, like pandas 'W-FRI')."""
    days = np.asarray(dates, dtype="datetime64[D]").astype(np.int64)
    weekday = (days + 3) % 7  # 1970-01-01 was a Thursday (3)
    return (days + (FRIDAY - weekday) % 7).astype("datetime64[D]")


def weekly_last(dates, prices, labels):
    """Last price in each week ending on `labels` (sorted Fridays); NaN for a week without a price."""
    out = np.full(len(labels), np.nan)
    if len(dates) == 0:
        return out
    ends = week_ending(dates)
    last_in_week = np.r_[ends[1:] != ends[:-1], True]  # dates are increasing
    weeks, values = ends[last_in_week], prices[last_in_week]
    pos = np.searchsorted(labels, weeks)
    hit = pos < len(labels)
    hit[hit] = labels[pos[hit]] == weeks[hit]
    out[pos[hit]] = values[hit]
    return out


@dataclass
class WeeklyPrices:
    dates: np.ndarray   # (T,) week-ending Fridays, datetime64[D]
    tickers: list       # 29 assets in frozen order, then SPY
    prices: np.ndarray  # (T, len(tickers)) weekly closes
    daily_window: tuple = None  # (first day, last day) of daily data used, when built from the archive


def build_weekly(archive_path=ARCHIVE_PATH, tickers=ALL_TICKERS):
    """Weekly closes for `tickers` from the local Kaggle archive. Fails loudly instead of dropping data."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"{archive_path} not found. Download it once (no login needed):\n"
                                f"  curl -L -o {archive_path} {DOWNLOAD_URL}")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        series = {}
        for ticker in tickers:
            member = resolve_member(names, ticker)
            series[ticker] = parse_daily(archive.read(member).decode("utf-8"), member)
    start = max(d[0] for d, _ in series.values())
    end = min(d[-1] for d, _ in series.values())
    if start > end:
        raise KaggleDataError(f"no common window: latest start {start} is after earliest end {end}")
    first_week, last_week = week_ending(np.array([start, end]))
    labels = np.arange(first_week, last_week + np.timedelta64(1, "D"), np.timedelta64(7, "D"))
    columns = []
    for ticker in tickers:
        dates, prices = series[ticker]
        inside = (dates >= start) & (dates <= end)
        columns.append(weekly_last(dates[inside], prices[inside], labels))
    prices = np.column_stack(columns)
    missing = np.isnan(prices)
    if missing.any():
        row, col = map(int, np.argwhere(missing)[0])
        raise KaggleDataError(f"{int(missing.sum())} weeks have no price (first: {tickers[col]}, week ending {labels[row]})")
    return WeeklyPrices(labels, list(tickers), prices, (start, end))


def write_csv(weekly, path=CSV_PATH):
    """date,<29 assets>,SPY with ISO dates; floats use repr, which round-trips exactly."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["date", *weekly.tickers])
        for date, row in zip(weekly.dates, weekly.prices):
            writer.writerow([str(date), *(repr(float(p)) for p in row)])


def load_csv(path=CSV_PATH, expected_tickers=ALL_TICKERS):
    """Load the shipped weekly CSV, checking columns, weekly date order and prices."""
    path = Path(path)
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise KaggleDataError(f"{path}: no data rows")
    header, body = rows[0], rows[1:]
    if header[0] != "date" or (expected_tickers is not None and header[1:] != list(expected_tickers)):
        raise KaggleDataError(f"{path}: columns must be date,{','.join(expected_tickers or [])}; got {header[:4]}...")
    for lineno, row in enumerate(body, start=2):
        if len(row) != len(header) or any(cell.strip() == "" for cell in row):
            raise KaggleDataError(f"{path}:{lineno}: expected {len(header)} non-empty cells")
        if not _ISO_DATE.fullmatch(row[0]):
            raise KaggleDataError(f"{path}:{lineno}: date {row[0]!r} is not YYYY-MM-DD")
    dates = np.array([row[0] for row in body], dtype="datetime64[D]")
    if np.any(np.diff(dates).astype(int) != 7):
        raise KaggleDataError(f"{path}: dates must be consecutive weeks, 7 days apart, in order")
    try:
        prices = np.array([row[1:] for row in body], dtype=float)
    except ValueError as err:
        raise KaggleDataError(f"{path}: non-numeric price ({err})") from None
    if not np.all(np.isfinite(prices) & (prices > 0)):
        raise KaggleDataError(f"{path}: prices must be finite and positive")
    return WeeklyPrices(dates, header[1:], prices)


def write_metadata(path=METADATA_PATH):
    """ticker,name,group,role,is_qubit for the 29 assets (frozen order) and the SPY benchmark."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["ticker", "name", "group", "role", "is_qubit"])
        for ticker, name, group in UNIVERSE:
            writer.writerow([ticker, name, group, "asset", "true"])
        writer.writerow([*BENCHMARK, "benchmark", "false"])


def load_metadata(path=METADATA_PATH):
    """Rows of the metadata file as dicts; checks they match the frozen universe."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    expected = [dict(ticker=t, name=n, group=g, role="asset", is_qubit="true") for t, n, g in UNIVERSE]
    expected.append(dict(ticker=BENCHMARK[0], name=BENCHMARK[1], group=BENCHMARK[2], role="benchmark", is_qubit="false"))
    if rows != expected:
        raise KaggleDataError(f"{path}: metadata does not match the frozen universe in stocks/kaggle_data.py")
    return rows


LICENCE_MARKERS = ("CC0: Public Domain", "borismarjanovic", DATASET_URL)


def check_licence(path=LICENSE_PATH):
    """(ok, message) for the local licence note (it cannot re-check Kaggle itself)."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False, f"NEEDS REVIEW ({path.name} missing or empty)"
    text = path.read_text()
    missing = [m for m in LICENCE_MARKERS if m not in text]
    if missing:
        return False, f"NEEDS REVIEW ({path.name} lacks {missing})"
    return True, f"present ({path.name}: CC0 as reported by Kaggle, attribution and caveats recorded)"


def main():
    weekly = build_weekly()
    write_csv(weekly)
    write_metadata()
    reloaded = load_csv()
    same = reloaded.tickers == weekly.tickers and np.array_equal(reloaded.dates, weekly.dates) \
        and np.array_equal(reloaded.prices, weekly.prices)
    load_metadata()
    print(f"daily window {weekly.daily_window[0]} to {weekly.daily_window[1]}; weekly labels "
          f"{weekly.dates[0]} to {weekly.dates[-1]}: {weekly.prices.shape[0]} weeks x {weekly.prices.shape[1]} series")
    print(f"wrote {CSV_PATH.relative_to(DATA_DIR.parent)} ({CSV_PATH.stat().st_size:,} bytes), "
          f"reload identical: {'yes' if same else 'NO'}; wrote {METADATA_PATH.relative_to(DATA_DIR.parent)}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
