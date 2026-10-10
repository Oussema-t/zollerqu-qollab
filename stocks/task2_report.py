"""Task 2 run report: actual measured values from the shipped Kaggle CSV (local tool).

Run from the repo root:  python -m stocks.task2_report
"""
import io
import subprocess
import sys
import unittest

import numpy as np

from stocks import graph, kaggle_data, objective, returns

ROOT = kaggle_data.DATA_DIR.parent
SIZES = (12, 29)
ETA_SENSITIVITY = (0.25, 0.5, 1.0, 2.0)  # predeclared, shown only, never used to choose eta


def run_tests():
    """All unittest files in tests/. Returns (passed, failed, skipped, log)."""
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    failed = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    return result.testsRun - failed - skipped, failed, skipped, log.getvalue()


def git_checks():
    """Is the raw archive ignored, untracked and unstaged? (None when git is unavailable.)"""
    archive = str(kaggle_data.ARCHIVE_PATH.relative_to(ROOT))
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        ignored = git("check-ignore", "-q", archive).returncode == 0
        tracked = bool(git("ls-files", archive).stdout.strip())
        staged = git("diff", "--cached", "--name-only").stdout.split()
    except OSError:
        return None
    return {"ignored": ignored, "tracked": tracked, "staged": archive in staged}


def main(train_fraction=returns.DEFAULT_TRAIN_FRACTION, eta=objective.DEFAULT_RISK_RETURN_WEIGHT,
         sizes=SIZES, with_tests=True):
    weekly = kaggle_data.load_csv()
    kaggle_data.load_metadata()
    r = returns.log_returns(weekly.prices)
    dates = weekly.dates[1:]  # each return is labelled with its ending week
    split = returns.chronological_split(len(r), train_fraction)
    crash = returns.window_rows(dates)
    crash_in_train = bool(len(crash) and np.all(crash < split.split_index))
    n_bad = int(np.sum(~np.isfinite(weekly.prices) | (weekly.prices <= 0)))

    print("[TASK 2] Kaggle returns, split, covariance graph, risk-return diagnostics")
    print(f"- CSV: {kaggle_data.CSV_PATH.relative_to(ROOT)}, {weekly.prices.shape[0]} weeks x {weekly.prices.shape[1]} "
          f"series, {kaggle_data.CSV_PATH.stat().st_size:,} bytes, {weekly.dates[0]} to {weekly.dates[-1]}, "
          f"missing or non-positive prices: {n_bad}")
    print(f"- Columns: date, {', '.join(weekly.tickers)}")
    print(f"- Weekly log returns: {len(r)} rows ({dates[0]} to {dates[-1]})")
    print(f"- Split (floor({train_fraction} x {len(r)})): train {split.split_index} rows "
          f"({split.split_index / len(r):.1%}) {dates[split.train][0]} to {dates[split.train][-1]}; "
          f"test {len(r) - split.split_index} rows ({1 - split.split_index / len(r):.1%}) "
          f"{dates[split.test][0]} to {dates[split.test][-1]}")
    print(f"- 2008 crash ({returns.CRASH_START} to {returns.CRASH_END}): return rows {crash[0]}..{crash[-1]} "
          f"({len(crash)} weeks), all in training: {'yes' if crash_in_train else 'NO'}")

    for n in sizes:
        tickers = kaggle_data.select_assets(n)
        columns = [weekly.tickers.index(t) for t in tickers]
        groups = [kaggle_data.GROUP_OF[t] for t in tickers]
        stats = graph.training_stats(r[split.train][:, columns], tickers)
        checks = graph.matrix_checks(stats)
        neg = graph.negative_links(stats, groups, list(kaggle_data.GROUPS))
        sparse = graph.sparsify(stats.cov)
        rep = sparse.report
        print(f"- n={n}: {', '.join(tickers)}")
        print(f"    training rows used for mu, C, rho: {stats.n_obs}; matrix checks all pass: "
              f"{all(v for k, v in checks.items() if k != 'corr_diag_max_deviation')}")
        print(f"    negative links: {neg['n_negative_cov']} of {neg['n_pairs']} (cov) / {neg['n_negative_corr']} (corr), "
              f"sign disagreements: {len(neg['sign_disagreements'])}")
        top = ", ".join(f"{name} {count}" for name, count in neg["by_group_pair"][:4])
        print(f"    by group pair (top): {top}")
        isolated = [tickers[k] for k in rep["isolated_nodes"]]
        print(f"    sparse graph: kept {rep['kept_edges']} of {rep['total_edges']} edges, dropped |C| share "
              f"{rep['dropped_abs_share']:.1%}, kept negative edges {rep['kept_negative_edges']}, "
              f"isolated: {', '.join(isolated) or 'none'}")
        for label, spins in objective.example_spins(n).items():
            s = objective.risk_return_score(spins, stats.mu, stats.cov, eta)
            print(f"    example '{label}': V_p/V_ref {s['risk_term']:.4f}, eta*E_p/sigma_ref "
                  f"{s['return_term']:+.4f}, J {s['score']:+.4f} (eta {eta}, provisional)")

    ok_licence, licence_msg = kaggle_data.check_licence()
    print(f"- Licence notice: {licence_msg}")
    git = git_checks()
    if git is None:
        print("- Raw archive: git not available, not checked")
    else:
        print(f"- Raw archive: ignored {git['ignored']}, tracked {git['tracked']}, staged {git['staged']}")
    failed = 0
    if with_tests:
        passed, failed, skipped, log = run_tests()
        print(f"- Tests (python -m unittest discover -s tests -t .): {passed} passed, {failed} failed, {skipped} skipped")
        if failed:
            print(log)
    ok = crash_in_train and n_bad == 0 and ok_licence and failed == 0 and (git is None or not (git["tracked"] or git["staged"]))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
