"""Training-window statistics and covariance-graph diagnostics (NumPy only).

Diagnostics only: no Ising encoding, no Max-Cut and no optimisation here.
The dense covariance matrix stays the reference objective; sparsification only produces a mask.
"""
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

DEFAULT_KEEP_FRACTION = 0.30  # keep the strongest 30 % of links by |C_ij|
TOL = 1e-12


@dataclass
class TrainingStats:
    tickers: list
    n_obs: int
    mu: np.ndarray    # (n,) mean weekly log return
    cov: np.ndarray   # (n, n) sample covariance, ddof = 1
    corr: np.ndarray  # (n, n) Pearson correlation from the same rows


def training_stats(train_returns, tickers):
    """mu, covariance (ddof=1) and correlation of training returns (rows = weeks, columns = tickers)."""
    r = np.asarray(train_returns, dtype=float)
    if r.ndim != 2 or r.shape[1] != len(tickers):
        raise ValueError(f"returns shape {r.shape} does not match {len(tickers)} tickers")
    if r.shape[0] < 3:
        raise ValueError(f"need at least 3 training rows, got {r.shape[0]}")
    if not np.all(np.isfinite(r)):
        raise ValueError("training returns contain non-finite values")
    # Checked on the data itself: the computed variance of a constant column is rounding noise, not 0.
    constant = [t for t, column in zip(tickers, r.T) if np.ptp(column) == 0]
    if constant:
        raise ValueError(f"zero variance in the training window, correlation undefined for: {constant}")
    cov = np.cov(r, rowvar=False, ddof=1)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return TrainingStats(list(tickers), r.shape[0], r.mean(axis=0), cov, corr)


def matrix_checks(stats):
    """Shape, symmetry, finiteness and diagonal checks, as a dict of results."""
    n = len(stats.tickers)
    cov, corr = stats.cov, stats.corr
    scale = max(float(np.abs(cov).max()), TOL)
    return {
        "shape_ok": cov.shape == corr.shape == (n, n) and stats.mu.shape == (n,),
        "finite": bool(np.all(np.isfinite(cov)) and np.all(np.isfinite(corr)) and np.all(np.isfinite(stats.mu))),
        "cov_symmetric": bool(np.max(np.abs(cov - cov.T)) <= TOL * scale),
        "corr_symmetric": bool(np.max(np.abs(corr - corr.T)) <= TOL),
        "cov_diag_nonnegative": bool(np.all(np.diag(cov) >= -TOL * scale)),
        "corr_diag_max_deviation": float(np.max(np.abs(np.diag(corr) - 1.0))),
        "corr_within_bounds": bool(np.all(np.abs(corr) <= 1.0 + 1e-9)),
    }


def pair_indices(n):
    """All pairs i < j in pair-index order (0,1), (0,2), ..., (n-2,n-1)."""
    return np.triu_indices(n, k=1)


def group_pair(group_a, group_b, group_order):
    """Unordered group pair, written in the frozen group order."""
    a, b = sorted((group_a, group_b), key=group_order.index)
    return f"{a} – {b}"


def pair_table(stats, groups):
    """One dict per pair i < j: tickers, groups, covariance and correlation."""
    iu, ju = pair_indices(len(stats.tickers))
    return [dict(i=int(i), j=int(j), a=stats.tickers[i], b=stats.tickers[j], group_a=groups[i], group_b=groups[j],
                 cov=float(stats.cov[i, j]), corr=float(stats.corr[i, j])) for i, j in zip(iu, ju)]


def negative_links(stats, groups, group_order):
    """Pairs with negative covariance or correlation, sign agreement and counts per group pair."""
    pairs = pair_table(stats, groups)
    negative = sorted((p for p in pairs if p["cov"] < 0 or p["corr"] < 0), key=lambda p: p["corr"])
    disagree = [p for p in pairs if (p["cov"] < 0) != (p["corr"] < 0)]
    by_group = Counter(group_pair(p["group_a"], p["group_b"], group_order) for p in negative)
    return {
        "n_pairs": len(pairs),
        "n_negative_cov": sum(p["cov"] < 0 for p in pairs),
        "n_negative_corr": sum(p["corr"] < 0 for p in pairs),
        "sign_disagreements": disagree,
        "pairs": negative,
        "by_group_pair": by_group.most_common(),
    }


def extreme_correlations(stats, groups, k=5):
    """The k most negative and the k strongest positive correlations (ties broken by pair index)."""
    pairs = pair_table(stats, groups)
    by_corr = sorted(pairs, key=lambda p: (p["corr"], p["i"], p["j"]))
    return by_corr[:k], sorted(pairs, key=lambda p: (-p["corr"], p["i"], p["j"]))[:k]


@dataclass
class SparseGraph:
    edges: list       # kept pairs (i, j), strongest |C_ij| first
    mask: np.ndarray  # (n, n) bool, symmetric, False on the diagonal
    report: dict


def sparsify(cov, keep_fraction=DEFAULT_KEEP_FRACTION):
    """Keep the ceil(keep_fraction * n(n-1)/2) pairs with the largest |C_ij|.

    Ties in |C_ij| are broken by pair index order (lower first). The covariance matrix itself is not changed.
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 2:
        raise ValueError(f"need a square covariance matrix with n >= 2, got {c.shape}")
    if not 0 < keep_fraction <= 1:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    n = c.shape[0]
    iu, ju = pair_indices(n)
    weight = np.abs(c[iu, ju])
    total_edges = len(weight)
    keep = math.ceil(round(keep_fraction * total_edges, 9))
    order = np.lexsort((np.arange(total_edges), -weight))  # |C| descending, then pair index
    kept, dropped = order[:keep], order[keep:]
    mask = np.zeros((n, n), dtype=bool)
    mask[iu[kept], ju[kept]] = True
    mask |= mask.T
    total_weight = float(weight.sum())
    # All-zero covariance: nothing carries weight, so nothing is lost by dropping links.
    dropped_share = float(weight[dropped].sum()) / total_weight if total_weight > 0 else 0.0
    report = {
        "n": n,
        "total_edges": total_edges,
        "kept_edges": keep,
        "dropped_edges": total_edges - keep,
        "density": keep / total_edges,
        "keep_rule": f"ceil({keep_fraction} * {total_edges}) = {keep} strongest |C_ij|, ties by pair index",
        "smallest_kept_abs_cov": float(weight[kept[-1]]),
        "largest_dropped_abs_cov": float(weight[dropped[0]]) if len(dropped) else None,
        "tie_at_threshold": bool(len(dropped) and weight[dropped[0]] == weight[kept[-1]]),
        "dropped_abs_share": dropped_share,
        "zero_total_weight": total_weight == 0,
        "kept_negative_edges": int(np.sum(c[iu[kept], ju[kept]] < 0)),
        "isolated_nodes": [int(k) for k in np.flatnonzero(mask.sum(axis=1) == 0)],
    }
    edges = [(int(iu[k]), int(ju[k])) for k in kept]
    return SparseGraph(edges, mask, report)


def kept_group_pairs(graph, groups, group_order):
    """Kept edges per unordered group pair."""
    return Counter(group_pair(groups[i], groups[j], group_order) for i, j in graph.edges).most_common()
