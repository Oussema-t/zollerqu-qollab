"""Held-out evaluation of a fixed assignment with SIMPLE weekly returns (NumPy only).

Portfolio P&L adds weighted simple returns P[t]/P[t-1] - 1 (log returns do not add across assets).
Mode A: weekly return = sum_i s_i R_i / n.   Mode B: sum_i x_i R_i / n, unselected money in cash at 0 return.
The assignment is held fixed and rebalanced to equal nominal weights every week.
Not modelled: short-borrow fees and availability, leverage/margin, trading costs, taxes, slippage.
"""
import numpy as np

from stocks.classical_solvers import random_spins
from stocks.ising import MODE_LONG_SHORT, as_spins


def simple_returns(prices):
    """R[t] = P[t] / P[t-1] - 1 along axis 0; same row alignment as returns.log_returns."""
    p = np.asarray(prices, dtype=float)
    if p.ndim not in (1, 2) or p.shape[0] < 2:
        raise ValueError(f"need at least 2 prices along axis 0, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("prices must be finite and positive")
    return p[1:] / p[:-1] - 1.0


def weights(spins, mode):
    """Nominal weights: s / n (mode A) or x / n with x = (1 + s) / 2 (mode B)."""
    s = np.asarray(spins, dtype=float)
    n = s.shape[-1]
    return (s if mode == MODE_LONG_SHORT else (1.0 + s) / 2.0) / n


def exposure(spins, mode):
    """Mode A: net exposure sum(s) / n. Mode B: invested fraction sum(x) / n."""
    return float(np.sum(weights(spins, mode)))


def series_metrics(weekly_returns):
    r = np.asarray(weekly_returns, dtype=float)
    return {"mean": float(r.mean()), "volatility": float(r.std(ddof=1)), "worst_week": float(r.min())}


def heldout_metrics(spins, mode, asset_returns, benchmark_returns):
    """Mean, volatility and worst week of the strategy, its exposure and its correlation with the benchmark."""
    r = np.asarray(asset_returns, dtype=float)
    s = as_spins(spins, r.shape[1])[0]
    strategy = r @ weights(s, mode)
    out = series_metrics(strategy)
    out["exposure"] = exposure(s, mode)
    # correlation is undefined for a constant series (e.g. mode B with nothing bought)
    out["corr_benchmark"] = float(np.corrcoef(strategy, benchmark_returns)[0, 1]) if np.ptp(strategy) > 0 else None
    return out


def random_portfolio_metrics(model, asset_returns, n_samples, rng):
    """Mean and volatility of n_samples random portfolios obeying the model's mode and balance rule."""
    states = random_spins(n_samples, model, rng)
    strategy = np.asarray(asset_returns, dtype=float) @ weights(states, model.mode).T  # (weeks, samples)
    return {"mean": strategy.mean(axis=0), "volatility": strategy.std(axis=0, ddof=1)}


def percentile_rank(value, sample):
    """Percentage of the sample that is <= value (0 to 100)."""
    return 100.0 * float(np.mean(np.asarray(sample) <= value))
