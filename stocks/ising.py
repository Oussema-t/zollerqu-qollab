"""Ising encoding of the risk + expected-return score for two portfolio modes (NumPy only, no circuits).

Energy, lower is better:
    E(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i + const,      s_i in {+1, -1}

Mode A "long_short":   s_i = +1 long (buy), -1 short (bet against); weights s_i / n
    J_A(s) = s^T C s / (n^2 V_ref) - eta * mu.s / (n sigma_ref)
    J_ij = 2 C_ij / (n^2 V_ref),  h_i = -eta mu_i / (n sigma_ref),  const = trace(C) / (n^2 V_ref) = 1/n
Mode B "buy_or_skip":  x_i = (1 + s_i) / 2; s_i = +1 buy, -1 skip (that share stays in cash); weights x_i / n
    J_B(x) = x^T C x / (n^2 V_ref) - eta * mu.x / (n sigma_ref)
    J_ij = C_ij / (2 n^2 V_ref),  h_i = r_i / (2 n^2 V_ref) - eta mu_i / (2 n sigma_ref),  r_i = sum_j C_ij
    const = (sum(C) + trace(C)) / (4 n^2 V_ref) - eta sum(mu) / (2 n sigma_ref)
with V_ref = mean(diag C), sigma_ref = sqrt(V_ref), mu and C from the training window.

Thinning removes pair couplings only; fields and the constant always come from the dense C and mu.
A thinned model is therefore a SPARSE SURROGATE: its energy is not the dense score.

Balance: penalty lambda * (sum_i s_i - m)^2 adds 2*lambda to every pair, -2*lambda*m to every field and
lambda * (n + m^2) to the constant. Mode A: m = 0. Mode B: m = 2 * target_buys - n.
Large sizes use a hard constraint in the classical solvers instead (no dense penalty in the model).

Future circuit convention (no Qiskit here): s_i = +1 <-> qubit i in |0>, and in a Qiskit count key
qubit 0 is the RIGHTMOST character.
"""
import math
from dataclasses import dataclass

import numpy as np

from stocks.graph import pair_indices

MODE_LONG_SHORT, MODE_BUY_OR_SKIP = "long_short", "buy_or_skip"
MODES = (MODE_LONG_SHORT, MODE_BUY_OR_SKIP)
ETA_GRID = (0.25, 0.5, 1.0, 2.0)
DEFAULT_KEEP_FRACTION = {12: 1.0, 29: 0.60}  # share of pair couplings kept, ranked by |rho_ij|
BALANCE_NONE, BALANCE_PENALTY, BALANCE_HARD = "none", "penalty", "hard_constraint"
PENALTY_MARGIN = 0.01  # lambda is set 1 % above the smallest value that is provably sufficient
MAX_ENUMERATION_N = 20


def keep_fraction_for(n, table=None):
    """Keep fraction for size n; a size not listed uses the entry of the nearest listed size."""
    table = DEFAULT_KEEP_FRACTION if table is None else table
    if n in table:
        return table[n]
    return table[min(table, key=lambda size: (abs(size - n), size))]


def _check_inputs(mu, cov):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if mu.ndim != 1 or cov.shape != (len(mu), len(mu)) or len(mu) < 2:
        raise ValueError(f"mu {mu.shape} and cov {cov.shape} must be (n,) and (n, n) with n >= 2")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(cov))):
        raise ValueError("mu and cov must be finite")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14 * max(float(np.abs(cov).max()), 1e-300)):
        raise ValueError("cov must be symmetric")
    v_ref = float(np.mean(np.diag(cov)))
    if not v_ref > 0:
        raise ValueError(f"reference variance mean(diag C) = {v_ref} must be positive")
    return mu, cov, len(mu), v_ref, math.sqrt(v_ref)


def _check_mode_eta(mode, eta):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if isinstance(eta, bool) or not isinstance(eta, (int, float, np.floating, np.integer)) \
            or not math.isfinite(eta) or eta < 0:
        raise ValueError(f"eta must be a finite number >= 0, got {eta!r}")


def as_spins(spins, n):
    """Validate one spin vector (n,) or a batch (m, n) of +1/-1 values; returns a float array (m, n)."""
    s = np.asarray(spins)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2 or s.shape[1] != n:
        raise ValueError(f"spins must have {n} entries per state, got shape {np.asarray(spins).shape}")
    if not np.all((s == 1) | (s == -1)):
        raise ValueError("spins must be +1 or -1")
    return s.astype(float)


def spins_to_x(spins):
    """Mode B selection x = (1 + s) / 2: +1 -> 1 (buy), -1 -> 0 (skip)."""
    return (1 + np.asarray(spins)) // 2 if np.issubdtype(np.asarray(spins).dtype, np.integer) \
        else (1.0 + np.asarray(spins, dtype=float)) / 2.0


def x_to_spins(x):
    x = np.asarray(x)
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("selection x must be 0 or 1")
    return (2 * x - 1).astype(np.int8)


def spins_to_bitstring(spins):
    """Qiskit-style key: qubit 0 is the rightmost character; s = +1 -> '0', s = -1 -> '1'."""
    s = as_spins(spins, len(np.asarray(spins)))[0]
    return "".join("0" if v > 0 else "1" for v in s[::-1])


def bitstring_to_spins(key):
    if not key or set(key) - {"0", "1"}:
        raise ValueError(f"bitstring must contain only 0 and 1, got {key!r}")
    return np.array([1 if c == "0" else -1 for c in key[::-1]], dtype=np.int8)


def full_mask(n):
    return ~np.eye(n, dtype=bool)


def correlation_mask(rho, keep_fraction):
    """Keep the ceil(keep_fraction * pairs) pairs with the largest |rho_ij|; ties by pair index.

    Returns (mask, report). mask is (n, n) bool, symmetric, False on the diagonal.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] < 2:
        raise ValueError(f"rho must be square with n >= 2, got {rho.shape}")
    if not np.all(np.isfinite(rho)) or not np.allclose(rho, rho.T, atol=1e-12):
        raise ValueError("rho must be finite and symmetric")
    if not 0 < keep_fraction <= 1:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    n = rho.shape[0]
    iu, ju = pair_indices(n)
    values = rho[iu, ju]
    strength = np.abs(values)
    total = len(strength)
    keep = math.ceil(round(keep_fraction * total, 9))
    order = np.lexsort((np.arange(total), -strength))  # |rho| descending, then pair index
    kept, dropped = order[:keep], order[keep:]
    mask = np.zeros((n, n), dtype=bool)
    mask[iu[kept], ju[kept]] = True
    mask |= mask.T
    report = {
        "n": n,
        "keep_fraction": float(keep_fraction),
        "total_edges": total,
        "kept_edges": keep,
        "dropped_edges": total - keep,
        "rule": f"ceil({keep_fraction} * {total}) = {keep} pairs with the largest |rho_ij|, ties by pair index",
        "smallest_kept_abs_rho": float(strength[kept[-1]]),
        "largest_dropped_abs_rho": float(strength[dropped[0]]) if len(dropped) else None,
        "tie_at_threshold": bool(len(dropped) and strength[dropped[0]] == strength[kept[-1]]),
        "negative_edges_total": int(np.sum(values < 0)),
        "kept_negative_edges": int(np.sum(values[kept] < 0)),
        "isolated_nodes": [int(k) for k in np.flatnonzero(mask.sum(axis=1) == 0)],
        "dropped_pairs": [(int(iu[k]), int(ju[k]), float(values[k])) for k in dropped],
    }
    return mask, report


def _check_mask(keep_mask, n):
    if keep_mask is None:
        return full_mask(n)
    mask = np.asarray(keep_mask)
    if mask.dtype != bool or mask.shape != (n, n):
        raise ValueError(f"keep_mask must be a boolean ({n}, {n}) array")
    if not np.array_equal(mask, mask.T):
        raise ValueError("keep_mask must be symmetric")
    mask = mask.copy()
    np.fill_diagonal(mask, False)
    return mask


def dense_scores(spins, mu, cov, mode, eta):
    """The ORIGINAL dense score J_A or J_B for one state or a batch; never thinned, no penalty."""
    mu, cov, n, v_ref, sigma_ref = _check_inputs(mu, cov)
    _check_mode_eta(mode, eta)
    s = as_spins(spins, n)
    w = s if mode == MODE_LONG_SHORT else (1.0 + s) / 2.0
    scores = np.einsum("mi,ij,mj->m", w, cov, w) / (n ** 2 * v_ref) - eta * (w @ mu) / (n * sigma_ref)
    return float(scores[0]) if np.asarray(spins).ndim == 1 else scores


def score_parts(spins, mu, cov, mode):
    """Training proxies of one state: expected return E_p = mu.w and variance V_p = w^T C w, w = weights."""
    mu, cov, n, _, _ = _check_inputs(mu, cov)
    s = as_spins(spins, n)[0]
    w = (s if mode == MODE_LONG_SHORT else (1.0 + s) / 2.0) / n
    return float(mu @ w), float(w @ cov @ w)


def sparse_surrogate_energy(spins, mu, cov, mode, eta, keep_mask):
    """Direct evaluation of the sparse surrogate: the dense score minus the dropped pair terms.

    Written with an explicit loop, independently of IsingModel.energy, for testing.
    """
    mu, cov, n, v_ref, _ = _check_inputs(mu, cov)
    mask = _check_mask(keep_mask, n)
    s = as_spins(spins, n)[0]
    energy = dense_scores(s, mu, cov, mode, eta)
    factor = 2.0 / (n ** 2 * v_ref) if mode == MODE_LONG_SHORT else 1.0 / (2.0 * n ** 2 * v_ref)
    for i in range(n):
        for j in range(i + 1, n):
            if not mask[i, j]:
                energy -= factor * cov[i, j] * s[i] * s[j]
    return energy


@dataclass
class IsingModel:
    n: int
    couplings: np.ndarray    # (n, n) symmetric, zero diagonal; entry [i, j] is J_ij, counted once for i < j
    fields: np.ndarray       # (n,)
    const: float
    mode: str
    eta: float
    keep_mask: np.ndarray    # (n, n) bool: which objective pair couplings were kept
    balance: bool = False
    balance_method: str = BALANCE_NONE
    target_sum: int = None   # m in (sum s - m)^2
    target_buys: int = None  # mode B only
    penalty_lambda: float = 0.0

    def energy(self, spins):
        """Model energy of one state (float) or a batch (array). Includes the penalty when there is one."""
        s = as_spins(spins, self.n)
        e = 0.5 * np.einsum("mi,ij,mj->m", s, self.couplings, s) + s @ self.fields + self.const
        return float(e[0]) if np.asarray(spins).ndim == 1 else e

    def pair_terms(self):
        """[(i, j, J_ij)] for i < j with a non-zero coupling (what a circuit would need)."""
        iu, ju = pair_indices(self.n)
        return [(int(i), int(j), float(self.couplings[i, j])) for i, j in zip(iu, ju) if self.couplings[i, j] != 0.0]

    @property
    def kept_edges(self):
        return int(self.keep_mask.sum() // 2)

    def imbalance(self, spins):
        """|sum(s) - m| per state (0 when no balance is requested)."""
        s = as_spins(spins, self.n)
        return np.abs(s.sum(axis=1) - self.target_sum) if self.balance else np.zeros(len(s))

    def is_feasible(self, spins):
        """Balanced states: |sum(s) - m| at its smallest reachable value (1 for mode A with odd n)."""
        s = as_spins(spins, self.n)
        ok = self.imbalance(s) == min_imbalance(self.n, self.target_sum) if self.balance else np.ones(len(s), bool)
        return bool(ok[0]) if np.asarray(spins).ndim == 1 else ok


def reachable_sums(n):
    return np.arange(-n, n + 1, 2)


def min_imbalance(n, target_sum):
    return int(np.min(np.abs(reachable_sums(n) - target_sum)))


def all_states(n):
    """All 2^n spin states; bit i of the row index is qubit i, s = 1 - 2 * bit."""
    if n > MAX_ENUMERATION_N:
        raise ValueError(f"enumeration is limited to n <= {MAX_ENUMERATION_N}, got {n}")
    index = np.arange(1 << n, dtype=np.int64)
    return (1 - 2 * ((index[:, None] >> np.arange(n)) & 1)).astype(np.int8)


def safe_penalty_lambda(base_model, target_sum):
    """A lambda for which no infeasible state can beat the best feasible one (by enumeration).

    With R = max - min of the base energy over all states and G = the gap between the smallest and the
    second-smallest penalty value (sum s - m)^2, any lambda > R / G is sufficient. We use (1 + margin) * R / G.
    """
    energies = base_model.energy(all_states(base_model.n))
    spread = float(energies.max() - energies.min())
    penalties = np.unique((reachable_sums(base_model.n) - target_sum) ** 2)
    if len(penalties) < 2:
        raise ValueError("balance penalty needs at least two different imbalance levels")
    gap = float(penalties[1] - penalties[0])
    return (1.0 + PENALTY_MARGIN) * spread / gap if spread > 0 else 1.0


def build_ising(mu, cov, mode, eta, keep_mask=None, balance=False, balance_method=None, target_buys=None,
                penalty_lambda=None):
    """Ising model of the score. See the module docstring for the formulas.

    keep_mask thins pair couplings only (fields and constant stay dense).
    balance_method: "penalty" (needs enumeration, small n) or "hard_constraint" (solvers keep states balanced).
    """
    mu, cov, n, v_ref, sigma_ref = _check_inputs(mu, cov)
    _check_mode_eta(mode, eta)
    mask = _check_mask(keep_mask, n)

    off_diagonal = cov - np.diag(np.diag(cov))
    if mode == MODE_LONG_SHORT:
        couplings = 2.0 * off_diagonal / (n ** 2 * v_ref)
        fields = -eta * mu / (n * sigma_ref)
        const = float(np.trace(cov)) / (n ** 2 * v_ref)
    else:
        couplings = off_diagonal / (2.0 * n ** 2 * v_ref)
        fields = cov.sum(axis=1) / (2.0 * n ** 2 * v_ref) - eta * mu / (2.0 * n * sigma_ref)
        const = float(cov.sum() + np.trace(cov)) / (4.0 * n ** 2 * v_ref) - eta * float(mu.sum()) / (2.0 * n * sigma_ref)
    couplings = np.where(mask, couplings, 0.0)
    model = IsingModel(n, couplings, fields, const, mode, float(eta), mask)

    if not balance:
        if balance_method not in (None, BALANCE_NONE) or target_buys is not None or penalty_lambda is not None:
            raise ValueError("balance_method, target_buys and penalty_lambda need balance=True")
        return model

    method = BALANCE_PENALTY if balance_method is None else balance_method
    if method not in (BALANCE_PENALTY, BALANCE_HARD):
        raise ValueError(f"balance_method must be '{BALANCE_PENALTY}' or '{BALANCE_HARD}', got {balance_method!r}")
    if mode == MODE_LONG_SHORT:
        if target_buys is not None:
            raise ValueError("target_buys applies to mode 'buy_or_skip' only")
        target_sum = 0
    else:
        target_buys = n // 2 if target_buys is None else target_buys
        if isinstance(target_buys, bool) or not isinstance(target_buys, (int, np.integer)) or not 0 <= target_buys <= n:
            raise ValueError(f"target_buys must be an integer from 0 to {n}, got {target_buys!r}")
        target_buys = int(target_buys)
        target_sum = 2 * target_buys - n
    model.balance, model.balance_method = True, method
    model.target_sum, model.target_buys = target_sum, target_buys
    if method == BALANCE_HARD:
        if penalty_lambda is not None:
            raise ValueError("penalty_lambda is not used with balance_method='hard_constraint'")
        return model

    lam = safe_penalty_lambda(model, target_sum) if penalty_lambda is None else float(penalty_lambda)
    if not (math.isfinite(lam) and lam > 0):
        raise ValueError(f"penalty_lambda must be positive and finite, got {penalty_lambda!r}")
    model.couplings = model.couplings + 2.0 * lam * full_mask(n)
    model.fields = model.fields - 2.0 * lam * target_sum
    model.const = model.const + lam * (n + target_sum ** 2)
    model.penalty_lambda = lam
    return model


def describe(spins, tickers, mode):
    """Readable assignment: long/short lists for mode A, buy/skip lists for mode B."""
    s = as_spins(spins, len(tickers))[0]
    plus = [t for t, v in zip(tickers, s) if v > 0]
    minus = [t for t, v in zip(tickers, s) if v < 0]
    return {"long": plus, "short": minus} if mode == MODE_LONG_SHORT else {"buy": plus, "skip": minus}
