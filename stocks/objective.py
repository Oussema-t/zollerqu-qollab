"""Risk + expected-return score for a GIVEN long/short spin vector (diagnostic only, no optimisation).

Spins s_i = +1 (long) or -1 (short), weights w_i = s_i / n, training estimates mu (mean weekly log
return) and C (sample covariance):
    E_p(s) = (1/n)   * sum_i mu_i s_i         expected-return proxy
    V_p(s) = (1/n^2) * s^T C s                variance proxy
    J(s; eta) = V_p / V_ref - eta * E_p / sigma_ref,   V_ref = mean(diag C), sigma_ref = sqrt(V_ref)
Lower J is preferred. eta (risk_return_weight) = 1.0 is a PROVISIONAL default, not tuned or approved.
This is a simplified objective on weighted log returns, not a profit forecast or a full P&L model.
"""
import math

import numpy as np

DEFAULT_RISK_RETURN_WEIGHT = 1.0


def risk_return_score(spins, mu, cov, eta=DEFAULT_RISK_RETURN_WEIGHT):
    """Score breakdown for one spin vector, as a dict."""
    s = np.asarray(spins)
    mu = np.asarray(mu, dtype=float)
    c = np.asarray(cov, dtype=float)
    if s.ndim != 1 or len(s) < 1:
        raise ValueError(f"spins must be a 1-D vector, got shape {s.shape}")
    n = len(s)
    if not np.all((s == 1) | (s == -1)):
        raise ValueError("spins must be +1 (long) or -1 (short)")
    if mu.shape != (n,) or c.shape != (n, n):
        raise ValueError(f"mu {mu.shape} and cov {c.shape} must match {n} spins")
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(c)) and math.isfinite(eta)):
        raise ValueError("mu, cov and eta must be finite")
    if eta < 0:
        raise ValueError(f"eta must be >= 0, got {eta}")
    v_ref = float(np.mean(np.diag(c)))
    if not v_ref > 0:
        raise ValueError(f"reference variance mean(diag C) = {v_ref} must be positive")
    s = s.astype(float)
    expected = float(mu @ s) / n
    variance = float(s @ c @ s) / n ** 2
    sigma_ref = math.sqrt(v_ref)
    risk_term = variance / v_ref
    return_term = eta * expected / sigma_ref
    return {
        "n": n,
        "eta": float(eta),
        "variance": variance,
        "expected_return": expected,
        "v_ref": v_ref,
        "sigma_ref": sigma_ref,
        "risk_term": risk_term,
        "return_term": return_term,
        "score": risk_term - return_term,
    }


def example_spins(n, seed=2026):
    """Fixed example patterns for diagnostics. They are NOT optimised portfolios."""
    rng = np.random.default_rng(seed)
    return {
        "all long": np.ones(n, dtype=int),
        "alternating (+1 first, selection order)": np.array([1 if k % 2 == 0 else -1 for k in range(n)]),
        f"random (seed {seed})": rng.choice([-1, 1], size=n),
    }
