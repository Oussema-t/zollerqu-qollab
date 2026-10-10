"""Exact enumeration and classical baselines for an IsingModel (NumPy only, written by us).

Baselines: uniform random sampling, greedy descent, simulated annealing. With model.balance = True every
solver stays inside the balanced states (swap moves: one +1 with one -1), whatever the balance method.
"""
from dataclasses import dataclass

import numpy as np

from stocks.ising import all_states, reachable_sums

TIE_RTOL = 1e-9        # states within TIE_RTOL * (max - min) of the minimum count as tied optima
SA_T_END_FACTOR = 1e-3
MAX_GREEDY_STEPS = 10_000  # a correct descent on n <= 29 needs far fewer; more means a bug


@dataclass
class ExactResult:
    n_states: int           # all states enumerated (2^n)
    energies: np.ndarray    # model energy of every state, in index order
    feasible: np.ndarray    # balanced states (all True without balance)
    e_min: float            # over the search space (feasible states)
    e_max: float
    optimal_states: np.ndarray  # (k, n): every state tied at the minimum
    tolerance: float

    @property
    def n_optimal(self):
        return len(self.optimal_states)

    @property
    def search_space(self):
        return int(self.feasible.sum())


def exact_solve(model):
    """Enumerate all 2^n states (no spin is fixed: the fields break the spin-flip symmetry)."""
    states = all_states(model.n)
    energies = model.energy(states)
    feasible = model.is_feasible(states)
    space = energies[feasible]
    e_min, e_max = float(space.min()), float(space.max())
    tolerance = TIE_RTOL * (e_max - e_min)
    optimal = states[feasible & (energies <= e_min + tolerance)]
    return ExactResult(len(states), energies, feasible, e_min, e_max, optimal, tolerance)


def normalized_quality(energy, e_min, e_max):
    """(E_max - E) / (E_max - E_min): 1 at the optimum, 0 at the worst state; 1.0 if all states are equal."""
    spread = e_max - e_min
    return 1.0 if spread <= 0 else float((e_max - energy) / spread)


def verify_penalty(exact):
    """For a penalty model: does every infeasible state lose to the best feasible state?"""
    infeasible = exact.energies[~exact.feasible]
    if len(infeasible) == 0:
        return {"ok": True, "margin": None, "n_infeasible": 0}
    margin = float(infeasible.min() - exact.e_min)
    return {"ok": bool(margin > 0), "margin": margin, "n_infeasible": int(len(infeasible))}


def random_spins(n_samples, model, rng):
    """Uniform random states; with balance, uniform over the balanced states."""
    n = model.n
    if not model.balance:
        return (1 - 2 * rng.integers(0, 2, size=(n_samples, n))).astype(np.int8)
    sums = reachable_sums(n)
    distance = np.abs(sums - model.target_sum)
    allowed = sums[distance == distance.min()]          # one value, or +1/-1 for mode A with odd n
    n_plus = (n + rng.choice(allowed, size=n_samples)) // 2
    ranks = np.argsort(np.argsort(rng.random((n_samples, n)), axis=1), axis=1)
    return np.where(ranks < n_plus[:, None], 1, -1).astype(np.int8)


def greedy_descent(model, start):
    """Steepest descent to a local minimum: single flips, or +1/-1 swaps when the model is balanced."""
    s = np.array(start, dtype=float)
    couplings, fields = model.couplings, model.fields
    tolerance = 1e-12 * max(float(np.abs(couplings).sum() + np.abs(fields).sum()), 1e-300)
    for _ in range(MAX_GREEDY_STEPS):
        flip_gain = -2.0 * s * (couplings @ s + fields)  # energy change of flipping each spin alone
        if model.balance:
            plus, minus = np.flatnonzero(s > 0), np.flatnonzero(s < 0)
            if len(plus) == 0 or len(minus) == 0:
                break
            swap = flip_gain[plus][:, None] + flip_gain[minus][None, :] - 4.0 * couplings[np.ix_(plus, minus)]
            a, b = np.unravel_index(np.argmin(swap), swap.shape)
            if swap[a, b] >= -tolerance:
                break
            s[plus[a]], s[minus[b]] = -1.0, 1.0
        else:
            k = int(np.argmin(flip_gain))
            if flip_gain[k] >= -tolerance:
                break
            s[k] = -s[k]
    else:
        raise RuntimeError(f"greedy descent did not reach a local minimum in {MAX_GREEDY_STEPS} steps")
    return s.astype(np.int8)


@dataclass
class SolverResult:
    best_spins: np.ndarray
    best_energy: float
    energies: np.ndarray  # one per sample / restart


def _result(model, states):
    energies = model.energy(states)
    best = int(np.argmin(energies))
    return SolverResult(np.asarray(states[best], dtype=np.int8), float(energies[best]), energies)


def random_search(model, n_samples, rng):
    return _result(model, random_spins(n_samples, model, rng))


def greedy_restarts(model, n_restarts, rng):
    starts = random_spins(n_restarts, model, rng)
    return _result(model, np.array([greedy_descent(model, s) for s in starts]))


def simulated_annealing(model, n_restarts, sweeps, rng, t_start=None, t_end=None):
    """Metropolis annealing, all restarts in parallel, then a greedy polish.

    One sweep = n proposed moves per restart. Temperatures fall geometrically from t_start to t_end.
    Defaults: t_start = 2 * max_k (sum_j |J_kj| + |h_k|), the largest possible single-flip energy change
    (almost every move accepted), t_end = t_start / 1000 (frozen). Heuristic defaults, not tuned.
    """
    n, couplings, fields = model.n, model.couplings, model.fields
    s = random_spins(n_restarts, model, rng).astype(float)
    scale = 2.0 * float(np.max(np.abs(couplings).sum(axis=1) + np.abs(fields)))
    can_swap = not model.balance or (np.any(s[0] > 0) and np.any(s[0] < 0))
    if scale > 0 and can_swap and sweeps > 0:
        t_start = scale if t_start is None else t_start
        t_end = SA_T_END_FACTOR * t_start if t_end is None else t_end
        local = s @ couplings + fields
        rows = np.arange(n_restarts)
        for temperature in np.geomspace(t_start, t_end, sweeps):
            if model.balance:
                for _ in range(n):
                    i = np.argmax(rng.random((n_restarts, n)) * (s > 0), axis=1)  # a random +1 spin per restart
                    j = np.argmax(rng.random((n_restarts, n)) * (s < 0), axis=1)  # a random -1 spin per restart
                    delta = -2.0 * local[rows, i] + 2.0 * local[rows, j] - 4.0 * couplings[i, j]
                    accept = (delta <= 0) | (rng.random(n_restarts) < np.exp(-np.maximum(delta, 0.0) / temperature))
                    a = rows[accept]
                    s[a, i[a]], s[a, j[a]] = -1.0, 1.0
                    local[a] += -2.0 * couplings[i[a]] + 2.0 * couplings[j[a]]
            else:
                for k in rng.permutation(n):
                    delta = -2.0 * s[:, k] * local[:, k]
                    accept = (delta <= 0) | (rng.random(n_restarts) < np.exp(-np.maximum(delta, 0.0) / temperature))
                    a = rows[accept]
                    s[a, k] = -s[a, k]
                    local[a] += 2.0 * s[a, k][:, None] * couplings[k][None, :]
    return _result(model, np.array([greedy_descent(model, row) for row in s]))
