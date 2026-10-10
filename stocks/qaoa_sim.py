"""NumPy statevector simulator for QAOA on a diagonal Ising cost. Independent of Qiskit (offline tool).

Index convention (same as Qiskit and ising.all_states): basis state k has qubit i = bit i of k;
bit 0 (|0>) is spin +1, bit 1 (|1>) is spin -1.

Circuit simulated, layers k = 1..p, starting from H on every qubit (uniform amplitudes):
    phase  exp(-i * (gamma_k * E_pair(z) + gammaF_k * E_field(z)) / scale)     [the rzz and rz gates]
    mixer  RX(2 * beta_k) on every qubit
    extra  RY(2 * alpha_k) on every qubit                                       [only for the DC variant]
with E_pair(z) = sum_{i<j} J_ij z_i z_j and E_field(z) = sum_i h_i z_i. The model's constant is a global
phase: it is used for energies, never for the state. Standard QAOA has gammaF = gamma and no RY.
"""
import numpy as np

from stocks import ising

MAX_SIM_QUBITS = 20
SCALE_FLOOR = 1e-12
SCALE_POLICY = "scale = max(max|J_ij|, max|h_i|) of the model (one scale for couplings and fields); 1.0 if below 1e-12"


def cost_scale(model):
    """One common normalisation for couplings and fields, so the largest term has magnitude 1."""
    largest = max(float(np.abs(model.couplings).max()), float(np.abs(model.fields).max()))
    return largest if largest > SCALE_FLOOR else 1.0


def _by_qubit(amplitudes, qubit):
    """View with shape (high bits, bit of `qubit`, low bits): index k = high * 2^(q+1) + bit * 2^q + low."""
    return amplitudes.reshape(-1, 2, 1 << qubit)


def apply_rx(amplitudes, n, qubit, beta):
    """RX(2 beta) = [[cos b, -i sin b], [-i sin b, cos b]] on one qubit."""
    t = _by_qubit(amplitudes, qubit)
    return (np.cos(beta) * t - 1j * np.sin(beta) * t[:, ::-1, :]).reshape(-1)


_RY_SIGN = np.array([-1.0, 1.0]).reshape(1, 2, 1)


def apply_ry(amplitudes, n, qubit, alpha):
    """RY(2 alpha) = [[cos a, -sin a], [sin a, cos a]] on one qubit."""
    t = _by_qubit(amplitudes, qubit)
    return (np.cos(alpha) * t + np.sin(alpha) * _RY_SIGN * t[:, ::-1, :]).reshape(-1)


_BLOCKS = {}


def _block_tables(m):
    """For m qubits: D[a, b] = number of bits where a and b differ; U[a, b] = bits that are 0 in a and 1 in b."""
    if m not in _BLOCKS:
        index = np.arange(1 << m)
        popcount = np.array([bin(v).count("1") for v in index])
        _BLOCKS[m] = (popcount[index[:, None] ^ index[None, :]], popcount[~index[:, None] & index[None, :]])
    return _BLOCKS[m]


def rx_block(beta, m):
    """RX(2 beta) on each of m qubits as one 2^m x 2^m matrix: cos^(m-d) * (-i sin)^d, d = differing bits."""
    differ, _ = _block_tables(m)
    powers = np.arange(m + 1)
    return (np.cos(beta) ** powers)[m - differ] * ((-1j * np.sin(beta)) ** powers)[differ]


def ry_block(alpha, m):
    """RY(2 alpha) on each of m qubits: cos^(m-d) * sin^d with sign -1 for every bit going 0 -> 1 (row 0, column 1)."""
    differ, up = _block_tables(m)
    powers = np.arange(m + 1)
    return ((np.cos(alpha) ** powers)[m - differ] * (np.sin(alpha) ** powers)[differ] * (-1.0) ** up).astype(complex)


def apply_to_all(amplitudes, n, block, angle):
    """The same single-qubit rotation on every qubit: (block_high) A (block_low)^T with A = amplitudes[high, low].

    `block` is rx_block or ry_block. Same result as applying the gate qubit by qubit, but much faster.
    """
    low = n // 2
    a = amplitudes.reshape(1 << (n - low), 1 << low)
    return (block(angle, n - low) @ a @ block(angle, low).T).reshape(-1)


def apply_xy(amplitudes, n, qubit_a, qubit_b, theta):
    """XXPlusYY(theta) = exp(-i theta/2 * (XX + YY)/2) on two qubits: mixes |01> and |10>, keeps |00>, |11>.

    It never changes the number of 1 bits, so it keeps a balanced state balanced.
    """
    index = np.arange(len(amplitudes))
    differ = ((index >> qubit_a) & 1) != ((index >> qubit_b) & 1)
    swapped = index ^ ((1 << qubit_a) | (1 << qubit_b))
    out = amplitudes.astype(complex, copy=True)
    out[differ] = np.cos(theta / 2) * amplitudes[differ] - 1j * np.sin(theta / 2) * amplitudes[swapped[differ]]
    return out


class QaoaSimulator:
    """Exact probabilities of a QAOA circuit for one IsingModel. Counts how many circuits were simulated."""

    def __init__(self, model, scale=None):
        if model.n > MAX_SIM_QUBITS:
            raise ValueError(f"statevector simulation is limited to {MAX_SIM_QUBITS} qubits, got {model.n}")
        self.n = model.n
        self.dim = 1 << model.n
        self.scale = cost_scale(model) if scale is None else float(scale)
        if not self.scale > 0:
            raise ValueError(f"scale must be positive, got {scale}")
        states = ising.all_states(model.n).astype(float)
        self.pair = 0.5 * np.sum((states @ model.couplings) * states, axis=1)  # E_pair(z), no constant
        self.field = states @ model.fields                                      # E_field(z)
        self.energies = self.pair + self.field + model.const                    # model energy of every state
        self._pair_phase, self._field_phase = self.pair / self.scale, self.field / self.scale
        self._cost_phase = self._pair_phase + self._field_phase
        self.evaluations = 0

    def uniform(self):
        return np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=complex)

    def amplitudes(self, gammas, betas, field_gammas=None, alphas=None, initial=None, xy_pairs=None):
        """Final amplitudes. With xy_pairs the mixer is XXPlusYY(2 beta) on those pairs instead of RX."""
        gammas, betas = np.atleast_1d(gammas).astype(float), np.atleast_1d(betas).astype(float)
        if gammas.shape != betas.shape or gammas.ndim != 1:
            raise ValueError(f"gammas {gammas.shape} and betas {betas.shape} must be 1-D and the same length")
        if field_gammas is not None:
            field_gammas = np.atleast_1d(field_gammas).astype(float)
        if (field_gammas is not None and field_gammas.shape != gammas.shape) \
                or (alphas is not None and np.shape(alphas) != gammas.shape):
            raise ValueError("field_gammas and alphas must have one value per layer")
        amp = self.uniform() if initial is None else np.asarray(initial, dtype=complex).copy()
        for k in range(len(gammas)):
            if field_gammas is None:
                phase = gammas[k] * self._cost_phase
            else:
                phase = gammas[k] * self._pair_phase + field_gammas[k] * self._field_phase
            amp = amp * np.exp(-1j * phase)
            if xy_pairs is None:
                amp = apply_to_all(amp, self.n, rx_block, betas[k])
            else:
                for a, b in xy_pairs:
                    amp = apply_xy(amp, self.n, a, b, 2.0 * betas[k])
            if alphas is not None:
                amp = apply_to_all(amp, self.n, ry_block, alphas[k])
        self.evaluations += 1
        return amp

    def probabilities(self, gammas, betas, **kwargs):
        return np.abs(self.amplitudes(gammas, betas, **kwargs)) ** 2

    def expectation(self, probabilities):
        return float(probabilities @ self.energies)


def cvar(probabilities, energies, alpha):
    """Mean energy of the best (lowest-energy) alpha share of the probability mass, 0 < alpha <= 1."""
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    order = np.argsort(energies, kind="stable")
    p, e = probabilities[order], energies[order]
    before = np.cumsum(p) - p
    weight = np.clip(alpha - before, 0.0, p)  # full weight until the mass alpha is used up, then the remainder
    return float(weight @ e / weight.sum())


def expected_best_of_shots(probabilities, energies, shots):
    """Exact expected value of the lowest energy seen in `shots` independent samples."""
    order = np.argsort(energies, kind="stable")
    e = energies[order]
    none_up_to = (1.0 - np.clip(np.cumsum(probabilities[order]), 0.0, 1.0)) ** shots
    none_before = np.r_[1.0, none_up_to[:-1]]
    return float(e @ (none_before - none_up_to))


def distribution_metrics(probabilities, energies, e_min, e_max, tolerance, shots=1000, near=0.01):
    """Quality of a probability distribution over states. Lower energy is better.

    quality = (e_max - E) / (e_max - e_min), 1 at the optimum; 1.0 if all states have the same energy.
    'within 1 %' = energy <= e_min + 0.01 * (e_max - e_min), the definition used in Task 3.
    """
    spread = e_max - e_min
    quality = (lambda e: 1.0 if spread <= 0 else float((e_max - e) / spread))
    p_opt = float(probabilities[energies <= e_min + tolerance].sum())
    expectation = float(probabilities @ energies)
    best = expected_best_of_shots(probabilities, energies, shots)
    top = int(np.argmax(probabilities))
    return {
        "expectation": expectation, "quality": quality(expectation),
        "p_optimum": p_opt, "p_within_1pct": float(probabilities[energies <= e_min + near * spread].sum()),
        "p_optimum_in_shots": float(1.0 - (1.0 - min(p_opt, 1.0)) ** shots),
        "expected_best_in_shots": best, "expected_best_quality": quality(best), "shots": int(shots),
        "most_probable_state": top, "most_probable_probability": float(probabilities[top]),
        "most_probable_energy": float(energies[top]),
    }


def condition_on(probabilities, keep):
    """(share of probability on the kept states, renormalised distribution or None if that share is 0)."""
    share = float(probabilities[keep].sum())
    if share <= 0:
        return 0.0, None
    return share, np.where(keep, probabilities, 0.0) / share


def sample(probabilities, shots, rng):
    """State indices of `shots` samples (finite-shot estimate, as a backend would return)."""
    return rng.choice(len(probabilities), size=shots, p=probabilities / probabilities.sum())


def boltzmann(energies, tau):
    """Exact imaginary-time evolution of the uniform state for a DIAGONAL cost: p(z) ~ exp(-2 tau E(z)).

    This is a classical Boltzmann distribution, shown only as a reference curve.
    """
    weights = np.exp(-2.0 * tau * (energies - energies.min()))
    return weights / weights.sum()
