"""Multi-threaded exact statevector simulation of standard QAOA at up to 29 qubits (numba; offline only).

Same circuit and conventions as large_sim.LargeQaoa and qaoa_sim.QaoaSimulator: bit i of the state index is
qubit i, bit 0 (|0>) is spin +1, cost phase exp(-i gamma E / scale), mixer rx(2 beta) on every qubit. It is
in place (one complex64 vector, 4.3 GB at 29 qubits) and parallel:
- the cost phase and the gates on the LOW qubits are applied chunk by chunk (a chunk stays in the cache);
- the HIGH qubits are applied two at a time, one pass over memory per pair of qubits;
- one more pass returns the norm, P(optimum) and a fine energy histogram (probability and probability-weighted
  energy per bin). Expectation, CVaR, P(within x %) and the expected best of B shots follow from the histogram
  without sampling; the bin width is 1/65536 of the energy range, which bounds their error.
"""
import math

import numba as nb
import numpy as np

from stocks import qaoa_sim

LOW_QUBITS = 14        # 2^14 amplitudes = 128 kB per chunk
BINS = 1 << 16


@nb.njit(parallel=True, cache=True)
def _fill(amp, value):
    for i in nb.prange(amp.size):
        amp[i] = value


@nb.njit(parallel=True, cache=True)
def _layer_low(amp, energies, k, c, m, low):
    """exp(-i k E) on every amplitude, then rx on qubits 0..low-1 (matrix [[c, m], [m, c]]), chunk by chunk."""
    chunk = 1 << low
    for ch in nb.prange(amp.size >> low):
        base = ch << low
        for i in range(base, base + chunk):
            angle = k * energies[i]
            amp[i] = amp[i] * np.complex64(complex(math.cos(angle), -math.sin(angle)))
        for q in range(low):
            stride = 1 << q
            for start in range(base, base + chunk, stride << 1):
                for i in range(start, start + stride):
                    a = amp[i]
                    b = amp[i + stride]
                    amp[i] = c * a + m * b
                    amp[i + stride] = m * a + c * b


@nb.njit(parallel=True, cache=True)
def _rx_one(amp, c, m, q):
    stride = 1 << q
    low_mask = stride - 1
    for t in nb.prange(amp.size >> 1):
        i0 = ((t >> q) << (q + 1)) | (t & low_mask)
        i1 = i0 | stride
        a = amp[i0]
        b = amp[i1]
        amp[i0] = c * a + m * b
        amp[i1] = m * a + c * b


@nb.njit(parallel=True, cache=True)
def _rx_two(amp, c2, cm, m2, q1, q2):
    """rx on qubits q1 < q2 in one pass: the 4 x 4 block kron(rx, rx) has entries c^2, c*m and m^2 only."""
    s1 = 1 << q1
    s2 = 1 << q2
    low_mask = s1 - 1
    mid_mask = (1 << (q2 - q1 - 1)) - 1
    for t in nb.prange(amp.size >> 2):
        i00 = ((t >> (q2 - 1)) << (q2 + 1)) | (((t >> q1) & mid_mask) << (q1 + 1)) | (t & low_mask)
        i01 = i00 | s1
        i10 = i00 | s2
        i11 = i01 | s2
        a = amp[i00]
        b = amp[i01]
        c = amp[i10]
        d = amp[i11]
        amp[i00] = c2 * a + cm * (b + c) + m2 * d
        amp[i01] = c2 * b + cm * (a + d) + m2 * c
        amp[i10] = c2 * c + cm * (a + d) + m2 * b
        amp[i11] = c2 * d + cm * (b + c) + m2 * a


@nb.njit(parallel=True, cache=True)
def _bin_index(energies, e_min, per_unit, nbins):
    out = np.empty(energies.size, dtype=np.uint16)
    for i in nb.prange(energies.size):
        b = int((energies[i] - e_min) * per_unit)
        out[i] = min(max(b, 0), nbins - 1)
    return out


@nb.njit(parallel=True, cache=True)
def _summary(amp, energies, bins, nbins):
    threads = nb.get_num_threads()
    mass = np.zeros((threads, nbins), dtype=np.float64)
    weighted = np.zeros((threads, nbins), dtype=np.float64)
    size = amp.size
    block = (size + threads - 1) // threads
    for th in nb.prange(threads):
        for i in range(th * block, min(size, (th + 1) * block)):
            a = amp[i]
            p = np.float64(a.real) * np.float64(a.real) + np.float64(a.imag) * np.float64(a.imag)
            b = bins[i]
            mass[th, b] += p
            weighted[th, b] += p * np.float64(energies[i])
    return mass.sum(axis=0), weighted.sum(axis=0)


class Distribution:
    """Output distribution of one circuit, as normalised probability mass per energy bin (ascending energy)."""

    def __init__(self, mass, weighted, p_optimum, norm, e_min, e_max):
        self.mass, self.weighted, self.p_optimum, self.norm, self.e_min, self.e_max = mass, weighted, p_optimum, norm, e_min, e_max
        self.expectation = float(weighted.sum())
        self.mean_energy = np.divide(weighted, mass, out=np.zeros_like(weighted), where=mass > 0)
        self.before = np.concatenate([[0.0], np.cumsum(mass)[:-1]])

    @property
    def quality(self):
        return (self.e_max - self.expectation) / (self.e_max - self.e_min)

    def hit(self, budget):
        """Probability that `budget` shots contain the optimum at least once."""
        return float(-np.expm1(budget * np.log1p(-min(self.p_optimum, 1.0 - 1e-15))))

    def cvar(self, alpha):
        weight = np.clip(alpha - self.before, 0.0, self.mass)
        return float(weight @ self.mean_energy / weight.sum())

    def within(self, fraction):
        """Probability of an energy within `fraction` of the range above the optimum (bin resolution)."""
        return float(self.mass[:max(int(fraction * len(self.mass)), 1)].sum())

    def expected_best(self, budget):
        """Expected lowest energy among `budget` shots (the lowest occupied bin is credited with its mean energy)."""
        survive = np.exp(budget * np.log1p(-np.clip(self.before, 0.0, 1.0 - 1e-15)))
        after = np.exp(budget * np.log1p(-np.clip(self.before + self.mass, 0.0, 1.0 - 1e-15)))
        return float((survive - after) @ self.mean_energy / (survive - after).sum())

    def sample_mean(self, shots, rng):
        """Mean energy of `shots` shots drawn from the distribution (what a device would estimate)."""
        counts = rng.multinomial(shots, self.mass / self.mass.sum())
        return float(counts @ self.mean_energy / shots)

    def summary(self, budgets=(1000,)):
        spread = self.e_max - self.e_min
        out = {"expectation": self.expectation, "quality": self.quality, "p_optimum": self.p_optimum, "probability_sum": self.norm,
               "cvar_0.1": self.cvar(0.1), "p_within_1pct": self.within(0.01)}
        for budget in budgets:
            out[f"hit@{budget}"] = self.hit(budget)
            out[f"best_quality@{budget}"] = (self.e_max - self.expected_best(budget)) / spread
        return out


class FastQaoa:
    """circuit_model / circuit_energies define the cost layer; grade_energies (default: the same) score the output."""

    def __init__(self, circuit_model, circuit_energies, grade_energies=None, low=LOW_QUBITS, bins=BINS):
        self.n = circuit_model.n
        self.scale = qaoa_sim.cost_scale(circuit_model)
        self.phase_energies = circuit_energies
        self.energies = circuit_energies if grade_energies is None else grade_energies
        self.low = min(low, self.n)
        self.e_min, self.e_max = float(self.energies.min()), float(self.energies.max())
        self.tolerance = 1e-6 * (self.e_max - self.e_min)       # float32 energies: same tie rule as large_sim
        self.optimal = np.flatnonzero(self.energies <= self.e_min + self.tolerance)
        self.nbins = bins
        self.bins = _bin_index(self.energies, self.e_min, bins / (self.e_max - self.e_min), bins)
        self.amp = np.empty(1 << self.n, dtype=np.complex64)
        self.evaluations = 0

    def state(self, gammas, betas):
        amp = self.amp
        _fill(amp, np.complex64(1.0 / math.sqrt(amp.size)))
        for gamma, beta in zip(np.atleast_1d(gammas), np.atleast_1d(betas)):
            block = qaoa_sim.rx_block(float(beta), 1)
            c, m = np.complex64(block[0, 0]), np.complex64(block[0, 1])
            _layer_low(amp, self.phase_energies, float(gamma) / self.scale, c, m, self.low)
            q = self.low
            while q + 1 < self.n:
                _rx_two(amp, np.complex64(c * c), np.complex64(c * m), np.complex64(m * m), q, q + 1)
                q += 2
            if q < self.n:
                _rx_one(amp, c, m, q)
        self.evaluations += 1
        return amp

    def probabilities(self, gammas, betas):
        return np.abs(self.state(gammas, betas).astype(np.complex128)) ** 2

    def evaluate(self, gammas, betas):
        amp = self.state(gammas, betas)
        mass, weighted = _summary(amp, self.energies, self.bins, self.nbins)
        norm = float(mass.sum())
        p_optimum = float(np.sum(np.abs(amp[self.optimal].astype(np.complex128)) ** 2)) / norm
        return Distribution(mass / norm, weighted / norm, p_optimum, norm, self.e_min, self.e_max)
