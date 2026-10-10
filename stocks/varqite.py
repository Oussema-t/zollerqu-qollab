"""varQITE: variational quantum imaginary-time evolution with a Pauli-term update rule, trained offline.

Reference: IonQ and ORNL, arXiv 2404.16135. Implemented from the method description only (no code reused).
Everything here is a classical statevector calculation; later only the final circuit is sampled on a backend.

Cost Hamiltonian (diagonal): H = sum_{i<j} J_ij Z_i Z_j + sum_i h_i Z_i  (+ const, a number).
Ansatz A: H on all qubits, then for every kept edge (i, j): ry(theta_e) on j, then cx(i, j).
Ansatz B: H on all qubits, then rzz(theta_e, i, j) per edge and one rx(theta_last) layer (IonQ-native gates).
One parameter per edge (plus one for ansatz B), all starting at 0: ansatz A then starts at |+...+>.

One step (no classical optimiser):
    E = <H>;   for every Pauli term P_k of H:  d_k = -(<P_k H> - <P_k><H>)     (imaginary-time direction)
    G[k, j] = d<P_k>/d theta_j   (parameter shift, +/- pi/2)
    solve G dtheta = d by least squares (rcond), theta <- theta + lr * dtheta
Expectations are exact (statevector); <P_k H> for diagonal operators is sum_z p(z) P_k(z) E(z).
"""
import numpy as np

from stocks import ising
from stocks.qaoa_sim import MAX_SIM_QUBITS, distribution_metrics

DEFAULTS = {"lr": 0.2, "steps": 60, "rcond": 1e-2, "patience": 10, "min_improvement": 1e-10}
LR_CANDIDATES = (0.1, 0.4, 0.8)   # 0.05 to 0.4 as described, plus 0.8: ansatz A was still improving at 0.4
# Ansatz B at theta = 0 is a stationary point: every parameter-shift derivative is exactly 0, so the update
# rule cannot move it. It therefore starts from small random angles (fixed seed); ansatz A starts at 0.
B_START_SCALE, B_START_SEED = 0.1, 2026
CITATION = "varQITE: IonQ and ORNL, arXiv 2404.16135 (implemented from the method description; no code reused)"


def pauli_terms(model):
    """[(qubits, coefficient)] of the diagonal cost: ZZ terms for kept pairs, Z terms for non-zero fields."""
    terms = [((i, j), coupling) for i, j, coupling in model.pair_terms()]
    terms += [((i,), float(h)) for i, h in enumerate(model.fields) if h != 0.0]
    return terms


def _view(amplitudes, qubit):
    """(high bits, bit of `qubit`, low bits) view of the amplitude vector."""
    return amplitudes.reshape(-1, 2, 1 << qubit)


def apply_ry(amplitudes, qubit, theta):
    """ry(theta) = [[cos t/2, -sin t/2], [sin t/2, cos t/2]] on one qubit (Qiskit convention)."""
    v = _view(amplitudes, qubit)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    out = np.empty_like(v)
    out[:, 0, :] = c * v[:, 0, :] - s * v[:, 1, :]
    out[:, 1, :] = s * v[:, 0, :] + c * v[:, 1, :]
    return out.reshape(-1)


def apply_cx(amplitudes, control, target):
    """cx: flip the target where the control bit is 1."""
    index = np.arange(len(amplitudes))
    on = ((index >> control) & 1) == 1
    out = amplitudes.copy()
    out[on] = amplitudes[index[on] ^ (1 << target)]
    return out


class Ansatz:
    """Statevector of ansatz A or B for an IsingModel; counts every circuit it evaluates."""

    def __init__(self, model, kind="A"):
        if model.n > MAX_SIM_QUBITS:
            raise ValueError(f"varQITE training is limited to {MAX_SIM_QUBITS} qubits, got {model.n}")
        if kind not in ("A", "B"):
            raise ValueError("kind must be 'A' (ry + cx per edge) or 'B' (rzz per edge + rx layer)")
        self.model, self.kind, self.n = model, kind, model.n
        self.edges = [(i, j) for i, j, _ in model.pair_terms()]
        self.n_params = len(self.edges) + (1 if kind == "B" else 0)
        states = ising.all_states(model.n).astype(float)
        self.energies = 0.5 * np.sum((states @ model.couplings) * states, axis=1) + states @ model.fields + model.const
        self.spins = states                      # (dim, n): spin value of each qubit in each basis state
        self.evaluations = 0

    def amplitudes(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"ansatz {self.kind} needs {self.n_params} parameters, got {theta.shape}")
        amp = np.full(1 << self.n, 1.0 / np.sqrt(1 << self.n), dtype=complex)
        if self.kind == "A":
            for (i, j), t in zip(self.edges, theta):
                amp = apply_ry(amp, j, t)
                amp = apply_cx(amp, i, j)
        else:
            for (i, j), t in zip(self.edges, theta[:-1]):
                amp = amp * np.exp(-0.5j * t * self.spins[:, i] * self.spins[:, j])   # rzz(t) = exp(-i t/2 Z Z)
            c, s = np.cos(theta[-1] / 2), -1j * np.sin(theta[-1] / 2)                   # rx(t) on every qubit
            for q in range(self.n):
                v = _view(amp, q)
                amp = (c * v + s * v[:, ::-1, :]).reshape(-1)
        self.evaluations += 1
        return amp

    def probabilities(self, theta):
        return np.abs(self.amplitudes(theta)) ** 2

    def two_qubit_gates(self):
        return len(self.edges)

    def to_qiskit(self, theta, measure=True):
        """The same circuit in Qiskit, for cross-checks and for the later backend batch."""
        from qiskit import QuantumCircuit
        circuit = QuantumCircuit(self.n)
        circuit.h(range(self.n))
        if self.kind == "A":
            for (i, j), t in zip(self.edges, theta):
                circuit.ry(float(t), j)
                circuit.cx(i, j)
        else:
            for (i, j), t in zip(self.edges, theta[:-1]):
                circuit.rzz(float(t), i, j)
            for q in range(self.n):
                circuit.rx(float(theta[-1]), q)
        if measure:
            circuit.measure_all()
        return circuit


def term_values(ansatz, terms):
    """P_k(z) for every term and basis state: the product of the spins on the term's qubits. Shape (terms, dim)."""
    return np.array([np.prod(ansatz.spins[:, list(qubits)], axis=1) for qubits, _ in terms])


def default_start(ansatz, seed=B_START_SEED):
    if ansatz.kind == "A":
        return np.zeros(ansatz.n_params)
    return np.random.default_rng(seed).uniform(-B_START_SCALE, B_START_SCALE, ansatz.n_params)


def train(model, kind="A", lr=DEFAULTS["lr"], steps=DEFAULTS["steps"], rcond=DEFAULTS["rcond"],
          patience=DEFAULTS["patience"], min_improvement=DEFAULTS["min_improvement"], theta0=None):
    """varQITE training. Returns (best theta, record). Stops early when the energy stops improving."""
    ansatz = Ansatz(model, kind)
    terms = pauli_terms(model)
    values = term_values(ansatz, terms)                     # (K, dim)
    energies = ansatz.energies
    theta = default_start(ansatz) if theta0 is None else np.array(theta0, dtype=float)
    shift = np.pi / 2
    history, best_energy, best_theta, since_best = [], np.inf, theta.copy(), 0
    for step in range(steps):
        p = ansatz.probabilities(theta)
        energy = float(p @ energies)
        history.append(energy)
        if energy < best_energy - min_improvement:
            best_energy, best_theta, since_best = energy, theta.copy(), 0
        else:
            since_best += 1
            if since_best >= patience:
                break
        expect = values @ p                                  # <P_k>
        d = -((values * energies) @ p - expect * energy)     # -(<P_k H> - <P_k><H>)
        gradient = np.empty((len(terms), ansatz.n_params))   # G[k, j] = d<P_k>/d theta_j by parameter shift
        for j in range(ansatz.n_params):
            plus, minus = theta.copy(), theta.copy()
            plus[j] += shift
            minus[j] -= shift
            gradient[:, j] = 0.5 * (values @ ansatz.probabilities(plus) - values @ ansatz.probabilities(minus))
        dtheta = np.linalg.lstsq(gradient, d, rcond=rcond)[0]
        theta = theta + lr * dtheta
    record = {"kind": kind, "lr": lr, "rcond": rcond, "steps_run": len(history), "steps_max": steps,
              "energy_per_step": history, "final_energy": history[-1], "best_energy": best_energy,
              "best_step": int(np.argmin(history)), "n_params": ansatz.n_params, "pauli_terms": len(terms),
              "evaluations": ansatz.evaluations, "two_qubit_gates": ansatz.two_qubit_gates(),
              "stopped_early": len(history) < steps, "start": "zeros" if kind == "A" else f"uniform(-{B_START_SCALE}, {B_START_SCALE}), seed {B_START_SEED}",
              "citation": CITATION}
    return best_theta, record


def train_one(model, kind, lr, evaluate=None, **kwargs):
    """One training run as a JSON-compatible record (theta included; metrics if `evaluate` is given)."""
    theta, record = train(model, kind, lr=lr, **kwargs)
    record["theta"] = [float(t) for t in theta]
    if evaluate is not None:
        record["metrics"] = evaluate(Ansatz(model, kind).probabilities(theta))
    return record


def select(runs):
    """The run with the lowest trained energy (training objective only)."""
    best = min(runs, key=lambda r: r["best_energy"])
    return {"runs": runs, "best": best, "selection_rule": "lowest trained energy (training objective only)"}


def tune(model, kinds=("A", "B"), lrs=LR_CANDIDATES, evaluate=None, **kwargs):
    """Train every (ansatz, lr) and pick the lowest final energy. evaluate: probabilities -> metrics dict."""
    return select([train_one(model, kind, lr, evaluate, **kwargs) for kind in kinds for lr in lrs])
