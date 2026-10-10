"""QAOA circuits in Qiskit for an IsingModel (offline: no backend, no Aer, no qiskit-algorithms).

Energy E(z) = sum_{i<j} J_ij z_i z_j + sum_i h_i z_i + const, spin z_i = +1 <-> qubit i in |0>.
Circuit: H on all qubits, then for each layer k
    rzz(2 * gamma_k * J_ij / scale) on every pair term          = exp(-i gamma_k J_ij Z_i Z_j / scale)
    rz (2 * gammaF_k * h_i / scale) on every non-zero field     = exp(-i gammaF_k h_i Z_i / scale)
    rx (2 * beta_k) on every qubit                              (mixer)
    ry (2 * alpha_k) on every qubit                             (only for the counterdiabatic variant)
SIGN: rx(2 beta) is exp(-i beta X), and H on all qubits prepares the HIGHEST state of +sum X. So with gamma and
beta of the same sign the circuit moves towards HIGH energy; to lower the energy they must have opposite signs
(for example gamma > 0, beta < 0). Measured in Task 4: the 0.6 / 0.3 ramp as written is worse than random.
The constant is a global phase and is not a gate. All rzz gates of a layer commute, so they are ordered
into rounds of disjoint pairs to keep the two-qubit depth low; this does not change the unitary.
In a Qiskit count key qubit 0 is the RIGHTMOST character (see ising.bitstring_to_spins).
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from stocks.qaoa_sim import MAX_SIM_QUBITS, SCALE_POLICY, cost_scale

DEFAULT_DGAMMA, DEFAULT_DBETA = 0.6, 0.3  # LR-QAOA starting point
ROUTING_BASIS = ["rzz", "rz", "rx", "sx", "x"]  # rzz native on both targets, so only connectivity differs
ROUTING_SEED = 2026


def lr_schedule(p, dgamma=DEFAULT_DGAMMA, dbeta=DEFAULT_DBETA):
    """Linear ramp: gamma_k = (k/p) dgamma, beta_k = (1 - (k-1)/p) dbeta, k = 1..p. p = 0 gives no layers."""
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 0:
        raise ValueError(f"p must be an integer >= 0, got {p!r}")
    k = np.arange(1, p + 1)
    return (k / p) * dgamma if p else np.zeros(0), (1.0 - (k - 1) / p) * dbeta if p else np.zeros(0)


def layered_pair_terms(pair_terms):
    """The same terms, reordered into rounds in which no qubit appears twice (greedy, deterministic)."""
    remaining, ordered = list(pair_terms), []
    while remaining:
        busy, later = set(), []
        for term in remaining:
            i, j = term[0], term[1]
            if i in busy or j in busy:
                later.append(term)
            else:
                ordered.append(term)
                busy.update((i, j))
        remaining = later
    return ordered


def build_circuit(model, gammas, betas, scale=None, measure=True, field_gammas=None, alphas=None, layered=True):
    """QAOA circuit with fixed angles. len(gammas) = len(betas) = p; p = 0 is H on all qubits only."""
    gammas, betas = np.atleast_1d(gammas).astype(float), np.atleast_1d(betas).astype(float)
    if gammas.shape != betas.shape or gammas.ndim != 1:
        raise ValueError(f"gammas {gammas.shape} and betas {betas.shape} must be 1-D and the same length")
    field_gammas = gammas if field_gammas is None else np.atleast_1d(field_gammas).astype(float)
    if field_gammas.shape != gammas.shape or (alphas is not None and np.shape(alphas) != gammas.shape):
        raise ValueError("field_gammas and alphas must have one value per layer")
    scale = cost_scale(model) if scale is None else float(scale)
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale}")
    terms = layered_pair_terms(model.pair_terms()) if layered else model.pair_terms()
    n = model.n
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for k in range(len(gammas)):
        for i, j, coupling in terms:
            circuit.rzz(2.0 * gammas[k] * coupling / scale, i, j)
        for i in range(n):
            if model.fields[i] != 0.0:
                circuit.rz(2.0 * field_gammas[k] * model.fields[i] / scale, i)
        for q in range(n):
            circuit.rx(2.0 * betas[k], q)
        if alphas is not None:
            for q in range(n):
                circuit.ry(2.0 * alphas[k], q)
    if measure:
        circuit.measure_all()
    return circuit


def statevector_probabilities(circuit):
    """Exact probabilities from Qiskit's Statevector; index k has qubit i = bit i of k. Small circuits only."""
    if circuit.num_qubits > MAX_SIM_QUBITS:
        raise ValueError(f"refusing to simulate {circuit.num_qubits} qubits (limit {MAX_SIM_QUBITS}); "
                         f"large circuits are for construction and gate counts only")
    bare = circuit.remove_final_measurements(inplace=False)
    return np.asarray(Statevector.from_instruction(bare).probabilities())


def circuit_metrics(circuit):
    """Gate counts and depths of the circuit as built (measurements left out)."""
    bare = circuit.remove_final_measurements(inplace=False)
    ops = {name: int(count) for name, count in bare.count_ops().items()}
    return {
        "qubits": bare.num_qubits,
        "gates": ops,
        "total_gates": int(sum(ops.values())),
        "two_qubit_gates": int(sum(1 for inst in bare.data if inst.operation.num_qubits == 2)),
        "depth": int(bare.depth()),
        "two_qubit_depth": int(bare.depth(lambda inst: inst.operation.num_qubits == 2)),
    }


def routed_metrics(circuit, coupling_map=None, seed=ROUTING_SEED):
    """Two-qubit gates after transpiling (optimization_level 1) to ROUTING_BASIS, optionally on a coupling map.

    coupling_map=None means all-to-all (trapped ions). Routing on a sparse map adds SWAPs, which count here.
    """
    bare = circuit.remove_final_measurements(inplace=False)
    compiled = transpile(bare, basis_gates=ROUTING_BASIS, coupling_map=coupling_map, optimization_level=1,
                         seed_transpiler=seed)
    return {
        "two_qubit_gates": int(sum(1 for inst in compiled.data if inst.operation.num_qubits == 2)),
        "two_qubit_depth": int(compiled.depth(lambda inst: inst.operation.num_qubits == 2)),
        "gates": {name: int(count) for name, count in compiled.count_ops().items()},
    }


def describe(model, kind, gammas, betas, scale=None, **extra):
    """JSON-compatible description of one fixed-angle circuit."""
    scale = cost_scale(model) if scale is None else float(scale)
    out = {
        "qaoa_type": kind, "p": int(len(np.atleast_1d(gammas))),
        "gammas": [float(v) for v in np.atleast_1d(gammas)], "betas": [float(v) for v in np.atleast_1d(betas)],
        "scale": scale, "scale_policy": SCALE_POLICY,
        "mode": model.mode, "eta": model.eta, "balance": model.balance, "balance_method": model.balance_method,
        "pair_terms": len(model.pair_terms()), "kept_links": model.kept_edges,
    }
    out.update(extra)
    return out
