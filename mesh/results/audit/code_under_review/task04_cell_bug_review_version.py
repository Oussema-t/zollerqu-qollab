# TASK_04_IMPLEMENTATION_CELL_START
# Task 04: local noiseless QAOA simulation (several variants and optimisers), classical baselines on the same
# Ising objective, a crude analytical noise estimate and a modelled runtime / electricity outcome.
# Nothing here runs on IonQ or Qollab. "Wall-clock" = measured on this computer; "runtime", "electricity" = MODELLED.
import csv
import gzip
import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

TASK04 = dict(
    code_version="task04-v1",
    # agreed shared balance weights (Task 03 large-mode selection); Task 03's own instances are left untouched
    lambda_by_scenario={"2 equal devices": 10.0, "CPU + GPU (GPU 4x)": 3.98},
    mu=float(selected_mu),                         # priority weight kept from Task 03 (0 = priority penalty inactive)
    near_optimal_factor=1.05,                      # main metric: cost <= 1.05 x optimum
    epsilon=ISING_EPSILON,                         # balance tolerance
    # optimiser study
    eval_budget=200,                               # objective evaluations per optimisation run, identical for all optimisers
    random_starts=3,                               # seeded random starts in addition to the LR-QAOA start
    start_seed=4004,
    optimizers=["COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B", "SPSA"],
    random_gamma_max=1.0, random_beta_max=float(np.pi / 2),
    # SPSA gains follow Spall (1998): exponents 0.602 / 0.101, stability constant 10 % of the iterations,
    # step size set from a few initial gradient estimates so that the first move is about `first_step` (all inside the budget)
    # The perturbation size and the first-step size are chosen once per scenario on the calibration meshes (grids below).
    spsa=dict(alpha=0.602, gamma=0.101, stability_fraction=0.1, calibration_pairs=5,
              perturbation_grid=(0.02, 0.05, 0.1), first_step_grid=(0.01, 0.03, 0.1)),
    cvar_fraction=0.10, cvar_optimizers=["COBYLA", "SPSA"],
    warm_start_epsilon_rounded=0.25,               # Egger et al. (2021), Sec. 2.3
    warm_start_epsilon_continuous=0.05,            # our choice: the paper only gives the range [0, 0.5]
    warm_start_optimizers=["COBYLA"],
    lr_paper_default=(0.6, 0.3),                   # (delta_gamma, delta_beta) used on hardware in the LR-QAOA paper
    symmetry_optimizers=["COBYLA"],
    stress_layers=[3, 4, 5],                       # extra depths, LR-QAOA only (no optimiser loop)
    full_sweep_max_qubits=16,                      # above this size only LR-QAOA and COBYLA (LR start) are run
    # LR-QAOA schedule: chosen once per scenario on SEPARATE calibration meshes (other seeds), never on scored instances
    lr_grid_gamma=np.linspace(0.02, 2.0, 100), lr_grid_beta=np.linspace(0.05, 1.5, 30),
    calibration_cases=["2 main", "6 three sources"], calibration_seeds=[101, 102], calibration_n=12,
    # classical baselines: same objective, same budget of candidate evaluations as the QAOA shots
    classical_budget=int(RUN["shots"]), classical_repeats=100, classical_seed=5005,
    # simulated annealing: start temperature (as a fraction of the energy spread) and end/start ratio are also chosen
    # once per scenario on the calibration meshes, so the classical baseline is tuned as carefully as the quantum one
    sa_start_fraction_grid=(1.0, 0.3, 0.1, 0.03, 0.01), sa_end_ratio_grid=(1e-2, 1e-3), sa_calibration_repeats=20,
    # crude analytical noise estimate (NOT an IonQ result)
    noise_error_per_two_qubit_gate=0.005,
    # modelled outcome (assumptions): job length from Task 03b, communication as a share of the reference job time
    job_reference_hours=float(JOB_REFERENCE_HOURS), comm_fraction_if_all_borders_cut=0.05,
    use_checkpoint=True,
)
TASK04.update(globals().get("TASK04_OVERRIDES", {}))   # optional: define TASK04_OVERRIDES before this cell for quick tests
assert RUN["quantum_cases"], "RUN['quantum_cases'] is empty: Task 04 benchmarks QAOA, so it needs at least one quantum case"
T4_TOL = 1e-9
T4_RESULTS_DIR = Path("results")
T4_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
T4_CHECKPOINT = T4_RESULTS_DIR / f"task04_checkpoint_{RUN_MODE}.json.gz"
T4_SPSA_SETTINGS, T4_SA_SETTINGS = {}, {}        # per scenario, filled by the calibration below
T4_SHOTS = int(RUN["shots"])
T4_LAYERS = [int(p) for p in RUN["qaoa_layers"]]
TASK04_REFERENCES = {}        # filled below with the verified literature references


def t4_jsonable(value):
    """Convert numpy types and arrays to plain Python so results can be saved as JSON."""
    if isinstance(value, dict):
        return {str(k): t4_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [t4_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


T4_CONFIG_HASH = hashlib.sha256(json.dumps(t4_jsonable(TASK04), sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ simulator ----
def fast_kron(a, b):
    """Kronecker product of two small matrices (much faster than np.kron for these sizes)."""
    return (a[:, None, :, None] * b[None, :, None, :]).reshape(a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])


def kron_block(gates):
    """Kronecker product gates[-1] x ... x gates[0] (qubit 0 least significant), built as a balanced tree."""
    if len(gates) == 1:
        return gates[0]
    middle = len(gates) // 2
    return fast_kron(kron_block(gates[middle:]), kron_block(gates[:middle]))


def apply_qubit_gates(state, gates, per_qubit=False):
    """Apply one 2x2 gate per qubit (gates[q]) to a statevector. Qubit q is bit q of the state index (qubit 0 = least significant)."""
    n = len(gates)
    if 2 <= n <= 13 and not per_qubit:            # two dense blocks: (high block) @ state @ (low block)^T
        low = n // 2
        return (kron_block(gates[low:]) @ state.reshape(2 ** (n - low), 2**low) @ kron_block(gates[:low]).T).reshape(-1)
    for q in range(n):                            # larger registers: one qubit at a time
        view = state.reshape(-1, 2, 2**q)
        new = np.empty_like(view)
        new[:, 0, :] = gates[q][0, 0] * view[:, 0, :] + gates[q][0, 1] * view[:, 1, :]
        new[:, 1, :] = gates[q][1, 0] * view[:, 0, :] + gates[q][1, 1] * view[:, 1, :]
        state = new.reshape(-1)
    return state


def mixer_gates(beta, axis_angles):
    """One gate per qubit: exp(+i beta (sin(t) X + cos(t) Z)) for the axis angle t of that qubit.
    t = pi/2: the standard mixer exp(+i beta X) = RX(-2 beta), i.e. exp(-i beta H_M) with H_M = -X.
    t = +theta_q: warm-start mixer A of Egger et al. (2021), Eq. (2): RY(theta) RZ(-2 beta) RY(-theta); its ground state is the
                  warm-start initial state RY(theta)|0>.
    t = -theta_q: their modified mixer B for rounded solutions (Sec. 2.3): RY(-theta) RZ(-2 beta) RY(theta)."""
    c, s = np.cos(beta), 1j * np.sin(beta)
    gates = np.empty((len(axis_angles), 2, 2), dtype=complex)
    gates[:, 0, 0], gates[:, 1, 1] = c + s * np.cos(axis_angles), c - s * np.cos(axis_angles)
    gates[:, 0, 1] = gates[:, 1, 0] = s * np.sin(axis_angles)
    return gates


T4_HADAMARD_CACHE = {}


def uniform_x_mixer_block(beta, k):
    """exp(+i beta sum X) on k qubits as a dense matrix, via X = H Z H: H diag(exp(i beta (k - 2 popcount))) H."""
    if k not in T4_HADAMARD_CACHE:
        hadamard = np.ones((1, 1))
        for _ in range(k):
            hadamard = np.kron(hadamard, np.array([[1.0, 1.0], [1.0, -1.0]])) / np.sqrt(2.0)
        popcount = np.array([bin(index).count("1") for index in range(2**k)])
        T4_HADAMARD_CACHE[k] = (hadamard, k - 2 * popcount)
    hadamard, weights = T4_HADAMARD_CACHE[k]
    return (hadamard * np.exp(1j * beta * weights)) @ hadamard


def apply_uniform_x_mixer(state, beta, n):
    """Standard mixer exp(+i beta sum_q X_q) = prod_q RX(-2 beta) on all n qubits."""
    if 2 <= n <= 13:
        low = n // 2
        block_high = uniform_x_mixer_block(beta, n - low)
        block_low = block_high if n - low == low else uniform_x_mixer_block(beta, low)
        return (block_high @ state.reshape(2 ** (n - low), 2**low) @ block_low.T).reshape(-1)
    return apply_qubit_gates(state, mixer_gates(beta, np.full(n, np.pi / 2)))


def qaoa_statevector(energy_scaled, gammas, betas, n, thetas=None, mixer_axis=None):
    """Noiseless QAOA state. Cost layer: multiply state k by exp(-i gamma E_scaled[k]); then the mixer on every qubit.
    Standard QAOA: thetas = None (start |+>^n, mixer exp(+i beta X)). Warm start: start prod_q RY(theta_q)|0> and the
    mixer axis angles given in `mixer_axis` (+theta_q for mixer A, -theta_q for mixer B)."""
    standard = thetas is None
    if standard:
        state = np.full(2**n, 2.0 ** (-n / 2), dtype=complex)
    else:
        state = np.ones(1, dtype=complex)
        for q in range(n - 1, -1, -1):
            state = np.kron(state, np.array([np.cos(thetas[q] / 2), np.sin(thetas[q] / 2)]))
        mixer_axis = np.asarray(thetas if mixer_axis is None else mixer_axis, dtype=float)
    for gamma, beta in zip(gammas, betas):
        phase = gamma * energy_scaled
        state = state * (np.cos(phase) - 1j * np.sin(phase))            # = state * exp(-i gamma E_scaled)
        state = apply_uniform_x_mixer(state, beta, n) if standard else apply_qubit_gates(state, mixer_gates(beta, mixer_axis))
    return state


def qaoa_probabilities(energy_scaled, gammas, betas, n, thetas=None, mixer_axis=None):
    return np.abs(qaoa_statevector(energy_scaled, gammas, betas, n, thetas, mixer_axis)) ** 2


def view_probabilities(view, angles):
    """Outcome distribution of a QAOA run described by a view (energies, start state, mixer) and its angles [gammas, betas]."""
    return qaoa_probabilities(view["energy_scaled"], *split_angles(angles), view["n"], view["thetas"], view["mixer_axis"])


def lr_angles(p, delta_gamma, delta_beta):
    """LR-QAOA linear ramp for layers k = 1..p: gamma_k = (k/p) dgamma, beta_k = (1 - (k-1)/p) dbeta. Returns [gammas, betas]."""
    k = np.arange(1, p + 1)
    return np.concatenate([k / p * delta_gamma, (1.0 - (k - 1) / p) * delta_beta])


def split_angles(angles):
    p = len(angles) // 2
    return np.asarray(angles[:p], float), np.asarray(angles[p:], float)


def round_robin_pairs(n):
    """All pairs of n qubits arranged in rounds of disjoint pairs (circle method), so a dense rzz layer has minimal depth."""
    players = list(range(n)) + ([None] if n % 2 else [])
    m, rounds = len(players), []
    for _ in range(m - 1):
        rounds.append([(min(a, b), max(a, b)) for a, b in zip(players[: m // 2], players[::-1][: m // 2])
                       if a is not None and b is not None])
        players = [players[0], players[-1]] + players[1:-1]
    return rounds


def build_qaoa_circuit(J_scaled, h_scaled, gammas, betas, thetas=None, modified_mixer=False):
    """The Qiskit circuit that later runs on IonQ: H (or RY) start, rzz(2 gamma J) on every pair, rz(2 gamma h), rx(-2 beta).
    Warm start: mixer A = RY(theta) RZ(-2 beta) RY(-theta) (operator order), mixer B (modified_mixer) has the two RY swapped."""
    n = len(h_scaled)
    circuit = QuantumCircuit(n)
    for q in range(n):
        circuit.h(q) if thetas is None else circuit.ry(float(thetas[q]), q)
    for gamma, beta in zip(gammas, betas):
        for pairs in round_robin_pairs(n):
            for i, j in pairs:
                if J_scaled[i, j] != 0:
                    circuit.rzz(float(2 * gamma * J_scaled[i, j]), i, j)
        for q in range(n):
            if h_scaled[q] != 0:
                circuit.rz(float(2 * gamma * h_scaled[q]), q)
        for q in range(n):
            if thetas is None:
                circuit.rx(float(-2 * beta), q)
            else:                                   # operator products act right to left, circuits list gates in time order
                sign = -1.0 if modified_mixer else 1.0
                circuit.ry(float(-sign * thetas[q]), q)
                circuit.rz(float(-2 * beta), q)
                circuit.ry(float(sign * thetas[q]), q)
    return circuit


# ------------------------------------------------------------------ instances ----
def t4_power(device_name):
    """Assumed busy / idle power (W) of a scenario device: the average of the synthetic pool devices of the same kind."""
    kind = "GPU" if "GPU" in device_name.upper() else "CPU"
    members = [d for d in hpc_pool if d["kind"] == kind]
    assert members, f"no {kind} in the synthetic pool: power for '{device_name}' is unavailable"
    return float(np.mean([d["active_w"] for d in members])), float(np.mean([d["idle_w"] for d in members]))


def t4_build_instance(regions, case_name, n, seed, scenario):
    """Task-04-local Ising instance with the agreed lambda; exact reference by enumeration where the size allows."""
    alpha, lam = TARGET_SHARES[scenario], TASK04["lambda_by_scenario"][scenario]
    (name_a, speed_a), (name_b, speed_b) = DEVICE_SCENARIOS[scenario].items()
    priority_active = bool(regions["region_priority_work"].sum() > 0 and speed_b > speed_a and TASK04["mu"] > 0)
    mu = TASK04["mu"] if priority_active else 0.0
    model = build_partition_ising(regions["region_work"], regions["region_pairs"], regions["region_boundary_cost"], alpha,
                                  lam=lam, priority_work=regions["region_priority_work"], mu=mu)
    J, h, const = model["J"], model["h"], model["const"]
    check_rng = np.random.default_rng([TASK04["start_seed"], n, seed])
    for z in 1 - 2 * check_rng.integers(0, 2, size=(50, n)):          # same objective as Task 03: Ising energy = direct cost
        assert abs(ising_energy(z, J, h, const) - direct_cost(z, model)) < T4_TOL
    inst = dict(case=case_name, n=n, seed=seed, scenario=scenario, alpha=alpha, lam=lam, mu=mu, model=model, regions=regions,
                priority_active=priority_active, speeds=(float(speed_a), float(speed_b)), device_names=(name_a, name_b),
                power=(t4_power(name_a), t4_power(name_b)),
                e_std_formula=float(np.sqrt(np.sum(J**2) + np.sum(h**2))),      # exact std of E under uniform sampling
                fingerprint=hashlib.sha256(J.tobytes() + h.tobytes() + np.float64(const).tobytes()).hexdigest()[:16],
                exact=n <= EXACT_BRUTE_FORCE_MAX_QUBITS, statevector=n <= LOCAL_STATEVECTOR_MAX_QUBITS)
    if inst["exact"]:
        cut, share_b, prio_slow = enumerate_components(model)
        energies = cut + lam * 4.0 * (alpha - share_b) ** 2 + mu * prio_slow
        if n <= 16:                                 # full cross-check of the component formula against J, h, const
            assert np.max(np.abs(energies - ising_energies_all(J, h, const, n))) < T4_TOL
        e_min, e_max = float(energies.min()), float(energies.max())
        near_limit = TASK04["near_optimal_factor"] * e_min if e_min > T4_TOL else T4_TOL   # optimum 0: absolute tolerance
        imbalance = np.abs(share_b - alpha)
        inst.update(energies=energies, cut=cut, share_b=share_b, imbalance=imbalance, e_min=e_min, e_max=e_max,
                    e_mean=float(energies.mean()), e_std=float(energies.std()),
                    optimal_states=np.flatnonzero(energies <= e_min + T4_TOL), near_mask=energies <= near_limit + T4_TOL,
                    balanced_mask=imbalance <= TASK04["epsilon"], delta_star=float(imbalance.min()),
                    energy_scaled=(energies - const) / model["scale"])
        hours = TASK04["job_reference_hours"]
        inst["modelled_hours_all"] = (hours * np.maximum((1.0 - share_b) / speed_a, share_b / speed_b)
                                      + TASK04["comm_fraction_if_all_borders_cut"] * hours * cut)
        assert abs(inst["e_std"] - inst["e_std_formula"]) < 1e-6 * max(1.0, inst["e_std"]), "uniform-std formula is wrong"
    return inst


def t4_view(inst, thetas=None, mixer_axis=None):
    """What a QAOA run needs: energies, scaled energies, start state / mixer and reference sets of the (full) problem."""
    return dict(n=inst["n"], energies=inst["energies"], energy_scaled=inst["energy_scaled"], thetas=thetas, mixer_axis=mixer_axis,
                optimal_states=inst["optimal_states"], near_mask=inst["near_mask"], balanced_mask=inst["balanced_mask"],
                e_min=inst["e_min"], e_max=inst["e_max"], e_mean=inst["e_mean"], e_std=inst["e_std"], shift=0)


def t4_reduced_view(inst):
    """Symmetry reduction for alpha = 0.5 without priority: E(z) = E(-z), so region 0 is fixed to device A (x_0 = 0)
    and the problem lives on n - 1 qubits: J' = J[1:, 1:], h'_i = h_i + J_0i, const' = const + h_0."""
    model = inst["model"]
    J, h, const, n = model["J"], model["h"], model["const"], inst["n"]
    J_red, h_red, const_red = J[1:, 1:], h[1:] + J[0, 1:], const + h[0]
    energies_red = inst["energies"][0::2]                        # states with x_0 = 0 are the even state indices
    assert np.max(np.abs(energies_red - ising_energies_all(J_red, h_red, const_red, n - 1))) < T4_TOL, "reduction is wrong"
    assert abs(energies_red.min() - inst["e_min"]) < T4_TOL, "reduced optimum differs from the full optimum"
    scale_red = max(np.abs(J_red).max(), np.abs(h_red).max())
    optimal_red = np.flatnonzero(energies_red <= energies_red.min() + T4_TOL)
    assert np.all(np.isin(optimal_red << 1, inst["optimal_states"])), "a reduced optimum does not map to a full optimum"
    return dict(n=n - 1, energies=energies_red, energy_scaled=(energies_red - const_red) / scale_red, thetas=None, mixer_axis=None,
                optimal_states=optimal_red, near_mask=inst["near_mask"][0::2], balanced_mask=inst["balanced_mask"][0::2],
                e_min=inst["e_min"], e_max=float(energies_red.max()), e_mean=float(energies_red.mean()),
                e_std=float(energies_red.std()), shift=1, J_scaled=J_red / scale_red, h_scaled=h_red / scale_red)


# -------------------------------------------------------------------- metrics ----
def distribution_metrics(probs, view):
    """Exact metrics of an outcome distribution over all states of a view."""
    p_opt, p_near = float(probs[view["optimal_states"]].sum()), float(probs[view["near_mask"]].sum())
    mean_cost = float(probs @ view["energies"])
    spread = view["e_max"] - view["e_min"]
    return dict(p_opt=p_opt, p_near=p_near, mean_cost=mean_cost,
                mean_ratio=float((view["e_max"] - mean_cost) / spread) if spread > 0 else 1.0,
                p_balanced=float(probs[view["balanced_mask"]].sum()),
                hit_opt=float(1.0 - (1.0 - min(p_opt, 1.0)) ** T4_SHOTS), hit_near=float(1.0 - (1.0 - min(p_near, 1.0)) ** T4_SHOTS))


def sampled_metrics(probs, view, seed):
    """The same metrics from T4_SHOTS seeded samples (finite-shot variation). Returns (metrics, sampled FULL-problem states)."""
    samples = np.random.default_rng(seed).choice(len(probs), size=T4_SHOTS, p=probs / probs.sum())
    costs = view["energies"][samples]
    metrics = dict(sampled_p_opt=float(np.isin(samples, view["optimal_states"]).mean()), sampled_p_near=float(view["near_mask"][samples].mean()),
                   sampled_mean_cost=float(costs.mean()), sampled_p_balanced=float(view["balanced_mask"][samples].mean()),
                   sampled_best_cost=float(costs.min()), sampled_hit_opt=bool(costs.min() <= view["e_min"] + T4_TOL),
                   best_state=int(samples[np.argmin(costs)]) << view["shift"], sample_seed=t4_jsonable(seed))
    return metrics, samples << view["shift"]


def modelled_outcome(inst, state):
    """MODELLED runtime and electricity of one split (assumptions, not measurements):
    time = max(share_A / s_A, share_B / s_B) x job hours + communication, communication = comm fraction x job hours x cut fraction;
    each device draws busy power while computing and idle power for the rest of the run (communication counted as idle time)."""
    hours, (speed_a, speed_b) = TASK04["job_reference_hours"], inst["speeds"]
    if "share_b" in inst:                            # enumerated arrays still in memory
        share_b, cut_fraction = float(inst["share_b"][state]), float(inst["cut"][state])
    else:
        cut_fraction, _, _ = direct_cost_parts(1 - 2 * state_index_to_x(state, inst["n"]), inst["model"])
        share_b = float(inst["model"]["a_tilde"] @ state_index_to_x(state, inst["n"]))
    busy_a, busy_b = hours * (1.0 - share_b) / speed_a, hours * share_b / speed_b
    comm = TASK04["comm_fraction_if_all_borders_cut"] * hours * float(cut_fraction)
    total = max(busy_a, busy_b) + comm
    (active_a, idle_a), (active_b, idle_b) = inst["power"]
    energy = (active_a * busy_a + idle_a * (total - busy_a) + active_b * busy_b + idle_b * (total - busy_b)) / 1000.0
    pieces = device_piece_counts(state_index_to_x(state, inst["n"]), inst["regions"]["region_pairs"])
    work = float(inst["model"]["W"])
    return dict(state=int(state), share_b=share_b, load_a=(1.0 - share_b) * work, load_b=share_b * work,
                imbalance=abs(share_b - inst["alpha"]), cut_fraction=float(cut_fraction), pieces_a=int(pieces[0]), pieces_b=int(pieces[1]),
                compute_hours=float(max(busy_a, busy_b)), comm_hours=float(comm), modelled_hours=float(total), modelled_kwh=float(energy))


def state_cost(inst, state):
    """Ising cost of one state: array lookup when enumerated, otherwise from J, h, const."""
    if "energies" in inst:
        return float(inst["energies"][state])
    return float(ising_energy(1 - 2 * state_index_to_x(state, inst["n"]), inst["model"]["J"], inst["model"]["h"], inst["model"]["const"]))


# ----------------------------------------------------------------- optimisers ----
class BudgetExhausted(Exception):
    pass


class CountedObjective:
    """Wraps an objective: counts evaluations, enforces the budget exactly and keeps the traces."""

    def __init__(self, function, budget):
        self.function, self.budget = function, budget
        self.n_evals, self.best_value, self.best_x = 0, np.inf, None
        self.exact_cost_trace = []                 # exact mean cost at every evaluated point (analysis only)

    def __call__(self, angles):
        if self.n_evals >= self.budget:
            raise BudgetExhausted
        value, exact_cost = self.function(np.asarray(angles, float), self.n_evals)
        self.n_evals += 1
        self.exact_cost_trace.append(exact_cost)
        if value < self.best_value:
            self.best_value, self.best_x = value, np.array(angles, float)
        return value


def cvar_of_samples(energies, fraction):
    """CVaR for a minimisation problem (Barkoutsos et al. 2020): mean of the lowest ceil(fraction * K) of K sampled energies."""
    ordered = np.sort(np.asarray(energies, dtype=float))
    return float(ordered[: int(np.ceil(fraction * ordered.size - 1e-9))].mean())


def make_objective(view, kind, run_seed):
    """Objective for the optimisers, normalised by the uniform-sampling mean and spread of the energy.
    'expectation': exact mean energy. 'shot mean': mean of T4_SHOTS sampled energies (what hardware gives).
    'cvar': mean of the LOWEST cvar_fraction of the sampled energies (CVaR, Barkoutsos et al. 2020)."""
    def objective(angles, eval_index):
        probs = view_probabilities(view, angles)
        exact_cost = float(probs @ view["energies"])
        if kind == "expectation":
            value = exact_cost
        else:
            rng = np.random.default_rng([int(run_seed), int(eval_index)])
            sample = view["energies"][rng.choice(len(probs), size=T4_SHOTS, p=probs / probs.sum())]
            value = float(sample.mean()) if kind == "shot mean" else cvar_of_samples(sample, TASK04["cvar_fraction"])
        return (value - view["e_mean"]) / view["e_std"], exact_cost
    return objective


def spsa_minimize(counted, start, budget, seed, tuned):
    """Simultaneous perturbation stochastic approximation (Spall 1998): two noisy evaluations per iteration.
    `tuned` holds the perturbation size and the first-step size chosen on the calibration meshes."""
    settings, rng = dict(TASK04["spsa"], **tuned), np.random.default_rng([int(seed), 99])
    angles = np.array(start, float)

    def gradient_estimate(shift):
        direction = rng.choice([-1.0, 1.0], size=len(angles))
        return (counted(angles + shift * direction) - counted(angles - shift * direction)) / (2.0 * shift) * direction

    iterations = (budget - 2 * settings["calibration_pairs"]) // 2
    stability = settings["stability_fraction"] * iterations
    # Spall's rule for the step size: the first move should be about `first_step`, given the typical gradient size at the start
    typical_gradient = np.mean([np.abs(gradient_estimate(settings["perturbation"])).mean() for _ in range(settings["calibration_pairs"])])
    gain = settings["first_step"] * (stability + 1) ** settings["alpha"] / max(typical_gradient, 1e-12)
    for k in range(iterations):
        step = gain / (stability + k + 1) ** settings["alpha"]
        angles = angles - step * gradient_estimate(settings["perturbation"] / (k + 1) ** settings["gamma"])
    return angles, f"SPSA finished {iterations} iterations after {settings['calibration_pairs']} calibration pairs"


def run_optimizer(name, view, kind, start, run_seed, scenario=None):
    """One optimisation run with the fixed evaluation budget. Returns the counted objective, final angles and stop reason."""
    budget = TASK04["eval_budget"]
    counted = CountedObjective(make_objective(view, kind, run_seed), budget)
    returned = None
    try:
        if name == "SPSA":
            returned, reason = spsa_minimize(counted, start, budget, run_seed, T4_SPSA_SETTINGS[scenario])
        else:
            # stopping tolerances: angles resolved to about 1e-4 rad (objective is normalised to order 1); the budget is the hard cap
            options = {"COBYLA": dict(maxiter=budget, rhobeg=0.1, tol=1e-4),
                       "Nelder-Mead": dict(maxfev=budget, xatol=1e-4, fatol=1e-7),
                       "Powell": dict(maxfev=budget, xtol=1e-4, ftol=1e-7),
                       "L-BFGS-B": dict(maxfun=budget, ftol=1e-10, gtol=1e-6)}[name]
            result = minimize(counted, np.array(start, float), method=name, options=options)
            returned, reason = result.x, str(result.message)
    except BudgetExhausted:
        reason = "evaluation budget reached"
    # exact objective: the best evaluated point. Noisy objective: the optimiser's own final point
    # (or, if the budget stopped it, the point with the best estimate), because exact values are not visible to it.
    final = counted.best_x if (kind == "expectation" or returned is None) else np.asarray(returned, float)
    return counted, final, reason


def evaluate_run(inst, view, angles, sample_seed, extra):
    """Exact and sampled metrics of one set of angles, plus the modelled outcome of its best sampled split."""
    probs = view_probabilities(view, angles)
    assert np.all(np.isfinite(probs)) and abs(probs.sum() - 1.0) < 1e-9, "probabilities are not normalised"
    sampled, full_states = sampled_metrics(probs, view, sample_seed)
    run = dict(distribution_metrics(probs, view), **sampled, **extra)
    run["final_angles"] = [float(a) for a in angles]
    run["outcome"] = modelled_outcome(inst, run["best_state"])                      # the lowest-cost sampled split
    # the sampled split with the best MODELLED runtime (every shot scored with the runtime model)
    run["outcome_runtime_scored"] = modelled_outcome(inst, int(full_states[np.argmin(inst["modelled_hours_all"][full_states])]))
    return run


def aggregate_runs(runs):
    """Summary over the starts of one (variant, optimiser, p): the run from the given start and the spread over random starts."""
    random_runs = [r for r in runs if r["init"] == "random"]
    given_runs = [r for r in runs if r["init"] != "random"]
    summary = {}
    for field in ("p_opt", "p_near", "mean_cost", "mean_ratio", "p_balanced", "hit_opt", "hit_near"):
        if given_runs:
            summary[f"{field}_given_start"] = given_runs[0][field]
        if random_runs:
            values = np.array([r[field] for r in random_runs])
            summary.update({f"{field}_random_mean": float(values.mean()), f"{field}_random_std": float(values.std()),
                            f"{field}_random_min": float(values.min()), f"{field}_random_max": float(values.max())})
    summary["n_runs"] = len(runs)
    return summary


def optimiser_group(inst, view, variant, optimiser, kind, p, lr_start, key_seed, random_starts, start_label="LR-QAOA angles", beta_max=None):
    """All starts of one optimiser on one instance: the given start plus seeded random starts, same budget each."""
    starts = [(start_label, np.array(lr_start, float), None)]
    beta_max = TASK04["random_beta_max"] if beta_max is None else beta_max
    for r in range(random_starts):
        rng = np.random.default_rng([TASK04["start_seed"], key_seed, p, r])
        starts.append(("random", np.concatenate([rng.uniform(0, TASK04["random_gamma_max"], p), rng.uniform(0, beta_max, p)]), r))
    runs = []
    for index, (init, start, init_seed) in enumerate(starts):
        run_seed = int(np.random.SeedSequence([TASK04["start_seed"], key_seed, p, index,
                                               zlib.crc32(f"{variant}|{optimiser}".encode())]).generate_state(1)[0])
        clock = time.perf_counter()
        counted, final, reason = run_optimizer(optimiser, view, kind, start, run_seed, inst["scenario"])
        wall = time.perf_counter() - clock
        assert counted.n_evals <= TASK04["eval_budget"], "evaluation budget exceeded"
        runs.append(evaluate_run(inst, view, final, [run_seed, 7], dict(
            init=init, init_seed=init_seed, run_seed=run_seed, initial_angles=[float(a) for a in start], objective=kind,
            n_evals=counted.n_evals, eval_budget=TASK04["eval_budget"], stopping_reason=reason, wall_seconds=wall,
            history=[round(float(v), 6) for v in np.minimum.accumulate(counted.exact_cost_trace)])))
    return dict(variant=variant, optimiser=optimiser, p=p, n_qubits=view["n"], runs=runs, aggregate=aggregate_runs(runs))


def schedule_group(inst, view, variant, label, p, angles, key_seed):
    """A fixed schedule (no optimiser loop): one run."""
    clock = time.perf_counter()
    run = evaluate_run(inst, view, angles, [TASK04["start_seed"], key_seed, p, 1], dict(
        init="fixed schedule", init_seed=None, run_seed=None, initial_angles=[float(a) for a in angles], objective="none",
        n_evals=0, eval_budget=0, stopping_reason="no optimiser", history=[]))
    run["wall_seconds"] = time.perf_counter() - clock
    return dict(variant=variant, optimiser=label, p=p, n_qubits=view["n"], runs=[run], aggregate=aggregate_runs([run]))


# ------------------------------------------------------- classical baselines ----
def bits_to_state(x):
    return int(np.dot(np.asarray(x, dtype=np.int64), 1 << np.arange(len(x), dtype=np.int64)))


def simulated_annealing(inst, rng, budget):
    """Single-spin-flip Metropolis with a geometric temperature schedule; every proposed flip is one candidate evaluation."""
    model, n = inst["model"], inst["n"]
    J_sym, h = model["J"] + model["J"].T, model["h"]
    z = 1.0 - 2.0 * rng.integers(0, 2, size=n)
    local = J_sym @ z + h
    energy = float(ising_energy(z, model["J"], h, model["const"]))
    best_energy, best_z = energy, z.copy()
    start_temperature = T4_SA_SETTINGS[inst["scenario"]]["start_fraction"] * inst["e_std_formula"]
    temperatures = np.geomspace(start_temperature, T4_SA_SETTINGS[inst["scenario"]]["end_ratio"] * start_temperature, budget - 1)
    for temperature in temperatures:
        i = int(rng.integers(n))
        delta = -2.0 * z[i] * local[i]                         # energy change of flipping spin i
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            z[i] = -z[i]
            local += 2.0 * z[i] * J_sym[:, i]
            energy += delta
            if energy < best_energy:
                best_energy, best_z = energy, z.copy()
    return best_energy, bits_to_state((1 - best_z) // 2)


def local_search(inst, rng, budget):
    """Steepest-descent single flips from random starts, restarted until the evaluation budget is used."""
    model, n = inst["model"], inst["n"]
    J_sym, h = model["J"] + model["J"].T, model["h"]
    used, best_energy, best_z = 0, np.inf, None
    while used < budget:
        z = 1.0 - 2.0 * rng.integers(0, 2, size=n)
        local = J_sym @ z + h
        energy = float(ising_energy(z, model["J"], h, model["const"]))
        used += 1
        while used + n <= budget:
            deltas = -2.0 * z * local                          # one evaluation per neighbour
            used += n
            i = int(np.argmin(deltas))
            if deltas[i] >= -1e-12:
                break
            z[i] = -z[i]
            local += 2.0 * z[i] * J_sym[:, i]
            energy += deltas[i]
        if energy < best_energy:
            best_energy, best_z = energy, z.copy()
    return best_energy, bits_to_state((1 - best_z) // 2)


def spectral_split(inst):
    """Spectral bisection: order regions by the Fiedler vector of the border-cost graph, then take the best of the
    2 (n - 1) threshold cuts under the SAME Ising objective (which already rewards the target work share)."""
    regions, n = inst["regions"], inst["n"]
    weights = np.zeros((n, n))
    weights[regions["region_pairs"][:, 0], regions["region_pairs"][:, 1]] = regions["region_boundary_cost"]
    weights = weights + weights.T
    fiedler = np.linalg.eigh(np.diag(weights.sum(axis=1)) - weights)[1][:, 1]
    order = np.argsort(fiedler, kind="stable")
    best_energy, best_state = np.inf, None
    for cut_position in range(1, n):
        for side in (order[:cut_position], order[cut_position:]):
            x = np.zeros(n, dtype=np.int64)
            x[side] = 1
            state = bits_to_state(x)
            energy = state_cost(inst, state)
            if energy < best_energy - 1e-12:
                best_energy, best_state = energy, state
    return best_energy, best_state, 2 * (n - 1)


def classical_record(inst, method, best_energies, best_states, evaluations, wall, note=""):
    """Summary of a classical method over its repeats, with the modelled outcome of its best split."""
    best_energies = np.asarray(best_energies, float)
    best_index = int(np.argmin(best_energies))
    record = dict(method=method, repeats=len(best_energies), evaluations_per_repeat=int(evaluations), wall_seconds=float(wall), note=note,
                  best_costs=[float(v) for v in best_energies], best_cost=float(best_energies.min()), mean_best_cost=float(best_energies.mean()),
                  best_state=int(best_states[best_index]), outcome=modelled_outcome(inst, int(best_states[best_index])))
    return record


def run_classical(inst):
    """Every classical baseline on one instance, each with the same budget of candidate evaluations."""
    budget, repeats, n = TASK04["classical_budget"], TASK04["classical_repeats"], inst["n"]
    seed_base = [TASK04["classical_seed"], n, inst["seed"], BENCHMARK_CASES.index(inst["case"]), list(TARGET_SHARES).index(inst["scenario"])]
    records = {}
    for method, sampler in (("uniform random sampling", lambda rng: rng.integers(0, 2, size=(budget, n))),
                            ("random with target share", lambda rng: (rng.random((budget, n)) < inst["alpha"]).astype(np.int64))):
        clock, energies, states = time.perf_counter(), [], []
        for repeat in range(repeats):
            candidates = sampler(np.random.default_rng(seed_base + [len(method), repeat]))
            candidate_states = candidates @ (1 << np.arange(n, dtype=np.int64))
            costs = inst["energies"][candidate_states] if "energies" in inst else np.array([state_cost(inst, s) for s in candidate_states])
            energies.append(costs.min())
            states.append(candidate_states[np.argmin(costs)])
        records[method] = classical_record(inst, method, energies, states, budget, time.perf_counter() - clock)
    for method, function in (("simulated annealing", simulated_annealing), ("local search (restarts)", local_search)):
        clock, energies, states = time.perf_counter(), [], []
        for repeat in range(repeats):
            energy, state = function(inst, np.random.default_rng(seed_base + [len(method), repeat]), budget)
            energies.append(energy)
            states.append(state)
        records[method] = classical_record(inst, method, energies, states, budget, time.perf_counter() - clock)
    clock = time.perf_counter()
    energy, state, evaluations = spectral_split(inst)
    records["spectral bisection"] = classical_record(inst, "spectral bisection", [energy], [state], evaluations, time.perf_counter() - clock,
                                                     note="deterministic; best of the 2(n-1) threshold cuts of the Fiedler ordering")
    # reference: exact optimum where enumerated, otherwise the best cost any classical method found (labelled)
    reference = inst["e_min"] if inst["exact"] else min(r["best_cost"] for r in records.values())
    near_limit = TASK04["near_optimal_factor"] * reference if reference > T4_TOL else T4_TOL
    for record in records.values():
        costs = np.array(record["best_costs"])
        record.update(reference_cost=float(reference), reference_kind="exact optimum" if inst["exact"] else "best classical result (not exact)",
                      hit_opt=float(np.mean(costs <= reference + T4_TOL)), hit_near=float(np.mean(costs <= near_limit + T4_TOL)))
    if inst["exact"]:                                # exact per-candidate probabilities of the two sampling baselines
        records["uniform random sampling"]["per_shot"] = distribution_metrics(np.full(2**n, 2.0 ** (-n)), t4_view(inst))
        share_probs = np.ones(1)                     # every region goes to device B with probability alpha, independently
        for _ in range(n):
            share_probs = np.kron(share_probs, np.array([1.0 - inst["alpha"], inst["alpha"]]))
        records["random with target share"]["per_shot"] = distribution_metrics(share_probs, t4_view(inst))
    return records


# ------------------------------------------------- literature (verified) ----
# References are filled in by t4_set_references() after verification; see TASK04_REFERENCES.


# ------------------------------------------------------------ warm starts ----
def relaxation_value_and_gradient(inst, c):
    """Continuous relaxation over c in [0, 1]^n of the same objective (equal to it on binary c), a convex quadratic:
    sum_adj c_ij (c_i - c_j)^2 + 4 lam (sum_i a_i c_i - alpha)^2 + mu sum_i p_i (1 - c_i)."""
    model, alpha, lam, mu = inst["model"], inst["alpha"], inst["lam"], inst["mu"]
    i, j = model["pairs"][:, 0], model["pairs"][:, 1]
    difference, balance = c[i] - c[j], model["a_tilde"] @ c - alpha
    gradient = 8.0 * lam * balance * model["a_tilde"] - mu * model["p_tilde"]
    np.add.at(gradient, i, 2.0 * model["c_tilde"] * difference)
    np.add.at(gradient, j, -2.0 * model["c_tilde"] * difference)
    return float(np.sum(model["c_tilde"] * difference**2) + 4.0 * lam * balance**2 + mu * np.sum(model["p_tilde"] * (1.0 - c))), gradient


def relaxed_solution(inst):
    """Optimum of the relaxation. NOTE: without a priority term, c_i = alpha for every region makes both the cut and the
    balance term zero, so the relaxation optimum carries no information about the split (reported by the cell)."""
    result = minimize(lambda c: relaxation_value_and_gradient(inst, c), np.full(inst["n"], 0.5), jac=True, method="L-BFGS-B",
                      bounds=[(0.0, 1.0)] * inst["n"], options=dict(ftol=1e-15, gtol=1e-12, maxiter=2000))
    return result.x


T4_WARM_FLAVOURS = ("relaxation, mixer A", "rounded spectral, mixer A", "rounded spectral, mixer B")


def warm_start_setup(inst, flavour, spectral_state):
    """Warm-start QAOA after Egger, Marecek, Woerner (Quantum 5, 479, 2021). Returns (thetas, mixer axis angles, c, note).
    theta_i = 2 arcsin(sqrt(c_i)) with c_i clipped into [eps, 1 - eps]; start state prod_i RY(theta_i)|0>.
    - 'relaxation, mixer A': c = optimum of the convex relaxation; mixer of their Eq. (2) (the start state is its ground state).
    - 'rounded spectral, mixer A': c from the spectral split with eps = 0.25, same Eq. (2) mixer.
    - 'rounded spectral, mixer B': their Sec. 2.3 modified mixer. At (beta, gamma) = (pi/2, 0) it returns the COMPLEMENT of the
      start bitstring. That is the same split only when E(z) = E(-z) (no fields). When fields are present we start from the
      complement of the spectral split instead, so the same angles return the spectral split itself: OUR adaptation, not in the paper."""
    x_spectral = state_index_to_x(spectral_state, inst["n"]).astype(float)
    if flavour == "relaxation, mixer A":
        c = np.clip(relaxed_solution(inst), TASK04["warm_start_epsilon_continuous"], 1.0 - TASK04["warm_start_epsilon_continuous"])
        note = "continuous warm start (Egger et al., Sec. 2.2)"
    else:
        eps = TASK04["warm_start_epsilon_rounded"]
        has_fields = bool(np.any(inst["model"]["h"] != 0))
        adapted = flavour.endswith("mixer B") and has_fields
        c = np.clip(1.0 - x_spectral if adapted else x_spectral, eps, 1.0 - eps)
        note = ("rounded warm start with the modified mixer (Egger et al., Sec. 2.3)" + ("; started from the complement because the problem has fields (our adaptation)" if adapted else "")
                if flavour.endswith("mixer B") else "rounded start with the Eq. (2) mixer (as in Qiskit's WarmStartQAOAOptimizer)")
    thetas = 2.0 * np.arcsin(np.sqrt(c))
    return thetas, (-thetas if flavour.endswith("mixer B") else thetas), c, note


# ------------------------------------------------- calibration of LR-QAOA ----
t4_clock_start = time.perf_counter()
calibration_regions = {}
for case_name in TASK04["calibration_cases"]:
    for calibration_seed in TASK04["calibration_seeds"]:
        assert calibration_seed not in RUN["seeds"], "calibration seeds must differ from the scored seeds"
        spec = MESH_CASES[case_name]
        calibration_points, calibration_triangles = build_case_mesh(spec, calibration_seed)
        calibration_mesh = add_case_weights(spec, check_case_mesh(case_name, spec, calibration_points, calibration_triangles))
        regions = coarsen_mesh(calibration_mesh, TASK04["calibration_n"], calibration_seed)
        check_region_case((case_name, TASK04["calibration_n"], calibration_seed), regions, TASK04["calibration_n"])
        calibration_regions[(case_name, calibration_seed)] = regions

lr_calibration, calibration_instances = {}, {}
for scenario in TARGET_SHARES:
    calibration_instances[scenario] = [t4_build_instance(regions, case_name, TASK04["calibration_n"], seed, scenario)
                                       for (case_name, seed), regions in calibration_regions.items()]
    views = [t4_view(inst) for inst in calibration_instances[scenario]]
    grid_ratio = np.zeros((len(TASK04["lr_grid_gamma"]), len(TASK04["lr_grid_beta"])))
    grid_near = np.zeros_like(grid_ratio)
    for gi, delta_gamma in enumerate(TASK04["lr_grid_gamma"]):
        for bi, delta_beta in enumerate(TASK04["lr_grid_beta"]):
            ratios, nears = [], []
            for view in views:
                for p in T4_LAYERS:
                    probs = view_probabilities(view, lr_angles(p, delta_gamma, delta_beta))
                    ratios.append((view["e_max"] - probs @ view["energies"]) / (view["e_max"] - view["e_min"]))
                    nears.append(probs[view["near_mask"]].sum())
            grid_ratio[gi, bi], grid_near[gi, bi] = np.mean(ratios), np.mean(nears)
    by_energy = np.unravel_index(np.argmax(grid_ratio), grid_ratio.shape)          # ties: first = smallest delta_gamma, delta_beta
    by_success = np.unravel_index(np.argmax(grid_near), grid_near.shape)
    lr_calibration[scenario] = dict(
        energy=(float(TASK04["lr_grid_gamma"][by_energy[0]]), float(TASK04["lr_grid_beta"][by_energy[1]])),
        success=(float(TASK04["lr_grid_gamma"][by_success[0]]), float(TASK04["lr_grid_beta"][by_success[1]])),
        energy_score=float(grid_ratio[by_energy]), energy_near=float(grid_near[by_energy]),
        success_score=float(grid_near[by_success]), success_ratio=float(grid_ratio[by_success]),
        grid_ratio=grid_ratio, grid_near=grid_near, n_instances=len(views),
        at_grid_edge=bool(by_energy[0] in (0, len(TASK04["lr_grid_gamma"]) - 1) or by_energy[1] in (0, len(TASK04["lr_grid_beta"]) - 1)))
# three fixed LR schedules: calibrated by mean energy (our choice for unknown optima), calibrated by success probability
# (the LR-QAOA paper selects by the probability of the optimum), and the paper's hardware default without calibration
T4_LR_DELTAS = {scenario: {"LR-QAOA": lr_calibration[scenario]["energy"],
                           "LR-QAOA (success-calibrated)": lr_calibration[scenario]["success"],
                           "LR-QAOA (paper default 0.6/0.3)": tuple(TASK04["lr_paper_default"])} for scenario in TARGET_SHARES}

# SPSA and simulated annealing get the same treatment: their free settings are chosen on the calibration meshes only
spsa_calibration, sa_calibration = {}, {}
for scenario in TARGET_SHARES:
    views, p_cal = [t4_view(inst) for inst in calibration_instances[scenario]], max(T4_LAYERS)
    scores = {}
    for perturbation in TASK04["spsa"]["perturbation_grid"]:
        for first_step in TASK04["spsa"]["first_step_grid"]:
            T4_SPSA_SETTINGS[scenario] = dict(perturbation=perturbation, first_step=first_step)
            ratios = []
            for index, view in enumerate(views):
                _, final, _ = run_optimizer("SPSA", view, "shot mean", lr_angles(p_cal, *lr_calibration[scenario]["energy"]), 7000 + index, scenario)
                ratios.append((view["e_max"] - view_probabilities(view, final) @ view["energies"]) / (view["e_max"] - view["e_min"]))
            scores[(perturbation, first_step)] = float(np.mean(ratios))
    best = max(scores, key=lambda setting: (scores[setting], -setting[0], -setting[1]))
    T4_SPSA_SETTINGS[scenario] = dict(perturbation=best[0], first_step=best[1])
    spsa_calibration[scenario] = dict(perturbation=best[0], first_step=best[1], mean_ratio=scores[best], grid={f"{k[0]}, {k[1]}": v for k, v in scores.items()})
    scores = {}
    for start_fraction in TASK04["sa_start_fraction_grid"]:
        for end_ratio in TASK04["sa_end_ratio_grid"]:
            T4_SA_SETTINGS[scenario] = dict(start_fraction=start_fraction, end_ratio=end_ratio)
            hits, costs = [], []
            for index, inst in enumerate(calibration_instances[scenario]):
                near_limit = TASK04["near_optimal_factor"] * inst["e_min"]
                for repeat in range(TASK04["sa_calibration_repeats"]):
                    energy, _ = simulated_annealing(inst, np.random.default_rng([8000, index, repeat]), TASK04["classical_budget"])
                    hits.append(energy <= near_limit + T4_TOL)
                    costs.append((inst["e_max"] - energy) / (inst["e_max"] - inst["e_min"]))
            scores[(start_fraction, end_ratio)] = (float(np.mean(hits)), float(np.mean(costs)))
    best = max(scores, key=lambda setting: scores[setting])
    T4_SA_SETTINGS[scenario] = dict(start_fraction=best[0], end_ratio=best[1])
    sa_calibration[scenario] = dict(start_fraction=best[0], end_ratio=best[1], hit_near=scores[best][0], grid={f"{k[0]}, {k[1]}": v for k, v in scores.items()})
t4_calibration_seconds = time.perf_counter() - t4_clock_start


# --------------------------------------------------------------- benchmark ----
def t4_key_seed(key):
    return 1000 * BENCHMARK_CASES.index(key[0]) + 100 * list(TARGET_SHARES).index(key[3]) + 10 * key[1] + key[2]


def run_qaoa_groups(inst, key):
    """Every QAOA variant on one instance. Returns {'variant|optimiser|p': group}."""
    view, key_seed, scenario = t4_view(inst), t4_key_seed(key), key[3]
    full_sweep = inst["n"] <= TASK04["full_sweep_max_qubits"]
    uniform = np.full(2 ** inst["n"], 2.0 ** (-inst["n"]))
    groups = []
    control = schedule_group(inst, view, "uniform control", "none", 0, np.array([]), key_seed)
    assert np.allclose(qaoa_probabilities(view["energy_scaled"], [], [], inst["n"]), uniform, atol=1e-15), "p = 0 is not uniform"
    groups.append(control)
    for variant, deltas in T4_LR_DELTAS[scenario].items():
        for p in T4_LAYERS + list(TASK04["stress_layers"]):
            groups.append(schedule_group(inst, view, variant, "fixed schedule", p, lr_angles(p, *deltas), key_seed))
    if np.any(inst["model"]["h"] != 0):              # ablation: cost layer with only the fields, or only the couplings
        x_all = state_index_to_x(np.arange(2 ** inst["n"]), inst["n"])
        fields_scaled = (1.0 - 2.0 * x_all) @ inst["model"]["h"] / inst["model"]["scale"]
        for label, scaled in (("fields only", fields_scaled), ("couplings only", view["energy_scaled"] - fields_scaled)):
            ablation_view = dict(view, energy_scaled=scaled)
            for p in T4_LAYERS:
                groups.append(schedule_group(inst, ablation_view, f"LR-QAOA ablation ({label})", "fixed schedule", p,
                                             lr_angles(p, *T4_LR_DELTAS[scenario]["LR-QAOA"]), key_seed))
    _, spectral_state, _ = spectral_split(inst)
    symmetric = inst["alpha"] == 0.5 and inst["mu"] == 0 and np.allclose(inst["energies"], inst["energies"][::-1], atol=T4_TOL)
    for p in T4_LAYERS:
        lr_start = lr_angles(p, *lr_calibration[scenario]["energy"])
        starts = TASK04["random_starts"] if full_sweep else 0
        for optimiser in (TASK04["optimizers"] if full_sweep else ["COBYLA"]):
            groups.append(optimiser_group(inst, view, "QAOA", optimiser, "shot mean" if optimiser == "SPSA" else "expectation",
                                          p, lr_start, key_seed, starts))
        if not full_sweep:
            continue
        for optimiser in TASK04["cvar_optimizers"]:
            groups.append(optimiser_group(inst, view, "CVaR-QAOA", optimiser, "cvar", p, lr_start, key_seed, starts))
        for flavour in T4_WARM_FLAVOURS:
            thetas, mixer_axis, c_values, note = warm_start_setup(inst, flavour, spectral_state)
            warm_view, variant = t4_view(inst, thetas, mixer_axis), f"warm-start QAOA ({flavour})"
            if p == T4_LAYERS[0]:                    # what the start state alone gives (no layers): the fair control for this variant
                groups.append(schedule_group(inst, warm_view, variant, "initial state only", 0, np.array([]), key_seed))
            if flavour.endswith("mixer B"):          # the paper's recovery angles: beta_1 = pi/2, everything else 0
                given, label, beta_max = np.concatenate([np.zeros(p), [np.pi / 2], np.zeros(p - 1)]), "recovery angles", float(np.pi)
            else:
                given, label, beta_max = lr_start, "LR-QAOA angles", None
            group = schedule_group(inst, warm_view, variant, f"fixed: {label}", p, given, key_seed)
            group.update(warm_start_c=[float(v) for v in c_values], note=note)
            groups.append(group)
            for optimiser in TASK04["warm_start_optimizers"]:
                groups.append(optimiser_group(inst, warm_view, variant, optimiser, "expectation", p, given, key_seed, starts,
                                              start_label=label, beta_max=beta_max))
        if symmetric:
            reduced = t4_reduced_view(inst)
            groups.append(schedule_group(inst, reduced, "symmetry-reduced QAOA", "fixed schedule", p, lr_start, key_seed))
            for optimiser in TASK04["symmetry_optimizers"]:
                groups.append(optimiser_group(inst, reduced, "symmetry-reduced QAOA", optimiser, "expectation", p, lr_start, key_seed, starts))
    for group in groups:                             # what uniform guessing gives in the same space, and the dense gate count
        n_qubits = group["n_qubits"]
        group["uniform_p_opt"] = float(len(inst["optimal_states"]) / 2 ** inst["n"]) if n_qubits == inst["n"] else float(
            np.count_nonzero(inst["optimal_states"] % 2 == 0) / 2**n_qubits)
        near_states = np.flatnonzero(inst["near_mask"])
        group["uniform_p_near"] = float(len(near_states) / 2 ** inst["n"]) if n_qubits == inst["n"] else float(
            np.count_nonzero(near_states % 2 == 0) / 2**n_qubits)
        group["two_qubit_gates"] = int(group["p"] * n_qubits * (n_qubits - 1) // 2)
    return {f"{g['variant']}|{g['optimiser']}|{g['p']}": g for g in groups}


def t4_code_digest(code):
    """Fingerprint of a function's compiled code (bytecode, names, constants, nested functions); line numbers do not matter."""
    parts = [code.co_code, repr(code.co_names).encode(), repr(code.co_varnames).encode()]
    for constant in code.co_consts:
        if hasattr(constant, "co_code"):
            parts.append(t4_code_digest(constant))
        elif isinstance(constant, frozenset):         # set order depends on the per-process string hash seed
            parts.append(repr(sorted(map(repr, constant))).encode())
        else:
            parts.append(repr(constant).encode())
    return hashlib.sha256(b"|".join(parts)).digest()


# the checkpoint is only reused if the settings AND the code of every function defined in this cell are unchanged
t4_cell_file = t4_build_instance.__code__.co_filename
T4_CODE_HASH = hashlib.sha256(b"".join(t4_code_digest(value.__code__) for name, value in sorted(globals().items())
                                       if callable(value) and getattr(getattr(value, "__code__", None), "co_filename", None) == t4_cell_file)).hexdigest()[:16]
t4_cache = {}
if TASK04["use_checkpoint"] and T4_CHECKPOINT.exists():
    with gzip.open(T4_CHECKPOINT, "rt") as handle:
        loaded_checkpoint = json.load(handle)
    if (loaded_checkpoint.get("config_hash") == T4_CONFIG_HASH and loaded_checkpoint.get("code_hash") == T4_CODE_HASH
            and loaded_checkpoint.get("shots") == T4_SHOTS and loaded_checkpoint.get("layers") == T4_LAYERS):
        t4_cache = loaded_checkpoint["instances"]

task04_ising_cases, qaoa_results, classical_results, task04_references = {}, {}, {}, {}
task04_skipped, task04_failures, t4_reused, t4_computed = [], [], 0, 0
t4_instance_keys = [(case_name, n, seed, scenario) for case_name in BENCHMARK_CASES for n in RUN["region_counts"]
                    for seed in RUN["seeds"] for scenario in TARGET_SHARES]
t4_heavy_fields = ("energies", "cut", "share_b", "imbalance", "near_mask", "balanced_mask", "energy_scaled", "modelled_hours_all")
t4_examples = {}                                     # one full instance per scenario, kept for the Qiskit cross-check and tests
for key in t4_instance_keys:
    case_name, n, seed, scenario = key
    inst = t4_build_instance(region_cases[(case_name, n, seed)], case_name, n, seed, scenario)
    cache_key = "|".join(map(str, key))
    wants_qaoa = case_name in RUN["quantum_cases"]
    cached = t4_cache.get(cache_key)
    # reuse only a complete record: same problem data, no recorded failure, and QAOA present if QAOA is wanted now
    if (cached is not None and cached.get("fingerprint") == inst["fingerprint"] and not cached.get("qaoa_error")
            and (cached["qaoa"] or not (wants_qaoa and inst["statevector"]))):
        record = cached
        t4_reused += 1
    else:
        record = dict(fingerprint=inst["fingerprint"], classical=run_classical(inst), qaoa={}, references={})
        if inst["exact"]:
            record["references"] = dict(ising_optimum=modelled_outcome(inst, int(inst["optimal_states"][0])),
                                        runtime_optimum=modelled_outcome(inst, int(np.argmin(inst["modelled_hours_all"]))))
        if wants_qaoa and inst["statevector"]:
            try:
                record["qaoa"] = run_qaoa_groups(inst, key)
            except Exception as error:               # keep going; the failure is recorded, never replaced by made-up values
                task04_failures.append(dict(key=cache_key, error=f"{type(error).__name__}: {error}"))
                record["qaoa_error"] = f"{type(error).__name__}: {error}"   # a failed record is recomputed on the next run
        record = t4_jsonable(record)
        t4_cache[cache_key] = record
        t4_computed += 1
        if TASK04["use_checkpoint"]:
            with gzip.open(T4_CHECKPOINT, "wt") as handle:
                json.dump(dict(config_hash=T4_CONFIG_HASH, code_hash=T4_CODE_HASH, shots=T4_SHOTS, layers=T4_LAYERS, instances=t4_cache), handle)
    if wants_qaoa and not inst["statevector"]:
        task04_skipped.append(f"{cache_key}: QAOA statevector skipped (n = {n} > {LOCAL_STATEVECTOR_MAX_QUBITS}); classical baselines and gate/noise estimates only")
    if not wants_qaoa:
        task04_skipped.append(f"{cache_key}: QAOA not run (case not in RUN['quantum_cases']); classical baselines only")
    if wants_qaoa and inst["statevector"] and n > TASK04["full_sweep_max_qubits"]:
        task04_skipped.append(f"{cache_key}: reduced QAOA sweep (n = {n} > {TASK04['full_sweep_max_qubits']}): LR-QAOA and COBYLA from the LR start only")
    for method, classical in record["classical"].items():
        classical_results[key + (method,)] = classical
    for group in record["qaoa"].values():
        qaoa_results[key + (group["variant"], group["optimiser"], group["p"])] = group
    task04_references[key] = record["references"]
    if scenario not in t4_examples and inst["statevector"] and record["qaoa"] and n == min(RUN["region_counts"]):
        t4_examples[scenario] = inst                 # keeps its arrays
    else:
        for field in t4_heavy_fields:
            inst.pop(field, None)
    task04_ising_cases[key] = inst
t4_benchmark_seconds = time.perf_counter() - t4_clock_start - t4_calibration_seconds


# ------------------------------------------------------- verified literature ----
TASK04_REFERENCES = {
    "LR-QAOA": "J. A. Montanez-Barrera, K. Michielsen, 'Toward a linear-ramp QAOA protocol: evidence of a scaling advantage in solving "
               "some combinatorial optimization problems', npj Quantum Information 11, 131 (2025), doi:10.1038/s41534-025-01082-1, arXiv:2405.09169",
    "CVaR": "P. Kl. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, S. Woerner, 'Improving Variational Quantum Optimization using CVaR', "
            "Quantum 4, 256 (2020), doi:10.22331/q-2020-04-20-256, arXiv:1907.04769",
    "warm-start QAOA": "D. J. Egger, J. Marecek, S. Woerner, 'Warm-starting quantum optimization', Quantum 5, 479 (2021), "
                       "doi:10.22331/q-2021-06-17-479, arXiv:2009.10095",
    "SPSA": "J. C. Spall, 'Implementation of the simultaneous perturbation algorithm for stochastic optimization', "
            "IEEE Trans. Aerospace and Electronic Systems 34(3), 817-823 (1998), doi:10.1109/7.705889",
    "Ising graph partitioning": "A. Lucas, 'Ising formulations of many NP problems', Frontiers in Physics 2:5 (2014), "
                                "doi:10.3389/fphy.2014.00005, arXiv:1302.5843",
    "verification": "citations and formulas were checked against the primary sources on 2026-10-10 before coding; our own choices "
                    "(calibration by mean energy, epsilon = 0.05 for the continuous warm start, complement start for mixer B with fields, "
                    "SPSA first-step size) are marked as ours in the code",
}

# -------------------------------------------------------------------- tests ----
t4_tests = {}
test_rng = np.random.default_rng(TASK04["start_seed"])
# 1-2. gates are unitary, the mixer keeps the norm, and the two code paths (dense blocks / qubit by qubit) agree
for n_test in (11, 12):
    random_state = test_rng.normal(size=2**n_test) + 1j * test_rng.normal(size=2**n_test)
    random_state /= np.linalg.norm(random_state)
    for axis in (np.full(n_test, np.pi / 2), test_rng.uniform(0.2, 2.9, n_test), -test_rng.uniform(0.2, 2.9, n_test)):
        gates = mixer_gates(0.37, axis)
        assert np.allclose(gates @ gates.conj().transpose(0, 2, 1), np.eye(2), atol=1e-12), "mixer gate is not unitary"
        by_blocks, by_qubit = apply_qubit_gates(random_state, gates), apply_qubit_gates(random_state, gates, per_qubit=True)
        assert abs(np.linalg.norm(by_blocks) - 1.0) < 1e-12 and np.allclose(by_blocks, by_qubit, atol=1e-12), "mixer paths disagree"
    assert np.allclose(apply_uniform_x_mixer(random_state, 0.37, n_test), apply_qubit_gates(random_state, mixer_gates(0.37, np.full(n_test, np.pi / 2)), per_qubit=True), atol=1e-12)
t4_tests["mixer unitary, norm kept, three code paths agree"] = "passed (11 and 12 qubits)"
# warm-start identities from Egger et al. (2021)
pauli_x, pauli_z = np.array([[0, 1], [1, 0]], complex), np.array([[1, 0], [0, -1]], complex)
assert np.allclose(mixer_gates(0.37, np.array([np.pi / 2]))[0], np.cos(0.37) * np.eye(2) + 1j * np.sin(0.37) * pauli_x), "standard mixer is not exp(+i beta X)"
for c_value in (0.1, 0.25, 0.6, 0.75, 0.93):
    theta = 2 * np.arcsin(np.sqrt(c_value))
    start = np.array([np.cos(theta / 2), np.sin(theta / 2)])
    assert np.isclose(start[1] ** 2, c_value)
    assert np.allclose(mixer_gates(0.37, np.array([theta]))[0] @ start, np.exp(0.37j) * start), "mixer A: start state is not its eigenstate"
for c_value, expected in ((0.25, 1), (0.75, 0)):       # mixer B, eps = 0.25, beta = pi/2: the start bit is flipped with certainty
    theta = 2 * np.arcsin(np.sqrt(c_value))
    out = mixer_gates(np.pi / 2, np.array([-theta]))[0] @ np.array([np.cos(theta / 2), np.sin(theta / 2)])
    assert np.isclose(abs(out[expected]) ** 2, 1.0), "mixer B recovery identity fails"
t4_tests["warm-start identities (mixer A eigenstate, mixer B recovery at eps = 0.25)"] = "passed"
# CVaR estimator: fraction 1 is the plain mean; fraction 0.1 of 1000 samples is the mean of the 100 lowest
cvar_sample = test_rng.normal(size=1000)
assert np.isclose(cvar_of_samples(cvar_sample, 1.0), cvar_sample.mean()) and np.isclose(cvar_of_samples(cvar_sample, 0.1), np.sort(cvar_sample)[:100].mean())
t4_tests["CVaR estimator"] = "passed"
# round-robin gate order covers every pair exactly once
for n_test in (11, 12, 16):
    listed = [pair for pairs in round_robin_pairs(n_test) for pair in pairs]
    assert len(listed) == len(set(listed)) == n_test * (n_test - 1) // 2
# 7. count keys: the rightmost character is region 0
assert count_key_to_spins("0001")[0].tolist() == [1, 0, 0, 0] and state_index_to_x(1, 4).tolist() == [1, 0, 0, 0]
t4_tests["count-key decoder (rightmost bit = region 0)"] = "passed"

# 3. Qiskit circuit versus the numpy simulator, on one 12-qubit instance per scenario
qiskit_checks = []
for scenario, inst in t4_examples.items():
    view, model, p_check = t4_view(inst), inst["model"], max(T4_LAYERS)
    angles = lr_angles(p_check, *T4_LR_DELTAS[scenario]["LR-QAOA"])
    circuit = build_qaoa_circuit(model["J_scaled"], model["h_scaled"], *split_angles(angles))
    error = float(np.max(np.abs(Statevector(circuit).probabilities() - view_probabilities(view, angles))))
    assert error < 1e-9, f"Qiskit and numpy disagree for {scenario}: {error}"
    _, spectral_state, _ = spectral_split(inst)
    warm_errors = {}
    for flavour in T4_WARM_FLAVOURS:
        thetas, mixer_axis, _, _ = warm_start_setup(inst, flavour, spectral_state)
        warm_circuit = build_qaoa_circuit(model["J_scaled"], model["h_scaled"], *split_angles(angles), thetas=thetas, modified_mixer=flavour.endswith("mixer B"))
        warm_errors[flavour] = float(np.max(np.abs(Statevector(warm_circuit).probabilities() - view_probabilities(t4_view(inst, thetas, mixer_axis), angles))))
        assert warm_errors[flavour] < 1e-9, f"warm-start circuit disagrees ({flavour}, {scenario})"
        if flavour.endswith("mixer B"):              # recovery angles return one classical bitstring with certainty
            recovered = view_probabilities(t4_view(inst, thetas, mixer_axis), np.array([0.0, np.pi / 2]))
            assert np.isclose(recovered.max(), 1.0), "mixer B recovery does not return a single bitstring"
            target = spectral_state if np.any(model["h"] != 0) else (2 ** inst["n"] - 1) ^ spectral_state
            assert int(np.argmax(recovered)) == target, "mixer B recovery returns the wrong bitstring"
    # warm start with c = 0.5 everywhere must be exactly standard QAOA
    half = np.full(inst["n"], np.pi / 2)
    assert np.allclose(view_probabilities(t4_view(inst, half, half), angles), view_probabilities(view, angles), atol=1e-12)
    # mutation test of the sign convention: flipping the mixer sign must make the mean cost WORSE than uniform guessing
    correct_cost = float(view_probabilities(view, angles) @ inst["energies"])
    wrong_cost = float(view_probabilities(view, np.concatenate([angles[:p_check], -angles[p_check:]])) @ inst["energies"])
    assert correct_cost < inst["e_mean"] < wrong_cost, "sign test failed on the example instance"
    operations = circuit.count_ops()
    qiskit_checks.append(dict(scenario=scenario, case=inst["case"], n=inst["n"], p=p_check, max_probability_error=error,
                              warm_start_errors=warm_errors, rzz_gates=int(operations.get("rzz", 0)),
                              expected_rzz=int(p_check * inst["n"] * (inst["n"] - 1) // 2), rzz_per_layer=int(operations.get("rzz", 0) // p_check),
                              depth=int(circuit.depth()), mean_cost=correct_cost, uniform_mean_cost=inst["e_mean"], wrong_sign_mean_cost=wrong_cost))
t4_tests["Qiskit circuit = numpy simulator (standard and three warm-start circuits)"] = (
    "passed, max probability error " + ", ".join(f"{c['max_probability_error']:.1e}" for c in qiskit_checks) if qiskit_checks
    else "NOT RUN: no instance with QAOA results")
# actual gate counts of the Qiskit circuit for every configured size (no simulation needed)
gate_counts = {}
for n in RUN["region_counts"]:
    sample_model = next(inst["model"] for key, inst in task04_ising_cases.items() if key[1] == n)
    for p in T4_LAYERS:
        counted_circuit = build_qaoa_circuit(sample_model["J_scaled"], sample_model["h_scaled"], np.full(p, 0.1), np.full(p, 0.1))
        gate_counts[(n, p)] = dict(rzz=int(counted_circuit.count_ops().get("rzz", 0)), expected=p * n * (n - 1) // 2, depth=int(counted_circuit.depth()))
        assert gate_counts[(n, p)]["rzz"] == gate_counts[(n, p)]["expected"], "dense pair count differs from n(n-1)/2 per layer"

# flat table: one row per run, with the crude noise estimate attached
noise_error = TASK04["noise_error_per_two_qubit_gate"]
task04_table = []
for key, group in qaoa_results.items():
    fidelity = (1.0 - noise_error) ** group["two_qubit_gates"]
    for run in group["runs"]:
        noisy_opt = fidelity * run["p_opt"] + (1.0 - fidelity) * group["uniform_p_opt"]
        noisy_near = fidelity * run["p_near"] + (1.0 - fidelity) * group["uniform_p_near"]
        task04_table.append(dict(
            case=key[0], n=key[1], seed=key[2], scenario=key[3], variant=key[4], optimiser=key[5], p=key[6], n_qubits=group["n_qubits"],
            init=run["init"], init_seed=run["init_seed"], objective=run["objective"], n_evals=run["n_evals"], eval_budget=run["eval_budget"],
            measured_wall_seconds=run["wall_seconds"], p_opt=run["p_opt"], p_near=run["p_near"], mean_cost=run["mean_cost"],
            mean_ratio=run["mean_ratio"], p_balanced=run["p_balanced"], hit_opt=run["hit_opt"], hit_near=run["hit_near"],
            sampled_p_opt=run["sampled_p_opt"], sampled_p_near=run["sampled_p_near"], sampled_best_cost=run["sampled_best_cost"],
            sampled_hit_opt=run["sampled_hit_opt"], uniform_p_opt=group["uniform_p_opt"], uniform_p_near=group["uniform_p_near"],
            two_qubit_gates=group["two_qubit_gates"], noise_fidelity_estimate=fidelity, p_opt_noise_estimate=noisy_opt,
            p_near_noise_estimate=noisy_near, hit_near_noise_estimate=1.0 - (1.0 - noisy_near) ** T4_SHOTS,
            hit_opt_noise_estimate=1.0 - (1.0 - noisy_opt) ** T4_SHOTS,
            modelled_hours=run["outcome"]["modelled_hours"], modelled_kwh=run["outcome"]["modelled_kwh"], share_b=run["outcome"]["share_b"],
            imbalance=run["outcome"]["imbalance"], cut_fraction=run["outcome"]["cut_fraction"], pieces_a=run["outcome"]["pieces_a"],
            pieces_b=run["outcome"]["pieces_b"], modelled_hours_runtime_scored=run["outcome_runtime_scored"]["modelled_hours"],
            stopping_reason=run["stopping_reason"]))

# 4. sign convention on every instance: LR-QAOA must have a lower mean cost than uniform guessing
sign_rows = [row for row in task04_table if row["variant"] == "LR-QAOA" and row["p"] in T4_LAYERS]
uniform_cost = {key[:4]: group["runs"][0]["mean_cost"] for key, group in qaoa_results.items() if key[4] == "uniform control"}
sign_failures = [(row["case"], row["n"], row["seed"], row["scenario"], row["p"]) for row in sign_rows
                 if not row["mean_cost"] < uniform_cost[(row["case"], row["n"], row["seed"], row["scenario"])]]
t4_tests["sign convention: LR-QAOA mean cost below the uniform mean"] = f"{len(sign_rows) - len(sign_failures)}/{len(sign_rows)} instance-layer pairs" + (
    f"; FAILURES: {sign_failures}" if sign_failures else "; mixer-sign flip makes it worse than uniform on the example instances")
# 5-6. p = 0 is uniform, and its exact and sampled numbers are consistent
for key, group in qaoa_results.items():
    if key[4] == "uniform control":
        run = group["runs"][0]
        assert abs(run["p_opt"] - group["uniform_p_opt"]) < 1e-12 and abs(run["p_near"] - group["uniform_p_near"]) < 1e-12
        assert abs(run["sampled_p_near"] - run["p_near"]) <= 5 * np.sqrt(run["p_near"] * (1 - run["p_near"]) / T4_SHOTS) + 1.0 / T4_SHOTS
t4_tests["p = 0 control is uniform; exact and sampled values consistent"] = "passed"
# 8. exact optimum agrees with Task 03 wherever the same lambda and mu were used there
same_objective = [key for key, inst in task04_ising_cases.items() if inst["exact"] and key in ising_cases and ising_cases[key]["brute_force"] == "done"
                  and abs(ising_cases[key]["lam"] - inst["lam"]) < 1e-12 and ising_cases[key]["mu"] == inst["mu"]]
for key in same_objective:
    assert abs(ising_cases[key]["E_min"] - task04_ising_cases[key]["e_min"]) < T4_TOL
    assert np.array_equal(np.sort(ising_cases[key]["optimal_state_indices"]), np.sort(task04_ising_cases[key]["optimal_states"]))
t4_tests["exact optimum and degeneracy equal Task 03 (same lambda)"] = f"{len(same_objective)} instances compared; the others use a different lambda and were re-enumerated"
# 9. symmetry reduction only where the energy is symmetric
reduced_keys = [key for key in qaoa_results if key[4] == "symmetry-reduced QAOA"]
assert all(TARGET_SHARES[key[3]] == 0.5 for key in reduced_keys), "symmetry reduction was applied to a non-symmetric instance"
for scenario, inst in t4_examples.items():
    symmetric_energy = np.allclose(inst["energies"], inst["energies"][::-1], atol=T4_TOL)
    assert symmetric_energy == (inst["alpha"] == 0.5 and inst["mu"] == 0), "unexpected symmetry"
t4_tests["symmetry reduction keeps the optimum and is used only at alpha = 0.5"] = f"passed ({len(reduced_keys)} reduced groups)"
# 10. evaluation budgets
evaluations = {}
for row in task04_table:
    if row["eval_budget"]:
        assert row["n_evals"] <= row["eval_budget"] == TASK04["eval_budget"], "budget violated"
        evaluations.setdefault(row["optimiser"], []).append(row["n_evals"])
t4_tests["same evaluation budget for every optimiser"] = f"budget {TASK04['eval_budget']}; evaluations used (min-max): " + ", ".join(
    f"{name} {min(v)}-{max(v)}" for name, v in evaluations.items())
# 11. every classical baseline is scored with the same Ising objective
for key, record in classical_results.items():
    assert abs(record["best_cost"] - state_cost(task04_ising_cases[key[:4]], record["best_state"])) < T4_TOL, "classical cost mismatch"
t4_tests["classical baselines use the same Ising objective"] = f"passed ({len(classical_results)} records)"
# 13. reproducibility: the same seeds give the same optimiser run and the same annealing run
repro_key = next((key[:4] for key in qaoa_results if key[4:] == ("QAOA", "COBYLA", T4_LAYERS[0]) and key[1] <= TASK04["full_sweep_max_qubits"]), None)
if repro_key is None:
    t4_tests["reproducible with fixed seeds (optimiser run, annealing run)"] = "NOT RUN: no successful optimiser group to repeat"
else:
    repro_inst = t4_build_instance(region_cases[repro_key[:3]], *repro_key)          # rebuilt with its arrays
    repeat_a = optimiser_group(repro_inst, t4_view(repro_inst), "QAOA", "COBYLA", "expectation", T4_LAYERS[0],
                               lr_angles(T4_LAYERS[0], *T4_LR_DELTAS[repro_key[3]]["LR-QAOA"]), t4_key_seed(repro_key), TASK04["random_starts"])
    stored = qaoa_results[repro_key + ("QAOA", "COBYLA", T4_LAYERS[0])]
    assert len(repeat_a["runs"]) == len(stored["runs"]) and all(
        np.allclose(a["final_angles"], b["final_angles"], atol=1e-12) for a, b in zip(repeat_a["runs"], stored["runs"])), "optimiser run not reproducible"
    assert simulated_annealing(repro_inst, np.random.default_rng(11), 200) == simulated_annealing(repro_inst, np.random.default_rng(11), 200)
    t4_tests["reproducible with fixed seeds (optimiser run, annealing run)"] = f"passed ({repro_key[0]}, {SCENARIO_SLUGS.get(repro_key[3], repro_key[3])})"
# ---- unit tests added after mutation testing: each one catches a specific silent error ----
unit_rng = np.random.default_rng(TASK04["start_seed"] + 1)
# LR schedule direction: gamma ramps up, beta ramps down
assert np.allclose(lr_angles(3, 0.9, 0.6), [0.3, 0.6, 0.9, 0.6, 0.4, 0.2]), "LR-QAOA schedule is wrong"
# exact metrics of a hand-made distribution (4 states, energies 3, 1, 2, 1.04; optimum = state 1)
toy_view = dict(energies=np.array([3.0, 1.0, 2.0, 1.04]), optimal_states=np.array([1]), near_mask=np.array([False, True, False, True]),
                balanced_mask=np.array([True, False, True, True]), e_min=1.0, e_max=3.0)
toy = distribution_metrics(np.array([0.1, 0.2, 0.3, 0.4]), toy_view)
assert np.allclose([toy["p_opt"], toy["p_near"], toy["mean_cost"], toy["p_balanced"]], [0.2, 0.6, 1.516, 0.8])
assert np.isclose(toy["mean_ratio"], (3.0 - 1.516) / 2.0)
rare = distribution_metrics(np.array([0.997, 0.001, 0.0, 0.002]), toy_view)     # small probabilities: the shot count matters
assert np.isclose(rare["hit_opt"], 1 - 0.999 ** T4_SHOTS) and np.isclose(rare["hit_near"], 1 - 0.997 ** T4_SHOTS), "hit probability formula is wrong"
# modelled outcome of a hand-made 2-region instance: 300 h job, speeds 1 and 4, state x = (0, 1)
toy_inst = dict(n=2, alpha=0.8, speeds=(1.0, 4.0), power=((100.0, 50.0), (200.0, 20.0)), share_b=np.array([0.0, 0.25, 0.75, 1.0]),
                cut=np.array([0.0, 1.0, 1.0, 0.0]), model=dict(W=4.0), regions=dict(region_pairs=np.array([[0, 1]])))
toy_out = modelled_outcome(dict(toy_inst, **{"exact": True}), 2)
hours = TASK04["job_reference_hours"]
expected_total = max(0.25 * hours, 0.75 * hours / 4) + TASK04["comm_fraction_if_all_borders_cut"] * hours
expected_kwh = (100 * 0.25 * hours + 50 * (expected_total - 0.25 * hours) + 200 * 0.75 * hours / 4 + 20 * (expected_total - 0.75 * hours / 4)) / 1000
assert np.isclose(toy_out["modelled_hours"], expected_total) and np.isclose(toy_out["modelled_kwh"], expected_kwh) and (toy_out["pieces_a"], toy_out["pieces_b"]) == (1, 1)
# SPSA must decrease a simple quadratic
spsa_target = np.array([0.3, -0.2, 0.5, 0.1])
spsa_counted = CountedObjective(lambda x, index: (float(np.sum((x - spsa_target) ** 2)),) * 2, 200)
spsa_final, _ = spsa_minimize(spsa_counted, np.zeros(4), 200, 3, dict(perturbation=0.1, first_step=0.1))
assert np.sum((spsa_final - spsa_target) ** 2) < 0.1 * np.sum(spsa_target ** 2), "SPSA does not minimise"
unit_checked = 0
for scenario, inst in t4_examples.items():
    # masks recomputed independently
    assert np.array_equal(inst["near_mask"], inst["energies"] <= TASK04["near_optimal_factor"] * inst["e_min"] + T4_TOL)
    assert np.array_equal(inst["balanced_mask"], np.abs(inst["share_b"] - inst["alpha"]) <= TASK04["epsilon"])
    # scaled energies agree with J_scaled, h_scaled for the full and (where used) the reduced problem
    views = [(t4_view(inst), inst["model"]["J_scaled"], inst["model"]["h_scaled"])]
    if inst["alpha"] == 0.5 and inst["mu"] == 0:
        reduced = t4_reduced_view(inst)
        views.append((reduced, reduced["J_scaled"], reduced["h_scaled"]))
    for view, J_s, h_s in views:
        for k in unit_rng.integers(0, 2 ** view["n"], 50):
            z = 1 - 2 * state_index_to_x(k, view["n"])
            assert abs(view["energy_scaled"][k] - (z @ J_s @ z + h_s @ z)) < 1e-9, "scaled energies disagree with J_scaled, h_scaled"
    # the relaxation equals the binary objective on binary points, and its gradient is right
    for k in unit_rng.integers(0, 2 ** inst["n"], 20):
        value, _ = relaxation_value_and_gradient(inst, state_index_to_x(k, inst["n"]).astype(float))
        assert abs(value - inst["energies"][k]) < 1e-9, "relaxation differs from the objective on a binary point"
    c_point = unit_rng.uniform(0, 1, inst["n"])
    numeric = np.array([(relaxation_value_and_gradient(inst, c_point + 1e-6 * e)[0] - relaxation_value_and_gradient(inst, c_point - 1e-6 * e)[0]) / 2e-6
                        for e in np.eye(inst["n"])])
    assert np.allclose(numeric, relaxation_value_and_gradient(inst, c_point)[1], atol=1e-6), "relaxation gradient is wrong"
    # outcome without the enumerated arrays (the path used above 22 regions) = outcome with them
    stripped = {k: v for k, v in inst.items() if k not in ("share_b", "cut", "energies")}
    for k in unit_rng.integers(0, 2 ** inst["n"], 10):
        assert np.isclose(modelled_outcome(stripped, int(k))["modelled_hours"], modelled_outcome(inst, int(k))["modelled_hours"])
        assert np.isclose(state_cost(stripped, int(k)), state_cost(inst, int(k)))
    # annealing at almost zero temperature and local search both end in a single-flip local minimum
    J_sym, h_vec = inst["model"]["J"] + inst["model"]["J"].T, inst["model"]["h"]
    def is_local_minimum(state):
        z = 1.0 - 2.0 * state_index_to_x(state, inst["n"])
        return bool(np.all(-2.0 * z * (J_sym @ z + h_vec) >= -1e-12))
    T4_SA_SETTINGS["unit test"] = dict(start_fraction=1e-12, end_ratio=0.5)
    cold = dict(inst, scenario="unit test")
    for repeat in range(3):
        assert is_local_minimum(simulated_annealing(cold, np.random.default_rng([99, repeat]), 600)[1]), "cold annealing did not descend"
        assert is_local_minimum(local_search(inst, np.random.default_rng([98, repeat]), 600)[1]), "local search did not end in a local minimum"
    del T4_SA_SETTINGS["unit test"]
    # spectral bisection returns a threshold cut of the Fiedler ordering
    regions = inst["regions"]
    weights = np.zeros((inst["n"], inst["n"]))
    weights[regions["region_pairs"][:, 0], regions["region_pairs"][:, 1]] = regions["region_boundary_cost"]
    weights += weights.T
    order = np.argsort(np.linalg.eigh(np.diag(weights.sum(axis=1)) - weights)[1][:, 1], kind="stable")
    side_b = set(np.flatnonzero(state_index_to_x(spectral_split(inst)[1], inst["n"])))
    assert any(side_b == set(order[:k]) or side_b == set(order[k:]) for k in range(1, inst["n"])), "spectral split is not a Fiedler threshold cut"
    unit_checked += 1
# pieces per device in the stored outcomes, recounted with an independent union-find
def union_find_pieces(x, pairs):
    parent = list(range(len(x)))
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in pairs:
        if x[i] == x[j]:
            parent[root(i)] = root(j)
    return tuple(len({root(i) for i in range(len(x)) if x[i] == side}) for side in (0, 1))
for key, group in list(qaoa_results.items())[:40]:
    for run in group["runs"]:
        x = state_index_to_x(run["outcome"]["state"], key[1])
        assert union_find_pieces(x, region_cases[key[:3]]["region_pairs"]) == (run["outcome"]["pieces_a"], run["outcome"]["pieces_b"]), "piece count is wrong"
# crude noise estimate recomputed from its definition for every run
for row in task04_table:
    fidelity = (1.0 - noise_error) ** row["two_qubit_gates"]
    assert np.isclose(row["p_near_noise_estimate"], fidelity * row["p_near"] + (1.0 - fidelity) * row["uniform_p_near"])
t4_tests["unit tests added after mutation testing (LR schedule, metrics, outcome model, SPSA, masks, scaled energies, relaxation, "
         "annealing / local search descent, spectral threshold, piece count, noise formula)"] = f"passed ({unit_checked} example instances)"
# 14. no missing or non-finite values
numeric_fields = ("p_opt", "p_near", "mean_cost", "mean_ratio", "p_balanced", "hit_opt", "hit_near", "modelled_hours", "modelled_kwh")
assert all(np.isfinite(row[field]) for row in task04_table for field in numeric_fields), "non-finite value in the results"
assert all(0.0 <= row["p_opt"] <= row["p_near"] + 1e-12 <= 1.0 + 1e-9 for row in task04_table), "probabilities out of range"
t4_tests["no missing or non-finite results"] = f"passed ({len(task04_table)} runs)"


# ----------------------------------------------------------------- analysis ----
T4_N_PLOT, T4_P_MAIN = min(RUN["region_counts"]), max(T4_LAYERS)
T4_QUANTUM_CASES = [case for case in BENCHMARK_CASES if case in RUN["quantum_cases"]]
T4_METHODS = [   # (short label, variant, optimiser)
    ("LR-QAOA", "LR-QAOA", "fixed schedule"),
    ("LR (success-cal.)", "LR-QAOA (success-calibrated)", "fixed schedule"),
    ("LR (0.6/0.3)", "LR-QAOA (paper default 0.6/0.3)", "fixed schedule"),
    ("QAOA COBYLA", "QAOA", "COBYLA"), ("QAOA Nelder-Mead", "QAOA", "Nelder-Mead"), ("QAOA Powell", "QAOA", "Powell"),
    ("QAOA L-BFGS-B", "QAOA", "L-BFGS-B"), ("QAOA SPSA (shots)", "QAOA", "SPSA"),
    ("CVaR COBYLA", "CVaR-QAOA", "COBYLA"), ("CVaR SPSA", "CVaR-QAOA", "SPSA"),
    ("WS relax. A", "warm-start QAOA (relaxation, mixer A)", "COBYLA"),
    ("WS rounded A", "warm-start QAOA (rounded spectral, mixer A)", "COBYLA"),
    ("WS rounded B", "warm-start QAOA (rounded spectral, mixer B)", "COBYLA"),
    ("sym-reduced", "symmetry-reduced QAOA", "COBYLA")]
T4_STYLE = {label: dict(color=color, marker=marker, ls=ls) for (label, _, _), color, marker, ls in zip(
    T4_METHODS, ["#000000", "#555555", "#999999", "#0072B2", "#56B4E9", "#009E73", "#F0E442", "#E69F00", "#D55E00", "#CC79A7", "#0072B2", "#009E73", "#D55E00", "#56B4E9"],
    ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h", "H", "d", "p"], ["-", "--", ":", "-", "-", "-", "-", "-", "-.", "-.", "--", "--", "--", ":"])}


def t4_rows(scenario, variant, optimiser, p, n=None, start="given"):
    """Table rows of one method on the plotted size: the run from the given start ('given'), or the random-start runs."""
    return [row for row in task04_table if row["scenario"] == scenario and row["variant"] == variant and row["optimiser"] == optimiser
            and row["p"] == p and row["n"] == (T4_N_PLOT if n is None else n) and (row["init"] == "random") == (start == "random")]


def t4_mean(rows, field):
    return float(np.mean([row[field] for row in rows])) if rows else float("nan")


def t4_classical(scenario, method, field, n=None):
    """Per-instance value of a classical method. The two pure sampling baselines use their exact per-shot distribution
    where it was enumerated (no sampling noise); annealing and local search use the share of their seeded repeats."""
    return [record["per_shot"][field] if (field in ("hit_opt", "hit_near") and "per_shot" in record) else record[field]
            for key, record in classical_results.items()
            if key[3] == scenario and key[4] == method and key[1] == (T4_N_PLOT if n is None else n) and key[0] in T4_QUANTUM_CASES]


t4_plot_paths = []
T4_NOTE = "noiseless local simulation, exact enumeration as reference; not an IonQ run"

# Plot 1: probability of an optimum per shot versus the number of layers
fig, axes = plt.subplots(1, len(TARGET_SHARES), figsize=(8.5 * len(TARGET_SHARES), 6.2), constrained_layout=True, sharey=True)
for ax, scenario in zip(np.atleast_1d(axes), TARGET_SHARES):
    control = t4_mean(t4_rows(scenario, "uniform control", "none", 0), "p_opt")
    ax.axhline(control, color="black", ls=(0, (1, 1)), lw=2, label=f"uniform guessing, p = 0 ({control:.1e})")
    for label, variant, optimiser in T4_METHODS:
        layers = sorted({row["p"] for row in task04_table if row["scenario"] == scenario and row["variant"] == variant and row["optimiser"] == optimiser and row["p"] > 0})
        values = [t4_mean(t4_rows(scenario, variant, optimiser, p), "p_opt") for p in layers]
        if layers:
            ax.plot(layers, np.maximum(values, 1e-6), lw=1.4, ms=7, label=label, **T4_STYLE[label])
    ax.set_yscale("log")
    ax.set_xticks(sorted(set(T4_LAYERS + list(TASK04["stress_layers"]))))
    ax.set_xlabel("QAOA layers p")
    ax.set_title(f"{scenario} (alpha = {TARGET_SHARES[scenario]})")
    ax.grid(alpha=0.3, which="both")
np.atleast_1d(axes)[0].set_ylabel("probability of an exact optimum per shot (mean over cases)")
np.atleast_1d(axes)[-1].legend(fontsize=7.5, loc="lower right", ncol=2)
fig.suptitle(f"Probability of an optimum per shot, n = {T4_N_PLOT} regions; optimisers started from the LR-QAOA angles "
             f"(WS rounded B: from its recovery angles, which simply return the classical spectral split)\n{T4_NOTE}", fontsize=10)
t4_plot_paths.append(T4_RESULTS_DIR / f"task04_optimum_probability_{RUN_MODE}.png")
fig.savefig(t4_plot_paths[-1], dpi=110)
plt.show()

# Plot 2: optimiser convergence on the example instances
fig, axes = plt.subplots(1, len(t4_examples), figsize=(8.5 * len(t4_examples), 5.6), constrained_layout=True)
for ax, (scenario, inst) in zip(np.atleast_1d(axes), t4_examples.items()):
    example_key = (inst["case"], inst["n"], inst["seed"], scenario)
    for label, variant, optimiser in T4_METHODS:
        group = qaoa_results.get(example_key + (variant, optimiser, T4_P_MAIN))
        if variant != "QAOA" or group is None:
            continue
        for run in group["runs"]:
            given = run["init"] != "random"
            ax.plot(np.arange(1, len(run["history"]) + 1), run["history"], color=T4_STYLE[label]["color"], lw=2.0 if given else 0.9,
                    ls="-" if given else "--", label=f"{optimiser} (LR start)" if given else (f"{optimiser} (random starts)" if run["init_seed"] == 0 else None))
    ax.axhline(inst["e_min"], color="black", ls=":", lw=1.5, label="exact optimum cost")
    ax.axhline(inst["e_mean"], color="grey", ls=":", lw=1.5, label="mean cost of uniform guessing")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("best exact mean cost among evaluated angles")
    ax.set_yscale("log")
    ax.set_title(f"{scenario}: case {inst['case']}, n = {inst['n']}, p = {T4_P_MAIN}")
    ax.grid(alpha=0.3, which="both")
np.atleast_1d(axes)[-1].legend(fontsize=7.5, ncol=2)
fig.suptitle(f"Optimiser convergence, budget {TASK04['eval_budget']} evaluations (solid: LR-QAOA start, dashed: random starts; "
             f"for SPSA the exact cost is recorded for analysis only)\n{T4_NOTE}", fontsize=10)
t4_plot_paths.append(T4_RESULTS_DIR / f"task04_convergence_{RUN_MODE}.png")
fig.savefig(t4_plot_paths[-1], dpi=110)
plt.show()

# Plot 3: difficulty heat map, case x method, P(cost <= 1.05 x optimum) per shot
from matplotlib.colors import LogNorm
heat_columns = [("uniform", "uniform control", "none", 0)] + [(label, variant, optimiser, T4_P_MAIN) for label, variant, optimiser in T4_METHODS]
fig, axes = plt.subplots(1, len(TARGET_SHARES), figsize=(9.5 * len(TARGET_SHARES), 0.55 * len(T4_QUANTUM_CASES) + 3.2), constrained_layout=True)
heat_values = {}
for ax, scenario in zip(np.atleast_1d(axes), TARGET_SHARES):
    grid = np.full((len(T4_QUANTUM_CASES), len(heat_columns)), np.nan)
    for row_index, case in enumerate(T4_QUANTUM_CASES):
        for column_index, (_, variant, optimiser, p) in enumerate(heat_columns):
            rows = [row for row in t4_rows(scenario, variant, optimiser, p) if row["case"] == case]
            grid[row_index, column_index] = t4_mean(rows, "p_near")
    heat_values[scenario] = grid
    colour_map = plt.get_cmap("viridis").copy()
    colour_map.set_bad("#DDDDDD")                     # not applicable (e.g. no symmetry reduction at alpha = 0.8)
    image = ax.imshow(np.ma.masked_invalid(np.maximum(grid, 1e-4)), cmap=colour_map, norm=LogNorm(vmin=1e-4, vmax=1.0), aspect="auto")
    for (row_index, column_index), value in np.ndenumerate(grid):
        text = "n/a" if np.isnan(value) else ("<1e-4" if value < 1e-4 else f"{value:.2g}")
        ax.text(column_index, row_index, text, ha="center", va="center", fontsize=7,
                color="black" if (np.isnan(value) or value >= 0.03) else "white")
    ax.set_xticks(range(len(heat_columns)))
    ax.set_xticklabels([column[0] for column in heat_columns], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(T4_QUANTUM_CASES)))
    ax.set_yticklabels(T4_QUANTUM_CASES, fontsize=8)
    ax.set_title(f"{scenario} (alpha = {TARGET_SHARES[scenario]})", fontsize=10)
fig.colorbar(image, ax=np.atleast_1d(axes).tolist(), label="P(cost <= 1.05 x optimum) per shot (fixed log scale)", shrink=0.9)
fig.suptitle(f"Where QAOA struggles: P(cost <= 1.05 x optimum) per shot, n = {T4_N_PLOT}, p = {T4_P_MAIN} (uniform: p = 0); "
             f"'WS rounded' columns inherit information from the classical spectral split\n{T4_NOTE}", fontsize=10)
t4_plot_paths.append(T4_RESULTS_DIR / f"task04_difficulty_heatmap_{RUN_MODE}.png")
fig.savefig(t4_plot_paths[-1], dpi=110)
plt.show()

# Plot 4: QAOA versus classical methods with the same budget of candidate splits
comparison = [("uniform random", "classical", "uniform random sampling"), ("random, target share", "classical", "random with target share"),
              ("local search", "classical", "local search (restarts)"), ("simulated annealing", "classical", "simulated annealing"),
              ("spectral bisection", "classical", "spectral bisection")]
comparison += [(f"LR-QAOA p={p}", "quantum", ("LR-QAOA", "fixed schedule", p)) for p in T4_LAYERS]
comparison += [(f"QAOA COBYLA p={T4_P_MAIN}", "quantum", ("QAOA", "COBYLA", T4_P_MAIN)), (f"CVaR COBYLA p={T4_P_MAIN}", "quantum", ("CVaR-QAOA", "COBYLA", T4_P_MAIN)),
               (f"WS rounded A p={T4_P_MAIN}", "quantum", ("warm-start QAOA (rounded spectral, mixer A)", "COBYLA", T4_P_MAIN)),
               (f"LR-QAOA p={T4_P_MAIN} + noise est.", "noise", ("LR-QAOA", "fixed schedule", T4_P_MAIN))]
comparison_values = {}
fig, axes = plt.subplots(2, len(TARGET_SHARES), figsize=(8.5 * len(TARGET_SHARES), 9), constrained_layout=True, sharex=True)
axes = np.asarray(axes).reshape(2, -1)
for column, scenario in enumerate(TARGET_SHARES):
    for row_index, (field, title) in enumerate((("hit_opt", "contains an exact optimum"), ("hit_near", "contains a split with cost <= 1.05 x optimum"))):
        ax = axes[row_index, column]
        for position, (label, kind, spec) in enumerate(comparison):
            if kind == "classical":
                values = t4_classical(scenario, spec, field)
            else:
                rows = t4_rows(scenario, *spec)
                values = [row[f"{field}_noise_estimate"] if kind == "noise" else row[field] for row in rows]
            if not values:
                continue
            mean = float(np.mean(values))
            comparison_values[(scenario, field, label)] = (mean, float(np.min(values)), float(np.max(values)))
            ax.barh(position, mean, color={"classical": "#56B4E9", "quantum": "#E69F00", "noise": "#F0E442"}[kind], edgecolor="black",
                    hatch={"classical": "", "quantum": "///", "noise": "xx"}[kind],
                    xerr=[[mean - np.min(values)], [np.max(values) - mean]], capsize=3)
            ax.text(min(mean + 0.02, 0.9), position, f"{mean:.2f}", va="center", fontsize=8)
        ax.set_yticks(range(len(comparison)))
        ax.set_yticklabels([label for label, _, _ in comparison], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_title(f"{scenario}: {T4_SHOTS} candidates {title}", fontsize=9.5)
        ax.grid(axis="x", alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("probability (mean over cases, whiskers: min and max)")
fig.suptitle(f"Same objective, same budget of {T4_SHOTS} candidate splits, n = {T4_N_PLOT}: blue = classical, striped orange = QAOA (noiseless), "
             f"crossed yellow = QAOA with the crude noise estimate\n{T4_NOTE}", fontsize=10)
t4_plot_paths.append(T4_RESULTS_DIR / f"task04_quantum_vs_classical_{RUN_MODE}.png")
fig.savefig(t4_plot_paths[-1], dpi=110)
plt.show()

# Plot 5: crude analytical noise estimate
fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
sizes = np.arange(8, IONQ_SIMULATOR_MAX_QUBITS + 1)
for index, p in enumerate(sorted(set(T4_LAYERS + list(TASK04["stress_layers"])))):
    axes[0].plot(sizes, (1 - noise_error) ** (p * sizes * (sizes - 1) / 2), color=T4_STYLE[T4_METHODS[index][0]]["color"],
                 marker=T4_STYLE[T4_METHODS[index][0]]["marker"], ms=4, lw=1.4, label=f"p = {p}")
for n in RUN["region_counts"]:
    axes[0].axvline(n, color="grey", ls=":", lw=1)
axes[0].set_xlabel("regions = qubits n (dotted lines: sizes in this run)")
axes[0].set_ylabel("estimated share of error-free shots  F = (1 - e)^(p n (n - 1) / 2)")
axes[0].set_title(f"Assumed error per two-qubit gate e = {noise_error}")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
noise_curves = {}
for scenario, colour in zip(TARGET_SHARES, ("#0072B2", "#D55E00")):
    for n, style in zip([n for n in RUN["region_counts"] if n <= LOCAL_STATEVECTOR_MAX_QUBITS], ("o", "s", "^", "D")):
        layers = sorted(set(T4_LAYERS + list(TASK04["stress_layers"])))
        ideal = [t4_mean(t4_rows(scenario, "LR-QAOA", "fixed schedule", p, n=n), "p_near") for p in layers]
        noisy = [t4_mean(t4_rows(scenario, "LR-QAOA", "fixed schedule", p, n=n), "p_near_noise_estimate") for p in layers]
        uniform = t4_mean(t4_rows(scenario, "uniform control", "none", 0, n=n), "p_near")
        noise_curves[(scenario, n)] = dict(layers=layers, ideal=ideal, noise_estimate=noisy, uniform=uniform)
        if not np.all(np.isnan(ideal)):
            axes[1].plot(layers, ideal, color=colour, marker=style, lw=1.6, label=f"{scenario}, n = {n}: noiseless")
            axes[1].plot(layers, noisy, color=colour, marker=style, lw=1.6, ls="--", mfc="white", label=f"{scenario}, n = {n}: with noise estimate")
            axes[1].axhline(uniform, color=colour, ls=":", lw=1)
axes[1].set_yscale("log")
axes[1].set_xticks(sorted(set(T4_LAYERS + list(TASK04["stress_layers"]))))
axes[1].set_xlabel("QAOA layers p (LR-QAOA)")
axes[1].set_ylabel("P(cost <= 1.05 x optimum) per shot (dotted: uniform guessing)")
axes[1].set_title("More layers help without noise but cost fidelity")
axes[1].legend(fontsize=7.5)
axes[1].grid(alpha=0.3, which="both")
fig.suptitle("ANALYTICAL NOISE ESTIMATE (global depolarising model, assumed gate error) - not a result of IonQ hardware or of its noise-model simulator", fontsize=10)
t4_plot_paths.append(T4_RESULTS_DIR / f"task04_noise_estimate_{RUN_MODE}.png")
fig.savefig(t4_plot_paths[-1], dpi=110)
plt.show()

# Plot 6: modelled runtime and electricity of the best split of each method, relative to the runtime-optimal split
outcome_methods = [("Ising optimum (exact)", "reference", "ising_optimum"), ("uniform random", "classical", "uniform random sampling"),
                   ("local search", "classical", "local search (restarts)"), ("simulated annealing", "classical", "simulated annealing"),
                   ("spectral bisection", "classical", "spectral bisection"),
                   (f"LR-QAOA p={T4_P_MAIN}: lowest-cost shot", "quantum", ("LR-QAOA", "fixed schedule", T4_P_MAIN, "modelled_hours", "modelled_kwh")),
                   (f"LR-QAOA p={T4_P_MAIN}: runtime-scored shot", "quantum", ("LR-QAOA", "fixed schedule", T4_P_MAIN, "modelled_hours_runtime_scored", None)),
                   (f"QAOA COBYLA p={T4_P_MAIN}: lowest-cost shot", "quantum", ("QAOA", "COBYLA", T4_P_MAIN, "modelled_hours", "modelled_kwh"))]
outcome_values = {}
fig, axes = plt.subplots(2, len(TARGET_SHARES), figsize=(8.5 * len(TARGET_SHARES), 8), constrained_layout=True, sharex="col")
axes = np.asarray(axes).reshape(2, -1)
for column, scenario in enumerate(TARGET_SHARES):
    keys = [key for key in task04_references if key[3] == scenario and key[1] == T4_N_PLOT and key[0] in T4_QUANTUM_CASES and task04_references[key]]
    for row_index, quantity in enumerate(("modelled_hours", "modelled_kwh")):
        ax = axes[row_index, column]
        for position, (label, kind, spec) in enumerate(outcome_methods):
            excess = []
            for key in keys:
                base = task04_references[key]["runtime_optimum"][quantity]
                if kind == "reference":
                    value = task04_references[key][spec][quantity]
                elif kind == "classical":
                    value = classical_results[key + (spec,)]["outcome"][quantity]
                else:
                    field = spec[3] if quantity == "modelled_hours" else spec[4]
                    rows = [row for row in t4_rows(scenario, spec[0], spec[1], spec[2]) if (row["case"], row["seed"]) == (key[0], key[2])]
                    value = rows[0][field] if rows and field else None
                if value is not None:
                    excess.append(100.0 * (value / base - 1.0))
            if not excess:
                continue
            mean = float(np.mean(excess))
            outcome_values[(scenario, quantity, label)] = (mean, float(np.min(excess)), float(np.max(excess)))
            ax.barh(position, mean, color={"reference": "#999999", "classical": "#56B4E9", "quantum": "#E69F00"}[kind], edgecolor="black",
                    hatch={"reference": "..", "classical": "", "quantum": "///"}[kind], xerr=[[mean - np.min(excess)], [np.max(excess) - mean]], capsize=3)
            ax.text(np.max(excess) + 0.3, position, f"{mean:+.1f} %", va="center", fontsize=8)
        ax.set_yticks(range(len(outcome_methods)))
        ax.set_yticklabels([label for label, _, _ in outcome_methods], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{scenario}: MODELLED {'runtime' if quantity == 'modelled_hours' else 'electricity'} of each method's best split", fontsize=9.5)
        ax.grid(axis="x", alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("difference to the split with the best modelled runtime, in % (mean over cases; whiskers: min, max)")
fig.suptitle(f"MODELLED outcome (assumptions: job = {TASK04['job_reference_hours']:g} h on a speed-1 device, communication = "
             f"{TASK04['comm_fraction_if_all_borders_cut']:.0%} of that if every border is cut, pool-average power) - no HPC measurement", fontsize=10)
t4_plot_paths.append(T4_RESULTS_DIR / f"task04_modelled_outcome_{RUN_MODE}.png")
fig.savefig(t4_plot_paths[-1], dpi=110)
plt.show()
for path in t4_plot_paths:
    assert path.exists() and path.stat().st_size < 2_000_000, f"{path} is missing or over 2 MB"

# ------------------------------------------------------------- result files ----
t4_method_summary = []
for scenario in TARGET_SHARES:
    controls = [("uniform control", "uniform control", "none")] + [(f"{label}: start only", variant, "initial state only")
                                                                for label, variant, _ in T4_METHODS if variant.startswith("warm-start")]
    ablations = [("ablation: fields only", "LR-QAOA ablation (fields only)", "fixed schedule"),
                 ("ablation: couplings only", "LR-QAOA ablation (couplings only)", "fixed schedule")]
    for label, variant, optimiser in controls + T4_METHODS + ablations:
        for p in sorted({row["p"] for row in task04_table if row["scenario"] == scenario and row["variant"] == variant and row["optimiser"] == optimiser}):
            given, randoms = t4_rows(scenario, variant, optimiser, p), t4_rows(scenario, variant, optimiser, p, start="random")
            if not given:
                continue
            share_sampler = [classical_results[(row["case"], row["n"], row["seed"], row["scenario"], "random with target share")]["per_shot"]["p_near"]
                             for row in given]
            t4_method_summary.append(dict(
                scenario=scenario, method=label, variant=variant, optimiser=optimiser, p=p, n=T4_N_PLOT, instances=len(given),
                p_near_target_share_sampler=float(np.mean(share_sampler)),
                p_opt=t4_mean(given, "p_opt"), p_near=t4_mean(given, "p_near"), mean_ratio=t4_mean(given, "mean_ratio"),
                p_balanced=t4_mean(given, "p_balanced"), hit_opt=t4_mean(given, "hit_opt"), hit_near=t4_mean(given, "hit_near"),
                p_near_noise_estimate=t4_mean(given, "p_near_noise_estimate"), evaluations=t4_mean(given, "n_evals"),
                measured_wall_seconds=t4_mean(given, "measured_wall_seconds"),
                p_near_random_starts_mean=t4_mean(randoms, "p_near"), p_near_random_starts_std=float(np.std([r["p_near"] for r in randoms])) if randoms else float("nan"),
                instances_below_uniform=int(sum(row["p_near"] < row["uniform_p_near"] for row in given))))
t4_csv_path = T4_RESULTS_DIR / f"task04_runs_{RUN_MODE}.csv"
with open(t4_csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(task04_table[0].keys()))
    writer.writeheader()
    writer.writerows(task04_table)
t4_total_seconds = time.perf_counter() - t4_clock_start
t4_summary_path = T4_RESULTS_DIR / f"task04_summary_{RUN_MODE}.json"
t4_summary_path.write_text(json.dumps(t4_jsonable(dict(
    task="Task 04: local noiseless QAOA benchmark", run_mode=RUN_MODE, run_settings={k: RUN[k] for k in ("region_counts", "seeds", "qaoa_layers", "shots", "quantum_cases")},
    config={k: v for k, v in TASK04.items() if not k.startswith("lr_grid")}, config_hash=T4_CONFIG_HASH,
    assumptions=dict(
        noise="global depolarising estimate with an assumed error per two-qubit gate; not an IonQ result",
        runtime="modelled: time = max(share_A/s_A, share_B/s_B) x job hours + comm fraction x job hours x cut fraction",
        electricity="modelled: busy power while computing, idle power for the rest of the run (communication counted as idle time); "
                    "power of a scenario device = average of the synthetic pool devices of that kind",
        power_watts={scenario: dict(zip(inst["device_names"], inst["power"])) for scenario, inst in t4_examples.items()},
        wall_clock="measured on the local computer; everything called runtime or electricity is modelled"),
    references=TASK04_REFERENCES,
    lr_calibration={scenario: {k: v for k, v in c.items() if not k.startswith("grid")} for scenario, c in lr_calibration.items()},
    lr_calibration_instances=[f"{case}, n = {TASK04['calibration_n']}, seed {seed}" for case, seed in calibration_regions],
    spsa_calibration=spsa_calibration, simulated_annealing_calibration=sa_calibration,
    qiskit_checks=qiskit_checks, gate_counts={f"n={n}, p={p}": v for (n, p), v in gate_counts.items()}, tests=t4_tests,
    method_summary=t4_method_summary,
    quantum_vs_classical={f"{s} | {f} | {l}": v for (s, f, l), v in comparison_values.items()},
    modelled_outcome_excess_percent={f"{s} | {q} | {l}": v for (s, q, l), v in outcome_values.items()},
    noise_estimate={f"{s}, n = {n}": v for (s, n), v in noise_curves.items()},
    skipped=task04_skipped, failures=task04_failures,
    workload=dict(instances=len(t4_instance_keys), qaoa_groups=len(qaoa_results), qaoa_runs=len(task04_table), classical_records=len(classical_results),
                  objective_evaluations=int(sum(row["n_evals"] for row in task04_table)), instances_computed=t4_computed,
                  instances_from_checkpoint=t4_reused, measured_wall_seconds=t4_total_seconds))), indent=1))
assert t4_summary_path.stat().st_size < 2_000_000 and t4_csv_path.stat().st_size < 2_000_000, "a result file is over 2 MB"


# ------------------------------------------------------------------ summary ----
def t4_find(scenario, label, p):
    return next((m for m in t4_method_summary if m["scenario"] == scenario and m["method"] == label and m["p"] == p), None)


print(f"TASK 04 RESULT: local noiseless QAOA benchmark (mode '{RUN_MODE}'): {len(t4_instance_keys)} instances, {len(qaoa_results)} QAOA groups, "
      f"{len(task04_table)} QAOA runs, {int(sum(row['n_evals'] for row in task04_table))} objective evaluations, {len(classical_results)} classical records")
if task04_failures:
    print(f"  WARNING: {len(task04_failures)} instance(s) FAILED and are missing from every table below: "
          + "; ".join(f"{failure['key']} ({failure['error']})" for failure in task04_failures))
print(f"  measured wall-clock: {t4_total_seconds:.0f} s in this run ({t4_computed} instances computed, {t4_reused} loaded from the checkpoint {T4_CHECKPOINT}); "
      f"calibration {t4_calibration_seconds:.0f} s")
print(f"  objective: Task-04-local instances with lambda = " + ", ".join(f"{v:g} ({k})" for k, v in TASK04["lambda_by_scenario"].items())
      + f"; mu = {TASK04['mu']:g}" + (" -> the priority penalty is INACTIVE, so case 9 equals case 8 here" if TASK04["mu"] == 0 else ""))
print(f"  LR-QAOA angles, chosen once per scenario on {len(calibration_regions)} separate calibration meshes "
      f"({', '.join(f'{case} seed {seed}' for case, seed in calibration_regions)}; layers {T4_LAYERS}):")
for scenario, calibration in lr_calibration.items():
    print(f"    {scenario}: by mean energy (delta_gamma, delta_beta) = ({calibration['energy'][0]:.3f}, {calibration['energy'][1]:.3f}); "
          f"by success probability ({calibration['success'][0]:.3f}, {calibration['success'][1]:.3f}); paper default (0.6, 0.3)"
          + ("; NOTE: the energy optimum lies on the edge of the search grid" if calibration["at_grid_edge"] else ""))
print("  SPSA and simulated annealing settings, chosen on the same calibration meshes: " + "; ".join(
    f"{SCENARIO_SLUGS.get(s, s)}: SPSA perturbation {spsa_calibration[s]['perturbation']}, first step {spsa_calibration[s]['first_step']}; "
    f"annealing start temperature {sa_calibration[s]['start_fraction']} x energy spread, end/start {sa_calibration[s]['end_ratio']}" for s in TARGET_SHARES))
print("  Qiskit circuit versus numpy simulator:")
for check in qiskit_checks:
    print(f"    {check['scenario']}, case {check['case']}, {check['n']} qubits, p = {check['p']}: max probability difference {check['max_probability_error']:.1e}; "
          f"rzz gates {check['rzz_gates']} (expected {check['expected_rzz']}, {check['rzz_per_layer']} per layer), depth {check['depth']}; "
          f"mean cost {check['mean_cost']:.3f} vs uniform {check['uniform_mean_cost']:.3f} vs wrong mixer sign {check['wrong_sign_mean_cost']:.3f}")
print("  gate counts of the circuit: " + "; ".join(f"n = {n}, p = {p}: {v['rzz']} rzz, depth {v['depth']}" for (n, p), v in gate_counts.items()))
print("  tests:")
for name, result in t4_tests.items():
    print(f"    {name}: {result}")

print(f"  QAOA methods at n = {T4_N_PLOT} (mean over {len(T4_QUANTUM_CASES)} cases; per shot unless stated; optimisers from their given start; noiseless):")
print(f"    {'scenario':<9}{'method':<25}{'p':>2}{'P(opt)':>9}{'P(<=1.05)':>10}{'x unif.':>8}{'x share':>8}{'mean r':>8}{'P(bal.)':>8}{'hit 1.05':>9}"
      f"{'noise est.':>11}{'rand.starts':>16}{'evals':>6}{'below unif.':>12}")
print("    ('x share' = ratio to the classical sampler that puts each region on device B with probability alpha, exact per-shot value)")
for scenario in TARGET_SHARES:
    uniform = t4_find(scenario, "uniform control", 0)
    for m in [m for m in t4_method_summary if m["scenario"] == scenario and (m["p"] in T4_LAYERS or m["p"] == 0)]:
        randoms = f"{m['p_near_random_starts_mean']:.4f}+-{m['p_near_random_starts_std']:.4f}" if not np.isnan(m["p_near_random_starts_mean"]) else "-"
        print(f"    {SCENARIO_SLUGS.get(scenario, scenario):<9}{m['method']:<25}{m['p']:>2}{m['p_opt']:>9.4f}{m['p_near']:>10.4f}{m['p_near'] / uniform['p_near']:>8.1f}"
              f"{m['p_near'] / m['p_near_target_share_sampler']:>8.1f}"
              f"{m['mean_ratio']:>8.4f}{m['p_balanced']:>8.3f}{m['hit_near']:>9.3f}{m['p_near_noise_estimate']:>11.4f}{randoms:>16}{m['evaluations']:>6.0f}"
              f"{m['instances_below_uniform']:>9}/{m['instances']}")
print(f"  depth stress test (LR-QAOA, n = {T4_N_PLOT}), P(cost <= 1.05 x optimum) per shot noiseless -> with the crude noise estimate (e = {noise_error}):")
for (scenario, n), curve in noise_curves.items():
    if not np.all(np.isnan(curve["ideal"])):
        print(f"    {SCENARIO_SLUGS.get(scenario, scenario)}, n = {n}: uniform {curve['uniform']:.4f}; " + ", ".join(
            f"p={p}: {ideal:.4f} -> {noisy:.4f}" for p, ideal, noisy in zip(curve["layers"], curve["ideal"], curve["noise_estimate"])))

print(f"  complexity ladder and other cases, P(cost <= 1.05 x optimum) per shot at p = {T4_P_MAIN}, n = {T4_N_PLOT} (classical columns: share of "
      f"{TASK04['classical_repeats']} repeats that found such a split with {TASK04['classical_budget']} candidates):")
ladder_columns = [("uniform", 0), ("LR-QAOA", 1 + 0), ("QAOA COBYLA", 1 + 3), ("CVaR COBYLA", 1 + 8), ("WS rounded A", 1 + 11)]
for scenario in TARGET_SHARES:
    print(f"    {SCENARIO_SLUGS.get(scenario, scenario)}: {'case':<17}" + "".join(f"{name:>13}" for name, _ in ladder_columns) + f"{'annealing':>11}{'local srch':>11}{'spectral':>9}")
    for row_index, case in enumerate(T4_QUANTUM_CASES):
        classical_hits = [np.mean([record["hit_near"] for key, record in classical_results.items()
                                   if key[0] == case and key[1] == T4_N_PLOT and key[3] == scenario and key[4] == method])
                          for method in ("simulated annealing", "local search (restarts)", "spectral bisection")]
        print(f"    {'':<{len(SCENARIO_SLUGS.get(scenario, scenario)) + 2}}{case:<17}" + "".join(f"{heat_values[scenario][row_index, column]:>13.4f}" for _, column in ladder_columns)
              + "".join(f"{value:>11.2f}" for value in classical_hits[:2]) + f"{classical_hits[2]:>9.2f}")

print(f"  same budget of {T4_SHOTS} candidate splits, probability that they contain (an optimum | a split with cost <= 1.05 x optimum), mean over cases:")
for scenario in TARGET_SHARES:
    parts = []
    for label, _, _ in comparison:
        opt, near = comparison_values.get((scenario, "hit_opt", label)), comparison_values.get((scenario, "hit_near", label))
        if near is not None:
            parts.append(f"{label} ({'-' if opt is None else f'{opt[0]:.2f}'} | {near[0]:.2f})")
    print(f"    {SCENARIO_SLUGS.get(scenario, scenario)}: " + "; ".join(parts))

print("  MODELLED outcome of each method's best split (assumptions, no measurement), difference to the split with the best modelled runtime:")
for scenario in TARGET_SHARES:
    reference_keys = [key for key in task04_references if key[3] == scenario and key[1] == T4_N_PLOT and key[0] in T4_QUANTUM_CASES and task04_references[key]]
    base_hours = np.mean([task04_references[key]["runtime_optimum"]["modelled_hours"] for key in reference_keys])
    base_kwh = np.mean([task04_references[key]["runtime_optimum"]["modelled_kwh"] for key in reference_keys])
    print(f"    {SCENARIO_SLUGS.get(scenario, scenario)} (best modelled runtime: {base_hours:.1f} h, {base_kwh:.1f} kWh on average): " + "; ".join(
        f"{label} {outcome_values[(scenario, 'modelled_hours', label)][0]:+.1f} % time"
        + (f", {outcome_values[(scenario, 'modelled_kwh', label)][0]:+.1f} % kWh" if (scenario, "modelled_kwh", label) in outcome_values else "")
        for label, _, _ in outcome_methods if (scenario, "modelled_hours", label) in outcome_values))

# interpretation, computed from the numbers above
print("  interpretation (computed from these results):")
for scenario in TARGET_SHARES:
    slug = SCENARIO_SLUGS.get(scenario, scenario)
    uniform, lr, cobyla = t4_find(scenario, "uniform control", 0), t4_find(scenario, "LR-QAOA", T4_P_MAIN), t4_find(scenario, "QAOA COBYLA", T4_P_MAIN)
    standard = [m for m in t4_method_summary if m["scenario"] == scenario and m["p"] == T4_P_MAIN and m["variant"] in ("QAOA", "CVaR-QAOA", "LR-QAOA")]
    if uniform is None or lr is None or cobyla is None or not standard:
        print(f"    {slug}: NO QAOA RESULTS for this scenario - see the failures listed below")
        continue
    best_standard = max(standard, key=lambda m: m["p_near"])
    annealing, local = comparison_values[(scenario, "hit_near", "simulated annealing")][0], comparison_values[(scenario, "hit_near", "local search")][0]
    best_classical = max(annealing, local)
    # sampling error of a classical share over (cases x repeats) runs; differences below two standard errors count as a tie
    runs = TASK04["classical_repeats"] * lr["instances"]
    margin = max(2.0 * np.sqrt(best_classical * (1.0 - best_classical) / runs), 2.0 / runs)
    def verdict(value):
        return "ahead of" if value > best_classical + margin else ("BEHIND" if value < best_classical - margin else "level with (within sampling error)")
    print(f"    {slug}: vs uniform guessing: LR-QAOA p = {T4_P_MAIN} raises P(cost <= 1.05 x optimum) per shot from {uniform['p_near']:.4f} to {lr['p_near']:.4f} "
          f"({lr['p_near'] / uniform['p_near']:.1f}x); below uniform on {lr['instances_below_uniform']} of {lr['instances']} cases. "
          f"The fairer random reference is the sampler with the target share ({lr['p_near_target_share_sampler']:.4f} per shot): "
          f"LR-QAOA is {lr['p_near'] / lr['p_near_target_share_sampler']:.1f}x that, QAOA COBYLA {cobyla['p_near'] / cobyla['p_near_target_share_sampler']:.1f}x.")
    fields_only, couplings_only = t4_find(scenario, "ablation: fields only", T4_P_MAIN), t4_find(scenario, "ablation: couplings only", T4_P_MAIN)
    if fields_only and couplings_only:
        print(f"    {slug}: ablation (LR-QAOA p = {T4_P_MAIN}, same angles): cost layer with only the single-region terms {fields_only['p_near']:.4f}, "
              f"only the pair terms {couplings_only['p_near']:.4f}, both {lr['p_near']:.4f} per shot -> the single-region terms alone only bias each region "
              f"towards device B (what the target-share sampler does classically); the pair terms add the rest.")
    print(f"    {slug}: reliability: QAOA COBYLA (the standard choice, fixed in advance) gives an exact optimum with probability {cobyla['p_opt']:.4f} per shot and a split "
          f"within 5 % with {cobyla['p_near']:.4f}; the best of {len(standard)} standard variants, picked AFTER seeing the results, is {best_standard['method']} with {best_standard['p_near']:.4f}.")
    print(f"    {slug}: vs classical, {T4_SHOTS} candidates each, probability of containing a split within 5 %: simulated annealing {annealing:.2f}, local search {local:.2f} "
          f"(+-{margin:.2f}); LR-QAOA {lr['hit_near']:.2f} -> {verdict(lr['hit_near'])} the best classical heuristic; QAOA COBYLA {cobyla['hit_near']:.2f} -> {verdict(cobyla['hit_near'])} it.")
    print(f"    {slug}: with the crude noise estimate P(cost <= 1.05 x optimum) per shot for LR-QAOA p = {T4_P_MAIN} drops from {lr['p_near']:.4f} to {lr['p_near_noise_estimate']:.4f}.")
print(f"    Budget caveat: the {T4_SHOTS}-candidate comparison counts only the final shots. The optimised variants also spent up to {TASK04['eval_budget']} objective evaluations "
      f"to find their angles (on hardware: that many circuits of {T4_SHOTS} shots each); LR-QAOA uses a fixed schedule and spends none.")
print("    A classical candidate (one energy evaluation) is also far cheaper than a quantum shot, so equal candidate counts favour QAOA.")
print(f"    Scale caveat: at n = {T4_N_PLOT} there are only {2 ** T4_N_PLOT} splits, so checking ALL of them costs about {2 ** T4_N_PLOT / T4_SHOTS:.0f} such budgets and is exact.")
print("    The rounded warm-start variants start from the classical spectral split: their success is partly classical (compare with 'spectral bisection' and their "
      "'start only' rows). Mixer B at its recovery angles returns the spectral split itself with certainty (a stationary point that COBYLA does not leave), "
      "so its row is the spectral split, not a quantum result.")
relaxation_spread = [float(np.ptp(group["warm_start_c"])) for key, group in qaoa_results.items()
                     if key[4] == "warm-start QAOA (relaxation, mixer A)" and "warm_start_c" in group]
if relaxation_spread and max(relaxation_spread) < 1e-6:
    print(f"    The relaxation warm start is degenerate here: its optimum is c_i = alpha for EVERY region (largest spread {max(relaxation_spread):.1e}), "
          "because equal values make both the cut and the balance term zero. At alpha = 0.5 it is identical to standard QAOA; "
          "at alpha = 0.8 its start state is the target-share sampler. It carries no information about the split.")
print("    'LR (success-cal.)' transfers badly: for equal devices its beta lies close to pi/2, which in that symmetric case acts like a small "
      "NEGATIVE beta (the landscape repeats every pi/2); the schedule chosen by success probability on the calibration meshes does not generalise.")
print("    Every result above is a noiseless simulation or a model. Nothing here shows quantum advantage, a speed-up or an energy saving.")

if task04_skipped:
    reasons = {}
    for line in task04_skipped:
        reasons[line.split(": ", 1)[1]] = reasons.get(line.split(": ", 1)[1], 0) + 1
    print("  skipped / reduced configurations:")
    for reason, count in reasons.items():
        print(f"    {count} instance(s): {reason}")
print(f"  failures: {len(task04_failures)}" + ("".join(f"\n    {failure['key']}: {failure['error']}" for failure in task04_failures) if task04_failures else " (none)"))
print("  references (verified against the primary sources before coding):")
for name, reference in TASK04_REFERENCES.items():
    print(f"    {name}: {reference}")
print(f"  files: {t4_summary_path}, {t4_csv_path}, {T4_CHECKPOINT} (full run records incl. convergence histories)")
print(f"  plots: {', '.join(str(path) for path in t4_plot_paths)}")
# TASK_04_IMPLEMENTATION_CELL_END
