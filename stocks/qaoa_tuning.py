"""Offline tuning of QAOA angles on the exact NumPy statevector. Training objective only, never a backend.

What is tuned is always the model energy of the TRAINING-derived Ising model. Test-period returns are never
read here. Every optimiser gets the same objective, the same starting points and the same hard cap on the
number of simulated circuits ("evaluations"); the result of a run is the best point it evaluated.
Optimisers minimise the energy divided by the circuit's scale (energy in units of the largest term), so
that default step sizes and tolerances mean the same for every model. Results are saved in model units.

Variants ("kind") and their parameter vectors, p layers:
    standard     [gamma_1..p, beta_1..p]                    2p numbers
    multi_angle  [gamma_pair_1..p, gamma_field_1..p, beta]  3p: separate angle for links and for single assets
    dc           [gamma_1..p, beta_1..p, alpha_1..p]        3p: extra RY(2 alpha) on every qubit after the mixer
Objectives: "expectation" (mean energy) or ("cvar", a): mean of the best share a of the probability mass.
"""
import numpy as np
from scipy import optimize

from stocks import qaoa, qaoa_sim

# SIGN: with the mixer rx(+2 beta), lowering the energy needs gamma and beta of OPPOSITE sign. A ramp with
# both positive (the 0.6 / 0.3 recipe as written) steers towards the highest energy. The grid covers both signs.
LR_GRID = {"dgamma": [round(0.1 * k, 1) for k in range(1, 16)],                    # 0.1 .. 1.5
           "dbeta": [round(0.1 * k, 1) for k in range(-10, 11) if k != 0]}         # -1.0 .. -0.1 and 0.1 .. 1.0
SIGN_CORRECTED_DEFAULT = (qaoa.DEFAULT_DGAMMA, -qaoa.DEFAULT_DBETA)                 # (0.6, -0.3)
LR_P_VALUES = (1, 2, 3, 4, 5)
SCIPY_OPTIMIZERS = ("COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B")
OPTIMIZERS = SCIPY_OPTIMIZERS + ("SPSA",)
EVALUATIONS_PER_LAYER = 200           # hard cap = 200 * p simulated circuits, for every optimiser and variant
RANDOM_START_BOX = {"gamma": (0.0, 1.5), "beta": (-1.0, 1.0), "alpha": (-0.5, 0.5)}
N_RANDOM_STARTS = 3
START_SEED = 2026
SCIPY_OPTIONS = {  # library defaults except the caps and COBYLA's first step; not tuned
    "COBYLA": lambda cap: {"maxiter": cap, "rhobeg": 0.2},
    "Nelder-Mead": lambda cap: {"maxfev": cap, "maxiter": cap},
    "Powell": lambda cap: {"maxfev": cap},
    "L-BFGS-B": lambda cap: {"maxfun": cap},
}
SPSA_SETTINGS = {"a": 0.2, "c": 0.1, "alpha": 0.602, "gamma": 0.101, "stability": 0.1}  # Spall's standard exponents
VARQITE_SETTINGS = {"dtau": 0.2, "regularisation": 1e-3, "difference": 1e-4}

KIND_LAYOUT = {"standard": ("gamma", "beta"), "multi_angle": ("gamma", "gamma_field", "beta"), "dc": ("gamma", "beta", "alpha")}


def evaluation_cap(p):
    return EVALUATIONS_PER_LAYER * p


def unpack(kind, x, p):
    """Split a parameter vector into the simulator's keyword arguments."""
    layout = KIND_LAYOUT[kind]
    x = np.asarray(x, dtype=float)
    if x.shape != (len(layout) * p,):
        raise ValueError(f"{kind} with p = {p} needs {len(layout) * p} parameters, got {x.shape}")
    parts = dict(zip(layout, x.reshape(len(layout), p)))
    out = {"gammas": parts["gamma"], "betas": parts["beta"]}
    if "gamma_field" in parts:
        out["field_gammas"] = parts["gamma_field"]
    if "alpha" in parts:
        out["alphas"] = parts["alpha"]
    return out


def lr_start(kind, p, dgamma, dbeta):
    """Linear-ramp angles as a starting point (the extra angles start at the standard-QAOA values)."""
    gammas, betas = qaoa.lr_schedule(p, dgamma, dbeta)
    parts = {"gamma": gammas, "beta": betas, "gamma_field": gammas, "alpha": np.zeros(p)}
    return np.concatenate([parts[name] for name in KIND_LAYOUT[kind]])


def random_start(kind, p, rng):
    box = dict(RANDOM_START_BOX, gamma_field=RANDOM_START_BOX["gamma"])
    return np.concatenate([rng.uniform(*box[name], size=p) for name in KIND_LAYOUT[kind]])


def starts_for(kind, p, lr_tuned):
    """The same named starts for every optimiser: default ramp, tuned ramp, and fixed-seed random points."""
    starts = {"lr_default": lr_start(kind, p, qaoa.DEFAULT_DGAMMA, qaoa.DEFAULT_DBETA),
              "lr_tuned": lr_start(kind, p, *lr_tuned)}
    for k in range(N_RANDOM_STARTS):
        seed = START_SEED + 1000 * p + k
        starts[f"random_seed_{seed}"] = random_start(kind, p, np.random.default_rng(seed))
    return starts


def objective_value(probabilities, energies, objective):
    if objective == "expectation":
        return float(probabilities @ energies)
    name, alpha = objective
    if name != "cvar":
        raise ValueError(f"unknown objective {objective!r}")
    return qaoa_sim.cvar(probabilities, energies, alpha)


class EvaluationCapReached(Exception):
    pass


class Objective:
    """Objective with a hard evaluation cap. Remembers the best point evaluated."""

    def __init__(self, sim, kind, p, objective="expectation", cap=None):
        self.sim, self.kind, self.p, self.objective = sim, kind, p, objective
        self.cap = evaluation_cap(p) if cap is None else cap
        self.count, self.best_value, self.best_x = 0, np.inf, None

    def state(self, x):
        """Amplitudes at x; counts as one evaluation and updates the best point."""
        if self.count >= self.cap:
            raise EvaluationCapReached
        amplitudes = self.sim.amplitudes(**unpack(self.kind, x, self.p))
        self.count += 1
        value = objective_value(np.abs(amplitudes) ** 2, self.sim.energies, self.objective) / self.sim.scale
        if value < self.best_value:
            self.best_value, self.best_x = value, np.array(x, dtype=float)
        return amplitudes, value

    def __call__(self, x):
        return self.state(x)[1]


def spsa(objective, x0, rng, settings=SPSA_SETTINGS):
    """Simultaneous-perturbation stochastic approximation: two evaluations per step, any number of parameters."""
    x = np.array(x0, dtype=float)
    steps = max((objective.cap - 1) // 2, 1)
    for k in range(steps):
        a_k = settings["a"] / (k + 1 + settings["stability"] * steps) ** settings["alpha"]
        c_k = settings["c"] / (k + 1) ** settings["gamma"]
        delta = rng.choice([-1.0, 1.0], size=len(x))
        gradient = (objective(x + c_k * delta) - objective(x - c_k * delta)) / (2.0 * c_k) * delta
        x = x - a_k * gradient
    objective(x)


def varqite(objective, x0, settings=VARQITE_SETTINGS):
    """Variational imaginary-time evolution (McLachlan) of the angles, OFFLINE on the statevector.

    theta <- theta - dtau * (A + reg I)^-1 C, with A_ij = Re(<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>)
    and C_i = Re <d_i psi| H/scale |psi>. Derivatives by central differences: 2 * (parameters) + 1 simulated
    circuits per step. This is a classical way to TRAIN the angles; it is not run on a quantum computer.
    """
    x = np.array(x0, dtype=float)
    h, sim = settings["difference"], objective.sim
    cost = (sim.energies - sim.energies.mean()) / sim.scale
    while True:
        psi, _ = objective.state(x)
        derivatives = []
        for i in range(len(x)):
            step = np.zeros(len(x))
            step[i] = h
            derivatives.append((objective.state(x + step)[0] - objective.state(x - step)[0]) / (2.0 * h))
        d = np.array(derivatives)
        overlap = d.conj() @ psi
        a = np.real(d.conj() @ d.T - np.outer(overlap, overlap.conj()))
        c = np.real(d.conj() @ (cost * psi))
        x = x - settings["dtau"] * np.linalg.solve(a + settings["regularisation"] * np.eye(len(x)), c)


def optimize_angles(sim, kind, p, optimizer, x0, objective="expectation", cap=None, seed=START_SEED):
    """One optimisation run. Returns a JSON-compatible dict; errors are recorded, not raised."""
    target = Objective(sim, kind, p, objective, cap)
    note, error = "", None
    try:
        if optimizer in SCIPY_OPTIMIZERS:
            result = optimize.minimize(target, np.array(x0, dtype=float), method=optimizer,
                                       options=SCIPY_OPTIONS[optimizer](target.cap))
            note = str(result.message)
        elif optimizer == "SPSA":
            spsa(target, x0, np.random.default_rng(seed))
            note = "finished its steps"
        elif optimizer == "VarQITE":
            varqite(target, x0)
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}")
    except EvaluationCapReached:
        note = "stopped at the evaluation cap"
    except Exception as err:  # keep failed runs in the record
        error = f"{type(err).__name__}: {err}"
    out = {"qaoa_type": kind, "p": int(p), "optimizer": optimizer, "evaluations": int(target.count),
           "cap": int(target.cap), "objective": objective if isinstance(objective, str) else f"cvar_{objective[1]}",
           "note": note, "error": error, "objective_value": None, "angles": None}
    if target.best_x is not None:
        out["objective_value"] = float(target.best_value * sim.scale)
        out["angles"] = {name: [float(v) for v in values] for name, values in unpack(kind, target.best_x, p).items()}
    return out


def probabilities_for(sim, angles):
    """Distribution for an 'angles' dict saved by optimize_angles (does not count against any cap)."""
    return sim.probabilities(**{name: np.array(values) for name, values in angles.items()})


def lr_grid_search(sim, optimal, evaluate, p_values=LR_P_VALUES, grid=LR_GRID):
    """Linear-ramp grid: expected model energy and per-shot P(optimum) for every (dgamma, dbeta) and every p.

    optimal: boolean mask of the optimal state(s); evaluate: probabilities -> metrics dict (used at 3 points per p).
    The best point is the one with the lowest expected energy (the tuning objective), never the best P(optimum),
    because the optimum is not known for a problem one actually needs to solve.
    """
    energies, out = sim.energies, {}
    for p in p_values:
        expectation = np.empty((len(grid["dgamma"]), len(grid["dbeta"])))
        p_optimum = np.empty_like(expectation)
        for a, dgamma in enumerate(grid["dgamma"]):
            for b, dbeta in enumerate(grid["dbeta"]):
                probabilities = sim.probabilities(*qaoa.lr_schedule(p, dgamma, dbeta))
                expectation[a, b] = probabilities @ energies
                p_optimum[a, b] = probabilities[optimal].sum()
        a, b = np.unravel_index(np.argmin(expectation), expectation.shape)
        best = (grid["dgamma"][a], grid["dbeta"][b])
        gammas, betas = qaoa.lr_schedule(p, *best)
        out[p] = {
            "expectation": expectation.tolist(), "p_optimum": p_optimum.tolist(),
            "best_dgamma": best[0], "best_dbeta": best[1],
            "gammas": [float(v) for v in gammas], "betas": [float(v) for v in betas],
            "best": evaluate(sim.probabilities(gammas, betas)),
            "recipe_as_written": evaluate(sim.probabilities(*qaoa.lr_schedule(p))),
            "sign_corrected_default": evaluate(sim.probabilities(*qaoa.lr_schedule(p, *SIGN_CORRECTED_DEFAULT))),
            "best_on_grid_edge": bool(a in (0, len(grid["dgamma"]) - 1) or b in (0, len(grid["dbeta"]) - 1)),
            "grid_points": int(expectation.size),
        }
    return out
