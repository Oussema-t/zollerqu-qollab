"""29-qubit benchmark (HPC): the best QAOA settings found at <= 20 qubits versus the classical methods, all graded against
the exact optimum (the simulation needs the energy of every split anyway, so the optimum comes for free).

usage: python scripts/run_29_qubits.py <n> <seeds, comma-separated> <parallel instances> [test]
  test: also run the notebook's exact simulator on the same instance (n <= 16) and compare every probability.
Memory per instance at n = 29: energies float32 (2.1 GB) + statevector complex64 (4.3 GB) + small buffers.
Nothing here is tuned on the n = 29 instances: the angles and depths come from scripts/q29_choice.json (chosen at n = 20).
"""
import contextlib
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 29
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "7").split(",")]
PARALLEL = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MODE = sys.argv[4] if len(sys.argv) > 4 else "run"     # run | test | freeze (save the region assignments, run on the Mac)
TEST = MODE == "test"
CASES = ["2 main", "9 priority"]
SHOTS = 1000
LOW = (N + 1) // 2                      # state k = lo + 2^LOW * hi: energies are built as (hi rows) x (lo columns)
BLOCK_BITS = min(int(os.environ.get("Q29_BLOCK_BITS", 18)), N)   # mixer: qubits below this are applied block by block, in cache
CHUNK_BITS = min(int(os.environ.get("Q29_CHUNK_BITS", 20)), N)   # streaming passes work on 2^CHUNK_BITS amplitudes at a time
assert CHUNK_BITS >= LOW, "a streaming chunk must hold whole rows of the energy table"
OUT = ROOT / "results" / f"hpc_n{N}"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:7.1f} s] {message}", flush=True)


def load_namespace():
    """The notebook's code cells up to the end of the Task 04 calibration (functions, meshes, regions, LR schedule)."""
    nb = nbformat.read(ROOT / "mesh_idea_ZollerQu.ipynb", as_version=4)
    ns = {}
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or cell.source.startswith("# TASK_01_IMPLEMENTATION_CELL_START"):
            continue
        source, last = cell.source, cell.source.startswith("# TASK_04_IMPLEMENTATION")
        if last:
            source = source.split("# --------------------------------------------------------------- benchmark ----")[0]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f"cell{index}", "exec"), ns)
        if last:
            return ns
    raise RuntimeError("Task 04 cell not found")


# ------------------------------------------------------------------ exact energies of all 2^n splits ----
def bit_matrix(count, bits):
    k = np.arange(count, dtype=np.int64)
    return ((k[:, None] >> np.arange(bits, dtype=np.int64)) & 1).astype(np.float64)


class Energies:
    """E(z) = z J z + h z + const for every state, built from a low half and a high half of the bits:
    E[hi, lo] = e_hi[hi] + e_lo[lo] + z_hi C z_lo, with C the couplings between the halves (one small matrix product per block)."""

    def __init__(self, model, n):
        J, h = model["J"], model["h"]
        self.n, self.const, self.low, self.high = n, float(model["const"]), LOW, n - LOW
        z_lo, z_hi = 1 - 2 * bit_matrix(2**self.low, self.low), 1 - 2 * bit_matrix(2**self.high, self.high)
        lo, hi = slice(0, self.low), slice(self.low, n)
        self.e_lo = np.sum((z_lo @ J[lo, lo]) * z_lo, axis=1) + z_lo @ h[lo]
        self.e_hi = np.sum((z_hi @ J[hi, hi]) * z_hi, axis=1) + z_hi @ h[hi]
        self.cross = z_lo @ (J[lo, hi] + J[hi, lo].T)          # (2^low, high)
        self.z_hi = z_hi
        a = np.asarray(model["a_tilde"], dtype=np.float64)
        self.share_lo, self.share_hi = bit_matrix(2**self.low, self.low) @ a[lo], bit_matrix(2**self.high, self.high) @ a[hi]
        self.rows = max(1, 2 ** (CHUNK_BITS - self.low))

    def blocks(self):
        """Yield (first state index, exact float64 energies E - const of the next rows x 2^low states)."""
        for r0 in range(0, 2**self.high, self.rows):
            r1 = min(r0 + self.rows, 2**self.high)
            block = self.e_hi[r0:r1, None] + self.e_lo[None, :] + self.z_hi[r0:r1] @ self.cross.T
            yield r0 << self.low, block.reshape(-1)

    def shares(self, start, count):
        r0, rows = start >> self.low, count >> self.low
        return (self.share_hi[r0:r0 + rows, None] + self.share_lo[None, :]).reshape(-1)


def direct_energies(states, model, n):
    """Exact float64 energies of a few states, straight from J, h, const (independent of the block construction)."""
    z = 1.0 - 2.0 * ((np.asarray(states, dtype=np.int64)[:, None] >> np.arange(n)) & 1)
    return np.einsum("si,si->s", z @ model["J"], z) + z @ model["h"] + model["const"]


def enumerate_all(model, n, near_factor):
    """float32 array of E - const for every state, plus exact optimum, maximum, optimal states and states within 5 %."""
    energy = Energies(model, n)
    shifted = np.empty(2**n, dtype=np.float32)
    e_min, e_max = np.inf, -np.inf
    for start, block in energy.blocks():
        shifted[start:start + len(block)] = block
        e_min, e_max = min(e_min, block.min()), max(e_max, block.max())
    e_min, e_max = e_min + energy.const, e_max + energy.const
    assert e_min > 1e-9, "zero optimum: the 5 % window would need an absolute tolerance"
    optimal, near = [], []
    for start, block in energy.blocks():                       # second exact pass (cheap) for the two reference sets
        full = block + energy.const
        optimal.append(start + np.flatnonzero(full <= e_min + 1e-9))
        near.append(start + np.flatnonzero(full <= near_factor * e_min + 1e-9))
    return energy, shifted, float(e_min), float(e_max), np.concatenate(optimal), np.concatenate(near)


# ------------------------------------------------------------------ QAOA statevector (complex64, in place) ----
def apply_phase(psi, shifted, gamma_over_scale):
    step = 2**CHUNK_BITS
    for start in range(0, len(psi), step):
        phase = shifted[start:start + step].astype(np.float64) * gamma_over_scale
        psi[start:start + step] *= (np.cos(phase) - 1j * np.sin(phase)).astype(np.complex64)


def rotate_pair(a0, a1, c, s, t1, t2):
    """(a0, a1) <- (c a0 + s a1, s a0 + c a1): the single-qubit gate exp(+i beta X) with c = cos beta, s = i sin beta."""
    np.copyto(t1, a0)
    a0 *= c
    np.multiply(a1, s, out=t2)
    a0 += t2
    a1 *= c
    np.multiply(t1, s, out=t2)
    a1 += t2


def apply_mixer(psi, beta, n, t1, t2):
    c, s = np.complex64(np.cos(beta)), np.complex64(1j * np.sin(beta))
    block, half_block = 2**BLOCK_BITS, 2 ** (BLOCK_BITS - 1)
    for start in range(0, len(psi), block):                    # low qubits: all of them while the block is in cache
        piece = psi[start:start + block]
        for q in range(BLOCK_BITS):
            view = piece.reshape(-1, 2, 2**q)
            rotate_pair(view[:, 0, :], view[:, 1, :], c, s, t1.reshape(-1, 2**q), t2.reshape(-1, 2**q))
    for q in range(BLOCK_BITS, n):                             # high qubits: partner amplitudes are 2^q apart
        half = 2**q
        for base in range(0, len(psi), 2 * half):
            for offset in range(0, half, half_block):
                lo = base + offset
                rotate_pair(psi[lo:lo + half_block], psi[lo + half:lo + half + half_block], c, s, t1, t2)


def qaoa_state(psi, shifted, scale, angles, n, t1, t2):
    """Standard QAOA from |+>^n: per layer exp(-i gamma (E - const) / scale), then exp(+i beta X) on every qubit
    (the same convention as the notebook's simulator; checked against it in test mode)."""
    p = len(angles) // 2
    psi[:] = np.complex64(2.0 ** (-n / 2))
    for gamma, beta in zip(angles[:p], angles[p:]):
        apply_phase(psi, shifted, gamma / scale)
        apply_mixer(psi, beta, n, t1, t2)
    return psi


def state_metrics(psi, shifted, energy, ref, alpha, epsilon, n, seed):
    """Exact metrics of |psi|^2 (streamed) and SHOTS seeded samples (inverse CDF, streamed)."""
    step = 2**CHUNK_BITS
    total = mean_shifted = p_bal = 0.0
    for start in range(0, len(psi), step):
        probs = np.abs(psi[start:start + step].astype(np.complex128)) ** 2
        total += probs.sum()
        mean_shifted += probs @ shifted[start:start + step].astype(np.float64)
        p_bal += probs[np.abs(energy.shares(start, len(probs)) - alpha) <= epsilon].sum()
    draws = np.sort(np.random.default_rng(seed).random(SHOTS)) * total
    samples, cumulative = [], 0.0
    for start in range(0, len(psi), step):
        cdf = cumulative + np.cumsum(np.abs(psi[start:start + step].astype(np.complex128)) ** 2)
        inside = draws[(draws >= cumulative) & (draws < cdf[-1])]
        samples.append(start + np.minimum(np.searchsorted(cdf, inside, side="right"), len(cdf) - 1))
        cumulative = cdf[-1]
    samples = np.concatenate(samples + [np.full(SHOTS - sum(map(len, samples)), len(psi) - 1)])   # round-off at the very end
    costs = direct_energies(samples, ref["model"], n)
    p_opt = float(np.sum(np.abs(psi[ref["optimal"]].astype(np.complex128)) ** 2) / total)
    p_near = float(np.sum(np.abs(psi[ref["near"]].astype(np.complex128)) ** 2) / total)
    mean_cost = mean_shifted / total + energy.const
    return dict(norm=float(total), p_opt=p_opt, p_near=p_near, mean_cost=float(mean_cost),
                mean_ratio=float((ref["e_max"] - mean_cost) / (ref["e_max"] - ref["e_min"])), p_balanced=float(p_bal / total),
                hit_opt=float(1 - (1 - p_opt) ** SHOTS), hit_near=float(1 - (1 - p_near) ** SHOTS),
                sampled_p_near=float(np.mean(costs <= 1.05 * ref["e_min"] + 1e-9)),
                sampled_hit_opt=bool(costs.min() <= ref["e_min"] + 1e-9), sampled_hit_near=bool(costs.min() <= 1.05 * ref["e_min"] + 1e-9),
                sampled_best_cost=float(costs.min()), sampled_mean_cost=float(costs.mean()),
                best_state=int(samples[np.argmin(costs)]), best_gap_percent=float(100 * (costs.min() / ref["e_min"] - 1)),
                sample_seed=[int(v) for v in seed])


# ------------------------------------------------------------------ one instance ----
def instance_mesh(ns, case_name, seed):
    if seed == 7:
        return ns["mesh_cases"][case_name]
    spec = ns["MESH_CASES"][case_name]
    source = spec.get("mesh_of", case_name)
    points, triangles = ns["build_case_mesh"](ns["MESH_CASES"][source], seed)
    return ns["add_case_weights"](spec, ns["check_case_mesh"](source, ns["MESH_CASES"][source], points, triangles))


def quantum_configs(ns, scenario, choice):
    """(label, role, p, angles): the best fixed schedule at n = 20, the COBYLA angles transferred from n = 20, and LR-QAOA p = 2."""
    best, trained = choice["best_fixed"], choice["trained"]
    best_deltas = ns["T4_LR_DELTAS"][scenario][best["variant"]]
    return [(f"{best['variant']}, p = {best['p']} (best fixed schedule at n = 20)", "best fixed schedule", best["p"],
             ns["lr_angles"](best["p"], *best_deltas)),
            (f"QAOA + COBYLA p = {trained['p']}, angles transferred from n = 20", "transferred trained angles", trained["p"],
             np.array(trained["angles"])),
            ("LR-QAOA p = 2 (scaling-table row)", "continuity", 2, ns["lr_angles"](2, *ns["T4_LR_DELTAS"][scenario]["LR-QAOA"]))]


def run_instance(ns, case_name, seed, scenario, choice):
    clock = time.time()
    T4 = ns["TASK04"]
    regions = ns["coarsen_mesh"](instance_mesh(ns, case_name, seed), N, seed)
    ns["check_region_case"]((case_name, N, seed), regions, N)
    inst = ns["t4_build_instance"](regions, case_name, N, seed, scenario)
    model, alpha = inst["model"], inst["alpha"]
    energy, shifted, e_min, e_max, optimal, near = enumerate_all(model, N, T4["near_optimal_factor"])
    ref = dict(model=model, e_min=e_min, e_max=e_max, optimal=optimal, near=near)
    checks = np.random.default_rng([seed, N]).integers(0, 2**N, 200)
    direct = np.array([ns["direct_cost"](1 - 2 * ns["state_index_to_x"](int(k), N), model) for k in checks])
    assert np.allclose(direct, direct_energies(checks, model, N), rtol=1e-12, atol=1e-12), "J, h, const differ from the direct cost"
    assert np.allclose(shifted[checks].astype(np.float64) + energy.const, direct, rtol=1e-6, atol=1e-6), "block energies are wrong"
    x_opt = ns["state_index_to_x"](int(optimal[0]), N)
    share_opt = float(model["a_tilde"] @ x_opt)
    record = dict(case=case_name, n=N, seed=seed, scenario=scenario, alpha=alpha, lam=inst["lam"], mu=inst["mu"],
                  fingerprint=inst["fingerprint"], e_min=e_min, e_max=e_max, n_optimal=len(optimal), optimal_states=optimal.tolist(),
                  n_near=len(near), optimum_share_b=share_opt, optimum_balanced=bool(abs(share_opt - alpha) <= T4["epsilon"]),
                  optimum_outcome=ns["modelled_outcome"](inst, int(optimal[0])), region_pairs=len(regions["region_pairs"]),
                  two_qubit_gates_per_layer=N * (N - 1) // 2, enumeration_seconds=time.time() - clock)
    log(f"{case_name} seed {seed} {scenario}: optimum {e_min:.6f} ({len(optimal)} state(s)), {len(near)} states within 5 %")

    # classical baselines: the Task 04 functions and budgets (1,000 candidates, 100 repeats), graded against the exact optimum
    clock2 = time.time()
    classical = ns["run_classical"](inst)
    record["classical"] = {}
    for method, rec in classical.items():
        costs = np.array(rec["best_costs"])
        record["classical"][method] = dict(hit_opt=float(np.mean(costs <= e_min + 1e-9)), hit_near=float(np.mean(costs <= 1.05 * e_min + 1e-9)),
                                           mean_best_cost=rec["mean_best_cost"], best_cost=rec["best_cost"], repeats=rec["repeats"],
                                           evaluations=rec["evaluations_per_repeat"], best_gap_percent=100 * (rec["best_cost"] / e_min - 1),
                                           best_state=rec["best_state"], outcome=rec["outcome"], wall_seconds=rec["wall_seconds"])

    def target_share_probability(states):                     # every region on device B with probability alpha, independently
        ones = np.sum((states[:, None] >> np.arange(N)) & 1, axis=1)
        return float(np.sum(alpha**ones * (1 - alpha) ** (N - ones)))

    record["per_shot_random"] = dict(uniform_p_opt=len(optimal) / 2**N, uniform_p_near=len(near) / 2**N,
                                     target_share_p_opt=target_share_probability(optimal), target_share_p_near=target_share_probability(near))
    record["classical_seconds"] = time.time() - clock2

    # QAOA with the settings chosen at n <= 20 (never tuned on these instances)
    psi = np.empty(2**N, dtype=np.complex64)
    t1, t2 = np.empty(2 ** (BLOCK_BITS - 1), dtype=np.complex64), np.empty(2 ** (BLOCK_BITS - 1), dtype=np.complex64)
    record["quantum"] = []
    for label, role, p, angles in quantum_configs(ns, scenario, choice):
        clock3 = time.time()
        qaoa_state(psi, shifted, model["scale"], angles, N, t1, t2)
        sim_seconds = time.time() - clock3
        metrics = state_metrics(psi, shifted, energy, ref, alpha, T4["epsilon"], N, [seed, p, len(role), N])
        gates = p * N * (N - 1) // 2
        fidelity = (1 - T4["noise_error_per_two_qubit_gate"]) ** gates
        metrics.update(label=label, role=role, p=p, angles=[float(v) for v in angles], two_qubit_gates=gates,
                       noise_fidelity_estimate=fidelity, p_near_noise_estimate=fidelity * metrics["p_near"] + (1 - fidelity) * len(near) / 2**N,
                       shots=SHOTS, best_outcome=ns["modelled_outcome"](inst, metrics["best_state"]),
                       simulation_seconds=sim_seconds, seconds=time.time() - clock3)
        if TEST:                                                # compare with the notebook's exact simulator
            reference = ns["qaoa_probabilities"](inst["energy_scaled"], angles[:p], angles[p:], N)
            mine = np.abs(psi.astype(np.complex128)) ** 2
            metrics["test_max_probability_difference"] = float(np.max(np.abs(mine - reference)))
            metrics["test_reference"] = ns["distribution_metrics"](reference, ns["t4_view"](inst))
        record["quantum"].append(metrics)
        log(f"  {label}: P(within 5 %) {metrics['p_near']:.3e}, P(optimum) {metrics['p_opt']:.3e}, {sim_seconds:.0f} s")
    if TEST:
        record["test_exact_e_min"] = inst["e_min"]
        record["test_exact_n_near"] = int(inst["near_mask"].sum())
    record["seconds"] = time.time() - clock
    del psi, shifted
    gc.collect()
    return record


def target_path(case_name, seed, alpha):
    return OUT / f"{case_name.replace(' ', '_')}_seed{seed}_{'equal' if abs(alpha - 0.5) < 1e-12 else 'cpu_gpu'}.json"


if __name__ == "__main__":
    ns = load_namespace()
    log(f"notebook cells loaded (frozen regions: {ns['frozen_region_log']}); LR deltas "
        + "; ".join(f"{s}: {v['LR-QAOA']}" for s, v in ns["T4_LR_DELTAS"].items()))
    if MODE == "freeze":                                       # region assignments are computed once, on the team's Mac
        for case, seed in [(case, seed) for seed in SEEDS for case in CASES]:
            regions = ns["coarsen_mesh"](instance_mesh(ns, case, seed), N, seed)
            ns["check_region_case"]((case, N, seed), regions, N)
        np.savez_compressed(ns["FROZEN_REGIONS_PATH"], **ns["frozen_regions"])
        log(f"frozen regions: {ns['frozen_region_log']}, {len(ns['frozen_regions'])} assignments in {ns['FROZEN_REGIONS_PATH']}")
        sys.exit(0)
    choices = json.loads((ROOT / "scripts" / "q29_choice.json").read_text())["choice"]
    jobs = [(case, seed, scenario) for seed in SEEDS for case in CASES for scenario in ns["TARGET_SHARES"]]
    running = {}
    for case, seed, scenario in jobs:
        target = target_path(case, seed, ns["TARGET_SHARES"][scenario])
        if target.exists():
            log(f"skip {target.name} (already done)")
            continue
        if PARALLEL <= 1:                                      # no fork (macOS: the BLAS library is not fork-safe)
            record = run_instance(ns, case, seed, scenario, choices[scenario])
            target.write_text(json.dumps(ns["t4_jsonable"](record)))
            continue
        while len(running) >= PARALLEL:
            pid, status = os.wait()
            log(f"finished {running.pop(pid)} (exit status {status})")
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                record = run_instance(ns, case, seed, scenario, choices[scenario])
                target.write_text(json.dumps(ns["t4_jsonable"](record)))
            except BaseException as error:                      # noqa: BLE001 - written next to the results
                import traceback
                target.with_suffix(".error.txt").write_text(traceback.format_exc())
                log(f"ERROR in {target.name}: {type(error).__name__}: {error}")
                code = 1
            finally:
                os._exit(code)
        running[pid] = target.name
        log(f"started {target.name}")
    while running:
        pid, status = os.wait()
        log(f"finished {running.pop(pid)} (exit status {status})")
    log("all instances done")
    (OUT / "DONE").write_text("done")
