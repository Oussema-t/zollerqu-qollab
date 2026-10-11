"""Memory-lean exact statevector simulation of LR-QAOA at up to 29 qubits (offline, many-core machine).

Differences from qaoa_sim.QaoaSimulator, which stores all 2^n spin states:
- energies of all 2^n states are built bit by bit (float32, 2^n * 4 bytes: 2.1 GB at 29 qubits);
- amplitudes are complex64 (4.3 GB at 29 qubits); the cost phase is applied in chunks;
- the mixer rx(2 beta) on every qubit is applied in groups of GROUP qubits with one Kronecker block each.
Index convention as everywhere in this project: bit i of the state index is qubit i; bit 0 (|0>) is spin +1.
Enumerating all energies also gives the EXACT optimum at 29 assets (brute force over 2^29 states).

Run (29 qubits, both modes, writes results/stocks_qaoa_n29.json):   python -m stocks.large_sim 29
"""
import datetime
import json
import sys
import time

import numpy as np

from stocks import classical_solvers as solvers
from stocks import fair_budget, ising, kaggle_data, qaoa, qaoa_sim, task3_report as t3, task4_report as t4

ROOT = kaggle_data.DATA_DIR.parent
GROUP = 4             # qubits per mixer block (block matrix 16 x 16)
CHUNK = 1 << 24       # elements per chunk for the cost phase
LR_P_VALUES = (1, 2, 3, 4, 5)
DENSE_P_VALUES = (1, 2, 3)
BUDGET, SEEDS = 1000, 200
TOP_STATES = 5


def energies_all_states(model, dtype=np.float32):
    """E(z) for every state index z (float32 by default), without storing the spin states.

    Adding qubit q to the (q-1)-qubit energies adds z_q * (h_q + sum_{i<q} J_iq z_i).
    """
    couplings, fields = model.couplings, model.fields
    energies = np.array([model.const], dtype=dtype)
    for q in range(model.n):
        local = np.zeros(1 << q, dtype=dtype)
        for i in range(q):
            if couplings[i, q] != 0.0:
                view = local.reshape(-1, 2, 1 << i)
                view[:, 0, :] += dtype(couplings[i, q])   # bit i = 0 -> spin +1
                view[:, 1, :] -= dtype(couplings[i, q])
        field = dtype(fields[q]) + local
        energies = np.concatenate([energies + field, energies - field])  # bit q = 0 first (spin +1)
    return energies


def dot64(a, b, chunk=CHUNK):
    """sum(a * b) accumulated in float64, chunked."""
    total = 0.0
    for start in range(0, len(a), chunk):
        total += float(np.dot(a[start:start + chunk].astype(np.float64), b[start:start + chunk].astype(np.float64)))
    return total


def spins_of_index(index, n):
    bits = (index >> np.arange(n)) & 1
    return (1 - 2 * bits).astype(np.int8)


class LargeQaoa:
    """Exact probabilities of a QAOA circuit, built for n up to 29 on a 32 GB machine."""

    def __init__(self, model, energies=None, dtype=np.complex64):
        self.n, self.dtype = model.n, dtype
        self.scale = qaoa_sim.cost_scale(model)
        self.energies = energies_all_states(model) if energies is None else energies
        self.evaluations = 0

    def amplitudes(self, gammas, betas):
        n, size = self.n, 1 << self.n
        amp = np.full(size, 1.0 / np.sqrt(size), dtype=self.dtype)
        for gamma, beta in zip(np.atleast_1d(gammas), np.atleast_1d(betas)):
            k = float(gamma) / self.scale
            for start in range(0, size, CHUNK):
                amp[start:start + CHUNK] *= np.exp(-1j * k * self.energies[start:start + CHUNK].astype(np.float64)).astype(self.dtype)
            q = 0
            while q < n:
                m = min(GROUP, n - q)
                block = qaoa_sim.rx_block(float(beta), m).astype(self.dtype)
                low = 1 << q
                view = amp.reshape(-1, 1 << m, low)
                if low == 1:
                    amp = (view[:, :, 0] @ block.T).reshape(-1)
                else:
                    amp = np.matmul(block, view).reshape(-1)
                q += m
        self.evaluations += 1
        return amp

    def probabilities(self, gammas, betas):
        amp = self.amplitudes(gammas, betas)
        probabilities = np.abs(amp)
        del amp
        np.square(probabilities, out=probabilities)
        return probabilities


def metrics(probabilities, energies, e_min, e_max, tolerance, shots=BUDGET, within=0.01):
    """Same definitions as qaoa_sim.distribution_metrics, computed without float32 accumulation errors."""
    total = dot64(probabilities, np.ones(1, dtype=np.float32).repeat(1)) if False else float(np.sum(probabilities, dtype=np.float64))
    expectation = dot64(probabilities, energies) / total
    spread = e_max - e_min
    optimal = energies <= e_min + tolerance
    p_opt = float(np.sum(probabilities[optimal], dtype=np.float64)) / total
    near = energies <= e_min + within * spread
    p_near = float(np.sum(probabilities[near], dtype=np.float64)) / total
    top = int(np.argmax(probabilities))
    return {"expectation": expectation, "quality": 1.0 if spread <= 0 else (e_max - expectation) / spread,
            "p_optimum": p_opt, "p_within_1pct": p_near, "p_optimum_in_shots": 1.0 - (1.0 - min(p_opt, 1.0)) ** shots,
            "most_probable_state": top, "most_probable_probability": float(probabilities[top]) / total,
            "most_probable_energy": float(energies[top]), "probability_sum": total}


def sampled_hits(probabilities, energies, e_min, tolerance, budget, seeds, rng):
    """hit@budget measured by sampling `seeds` repetitions of `budget` shots from the exact distribution."""
    cdf = np.cumsum(probabilities, dtype=np.float64)
    draws = rng.random(seeds * budget) * cdf[-1]
    index = np.searchsorted(cdf, draws)
    del cdf
    best = energies[index].reshape(seeds, budget).min(axis=1)
    return float(np.mean(best <= e_min + tolerance))


def top_states(probabilities, energies, k, n, tickers, mode):
    order = np.argpartition(-probabilities, k)[:k]
    order = order[np.argsort(-probabilities[order])]
    return [{"index": int(i), "probability": float(probabilities[i]), "energy": float(energies[i]),
             "bitstring": ising.spins_to_bitstring(spins_of_index(int(i), n)),
             "assignment": ising.describe(spins_of_index(int(i), n), tickers, mode)} for i in order]


def classical_at_budget(model, e_min, tolerance, budget, seeds):
    rows = {}
    rng = np.random.default_rng(2026)
    rows["random sampling"] = float(np.mean([fair_budget.random_within_budget(model, budget, rng)[0] <= e_min + tolerance for _ in range(seeds)]))
    rng = np.random.default_rng(2027)
    greedy = [fair_budget.greedy_within_budget(model, budget, rng) for _ in range(seeds)]
    rows["greedy descent (restarted)"] = float(np.mean([g[0] <= e_min + tolerance for g in greedy]))
    rows["greedy mean restarts"] = float(np.mean([g[2] for g in greedy]))
    rng = np.random.default_rng(2028)
    rows["simulated annealing (one run)"] = float(np.mean([fair_budget.annealing_within_budget(model, budget, rng)[0] <= e_min + tolerance for _ in range(seeds)]))
    return rows


def run(n=29, eta=1.0, lr_p=LR_P_VALUES, dense_p=DENSE_P_VALUES, budget=BUDGET, seeds=SEEDS, log=print):
    started = time.time()
    parameters = json.loads(t4.PARAMETERS_PATH.read_text())
    task3 = json.loads((ROOT / "results" / "stocks_task3_classical.json").read_text())
    ctx = t3.load_context()
    thin_setup = t3.size_setup(ctx, n)
    dense_setup = t3.size_setup(ctx, n, 1.0)
    tickers = thin_setup["tickers"]
    out = {"n": n, "eta": eta, "budget": budget, "repetitions": seeds, "tickers": tickers,
           "keep_fraction_thinned": thin_setup["keep_fraction"], "kept_links": thin_setup["mask_report"]["kept_edges"],
           "total_links": thin_setup["mask_report"]["total_edges"],
           "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "note": "exact statevector (complex64) of LR-QAOA with the ramp tuned at 12 assets (not re-tuned); exact optimum by "
                   "enumerating all 2^n energies; classical heuristics under the same 1,000-evaluation budget; noiseless",
           "modes": {}}
    for mode in ising.MODES:
        t0 = time.time()
        dense_model = t3.build_model(dense_setup, mode, eta, False)
        thin_model = t3.build_model(thin_setup, mode, eta, False)
        e_dense = energies_all_states(dense_model)
        e_thin = energies_all_states(thin_model)
        opt_dense, opt_thin = int(np.argmin(e_dense)), int(np.argmin(e_thin))
        e_min, e_max = float(e_dense[opt_dense]), float(e_dense.max())
        tolerance = 1e-6 * (e_max - e_min)   # float32 energies: ties within 1e-6 of the range count as optimal
        n_opt = int(np.sum(e_dense <= e_min + tolerance))
        best_found_task3 = next((r["dense_best_found"] for r in task3["results"]
                                 if r["n"] == n and r["mode"] == mode and r["eta"] == eta and not r["balance"]), None)
        block = {
            "exact_dense": {"e_min": e_min, "e_max": e_max, "n_optimal": n_opt, "states": int(1 << n),
                            "bitstring": ising.spins_to_bitstring(spins_of_index(opt_dense, n)),
                            "assignment": ising.describe(spins_of_index(opt_dense, n), tickers, mode)},
            "exact_thinned_surrogate": {"e_min_surrogate": float(e_thin[opt_thin]), "dense_score_of_surrogate_optimum": float(e_dense[opt_thin]),
                                        "same_state_as_dense_optimum": bool(opt_thin == opt_dense)},
            "task3_best_found_dense": best_found_task3,
            "task3_found_exact_optimum": None if best_found_task3 is None else bool(best_found_task3 <= e_min + tolerance),
            "uniform": {"p_optimum": 1.0 / (1 << n), "quality": (e_max - float(np.mean(e_dense, dtype=np.float64))) / (e_max - e_min)},
            "circuits": {},
        }
        log(f"{mode}: energies built in {time.time() - t0:.0f} s; exact optimum {e_min:+.5f} ({n_opt} state), "
            f"Task 3 best found {best_found_task3}, surrogate optimum is the dense optimum: {opt_thin == opt_dense}")
        lr = parameters["modes"][mode]["linear_ramp"]
        sims = {"thinned": LargeQaoa(thin_model, e_thin), "dense": LargeQaoa(dense_model, e_dense)}
        for kind, p_values in (("thinned", lr_p), ("dense", dense_p)):
            for p in p_values:
                t1 = time.time()
                gammas, betas = qaoa.lr_schedule(p, lr[str(p)]["best_dgamma"], lr[str(p)]["best_dbeta"])
                probabilities = sims[kind].probabilities(gammas, betas)
                m = metrics(probabilities, e_dense, e_min, e_max, tolerance, budget)
                m["hit_at_budget_sampled"] = sampled_hits(probabilities, e_dense, e_min, tolerance, budget, seeds, np.random.default_rng(2029))
                m["surrogate_expectation"] = dot64(probabilities, e_thin) / m["probability_sum"] if kind == "thinned" else None
                m["top_states"] = top_states(probabilities, e_dense, TOP_STATES, n, tickers, mode)
                m.update({"p": p, "dgamma": lr[str(p)]["best_dgamma"], "dbeta": lr[str(p)]["best_dbeta"], "circuit_model": kind,
                          "seconds": round(time.time() - t1, 1)})
                del probabilities
                block["circuits"][f"{kind}_p{p}"] = m
                log(f"  {mode} LR-QAOA on the {kind} model, p={p}: quality {m['quality']:.3f}, P(opt)/shot {m['p_optimum']:.2e}, "
                    f"hit@{budget} exact {m['p_optimum_in_shots']:.3f} sampled {m['hit_at_budget_sampled']:.3f}, "
                    f"most probable = optimum: {m['most_probable_state'] == opt_dense} ({m['seconds']} s)")
        del sims, e_thin
        t2 = time.time()
        block["classical_same_budget"] = classical_at_budget(dense_model, e_min, tolerance, budget, seeds)
        block["classical_task3_hit_rates"] = {name: v["hit_rate_per_try"] for name, v in next(
            r for r in task3["results"] if r["n"] == n and r["mode"] == mode and r["eta"] == eta and not r["balance"])["solvers"].items()}
        log(f"  {mode} classical at the same budget: {block['classical_same_budget']} ({time.time() - t2:.0f} s)")
        del e_dense
        out["modes"][mode] = block
    out["seconds"] = round(time.time() - started, 1)
    return out


def main(n=29):
    out = run(n)
    path = ROOT / "results" / f"stocks_qaoa_n{n}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("saved:", path.relative_to(ROOT), "in", out["seconds"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 29))
