# ZollerQu — heterogeneous mesh partitioning with QAOA and varQITE

**This effort is supported by Qollab & IonQ.**

**What is new.** Quantum graph partitioning has so far used *balanced* halves (IonQ with Ansys/Synopsys in LS-DYNA). Real clusters mix CPUs and GPUs, so a mesh must be split with an *unequal* target share α that matches device speed. We study this heterogeneous split and find that it breaks a published varQITE ansatz: a qubit that only ever acts as the control of the ansatz's two-qubit gates keeps a 50/50 marginal for every angle, so the probability of a unique optimum is capped at 1/2. Equal shares hide this because their optimum comes as a mirror pair. We prove it (report, Lemma 1), measure it on every trained circuit (27 of 27 per scenario) and show a one-layer change (A+) that removes the cap.

**Setup.** A simulation mesh (a classical PDE problem) is split between two devices: each device gets its share of the work (α = 0.5 for equal devices, α = 0.8 for a CPU + a GPU assumed 4× faster) and the border between the parts stays short. The split is an Ising problem (one qubit per mesh region), solved by QAOA variants and varQITE, compared with classical methods, and graded against the exact optimum, with the **total evaluations** of every method counted (training circuits × 1,000 shots + the final 1,000 shots for quantum methods).

**Short result.** varQITE with the paper's ansatz is excellent for equal devices (near-optimal split on 78 % of shots) and fails for α = 0.8, as the lemma predicts. QAOA beats random sampling at both shares and is level with local search at 12 qubits, but spends about 197,000 total evaluations against local search's 1,000; from 16 qubits on, classical heuristics (especially spectral bisection) win clearly.

**29 qubits on HPC.** We took the best QAOA settings found at 20 qubits and re-tested them, without further tuning, at 29 qubits (the size of IonQ's Forte 1 simulator): 12 instances, every method graded against the exact optimum over all 536,870,912 splits.
- QAOA still beats uniform guessing per shot (about 1,200× for equal devices and 2,300× for CPU+GPU), so the settings transfer.
- For CPU+GPU, a biased coin that sends each region to the GPU with probability α does better per shot (QAOA reaches 0.7× of it).
- 1,000 shots contain a near-optimal split with probability 0.7 % (QAOA), against 10 % / 43 % for simulated annealing (equal / CPU+GPU), 5 % / 45 % for local search and 50 % / 100 % for spectral bisection.

**No quantum advantage is claimed.** All numbers come from noiseless local simulation or stated models; the single IonQ job is prepared but has not been run yet.

The report is [`report/main.pdf`](report/main.pdf) (6 pages). Figure 1 shows all nine benchmark objects.

## Contents

| Path | What it is |
|---|---|
| `mesh_idea_ZollerQu.ipynb` | The notebook, run top to bottom with outputs (Tasks 01–06; small mode: 12 qubits, seed 7; Tasks 04d and 06 show the saved HPC results) |
| `results/` | Every result as a file: plots, CSV/JSON summaries, resumable checkpoints. **[`results/README.md`](results/README.md) lists what each file shows and which cell or script shows it again** |
| `results/hpc_n29/` | 29-qubit results: one JSON (metrics) and one `.raw.npz` per instance (the instance, all shots, the best cost of every classical repeat, cost distributions, region marginals, most likely splits), `summary_q29.csv`, `rerun_check.json` (a second run reproduced every metric exactly), server logs |
| `results/audit/` | Evidence behind the honesty claims: mutation testing (`mutation_testing/SUMMARY.md`: 30 of 30 injected bugs caught), fairness and physics reviews, validation logs, an independent reproduction |
| `results/hpc_validation/` | The 29-qubit simulator checked against the notebook's exact simulator at 12, 16 and 20 qubits |
| `results_large_hpc/` | Large mode (12, 16, 20, 24 qubits; seeds 7, 8, 9) run on a Hetzner `ccx33` node with `scripts/hpc_driver.py` |
| `data/` | Frozen meshes (`frozen_meshes/*.npz`) and region assignments (`frozen_regions.npz`), so every machine uses bit-identical inputs |
| `qollab/` | The Qollab project for the single IonQ job: `main.py`, `mesh_ionq.py`, `mesh_batch.json` (12 circuits), `results_ionq/` for the recorded run |
| `report/` | The LaTeX report (`main.tex`, `main.pdf`), its figures, and the scripts that build every number, table and figure from the result files: `make_report_data.py` (`numbers.tex`, `table_n12.tex`, `table_scaling.tex`, `table_ionq.tex`, `fig_results.pdf` + `fig_results_preview.png`, `scaling.json`), `make_mesh_figure.py` (`fig_meshes.pdf`, Fig. 1, the nine objects) and `make_q29_report.py` (`numbers_q29.tex` + `q29_numbers.json`, `table_q29.tex`, `fig_q29_split.pdf`, `results/hpc_n29/summary_q29.csv`, `rerun_check.json`). Rebuilding gives byte-identical files |
| `scripts/run_ionq_batch.py` | Runs the same IonQ batch through the IonQ cloud (Forte 1 noise model) with an API key in `IONQ_API_KEY` |
| `scripts/run_29_qubits.py` | 29-qubit benchmark: complex64 statevector, exact energies of all 2^29 splits, classical baselines; `test` mode compares with the notebook's simulator |
| `scripts/choose_29_qubit_settings.py` | Picks the 29-qubit QAOA settings from the n = 20 results only and writes `scripts/q29_choice.json` |
| `scripts/hpc/` | `launch_29_qubits.sh` (create a Hetzner server, upload, start) and `watch_and_delete.sh` (pull the results, then delete the server); no secrets inside |
| `docs/literature/` | Literature checks: structured reports with DOIs, sign conventions and quotes, plus numerical convention checks |
| `scripts/hpc_driver.py` | Runs the notebook's Task 04 / 04b cells on a server in parallel shards, merges them and finishes |
| `docs/` | The task prompt and handoff used to build the notebook |

## Notebook tasks

| Task | Content |
|---|---|
| Run settings | `RUN_MODE = "small"` (12 qubits) or `"large"` (12–24 qubits, three seeds) |
| 01, 01b, 01c, 01d | Meshes: an L-plate with a heat source, four benchmark plates, a five-step complexity ladder on one L-plate; frozen as data files |
| 02, 02b | Coarsening into connected regions (one region = one qubit); frozen as a data file |
| 03 | Weighted Ising objective (cut + balance + optional priority), exact reference by enumeration |
| 03b | Synthetic research-centre pool (CPUs/GPUs with speed, memory, power) and three researcher goals |
| 04 | Local noiseless QAOA benchmark: LR-QAOA, COBYLA / Nelder–Mead / Powell / L-BFGS-B / SPSA, CVaR, warm starts, symmetry reduction; classical baselines; noise estimate; modelled runtime and electricity |
| 04b | varQITE (Morris, Kaushik, Roetteler & Lotshaw, arXiv:2404.16135): paper ansatz and three variants, trained on exact expectations |
| 04c | The region graph before and after the cut: original graph, QAOA's split, varQITE's split, exact optimum |
| 04d | The saved large-mode HPC results (12–24 qubits), display only |
| 05 | Export of the single IonQ batch for Qollab (p = 0 control, LR-QAOA, QAOA + COBYLA, final varQITE circuits) |
| Code for Qollab | The self-contained `main.py` to paste into Qollab (one circuit per job); the cell runs it locally as a noiseless preview |
| 06 | 29 qubits on HPC: shows the saved results of `scripts/run_29_qubits.py` (best settings from n = 20 against the classical methods, exact optimum) |

## How to run

```bash
python3.12 -m venv venv && venv/bin/pip install numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2 qiskit==2.5.2 pandas nbformat nbclient ipykernel
venv/bin/jupyter nbconvert --to notebook --execute mesh_idea_ZollerQu.ipynb   # or open it in Jupyter / VS Code
```

29-qubit run (about 7 GB of memory per instance, 6.6 GB resident as seen with `ps` on the server; we used two 32 GB Hetzner servers, about 15 minutes):

```bash
python scripts/choose_29_qubit_settings.py            # settings from the n = 20 results
python scripts/run_29_qubits.py 29 7,8,9 4            # n, seeds, instances in parallel (Linux; on macOS use 1)
python scripts/run_29_qubits.py 16 7 1 test           # check against the notebook's simulator
python report/make_q29_report.py && cd report && SOURCE_DATE_EPOCH=1791676800 tectonic main.tex   # fixed date: byte-identical PDF
```

The checkpoints in `results/` let Task 04 and 04b resume. They are reused only when the settings, the problem data and the code are unchanged. The code check compares Python bytecode, so the checkpoints are reused when the notebook runs in Jupyter, VS Code or nbconvert with Python 3.12. A script that `exec()`s whole cells, or another Python version, recomputes everything (about 17 minutes in small mode); the recomputed results were identical.

## Run on Qollab (IonQ Forte 1 noise model)

Qollab's IonQ simulators accept **one circuit per job**. So `qollab/main.py` is self-contained, with the circuits, angles and exact optima embedded, and runs the circuit chosen in the Run dialog:

1. Create a Qollab project and paste `qollab/main.py`; it is the same as the notebook's last cell, "Code for Qollab".
2. Press **Run** and choose **IonQ Forte 1 (29q) noise model** in Select QPU. Never use the paid "Quantum Computer" entries.
3. Optional: set `circuit` in the Run dialog. The default is `varqite_zy_equal`.

| `circuit` | What it is | Noiseless P(optimum) | Crude noisy estimate | 2-qubit gates |
|---|---|---|---|---|
| `varqite_zy_equal` (default) | varQITE ZY, equal devices | 0.750 | 0.539 | 66 |
| `varqite_a_equal` | varQITE A, equal devices | 0.706 | 0.507 | 66 |
| `warm_recovery_p1_cpu_gpu` | warm start that returns the classical spectral split (a noise reference, not a quantum search) | 1.000 | 0.718 | 66 |
| `cvar_spsa_p2_cpu_gpu` | best pure-quantum circuit for CPU + GPU (CVaR-QAOA, p = 2) | 0.0058 | 0.0030 | 132 |
| `control_p0` | p = 0 control (uniform guessing), graded under both scenarios | 0.0005 / 0.0002 | same | 0 |
| `lemma_varqite_zy_cpu_gpu` | varQITE ZY trained for CPU + GPU: Lemma 1 predicts failure | ~0 | ~0 | 66 |

**What the program does**
- Re-checks the exact optimum by brute force.
- Grades every shot and prints the results next to the predictions.
- Ends with a `RAW COUNTS` line; save it as `qollab/results_ionq/<date>_<circuit>.json`.

**Recorded run (IonQ Forte 1 noise model on Qollab, 2026-10-11):** `varqite_zy_equal`, 1,000 shots. 27.8 % (±1.4 %) of the shots were the exact optimum, about 570× random guessing (0.05 %). The noiseless prediction was 75 % and our noise estimates 54–55 %, so the real noise is about twice as strong as we estimated. Counts and the Qollab console export are in `qollab/results_ionq/`.

**How the circuits were chosen** (from the existing 12-qubit results, no new training): `scripts/ionq_selection_extract.py` → `scripts/ionq_selection_evaluate.py` → `qollab/selected_circuits.json` → `scripts/make_qollab_main.py`. The full table of 104 candidates is in [`results/ionq_selection/candidates_table.md`](results/ionq_selection/candidates_table.md). It gives noiseless scores, two-qubit gates measured on IonQ Forte 1, the crude estimate, an approximate Aer noise check and samples to solution.

**Earlier versions**
- The multi-circuit version is kept in `qollab/legacy/main_batch.py`.
- With an IonQ API key, the 12-circuit batch can still be sent through the IonQ cloud: `scripts/run_ionq_batch.py`.

## Honesty notes

- Everything is noiseless local simulation (statevector), checked against Qiskit circuits; the noise numbers are a crude depolarising estimate, not an IonQ result.
- varQITE was trained on a noiseless local simulator with exact expectations. Only the final trained circuits are meant for the IonQ batch, together with the p = 0 control and the QAOA circuits.
- Runtimes, electricity and device speeds are modelled assumptions, not measurements.
- Schedules, step sizes and annealing temperatures were tuned on separate calibration meshes, never on scored instances.
- 30 deliberately injected bugs were all caught by the notebook's checks (mutation testing; per-mutant table in `results/audit/mutation_testing/SUMMARY.md`).
- At 29 qubits the QAOA angles are transferred from 20 qubits, never re-optimised or tuned on the 29-qubit instances. Their training cost (at n = 20) is not included in the 1,000 evaluations of the 29-qubit table. The simulation is noiseless: the depolarising estimate for the p = 5 circuit (2,030 two-qubit gates) is F ≈ 4 × 10⁻⁵, so on hardware it would sample essentially uniformly. varQITE was not run at 29 qubits.
- The large mode ran on Linux (x86-64) before the meshes were frozen: there, cases 1, 2, 4, 5 had slightly different meshes than on the Mac (floating-point details), so its case-2 instances are equivalent but not identical to the notebook's (see `results_large_hpc/fingerprints_*.json`). Since Tasks 01d / 02b, meshes and regions are data files written on the Mac, so every machine now uses identical inputs.

## Credits

Qiskit, NumPy, SciPy, Matplotlib. Methods: LR-QAOA (Montañez-Barrera & Michielsen, npj Quantum Inf. 2025), CVaR (Barkoutsos et al., Quantum 2020), warm-start QAOA (Egger, Mareček & Woerner, Quantum 2021), SPSA (Spall 1998), Ising partitioning (Lucas 2014), varQITE (Morris et al., arXiv:2404.16135). Prior art: IonQ + Ansys (arXiv:2503.13128), IonQ + Synopsys (arXiv:2603.15515). varQITE was implemented from the paper and the team's description; no code from other notebooks was copied. AI assistants (Claude Code, ChatGPT) were used for code and text; every number is produced by the code in this folder. Licence: MIT (repository root).
