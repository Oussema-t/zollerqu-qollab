# Results index

Every result of the project is saved as a file, so it can be shown again without recomputing anything. Open the notebook and run it top to bottom (about 1.5 minutes: the checkpoints below are reused), or open the files directly.

## Small mode: 12 qubits, nine cases (notebook, laptop)

| File | What it shows | Shown again by |
|---|---|---|
| `mesh_seed7.png`, `mesh_cases_seed7.png`, `mesh_complexity_cases_seed7.png`, `mesh_objects_all_cases_seed7.png` | The meshes: the main L-plate, the four benchmark plates, the complexity ladder, and all nine objects on one page (paper Fig. 1) | Tasks 01, 01b, 01c; `report/make_mesh_figure.py` |
| `regions_n12_seed7.png` | The nine meshes split into 12 regions (paper Fig. 2) | Task 02 |
| `lambda_scan_n12_seed7.png`, `ising_difficulty_map.png`, `ising_splits_n12_seed7_*.png` | Balance-weight scan, problem difficulty, exact optimal splits | Task 03 |
| `hpc_pool_goals_seed7.png`, **`hpc_pool_goals_seed7.json`** | Synthetic CPU/GPU pool and the three researcher goals (fastest / overnight / lowest energy): devices, shares, hours, kWh (modelled) | Task 03b; the JSON feeds the paper's device-choice numbers |
| `task04_checkpoint_small.json.gz` | **Every QAOA and classical run** at 12 qubits (18 instances, 2,016 QAOA runs, angles, traces, outcomes). The notebook reuses it instead of recomputing | Task 04 |
| `task04_runs_small.csv`, `task04_summary_small.json` | The same as a flat table (one row per run) and the summary used by the paper. `measured_wall_seconds` records the latest notebook run and is the only field that changes on a re-run | Task 04; `report/make_report_data.py` |
| `task04_*_small.png` | QAOA vs classical, P(optimum), convergence, noise estimate, modelled runtime and electricity, difficulty heatmap | Task 04 |
| `varqite_checkpoint_small.json.gz`, `varqite_summary_small.json`, `varqite_*_small.png` | Every varQITE training run (angles, energy curves, final distributions) and its plots | Task 04b |
| `graph_cut_*_n12_seed7.png` | Region graph before and after the cut: QAOA, varQITE and exact optimum (paper Fig. 3) | Task 04c |

## Large mode: 12–24 qubits, three seeds (HPC run) — `../results_large_hpc/`

Checkpoint, CSV, summaries, plots, server logs, Mac/Linux mesh fingerprints and `figures/` (Task 02/03 plots of that run). **Shown again by Task 04d** (display only) and by `report/make_report_data.py` (paper Fig. 4 and Table 2).

## 29 qubits (HPC run) — `hpc_n29/`

| File | What it is |
|---|---|
| `<case>_seed<s>_<equal or cpu_gpu>.json` (12 files) | Metrics per instance: exact optimum, every classical baseline (hit rates, best costs, modelled outcome), every QAOA setting (P(optimum), P(within 5 %), mean cost, balance, sampled best, region marginals, most likely split, noise estimate), run times and the machine it ran on |
| `<…>.raw.npz` (12 files) | **Raw arrays**: the instance (J, h, constant, region work, borders, centroids, cell → region labels, optimal and near-optimal splits), all 1,000 sampled shots and their costs per QAOA setting, the best cost of each of the 100 runs of every classical method (1 run for the deterministic spectral bisection; the best split itself is kept for the best run, in the JSON), per-shot cost distributions over E / E_min for QAOA, uniform and target-share sampling, region marginals, and the 50 most likely splits. Load with `numpy.load(path)` |
| `summary_q29.csv` | One row per instance and method (the numbers behind paper Table 3) |
| `rerun_check.json` | A second run with the same seeds reproduced every metric exactly (largest difference 0) |
| `first_run/` | The first run's metric files and logs, kept for that comparison |
| `logs/` | Server logs and the server lifecycle (results pulled, servers deleted, none left) |

**Shown again by:** notebook Task 06 (tables, hit-rate plot, cost distributions and region marginals: `q29_hit_rates.png`, `q29_distributions_and_marginals.png`) and `report/make_q29_report.py` (paper Table 3, the 29-qubit numbers, `graph_cut_2_main_n29_seed7.png` = paper Fig. 5).

`hpc_validation/hpc_n{12,16,20}/` hold the same outputs for small sizes, checked against the notebook's exact simulator (largest probability difference 1.8e-8). `hpc_validation/logs/` has the run logs.

## Audit evidence — `audit/`

Mutation testing (30 of 30 injected bugs caught; see `audit/mutation_testing/SUMMARY.md`), fairness and physics reviews, validation logs and an independent reproduction. Details in `audit/README.md`. The literature checks are in `../docs/literature/`.

## IonQ circuit selection — `ionq_selection/`

| File | What it is |
|---|---|
| `candidates_table.md` / `.csv` | All 104 candidate circuits at 12 qubits (case 2, both scenarios). Per circuit: noiseless P(optimum) and P(within 5 %); two-qubit gates and depth measured by transpiling to IonQ Forte 1; the crude estimate; the Aer noise check (approximate noise model, not IonQ); shots to solution |
| `selection_summary.json` | The chosen circuits, the fallback, and the classical samples to solution (median evaluations for simulated annealing and local search) |
| `candidates.pkl`, `aer_cache.json` | Exact parameters and reference distributions of every candidate, and the cached Aer results |

Made by `scripts/ionq_selection_extract.py` and `scripts/ionq_selection_evaluate.py`; `scripts/check_selected_circuits.py` checks that `../qollab/selected_circuits.json` rebuilds every circuit exactly.

## IonQ

`../qollab/main.py` is the self-contained Qollab program (one circuit per job; the circuits come from `../qollab/selected_circuits.json`). `../qollab/mesh_batch.json` is the earlier 12-circuit batch for the IonQ cloud. `../qollab/results_ionq/` will hold the recorded IonQ run; it is empty until the job is run.
