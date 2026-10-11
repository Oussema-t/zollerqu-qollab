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

## Many devices (option C) — `multidevice/`

| File | What it is |
|---|---|
| `stage_a_device_choice.json` | A. Which devices of a synthetic CPU/GPU pool to use (one qubit per device, 8–10 devices), per goal (fastest / overnight / lowest energy): every LR-QAOA and optimised-angle setting for p = 1–5 (angles, calibration score on 10 calibration pools, P(best set) on each of 5 test pools, random baseline), the cost-circuit size (general diagonal, CNOT per layer) and the check that a 2-local approximation loses the optimum |
| `stage_b_recursive_split.json` | B. Mesh "2 main" (24 regions) split among the chosen devices by recursive bisection, per goal and method (exact, QAOA, local search, simulated annealing, spectral bisection, random): every split (size, target share, lambda, gap, optimal or not, QAOA P(opt)), the region → device assignment, shares, memory overload, cut, modelled hours and kWh, and the plan's ideal values |
| `REPORT.md`, `figures/` | Tables and the two figures (device choice vs depth; hours per method) |

Made by `scripts/multidevice_c.py`; report by `scripts/make_multidevice_report.py`; shown again by notebook Task 07.

## QAOA tuning — `qaoa_tuning/`

| File | What it is |
|---|---|
| `tuning_n12.json` | 12 qubits, Task 04 calibration set (4 meshes): every setting per scenario and depth p = 1–8 (LR energy / success, all 2p angles with 5 optimisers, warm start), its angles, calibration score, evaluations used and test metrics on the 9 test meshes (mean and per mesh). Kept because it shows the over-fitting |
| `tuning_n12_large_calibration.json` | The same with 18 calibration meshes (all 9 cases × seeds 101 / 102): the main 12-qubit result. Run on the HPC server as 14 parts (scenario × depth), merged by `qaoa_tuning.py large merge`; the parts are in `hpc_parts/` |
| `frozen_regions_calibration*.npz` | Region assignments of the calibration meshes at 12, 16 and 20 qubits (computed once on the Mac; the notebook's `data/frozen_regions.npz` is never modified) |
| `scaling/n{16,20}_<scenario>_p<p>.json` | 16 and 20 qubits: the best 12-qubit angles transferred unchanged and re-optimised at n on calibration meshes there; test metrics on cases 2 and 9 × mesh seeds 7–9, the classical baselines on the same meshes, and the measured circuit size |
| `hpc/hpc_n24/`, `hpc/hpc_n29/` | 24 and 29 qubits: the settings frozen from n = 20 (`../scripts/q29_choice_tuned.json`) run by `scripts/run_29_qubits.py`, same output format as `hpc_n29/` (metrics JSON + raw arrays per instance) |
| `logs/` | Server logs of the pipeline and the Mac cross-check logs (the same 12-qubit rows reproduced to 4 digits on macOS / ARM and Linux / x86) |
| `REPORT.md`, `summary.csv`, `figures/` | The tables, a flat CSV (one row per size, scenario, depth and setting) and the figures |

Made by `scripts/qaoa_tuning.py`, `scripts/qaoa_scaling.py` and `scripts/run_29_qubits.py` (pipeline: `scripts/hpc/launch_tuning.sh` → `scripts/hpc/run_tuning_pipeline.sh`); report by `scripts/make_qaoa_tuning_report.py`; shown again by notebook Task 08. Quiver is not installed in this project's environments, so it is not compared.

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

`../qollab/main.py` is the self-contained Qollab program (one circuit per job; the circuits come from `../qollab/selected_circuits.json`). `../qollab/mesh_batch.json` is the earlier 12-circuit batch for the IonQ cloud. `../qollab/results_ionq/` holds the recorded Qollab run on the IonQ Forte 1 noise-model simulator (`2026-10-11_varqite_zy_equal_ionq_simulator.json`: all 1,000 shots' counts, re-graded against the exact optimum; `…_qollab_console.html`: the console export).
