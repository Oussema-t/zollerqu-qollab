# ZollerQu — heterogeneous mesh partitioning with QAOA and varQITE

**This effort is supported by Qollab & IonQ.**

**What is new.** Quantum graph partitioning has so far used *balanced* halves (IonQ with Ansys/Synopsys in LS-DYNA). Real clusters mix CPUs and GPUs, so a mesh must be split with an *unequal* target share α that matches device speed. We study this heterogeneous split and find that it breaks a published varQITE ansatz: a qubit that only ever acts as the control of the ansatz's two-qubit gates keeps a 50/50 marginal for every angle, so the probability of a unique optimum is capped at 1/2. Equal shares hide this because their optimum comes as a mirror pair. We prove it (report, Lemma 1), measure it on every trained circuit (27 of 27 per scenario) and show a one-layer change (A+) that removes the cap.

**Setup.** A simulation mesh (a classical PDE problem) is split between two devices: each device gets its share of the work (α = 0.5 for equal devices, α = 0.8 for a CPU + a GPU assumed 4× faster) and the border between the parts stays short. The split is an Ising problem (one qubit per mesh region), solved by QAOA variants and varQITE, compared with classical methods, and graded against the exact optimum, with the **total evaluations** of every method counted (training circuits × 1,000 shots + the final 1,000 shots for quantum methods).

**Short result.** varQITE with the paper's ansatz is excellent for equal devices (near-optimal split on 78 % of shots) and fails for α = 0.8, as the lemma predicts. QAOA beats random sampling at both shares and is level with local search at 12 qubits, but spends about 197,000 total evaluations against local search's 1,000; from 16 qubits on, classical heuristics (especially spectral bisection) win clearly. **No quantum advantage is claimed.** All numbers come from noiseless local simulation or stated models; the single IonQ job is prepared but has not been run yet.

The report is [`report/main.pdf`](report/main.pdf) (4 pages).

## Contents

| Path | What it is |
|---|---|
| `mesh_idea_ZollerQu.ipynb` | The notebook, run top to bottom with outputs (Tasks 01–04b, small mode: 12 qubits, seed 7) |
| `results/` | Plots, CSV and JSON summaries of the notebook run, and resumable checkpoints |
| `results_large_hpc/` | Large mode (12, 16, 20, 24 qubits; seeds 7, 8, 9) run on a Hetzner `ccx33` node with `scripts/hpc_driver.py` |
| `data/` | Frozen meshes (`frozen_meshes/*.npz`) and region assignments (`frozen_regions.npz`), so every machine uses bit-identical inputs |
| `qollab/` | The Qollab project for the single IonQ job: `main.py`, `mesh_ionq.py`, `mesh_batch.json` (12 circuits), `results_ionq/` for the recorded run |
| `report/` | The LaTeX report (`main.tex`, `main.pdf`), its figure, and `make_report_data.py`, which builds every number and figure in the report from the result files |
| `scripts/run_ionq_batch.py` | Runs the same IonQ batch through the IonQ cloud (Forte 1 noise model) with an API key in `IONQ_API_KEY` |
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
| 05 | Export of the single IonQ batch for Qollab (p = 0 control, LR-QAOA, QAOA + COBYLA, final varQITE circuits) |

## How to run

```bash
python3 -m venv venv && venv/bin/pip install numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2 qiskit==2.5.2 nbformat nbclient ipykernel
venv/bin/jupyter nbconvert --to notebook --execute mesh_idea_ZollerQu.ipynb   # or open it in Jupyter / VS Code
```

The checkpoints in `results/` let Task 04 and 04b resume. They are reused only when the settings, the problem data and the code are unchanged.

## Run the IonQ job (one job, 12 circuits)

- **Qollab:** create a project, upload the four files of `qollab/` (`main.py`, `mesh_ionq.py`, `mesh_batch.json`, `requirements.txt`), press **Run** and choose **IonQ Forte 1 (29q) noise model** (never the paid "Quantum Computer" entries). The console prints the measured results next to the noiseless predictions and a `RAW_RESULT_JSON` line with all counts; save that line as `qollab/results_ionq/<date>.json` so later runs show it as the recorded result.
- **Locally:** `pip install qiskit-ionq`, then `IONQ_API_KEY=... python scripts/run_ionq_batch.py`.

## Honesty notes

- Everything is noiseless local simulation (statevector), checked against Qiskit circuits; the noise numbers are a crude depolarising estimate, not an IonQ result.
- varQITE was trained on a noiseless local simulator with exact expectations. Only the final trained circuits are meant for the IonQ batch, together with the p = 0 control and the QAOA circuits.
- Runtimes, electricity and device speeds are modelled assumptions, not measurements.
- Schedules, step sizes and annealing temperatures were tuned on separate calibration meshes, never on scored instances.
- 30 deliberately injected bugs were all caught by the notebook's checks (mutation testing).
- The large mode ran on Linux (x86-64) before the meshes were frozen: there, cases 1, 2, 4, 5 had slightly different meshes than on the Mac (floating-point details), so its case-2 instances are equivalent but not identical to the notebook's (see `results_large_hpc/fingerprints_*.json`). Since Tasks 01d / 02b, meshes and regions are data files written on the Mac, so every machine now uses identical inputs.

## Credits

Qiskit, NumPy, SciPy, Matplotlib. Methods: LR-QAOA (Montañez-Barrera & Michielsen, npj Quantum Inf. 2025), CVaR (Barkoutsos et al., Quantum 2020), warm-start QAOA (Egger, Mareček & Woerner, Quantum 2021), SPSA (Spall 1998), Ising partitioning (Lucas 2014), varQITE (Morris et al., arXiv:2404.16135). Prior art: IonQ + Ansys (arXiv:2503.13128), IonQ + Synopsys (arXiv:2603.15515). varQITE was implemented from the paper and the team's description; no code from other notebooks was copied. AI assistants (Claude Code, ChatGPT) were used for code and text; every number is produced by the code in this folder. Licence: MIT (repository root).
