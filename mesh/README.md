# ZollerQu — heterogeneous mesh partitioning with QAOA and varQITE

**This effort is supported by Qollab & IonQ.**

A simulation mesh (a classical PDE problem) must be split between compute devices of different speed. Each device should get a share of the work that matches its speed (target share α), and the border between the parts should be short (little communication). We write the two-device split as an Ising problem (one qubit per mesh region) and benchmark quantum samplers against classical methods, grading every method against the exact optimum. The PDE solver stays classical; the quantum part only decides the allocation.

**Short result:** QAOA and varQITE beat random guessing, and QAOA is level with classical local search at 12 qubits. IonQ's varQITE is excellent for equal devices (near-optimal split on 78 % of shots) but fails for unequal shares, because of a structural limit of its ansatz that we identify. From 16 qubits on, classical heuristics (especially spectral bisection) win clearly. **No quantum advantage is claimed.** All numbers come from noiseless local simulation or stated models; nothing here has run on IonQ hardware yet.

The two-page report is [`report/main.pdf`](report/main.pdf).

## Contents

| Path | What it is |
|---|---|
| `mesh_idea_ZollerQu.ipynb` | The notebook, run top to bottom with outputs (Tasks 01–04b, small mode: 12 qubits, seed 7) |
| `results/` | Plots, CSV and JSON summaries of the notebook run, and resumable checkpoints |
| `results_large_hpc/` | Large mode (12, 16, 20, 24 qubits; seeds 7, 8, 9) run on a Hetzner `ccx33` node with `scripts/hpc_driver.py` |
| `report/` | The LaTeX report (`main.tex`, `main.pdf`), its figure, and `make_report_data.py`, which builds every number and figure in the report from the result files |
| `scripts/hpc_driver.py` | Runs the notebook's Task 04 / 04b cells on a server in parallel shards, merges them and finishes |
| `docs/` | The task prompt and handoff used to build the notebook |

## Notebook tasks

| Task | Content |
|---|---|
| Run settings | `RUN_MODE = "small"` (12 qubits) or `"large"` (12–24 qubits, three seeds) |
| 01, 01b, 01c | Meshes: an L-plate with a heat source, four benchmark plates, and a five-step complexity ladder on one L-plate |
| 02 | Coarsening into connected regions (one region = one qubit) |
| 03 | Weighted Ising objective (cut + balance + optional priority), exact reference by enumeration |
| 03b | Synthetic research-centre pool (CPUs/GPUs with speed, memory, power) and three researcher goals |
| 04 | Local noiseless QAOA benchmark: LR-QAOA, COBYLA / Nelder–Mead / Powell / L-BFGS-B / SPSA, CVaR, warm starts, symmetry reduction; classical baselines; noise estimate; modelled runtime and electricity |
| 04b | varQITE (Morris, Kaushik, Roetteler & Lotshaw, arXiv:2404.16135): paper ansatz and three variants, trained on exact expectations |

## How to run

```bash
python3 -m venv venv && venv/bin/pip install numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2 qiskit==2.5.2 nbformat nbclient ipykernel
venv/bin/jupyter nbconvert --to notebook --execute mesh_idea_ZollerQu.ipynb   # or open it in Jupyter / VS Code
```

The checkpoints in `results/` let Task 04 and 04b resume. They are reused only when the settings, the problem data and the code are unchanged.

## Honesty notes

- Everything is noiseless local simulation (statevector), checked against Qiskit circuits; the noise numbers are a crude depolarising estimate, not an IonQ result.
- varQITE was trained on a noiseless local simulator with exact expectations. Only the final trained circuits are meant for the IonQ batch, together with the p = 0 control and the QAOA circuits.
- Runtimes, electricity and device speeds are modelled assumptions, not measurements.
- Schedules, step sizes and annealing temperatures were tuned on separate calibration meshes, never on scored instances.
- 30 deliberately injected bugs were all caught by the notebook's checks (mutation testing).
- The notebook ran on macOS (arm64); the large mode ran on Linux (x86-64). Four of the nine meshes (cases 1, 2, 4, 5) differ at bit level between the two builds (floating-point differences in mesh generation), so the server's case-2 instances are equivalent but not identical to the notebook's; cases 3 and 6–9 are bit-identical (see `results_large_hpc/fingerprints_*.json`).

## Credits

Qiskit, NumPy, SciPy, Matplotlib. Methods: LR-QAOA (Montañez-Barrera & Michielsen, npj Quantum Inf. 2025), CVaR (Barkoutsos et al., Quantum 2020), warm-start QAOA (Egger, Mareček & Woerner, Quantum 2021), SPSA (Spall 1998), Ising partitioning (Lucas 2014), varQITE (Morris et al., arXiv:2404.16135). Prior art: IonQ + Ansys (arXiv:2503.13128), IonQ + Synopsys (arXiv:2603.15515). varQITE was implemented from the paper and the team's description; no code from other notebooks was copied. AI assistants (Claude Code, ChatGPT) were used for code and text; every number is produced by the code in this folder. Licence: MIT (repository root).
