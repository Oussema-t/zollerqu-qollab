# ZollerQu: four quantum projects on IonQ

**This effort is supported by Qollab & IonQ.**

Team ZollerQu's work for the Qollab × IonQ Hackathon (9–11 October 2026, theme: Quantum Impact). The repository holds four independent projects, one per branch. Each one takes a real problem, solves it with Qiskit circuits, and compares the result with the exact answer and with classical methods.

**Team:** Oussama Turki and Tushar Pandey. **Licence:** MIT.

## Projects at a glance

| # | Project | Branch | Quantum method | Qubits | Headline result |
|---|---|---|---|---|---|
| 1 | [Portfolio selection](#1-portfolio-selection-with-qaoa) | [`stocks`](https://github.com/Oussema-t/zollerqu-qollab/tree/stocks) | QAOA (8 circuit types), Iceberg error detection | 12–29 | IonQ Forte 1 noise model: 7 of 1,000 shots are the exact best of 4,096 portfolios, 29× random guessing |
| 2 | [Mesh partitioning for CPU + GPU](#2-mesh-partitioning-for-cpu--gpu) | [`mesh`](https://github.com/Oussema-t/zollerqu-qollab/tree/mesh/mesh) | QAOA variants, varQITE | 12–29 | IonQ Forte 1 noise model: 27.8 % of shots are the exact best split; random guessing gives 0.049 % |
| 3 | [Store placement](#3-store-placement-by-quantum-minimum-finding) | [`placement`](https://github.com/Oussema-t/zollerqu-qollab/tree/placement) | Dürr–Høyer minimum finding (Grover search) | 16–22 | 27 of 27 simulated runs reach the exact minimum; placements scored grow like 7 × √C(m,4) |
| 4 | [Logical teleportation ("Unlocked")](#4-logical-teleportation-on-a-tiny-surface-code-unlocked) | [`unlocked`](https://github.com/Oussema-t/zollerqu-qollab/tree/unlocked) | Distance-3 surface code, logical CNOT, T-state teleportation | 9–27 | Four interactive lessons; a T state teleported between two logical qubits on 27 physical qubits |

### How to read the results

Every number below is read from a result file in the project's branch. Three kinds of runs appear, and each result says which one it is:

| Label | Meaning |
|---|---|
| **IonQ noise model** | IonQ's noisy simulator (Forte 1 noise model), run through Qollab |
| **Noiseless** | Exact simulation on a classical computer, with no errors |
| **Local Aer** | Qiskit Aer on a laptop: the saved runs of projects 3 and 4, which also run live on Qollab |

No result comes from a real quantum computer, and **no project claims a quantum advantage**. At these sizes classical methods solve every problem here quickly; where they beat the circuits, the results say so.

---

## 1. Portfolio selection with QAOA

Branch [`stocks`](https://github.com/Oussema-t/zollerqu-qollab/tree/stocks) · [paper (PDF)](https://github.com/Oussema-t/zollerqu-qollab/blob/stocks/paper/zollerqu_stock_qaoa.pdf) · [slides (PDF)](https://github.com/Oussema-t/zollerqu-qollab/blob/stocks/slides/stocks_ZollerQu.pdf) · [notebook](https://github.com/Oussema-t/zollerqu-qollab/blob/stocks/stocks/stock_idea_ZollerQu.ipynb) · [Qollab code](https://github.com/Oussema-t/zollerqu-qollab/blob/stocks/qollab/main.py)

<img src="https://raw.githubusercontent.com/Oussema-t/zollerqu-qollab/stocks/slides/figures/main_result.png" width="620" alt="Dot chart on a log scale: evaluations until the best portfolio is first found, median. Noiseless simulation 12, greedy search 73, IonQ Forte 1 noise model 99, simulated annealing 1,516, random guessing 2,839.">

**Problem.** Which assets should a fund buy? Funds want low risk and a good return, and risk comes from assets that fall together. Every extra asset doubles the number of portfolios: 4,096 for 12 assets, 536,870,912 for 29.

**Method.** One qubit per asset (0 = buy, 1 = skip). Risk minus return becomes an Ising energy with one coupling per pair of assets, built from weekly prices of 12 US stocks and funds (2006–2014). The exact best portfolio, found by checking all 4,096, is Apple plus US Treasury bonds. We compared 8 circuit types (58 circuits) in noiseless simulation and sent the best one, CVaR-QAOA with 3 layers and 198 two-qubit gates, to IonQ's noise model.

**Recorded results**

| Experiment | Run type | Result |
|---|---|---|
| CVaR-QAOA, 3 layers, 12 qubits, 1,000 shots | IonQ noise model | 7 of 1,000 shots are the exact best portfolio (95 % interval 3–14) |
| Same, against random guessing | IonQ noise model | 29× more often (random: 0.24 per 1,000) |
| Same circuit without noise | Noiseless | 58 per 1,000; noise costs a factor of 8 |
| Evaluations until the best portfolio is found (median) | mixed | greedy search 73 · IonQ run 99 · simulated annealing 1,516 · random 2,839 |
| 29 assets: best portfolio found within 1,000 shots | Noiseless | 3 layers 6.6 % of runs · 16 layers 90 % · 24 layers 99.6 % |
| 29 assets: classical search, 1,000 evaluations | classical | 61 % of runs (100 % with 10,000 evaluations) |
| Two-qubit gates per layer, all-to-all vs heavy-hex | transpiler count | 66 vs 285 at 12 qubits (4.3×); 6.5× at 29 qubits |
| Iceberg code: all 5,904 single faults injected | Noiseless | every state-preparation, check and readout fault caught; 80 % of gate faults caught or harmless |
| Iceberg code, 1 layer, 17 qubits, 1,000 shots | IonQ noise model | 246 shots pass every check (25 %); best portfolio in 2 of 246 kept shots, against 5 and 11 per 1,000 for plain QAOA |

**What it shows.** The circuit finds the best portfolio far more often than random guessing, but noise costs a factor of 8 and greedy search is in the same range. At 29 assets more layers help a lot, but only without noise: 16 layers need 6,496 two-qubit gates. Iceberg error detection does catch errors, flagging 3 in 4 shots, but the kept shots are not clearly better because the code doubles the gates (133 vs 66).

---

## 2. Mesh partitioning for CPU + GPU

Branch [`mesh`](https://github.com/Oussema-t/zollerqu-qollab/tree/mesh/mesh) · [project README](https://github.com/Oussema-t/zollerqu-qollab/blob/mesh/mesh/README.md) · [report (PDF)](https://github.com/Oussema-t/zollerqu-qollab/blob/mesh/mesh/report/main.pdf) · [slides (PDF)](https://github.com/Oussema-t/zollerqu-qollab/blob/mesh/mesh/slides/mesh_ZollerQu.pdf) · [notebook](https://github.com/Oussema-t/zollerqu-qollab/blob/mesh/mesh/mesh_idea_ZollerQu.ipynb) · [Qollab code](https://github.com/Oussema-t/zollerqu-qollab/blob/mesh/mesh/qollab/main.py)

<img src="https://raw.githubusercontent.com/Oussema-t/zollerqu-qollab/mesh/mesh/slides/figures/fig_ionq_result.png" width="480" alt="Bar chart on a log scale: share of shots on the exact best split. Random guessing 0.049 %, IonQ Forte 1 noise model 27.8 %, noiseless simulation 75.0 %. Our noise estimate was 54 %.">

**Problem.** Large simulations, such as crash tests, cut their mesh into parts that run on different devices. A good split gives each device its fair share of work and keeps the border short. Earlier quantum work used two equal halves. Real clusters mix CPUs and GPUs, so we study the unequal split: a GPU that is four times faster should get 80 % of the work.

**Method.** The mesh is grouped into regions, one qubit per region (0 = device A, 1 = device B). The Ising energy adds the cut border and a penalty when the work share misses its target. We benchmarked QAOA and its variants, and IonQ's variational imaginary-time evolution (varQITE), on nine finite-element meshes: 104 candidate circuits at 12 qubits, each graded against the exact optimum.

**Recorded results**

| Experiment | Run type | Result |
|---|---|---|
| varQITE, equal devices, 12 qubits, 66 two-qubit gates, 1,000 shots | IonQ noise model | 27.8 % of shots are the exact best split; a second run gave 27.6 % |
| Same, references | — | noiseless 75 % · random guessing 0.049 % |
| varQITE with the published ansatz, equal devices | Noiseless | near-optimal split on 78 % of shots |
| varQITE with the published ansatz, CPU + GPU | Noiseless | fails, as our lemma predicts (see below) |
| Tuned QAOA, 8 layers, 12 qubits: shots within 5 % of the optimum | Noiseless | 7.1 % (equal) and 19.5 % (CPU + GPU), up from 0.36 % and 1.08 % |
| 24 qubits: near-optimal split found within 1,000 shots | Noiseless | 99 % and 100 % of instances |
| 29 qubits: near-optimal split found within 1,000 shots | Noiseless | 88 % and 99.6 % of instances; 0.7 % before tuning |
| 29 qubits: classical methods, 1,000 evaluations | classical | spectral bisection 50 % / 100 % · annealing 10 % / 43 % · local search 5 % / 46 % |
| Two-qubit gates per layer, all-to-all vs heavy-hex | transpiler count | 66 vs 216 at 12 qubits (3.3×) |

Pairs of numbers are equal devices / CPU + GPU.

**A finding about varQITE.** In the published ansatz, a qubit that only acts as the control of the two-qubit gates always stays at 50/50. So for an unequal split, where the best answer is unique, the chance of measuring it can never exceed 1/2. We prove this in the report (Lemma 1), measure it on every trained circuit (27 of 27 per scenario), and show that one extra layer of single-qubit gates removes the limit.

**What it shows.** On IonQ's noise model the best split is the most frequent answer, about 570× more frequent than random guessing, although noise costs about twice what we estimated. The 29-qubit results are noiseless simulations of circuits with 3,248 two-qubit gates, well beyond today's noise, and a shot is not the same work as a classical evaluation. The project's checks caught 30 of 30 deliberately injected bugs.

---

## 3. Store placement by quantum minimum finding

[**Run it on Qollab**](https://qollab.xyz/u/tusharp/zollerqu-placement-store-place) · Branch [`placement`](https://github.com/Oussema-t/zollerqu-qollab/tree/placement) · [project README](https://github.com/Oussema-t/zollerqu-qollab/blob/placement/README.md) · [demo video](https://github.com/Oussema-t/zollerqu-qollab/blob/placement/docs/demo.mp4)

**Problem.** Where should k stores open among m candidate sites? The instance is real: 158 census block groups of Champaign County, Illinois (2020 census, 205,865 people) and road driving times. The cost mixes travel time, staffing, crowding and warehouse trips.

**Method.** Dürr–Høyer minimum finding: a Grover search with a threshold that tightens round by round. The search starts in a Dicke state, so it covers exactly the C(m,k) placements that the classical exact method scores. The oracle computes each placement's cost inside the circuit with a quantum adder; it never reads a list of costs. The project is an interactive Qollab page with a map, sliders for every cost weight, and three classical baselines (exact brute force, greedy, local search).

**Live on Qollab.** The page runs in the browser on every Qollab backend, including IonQ's simulators. Pressing "Run quantum search" sends one circuit: the last search of the Dürr–Høyer loop, whose threshold marks only the cheapest placement. At the default size (6 sites, 2 stores, 15 placements, 16 qubits) theory gives a 93.7 % chance of measuring the exact optimum after 3 Grover iterations. IonQ accepts one circuit per job, so the full loop and the theory check are shown from the saved run.

**Recorded results.** The saved run in the repository is from Local Aer. No IonQ result is saved in the branch, so none is quoted here.

| Experiment | Result |
|---|---|
| Theory check: 17 circuits, 1,000 shots each | measured Grover success probabilities within 5.3 percentage points of the theory curve |
| One full search, 8 sites, 4 stores (70 placements, 20 qubits) | reached the exact minimum after scoring 46 placements; brute force scores all 70 |
| One Grover iteration at 8 sites | 8,185 gates, of which 2,716 CNOTs |

Scaling study, one measurement per round:

| Sites m | Placements C(m,4) | Runs | Mean placements scored (min–max) | Reached the exact minimum |
|---|---|---|---|---|
| 6 | 15 | 20 | 27.5 (17–36) | 20 of 20 |
| 8 | 70 | 5 | 56 (51–61) | 5 of 5 |
| 10 | 210 | 2 | 104 (87–121) | 2 of 2 |

**What it shows.** The circuits behave as the theory predicts: the number of placements scored grows like 7 × √C(m,4) while C(m,4) grows 14-fold. At the smallest size the quantum loop scores more placements than brute force, as the theory says it must. The study is small (5 and 2 runs at the larger sizes), and nothing here measures wall-clock time: a laptop scores 70 placements in a millisecond. The proven quadratic speedup needs a fault-tolerant machine.

---

## 4. Logical teleportation on a tiny surface code ("Unlocked")

Branch [`unlocked`](https://github.com/Oussema-t/zollerqu-qollab/tree/unlocked) · [project README](https://github.com/Oussema-t/zollerqu-qollab/blob/unlocked/README.md) · [demo video](https://github.com/Oussema-t/zollerqu-qollab/blob/unlocked/docs/demo.mp4) · [narrated explainer](https://github.com/Oussema-t/zollerqu-qollab/blob/unlocked/docs/explainer.mp4)

**Problem.** Logical qubits are how quantum computers will outlive their own noise, but they are hard to learn from papers. This project is a teaching page for learners and teachers.

**Method.** Three logical qubits, each a 3 × 3 patch of the distance-3 rotated surface code (9 physical qubits, prepared by a 16-gate encoder). Each patch is measured once at the end; its parity checks point at a single error, which is then corrected in software. The page has four lessons, each shown next to the same experiment on bare qubits, with an animation at the logical and the physical level:

1. **One patch, one error.** Inject an error on any qubit and watch the checks find it.
2. **Logical CNOT.** Nine physical CNOTs between two patches make a logical Bell pair.
3. **A T state on a patch.** The resource state for the T gate, which the code cannot apply directly.
4. **Teleporting the T state** from one patch to another through a Bell measurement, on 27 qubits.

**Live on Qollab.** Lessons 1 to 3 (9, 18 and 9 qubits) run live on every backend. Lesson 4 needs 27 qubits, so it runs live on the 29-qubit IonQ Forte simulators; IonQ accepts one circuit per job, so there the page runs the teleportation circuit and shows the other lessons from the saved run.

**Recorded results.** The saved run in the repository is from Local Aer with a depolarizing noise model, 500 shots per circuit. No IonQ result is saved in the branch, so none is quoted here. The value is the share of shots reading logical 1.

| Lesson | Qubits | Ideal | Encoded, after correction | Bare qubits | Shots corrected |
|---|---|---|---|---|---|
| 1. One patch, Z / X readout | 9 | 0 % | 1.6 % / 2.0 % | 0 % / 0 % | 4.4 % / 4.2 % |
| 2. Logical Bell pair, ZZ / XX parity | 18 | 0 % | 2.4 % / 2.6 % | 0 % / 0.6 % | 14.2 % / 10.6 % |
| 3. T state, X / Y readout | 9 | 14.6 % | 15.6 % / 20.6 % | 14.0 % / 16.0 % | 3.2 % / none |
| 4. Teleported T state, X / Y readout | 27 | 14.6 % | 18.1 % / 25.9 % | 13.8 % / 16.0 % | 15.4 % / 22.2 % |

These numbers are computed from `data/recorded_aria1.json` with the project's own `qec/analyze.py`. The Y readout has no parity checks, so in lesson 3 nothing can be corrected in that basis.

**What it shows.** The checks find and correct single errors, and the T state survives teleportation between logical qubits. At this noise level the encoded results are mostly further from ideal than the bare qubits. That is expected for this design: the encoder is not fault tolerant and the checks are read only once, the same compromises the first logical-qubit experiments made. The goal is teaching how the pieces work, not showing a benefit from error correction.

---

## Repository layout

| Branch | Content |
|---|---|
| `main` | This overview and the first Qollab smoke test (a 2-qubit Bell circuit in `main.py`) |
| [`stocks`](https://github.com/Oussema-t/zollerqu-qollab/tree/stocks) | Project 1: Python package `stocks/`, notebook, results, paper, slides, Qollab code |
| [`mesh`](https://github.com/Oussema-t/zollerqu-qollab/tree/mesh/mesh) | Project 2: everything is in the `mesh/` folder (notebook, results, report, slides, Qollab code) |
| [`placement`](https://github.com/Oussema-t/zollerqu-qollab/tree/placement) | Project 3: Qollab web page (`index.html`, `main.js`, `ui/`), Python package `qopt/`, tests |
| [`unlocked`](https://github.com/Oussema-t/zollerqu-qollab/tree/unlocked) | Project 4: Qollab web page, Python package `qec/`, tests |

## Running a project

```bash
git clone https://github.com/Oussema-t/zollerqu-qollab.git
cd zollerqu-qollab
git checkout stocks        # or mesh, placement, unlocked
```

Each branch has its own instructions. On Qollab, open the project, press **Run** and choose a backend in **Select QPU**; the IonQ Forte 1 (29q) noise model is free and is the one used for the IonQ results above. Qollab's IonQ simulators accept one circuit per job, so every project sends a single circuit per run.

## Credits

- **Libraries:** Qiskit 2.5.2, NumPy, SciPy, Matplotlib; Qiskit Aer for local checks.
- **Data:** Kaggle "Huge Stock Market Dataset" (CC0) for project 1; US Census Bureau TIGERweb and OSRM / OpenStreetMap (ODbL) for project 3. Each branch lists its sources and licences.
- **Papers:** each project's README or report cites the methods it builds on, among them linear-ramp QAOA (Montañez-Barrera and Michielsen, 2025), varQITE (Morris et al., arXiv:2404.16135), the Iceberg code (Self, Benedetti and Amaro, 2024), Dürr and Høyer's minimum finding (1996) and the rotated surface code (Bombin and Martin-Delgado, 2007).
- **AI tools:** Claude Code was used in all four projects for planning, code and text, and ChatGPT in project 2. The team chose the problems and methods and reviewed the results.
- **Licence:** MIT, see [LICENSE](LICENSE).
