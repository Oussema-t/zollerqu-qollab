# Speaker notes: Portfolio Selection with QAOA (ZollerQu, stocks project)

Spoken script for all 9 slides: **374 words** (about 2:46-2:53 at 130-135 words per minute; target 330-380 words).

Wording rules used: "noisy simulator" / "noise model" for the IonQ run, no advantage or speed-up claim, no wall-clock comparison.

## Slide 1: Portfolio Selection with QAOA

**Script (38 words):** Hi, we are team ZollerQu. Our question is simple: which stocks should a fund buy? We turn it into an Ising problem, one qubit per asset, and sample it with QAOA on the IonQ Forte 1 noise model.

**Alt text** (`slides/figures/hero_graph.png`): Network picture of the 12 assets: 12 dots on a circle, every pair joined by a line; thicker lines mean the two assets move more strongly together.

## Slide 2: Which assets should a fund buy?

**Script (39 words):** Banks and funds build portfolios every day. They want low risk and a good return. Risk comes from assets that fall together, so the choices interact. Twelve assets give 4,096 possible portfolios; twenty-nine give more than five hundred million.

**Source line on the slide:** Data: weekly prices of 12 US stocks and funds, 2006-2014 (Kaggle Huge Stock Market Dataset, CC0)

## Slide 3: One qubit per asset, one Ising energy

**Script (48 words):** Each asset becomes one qubit: zero means buy, one means skip. Risk minus return becomes an Ising energy with one coupling for every pair of assets, 66 couplings. For twelve assets we can check every portfolio, so we know the exact best one: Apple plus long-term Treasury bonds.

**Alt text** (`slides/figures/formula_score.png`): Formula: J of x equals x transposed C x over n squared V ref (the risk), minus eta times mu transposed x over n sigma ref (the return).

**Alt text** (`slides/figures/formula_ising.png`): Formula: E of s equals the sum over pairs of J i j times s i s j, plus the sum of h i times s i, plus a constant.

**Alt text** (`slides/figures/graph_optimum.png`): Network of the 12 assets with every pair linked. Two filled dots mark the exact best portfolio: buy AAPL and TLT; the other ten are skipped.

**Source line on the slide:** Exact best portfolio found by checking all 4,096 portfolios (classical); risk and return from 2006-2014 data

## Slide 4: QAOA circuits, trained offline, one IonQ job

**Script (47 words):** The circuit is QAOA: Hadamard gates, a cost layer with one ZZ gate per pair, then a mixer, repeated three times. We compared 58 circuits from seven method families, all trained offline on a classical simulator. The best one, CVaR-QAOA, went to IonQ as a single job.

**Alt text** (`slides/figures/circuit_sketch.png`): Circuit sketch: four of the twelve qubit lines. Each starts with an H gate, passes a cost layer block (one ZZ gate per pair, 66, and one Z rotation per qubit) and a mixer block (RX on every qubit), repeated three times, then is measured; 1,000 shots give 1,000 portfolios.

**Source line on the slide:** 58 circuits and a random-guess control scored by noiseless simulation (server), 12 qubits

## Slide 5: All-to-all links: 4.3× fewer two-qubit gates

**Script (42 words):** Why IonQ? Every pair of assets is coupled, and trapped ions link every pair of qubits directly. One layer needs 66 two-qubit gates. The same layer routed on a heavy-hex grid needs 285: 4.3 times more, and 6.5 times at twenty-nine qubits.

**Alt text** (`slides/figures/gate_counts.png`): Column chart of two-qubit gates in one QAOA layer. At 12 qubits: 66 on all-to-all against 285 on a heavy-hex grid. At 29 qubits: 244 against 1,576.

**Source line on the slide:** Gate counts from transpiling only (Qiskit 2.5.2, level 1); heavy-hex = median of 5 routing seeds / on a generic heavy-hex map. No circuit was run on a heavy-hex device.

## Slide 6: IonQ noise model: 29× random guessing

**Script (51 words):** The main result, on the IonQ Forte 1 noise model: 7 of 1,000 shots were the exact best portfolio, 29 times more than random guessing. The noiseless simulation gives 58, so noise costs a factor of 8. Greedy search, a classical method, is in the same range as our noisy run.

**Alt text** (`slides/figures/main_result.png`): Dot chart on a log scale: evaluations until the best portfolio is first found, median, fewer is better. Noiseless simulation of the same circuit 12; greedy descent 73; IonQ Forte 1 noise model 99 with a 95 % interval from 48 to 246; simulated annealing 1,516; random guessing 2,839.

**Source line on the slide:** IonQ Forte 1 noise model (Qollab): 1 job, 1,000 shots, 11 Oct 2026 · noiseless simulation / Classical: median of 50 runs. Sampler medians come from the hit rate per shot. Training not counted.

## Slide 7: Up to 29 qubits, classical heuristics win

**Script (47 words):** In noiseless simulation up to twenty-nine qubits the circuit stays far above random guessing, but classical heuristics win: they find the best portfolio in 61 percent of runs, the circuit in 7 percent. And our circuit first needed about 4,900 training circuits. We claim no quantum advantage.

**Alt text** (`slides/figures/scaling.png`): Line chart: share of runs that find the best portfolio within 1,000 evaluations, against the number of assets. QAOA in noiseless simulation: 100 % at 12, 89 % at 16, 66 % at 20 and 6.6 % at 29 qubits. Classical methods: 100 % at 12 and 61 % at 29. Random guessing: 22 % at 12 and about 0 % from 20 on.

**Source line on the slide:** 12-20 qubits: server simulation, 10 Oct 2026 · 29 qubits: server simulation of all 536,870,912 states, / 11 Oct 2026 · classical: 200 runs · the same 1,000 evaluations for every method

## Slide 8: Try it on Qollab: one click, one IonQ job

**Cut to the live Qollab demo at the marker below, then come back to the slide.**

**Script (49 words):** [CUT TO LIVE QOLLAB DEMO] Here is the project on Qollab, with the code in main.py. We press Run, select the IonQ Forte 1 noise model, and one job returns counts and plots. [BACK TO SLIDE] Our limits: a noisy simulator, one job, and strong classical baselines. Next: the other selected circuits, shorter circuits, and twenty-nine assets.

## Slide 9: Thank you for your attention

**Script (13 words):** Thank you for your attention. This effort is supported by Qollab and IonQ.

**Alt text** (`slides/figures/hero_graph.png`): The network picture of the 12 assets from the title slide: 12 dots on a circle, every pair joined by a line.

## Figures and the result files behind them

| Figure | Used on | Result files |
|---|---|---|
| `slides/figures/hero_graph.png` | slide 1 | data/kaggle_weekly_29.csv (training-window correlations, through stocks.task3_report) |
| `slides/figures/formula_score.png` | slide 3 | no data: the score definition in stocks/ising.py (mode B, buy or skip) |
| `slides/figures/formula_ising.png` | slide 3 | no data: the Ising form in stocks/ising.py |
| `slides/figures/graph_optimum.png` | slide 3 | data/kaggle_weekly_29.csv (training-window correlations, through stocks.task3_report); exact best portfolio from the Ising model in qollab/selected_circuits.json (all 4,096 states) |
| `slides/figures/circuit_sketch.png` | slide 4 | schematic; pair count from results/stocks_qaoa_tuning_n12.json (meta.kept_links), shots from results/ionq_B_cvar_p3_20261011-052116.json |
| `slides/figures/gate_counts.png` | slide 5 | results/stocks_qaoa_tuning_n12.json (circuits.n12 and circuits.large, routing_p1) |
| `slides/figures/main_result.png` | slide 6 | results/ionq_B_cvar_p3_20261011-052116.json (IonQ run), qollab/circuit_candidates.json (noiseless rate, classical medians), results/stocks_fair_budget_n12.json (random rate) |
| `slides/figures/scaling.png` | slide 7 | results/stocks_qaoa_tuning_n12.json (12, 16, 20 qubits), results/stocks_qaoa_n29.json (29 qubits), results/stocks_fair_budget_n12.json (classical and random at 12) |

## Open points

- No placeholder is left on the slides. Slide 8 is shown together with a live Qollab session (no screenshot).
- Links are not on the slides (Qollab project, Demo video): set LINKS in slides/build_deck.py to print them on the closing slide. The GitHub repository is private and is not linked.

## Not verified from the files

- The stocks run file records the backend only as 'ionq_simulator'. That the IonQ Forte 1 noise model was the simulator selected in the Qollab Run dialog is taken from the team, not from the file (the measured rate is far below the noiseless one, so a noise model was active).
- One IonQ job of 1,000 shots: no repeat run, so run-to-run variation on the noise model is not measured; the 95 % interval covers sampling error only.
- Heavy-hex gate counts come from Qiskit's router on a generic heavy-hex map (5 seeds); a better router or a real device layout could give other numbers.
- The medians for the three samplers on slide 6 are computed from the per-shot hit rate (geometric distribution), not measured by repeating runs.
- The PPTX was rendered with macOS Quick Look (slide by slide) and compared with the PDF; it was not opened in PowerPoint itself.

Rebuild everything with `./venv/bin/python slides/build_deck.py` (reads `results/` and `qollab/`; nothing is simulated).
