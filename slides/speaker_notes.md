# Speaker notes: Portfolio Selection with QAOA (ZollerQu, stocks project)

Spoken script for all 10 slides: **377 words** (about 2:48-2:54 at 130-135 words per minute; target 330-380 words).

Wording rules used: "noisy simulator" / "noise model" for the IonQ run, no advantage or speed-up claim, no wall-clock comparison.

## Slide 1: Portfolio Selection with QAOA

**Script (34 words):** Hi, we are team ZollerQu. Which stocks should a fund buy? We turn this into an Ising problem, one qubit per asset, and sample it with QAOA on the IonQ Forte 1 noise model.

**Alt text** (`slides/figures/hero_graph.png`): Network picture of the 12 assets: 12 dots on a circle, every pair joined by a line; thicker lines mean the two assets move more strongly together.

## Slide 2: Which assets should a fund buy?

**Script (33 words):** Banks and funds build portfolios every day. They want low risk and a good return. Risk comes from assets that fall together. Twelve assets give 4,096 portfolios; twenty-nine give over five hundred million.

**Source line on the slide:** Data: weekly prices of 12 US stocks and funds, 2006-2014 (Kaggle Huge Stock Market Dataset, CC0)

## Slide 3: One qubit per asset, one Ising energy

**Script (42 words):** Each asset becomes one qubit: zero means buy, one means skip. Risk minus return becomes an Ising energy with one coupling per pair of assets: 66 couplings. For twelve assets we check every portfolio: the best is Apple plus long-term Treasury bonds.

**Alt text** (`slides/figures/formula_score.png`): Formula: J of x equals x transposed C x over n squared V ref (the risk), minus eta times mu transposed x over n sigma ref (the return).

**Alt text** (`slides/figures/formula_ising.png`): Formula: E of s equals the sum over pairs of J i j times s i s j, plus the sum of h i times s i, plus a constant.

**Alt text** (`slides/figures/graph_optimum.png`): Network of the 12 assets with every pair linked. Two filled dots mark the exact best portfolio: buy AAPL and TLT; the other ten are skipped.

**Source line on the slide:** Exact best portfolio found by checking all 4,096 portfolios (classical); risk and return from 2006-2014 data

## Slide 4: QAOA circuits, trained offline, one IonQ job

**Script (41 words):** The circuit is QAOA: Hadamard gates, a cost layer with one ZZ gate per pair, then a mixer. At twelve qubits the best is CVaR-QAOA, sent to IonQ as one job; at twenty-nine we use linear-ramp QAOA with angles from twelve.

**Alt text** (`slides/figures/circuit_sketch.png`): Circuit sketch: four of the twelve qubit lines. Each starts with an H gate, passes a cost layer block (one ZZ gate per pair, 66, and one Z rotation per qubit) and a mixer block (RX on every qubit), repeated three times, then is measured; 1,000 shots give 1,000 portfolios.

**Source line on the slide:** 12 qubits: 58 circuits + a random-guess control, noiseless simulation · 16-29 qubits: server simulation

## Slide 5: 8 circuit types, 5 settings that mattered

**Script (35 words):** We tested eight circuit types, six kinds of QAOA and two imaginary-time methods. Five settings made the difference: the mixer sign, a CVaR objective, more layers, a wider ramp range, and choosing angles by CVaR.

**Source line on the slide:** 12 assets: noiseless simulation, 10 Oct 2026 · 29 assets: noiseless server simulation, 11 Oct 2026 / per shot = chance that one shot is the best portfolio · runs = 1,000-shot runs that contain it

## Slide 6: All-to-all links: 4.3× fewer two-qubit gates

**Script (37 words):** Every pair of assets is coupled, and trapped ions link every pair of qubits directly. One layer needs 66 two-qubit gates; on a heavy-hex grid it needs 285: 4.3 times more, and 6.5 times at twenty-nine qubits.

**Alt text** (`slides/figures/gate_counts.png`): Column chart of two-qubit gates in one QAOA layer. At 12 qubits: 66 on all-to-all against 285 on a heavy-hex grid. At 29 qubits: 244 against 1,576.

**Source line on the slide:** Gate counts from transpiling only (Qiskit 2.5.2, level 1); heavy-hex = median of 5 routing seeds / on a generic heavy-hex map. No circuit was run on a heavy-hex device.

## Slide 7: IonQ noise model: 29× random guessing

**Script (45 words):** The main result, on the IonQ Forte 1 noise model: 7 of 1,000 shots were the exact best portfolio, 29 times more than random guessing. The noiseless simulation gives 58, so noise costs a factor of 8. Classical greedy search is in the same range.

**Alt text** (`slides/figures/main_result.png`): Dot chart on a log scale: evaluations until the best portfolio is first found, median, fewer is better. Noiseless simulation of the same circuit 12; greedy descent 73; IonQ Forte 1 noise model 99 with a 95 % interval from 48 to 246; simulated annealing 1,516; random guessing 2,839.

**Source line on the slide:** IonQ Forte 1 noise model (Qollab): 1 job, 1,000 shots, 11 Oct 2026 · noiseless simulation / Classical: median of 50 runs. Sampler medians come from the hit rate per shot. Training not counted.

## Slide 8: 29 qubits: more layers help, only without noise

**Cut to the live Qollab demo at the marker below, then come back to the slide.**

**Script (48 words):** At twenty-nine assets the three-layer circuit finds the best portfolio in 7 percent of runs, classical methods in 61. With more layers, the noiseless circuit reaches 90 percent at sixteen. But that is six and a half thousand two-qubit gates, far beyond today's noise. No advantage is claimed.

**Alt text** (`slides/figures/depth29.png`): Line chart for 29 assets: share of 1,000-shot runs that contain the exact best portfolio, against the number of layers, in noiseless simulation. With the ramp tuned on a wider grid: 6.6 % at 3 layers, 62 % at 12, 90 % at 16, 99.6 % at 24 and 100 % at 32. With the ramp as in the paper: 45 % at 16 layers. Classical methods with 1,000 evaluations: 61 %.

**Source line on the slide:** Server simulation of all 536,870,912 states, 11 Oct 2026; ramp tuned at 12 assets, no tuning at 29 / Classical: greedy descent and annealing, 200 runs, the same 1,000 evaluations

## Slide 9: Try it on Qollab: one click, one IonQ job

**Script (49 words):** [CUT TO LIVE QOLLAB DEMO] Here is the project on Qollab, with the code in main.py. We press Run, select the IonQ Forte 1 noise model, and one job returns counts and plots. [BACK TO SLIDE] Our limits: a noisy simulator, one job, and strong classical baselines. Next: other circuits, shorter circuits, and the depth noise allows.

## Slide 10: Thank you for your attention

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
| `slides/figures/depth29.png` | slide 7 | results/stocks_exp29_summary.json (29-asset depth experiment and its classical baselines) |

## Open points

- No placeholder is left on the slides. Slide 8 is shown together with a live Qollab session (no screenshot).
- Links are not on the slides (Qollab project, Demo video): set LINKS in slides/build_deck.py to print them on the closing slide. The GitHub repository is private and is not linked.

## Caveats and things not verified from the files

- Slide 4, best circuit per size: at 12 qubits CVaR-QAOA (3 layers) has the highest hit rate of all 58 circuits. At 16, 20 and 29 qubits only linear-ramp QAOA was simulated, so no other family was compared there. Its best 29-qubit run has 4 layers (12.6 % of runs find the best portfolio, run 'thinned_p4'); slide 7 shows the 3-layer run (6.6 %) so that every size uses the same settings.
- The stocks run file records the backend only as 'ionq_simulator'. That the IonQ Forte 1 noise model was the simulator selected in the Qollab Run dialog is taken from the team, not from the file (the measured rate is far below the noiseless one, so a noise model was active).
- One IonQ job of 1,000 shots: no repeat run, so run-to-run variation on the noise model is not measured; the 95 % interval covers sampling error only.
- Heavy-hex gate counts come from Qiskit's router on a generic heavy-hex map (5 seeds); a better router or a real device layout could give other numbers.
- The medians for the three samplers on slide 6 are computed from the per-shot hit rate (geometric distribution), not measured by repeating runs.
- The PPTX was rendered with macOS Quick Look (slide by slide) and compared with the PDF; it was not opened in PowerPoint itself.

Rebuild everything with `./venv/bin/python slides/build_deck.py` (reads `results/` and `qollab/`; nothing is simulated).
