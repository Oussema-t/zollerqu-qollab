# Speaker notes: mesh partitioning on IonQ (Team ZollerQu)

Spoken script for slides 1-8: 379 words, about 2:30-2:50 at a calm pace. The appendix is not spoken.

## Slide 1: Title

Hi, we are team ZollerQu: Oussama Turki and Tushar Pandey. We split a simulation mesh between two computers, a CPU and a GPU, with a quantum circuit on IonQ's noisy simulator.

## Slide 2: Problem

Big simulations, like crash tests, cut their mesh into parts. A good split gives each device a fair share of work and keeps the border short, because data crosses it at every step. A GPU four times faster should get eighty percent.

## Slide 3: Formulation

We group the mesh into twelve regions, one qubit each: zero means device A, one means device B. The energy adds the border that is cut and a penalty when the work share misses its target. Twelve qubits let us check every split and know the exact best one.

## Slide 4: Quantum method

The circuit starts from all splits, applies a cost layer with one two-qubit gate per pair of regions, then a mixer. We compared one hundred and four circuits, including IonQ's varQITE. Training is offline and noiseless. On Qollab we send one fixed circuit per job.

## Slide 5: Why IonQ

Our balance term couples every pair of regions. IonQ connects every pair of qubits, so one layer needs sixty-six two-qubit gates. Routed on a heavy-hex layout it needs about three times more, counted with the transpiler only.

## Slide 6: Main IonQ result

The main result, for two equal devices. On the IonQ Forte 1 noise model, twenty-eight percent of one thousand shots were the exact best split, and it was the most frequent answer. Random guessing gives five hundredths of a percent. Without noise we reach seventy-five percent: noise costs twice what we estimated.

## Slide 7: Scaling and honest comparison

Now the honest part. As the mesh grows, classical methods win: at twenty-nine regions, simulated without noise, our best circuit finds a near-best split in under one percent of runs, spectral bisection in half or more. On IonQ's simulator we needed four shots per optimum against eleven hundred classical evaluations. But varQITE first needed seven thousand six hundred training circuits, and a shot is not the same work as a classical evaluation.

## Slide 8: Demo, limits and next steps

[CUT TO THE LIVE QOLLAB DEMO] On Qollab we press Run and choose IonQ Forte 1. Limits: a noisy simulator, not hardware. For a CPU plus a GPU the varQITE circuit fails, and we proved why. Beyond twelve qubits classical methods win. Next, we repair that circuit. Thank you. This effort is supported by Qollab and IonQ.

## Slide 9: Appendix: both device scenarios

Appendix, not spoken: the two device scenarios side by side. For CPU plus GPU the published varQITE circuit is capped by a qubit that stays fifty-fifty; the best other circuit reaches about half a percent without noise.

## Figure alt text

- **fig_problem.png**: An L-shaped triangle mesh split in two parts. The large teal part is computed by the GPU (81 % of the work), the small light part by the CPU. An orange line marks the border between the parts.  
  Source: data/frozen_meshes/2_main_seed7.npz, data/frozen_regions.npz, qollab/selected_circuits.json (exact optimum for CPU + GPU)
- **fig_formula.png**: The Ising energy: a sum over neighbouring regions of border cost times (1 - z_i z_j) / 2, plus lambda times the squared difference between the weighted sum of the z_i and (1 - 2 alpha). The first term is the border that is cut, the second a penalty if the work share is off target.  
  Source: the Ising energy used in the notebook (Task 03) and in qollab/main.py
- **fig_circuit.png**: Circuit sketch with four of the twelve qubit lines: an H gate on every qubit, a cost layer with one two-qubit rzz gate per pair of regions, an rx mixer gate on every qubit, then measurement. The cost layer and the mixer are repeated p times.  
  Source: sketch of the QAOA circuit built by qollab/main.py (build_circuit)
- **fig_gates.png**: Bar chart of two-qubit gate counts. One QAOA layer at 12 qubits: 66 on all-to-all connectivity, 216 routed on heavy-hex. The varQITE ZY circuit at 12 qubits: 66 against 237. One QAOA layer at 29 qubits: 406 against 1,498.  
  Source: results/ionq_selection/heavy_hex_gate_counts.json (scripts/heavy_hex_gate_counts.py, transpiler only)
- **fig_ionq_result.png**: Bar chart on a log scale of the share of shots that are the exact best split: random guessing 0.049 %, IonQ Forte 1 noise model 27.8 %, noiseless simulation 75.0 %. A dashed line marks our estimate of 54 %.  
  Source: qollab/results_ionq/2026-10-11_varqite_zy_equal_ionq_simulator.json (IonQ Forte 1 noise model on Qollab), qollab/selected_circuits.json (noiseless value, estimate, random guessing)
- **fig_scaling.png**: Two line charts, for two equal devices and for CPU plus GPU, of the share of runs with 1,000 tries that find a near-best split, at 12, 16, 20 and 29 qubits. The QAOA line falls from above 90 percent at 12 qubits to below 1 percent at 29 qubits. Local search and simulated annealing also fall but stay higher; spectral bisection stays at 50 percent or more.  
  Source: report/scaling.json (12, 16, 20 qubits: results_large_hpc/, server simulation 2026-10-10) and results/hpc_n29/summary_q29.csv (29 qubits, server simulation 2026-10-10); QAOA at 29 qubits = best fixed schedule p = 5
- **fig_samples.png**: Horizontal bar chart on a log scale of tries per optimum: random guessing 2,048, simulated annealing 1,181, local search 1,111, IonQ noise model 3.6, noiseless 1.3.  
  Source: qollab/results_ionq/2026-10-11_varqite_zy_equal_ionq_simulator.json (shots per optimum), qollab/selected_circuits.json (classical median evaluations from results/ionq_selection/, noiseless value)
- **fig_qollab_placeholder.png**: Placeholder for a live Qollab screenshot. It shows lines of the recorded console output: running on ionq_simulator, 1,000 shots, measured P(optimum) 0.2780, and the two optimal splits measured 146 and 132 times.  
  Source: qollab/results_ionq/2026-10-11_varqite_zy_equal_qollab_console.html (recorded console output); to be replaced by a live screenshot

## Placeholders still open

- Slide 8: Qollab project link `[pending]`, video link `[pending]`, live Qollab screenshot `[pending]` (the picture shows the recorded console output).
- Appendix: the IonQ Forte 1 noise-model runs for CPU + GPU `[pending]` (two cells).
