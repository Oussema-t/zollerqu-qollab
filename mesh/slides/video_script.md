# Video script (3 minutes max) — read over the slides

**Slide 1 — Title**
Hi, we're team ZollerQu: I'm Oussama Turki, with Tushar Pandey. We split a computer simulation between a CPU and a GPU, using a quantum circuit on IonQ's noisy simulator.

**Slide 2 — The problem**
Engineers simulate heat, stress or car crashes on a mesh of triangles. To use a CPU and a GPU together, the mesh is cut in two. A GPU four times faster should get about eighty percent of the work, with a short border, because data crosses that border at every step.

**Slide 3 — Formulation**
We group the mesh into twelve regions, one qubit each: zero means device A, one means device B. The energy counts the cut border, plus a penalty when the work share misses its target. With twelve regions we check all 4,096 splits, so we always know the best answer.

**Slide 4 — Nine objects**
We tested nine mesh objects: plates, L-shapes and a plate with a hole, with heat sources, heavy materials and a priority zone.

**Slide 5 — Quantum method**
We compared 104 circuits, QAOA variants and IonQ's varQITE. Training is offline and noiseless; on Qollab we send one fixed circuit per job.

**Slide 6 — Why IonQ**
IonQ connects every qubit to every other, so one layer needs sixty-six two-qubit gates. A heavy-hex chip would need about three times more.

**Slide 7 — Main result**
On IonQ's Forte 1 noise model, twenty-eight percent of a thousand shots were the exact best split, about five hundred and seventy times random guessing, and it was the most frequent answer. A second run gave almost the same. Without noise we reach seventy-five percent, so noise still costs a lot.

**Slide 8 — Honest comparison**
With our first settings, bigger meshes went to classical methods: at twenty-nine regions, spectral bisection found a near-best split in half the runs or more, our circuit in under one percent.

**Slide 9 — Tuning QAOA**
So we tuned QAOA itself, depth and all angles, on separate calibration meshes, and tested on unseen ones. At twelve qubits that gives about twenty times more near-best answers per shot. At twenty-nine qubits, simulated without noise, it finds one in eighty-eight percent of cases for equal devices, against fifty for spectral bisection. But those circuits need over three thousand two-qubit gates, so we claim no quantum advantage.

**Slide 10 — Demo and limits**
[show Qollab] On Qollab we press Run, choose the IonQ Forte 1 noise model, and every shot is graded against the exact answer. Limits: a noisy simulator, not hardware, and for a CPU plus a GPU the varQITE circuit fails; we proved why. Thank you. This effort is supported by Qollab and IonQ.
