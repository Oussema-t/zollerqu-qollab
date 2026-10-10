# Logical Teleportation on a Distance-2 Surface Code: interactive tutorial

Qollab x IonQ hackathon, theme Quantum Unlocked. Team ZollerQu. Build window Oct 9 19:00 ET to Oct 11 19:00 ET 2026.

## 1. Goal

An interactive Qollab JS/Qiskit project that makes four ideas click for a newcomer:

1. A logical qubit is several physical qubits plus checks (stabilizers) that flag errors.
2. Some logical gates are "free": a logical CNOT is four physical CNOTs.
3. Quantum teleportation works on logical qubits, and the Pauli corrections can be done in software.
4. The T gate is not free. It is itself a teleportation, using a magic state.

Everything runs on 18 or fewer qubits, so it works on the Qollab built-in simulator (24 qubits), the IonQ Aria 1 noise-model simulator (25) and the IonQ Forte simulators (29). It needs no mid-circuit measurement, no reset and no classical feed-forward.

Honesty line, repeated on the page: distance 2 detects any single error and corrects none. Accepted shots are post-selected. The encoder is not fault tolerant. This is the regime of the first logical-gate experiments on trapped ions and superconducting chips.

## 2. The code: [[4,1,2]] rotated surface code

One logical qubit per 2x2 patch. Local data indices 0..3 laid out as

```
q0 --- q1      top edge
|  X   |       XXXX face in the middle
q2 --- q3      bottom edge
```

| Object | Pauli (local indices) | Drawn as |
|---|---|---|
| Stabilizer | X0 X1 X2 X3 | the face |
| Stabilizer | Z0 Z1 | semicircle on the top edge |
| Stabilizer | Z2 Z3 | semicircle on the bottom edge |
| Logical Z | Z0 Z2 | left edge |
| Logical X | X0 X1 | top edge |
| Logical Y | Y0 X1 Z2 | derived, lesson 4 only |

Codewords: |0_L> = (|0000> + |1111>)/sqrt2, |1_L> = (|1100> + |0011>)/sqrt2, with the bit order q0 q1 q2 q3 left to right. Note Qiskit bitstrings are little-endian, so the analysis layer reverses them before indexing.

Encoder. The input state is prepared on local q0 first, then:

```
h q2; cx q2,q3; cx q0,q1; cx q2,q0; cx q2,q1
```

Verified numerically on Oct 9: maps a|0> + b|1> on q0 to a|0_L> + b|1_L>, transversal CNOT between two blocks equals logical CNOT, every single-qubit X, Y or Z anticommutes with XXXX or ZZZZ, and the T gadget below yields the encoded T state with keep fraction 0.5.

## 3. Qubit and bit layout

For a circuit with B blocks:

- Block b uses data qubits 4b .. 4b+3, local index i at 4b + i.
- Each block has two ancillas: `ax(b)` = 4B + 2b (measures XXXX) and `az(b)` = 4B + 2b + 1 (measures ZZZZ).
- One classical register of size 6B. Classical bit k records qubit k.
- Bare reference circuits use one qubit per logical qubit and no ancillas.

Total qubits: lesson 1: 6, lesson 2: 12, lesson 3: 18, lesson 4: 12.

Syndrome extraction, applied to every block after the lesson's logical operations and before the readout rotations:

```
XXXX: h ax; cx ax,q0; cx ax,q1; cx ax,q2; cx ax,q3; h ax
ZZZZ: cx q0,az; cx q1,az; cx q2,az; cx q3,az
```

Readout of a block in basis:

- Z: measure q0..q3. Logical value = parity(q0, q2). Readout check: parity(q0,q1) and parity(q2,q3) must be 0.
- X: h on q0..q3, measure. Logical value = parity(q0, q1). Readout check: parity(q0,q1,q2,q3) must be 0.
- Y: sdg q0; h q0; h q1; measure all. Logical value = parity(q0, q1, q2). No readout check.

All ancillas and data are measured at the very end of the circuit.

A shot is **flagged** if any ancilla bit of any block is 1 or any readout check of any block is odd. **Accepted** shots are the unflagged ones.

## 4. Lessons

Each lesson produces encoded circuits and bare physical reference circuits that undergo the same preparation, same optional injection and same readout. Ideal P(logical = 1) is listed so the analysis can compute deviation.

### Lesson 1: one patch, one error

Settings: injection = none, or Pauli in {X, Y, Z} on local data qubit in {0,1,2,3}, applied right after the encoder.

| Circuit | Prep on q0 | Readout | Ideal P(1) |
|---|---|---|---|
| enc_Z | none (|0_L>) | Z | 0 |
| enc_X | h (|+_L>) | X | 0 |
| bare_Z | none | Z | 0 |
| bare_X | h | X | 0 |

Bare circuits apply the injected Pauli to their single qubit when injection is on. Teaching point: the bare qubit cannot tell it was hit; the patch flags it.

### Lesson 2: two patches, logical CNOT

Block A = |+_L> (h q0 then encode), block B = |0_L>. Transversal CNOT: cx(A_i, B_i) for i in 0..3.

| Circuit | Readout A, B | Logical value | Ideal P(1) |
|---|---|---|---|
| enc_ZZ | Z, Z | Z_A xor Z_B | 0 |
| enc_XX | X, X | X_A xor X_B | 0 |
| bare_ZZ | Z, Z | same on 2 qubits | 0 |
| bare_XX | X, X | same | 0 |

### Lesson 3: three patches, logical teleportation

Block A = |psi_L> with psi = ry(pi/4)|0> on q0 before encoding. Block B = |+_L>, block C = |0_L>, then transversal CNOT B to C (logical Bell pair). Bell measurement: transversal CNOT A to B, A read in X, B read in Z. C is read in Z or X.

Pauli frame applied in analysis: Z readout of C is XORed with Z_B; X readout of C is XORed with X_A.

| Circuit | C readout | Ideal P(1) after frame |
|---|---|---|
| enc_Z | Z | sin^2(pi/8) = 0.146 |
| enc_X | X | (1 - sin(pi/4))/2 = 0.146 |
| bare_Z, bare_X | same on 3 qubits | 0.146 |

### Lesson 4: logical T by magic-state teleportation

Block A = |psi_L> with psi = ry(pi/4)|0>, the same state as lesson 3. Block B = |T_L>: h q0; t q0; encode. Transversal CNOT with B as control and A as target. A read in Z. Shots with Z_A = 1 are **discarded** (they would need a logical S correction, which needs feed-forward). B read in X or Y.

| Circuit | B readout | Ideal P(1) among kept shots |
|---|---|---|
| enc_X | X | 0.25 |
| enc_Y | Y | 0.25 |
| bare_X, bare_Y | same on 2 qubits | 0.25 |

The input is deliberately not |+>: T|+> is the magic state itself, so with |+> the output would look identical whether or not the gadget acted. With psi, the readouts give 0.25 in X and Y, against 0.146 and 0.5 without the T gate.

The page shows the discarded fraction (ideal 0.5) separately from the flagged fraction.

## 5. Metrics shown for every lesson and circuit

- Flagged fraction: flagged shots / all shots.
- Logical P(1) over accepted shots, with the ideal value next to it.
- Deviation: |P(1) accepted - ideal|. For lessons 1 and 2 the ideal is 0 so this is the logical error rate.
- Bare deviation: the same quantity for the bare reference circuit.
- Lesson 4 adds the kept fraction after post-selection on Z_A.
- Per block: fire rate of each of the three stabilizer faces (from the two ancillas and the readout checks), used to light up the patch drawing.
- Per block readout bases, so the page can mark checks that the chosen basis cannot read as not measured.

## 6. Architecture

Python owns the physics, JavaScript owns the page. Chosen over a JS-only build because the physics is then testable locally with pytest, and over a Python-only project because the theme asks for an interactive demo.

```
unlocked/
  index.html             body-only markup
  main.css
  main.js                boot, UI wiring, the single backend.run, rendering calls
  ui/patches.js          SVG patches, syndrome highlighting, hover on logical operators
  ui/charts.js           SVG bar charts and summary cards, no external library
  ui/lessons.js          lesson titles and explanatory text
  qec/__init__.py
  qec/code412.py         encoder, syndrome extraction, readout, bare references
  qec/lessons.py         build_all(settings) -> circuits, metadata
  qec/analyze.py         counts -> flagged, accepted, frame, metrics, face fire rates
  qec/bridge.py          JSON-in JSON-out functions called from main.js
  data/recorded_aria1.json
  tests/test_code412.py
  tests/test_lessons.py
  tests/test_analyze.py
  requirements.txt       qiskit[visualization]>=2.5.2,<3.0.0
  README.md              project page text
```

Python to JS boundary. All arguments and return values are JSON strings except circuits and figures, which stay Python proxies.

- `bridge.build(settings_json) -> metadata_json; bridge.circuits() returns the cached list`. `circuits` is a Python list of QuantumCircuit. Metadata is a list, one entry per circuit: lesson id, name, kind (enc or bare), block count, per-block readout basis, logical bit recipe, frame recipe, post-selection recipe, ideal P(1).
- `bridge.analyze(metadata_json, counts_json) -> results_json`. `counts_json` is a list of {bitstring: count} objects in circuit order. Returns per lesson and per circuit the metrics of section 5 plus per-block face fire rates.
- `bridge.draw(index) -> figure`. Matplotlib figure of one circuit; main.js turns it into an SVG URL via `URL.createObjectURL`.
- `bridge.load_recorded() -> json` reading `data/recorded_aria1.json` through a literal path string, since JavaScript cannot read project files.

`backend.run` is called from main.js only, because a Python-side call fails on IonQ backends with an XHR error. Counts are collected in JS with `result.get_counts(i)` for each circuit index and `.toJs()`, then serialised to JSON.

## 7. Run flow

Three page states.

1. **Setup.** Backend banner (name from the Qollab dialog, and a warning that IonQ backends allow one submission per run). Controls: shots (default 500, range 100 to 2000), lesson 1 injection (none or Pauli and qubit), buttons Submit and Load recorded take.
2. **Running.** Submit calls `bridge.build`, then one `backend.run(circuits, {shots})` with all 16 circuits, polls `job.status()` with the injected `setTimeout`, shows a progress line in an `aria-live` region. On local simulators Submit stays enabled for repeat runs. On IonQ backends it is disabled after the first job with the message to Stop and Run again.
3. **Explore.** `bridge.analyze` runs once. Lesson tabs switch views offline. Nothing talks to the backend again.

Recorded take: `data/recorded_aria1.json` holds settings, metadata and counts from one Aria 1 noise-model run made on Saturday Oct 10. Load recorded take jumps straight to Explore.

Errors: a job error or a backend limit error shows the backend message in the banner and returns to Setup. Nothing else is handled specially.

## 8. Page layout and accessibility

Layout, top to bottom on narrow screens, two columns on wide:

- Header: title, one sentence, backend banner.
- Setup panel, collapses after submission, reopenable.
- Lesson tabs 1 to 4, implemented as a tablist with arrow-key navigation and `aria-selected`.
- Patch area: one to three SVG patches. Corner dots are data qubits, the face and the two semicircles are stabilizers, filled with intensity proportional to fire rate and labelled with the percentage. Hover or focus on the left edge highlights logical Z, on the top edge logical X. Transversal CNOTs drawn as four thin links between corresponding corners.
- Explanation column: lesson text, what to look for, the honesty notes relevant to that lesson.
- Results: three summary cards (flagged, logical deviation accepted, bare deviation), one bar chart per readout basis with accepted, all shots, bare and ideal, and a `<details>` element "Show the physical circuit" that lazily loads the Qiskit drawing.

Accessibility rules: semantic buttons and tabs, visible focus rings, status text in `aria-live`, every SVG element with a `<title>`, syndrome state shown by colour and by a text list of fired checks, contrast at least 4.5:1, `prefers-reduced-motion` respected, page usable at 360 px width.

## 9. Testing

Local, with pytest, Qiskit 2.x and Aer or BasicSimulator:

- `test_code412.py`: statevector checks of section 2 (encoder, logical operators, transversal CNOT, detection of every single-qubit Pauli, T gadget with projection).
- `test_lessons.py`: `build_all` returns 16 circuits, correct qubit counts (6, 12, 18, 12 maximum per lesson), metadata fields present, every circuit ends with measurements of all qubits.
- `test_analyze.py`: run `build_all` on an ideal simulator with 4000 shots. Expect flagged fraction 0, deviation below 0.03 for every encoded circuit, lesson 4 kept fraction within 0.05 of 0.5. With each injection setting in lesson 1 expect flagged fraction 1 for enc circuits and 0 for bare. With Aer depolarising noise at p = 0.01, print accepted versus bare deviation for every lesson as an informational table.
- Bit-order test: a hand-built counts dict with known bitstrings maps to the expected per-qubit values.

On Qollab, manual, in this order: a hello-world JS project to confirm the Python import and `.callPromising` path; the full page on the built-in simulator; Aria 1 noise model at 100 shots; the recorded take at 500 shots; Forte simulator once.

## 10. Timeline (CDT)

- Fri night: spec, plan, scaffold, `code412.py` with tests.
- Sat morning: `lessons.py`, `analyze.py`, `bridge.py` with tests. First Qollab smoke test.
- Sat afternoon: page, patches, charts, tabs. Full run on the built-in simulator. Record the Aria 1 take.
- Sat evening: README and project page, accessibility pass, demo video.
- Sun morning: polish, publish, links to the captain by 14:00 CT. Hard deadline 18:00 CT.

## 11. Out of scope

Distance 3, decoding and correction, mid-circuit measurement, reset, feed-forward, custom noise models, hardware runs, saving state between runs.

## 12. Credits to include on the project page

Small-code error detection on trapped ions (Linke et al. 2017) and the [[4,2,2]] and [[4,1,2]] family (Gottesman 2016), magic-state T gadget (Bravyi and Kitaev 2005), quantum teleportation (Bennett et al. 1993), Qollab runtime docs for the multi-file and recorded-take patterns. AI tools used: Claude Code for planning and implementation, disclosed as the rules require. Required line: "This effort is supported by Qollab & IonQ".

## Addendum, Oct 10 2026: 3D spacetime view

Approved design (Tushar, Oct 10). Each lesson panel gains a three.js scene of the encoded circuit as a spacetime block, after Google Quantum AI, Nature 614, 676 (2023), Fig. 1b and 2a, with time pointing up instead of right.

- Ground plane: the patches as the 2D view draws them (gold data qubits at the corners, XXXX face, two blue Z caps, logical edges); XXXX ancilla at the face centre, ZZZZ ancilla just below the patch. Every qubit is a worldline rising through time.
- Gate layers: the circuit is cut into phases recorded by the builders (prepare, encode, inject, transversal CNOT, syndrome extraction, readout rotation, measure). Gates inside a phase are packed as early as possible; phases never overlap. Single-qubit gate: labelled box on the worldline. CNOT: bar between worldlines at its step with a dot at the control. Transversal CNOT: four parallel bars patch to patch. Syndrome extraction: the fan of bars from the corners into the ancilla.
- Top slice from the data: at the final step the tiles are drawn again filled by the measured fire rates and the ZZZZ ancilla rate, so the top of the block is the measured syndrome.
- Injected error (lesson 1): red marker on the hit worldline at its step; the tiles that catch it are outlined red at the top (spacelike pair).
- Logical worldsheets: Z_L and X_L edges extruded along time as translucent sheets (blue, gold). At a transversal CNOT the X sheet of the control continues onto the target and the Z sheet of the target onto the control, with a horizontal bridge at that step. Toggle, default on.
- Animation: time slider and play button; everything above the current time is hidden and a faint "now" plane rises. Orbit and zoom via OrbitControls. No autoplay under reduced motion. Text alternative: the phase list with time spans, clickable.
- Build: phase names recorded in `qc.metadata["phases"]` (gates unchanged), new `qec/spacetime.py` layout JSON, bridge entry `spacetime(index)`, new `ui/spacetime.js`. three.js is importable by name on Qollab (0.186.1, runtime docs). The Qiskit line drawing stays as a collapsed "Raw Qiskit circuit" and leaves the video.
- Dev harness (`dev/`) and scripted Playwright demo video are separate tooling, see the plan of Oct 10.
