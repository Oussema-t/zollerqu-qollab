# Distance 3 and the story page: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. One implementer per task, explicit-path commits, no attribution lines.

**Goal:** Replace the [[4,1,2]] patches with rotated [[9,1,3]] patches (9 data qubits, no helper qubits, checks read from the final measurement, single errors corrected by lookup), and ship the story page (logical and physical levels, per lesson) inside the tutorial. The three.js panel is removed.

**Decisions (approved by Tushar, Oct 10 late / Oct 11):** d=3; three-patch teleportation at 27 qubits, which runs live only on the Forte simulators (29 qubits); helper-free checks with lookup correction; story page per lesson with the Logical/Physical toggle; prototype at scratchpad preview/story.html is the visual reference.

**Deadline:** Sun Oct 11 19:00 ET. Order of work is chosen so that a working, tested d=3 tutorial exists before the story integration starts.

---

## Facts

| Fact | Source |
|---|---|
| Qollab backends: built-in 24 qubits, Braket 25, Aria 1/2 25, Forte 1 and Forte Enterprise 1 simulators 29; "ceilings, not guarantees" | compute-backends docs, fetched Oct 11 |
| IonQ backends: one `backend.run` per code run; pass all circuits in one call | runtime docs |
| `backend.name` string or function, tested by regex in `main.js:40,207` | `main.js` |
| Aer extended_stabilizer: 20 noisy shots of the 27-qubit lesson 3 circuit in 5.7 s; MPS 2 s/shot; needs Clifford+T, so ry(π/4) = sdg, h, t, h, s | measured Oct 11 |
| Encoder verified: cx 0→1, cx 0→2; h 4,5,6,7; cx 4→0,4→1,4→3; cx 5→2; cx 6→3; cx 7→0,7→1,7→2,7→3,7→8 maps \|0⟩,\|1⟩ on q0 to \|0_L⟩,\|1_L⟩ with overlap 1.000000 | measured Oct 11 |
| Stabilizers: X (0,1,3,4), (4,5,7,8), (2,5), (3,6); Z (1,2,4,5), (3,4,6,7), (0,1), (7,8). X_L = X0X1X2 (top row), Z_L = Z0Z3Z6 (left column) | verified commuting and logical, Oct 11 |
| Lookup tables (syndrome order as listed): Z readout {0000:(), 0010:0, 1010:1, 1000:2, 0100:3, 1100:4, 0101:7, 0001:8}; X readout {0000:(), 1000:0, 0010:2, 1001:3, 1100:4, 0110:5, 0001:6, 0100:7}; all single errors corrected to a stabilizer; weight 2: 16 discarded, 18 logical flips, 2 fine | measured Oct 11 |
| Existing meta fields and analysis row fields used by the UI | `qec/lessons.py:11-17`, `ui/charts.js`, `ui/patches.js:38-40` |
| Story prototype files: preview/story.html, story.js (timeline, legend rules, state panel, token, ribbons, step strip, intro and closing cards), story_data.js shapes | scratchpad |

## Qubit layout

Block b: data qubits 9b+i, i row-major in a 3x3 grid: row 0 = 0 1 2 (top), row 1 = 3 4 5, row 2 = 6 7 8. No helpers. n_qubits(n) = 9n. Lessons: 1 → 9, 2 → 18, 3 → 27, 4 → 18 qubits.

Plane coordinates for drawing (both patches.js and the story): local (col, row) = (i % 3, i // 3); block offset x0 = b × PITCH, PITCH = 4.4 (grid is 2 wide, gap 2.4).

## Readout recipes

- Z: checks = parities over Z plaquettes [(1,2,4,5), (3,4,6,7), (0,1), (7,8)]; logical = v0^v3^v6; correction table Z above; apply the correction (flip the bit of the table qubit) before computing the logical value.
- X: rotate h on all 9; checks over X plaquettes [(0,1,3,4), (4,5,7,8), (2,5), (3,6)]; logical = v0^v1^v2; table X.
- Y: Y_L = i X_L Z_L = Y0 X1 X2 Z3 Z6; rotate sdg,h on q0; h on q1,q2; logical = v0^v1^v2^v3^v6; no checks, no correction. (Sign to be fixed by the lesson 4 test: ideal P(1) with T = 0.25, without T = 0.5; if the test gives 0.75, flip the logical bit for Y.)

Per shot classification per block: clean (syndrome 0), corrected (syndrome in table, nonzero), discarded (syndrome not in table). A shot is kept if every value block is clean or corrected. The frame correction (lesson 3) and keep_if_zero (lesson 4) use corrected logical bits.

## Analysis row (new and renamed fields)

`n_total, n_clean, n_corrected, n_discarded, n_kept` (kept = after keep_if_zero), `p1_corrected` (over clean+corrected shots, after corrections and frames), `p1_clean` (over clean shots only), `p1_all` (no checks, no correction), `deviation_corrected, deviation_clean, deviation_all`, `discard_fraction`, `correct_fraction`, `kept_fraction`, `blocks[bi] = {checks: [4 rates in recipe order], corrections: [9 per-qubit correction rates], basis}`. Bare rows keep `p1_all` and `deviation_all`. Cards: "discarded by checks", "corrected by the code", "logical error after correction", "same experiment on bare qubits", lesson 4 "kept after post-selecting A = 0".

---

### Task 1: `qec/code913.py` (replaces code412.py) with tests

Files: create `qec/code913.py`, delete `qec/code412.py`, rewrite `tests/test_code913.py` (delete test_code412.py).

API (same names as before where it exists): `BLOCK_NAMES = "ABC"`, `data(b, i)`, `n_qubits(n)`, `new_circuit(n)` (n_qubits classical bits), `mark(qc, name)`, `X_STABS, Z_STABS, X_L = (0,1,2), Z_L = (0,3,6)`, `encode(qc, b)` (marks "encode {name}", then the verified 16-gate circuit on block b), `transversal_cx(qc, c, g)` (marks, 9 cx), `inject(qc, b, pauli, i)` (marks "inject P on qI", i in 0..8), `rotate_block(qc, b, basis)` (marks "readout rotation {name} ({basis})" for X and Y), `finish(qc, n, bases)` (rotations, mark "measure", measure all), `new_bare(n)`, `finish_bare(qc, bases)` (Y: sdg, h).

Tests (statevector, qiskit.quantum_info): encoder maps |0⟩→|0_L⟩ and |1⟩→|1_L⟩ (codewords built by projecting with the X stabilizers; overlap > 1-1e-9); every stabilizer has eigenvalue +1 on both codewords; X_L flips, Z_L signs; transversal CNOT equals the logical CNOT on all four basis pairs (18 qubits); single X error on each of 9 qubits after encoding changes the Z syndrome to the table entry; Y readout rotation gives the expected sign (a 9-qubit test with |+i_L⟩ prepared via encode of (|0⟩+i|1⟩)/√2 reads 0 with probability 1); marks recorded in order.

### Task 2: `qec/lessons.py`, `qec/analyze.py`, `qec/bridge.py`, `qec/spacetime.py`, `tools/record_take.py`, tests

- lessons: import code913; same four lessons and metas; add `"nqubits": 9*nblocks` to meta; lesson 1 injection qubit range 0..8 (index.html select gets 9 options).
- analyze: recipes and tables above; `block_readout(bits, nblocks, b, basis)` → `{logical, syndrome, status: 'clean'|'corrected'|'discarded', corrected_qubit or None}`; `analyze_one` computes the row fields above; `analyze_all` unchanged signature.
- bridge: unchanged API; `spacetime(index)` stays.
- spacetime.py: positions from the 3x3 layout, PITCH 4.4, no ancillas, `qubits[].kind = 'data'` only; phases: prepare, encode, inject, transversal, readout, measure; `inject.caught` = indices of checks (0..3 in recipe order) that fire for that single error in the readout basis of that block, plus `corrected: true` and `corrected_qubit`.
- record_take.py: Aer `extended_stabilizer`, a pass that rewrites `ry(π/4)` to sdg,h,t,h,s before running (function `clifford_t(qc)`), noise on 1q gates and cx as before, 500 shots, no injection (lesson 1 then shows the noise floor; injection is the live demo). Regenerate `data/recorded_aria1.json`; keep source "local-aer-stand-in", backend text "Aer extended-stabilizer stand-in (Forte run pending)".
- dev/serve.py `run()`: same `clifford_t` rewrite and `method="extended_stabilizer"`.
- tests: test_lessons (16 circuits, nqubits per lesson, names), test_analyze (table correctness on synthetic bitstrings; injection X on q4 → corrected 100 % in lesson 1 enc_Z with ideal noiseless Aer; two injected errors → discarded; lesson 3 and 4 ideal values with extended_stabilizer noiseless at 400 shots within 0.06), test_bridge (round trip, recorded take consistent), test_spacetime (positions, phases, 9 bars per transversal CNOT, inject caught set).

### Task 3: `main.js` backend capacity and lesson 3 fallback, `index.html` select

- `state.maxQubits`: 29 if `/forte|harness|aer/i` matches the backend name, 25 if `/aria|braket/i`, else 24. Circuits whose meta `nqubits` exceeds it are left out of the job (bridge gets a new `circuits_for(max_qubits)` returning the subset and `build` already returns meta with indices; simplest: JS passes the list of included indices to a new bridge entry `circuits_subset(indices_json)`).
- After a live run, lesson rows missing from the live results are filled from the recorded take (loaded via `load_recorded` and analysed), and the tab's status line says "Lesson 3 needs 27 qubits; showing the recorded take for this lesson". `explore` receives a `sources` map per lesson.
- Banner text explains the Forte requirement up front when the backend is not Forte.
- Injection qubit select: options 0..8.

### Task 4: `ui/patches.js` for 3x3

One SVG per patch: 3x3 dots, four bulk plaquettes as squares (X gold at (0,1,3,4) and (4,5,7,8), Z blue at (1,2,4,5) and (3,4,6,7)), four boundary half-plaquettes as semicircles (Z on top (0,1) and bottom (7,8); X on right (2,5) and left (3,6)), fill opacity from the check fire rate when the readout basis matches, dashed outline otherwise; per-qubit correction rate as a red ring whose opacity follows `corrections[i]`; Z_L left column and X_L top row as dashed/dotted edges with the same hover titles; injected qubit highlight; fired-list text lists plaquettes and corrections. Links line unchanged. Cards and charts in `ui/charts.js` updated to the new fields.

### Task 5: Story integration

- New `ui/story.js` (page shell, timeline, step strip, captions, state panel, legend rules, token, ribbons, intro and closing cards) ported from the prototype; `ui/storyPhysical.js` (3x3 patch animation: encoder arrows, 9 transversal arrows, injected burst, readout rotation rings, measure fill with check rates and correction rings); `ui/storyText.js` with per-lesson scripts: steps, logical symbols, captions, state expressions (lesson 3 from the prototype; lessons 1, 2, 4 from `docs/superpowers/specs/2026-10-10-story-captions-draft.md`, updated for correction: lesson 1's error is located and corrected, with the discarded two-error case mentioned).
- `index.html`: a "Watch the story" button in each lesson panel header; a `<section id="story" hidden>` full-page overlay; the three.js `#spacetime` section removed; `ui/spacetime.js` deleted; CSS for `.st-*` removed and story CSS added.
- `main.js`: open/close the story for the current lesson with the current rows and `bridge.spacetime(index)` layout; Escape closes; focus returns to the button.
- Reduced motion respected; keyboard as in the prototype.

### Task 6: Docs, spec, README, cleanup

README rewritten for d=3 (code, checks, correction, qubit counts per lesson, Forte requirement for lesson 3, story page, credits incl. the rotated surface code reference: Bombin and Martin-Delgado 2007, Tomita and Svore 2014; Google Nature 2023 for the figure conventions). Spec addendum. Remove the 3D plan's leftovers. `dev/record_demo.py` retargeted at the story page (lesson 3 logical then physical).

### Task 7: Qollab (needs Tushar) and video

Upload, run on the built-in simulator (lessons 1, 2, 4 live; lesson 3 from the take), run on Forte 1 simulator with 200 shots for the real take, export, replace the data file, publish. Record the video from the story page.

## Risks

- Time: Tasks 1 to 4 are a full rewrite of the Python core and the patch drawing; Task 5 is the largest UI piece. Parallelise: Task 1+2 (one agent), Task 4 (one agent, interface = analysis row fields above), Task 5 physical renderer (one agent, interface = layout JSON and row fields above). Task 3 after 2.
- Forte simulator speed and reliability at 27 qubits is unknown until Tushar runs it.
- The Y readout sign is fixed by test, not derived here.

## Amendments after implementation (Oct 11)

1. **Aer extended_stabilizer dropped.** Measured during Task 2: with the Clifford+T rewrite, noiseless lesson 3 returned 28 of 100 shots outside the code space and ideal P(1) = 0.46 instead of 0.146; the per-shot noisy path returned 40 of 40 shots outside the code space. The Metropolis sampler cannot move between codewords by single-bit flips. Replacement: `qec/aer_compat.py` picks an exact method per circuit (stabilizer for Clifford-only, statevector up to 20 qubits, matrix_product_state above) and holds the stand-in noise model; used by `tools/record_take.py`, `dev/serve.py` and the tests only. Cost: about 0.9 s per noisy shot for the two 27-qubit circuits, so a harness run at 100 shots takes about 3 minutes. The take's backend text is "Aer depolarizing stand-in (Forte run pending)". No `clifford_t` helper exists.
2. **Y readout sign:** no flip needed (|+i_L> reads 0 with probability 1; lesson 4 enc_Y ideal 0.235 at 1000 shots).
3. **`meta.nqubits`** is the circuit's real qubit count (9n encoded, n bare), so the capacity filter never drops a bare circuit.
4. **Lesson 4 keep rule:** block A is not a value block; its corrected bit drives keep_if_zero. All three P(1) values apply keep_if_zero. (Discard handling of A superseded by amendment 8.)
5. **Story steps without helpers:** at the physical level the Checks step covers the readout rotation and measurement layers (plaquettes light, correction rings appear); Readout is a one-layer step that highlights the logical operator. Lesson 1 without injection shows a one-layer "No error injected" step.
6. **Task 3 interface:** the bridge entry is `circuits_subset(indices_json)`; the harness `run` still executes all built circuits (harness capacity is 29).
7. **Task 6 split:** README, spec addendum and this amendment are written by the planner; `dev/record_demo.py` is retargeted after the harness smoke test, since the video is last.
8. **Keep rule:** a shot is kept only if every block with checks is clean or corrected; lesson 4's post-selection block A discards the shot too when its syndrome is outside the table.
9. **Qollab handoff: the bridge runs the job (Oct 11, after the placement page's Qollab runs).** On Qollab the page's `backend.run(pythonList)` failed with "This borrowed proxy was automatically destroyed at the end of a generator" (placement page, built-in BasicSimulator), and a scratch experiment showed JS-side handoff forms behave differently per backend. The form that cannot fail is running inside Python: the JS `backend` global is a proxy of the Python backend object, so passing it back gives Python the real backend. Change: `qec/bridge.py` gains `run_subset(backend, indices_json, shots)`: takes the built circuits at those indices (as circuits_subset does), calls `backend.run(circuits, shots=int(shots)).result()` and returns `json.dumps([result.get_counts(i) for i in range(len(circuits))])` in index order. main.js `submit()` replaces the circuits fetch, materialize, backend.run, the status polling loop and get_counts with `const sub = JSON.parse(await bridge.run_subset.callPromising(backend, JSON.stringify(included), settings.shots));` then `counts[included[k]] = sub[k]`; the status line before it says the job with the included circuits is running on the backend and that noise-model simulators run one shot at a time. materialize, sleep and toPlain are removed if unused. main.js ends with `await new Promise(() => {});` after its start-up call, because Qollab routes remote backend requests only while main.js is still running (Qollab's own multi-file example does the same). `dev/bridge_shim.js`: `run_subset = (_backend, ...a) => call('run_subset', a)` with `.callPromising` alias; `globalThis.backend = { name: ... }` only; `dev/serve.py` adds a harness backend class whose run(circuits, shots) returns an object with result().get_counts(i) from `simulate(list(circuits), shots, noise_model=NOISE)`, routes `/api/run_subset` to `bridge.run_subset(HarnessBackend(), *args)`, and drops `/api/run`, `run()`, and the count-returning `circuits`/`circuits_subset` branches that nothing calls any more. Tests first: `tests/test_bridge.py::test_run_subset_counts_in_order` with qiskit's BasicSimulator on lesson 1 circuits (build with the smallest settings the file already uses; each dict sums to shots); `tests/test_dev_server.py` round trip updated to run_subset. If BasicSimulator rejects a gate of the lesson circuits, stop and report the gate names instead of adding a transpile step.
