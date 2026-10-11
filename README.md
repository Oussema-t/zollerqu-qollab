# Logical teleportation on a tiny surface code

**What it is.** Three logical qubits, each a distance-3 rotated surface code patch, on IonQ's simulators: four lessons from one job, and a story page that animates each lesson at the logical and the physical level.

**Why it matters, and for whom.** Logical qubits are how quantum computers will outlive their own noise, and the first logical-gate experiments on trapped ions used exactly these small codes. This page is for learners and teachers: inject an error and watch the checks point at it, make a logical Bell pair, teleport a logical state between patches, and teleport in a T gate, each next to the same experiment on bare qubits.

**How the quantum part works.** A 16-gate encoder prepares each patch; every patch is measured once at the end and the four checks of that basis are parities of the measured bits; a lookup table corrects the one qubit a single error would produce, and Pauli corrections are tracked in software (details under "The code").

**How to run it.** Run on Qollab, pick Built-in simulator (24q) for lessons 1, 2 and 4 (lesson 3 then comes from the recorded take) or an IonQ Forte simulator (29q) for all four, press Submit. Demo video: `docs/demo.mp4` in the repo (public link added at submission).

Three logical qubits, each a 3x3 patch of the rotated surface code (distance 3), running on IonQ's simulators. Four lessons from one job, and a story page that plays each lesson as an animation at two levels: the logical circuit, and the physical gates on every qubit of every patch.

1. **One patch, one error.** Inject an X, Y or Z on any of the nine qubits. The checks point at it and the code corrects it in software; the bare qubit next to it has no idea.
2. **Logical CNOT.** Nine physical CNOTs between two patches make a logical Bell pair. No SWAPs on trapped ions.
3. **Logical teleportation.** A state crosses from patch A to patch C through a Bell measurement. Pauli corrections are applied in software (the Pauli frame), so no feed-forward is needed. 27 qubits.
4. **The teleported T gate.** T is not transversal. We teleport it in with a magic state, keep the shots that need no correction, and check that X and Y readouts both give 25 percent, which neither the input nor the magic state alone would.

## How to run

Press **Run on Qollab**, pick a backend, then:

- choose shots (100 to 2000) and, for lesson 1, an injected error,
- press **Submit one job with all circuits that fit**, or **Load recorded take** to see a saved run without waiting,
- click through the four lesson tabs. Hover any plaquette or logical operator for its meaning. **Watch the story** opens the animated page for the current lesson; the Logical / Physical toggle switches level at the same point in time.

Qubit budget: lessons 1, 2 and 4 use 9, 18 and 18 qubits and run on every backend. Lesson 3 uses 27 qubits, which fits only the 29-qubit Forte simulators. On other backends the page submits the circuits that fit and shows lesson 3 from the recorded take, and says so in the status line.

IonQ backends allow one job per run, so all circuits travel together. On local simulators you can resubmit freely.

## What the numbers mean

- **Discarded by checks:** the check pattern matched no single error, so the shot is thrown away.
- **Corrected by the code:** the check pattern pointed at one qubit and its recorded bit was flipped.
- **Logical error after correction:** gap between the measured and the ideal probability of reading logical 1, over the kept shots. The charts also show the clean shots alone and the raw result with no correction at all.
- **Same experiment on bare qubits:** one physical qubit per logical qubit, same gates, same noise.
- Lesson 4 also shows the **kept fraction** after post-selecting on patch A reading 0, ideally one half.

## The code

Nine data qubits in a 3x3 grid, row-major: q0 q1 q2 / q3 q4 q5 / q6 q7 q8. Stabilizers: X on (0,1,3,4), (4,5,7,8), (2,5), (3,6); Z on (1,2,4,5), (3,4,6,7), (0,1), (7,8). Logical X is the top row X0X1X2, logical Z the left column Z0Z3Z6. Encoder, 16 gates: `cx 0,1; cx 0,2; h 4,5,6,7; cx 4,0; cx 4,1; cx 4,3; cx 5,2; cx 6,3; cx 7,0; cx 7,1; cx 7,2; cx 7,3; cx 7,8`.

There are no helper qubits. Every patch is measured once at the end, in Z or in X after a layer of Hadamards; the four checks of that basis are parities of the measured bits. A lookup table maps each nonzero check pattern to the one qubit a single error on that qubit would produce, and the correction flips that bit before the logical value is read. Patterns outside the table are discarded. The Y readout of lesson 4 uses Y0 X1 X2 Z3 Z6 and has no checks. Everything is measured at the end, so no mid-circuit measurement, reset or feed-forward is used.

## Honesty notes

The encoder is not fault tolerant. Checks are read only once, from the final measurement, so only the errors the readout basis can see are corrected. Two errors in one patch can be miscorrected into a logical flip. Lesson 4 discards half the shots because the S correction needs feed-forward. The recorded take is a local Aer stand-in with depolarizing noise until a Forte simulator run replaces it. These are the same compromises the first logical-gate experiments on trapped ions and superconducting chips made.

## Files

`main.js` runs the page and the single job; `ui/` draws patches, cards, charts and the story page (`story.js`, `storyPhysical.js`, `storyText.js`). `qec/` holds the physics in Python: `code913.py` (code, encoder, lookup tables), `lessons.py` (16 circuits), `analyze.py` (counts to numbers), `spacetime.py` (gate layout for the story), `bridge.py` (JSON entry points), `aer_compat.py` (local Aer only). `tests/` run with `python -m pytest`. `data/recorded_aria1.json` is the saved run; `tools/record_take.py` regenerates it. `dev/` is a local harness that serves the page with Aer in place of the Qollab backend.

## Credits

Rotated surface code: Bombin and Martin-Delgado, Phys. Rev. A 76, 012305 (2007); Tomita and Svore, Phys. Rev. A 90, 062320 (2014). Figure conventions for patches and gates: Google Quantum AI, Nature 614, 676 (2023). Error correction with small codes on trapped ions: Linke et al., Sci. Adv. 3, e1701074 (2017). Magic states and the T gadget: Bravyi and Kitaev, Phys. Rev. A 71, 022316 (2005). Teleportation: Bennett et al., Phys. Rev. Lett. 70, 1895 (1993). Multi-file and recorded-take patterns from the Qollab runtime docs.

Built during the Qollab x IonQ hackathon, Oct 9 to 11, 2026, by team ZollerQu. AI tools used: Claude Code for planning and implementation. MIT licence.

This effort is supported by Qollab & IonQ.
