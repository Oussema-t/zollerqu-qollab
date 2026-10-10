# Logical teleportation on a tiny surface code

Three logical qubits, each a 2x2 patch of the [[4,1,2]] rotated surface code, running on IonQ's simulator. Four lessons from one job:

1. **One patch, one error.** Inject an X, Y or Z on any corner. The checks flag it; the bare qubit next to it has no idea.
2. **Logical CNOT.** Four physical CNOTs between two patches make a logical Bell pair. No SWAPs on trapped ions.
3. **Logical teleportation.** A state crosses from patch A to patch C through a Bell measurement. Pauli corrections are applied in software (the Pauli frame), so no feed-forward is needed.
4. **The teleported T gate.** T is not transversal. We teleport it in with a magic state, keep the shots that need no correction, and check that X and Y readouts both give 25 percent, which neither the input nor the magic state alone would.

## How to run

Press **Run on Qollab**, pick a backend (built-in simulator for instant results, IonQ Aria 1 noise model for realistic noise), then:

- choose shots (100 to 2000) and, for lesson 1, an injected error,
- press **Submit one job with all 16 circuits**, or **Load recorded take** to see a saved run without waiting (an IonQ Aria 1 noise-model take once recorded; until then a local stand-in),
- click through the four lesson tabs. Hover any face or edge for its meaning. "Show the physical circuit" draws the real gates.

IonQ backends allow one job per run, so all 16 circuits travel together. On local simulators you can resubmit freely.

## What the numbers mean

- **Flagged:** a check fired, the shot is discarded. Distance 2 detects any single error and corrects none.
- **Logical error of accepted shots:** gap between the measured and the ideal probability of reading logical 1.
- **Bare physical:** the same experiment with one physical qubit per logical qubit.
- Lesson 4 also shows the **kept fraction** after post-selecting on patch A reading 0, ideally one half.

## The code

Corner qubits q0 q1 / q2 q3. Checks: XXXX (face), Z0Z1 (top), Z2Z3 (bottom). Logical Z = Z0Z2, logical X = X0X1. Encoder: `h q2; cx q2,q3; cx q0,q1; cx q2,q0; cx q2,q1`. Every block gets two ancillas measured at the end, plus the checks that the final readout basis gives for free. Everything is measured at the end, so no mid-circuit measurement, reset or feed-forward is used.

## Honesty notes

The encoder is not fault tolerant. Post-selection throws shots away and would not scale. Lesson 4 discards half the shots because the S correction needs feed-forward. These are the same compromises the first logical-gate experiments on trapped ions and superconducting chips made.

## Files

`main.js` runs the page and the single job. `qec/` holds the physics in Python: `code412.py` (code), `lessons.py` (16 circuits), `analyze.py` (counts to numbers), `bridge.py` (JSON entry points). `tests/` run with `python -m pytest`. `data/recorded_aria1.json` is a saved run.

## Credits

Error detection with small codes on trapped ions: Linke et al., Sci. Adv. 3, e1701074 (2017). The [[4,2,2]] and [[4,1,2]] family: Gottesman, arXiv:1610.03507. Magic states and the T gadget: Bravyi and Kitaev, Phys. Rev. A 71, 022316 (2005). Teleportation: Bennett et al., Phys. Rev. Lett. 70, 1895 (1993). Multi-file and recorded-take patterns from the Qollab runtime docs.

Built during the Qollab x IonQ hackathon, Oct 9 to 11, 2026, by team ZollerQu. AI tools used: Claude Code for planning and implementation. MIT licence.

This effort is supported by Qollab & IonQ.
