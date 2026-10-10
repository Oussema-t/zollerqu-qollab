# Story page: steps, states and captions for lessons 1, 2 and 4 (draft for review)

Lesson 3 is the prototype (scratchpad preview/story.html). Same structure for the others: logical steps with a caption, a "State now" expression with the changed part highlighted, the physical level following the recorded phases. α = cos(π/8) ≈ 0.92, β = sin(π/8) ≈ 0.38, ω = e^{iπ/4}.

## Lesson 1: one patch, one error

Wires: A only. The story follows the enc_Z circuit; if an error is injected (settings), it appears as a step.

| Step | Caption | State now |
|---|---|---|
| Prepare | One logical qubit: four physical qubits on the corners of a square, plus two helpers. Encoding spreads |0⟩ over the patch so that no single corner holds it. | \|0_L⟩ = (\|0000⟩ + \|1111⟩)/√2 |
| Error strikes (only if injected) | A single X hits corner q{i}. The logical state is now outside the code space. Nothing in the hardware notices yet. | X_{i}\|0_L⟩ = (\|0100⟩ + \|1011⟩)/√2 for i = 1. Line: "The top check Z0Z1 now reads −1." |
| Checks | The helpers read XXXX and ZZZZ. The Z0Z1 and Z2Z3 parities come straight from the data readout. A flipped check flags the shot. | unchanged; line: "Flipped checks: Z0Z1, ZZZZ" (or "none" without injection) |
| Readout | A is read in Z. Logical Z is the left edge, Z0 Z2. | after measurement |
| Result | With the error, nearly every shot is flagged and discarded; the few that survive are the ones where a second fault hid the first. The bare qubit next to it has the same error and reports a wrong answer every time, with no warning. Without the error, the flagged share is the noise floor of the device. | cards: flagged %, coded pts off ideal (ideal 0), bare pts off ideal |

Physical level: prepare (none for Z), encode A, inject (red burst at q1), syndrome extraction A, measure. The injected corner stays outlined red from the strike onward; at measure the Z0Z1 cap and ZZZZ helper glow at their fire rates (near 100 %).

## Lesson 2: logical CNOT

Wires: A, B. Circuit enc_ZZ (both read in Z); the XX circuit is mentioned in the result.

| Step | Caption | State now |
|---|---|---|
| Prepare | A is encoded as \|+_L⟩, B as \|0_L⟩. | \|+⟩_A \|0⟩_B |
| Logical CNOT | A logical CNOT from A to B. It is four physical CNOTs, one between each pair of matching corners, and nothing else. No decoding, no re-encoding: the gate is transversal. | (\|00⟩ + \|11⟩)_AB / √2, highlighted; line: "A and B are a logical Bell pair." |
| Checks | Both patches are checked. The CNOTs copy any error from A onto B, which is why transversal gates are safe: an error never spreads inside a patch. | unchanged |
| Readout | Both read in Z. The two logical bits must agree. The same circuit read in X must agree too; only entanglement does both. | after measurement |
| Result | ZZ parity odd in {p1 %} of accepted shots (ideal 0). XX: {p1 %}. Bare qubits: {bare %}. | cards |

Physical level: prepare A (H on q0), encode A, encode B, transversal CNOT A→B (four white arrows), syndrome extraction A, B, measure.

## Lesson 4: the T gate

Wires: A (ψ), B (magic state). Circuit enc_X (B read in X); Y is mentioned in the result.

| Step | Caption | State now |
|---|---|---|
| Prepare | A carries ψ. B is prepared as the magic state T\|+⟩ before encoding: H then T on one corner, then the encoder. The T gate is applied here, on a fresh patch, not on the data. | \|ψ⟩_A ⊗ (\|0⟩ + ω\|1⟩)_B / √2 |
| Logical CNOT B→A | A transversal CNOT with the magic patch as control and the data patch as target. | ( \|0⟩_B (α\|0⟩ + β\|1⟩)_A + ω\|1⟩_B (α\|1⟩ + β\|0⟩)_A ) / √2 |
| Checks | Both patches are checked as before. | unchanged |
| Readout | A is read in Z. Grouping by A's outcome: when A reads 0, B holds Tψ. When A reads 1, B holds a state that would need an S gate to fix; that needs feed-forward, so here we keep only the A = 0 shots. Half the shots survive. | ½ [ \|0⟩_A (α\|0⟩ + ωβ\|1⟩)_B + \|1⟩_A (β\|0⟩ + ωα\|1⟩)_B ] with the first B factor highlighted and labelled "= Tψ" |
| Result | B read in X gives 1 in 25 % of kept shots ideally, and the same in Y. Without the T gate X would give 14.6 % and Y 50 %. Kept: {kept %} (ideal 50 %). | cards: kept %, coded pts off ideal (25), bare pts |

Physical level: prepare A (Ry), encode A, prepare B (H, T), encode B, transversal CNOT B→A, syndrome extraction A, B, readout rotation B (X), measure.

## Open questions for Tushar

1. Lesson 1 without injection has no "error strikes" step. Keep the step with the caption "No error injected this time" or drop it?
2. Lesson 2: show the ZZ circuit only, or alternate ZZ and XX in the story (two short readout steps)?
3. Lesson 4: show the discarded A = 1 branch in grey in the state panel (as above), or hide it?
