export const LESSONS = {
  1: {
    title: "One patch, one error",
    html: `
<p>A logical qubit here is nine physical qubits in a 3x3 grid, a patch of the <strong>rotated surface code</strong>. Eight <strong>checks</strong> watch it: four measure X on a group of neighbours, four measure Z. Logical Z runs down the left column (<em>Z0Z3Z6</em>), logical X along the top row (<em>X0X1X2</em>).</p>
<p>Any single error flips a pattern of checks that points at one qubit. The code can therefore not only <strong>flag</strong> the shot but <strong>correct</strong> it: we flip that qubit's recorded bit in software. That is error correction, distance 3. Two errors at once can fool it, and a few patterns match no single error; those shots are discarded.</p>
<p><strong>Try it:</strong> inject an error in the setup and resubmit. The bare physical qubit next to it gets the same error and has no idea. The patch lights up where the checks fire, and a ring marks the qubit that was corrected.</p>
<p class="note">The checks are read from the final measurement, not mid-circuit, so a Z readout sees only the Z checks (which catch X errors) and an X readout only the X checks. The encoder is not fault tolerant: an error inside it can slip through. Real experiments start the same way.</p>`,
  },
  2: {
    title: "Logical CNOT is nine physical CNOTs",
    html: `
<p>For this code a logical CNOT is <strong>transversal</strong>: CNOT qubit 0 to qubit 0, 1 to 1, and so on through all nine pairs. No gate ever touches two qubits of the same patch, so one faulty gate makes at most one error per patch, and the checks still catch and correct it.</p>
<p>We prepare patch A in |+&gt; and patch B in |0&gt;, apply the nine CNOTs and get a <strong>logical Bell pair</strong>. Read both in Z: the parities agree. Read both in X: they agree too. That pair of facts is entanglement.</p>
<p class="note">On IonQ's trapped ions every qubit talks to every other, so these nine CNOTs need no SWAP gates. On a grid chip the patches would have to sit next to each other.</p>
<p class="note">18 qubits. Each patch is corrected on its own before the parity is taken.</p>`,
  },
  3: {
    title: "A T state on a patch",
    html: `
<p>Every gate so far was Clifford. The <strong>T gate</strong> is not: it is not transversal in this code, or in any code of this kind, yet a computer needs it. The way out is a resource state, <strong>T|+&gt;</strong>, made once and used later. With it, a T gate can be teleported onto any data.</p>
<p>Physically it is made on one qubit: H then T on q0 of the patch, then the same 16-gate encoder as in lesson 1 spreads that one qubit over all nine. The result is the logical state T|+<sub>L</sub>&gt;.</p>
<p>Check: read the patch in X and in Y. Each should give 1 in <strong>14.6 percent</strong> of shots. A plain |+&gt; without the T would give 0 percent in X and 50 percent in Y, so both numbers together show the T phase is there.</p>
<p class="note">9 qubits. The X readout is checked and corrected; the Y readout has no checks. T on one bare qubit before encoding is unprotected: real machines distil many noisy T states into one clean one, and spend most of their effort doing so.</p>
<p class="note">The Y readout has no checks, so a Y error on any of its nine qubits goes uncorrected; expect the coded Y result to sit above the bare one on a noisy run. The X readout is checked and corrected.</p>`,
  },
  4: {
    title: "Teleporting the T state",
    html: `
<p>Patch A holds the T state from lesson 3, <em>T|+&gt;</em>. Patches B and C share a logical Bell pair. A Bell measurement on A and B is a transversal CNOT followed by reading A in X and B in Z. Patch C now holds the T state, up to a Pauli correction that depends on the two readouts.</p>
<p>No feed-forward is needed: we read C and apply the correction <strong>in software</strong>, flipping the recorded bit when the readouts say so. Hardware experiments do exactly this and call it the Pauli frame. When C is read in X, flip it if A read 1; when C is read in Y, flip it if A or B read 1, but not both.</p>
<p>Check: in both X and Y the chance of reading 1 should be 14.6 percent, the same as the T state before it travelled.</p>
<p class="note">Three patches, 27 qubits. That fits only the 29-qubit Forte simulators; on other backends this lesson is shown from the recorded take. Each patch is corrected before the frame is applied.</p>
<p class="note">C's Y readout has no checks, so a Y error on any of its nine qubits goes uncorrected; expect the coded Y result to sit above the bare one on a noisy run. The X readout is checked and corrected.</p>`,
  },
};
