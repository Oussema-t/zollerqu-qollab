export const LESSONS = {
  1: {
    title: "One patch, one error",
    html: `
<p>A logical qubit here is four physical qubits on the corners of a square. Three <strong>checks</strong> watch it:
the face measures <em>XXXX</em>, the top and bottom edges measure <em>Z0Z1</em> and <em>Z2Z3</em>.
Logical Z lives on the left edge (<em>Z0Z2</em>), logical X on the top edge (<em>X0X1</em>).</p>
<p>Any single X, Y or Z error on any corner flips at least one check. The code cannot say which corner,
so it cannot fix it. It can <strong>flag</strong> the shot, and we throw flagged shots away. That is error detection, distance 2.</p>
<p><strong>Try it:</strong> inject an error in the setup and resubmit. The bare physical qubit next to it gets the same error and
has no idea. The patch lights up.</p>
<p class="note">The encoder is not fault tolerant: an error inside those five gates can slip through. Real experiments
start the same way, then post-select.</p>`,
  },
  2: {
    title: "Logical CNOT is four physical CNOTs",
    html: `
<p>For this code family a logical CNOT is <strong>transversal</strong>: CNOT corner 0 to corner 0, 1 to 1, 2 to 2, 3 to 3.
No gate ever touches two qubits of the same patch, so one faulty gate makes at most one error per patch, and the checks still catch it.</p>
<p>We prepare patch A in |+&gt; and patch B in |0&gt;, apply the four CNOTs and get a <strong>logical Bell pair</strong>.
Read both in Z: the parities agree. Read both in X: they agree too. That pair of facts is entanglement.</p>
<p class="note">On IonQ's trapped ions every qubit talks to every other, so these four CNOTs need no SWAP gates.
On a grid chip the patches would have to sit next to each other.</p>`,
  },
  3: {
    title: "Teleporting a logical qubit",
    html: `
<p>Patch A holds a state, <em>ry(pi/4)|0&gt;</em>. Patches B and C share a logical Bell pair. A Bell measurement on A and B
is a transversal CNOT followed by reading A in X and B in Z. Patch C now holds the state, up to a Pauli correction
that depends on the two readouts.</p>
<p>No feed-forward is needed: we read C and apply the correction <strong>in software</strong>, flipping the recorded
bit when the readouts say so. Hardware experiments do exactly this and call it the Pauli frame.</p>
<p>Check: in both bases the chance of reading 1 should be 14.6 percent, the same as the input state.</p>`,
  },
  4: {
    title: "The T gate is a teleportation too",
    html: `
<p>Every gate so far was Clifford. The <strong>T gate</strong> is not transversal in this code, or in any code of this kind.
The way out is to prepare a <em>magic state</em> T|+&gt; in its own patch, then teleport the gate onto the data.</p>
<p>Patch B holds the magic state, patch A holds the same input as lesson 3, <em>ry(pi/4)|0&gt;</em>. A transversal CNOT from B to A and a Z readout of A leave
T|psi&gt; on B half the time. The other half needs an S correction that requires feed-forward, which IonQ's simulator
does not offer, so we <strong>keep only the shots where A read 0</strong>. The kept fraction should be about one half.</p>
<p>Check: reading B in X and in Y should both give 1 about 25 percent of the time. Without the T gate, X would give 14.6 percent and Y 50 percent.
We do not use |+&gt; as the input on purpose: T|+&gt; is the magic state itself, so nothing would visibly change.</p>
<p class="note">Fault-tolerant computers spend most of their effort making clean magic states. This lesson is why.</p>`,
  },
};
