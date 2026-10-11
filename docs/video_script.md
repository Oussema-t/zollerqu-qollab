# Voice script, Unlocked (target 2:30, screen: the Qollab tab, Code view, page running)

0:00  [Page loaded on the Built-in simulator, lesson 1 tab] "Suppose you want a quantum computer whose answers survive its own noise. The way to get there is a logical qubit: here, nine physical qubits in a three by three patch of the rotated surface code. Eight parity checks watch the patch; when one qubit is hit, the checks point at it and we flip that bit back in software."

0:20  [Lesson 1: pick inject X on qubit 4, press Submit; cards appear] "Lesson one shows that live. We inject an error on one qubit. The checks around it fire, the ring marks the qubit the code corrected, and the logical value comes out right. The bare qubit beside it takes the same hit and has no idea."

0:40  [Lesson 2 tab] "Lesson two is a gate between two patches. Nine physical CNOTs, each from a qubit of A to the same qubit of B, make one logical CNOT and leave a logical Bell pair. No gate touches two qubits of the same patch, so a faulty gate makes at most one error per patch and the checks still catch it."

1:00  [Lesson 3 tab, then Watch the story, Physical level] "Lesson three prepares the state we actually care about: the T state. Every gate so far was Clifford. The T gate is the one a surface code cannot apply directly, so the resource is the state T applied to plus. Physically: H on qubit zero of patch A, then T, then the sixteen gates of the encoder spread that one qubit over the whole patch. We read the patch in X and in Y; the T state gives one in 14.6 percent of shots in each basis, which no Clifford state would."

1:30  [Lesson 4 tab, Watch the story; it opens at the physical level] "Lesson four moves that T state from patch A to patch C. Prepare: A gets H, T and the encoder as before; B gets H and the encoder, so it is logical plus; C is encoded as logical zero. Bell pair: nine CNOTs from B to C, qubit to qubit. Bell measurement: nine CNOTs from A to B. Then every qubit of A is read in X and every qubit of B in Z; those two logical bits say which Pauli flip C needs, and we apply it in software. Checks on A and B catch single errors first."

2:00  [Story reaches Result; scroll to the static logical circuit underneath] "The ideal logical circuit is drawn underneath for reference; the animation above is what the 27 physical qubits actually do. Patch C ends holding the T state: read in X and in Y, 14.6 percent ones each, up to the noise of the run."

2:15  [Back to the lesson 4 cards] "The cards count it: shots discarded when a check pattern matches no single error, shots corrected, and the gap to ideal, coded against bare. Lesson four needs 27 qubits, so on the built-in simulator it comes from a saved run; on a Forte simulator it runs live. Code, 104 tests and the write-up are in the repo, MIT."

2:30  end
