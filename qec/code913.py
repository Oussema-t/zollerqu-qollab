"""Building blocks for the rotated [[9,1,3]] surface code.

Block b occupies data qubits 9b..9b+8 (local index i at 9b+i), row-major on a 3x3 grid:

    q0 q1 q2      X stabilizers: (0,1,3,4), (4,5,7,8) bulk; (2,5) right, (3,6) left
    q3 q4 q5      Z stabilizers: (1,2,4,5), (3,4,6,7) bulk; (0,1) top, (7,8) bottom
    q6 q7 q8      X_L = X0X1X2 (top row), Z_L = Z0Z3Z6 (left column)

No helper qubits: the checks of the readout basis are parities of the final measurement,
and single errors are corrected by the lookup tables below.
"""
from qiskit import QuantumCircuit

BLOCK_NAMES = "ABC"

X_STABS = ((0, 1, 3, 4), (4, 5, 7, 8), (2, 5), (3, 6))
Z_STABS = ((1, 2, 4, 5), (3, 4, 6, 7), (0, 1), (7, 8))
X_L = (0, 1, 2)
Z_L = (0, 3, 6)

# Syndrome (bits in stabilizer order) -> local qubit to flip, None for no correction.
# Z_TABLE reads the Z checks (Z readout, catches X errors); X_TABLE reads the X checks (X readout, Z errors).
# A syndrome not in the table is discarded.
Z_TABLE = {(0, 0, 0, 0): None, (0, 0, 1, 0): 0, (1, 0, 1, 0): 1, (1, 0, 0, 0): 2, (0, 1, 0, 0): 3,
           (1, 1, 0, 0): 4, (0, 1, 0, 1): 7, (0, 0, 0, 1): 8}
X_TABLE = {(0, 0, 0, 0): None, (1, 0, 0, 0): 0, (0, 0, 1, 0): 2, (1, 0, 0, 1): 3, (1, 1, 0, 0): 4,
           (0, 1, 1, 0): 5, (0, 0, 0, 1): 6, (0, 1, 0, 0): 7}


def data(b, i):
    return 9 * b + i


def n_qubits(nblocks):
    return 9 * nblocks


def mark(qc, name):
    """Record the start of a named phase for the spacetime view. Gates are unchanged."""
    qc.metadata.setdefault("phases", []).append((len(qc.data), name))


def new_circuit(nblocks):
    return QuantumCircuit(n_qubits(nblocks), n_qubits(nblocks))


def encode(qc, b):
    """Encode the state on local q0 into the block. Not fault tolerant."""
    mark(qc, f"encode {BLOCK_NAMES[b]}")
    q = [data(b, i) for i in range(9)]
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])
    for i in (4, 5, 6, 7):
        qc.h(q[i])
    for t in (0, 1, 3):
        qc.cx(q[4], q[t])
    qc.cx(q[5], q[2])
    qc.cx(q[6], q[3])
    for t in (0, 1, 2, 3, 8):
        qc.cx(q[7], q[t])


def transversal_cx(qc, ctrl_b, targ_b):
    mark(qc, f"transversal CNOT {BLOCK_NAMES[ctrl_b]}→{BLOCK_NAMES[targ_b]}")
    for i in range(9):
        qc.cx(data(ctrl_b, i), data(targ_b, i))


def inject(qc, b, pauli, i):
    mark(qc, f"inject {pauli} on q{i}")
    getattr(qc, pauli.lower())(data(b, i))


def rotate_block(qc, b, basis):
    """Rotate so that a Z measurement reports the wanted logical observable."""
    if basis == "Z":
        return
    mark(qc, f"readout rotation {BLOCK_NAMES[b]} ({basis})")
    q = [data(b, i) for i in range(9)]
    if basis == "X":
        for qi in q:
            qc.h(qi)
    elif basis == "Y":
        # Y_L = i X_L Z_L = Y0 X1 X2 Z3 Z6
        qc.sdg(q[0])
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
    else:
        raise ValueError(f"unknown basis {basis!r}")


def finish(qc, nblocks, bases):
    """Readout rotations, then measure everything."""
    for b, basis in enumerate(bases):
        rotate_block(qc, b, basis)
    mark(qc, "measure")
    qc.measure(range(qc.num_qubits), range(qc.num_clbits))


# Bare physical references: one qubit per logical qubit.

def new_bare(n):
    return QuantumCircuit(n, n)


def finish_bare(qc, bases):
    for q, basis in enumerate(bases):
        if basis == "X":
            qc.h(q)
        elif basis == "Y":
            qc.sdg(q)
            qc.h(q)
        elif basis != "Z":
            raise ValueError(f"unknown basis {basis!r}")
    qc.measure(range(qc.num_qubits), range(qc.num_clbits))
