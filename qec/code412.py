"""Building blocks for the [[4,1,2]] rotated surface code.

Block b occupies data qubits 4b..4b+3 (local index i at 4b+i). With n blocks,
ancilla ax(b) = 4n + 2b measures XXXX and az(b) = 4n + 2b + 1 measures ZZZZ.
Layout of one patch (local indices):

    q0 --- q1     top edge:    Z0Z1 stabilizer, X_L = X0X1
    |  X   |      face:        XXXX stabilizer
    q2 --- q3     bottom edge: Z2Z3 stabilizer, Z_L = Z0Z2 (left edge)
"""
from qiskit import QuantumCircuit


def data(b, i):
    return 4 * b + i


def anc_x(nblocks, b):
    return 4 * nblocks + 2 * b


def anc_z(nblocks, b):
    return 4 * nblocks + 2 * b + 1


def n_qubits(nblocks):
    return 6 * nblocks


def new_circuit(nblocks):
    return QuantumCircuit(n_qubits(nblocks), n_qubits(nblocks))


def encode(qc, b):
    """Encode the state on local q0 into the block. Not fault tolerant."""
    q = [data(b, i) for i in range(4)]
    qc.h(q[2])
    qc.cx(q[2], q[3])
    qc.cx(q[0], q[1])
    qc.cx(q[2], q[0])
    qc.cx(q[2], q[1])


def transversal_cx(qc, ctrl_b, targ_b):
    for i in range(4):
        qc.cx(data(ctrl_b, i), data(targ_b, i))


def inject(qc, b, pauli, i):
    getattr(qc, pauli.lower())(data(b, i))


def extract_syndromes(qc, nblocks, b):
    ax, az = anc_x(nblocks, b), anc_z(nblocks, b)
    q = [data(b, i) for i in range(4)]
    # XXXX ancilla CX order q0, q2, q1, q3 avoids the hook error: an X fault on ax
    # after the second CX would leave X2X3 = X_L * XXXX with order q0..q3, a logical
    # error no check sees. With this order the leftover is X1X2X3 (= X0 * XXXX),
    # X1X3 or X3, all caught by the Z checks. The ZZZZ ancilla keeps q0..q3: its
    # mid-sequence leftover Z2Z3 is a stabilizer, while reordering would give Z_L.
    qc.h(ax)
    for qi in (q[0], q[2], q[1], q[3]):
        qc.cx(ax, qi)
    qc.h(ax)
    for qi in q:
        qc.cx(qi, az)


def rotate_block(qc, b, basis):
    q = [data(b, i) for i in range(4)]
    _rotate(qc, q, basis)


def _rotate(qc, q, basis):
    """Rotate so that a Z measurement reports the wanted logical observable."""
    if basis == "Z":
        return
    if basis == "X":
        for qi in q:
            qc.h(qi)
        return
    if basis == "Y":
        qc.sdg(q[0])
        qc.h(q[0])
        qc.h(q[1])
        return
    raise ValueError(f"unknown basis {basis!r}")


def finish(qc, nblocks, bases):
    """Syndrome extraction on every block, readout rotations, measure everything."""
    for b in range(nblocks):
        extract_syndromes(qc, nblocks, b)
    for b, basis in enumerate(bases):
        rotate_block(qc, b, basis)
    qc.measure(range(qc.num_qubits), range(qc.num_clbits))


# Bare physical references: one qubit per logical qubit, no ancillas.

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
