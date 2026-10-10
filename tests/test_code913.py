import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from qec import code913 as c


def pauli(kind, qubits, n=9):
    """Pauli acting with `kind` on the given qubit numbers (qiskit labels are little-endian)."""
    lab = ["I"] * n
    for q in qubits:
        lab[n - 1 - q] = kind
    return Pauli("".join(lab))


def overlap(a, b):
    return abs(np.vdot(np.asarray(a), np.asarray(b)))


def codewords():
    """|0_L> by projecting |0...0> onto the +1 space of the X stabilizers; |1_L> = X_L |0_L>."""
    v = Statevector.from_label("0" * 9)
    for s in c.X_STABS:
        v = (v + v.evolve(pauli("X", s))) / 2
    v = v / np.linalg.norm(np.asarray(v))
    return v, v.evolve(pauli("X", c.X_L))


def encoded(state):
    qc = QuantumCircuit(9)
    qc.initialize(state, 0)
    c.encode(qc, 0)
    return Statevector(qc)


def test_indexing():
    assert c.BLOCK_NAMES == "ABC"
    assert c.data(0, 8) == 8 and c.data(1, 4) == 13 and c.data(2, 0) == 18
    assert c.n_qubits(1) == 9 and c.n_qubits(3) == 27
    qc = c.new_circuit(2)
    assert qc.num_qubits == 18 and qc.num_clbits == 18


def test_stabilizers_commute_and_logicals_are_logical():
    stabs = [pauli("X", s) for s in c.X_STABS] + [pauli("Z", s) for s in c.Z_STABS]
    for a, b in itertools.combinations(stabs, 2):
        assert a.commutes(b)
    xl, zl = pauli("X", c.X_L), pauli("Z", c.Z_L)
    for s in stabs:
        assert s.commutes(xl) and s.commutes(zl)
    assert not xl.commutes(zl)
    assert c.X_L == (0, 1, 2) and c.Z_L == (0, 3, 6)


def test_codewords_are_stabilized_and_logicals_act():
    z0, z1 = codewords()
    for v in (z0, z1):
        for s in c.X_STABS:
            assert overlap(v.evolve(pauli("X", s)), v) == pytest.approx(1.0)
            assert np.allclose(np.asarray(v.evolve(pauli("X", s))), np.asarray(v))
        for s in c.Z_STABS:
            assert np.allclose(np.asarray(v.evolve(pauli("Z", s))), np.asarray(v))
    assert np.allclose(np.asarray(z0.evolve(pauli("Z", c.Z_L))), np.asarray(z0))
    assert np.allclose(np.asarray(z1.evolve(pauli("Z", c.Z_L))), -np.asarray(z1))
    assert np.allclose(np.asarray(z0.evolve(pauli("X", c.X_L))), np.asarray(z1))


def test_encoder_maps_basis_and_generic_states():
    z0, z1 = codewords()
    assert overlap(encoded([1, 0]), z0) > 1 - 1e-9
    assert overlap(encoded([0, 1]), z1) > 1 - 1e-9
    a, b = np.cos(0.4), np.exp(0.7j) * np.sin(0.4)
    assert overlap(encoded([a, b]), a * z0 + b * z1) > 1 - 1e-9


def test_encode_is_sixteen_gates_after_its_mark():
    qc = c.new_circuit(2)
    c.encode(qc, 1)
    assert qc.metadata["phases"] == [(0, "encode B")]
    assert len(qc.data) == 16
    used = {qc.find_bit(q).index for ins in qc.data for q in ins.qubits}
    assert used == set(range(9, 18))


@pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_transversal_cx_is_logical_cnot(a, b):
    z = codewords()
    qc = QuantumCircuit(18)
    if a:
        qc.x(c.data(0, 0))
    if b:
        qc.x(c.data(1, 0))
    c.encode(qc, 0)
    c.encode(qc, 1)
    c.transversal_cx(qc, 0, 1)
    assert sum(ins.operation.name == "cx" for ins in qc.data[-9:]) == 9
    expected = Statevector(np.kron(np.asarray(z[a ^ b]), np.asarray(z[a])))
    assert overlap(Statevector(qc), expected) > 1 - 1e-9


def _syndrome(kind_err, i, stabs, kind_stab):
    e = pauli(kind_err, [i])
    return tuple(0 if pauli(kind_stab, s).commutes(e) else 1 for s in stabs)


@pytest.mark.parametrize("i", range(9))
def test_single_x_error_gives_the_z_table_entry(i):
    # measure every qubit in Z after |0_L> + X on qubit i: every outcome shows the same Z-check parities
    qc = QuantumCircuit(9)
    c.encode(qc, 0)
    qc.x(i)
    probs = Statevector(qc).probabilities_dict()
    synd = {tuple(sum(int(k[8 - q]) for q in s) % 2 for s in c.Z_STABS) for k, p in probs.items() if p > 1e-12}
    assert synd == {_syndrome("X", i, c.Z_STABS, "Z")}
    j = c.Z_TABLE[synd.pop()]
    assert j is not None
    # the correction X_j undoes X_i up to a stabilizer: X_i X_j |0_L> = |0_L>
    z0, _ = codewords()
    fixed = z0.evolve(pauli("X", [i])).evolve(pauli("X", [j]))
    assert np.allclose(np.asarray(fixed), np.asarray(z0))


@pytest.mark.parametrize("i", range(9))
def test_single_z_error_is_corrected_by_the_x_table(i):
    s = _syndrome("Z", i, c.X_STABS, "X")
    j = c.X_TABLE[s]
    assert j is not None
    z0, z1 = codewords()
    plus = (z0 + z1) / np.sqrt(2)
    fixed = plus.evolve(pauli("Z", [i])).evolve(pauli("Z", [j]))
    assert np.allclose(np.asarray(fixed), np.asarray(plus))


def test_tables_have_eight_entries_with_zero_meaning_no_correction():
    for t in (c.Z_TABLE, c.X_TABLE):
        assert len(t) == 8 and t[(0, 0, 0, 0)] is None
    assert c.Z_TABLE == {(0, 0, 0, 0): None, (0, 0, 1, 0): 0, (1, 0, 1, 0): 1, (1, 0, 0, 0): 2, (0, 1, 0, 0): 3,
                         (1, 1, 0, 0): 4, (0, 1, 0, 1): 7, (0, 0, 0, 1): 8}
    assert c.X_TABLE == {(0, 0, 0, 0): None, (1, 0, 0, 0): 0, (0, 0, 1, 0): 2, (1, 0, 0, 1): 3, (1, 1, 0, 0): 4,
                         (0, 1, 1, 0): 5, (0, 0, 0, 1): 6, (0, 1, 0, 0): 7}


def test_y_readout_of_plus_i_reads_zero():
    qc = QuantumCircuit(9)
    qc.h(0)
    qc.s(0)                                  # (|0> + i|1>)/sqrt2
    c.encode(qc, 0)
    c.rotate_block(qc, 0, "Y")
    probs = Statevector(qc).probabilities_dict()
    for k, p in probs.items():
        if p > 1e-12:
            v = [int(ch) for ch in reversed(k)]
            assert v[0] ^ v[1] ^ v[2] ^ v[3] ^ v[6] == 0, k


def test_inject_applies_named_pauli_and_marks():
    qc = c.new_circuit(1)
    c.inject(qc, 0, "Y", 7)
    assert [ins.operation.name for ins in qc.data] == ["y"]
    assert qc.find_bit(qc.data[0].qubits[0]).index == 7
    assert qc.metadata["phases"] == [(0, "inject Y on q7")]


def test_finish_rotates_and_measures_everything():
    qc = c.new_circuit(2)
    c.finish(qc, 2, ["X", "Y"])
    names = [(ins.operation.name, qc.find_bit(ins.qubits[0]).index) for ins in qc.data]
    assert sum(n == "measure" for n, _ in names) == 18
    rot = [x for x in names if x[0] != "measure"]
    assert rot == [("h", q) for q in range(9)] + [("sdg", 9), ("h", 9), ("h", 10), ("h", 11)]


def test_bare_helpers():
    qc = c.new_bare(2)
    assert qc.num_qubits == 2 and qc.num_clbits == 2
    c.finish_bare(qc, ["Z", "Y"])
    names = [ins.operation.name for ins in qc.data]
    assert names == ["sdg", "h", "measure", "measure"]


def test_phases_are_recorded_in_order_without_changing_gates():
    qc = c.new_circuit(2)
    c.mark(qc, "prepare A")
    qc.h(c.data(0, 0))
    c.encode(qc, 0)
    c.encode(qc, 1)
    c.transversal_cx(qc, 0, 1)
    c.finish(qc, 2, ["Z", "X"])
    names = [n for _, n in qc.metadata["phases"]]
    assert names == ["prepare A", "encode A", "encode B", "transversal CNOT A→B",
                     "readout rotation B (X)", "measure"]
    starts = [s for s, _ in qc.metadata["phases"]]
    assert starts == [0, 1, 17, 33, 42, 51]
    assert len(qc.data) == 1 + 16 + 16 + 9 + 9 + 18
