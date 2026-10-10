import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from qec import code412 as c

SQ2 = np.sqrt(2)


def z0():
    return (Statevector.from_label("0000") + Statevector.from_label("1111")) / SQ2


def z1():
    return (Statevector.from_label("1100") + Statevector.from_label("0011")) / SQ2


def overlap(a, b):
    return abs(np.vdot(np.asarray(a), np.asarray(b)))


def test_indexing():
    assert c.data(0, 3) == 3 and c.data(2, 1) == 9
    assert c.anc_x(3, 0) == 12 and c.anc_z(3, 0) == 13
    assert c.anc_x(3, 2) == 16 and c.anc_z(3, 2) == 17
    assert c.n_qubits(3) == 18


def test_codewords_are_stabilized():
    for s in ("XXXX", "IIZZ", "ZZII"):  # XXXX, Z0Z1, Z2Z3 in little-endian labels
        for v in (z0(), z1()):
            assert overlap(v.evolve(Pauli(s)), v) == pytest.approx(1.0)
    assert overlap(z0().evolve(Pauli("IZIZ")), z0()) == pytest.approx(1.0)   # Z_L = Z0Z2
    assert np.allclose(np.asarray(z1().evolve(Pauli("IZIZ"))), -np.asarray(z1()))
    assert overlap(z0().evolve(Pauli("IIXX")), z1()) == pytest.approx(1.0)   # X_L = X0X1


def test_encoder_encodes_generic_state():
    a, b = np.cos(0.4), np.exp(0.7j) * np.sin(0.4)
    qc = QuantumCircuit(4)
    qc.initialize([a, b], 0)
    c.encode(qc, 0)
    assert overlap(Statevector(qc), a * z0() + b * z1()) == pytest.approx(1.0)


def test_transversal_cx_is_logical_cnot():
    a, b = np.cos(0.4), np.exp(0.7j) * np.sin(0.4)
    qc = QuantumCircuit(8)
    qc.initialize([a, b], 0)
    c.encode(qc, 0)
    c.encode(qc, 1)
    c.transversal_cx(qc, 0, 1)
    expected = a * Statevector(np.kron(np.asarray(z0()), np.asarray(z0()))) \
        + b * Statevector(np.kron(np.asarray(z1()), np.asarray(z1())))
    assert overlap(Statevector(qc), expected) == pytest.approx(1.0)


@pytest.mark.parametrize("i", range(4))
@pytest.mark.parametrize("p", "XYZ")
def test_every_single_error_is_detected(p, i):
    lab = ["I"] * 4
    lab[3 - i] = p
    e = Pauli("".join(lab))
    assert not (Pauli("XXXX").commutes(e) and Pauli("ZZZZ").commutes(e))


def test_inject_applies_named_pauli():
    qc = QuantumCircuit(4)
    c.inject(qc, 0, "Y", 2)
    assert [ins.operation.name for ins in qc.data] == ["y"]
    assert qc.find_bit(qc.data[0].qubits[0]).index == 2


def test_t_gadget_post_selected_gives_t_state():
    qc = QuantumCircuit(8)
    qc.h(0); c.encode(qc, 0)                 # A = |+_L>
    qc.h(4); qc.t(4); c.encode(qc, 1)        # B = |T_L>
    c.transversal_cx(qc, 1, 0)               # B controls A
    sv = np.asarray(Statevector(qc))
    p0 = np.outer(np.asarray(z0()), np.asarray(z0()).conj())
    proj = np.kron(np.eye(16), p0) @ sv
    assert np.linalg.norm(proj) ** 2 == pytest.approx(0.5)
    proj /= np.linalg.norm(proj)
    t_l = (np.asarray(z0()) + np.exp(1j * np.pi / 4) * np.asarray(z1())) / SQ2
    assert overlap(proj, np.kron(t_l, np.asarray(z0()))) == pytest.approx(1.0)


def test_finish_measures_every_qubit_and_rotates_bases():
    qc = c.new_circuit(2)
    c.encode(qc, 0); c.encode(qc, 1)
    start = len(qc.data)
    c.finish(qc, 2, ["X", "Y"])
    names = [ins.operation.name for ins in qc.data]
    assert names.count("measure") == 12
    # block 0 X readout: four h on data 0..3 after extraction; block 1 Y readout: sdg on 4, h on 4 and 5
    rotated = [(ins.operation.name, qc.find_bit(ins.qubits[0]).index) for ins in qc.data[start:]
               if ins.operation.name in ("h", "sdg")]
    assert ("sdg", 4) in rotated and ("h", 4) in rotated and ("h", 5) in rotated
    assert all(("h", q) in rotated for q in range(4))
    assert ("h", 6) not in rotated and ("h", 7) not in rotated


def test_bare_helpers():
    qc = c.new_bare(2)
    assert qc.num_qubits == 2 and qc.num_clbits == 2
    c.finish_bare(qc, ["Z", "Y"])
    names = [ins.operation.name for ins in qc.data]
    assert names.count("measure") == 2 and "sdg" in names and "h" in names


def test_xxxx_ancilla_fault_mid_extraction_is_flagged():
    # X fault on the XXXX ancilla after its second CX must not leave an undetectable logical error.
    from qiskit.providers.basic_provider import BasicSimulator

    from qec.analyze import analyze_all

    ext = c.new_circuit(1)
    c.extract_syndromes(ext, 1, 0)
    ax = c.anc_x(1, 0)
    qc = c.new_circuit(1)
    c.encode(qc, 0)                          # |0_L>
    n_cx = 0
    for ins in ext.data:
        qc.append(ins.operation, [ext.find_bit(q).index for q in ins.qubits])
        if ins.operation.name == "cx" and ext.find_bit(ins.qubits[0]).index == ax:
            n_cx += 1
            if n_cx == 2:
                qc.x(ax)
    qc.measure(range(qc.num_qubits), range(qc.num_clbits))   # Z readout
    counts = BasicSimulator().run(qc, shots=200, seed_simulator=1234).result().get_counts()
    meta = {"index": 0, "lesson": 1, "name": "hook", "kind": "enc", "label": "Z", "nblocks": 1,
            "bases": ["Z"], "value_blocks": [0], "keep_if_zero": [], "ideal_p1": 0.0}
    assert analyze_all([meta], [counts])[0]["flagged_fraction"] == 1.0


def test_phases_are_recorded_without_changing_gates():
    from qec import code412 as c
    qc = c.new_circuit(2)
    c.mark(qc, "prepare A")
    qc.h(c.data(0, 0))
    c.encode(qc, 0)
    c.encode(qc, 1)
    c.transversal_cx(qc, 0, 1)
    c.finish(qc, 2, ["Z", "X"])
    names = [n for _, n in qc.metadata["phases"]]
    assert names == ["prepare A", "encode A", "encode B", "transversal CNOT A→B",
                     "syndrome extraction A", "syndrome extraction B", "readout rotation B (X)", "measure"]
    starts = [s for s, _ in qc.metadata["phases"]]
    assert starts == sorted(starts) and starts[0] == 0
    # gate count unchanged by marking: 1 + 5 + 5 + 4 + 10 + 10 + 4 + 12
    assert len(qc.data) == 51
