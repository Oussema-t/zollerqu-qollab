import math

import pytest

from qec.lessons import build_all, IDEAL_P1_ROTATED

NO_INJECT = {"shots": 100, "inject": None}


def names(meta):
    return [(m["lesson"], m["name"]) for m in meta]


def test_sixteen_circuits_in_fixed_order():
    circuits, meta = build_all(NO_INJECT)
    assert len(circuits) == len(meta) == 16
    assert names(meta) == [
        (1, "enc_Z"), (1, "enc_X"), (1, "bare_Z"), (1, "bare_X"),
        (2, "enc_ZZ"), (2, "enc_XX"), (2, "bare_ZZ"), (2, "bare_XX"),
        (3, "enc_X"), (3, "enc_Y"), (3, "bare_X"), (3, "bare_Y"),
        (4, "enc_X"), (4, "enc_Y"), (4, "bare_X"), (4, "bare_Y"),
    ]


def test_qubit_counts():
    circuits, meta = build_all(NO_INJECT)
    enc = {m["lesson"]: qc.num_qubits for qc, m in zip(circuits, meta) if m["kind"] == "enc"}
    bare = {m["lesson"]: qc.num_qubits for qc, m in zip(circuits, meta) if m["kind"] == "bare"}
    assert enc == {1: 9, 2: 18, 3: 9, 4: 27}
    assert bare == {1: 1, 2: 2, 3: 1, 4: 3}
    for qc, m in zip(circuits, meta):
        assert m["nqubits"] == qc.num_qubits


def test_every_circuit_measures_every_qubit_last():
    circuits, _ = build_all(NO_INJECT)
    for qc in circuits:
        measured = {qc.find_bit(ins.qubits[0]).index for ins in qc.data if ins.operation.name == "measure"}
        assert measured == set(range(qc.num_qubits))
        tail = [ins.operation.name for ins in qc.data[-qc.num_qubits:]]
        assert set(tail) == {"measure"}


def test_metadata_fields_and_recipes():
    _, meta = build_all(NO_INJECT)
    for m in meta:
        assert set(m) >= {"index", "lesson", "name", "kind", "label", "nblocks", "bases",
                          "value_blocks", "keep_if_zero", "ideal_p1", "nqubits"}
        assert len(m["bases"]) == m["nblocks"]
    by = {(m["lesson"], m["name"]): m for m in meta}
    assert by[(1, "enc_Z")]["value_blocks"] == [0] and by[(1, "enc_Z")]["ideal_p1"] == 0
    assert by[(2, "enc_XX")]["value_blocks"] == [0, 1] and by[(2, "enc_XX")]["bases"] == ["X", "X"]
    for kind in ("enc", "bare"):
        for basis in ("X", "Y"):
            m3, m4 = by[(3, f"{kind}_{basis}")], by[(4, f"{kind}_{basis}")]
            assert m3["bases"] == [basis] and m3["value_blocks"] == [0]
            assert m4["bases"] == ["X", "Z", basis]
            assert m4["value_blocks"] == ([2, 0] if basis == "X" else [2, 0, 1])
            assert m3["ideal_p1"] == m4["ideal_p1"] == pytest.approx(math.sin(math.pi / 8) ** 2)
    assert IDEAL_P1_ROTATED == pytest.approx(math.sin(math.pi / 8) ** 2)
    assert all(m["keep_if_zero"] == [] for m in meta)


def test_injection_only_touches_lesson_one():
    circuits, meta = build_all({"shots": 100, "inject": {"pauli": "Y", "qubit": 8}})
    for qc, m in zip(circuits, meta):
        ys = [qc.find_bit(ins.qubits[0]).index for ins in qc.data if ins.operation.name == "y"]
        if m["lesson"] == 1 and m["kind"] == "enc":
            assert ys == [8]
        elif m["lesson"] == 1:
            assert ys == [0]
        else:
            assert ys == []


def test_bad_injection_rejected():
    with pytest.raises(ValueError):
        build_all({"shots": 100, "inject": {"pauli": "H", "qubit": 0}})
    with pytest.raises(ValueError):
        build_all({"shots": 100, "inject": {"pauli": "X", "qubit": 9}})
    for q in range(9):
        build_all({"shots": 100, "inject": {"pauli": "X", "qubit": q}})
