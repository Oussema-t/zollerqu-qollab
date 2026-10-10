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
        (3, "enc_Z"), (3, "enc_X"), (3, "bare_Z"), (3, "bare_X"),
        (4, "enc_X"), (4, "enc_Y"), (4, "bare_X"), (4, "bare_Y"),
    ]


def test_qubit_counts():
    circuits, meta = build_all(NO_INJECT)
    enc = {m["lesson"]: qc.num_qubits for qc, m in zip(circuits, meta) if m["kind"] == "enc"}
    bare = {m["lesson"]: qc.num_qubits for qc, m in zip(circuits, meta) if m["kind"] == "bare"}
    assert enc == {1: 6, 2: 12, 3: 18, 4: 12}
    assert bare == {1: 1, 2: 2, 3: 3, 4: 2}


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
                          "value_blocks", "keep_if_zero", "ideal_p1"}
        assert len(m["bases"]) == m["nblocks"]
    by = {(m["lesson"], m["name"]): m for m in meta}
    assert by[(1, "enc_Z")]["value_blocks"] == [0] and by[(1, "enc_Z")]["ideal_p1"] == 0
    assert by[(2, "enc_XX")]["value_blocks"] == [0, 1] and by[(2, "enc_XX")]["bases"] == ["X", "X"]
    assert by[(3, "enc_Z")]["bases"] == ["X", "Z", "Z"] and by[(3, "enc_Z")]["value_blocks"] == [2, 1]
    assert by[(3, "enc_X")]["bases"] == ["X", "Z", "X"] and by[(3, "enc_X")]["value_blocks"] == [2, 0]
    assert by[(3, "enc_X")]["ideal_p1"] == pytest.approx(IDEAL_P1_ROTATED)
    assert by[(4, "enc_Y")]["bases"] == ["Z", "Y"] and by[(4, "enc_Y")]["keep_if_zero"] == [0]
    assert by[(4, "enc_Y")]["value_blocks"] == [1]
    assert by[(4, "bare_X")]["bases"] == ["Z", "X"] and by[(4, "bare_X")]["keep_if_zero"] == [0]


def test_injection_only_touches_lesson_one():
    circuits, meta = build_all({"shots": 100, "inject": {"pauli": "Y", "qubit": 3}})
    for qc, m in zip(circuits, meta):
        ys = [qc.find_bit(ins.qubits[0]).index for ins in qc.data if ins.operation.name == "y"]
        if m["lesson"] == 1 and m["kind"] == "enc":
            assert ys == [3]
        elif m["lesson"] == 1:
            assert ys == [0]
        else:
            assert ys == []


def test_bad_injection_rejected():
    with pytest.raises(ValueError):
        build_all({"shots": 100, "inject": {"pauli": "H", "qubit": 0}})
    with pytest.raises(ValueError):
        build_all({"shots": 100, "inject": {"pauli": "X", "qubit": 4}})
