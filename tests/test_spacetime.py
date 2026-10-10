import json

import pytest

from qec.lessons import build_all
from qec.spacetime import PITCH, layout


def enc_layouts(inject=None):
    circuits, meta = build_all({"shots": 100, "inject": inject})
    return [(m, layout(qc, m)) for qc, m in zip(circuits, meta) if m["kind"] == "enc"]


def pick(layouts, lesson, name):
    return next(L for m, L in layouts if m["lesson"] == lesson and m["name"] == name)


def test_layers_are_disjoint_per_qubit_and_phases_tile_time():
    for m, L in enc_layouts({"pauli": "X", "qubit": 1}):
        T = L["T"]
        seen = set()
        for op in L["ops"]:
            for q in op["q"]:
                assert (op["t"], q) not in seen, (m["name"], op)
                seen.add((op["t"], q))
            assert 0 <= op["t"] <= T
        spans = L["phases"]
        assert spans[0]["t0"] == 0 and spans[-1]["t1"] == T and spans[-1]["name"] == "measure"
        for a, b in zip(spans, spans[1:]):
            assert b["t0"] == a["t1"] + 1, (m["name"], a, b)
        assert {op["role"] for op in L["ops"]} <= {"prepare", "encode", "inject", "transversal", "readout", "measure"}


def test_qubit_positions_and_blocks():
    assert PITCH == 4.4
    L = pick(enc_layouts(), 3, "enc_Z")
    assert L["pitch"] == 4.4
    assert len(L["qubits"]) == 27 and len(L["blocks"]) == 3
    assert {q["kind"] for q in L["qubits"]} == {"data"}
    b1 = L["blocks"][1]
    assert b1 == {"name": "B", "data": list(range(9, 18)), "x0": 4.4, "y0": 0.0}
    q14 = L["qubits"][14]                       # block B, local 5: column 2, row 1
    assert q14["id"] == 14 and q14["block"] == 1 and q14["local"] == 5
    assert (q14["x"], q14["y"]) == pytest.approx((4.4 + 2, 1))
    assert (L["qubits"][24]["x"], L["qubits"][24]["y"]) == pytest.approx((8.8, 2))


def test_phase_names_for_lesson_three():
    L = pick(enc_layouts(), 3, "enc_X")
    assert [p["name"] for p in L["phases"]] == [
        "prepare A", "encode A", "prepare B", "encode B", "encode C",
        "transversal CNOT B→C", "transversal CNOT A→B",
        "readout rotation A (X)", "readout rotation C (X)", "measure"]


def test_transversal_cnot_is_nine_parallel_bars():
    L = pick(enc_layouts(), 3, "enc_Z")
    tv = [p for p in L["phases"] if p["name"].startswith("transversal")]
    assert [p["name"] for p in tv] == ["transversal CNOT B→C", "transversal CNOT A→B"]
    for p in tv:
        ops = [o for o in L["ops"] if o["phase"] == p["name"]]
        assert len(ops) == 9 and all(o["gate"] == "cx" and o["role"] == "transversal" for o in ops)
        assert len({o["t"] for o in ops}) == 1
    bc = [o for o in L["ops"] if o["phase"] == "transversal CNOT B→C"]
    assert sorted(o["q"] for o in bc) == [[9 + i, 18 + i] for i in range(9)]


def test_injection_marker_with_caught_checks_and_correction():
    layouts = enc_layouts({"pauli": "X", "qubit": 1})
    L = pick(layouts, 1, "enc_Z")
    inj = L["inject"]
    assert {k: inj[k] for k in ("q", "pauli", "block", "local")} == {"q": 1, "pauli": "X", "block": 0, "local": 1}
    enc = next(p for p in L["phases"] if p["name"] == "encode A")
    meas = next(p for p in L["phases"] if p["name"] == "measure")
    assert enc["t1"] < inj["t"] < meas["t0"]
    assert inj["caught"] == [0, 2]               # Z checks (1,2,4,5) and (0,1)
    assert inj["corrected_qubit"] == 1 and inj["corrected"] is True
    x = pick(layouts, 1, "enc_X")["inject"]      # an X error is invisible to the X checks
    assert x["caught"] == [] and x["corrected_qubit"] is None and x["corrected"] is False


@pytest.mark.parametrize("pauli,qubit,zc,xc", [
    ("Z", 5, [], [1, 2]),          # Z on q5: X checks (4,5,7,8) and (2,5); corrected as q5
    ("Y", 4, [0, 1], [0, 1]),
    ("X", 6, [1], []),             # X on q6: Z check (3,4,6,7) only
])
def test_caught_sets_follow_the_readout_basis(pauli, qubit, zc, xc):
    layouts = enc_layouts({"pauli": pauli, "qubit": qubit})
    assert pick(layouts, 1, "enc_Z")["inject"]["caught"] == zc
    assert pick(layouts, 1, "enc_X")["inject"]["caught"] == xc


def test_no_injection_means_null():
    m, L = enc_layouts()[0]
    assert L["inject"] is None


def test_sheets_propagate_through_transversal_cnot():
    L = pick(enc_layouts(), 2, "enc_ZZ")
    t = next(o["t"] for o in L["ops"] if o["role"] == "transversal")
    base = [s for s in L["sheets"] if s["from"] is None]
    assert sorted((s["type"], s["block"]) for s in base) == [("X", 0), ("X", 1), ("Z", 0), ("Z", 1)]
    assert all(s["t0"] == 0 and s["t1"] == L["T"] for s in base)
    prop = [s for s in L["sheets"] if s["from"] is not None]
    assert sorted((s["type"], s["block"], s["from"], s["t0"]) for s in prop) == [("X", 1, 0, t), ("Z", 0, 1, t)]


def test_bridge_spacetime_round_trip():
    from qec import bridge
    bridge.build(json.dumps({"shots": 100, "inject": None}))
    L = json.loads(bridge.spacetime(4))
    assert L["T"] > 0 and L["ops"] and L["phases"] and L["blocks"]
    assert bridge.spacetime(4) == bridge.spacetime(4)   # cached
