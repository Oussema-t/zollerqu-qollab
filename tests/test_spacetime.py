import json

from qec.lessons import build_all
from qec.spacetime import layout


def enc_layouts(inject=None):
    circuits, meta = build_all({"shots": 100, "inject": inject})
    return [(m, layout(qc, m)) for qc, m in zip(circuits, meta) if m["kind"] == "enc"]


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


def test_qubit_positions_and_blocks():
    m, L = [x for x in enc_layouts() if x[0]["name"] == "enc_Z" and x[0]["lesson"] == 3][0]
    assert len(L["qubits"]) == 18 and len(L["blocks"]) == 3
    b1 = L["blocks"][1]
    assert b1["name"] == "B" and b1["data"] == [4, 5, 6, 7] and b1["ax"] == 14 and b1["az"] == 15
    q5 = L["qubits"][5]
    assert q5["kind"] == "data" and q5["block"] == 1 and (q5["x"], q5["y"]) == (2.2 + 1, 0)
    assert L["qubits"][14]["kind"] == "ax" and (L["qubits"][14]["x"], L["qubits"][14]["y"]) == (2.2 + 0.5, 0.5)
    assert L["qubits"][15]["kind"] == "az" and L["qubits"][15]["y"] == 1.7


def test_transversal_cnot_is_four_parallel_bars():
    m, L = [x for x in enc_layouts() if x[0]["name"] == "enc_Z" and x[0]["lesson"] == 3][0]
    tv = [p for p in L["phases"] if p["name"].startswith("transversal")]
    assert [p["name"] for p in tv] == ["transversal CNOT B→C", "transversal CNOT A→B"]
    for p in tv:
        ops = [o for o in L["ops"] if o["phase"] == p["name"]]
        assert len(ops) == 4 and all(o["gate"] == "cx" and o["role"] == "transversal" for o in ops)
        assert len({o["t"] for o in ops}) == 1
    bc = [o for o in L["ops"] if o["phase"] == "transversal CNOT B→C"]
    assert sorted(o["q"] for o in bc) == [[4, 8], [5, 9], [6, 10], [7, 11]]


def test_injection_marker_sits_between_encode_and_extraction():
    m, L = [x for x in enc_layouts({"pauli": "X", "qubit": 1}) if x[0]["lesson"] == 1 and x[0]["name"] == "enc_Z"][0]
    inj = L["inject"]
    assert inj["q"] == 1 and inj["pauli"] == "X"
    enc = next(p for p in L["phases"] if p["name"] == "encode A")
    ext = next(p for p in L["phases"] if p["name"].startswith("syndrome"))
    assert enc["t1"] < inj["t"] < ext["t0"]
    assert inj["caught"] == ["z_top", "zzzz_anc"]


def test_no_injection_means_null():
    m, L = enc_layouts()[0]
    assert L["inject"] is None


def test_sheets_propagate_through_transversal_cnot():
    m, L = [x for x in enc_layouts() if x[0]["lesson"] == 2 and x[0]["name"] == "enc_ZZ"][0]
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
