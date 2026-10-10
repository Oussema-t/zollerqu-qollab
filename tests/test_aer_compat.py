from qec.aer_compat import method_for, simulate, stand_in_noise
from qec.lessons import build_all


def test_method_per_circuit():
    circuits, meta = build_all({"shots": 10, "inject": None})
    got = {(m["lesson"], m["name"]): method_for(qc) for qc, m in zip(circuits, meta)}
    assert got[(1, "enc_Z")] == got[(2, "enc_XX")] == got[(1, "bare_X")] == "stabilizer"
    assert got[(4, "enc_X")] == got[(4, "bare_Y")] == got[(3, "bare_Z")] == "statevector"
    assert got[(3, "enc_Z")] == got[(3, "enc_X")] == "matrix_product_state"


def test_simulate_keeps_circuit_order_and_shots():
    circuits, meta = build_all({"shots": 10, "inject": {"pauli": "X", "qubit": 0}})
    picked = [circuits[i] for i in (14, 0, 2, 15)]   # statevector, stabilizer, stabilizer, statevector
    counts = simulate(picked, 30, noise_model=stand_in_noise(), seed=7)
    assert len(counts) == 4 and all(sum(c.values()) == 30 for c in counts)
    assert [len(next(iter(c))) for c in counts] == [qc.num_clbits for qc in picked]
    assert counts == simulate(picked, 30, noise_model=stand_in_noise(), seed=7)
