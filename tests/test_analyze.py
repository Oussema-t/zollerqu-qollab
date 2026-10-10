import pytest

from qec import code913 as c
from qec.aer_compat import simulate, stand_in_noise
from qec.analyze import analyze_all, analyze_one, bits_of, block_readout
from qec.lessons import build_all, IDEAL_P1_ROTATED

ROW_KEYS = {"index", "lesson", "name", "kind", "label", "bases", "nblocks", "ideal_p1",
            "n_total", "n_clean", "n_corrected", "n_discarded", "n_kept",
            "discard_fraction", "correct_fraction", "kept_fraction",
            "p1_corrected", "p1_clean", "p1_all",
            "deviation_corrected", "deviation_clean", "deviation_all", "blocks"}


def run(settings, shots, lessons=(1, 2, 3, 4), noise=None):
    circuits, meta = build_all(settings)
    pick = [i for i, m in enumerate(meta) if m["lesson"] in lessons]
    counts = simulate([circuits[i] for i in pick], shots, noise_model=noise, seed=1234)
    return analyze_all([meta[i] for i in pick], counts)


def by_name(results):
    return {(r["lesson"], r["name"]): r for r in results}


def key_of(bits):
    """Inverse of bits_of: bits indexed by qubit -> little-endian key."""
    return "".join(str(b) for b in reversed(bits))


def block_bits(local, nblocks=1, b=0):
    bits = [0] * (9 * nblocks)
    for i in local:
        bits[c.data(b, i)] = 1
    return bits


def meta1(basis):
    return {"index": 0, "lesson": 1, "name": f"enc_{basis}", "kind": "enc", "label": basis, "nblocks": 1,
            "bases": [basis], "value_blocks": [0], "keep_if_zero": [], "ideal_p1": 0.0, "nqubits": 9}


def test_bits_of_is_little_endian():
    assert bits_of("0110") == [0, 1, 1, 0]
    assert bits_of("1000") == [0, 0, 0, 1]
    assert bits_of(key_of([1, 0, 0, 1, 1])) == [1, 0, 0, 1, 1]


@pytest.mark.parametrize("i", range(9))
def test_z_readout_corrects_every_single_flip(i):
    r = block_readout(block_bits([i]), 1, 0, "Z")
    assert r["status"] == "corrected"
    assert r["raw_logical"] == (1 if i in c.Z_L else 0)
    assert r["logical"] == 0
    assert r["syndrome"] == tuple(int(i in s) for s in c.Z_STABS)
    assert r["corrected_qubit"] == c.Z_TABLE[r["syndrome"]]


@pytest.mark.parametrize("i", range(9))
def test_x_readout_corrects_every_single_flip(i):
    r = block_readout(block_bits([i]), 1, 0, "X")
    assert r["status"] == "corrected"
    assert r["raw_logical"] == (1 if i in c.X_L else 0)
    assert r["logical"] == 0
    assert r["syndrome"] == tuple(int(i in s) for s in c.X_STABS)
    assert r["corrected_qubit"] == c.X_TABLE[r["syndrome"]]


def test_clean_logical_one_and_discarded_and_y():
    one = block_readout(block_bits([0, 1, 2]), 1, 0, "Z")          # a |1_L> string: checks quiet, Z_L odd
    assert one == {"logical": 1, "raw_logical": 1, "syndrome": (0, 0, 0, 0), "status": "clean", "corrected_qubit": None}
    bad = block_readout(block_bits([0, 8]), 1, 0, "Z")             # syndrome 0011 is not in the table
    assert bad["syndrome"] == (0, 0, 1, 1) and bad["status"] == "discarded" and bad["corrected_qubit"] is None
    assert bad["logical"] == bad["raw_logical"] == 1
    y = block_readout(block_bits([1, 6], nblocks=2, b=1), 2, 1, "Y")
    assert y == {"logical": 0, "raw_logical": 0, "syndrome": (), "status": "clean", "corrected_qubit": None}
    assert block_readout(block_bits([3], nblocks=2, b=1), 2, 1, "Y")["logical"] == 1


def test_row_fields_and_classification_on_synthetic_counts():
    counts = {
        key_of(block_bits([])): 5,           # clean, 0
        key_of(block_bits([0, 1, 2])): 3,    # clean, 1
        key_of(block_bits([4])): 4,          # corrected on q4, 0
        key_of(block_bits([1, 2])): 2,       # q0 missing: corrected on q0, 1 (raw 0)
        key_of(block_bits([0, 8])): 6,       # discarded, raw 1
    }
    r = analyze_one(meta1("Z"), counts)
    assert set(r) == ROW_KEYS
    assert (r["n_total"], r["n_clean"], r["n_corrected"], r["n_discarded"], r["n_kept"]) == (20, 8, 6, 6, 14)
    assert r["discard_fraction"] == pytest.approx(0.3) and r["correct_fraction"] == pytest.approx(0.3)
    assert r["kept_fraction"] == pytest.approx(0.7)
    assert r["p1_corrected"] == pytest.approx(5 / 14)
    assert r["p1_clean"] == pytest.approx(3 / 8)
    assert r["p1_all"] == pytest.approx((3 + 6) / 20)
    assert r["deviation_corrected"] == pytest.approx(5 / 14)
    assert r["deviation_all"] == pytest.approx(9 / 20)
    blk = r["blocks"][0]
    assert blk["basis"] == "Z"
    assert blk["checks"] == pytest.approx([4 / 20, 4 / 20, (2 + 6) / 20, 6 / 20])
    corr = [0.0] * 9
    corr[4], corr[0] = 4 / 20, 2 / 20
    assert blk["corrections"] == pytest.approx(corr)


def test_keep_if_zero_uses_corrected_bits():
    m = {"index": 0, "lesson": 4, "name": "enc_X", "kind": "enc", "label": "X", "nblocks": 2, "bases": ["Z", "X"],
         "value_blocks": [1], "keep_if_zero": [0], "ideal_p1": 0.25, "nqubits": 18}
    a_flip_corrected = block_bits([1, 2], nblocks=2)            # A reads 1 after correcting q0
    a_zero_b_one = block_bits([0, 3, 6], nblocks=2, b=1)        # A = 0, B = 1 (X_L odd, X checks quiet)
    r = analyze_one(m, {key_of(a_flip_corrected): 3, key_of(a_zero_b_one): 1})
    assert r["n_corrected"] == 0 and r["n_clean"] == 4          # block A is not a value block
    assert r["n_kept"] == 1 and r["kept_fraction"] == pytest.approx(0.25)
    assert r["p1_corrected"] == 1.0
    assert r["p1_all"] == pytest.approx(1 / 4)                   # raw A reads 0 on the first string


def test_key_without_leading_zeros_is_padded_to_circuit_width():
    key = key_of(block_bits([4])).lstrip("0")                   # "10000": q5..q8 dropped by the backend
    r = analyze_one(meta1("Z"), {key: 3})
    assert r["n_corrected"] == 3 and r["blocks"][0]["corrections"][4] == pytest.approx(1.0)


def test_keep_if_zero_block_with_uncorrectable_syndrome_discards_the_shot():
    m = {"index": 0, "lesson": 4, "name": "enc_X", "kind": "enc", "label": "X", "nblocks": 2, "bases": ["Z", "X"],
         "value_blocks": [1], "keep_if_zero": [0], "ideal_p1": 0.25, "nqubits": 18}
    a_bad = block_bits([0, 8], nblocks=2)                       # A: Z syndrome 0011, not in the table
    clean = block_bits([], nblocks=2)
    r = analyze_one(m, {key_of(a_bad): 2, key_of(clean): 2})
    assert (r["n_discarded"], r["n_clean"], r["n_kept"]) == (2, 2, 2)
    assert r["discard_fraction"] == pytest.approx(0.5) and r["kept_fraction"] == pytest.approx(0.5)


def test_bare_row_shape():
    m = {"index": 2, "lesson": 1, "name": "bare_Z", "kind": "bare", "label": "Z", "nblocks": 1, "bases": ["Z"],
         "value_blocks": [0], "keep_if_zero": [], "ideal_p1": 0.0, "nqubits": 1}
    r = analyze_one(m, {"0": 7, "1": 3})
    assert set(r) == ROW_KEYS
    assert (r["n_clean"], r["n_corrected"], r["n_discarded"], r["n_kept"]) == (10, 0, 0, 10)
    assert r["p1_corrected"] == r["p1_clean"] == r["p1_all"] == pytest.approx(0.3)
    assert r["blocks"] == []


def test_ideal_simulation_is_clean_with_right_statistics():
    res = run({"shots": 1000, "inject": None}, 1000)
    assert len(res) == 16
    for r in res:
        assert r["n_total"] == 1000
        assert r["n_discarded"] == 0 and r["n_corrected"] == 0, r
        assert r["deviation_corrected"] < 0.06, r
        assert r["p1_clean"] == r["p1_corrected"] == r["p1_all"]
    d = by_name(res)
    assert d[(3, "enc_Z")]["ideal_p1"] == pytest.approx(IDEAL_P1_ROTATED)
    for name in ("enc_X", "enc_Y", "bare_X", "bare_Y"):
        assert d[(4, name)]["p1_corrected"] == pytest.approx(0.25, abs=0.06)
        assert d[(4, name)]["kept_fraction"] == pytest.approx(0.5, abs=0.06)
    assert len(d[(3, "enc_X")]["blocks"]) == 3 and d[(3, "enc_X")]["blocks"][2]["basis"] == "X"
    assert d[(4, "enc_Y")]["blocks"][1]["checks"] == []


def test_lesson_four_without_t_reads_one_half_in_y():
    circuits, meta = build_all({"shots": 400, "inject": None})
    m = next(x for x in meta if (x["lesson"], x["name"]) == (4, "enc_Y"))
    qc = circuits[m["index"]].copy()
    qc.data = [ins for ins in qc.data if ins.operation.name != "t"]
    r = analyze_one(m, simulate([qc], 400, seed=1234)[0])
    assert r["p1_corrected"] == pytest.approx(0.5, abs=0.08)


@pytest.mark.parametrize("pauli", "XYZ")
def test_single_injected_error_on_q4_is_located_and_corrected(pauli):
    d = by_name(run({"shots": 100, "inject": {"pauli": pauli, "qubit": 4}}, 100, lessons=(1,)))
    z, x = d[(1, "enc_Z")], d[(1, "enc_X")]
    # the X part shows in the Z readout, the Z part in the X readout
    for r, visible in ((z, pauli in "XY"), (x, pauli in "ZY")):
        assert r["n_discarded"] == 0 and r["p1_corrected"] == 0
        if visible:
            assert r["n_corrected"] == 100 and r["blocks"][0]["corrections"][4] == 1.0
        else:
            assert r["n_clean"] == 100 and sum(r["blocks"][0]["corrections"]) == 0
    assert z["blocks"][0]["checks"] == ([1.0, 1.0, 0.0, 0.0] if pauli in "XY" else [0.0] * 4)


@pytest.mark.parametrize("qubit", range(9))
def test_any_single_x_error_is_corrected(qubit):
    r = by_name(run({"shots": 50, "inject": {"pauli": "X", "qubit": qubit}}, 50, lessons=(1,)))[(1, "enc_Z")]
    assert r["n_corrected"] == 50 and r["p1_corrected"] == 0


def test_two_errors_are_discarded():
    qc = c.new_circuit(1)
    c.encode(qc, 0)
    qc.x(0)
    qc.x(8)
    c.finish(qc, 1, ["Z"])
    r = analyze_one(meta1("Z"), simulate([qc], 50, seed=1)[0])
    assert r["n_discarded"] == 50 and r["p1_corrected"] is None and r["n_kept"] == 0


def test_noisy_run_shows_corrections_and_discards():
    res = run({"shots": 200, "inject": None}, 200, lessons=(1, 2), noise=stand_in_noise())
    enc = [r for r in res if r["kind"] == "enc"]
    assert sum(r["n_corrected"] for r in enc) > 0 and sum(r["n_discarded"] for r in enc) > 0
    for r in enc:
        assert r["n_clean"] + r["n_corrected"] + r["n_discarded"] == r["n_total"] == 200
