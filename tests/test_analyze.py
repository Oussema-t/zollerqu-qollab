import math

import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qec.analyze import analyze_all, bits_of, block_readout
from qec.lessons import build_all, IDEAL_P1_ROTATED


def run(settings, shots, noise=None):
    circuits, meta = build_all(settings)
    sim = AerSimulator(noise_model=noise) if noise else AerSimulator()
    tcs = transpile(circuits, sim, optimization_level=0)
    result = sim.run(tcs, shots=shots, seed_simulator=1234).result()
    counts = [result.get_counts(i) for i in range(len(circuits))]
    return analyze_all(meta, counts)


def by_name(results):
    return {(r["lesson"], r["name"]): r for r in results}


def test_bits_of_is_little_endian():
    assert bits_of("0110") == [0, 1, 1, 0]
    assert bits_of("1000") == [0, 0, 0, 1]


def test_block_readout_recipes():
    # block 0 of a 1-block circuit: data bits 0..3, ax=4, az=5
    bits = bits_of("00" + "0101")      # q0=1,q1=0,q2=1,q3=0 ; ancillas 0
    r = block_readout(bits, nblocks=1, b=0, basis="Z")
    assert r == {"logical": 0, "checks": [1, 1], "ax": 0, "az": 0}
    r = block_readout(bits, nblocks=1, b=0, basis="X")
    assert r == {"logical": 1, "checks": [0], "ax": 0, "az": 0}
    r = block_readout(bits, nblocks=1, b=0, basis="Y")
    assert r == {"logical": 0, "checks": [], "ax": 0, "az": 0}
    bits = bits_of("10" + "0000")      # az=1
    assert block_readout(bits, 1, 0, "Z")["az"] == 1


def test_ideal_simulation_has_no_flags_and_right_statistics():
    results = run({"shots": 4000, "inject": None}, 4000)
    res = by_name(results)
    for r in results:
        assert r["n_total"] == 4000
        assert r["flagged_fraction"] == 0
        assert abs(r["p1_accepted"] - r["ideal_p1"]) < 0.03, r
        assert r["deviation_accepted"] < 0.03
    assert res[(4, "enc_X")]["kept_fraction"] == pytest.approx(0.5, abs=0.05)
    assert res[(4, "bare_Y")]["kept_fraction"] == pytest.approx(0.5, abs=0.05)
    assert res[(3, "enc_Z")]["ideal_p1"] == pytest.approx(IDEAL_P1_ROTATED)


@pytest.mark.parametrize("qubit", range(4))
@pytest.mark.parametrize("pauli", "XYZ")
def test_single_injected_error_is_always_flagged(pauli, qubit):
    res = by_name(run({"shots": 200, "inject": {"pauli": pauli, "qubit": qubit}}, 200))
    for name in ("enc_Z", "enc_X"):
        r = res[(1, name)]
        assert r["flagged_fraction"] == 1.0, (name, r)
        assert r["n_accepted"] == 0
        assert r["p1_accepted"] is None
    for name in ("bare_Z", "bare_X"):
        assert res[(1, name)]["flagged_fraction"] == 0


def test_face_fire_rates_reported_per_block():
    res = by_name(run({"shots": 200, "inject": {"pauli": "X", "qubit": 0}}, 200))
    r = res[(1, "enc_Z")]
    assert len(r["blocks"]) == 1
    assert r["bases"] == ["Z"]
    faces = r["blocks"][0]
    assert set(faces) == {"x_face", "z_top", "z_bottom", "zzzz_anc"}
    assert faces["z_top"] == 1.0 and faces["zzzz_anc"] == 1.0 and faces["x_face"] == 0.0 and faces["z_bottom"] == 0.0
    r = res[(2, "enc_XX")]
    assert len(r["blocks"]) == 2
    assert res[(2, "enc_XX")]["bases"] == ["X", "X"]


def test_lesson_four_shows_the_t_gate():
    res = by_name(run({"shots": 4000, "inject": None}, 4000))
    for name in ("enc_X", "enc_Y", "bare_X", "bare_Y"):
        assert res[(4, name)]["p1_accepted"] == pytest.approx(0.25, abs=0.03)
        assert res[(4, name)]["kept_fraction"] == pytest.approx(0.5, abs=0.03)


def test_noisy_table_is_informational():
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["h", "x", "y", "z", "t", "sdg", "ry"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
    res = run({"shots": 200, "inject": None}, 200, noise)
    print("\nlesson name       flagged  dev_acc  dev_all")
    for r in res:
        if r["kind"] == "enc":
            print(f"{r['lesson']}      {r['name']:<10} {r['flagged_fraction']:.3f}    "
                  f"{(r['deviation_accepted'] if r['deviation_accepted'] is not None else math.nan):.3f}    {r['deviation_all']:.3f}")
    for r in res:
        if r["kind"] == "enc":
            assert 0 < r["flagged_fraction"] < 1
