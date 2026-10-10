# Logical Teleportation Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Qollab JS/Qiskit project that teaches error detection, logical CNOT, logical teleportation and the teleported T gate on the [[4,1,2]] surface code, in four interactive lessons, from one batched job.

**Architecture:** Python (`qec/`) builds every circuit and analyses counts; JavaScript (`main.js`, `ui/`) renders the page, makes the single `backend.run` call, and draws SVG patches and charts. The two sides exchange JSON strings only, plus the circuit list and Matplotlib figures as Pyodide proxies. Spec: `docs/superpowers/specs/2026-10-09-logical-teleportation-tutorial-design.md`.

**Tech Stack:** Python 3.12+, Qiskit 2.x, qiskit-aer (local tests only), pytest, plain ES-module JavaScript, SVG, Qollab runtime (Pyodide). No JS libraries.

**Working directory for every command:** `D:\Projects\zollerqu-unlocked`. Run Python as `python -m pytest`.

**Conventions that every task relies on**

- Block `b` uses data qubits `4b..4b+3`; local index `i` is at `4b+i`. With `n` blocks, ancilla `ax(b) = 4n + 2b` (measures XXXX) and `az(b) = 4n + 2b + 1` (measures ZZZZ). One classical register of size `6n`, bit `k` records qubit `k`.
- Qiskit bitstrings are little-endian. `bits = [int(c) for c in reversed(key)]` gives `bits[k]` for qubit `k`.
- Readout recipes per block (local bit values `v0..v3`): Z basis: logical `v0^v2`, checks `[v0^v1, v2^v3]`. X basis: logical `v0^v1`, checks `[v0^v1^v2^v3]`. Y basis: logical `v0^v1^v2`, checks `[]`.
- A circuit's reported value is the XOR of the logical bits of `value_blocks`; shots are kept only if every block in `keep_if_zero` reads logical 0.
- Settings JSON: `{"shots": 500, "inject": null}` or `{"shots": 500, "inject": {"pauli": "X", "qubit": 2}}`.

---

## File structure

| Path | Responsibility |
|---|---|
| `requirements.txt` | `qiskit[visualization]>=2.5.2,<3.0.0` for Qollab |
| `requirements-dev.txt` | local extras: `qiskit-aer`, `pytest` |
| `pytest.ini` | `pythonpath = .` so `qec` imports in tests |
| `qec/code412.py` | encoder, transversal CNOT, injection, syndrome extraction, readout rotation, bare helpers |
| `qec/lessons.py` | `build_all(settings)` producing 16 circuits and metadata |
| `qec/analyze.py` | counts to metrics and face fire rates |
| `qec/bridge.py` | JSON-in/JSON-out entry points for JavaScript, recorded take, export |
| `tools/record_take.py` | local Aer stand-in for `data/recorded_aria1.json` |
| `data/recorded_aria1.json` | recorded take (local stand-in until the Aria run on Saturday) |
| `index.html`, `main.css`, `main.js` | page, styles, state machine and the one job |
| `ui/patches.js` | SVG patches with syndrome highlighting |
| `ui/charts.js` | summary cards and SVG bar charts |
| `ui/lessons.js` | lesson copy |
| `README.md` | Qollab project page text |
| `tests/test_code412.py`, `tests/test_lessons.py`, `tests/test_analyze.py`, `tests/test_bridge.py` | pytest |

---

### Task 1: Scaffold

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `qec/__init__.py`, `tests/__init__.py`, `ui/.gitkeep`, `data/.gitkeep`, `tools/.gitkeep`

- [ ] **Step 1: Create the files**

`requirements.txt`:
```
qiskit[visualization]>=2.5.2,<3.0.0
```

`requirements-dev.txt`:
```
qiskit>=2.4
qiskit-aer>=0.17
pytest>=8
```

`pytest.ini`:
```
[pytest]
pythonpath = .
testpaths = tests
```

`qec/__init__.py` and `tests/__init__.py`: empty files. `ui/.gitkeep`, `data/.gitkeep`, `tools/.gitkeep`: empty files.

- [ ] **Step 2: Verify pytest collects nothing and exits cleanly**

Run: `python -m pytest -q`
Expected: `no tests ran` with exit code 5.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "Scaffold project layout and test config"
```

---

### Task 2: code412 building blocks

**Files:**
- Create: `qec/code412.py`
- Test: `tests/test_code412.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_code412.py`:
```python
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
    c.finish(qc, 2, ["X", "Y"])
    names = [ins.operation.name for ins in qc.data]
    assert names.count("measure") == 12
    # block 0 X readout: four h on data 0..3 after extraction; block 1 Y readout: sdg on 4, h on 4 and 5
    rotated = [(ins.operation.name, qc.find_bit(ins.qubits[0]).index) for ins in qc.data
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_code412.py -q`
Expected: ImportError / AttributeError, all fail.

- [ ] **Step 3: Implement `qec/code412.py`**

```python
"""Building blocks for the [[4,1,2]] rotated surface code.

Block b occupies data qubits 4b..4b+3 (local index i at 4b+i). With n blocks,
ancilla ax(b) = 4n + 2b measures XXXX and az(b) = 4n + 2b + 1 measures ZZZZ.
Layout of one patch (local indices):

    q0 --- q1     top edge:    Z0Z1 stabilizer, X_L = X0X1
    |  X   |      face:        XXXX stabilizer
    q2 --- q3     bottom edge: Z2Z3 stabilizer, Z_L = Z0Z2 (left edge)
"""
from qiskit import QuantumCircuit

BASES = ("Z", "X", "Y")


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
    qc.h(ax)
    for qi in q:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_code412.py -q`
Expected: all pass (19 tests).

- [ ] **Step 5: Commit**

```bash
git add qec/code412.py tests/test_code412.py
git commit -m "Add [[4,1,2]] code building blocks with statevector tests"
```

---

### Task 3: Lessons builder

**Files:**
- Create: `qec/lessons.py`
- Test: `tests/test_lessons.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_lessons.py`:
```python
import pytest

from qec.lessons import build_all, IDEAL_P1_ROTATED, IDEAL_P1_T

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
    assert by[(4, "enc_X")]["ideal_p1"] == by[(4, "bare_Y")]["ideal_p1"] == IDEAL_P1_T


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lessons.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement `qec/lessons.py`**

```python
"""Build the 16 circuits of the four lessons plus their metadata."""
import math

from qec import code412 as c

THETA = math.pi / 4
IDEAL_P1_ROTATED = math.sin(math.pi / 8) ** 2   # 0.146..., same for every rotated readout we use
IDEAL_P1_T = 0.25                               # T applied to ry(pi/4)|0>: <X> = <Y> = 1/2


def _meta(lesson, name, kind, label, nblocks, bases, value_blocks, ideal_p1, keep_if_zero=()):
    return {
        "lesson": lesson, "name": name, "kind": kind, "label": label,
        "nblocks": nblocks, "bases": list(bases),
        "value_blocks": list(value_blocks), "keep_if_zero": list(keep_if_zero),
        "ideal_p1": ideal_p1,
    }


def _check_injection(inject):
    if inject is None:
        return None
    pauli, qubit = inject.get("pauli"), inject.get("qubit")
    if pauli not in ("X", "Y", "Z") or qubit not in (0, 1, 2, 3):
        raise ValueError(f"bad injection {inject!r}")
    return pauli, int(qubit)


# ---------- lesson 1: one patch, one error ----------

def _lesson1(inject):
    out = []
    for basis, prep in (("Z", False), ("X", True)):
        qc = c.new_circuit(1)
        if prep:
            qc.h(c.data(0, 0))
        c.encode(qc, 0)
        if inject:
            c.inject(qc, 0, inject[0], inject[1])
        c.finish(qc, 1, [basis])
        out.append((qc, _meta(1, f"enc_{basis}", "enc", basis, 1, [basis], [0], 0.0)))
    for basis, prep in (("Z", False), ("X", True)):
        qc = c.new_bare(1)
        if prep:
            qc.h(0)
        if inject:
            getattr(qc, inject[0].lower())(0)
        c.finish_bare(qc, [basis])
        out.append((qc, _meta(1, f"bare_{basis}", "bare", basis, 1, [basis], [0], 0.0)))
    return out


# ---------- lesson 2: logical CNOT, logical Bell pair ----------

def _lesson2():
    out = []
    for basis in ("Z", "X"):
        qc = c.new_circuit(2)
        qc.h(c.data(0, 0))
        c.encode(qc, 0)
        c.encode(qc, 1)
        c.transversal_cx(qc, 0, 1)
        c.finish(qc, 2, [basis, basis])
        out.append((qc, _meta(2, f"enc_{basis}{basis}", "enc", basis + basis, 2, [basis, basis], [0, 1], 0.0)))
    for basis in ("Z", "X"):
        qc = c.new_bare(2)
        qc.h(0)
        qc.cx(0, 1)
        c.finish_bare(qc, [basis, basis])
        out.append((qc, _meta(2, f"bare_{basis}{basis}", "bare", basis + basis, 2, [basis, basis], [0, 1], 0.0)))
    return out


# ---------- lesson 3: logical teleportation ----------

def _lesson3():
    out = []
    for basis in ("Z", "X"):
        qc = c.new_circuit(3)
        qc.ry(THETA, c.data(0, 0))
        c.encode(qc, 0)                 # A = psi
        qc.h(c.data(1, 0))
        c.encode(qc, 1)                 # B = |+_L>
        c.encode(qc, 2)                 # C = |0_L>
        c.transversal_cx(qc, 1, 2)      # logical Bell pair B,C
        c.transversal_cx(qc, 0, 1)      # Bell measurement part 1
        c.finish(qc, 3, ["X", "Z", basis])
        frame = 1 if basis == "Z" else 0   # X correction from Z_B flips Z readout; Z correction from X_A flips X readout
        out.append((qc, _meta(3, f"enc_{basis}", "enc", basis, 3, ["X", "Z", basis], [2, frame], IDEAL_P1_ROTATED)))
    for basis in ("Z", "X"):
        qc = c.new_bare(3)
        qc.ry(THETA, 0)
        qc.h(1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        c.finish_bare(qc, ["X", "Z", basis])
        frame = 1 if basis == "Z" else 0
        out.append((qc, _meta(3, f"bare_{basis}", "bare", basis, 3, ["X", "Z", basis], [2, frame], IDEAL_P1_ROTATED)))
    return out


# ---------- lesson 4: teleported T gate ----------

def _lesson4():
    out = []
    for basis in ("X", "Y"):
        qc = c.new_circuit(2)
        qc.ry(THETA, c.data(0, 0))
        c.encode(qc, 0)                 # A = psi = ry(pi/4)|0>
        qc.h(c.data(1, 0))
        qc.t(c.data(1, 0))
        c.encode(qc, 1)                 # B = |T_L>
        c.transversal_cx(qc, 1, 0)      # B controls A
        c.finish(qc, 2, ["Z", basis])
        out.append((qc, _meta(4, f"enc_{basis}", "enc", basis, 2, ["Z", basis], [1], IDEAL_P1_T, keep_if_zero=[0])))
    for basis in ("X", "Y"):
        qc = c.new_bare(2)
        qc.ry(THETA, 0)
        qc.h(1)
        qc.t(1)
        qc.cx(1, 0)
        c.finish_bare(qc, ["Z", basis])
        out.append((qc, _meta(4, f"bare_{basis}", "bare", basis, 2, ["Z", basis], [1], IDEAL_P1_T, keep_if_zero=[0])))
    return out


def build_all(settings):
    """Return (circuits, metadata) for the given settings dict."""
    inject = _check_injection(settings.get("inject"))
    pairs = _lesson1(inject) + _lesson2() + _lesson3() + _lesson4()
    circuits, meta = [], []
    for index, (qc, m) in enumerate(pairs):
        m["index"] = index
        qc.name = f"L{m['lesson']}_{m['name']}"
        circuits.append(qc)
        meta.append(m)
    return circuits, meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lessons.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add qec/lessons.py tests/test_lessons.py
git commit -m "Build the sixteen lesson circuits with metadata"
```

---

### Task 4: Analysis of counts

**Files:**
- Create: `qec/analyze.py`
- Test: `tests/test_analyze.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_analyze.py`:
```python
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
    result = sim.run(tcs, shots=shots).result()
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
    res = run({"shots": 2000, "inject": None}, 2000, noise)
    print("\nlesson name       flagged  dev_acc  dev_all")
    for r in res:
        if r["kind"] == "enc":
            print(f"{r['lesson']}      {r['name']:<10} {r['flagged_fraction']:.3f}    "
                  f"{(r['deviation_accepted'] if r['deviation_accepted'] is not None else math.nan):.3f}    {r['deviation_all']:.3f}")
    for r in res:
        if r["kind"] == "enc":
            assert 0 < r["flagged_fraction"] < 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analyze.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement `qec/analyze.py`**

```python
"""Turn measurement counts into the numbers the page shows."""
from qec import code412 as c

RECIPES = {
    # basis: (logical bit from local values, list of check parities)
    "Z": (lambda v: v[0] ^ v[2], lambda v: [v[0] ^ v[1], v[2] ^ v[3]]),
    "X": (lambda v: v[0] ^ v[1], lambda v: [v[0] ^ v[1] ^ v[2] ^ v[3]]),
    "Y": (lambda v: v[0] ^ v[1] ^ v[2], lambda v: []),
}


def bits_of(key):
    """Qiskit keys are little-endian; return bits indexed by qubit number."""
    return [int(ch) for ch in reversed(key.replace(" ", ""))]


def block_readout(bits, nblocks, b, basis):
    v = [bits[c.data(b, i)] for i in range(4)]
    logical, checks = RECIPES[basis]
    return {
        "logical": logical(v),
        "checks": checks(v),
        "ax": bits[c.anc_x(nblocks, b)],
        "az": bits[c.anc_z(nblocks, b)],
    }


def _faces_update(acc, basis, r, count):
    if basis == "Z":
        acc["z_top"] += count * r["checks"][0]
        acc["z_bottom"] += count * r["checks"][1]
        acc["x_face"] += count * r["ax"]
    elif basis == "X":
        acc["x_face"] += count if (r["ax"] or r["checks"][0]) else 0
    else:
        acc["x_face"] += count * r["ax"]
    acc["zzzz_anc"] += count * r["az"]


def analyze_one(m, counts):
    nblocks, bases = m["nblocks"], m["bases"]
    enc = m["kind"] == "enc"
    n_total = n_flagged = n_kept = n_one = n_kept_all = n_one_all = 0
    faces = [{"x_face": 0, "z_top": 0, "z_bottom": 0, "zzzz_anc": 0} for _ in range(nblocks)]
    for key, count in counts.items():
        bits = bits_of(key)
        if enc:
            reads = [block_readout(bits, nblocks, b, bases[b]) for b in range(nblocks)]
            flagged = any(r["ax"] or r["az"] or any(r["checks"]) for r in reads)
            for b, r in enumerate(reads):
                _faces_update(faces[b], bases[b], r, count)
            logical = [r["logical"] for r in reads]
        else:
            flagged = False
            logical = bits[:nblocks]
        value = 0
        for b in m["value_blocks"]:
            value ^= logical[b]
        keep = all(logical[b] == 0 for b in m["keep_if_zero"])
        n_total += count
        if keep:
            n_kept_all += count
            n_one_all += count * value
        if flagged:
            n_flagged += count
            continue
        if keep:
            n_kept += count
            n_one += count * value

    def frac(a, b):
        return (a / b) if b else None

    p1_acc = frac(n_one, n_kept)
    p1_all = frac(n_one_all, n_kept_all)
    ideal = m["ideal_p1"]
    out = {
        "index": m["index"], "lesson": m["lesson"], "name": m["name"], "kind": m["kind"], "label": m["label"],
        "bases": list(bases),
        "ideal_p1": ideal,
        "n_total": n_total,
        "n_flagged": n_flagged,
        "n_accepted": n_total - n_flagged,
        "flagged_fraction": frac(n_flagged, n_total) if n_total else 0,
        "kept_fraction": frac(n_kept_all, n_total),
        "p1_accepted": p1_acc,
        "p1_all": p1_all,
        "deviation_accepted": None if p1_acc is None else abs(p1_acc - ideal),
        "deviation_all": None if p1_all is None else abs(p1_all - ideal),
        "blocks": [{k: (v / n_total if n_total else 0) for k, v in f.items()} for f in faces] if enc else [],
    }
    return out


def analyze_all(meta, counts_list):
    """meta and counts_list are parallel lists, one entry per circuit."""
    return [analyze_one(m, dict(counts)) for m, counts in zip(meta, counts_list)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analyze.py -q -s`
Expected: all pass; the informational table prints flagged fractions between 0 and 1 and, for most lessons, `dev_acc` below `dev_all`.

- [ ] **Step 5: Commit**

```bash
git add qec/analyze.py tests/test_analyze.py
git commit -m "Analyse counts into flags, post-selection and face fire rates"
```

---

### Task 5: Bridge, recorded take, export

**Files:**
- Create: `qec/bridge.py`, `tools/record_take.py`, `data/recorded_aria1.json` (generated)
- Test: `tests/test_bridge.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_bridge.py`:
```python
import json

from qec import bridge


def test_build_then_circuits_and_analyze_round_trip():
    meta_json = bridge.build(json.dumps({"shots": 50, "inject": None}))
    meta = json.loads(meta_json)
    circuits = bridge.circuits()
    assert len(meta) == len(circuits) == 16
    # fake counts: every shot reads all zeros
    counts = [{"0" * qc.num_clbits: 50} for qc in circuits]
    res = json.loads(bridge.analyze(meta_json, json.dumps(counts)))
    assert len(res) == 16
    assert res[0]["flagged_fraction"] == 0 and res[0]["p1_accepted"] == 0


def test_recorded_take_is_loadable_and_consistent():
    take = json.loads(bridge.load_recorded())
    assert set(take) >= {"source", "backend", "date", "settings", "meta", "counts"}
    assert len(take["meta"]) == len(take["counts"]) == 16
    res = json.loads(bridge.analyze(json.dumps(take["meta"]), json.dumps(take["counts"])))
    assert len(res) == 16


def test_draw_returns_matplotlib_figure():
    bridge.build(json.dumps({"shots": 50, "inject": None}))
    fig = bridge.draw(0)
    assert fig.__class__.__name__ == "Figure"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bridge.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement `qec/bridge.py`**

```python
"""Entry points called from main.js. JSON strings in and out; circuits and figures stay proxies."""
import json

from qec.analyze import analyze_all
from qec.lessons import build_all

RECORDED = "data/recorded_aria1.json"   # literal path so the Qollab runtime loads the file

_LAST = {"circuits": [], "meta": []}


def build(settings_json):
    circuits, meta = build_all(json.loads(settings_json))
    _LAST["circuits"], _LAST["meta"] = circuits, meta
    return json.dumps(meta)


def circuits():
    return _LAST["circuits"]


def analyze(meta_json, counts_json):
    return json.dumps(analyze_all(json.loads(meta_json), json.loads(counts_json)))


def draw(index):
    qc = _LAST["circuits"][int(index)]
    return qc.draw(output="mpl", fold=-1, scale=0.6)


def load_recorded():
    with open(RECORDED) as f:
        return f.read()


def export_take(backend_name, settings_json, meta_json, counts_json, date):
    """Hand a JSON file to the Qollab console (copy button) so a real run can be saved as data/recorded_aria1.json."""
    take = {
        "source": "qollab-run", "backend": backend_name, "date": date,
        "settings": json.loads(settings_json), "meta": json.loads(meta_json), "counts": json.loads(counts_json),
    }
    text = json.dumps(take)
    try:
        from js import sendFile  # only exists inside the Qollab runtime
        sendFile("recorded_take.json", "application/json", text)
    except ImportError:
        pass
    return text
```

- [ ] **Step 4: Implement `tools/record_take.py` and generate the stand-in take**

```python
"""Generate data/recorded_aria1.json locally with Aer as a stand-in until the real IonQ Aria 1 run.
Run from the project root: python tools/record_take.py
"""
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qec.lessons import build_all

SETTINGS = {"shots": 500, "inject": {"pauli": "X", "qubit": 1}}


def main():
    circuits, meta = build_all(SETTINGS)
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.0005, 1), ["h", "x", "y", "z", "t", "sdg", "ry"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.006, 2), ["cx"])
    sim = AerSimulator(noise_model=noise)
    result = sim.run(transpile(circuits, sim, optimization_level=0), shots=SETTINGS["shots"]).result()
    counts = [dict(result.get_counts(i)) for i in range(len(circuits))]
    take = {
        "source": "local-aer-stand-in",
        "backend": "AerSimulator depolarizing (stand-in, replace with IonQ Aria 1 noise model)",
        "date": datetime.date.today().isoformat(),
        "settings": SETTINGS, "meta": meta, "counts": counts,
    }
    out = Path(__file__).resolve().parents[1] / "data" / "recorded_aria1.json"
    out.write_text(json.dumps(take))
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
```

Run: `python tools/record_take.py`
Expected: `wrote ... bytes`, file well under 2 MB.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest -q`
Expected: everything passes.

- [ ] **Step 6: Commit**

```bash
git add qec/bridge.py tools/record_take.py data/recorded_aria1.json tests/test_bridge.py
git commit -m "Add JS bridge, recorded take stand-in and export"
```

---

### Task 6: Page skeleton, styles and state machine

**Files:**
- Create: `index.html`, `main.css`, `main.js`, `ui/lessons.js`

- [ ] **Step 1: Write `index.html` (body content only, the runtime strips html/head/body)**

```html
<div class="app" id="app">
  <header class="hdr">
    <h1>Logical teleportation on a tiny surface code</h1>
    <p class="sub">Three logical qubits, each a 2x2 patch of the [[4,1,2]] code. One job, four lessons, no feed-forward needed.</p>
    <p class="banner" id="banner" aria-live="polite"></p>
  </header>

  <section class="setup" id="setup" aria-labelledby="setup-h">
    <h2 id="setup-h">Set up the run</h2>
    <div class="row">
      <label>Shots <input id="shots" type="number" min="100" max="2000" step="100" value="500"></label>
      <label>Lesson 1 injected error
        <select id="inject-pauli">
          <option value="">none</option><option>X</option><option>Y</option><option>Z</option>
        </select>
      </label>
      <label>on data qubit
        <select id="inject-qubit"><option>0</option><option>1</option><option>2</option><option>3</option></select>
      </label>
    </div>
    <div class="row">
      <button id="submit" class="primary">Submit one job with all 16 circuits</button>
      <button id="load-recorded">Load recorded take</button>
      <button id="export" hidden>Export this run as JSON</button>
    </div>
    <p class="status" id="status" aria-live="polite"></p>
  </section>

  <nav class="tabs" role="tablist" aria-label="Lessons" id="tabs" hidden>
    <button role="tab" id="tab-1" aria-controls="panel" aria-selected="true" data-lesson="1">1. One patch, one error</button>
    <button role="tab" id="tab-2" aria-controls="panel" aria-selected="false" data-lesson="2" tabindex="-1">2. Logical CNOT</button>
    <button role="tab" id="tab-3" aria-controls="panel" aria-selected="false" data-lesson="3" tabindex="-1">3. Teleportation</button>
    <button role="tab" id="tab-4" aria-controls="panel" aria-selected="false" data-lesson="4" tabindex="-1">4. The T gate</button>
  </nav>

  <section class="panel" id="panel" role="tabpanel" aria-labelledby="tab-1" hidden>
    <div class="grid">
      <div class="patches" id="patches"></div>
      <div class="text" id="lesson-text"></div>
    </div>
    <div class="cards" id="cards"></div>
    <div class="charts" id="charts"></div>
    <details class="circuit" id="circuit-details">
      <summary>Show the physical circuit</summary>
      <div id="circuit-img"></div>
    </details>
  </section>

  <footer class="foot">
    <p>Distance 2 detects any single error and corrects none. Accepted shots are post-selected. The encoder is not fault tolerant.</p>
    <p>This effort is supported by Qollab &amp; IonQ.</p>
  </footer>
</div>
```

- [ ] **Step 2: Write `main.css`**

```css
:root {
  --bg: #0f1115; --fg: #e8eaf0; --muted: #a7adbb; --card: #181b22; --line: #2a2f3a;
  --x: #e0a040; --z: #4f9cff; --good: #39c37a; --bad: #ff5c5c; --focus: #ffd166;
}
@media (prefers-color-scheme: light) {
  :root { --bg: #fafbfd; --fg: #15181f; --muted: #525866; --card: #ffffff; --line: #d9dde6; }
}
* { box-sizing: border-box; }
.app { font: 16px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg); padding: 16px; max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 0 0 8px; }
.sub, .status, .foot p { color: var(--muted); margin: 4px 0; }
.banner { margin: 8px 0 0; padding: 8px 12px; border: 1px solid var(--line); border-radius: 8px; }
.banner.warn { border-color: var(--x); }
.banner.err { border-color: var(--bad); }
.setup, .panel { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-top: 16px; }
.row { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-bottom: 12px; }
label { display: grid; gap: 4px; font-size: 0.9rem; color: var(--muted); }
input, select, button { font: inherit; color: var(--fg); background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
button { cursor: pointer; }
button.primary { background: var(--z); color: #fff; border-color: var(--z); }
button:disabled { opacity: 0.5; cursor: not-allowed; }
:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
.tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.tabs [role="tab"][aria-selected="true"] { background: var(--card); border-color: var(--fg); font-weight: 600; }
.grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 800px) { .grid { grid-template-columns: 1.2fr 1fr; } }
.patches { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.patches svg { max-width: 100%; height: auto; }
.patch-title { font-size: 0.85rem; color: var(--muted); text-align: center; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }
.card { border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
.card .big { font-size: 1.6rem; font-weight: 700; }
.card .lbl { color: var(--muted); font-size: 0.85rem; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-top: 16px; }
.chart h3 { font-size: 0.95rem; margin: 0 0 6px; }
.chart svg { width: 100%; height: auto; }
.fired-list { font-size: 0.85rem; color: var(--muted); margin: 6px 0 0; }
.circuit { margin-top: 16px; }
.circuit img { max-width: 100%; background: #fff; border-radius: 8px; }
.note { border-left: 3px solid var(--x); padding-left: 10px; color: var(--muted); }
@media (prefers-reduced-motion: no-preference) { .face { transition: fill-opacity 200ms; } }
```

- [ ] **Step 3: Write `ui/lessons.js`**

```js
export const LESSONS = {
  1: {
    title: "One patch, one error",
    html: `
<p>A logical qubit here is four physical qubits on the corners of a square. Three <strong>checks</strong> watch it:
the face measures <em>XXXX</em>, the top and bottom edges measure <em>Z0Z1</em> and <em>Z2Z3</em>.
Logical Z lives on the left edge (<em>Z0Z2</em>), logical X on the top edge (<em>X0X1</em>).</p>
<p>Any single X, Y or Z error on any corner flips at least one check. The code cannot say which corner,
so it cannot fix it. It can <strong>flag</strong> the shot, and we throw flagged shots away. That is error detection, distance 2.</p>
<p><strong>Try it:</strong> inject an error in the setup and resubmit. The bare physical qubit next to it gets the same error and
has no idea. The patch lights up.</p>
<p class="note">The encoder is not fault tolerant: an error inside those five gates can slip through. Real experiments
start the same way, then post-select.</p>`,
  },
  2: {
    title: "Logical CNOT is four physical CNOTs",
    html: `
<p>For this code family a logical CNOT is <strong>transversal</strong>: CNOT corner 0 to corner 0, 1 to 1, 2 to 2, 3 to 3.
No gate ever touches two qubits of the same patch, so one faulty gate makes at most one error per patch, and the checks still catch it.</p>
<p>We prepare patch A in |+&gt; and patch B in |0&gt;, apply the four CNOTs and get a <strong>logical Bell pair</strong>.
Read both in Z: the parities agree. Read both in X: they agree too. That pair of facts is entanglement.</p>
<p class="note">On IonQ's trapped ions every qubit talks to every other, so these four CNOTs need no SWAP gates.
On a grid chip the patches would have to sit next to each other.</p>`,
  },
  3: {
    title: "Teleporting a logical qubit",
    html: `
<p>Patch A holds a state, <em>ry(pi/4)|0&gt;</em>. Patches B and C share a logical Bell pair. A Bell measurement on A and B
is a transversal CNOT followed by reading A in X and B in Z. Patch C now holds the state, up to a Pauli correction
that depends on the two readouts.</p>
<p>No feed-forward is needed: we read C and apply the correction <strong>in software</strong>, flipping the recorded
bit when the readouts say so. Hardware experiments do exactly this and call it the Pauli frame.</p>
<p>Check: in both bases the chance of reading 1 should be 14.6 percent, the same as the input state.</p>`,
  },
  4: {
    title: "The T gate is a teleportation too",
    html: `
<p>Every gate so far was Clifford. The <strong>T gate</strong> is not transversal in this code, or in any code of this kind.
The way out is to prepare a <em>magic state</em> T|+&gt; in its own patch, then teleport the gate onto the data.</p>
<p>Patch B holds the magic state, patch A holds the same input as lesson 3, <em>ry(pi/4)|0&gt;</em>. A transversal CNOT from B to A and a Z readout of A leave
T|psi&gt; on B half the time. The other half needs an S correction that requires feed-forward, which IonQ's simulator
does not offer, so we <strong>keep only the shots where A read 0</strong>. The kept fraction should be about one half.</p>
<p>Check: reading B in X and in Y should both give 1 about 25 percent of the time. Without the T gate, X would give 14.6 percent and Y 50 percent.
We do not use |+&gt; as the input on purpose: T|+&gt; is the magic state itself, so nothing would visibly change.</p>
<p class="note">Fault-tolerant computers spend most of their effort making clean magic states. This lesson is why.</p>`,
  },
};
```

- [ ] **Step 4: Write `main.js`**

```js
import * as bridge from 'qec.bridge';
import { LESSONS } from './ui/lessons.js';
import { renderPatches } from './ui/patches.js';
import { renderCards, renderCharts } from './ui/charts.js';

const $ = (id) => document.getElementById(id);
const state = { settings: null, metaJson: null, meta: null, countsJson: null, results: null, lesson: 1, backendName: '', isIonq: false, source: '' };

function backendLabel() {
  try { return String(backend.name && typeof backend.name === 'function' ? backend.name() : backend.name); }
  catch { return 'unknown backend'; }
}

function setBanner(text, cls = '') { const b = $('banner'); b.textContent = text; b.className = 'banner ' + cls; }
function setStatus(text) { $('status').textContent = text; }

function readSettings() {
  const shots = Math.min(2000, Math.max(100, parseInt($('shots').value, 10) || 500));
  const pauli = $('inject-pauli').value;
  const inject = pauli ? { pauli, qubit: parseInt($('inject-qubit').value, 10) } : null;
  return { shots, inject };
}

async function submit() {
  const settings = readSettings();
  state.settings = settings;
  $('submit').disabled = true;
  setStatus('Building 16 circuits...');
  try {
    state.metaJson = await bridge.build.callPromising(JSON.stringify(settings));
    state.meta = JSON.parse(state.metaJson);
    const circuits = await bridge.circuits.callPromising();
    setStatus(`Submitting one job with 16 circuits, ${settings.shots} shots each, to ${state.backendName}...`);
    const job = await backend.run(circuits, { shots: settings.shots });
    // Proxies to the same Python enum member are not === in JS, so compare by name.
    const statusName = async () => { const s = await job.status(); try { return String(s.name ?? s); } catch { return String(s); } };
    let status = await statusName();
    let ticks = 0;
    while (!['DONE', 'ERROR', 'CANCELLED'].some((k) => status.includes(k))) {
      await setTimeout(1000);
      status = await statusName();
      ticks += 1;
      setStatus(`Job ${status}... ${ticks}s. Noise-model simulators run one shot at a time, so this scales with shots.`);
    }
    if (!status.includes('DONE')) throw new Error('Job ended with status ' + status);
    const result = await job.result();
    const counts = [];
    for (let i = 0; i < state.meta.length; i++) {
      counts.push(Object.fromEntries(result.get_counts(i).toJs()));
    }
    state.countsJson = JSON.stringify(counts);
    state.source = `live run on ${state.backendName}`;
    await explore();
    $('export').hidden = false;
    if (!state.isIonq) $('submit').disabled = false;
    else setStatus('Done. IonQ backends allow one job per run: press Stop, then Run, to submit again.');
  } catch (e) {
    setBanner('Run failed: ' + (e && e.message ? e.message : String(e)), 'err');
    setStatus('');
    $('submit').disabled = false;
  }
}

async function loadRecorded() {
  setStatus('Loading recorded take...');
  const take = JSON.parse(await bridge.load_recorded.callPromising());
  state.settings = take.settings;
  state.metaJson = JSON.stringify(take.meta);
  state.meta = take.meta;
  state.countsJson = JSON.stringify(take.counts);
  state.source = `recorded take: ${take.backend}, ${take.date} (${take.source})`;
  await bridge.build.callPromising(JSON.stringify(take.settings));   // so circuit drawings match
  await explore();
}

async function explore() {
  setStatus('Analysing...');
  state.results = JSON.parse(await bridge.analyze.callPromising(state.metaJson, state.countsJson));
  $('tabs').hidden = false;
  $('panel').hidden = false;
  showLesson(state.lesson);
  setStatus(`Showing ${state.source}. Shots per circuit: ${state.settings.shots}.`);
  $('panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showLesson(n) {
  state.lesson = n;
  document.querySelectorAll('[role="tab"]').forEach((t) => {
    const on = Number(t.dataset.lesson) === n;
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
  });
  $('panel').setAttribute('aria-labelledby', 'tab-' + n);
  const rows = state.results.filter((r) => r.lesson === n);
  $('lesson-text').innerHTML = `<h2>${LESSONS[n].title}</h2>` + LESSONS[n].html;
  renderPatches($('patches'), n, rows, state.settings);
  renderCards($('cards'), rows);
  renderCharts($('charts'), rows);
  $('circuit-img').innerHTML = '';
  $('circuit-details').open = false;
}

async function drawCircuit() {
  if (!$('circuit-details').open || $('circuit-img').childElementCount) return;
  const row = state.results.find((r) => r.lesson === state.lesson && r.kind === 'enc');
  $('circuit-img').textContent = 'Drawing...';
  try {
    const fig = await bridge.draw.callPromising(row.index);
    const img = document.createElement('img');
    img.alt = `Physical circuit for lesson ${state.lesson}, ${row.name}`;
    img.src = URL.createObjectURL(fig);
    $('circuit-img').replaceChildren(img);
  } catch (e) {
    $('circuit-img').textContent = 'Drawing needs the Qiskit Visualizations library enabled in the Libraries list. ' + e;
  }
}

async function exportRun() {
  const text = await bridge.export_take.callPromising(state.backendName, JSON.stringify(state.settings), state.metaJson, state.countsJson, new Date().toISOString().slice(0, 10));
  console.log('Exported run JSON (also offered as a file in the console):');
  console.log(text.slice(0, 200) + '...');
}

function wire() {
  state.backendName = backendLabel();
  state.isIonq = /ionq|aria|forte/i.test(state.backendName);
  setBanner(state.isIonq
    ? `Backend: ${state.backendName}. IonQ backends accept one job per run, so all 16 circuits go in one submission.`
    : `Backend: ${state.backendName}. Local simulator, you can resubmit as often as you like.`, state.isIonq ? 'warn' : '');
  $('submit').addEventListener('click', submit);
  $('load-recorded').addEventListener('click', loadRecorded);
  $('export').addEventListener('click', exportRun);
  $('circuit-details').addEventListener('toggle', drawCircuit);
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => showLesson(Number(t.dataset.lesson)));
    t.addEventListener('keydown', (e) => {
      const dir = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!dir) return;
      e.preventDefault();
      const next = tabs[(i + dir + tabs.length) % tabs.length];
      next.focus();
      showLesson(Number(next.dataset.lesson));
    });
  });
  setStatus('Choose shots, optionally inject an error for lesson 1, then submit. Or load the recorded take.');
}

wire();
```

- [ ] **Step 5: Syntax-check the JavaScript**

Run: `node --check main.js && node --check ui/lessons.js && echo ok`
Expected: `ok` (imports of `qiskit...` and `qec.bridge` are only resolved on Qollab; `--check` parses without resolving).

- [ ] **Step 6: Commit**

```bash
git add index.html main.css main.js ui/lessons.js
git commit -m "Add page skeleton, styles, lesson copy and run state machine"
```

---

### Task 7: SVG patches

**Files:**
- Create: `ui/patches.js`

- [ ] **Step 1: Write `ui/patches.js`**

```js
// Draws one SVG patch per logical qubit for the current lesson, lit by syndrome fire rates.
const NS = 'http://www.w3.org/2000/svg';
const BLOCK_NAMES = { 1: ['A'], 2: ['A', 'B'], 3: ['A', 'B', 'C'], 4: ['A', 'B'] };
const ROLE = {
  1: { A: 'the logical qubit' },
  2: { A: 'control, |+>', B: 'target, |0>' },
  3: { A: 'input psi', B: 'half of Bell pair', C: 'output' },
  4: { A: 'data |+>', B: 'magic state T|+>' },
};
const LINKS = { 2: [['A', 'B']], 3: [['B', 'C'], ['A', 'B']], 4: [['B', 'A']] };

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function pct(x) { return `${(100 * x).toFixed(1)}%`; }

export function renderPatches(root, lesson, rows, settings) {
  root.replaceChildren();
  const enc = rows.find((r) => r.kind === 'enc') || rows[0];
  const names = BLOCK_NAMES[lesson];
  const blocks = enc.blocks || [];
  const W = 170, H = 190, n = names.length;
  const svg = el('svg', { viewBox: `0 0 ${W * n} ${H}`, role: 'img', width: W * n, height: H });
  el('title', {}, svg).textContent = `${n} patch${n > 1 ? 'es' : ''} of the [[4,1,2]] code with syndrome fire rates`;
  const fired = [];
  names.forEach((name, bi) => {
    const f = blocks[bi] || { x_face: 0, z_top: 0, z_bottom: 0, zzzz_anc: 0 };
    const basis = enc.bases[bi];
    const zRead = basis === 'Z';   // Z0Z1 and Z2Z3 are only readable from a Z readout
    const g = el('g', { transform: `translate(${bi * W + 25}, 30)` }, svg);
    const P = { q0: [0, 0], q1: [120, 0], q2: [0, 120], q3: [120, 120] };
    // face: XXXX
    const face = el('rect', { x: 0, y: 0, width: 120, height: 120, fill: 'var(--x)', 'fill-opacity': 0.12 + 0.88 * f.x_face, stroke: 'var(--line)', class: 'face' }, g);
    el('title', {}, face).textContent = basis === 'Y'
      ? `XXXX check from the ancilla, fired in ${pct(f.x_face)} of shots`
      : `XXXX check, fired in ${pct(f.x_face)} of shots`;
    // top and bottom semicircles: Z0Z1, Z2Z3; dashed outline when this readout cannot measure them
    const zLook = (rate) => (zRead
      ? { fill: 'var(--z)', 'fill-opacity': 0.12 + 0.88 * rate }
      : { fill: 'none', 'stroke-dasharray': '4 3' });
    const top = el('path', { d: 'M0,0 A60,60 0 0 1 120,0 Z', ...zLook(f.z_top), stroke: 'var(--line)', class: 'face' }, g);
    el('title', {}, top).textContent = zRead
      ? `Z0Z1 check, fired in ${pct(f.z_top)} of shots`
      : 'Z0Z1 check, not measured in this readout';
    const bot = el('path', { d: 'M0,120 A60,60 0 0 0 120,120 Z', ...zLook(f.z_bottom), stroke: 'var(--line)', class: 'face' }, g);
    el('title', {}, bot).textContent = zRead
      ? `Z2Z3 check, fired in ${pct(f.z_bottom)} of shots`
      : 'Z2Z3 check, not measured in this readout';
    // logical operators as edges with hover titles
    const zl = el('line', { x1: 0, y1: 0, x2: 0, y2: 120, stroke: 'var(--fg)', 'stroke-width': 4, 'stroke-dasharray': '6 4', tabindex: 0 }, g);
    el('title', {}, zl).textContent = 'Logical Z = Z0 Z2 (left edge)';
    const xl = el('line', { x1: 0, y1: 0, x2: 120, y2: 0, stroke: 'var(--fg)', 'stroke-width': 4, 'stroke-dasharray': '2 4', tabindex: 0 }, g);
    el('title', {}, xl).textContent = 'Logical X = X0 X1 (top edge)';
    // data qubits
    Object.entries(P).forEach(([q, [x, y]], i) => {
      const hit = settings && settings.inject && lesson === 1 && settings.inject.qubit === i;
      const dot = el('circle', { cx: x, cy: y, r: 10, fill: hit ? 'var(--bad)' : 'var(--card)', stroke: 'var(--fg)', 'stroke-width': 2 }, g);
      el('title', {}, dot).textContent = hit ? `data qubit ${i}, injected ${settings.inject.pauli} error` : `data qubit ${i}`;
      el('text', { x: x + (x ? 14 : -14), y: y + (y ? 24 : -14), 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' }, g).textContent = `q${i}`;
    });
    el('text', { x: 60, y: 150, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--fg)', 'font-weight': 600 }, g).textContent = `Patch ${name}`;
    el('text', { x: 60, y: 166, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }, g).textContent = ROLE[lesson][name];
    const list = [['XXXX', f.x_face], ['Z0Z1', f.z_top], ['Z2Z3', f.z_bottom], ['ZZZZ ancilla', f.zzzz_anc]]
      .filter(([, v]) => v > 0).map(([k, v]) => `${k} ${pct(v)}`);
    const unread = zRead ? '' : ` (Z edges not measured, basis ${basis})`;
    fired.push(`Patch ${name}: ${list.length ? list.join(', ') : 'no check fired'}${unread}`);
  });
  // transversal CNOT links between corresponding corners
  (LINKS[lesson] || []).forEach(([a, b]) => {
    const ia = names.indexOf(a), ib = names.indexOf(b);
    [[0, 0], [120, 0], [0, 120], [120, 120]].forEach(([x, y]) => {
      const l = el('line', { x1: ia * W + 25 + x, y1: 30 + y, x2: ib * W + 25 + x, y2: 30 + y, stroke: 'var(--muted)', 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, svg);
      el('title', {}, l).textContent = `transversal CNOT ${a} to ${b}`;
    });
  });
  root.appendChild(svg);
  const p = document.createElement('p');
  p.className = 'fired-list';
  p.textContent = 'Checks fired (share of all shots). ' + fired.join(' | ');
  root.appendChild(p);
}
```

- [ ] **Step 2: Syntax-check**

Run: `node --check ui/patches.js && echo ok`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add ui/patches.js
git commit -m "Draw SVG patches lit by syndrome fire rates"
```

---

### Task 8: Cards and charts

**Files:**
- Create: `ui/charts.js`

- [ ] **Step 1: Write `ui/charts.js`**

```js
const NS = 'http://www.w3.org/2000/svg';

function fmt(x) { return x === null || x === undefined ? 'n/a' : `${(100 * x).toFixed(1)}%`; }

function card(big, lbl, title) {
  const d = document.createElement('div');
  d.className = 'card';
  if (title) d.title = title;
  d.innerHTML = `<div class="big">${big}</div><div class="lbl">${lbl}</div>`;
  return d;
}

export function renderCards(root, rows) {
  root.replaceChildren();
  const encs = rows.filter((r) => r.kind === 'enc');
  const bares = rows.filter((r) => r.kind === 'bare');
  const mean = (xs) => { const v = xs.filter((x) => x !== null && x !== undefined); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  root.appendChild(card(fmt(mean(encs.map((r) => r.flagged_fraction))), 'shots flagged by a check and discarded'));
  root.appendChild(card(fmt(mean(encs.map((r) => r.deviation_accepted))), 'logical error of accepted shots', 'absolute gap between measured and ideal P(1), averaged over the readouts shown'));
  root.appendChild(card(fmt(mean(bares.map((r) => r.deviation_all))), 'same experiment on bare physical qubits'));
  if (encs[0] && encs[0].kept_fraction !== null && encs[0].name.startsWith('enc') && rows[0].lesson === 4) {
    root.appendChild(card(fmt(mean(encs.map((r) => r.kept_fraction))), 'kept after post-selecting A = 0 (ideal 50%)'));
  }
}

function bars(title, items) {
  // items: [{label, value, ideal?}] values in [0,1]
  const wrap = document.createElement('div');
  wrap.className = 'chart';
  const h = document.createElement('h3');
  h.textContent = title;
  wrap.appendChild(h);
  const W = 320, rowH = 26, pad = 110, H = items.length * rowH + 24;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('role', 'img');
  const t = document.createElementNS(NS, 'title');
  t.textContent = title + ': ' + items.map((i) => `${i.label} ${fmt(i.value)}`).join(', ');
  svg.appendChild(t);
  items.forEach((it, i) => {
    const y = 8 + i * rowH;
    const lab = document.createElementNS(NS, 'text');
    lab.setAttribute('x', pad - 6); lab.setAttribute('y', y + 14); lab.setAttribute('text-anchor', 'end');
    lab.setAttribute('font-size', 12); lab.setAttribute('fill', 'var(--fg)');
    lab.textContent = it.label;
    svg.appendChild(lab);
    const v = it.value === null || it.value === undefined ? 0 : it.value;
    const r = document.createElementNS(NS, 'rect');
    r.setAttribute('x', pad); r.setAttribute('y', y); r.setAttribute('height', 18);
    r.setAttribute('width', Math.max(1, (W - pad - 50) * v));
    r.setAttribute('fill', it.color || 'var(--z)');
    svg.appendChild(r);
    const val = document.createElementNS(NS, 'text');
    val.setAttribute('x', pad + (W - pad - 50) * v + 4); val.setAttribute('y', y + 14);
    val.setAttribute('font-size', 12); val.setAttribute('fill', 'var(--fg)');
    val.textContent = fmt(it.value);
    svg.appendChild(val);
    if (it.ideal !== undefined) {
      const x = pad + (W - pad - 50) * it.ideal;
      const m = document.createElementNS(NS, 'line');
      m.setAttribute('x1', x); m.setAttribute('x2', x); m.setAttribute('y1', y - 2); m.setAttribute('y2', y + 20);
      m.setAttribute('stroke', 'var(--good)'); m.setAttribute('stroke-width', 2);
      const mt = document.createElementNS(NS, 'title');
      mt.textContent = `ideal ${fmt(it.ideal)}`;
      m.appendChild(mt);
      svg.appendChild(m);
    }
  });
  wrap.appendChild(svg);
  return wrap;
}

export function renderCharts(root, rows) {
  root.replaceChildren();
  const labels = [...new Set(rows.map((r) => r.label))];
  labels.forEach((label) => {
    const enc = rows.find((r) => r.kind === 'enc' && r.label === label);
    const bare = rows.find((r) => r.kind === 'bare' && r.label === label);
    const items = [
      { label: 'accepted (encoded)', value: enc.p1_accepted, ideal: enc.ideal_p1, color: 'var(--z)' },
      { label: 'all shots (encoded)', value: enc.p1_all, ideal: enc.ideal_p1, color: 'var(--muted)' },
      { label: 'bare physical', value: bare ? bare.p1_all : null, ideal: enc.ideal_p1, color: 'var(--x)' },
    ];
    root.appendChild(bars(`P(logical = 1), readout ${label}. Green mark = ideal ${fmt(enc.ideal_p1)}`, items));
  });
}
```

- [ ] **Step 2: Syntax-check**

Run: `node --check ui/charts.js && echo ok`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add ui/charts.js
git commit -m "Add summary cards and SVG bar charts"
```

---

### Task 9: README / project page

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`** (also pasted into the Qollab project card, which takes Markdown and math)

```markdown
# Logical teleportation on a tiny surface code

Three logical qubits, each a 2x2 patch of the [[4,1,2]] rotated surface code, running on IonQ's simulator. Four lessons from one job:

1. **One patch, one error.** Inject an X, Y or Z on any corner. The checks flag it; the bare qubit next to it has no idea.
2. **Logical CNOT.** Four physical CNOTs between two patches make a logical Bell pair. No SWAPs on trapped ions.
3. **Logical teleportation.** A state crosses from patch A to patch C through a Bell measurement. Pauli corrections are applied in software (the Pauli frame), so no feed-forward is needed.
4. **The teleported T gate.** T is not transversal. We teleport it in with a magic state, keep the shots that need no correction, and check that X and Y readouts both give 25 percent, which neither the input nor the magic state alone would.

## How to run

Press **Run on Qollab**, pick a backend (built-in simulator for instant results, IonQ Aria 1 noise model for realistic noise), then:

- choose shots (100 to 2000) and, for lesson 1, an injected error,
- press **Submit one job with all 16 circuits**, or **Load recorded take** to see a saved IonQ run without waiting,
- click through the four lesson tabs. Hover any face or edge for its meaning. "Show the physical circuit" draws the real gates.

IonQ backends allow one job per run, so all 16 circuits travel together. On local simulators you can resubmit freely.

## What the numbers mean

- **Flagged:** a check fired, the shot is discarded. Distance 2 detects any single error and corrects none.
- **Logical error of accepted shots:** gap between the measured and the ideal probability of reading logical 1.
- **Bare physical:** the same experiment with one physical qubit per logical qubit.
- Lesson 4 also shows the **kept fraction** after post-selecting on patch A reading 0, ideally one half.

## The code

Corner qubits q0 q1 / q2 q3. Checks: XXXX (face), Z0Z1 (top), Z2Z3 (bottom). Logical Z = Z0Z2, logical X = X0X1. Encoder: `h q2; cx q2,q3; cx q0,q1; cx q2,q0; cx q2,q1`. Every block gets two ancillas measured at the end, plus the checks that the final readout basis gives for free. Everything is measured at the end, so no mid-circuit measurement, reset or feed-forward is used.

## Honesty notes

The encoder is not fault tolerant. Post-selection throws shots away and would not scale. Lesson 4 discards half the shots because the S correction needs feed-forward. These are the same compromises the first logical-gate experiments on trapped ions and superconducting chips made.

## Files

`main.js` runs the page and the single job. `qec/` holds the physics in Python: `code412.py` (code), `lessons.py` (16 circuits), `analyze.py` (counts to numbers), `bridge.py` (JSON entry points). `tests/` run with `python -m pytest`. `data/recorded_aria1.json` is a saved run.

## Credits

Error detection with small codes on trapped ions: Linke et al., Sci. Adv. 3, e1701074 (2017). The [[4,2,2]] and [[4,1,2]] family: Gottesman, arXiv:1610.03507. Magic states and the T gadget: Bravyi and Kitaev, Phys. Rev. A 71, 022316 (2005). Teleportation: Bennett et al., Phys. Rev. Lett. 70, 1895 (1993). Multi-file and recorded-take patterns from the Qollab runtime docs.

Built during the Qollab x IonQ hackathon, Oct 9 to 11, 2026, by team ZollerQu. AI tools used: Claude Code for planning and implementation. MIT licence.

This effort is supported by Qollab & IonQ.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add project page text"
```

---

### Task 10: Qollab smoke test and recorded take (manual, needs the Qollab login)

**Files:**
- Modify: `data/recorded_aria1.json` (replace the stand-in)

- [ ] **Step 1: Create the project on Qollab** as a JS/Qiskit project named "Logical teleportation on a tiny surface code". Upload every file keeping paths: `index.html`, `main.css`, `main.js`, `ui/*.js`, `qec/*.py`, `data/recorded_aria1.json`, `requirements.txt`, `README.md`. Enable **Qiskit Visualizations** in the Libraries list (it rewrites `requirements.txt` to the same line).

- [ ] **Step 2: Built-in simulator.** Run, Load recorded take, confirm the four tabs render patches, cards and charts. Then Submit with 500 shots and no injection: expect flagged near 0, deviations below 3 percent, lesson 4 kept near 50 percent. Open "Show the physical circuit".

- [ ] **Step 3: Injection check.** Submit with X on qubit 1: lesson 1 patch shows the top semicircle and ZZZZ ancilla at 100 percent, flagged 100 percent, accepted n/a, bare qubit unflagged and wrong.

- [ ] **Step 4: IonQ Aria 1 noise model, 100 shots.** Confirm one job carries 16 circuits and `get_counts(i)` works for every index. Note the wall-clock time in the console.

- [ ] **Step 5: Record the real take.** Run Aria 1 with 500 shots and X on qubit 1, press Export, copy `recorded_take.json` from the console, save it as `data/recorded_aria1.json` with `"source": "qollab-run"`, re-upload. Run `python -m pytest tests/test_bridge.py -q` locally; expected pass.

- [ ] **Step 6: Commit and publish.**

```bash
git add data/recorded_aria1.json
git commit -m "Replace stand-in with recorded IonQ Aria 1 take"
```

Publish on Qollab, set the version as default, paste README into the project card, add tags `qiskit`, `ionq`, `error-correction`, `teleportation`, `surface-code`, link the GitHub repository.

---

## Self-review

**Spec coverage.** Section 2 code: Task 2. Section 3 layout and recipes: Tasks 2 and 4. Section 4 lessons: Task 3. Section 5 metrics: Task 4 and Task 8 cards. Section 6 architecture and boundary: Tasks 5 and 6. Section 7 run flow, recorded take, export, one-job rule: Tasks 5 and 6. Section 8 layout and accessibility: Tasks 6, 7, 8 (tablist with arrow keys, aria-live, SVG titles, text list of fired checks, reduced motion, 360 px via flex wrap). Section 9 testing: Tasks 2 to 5 and Task 10. Section 12 credits: Task 9.

**Known risks to verify in Task 10, not resolvable locally:** `backend.run` accepting the Python list proxy; `result.get_counts(i)` on the IonQ multi-circuit result; `URL.createObjectURL(fig)` for the circuit drawing; `js.sendFile` from Python inside a JS project; the `backend.name` call shape. Each has a fallback message in `main.js`.

**Type consistency.** `bridge.build(settings_json)` returns metadata JSON and caches circuits; `bridge.circuits()` returns the list; `bridge.analyze(meta_json, counts_json)`; `bridge.draw(index)`; `bridge.load_recorded()`; `bridge.export_take(backend_name, settings_json, meta_json, counts_json, date)`. Result rows carry `index, lesson, name, kind, label, ideal_p1, n_total, n_flagged, n_accepted, flagged_fraction, kept_fraction, p1_accepted, p1_all, deviation_accepted, deviation_all, blocks[{x_face, z_top, z_bottom, zzzz_anc}]`, used identically in `patches.js` and `charts.js`.
