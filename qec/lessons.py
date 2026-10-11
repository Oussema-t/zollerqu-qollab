"""Build the 16 circuits of the four lessons plus their metadata."""
import math

from qec import code913 as c

IDEAL_P1_ROTATED = math.sin(math.pi / 8) ** 2   # 0.146..., same for every rotated readout we use


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
    if pauli not in ("X", "Y", "Z") or qubit not in range(9):
        raise ValueError(f"bad injection {inject!r}")
    return pauli, int(qubit)


# ---------- lesson 1: one patch, one error ----------

def _lesson1(inject):
    out = []
    for basis, prep in (("Z", False), ("X", True)):
        qc = c.new_circuit(1)
        if prep:
            c.mark(qc, "prepare A")
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
        c.mark(qc, "prepare A")
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


# ---------- lesson 3: a T state on one patch ----------

def _lesson3():
    out = []
    for basis in ("X", "Y"):
        qc = c.new_circuit(1)
        c.mark(qc, "prepare A")
        qc.h(c.data(0, 0))
        qc.t(c.data(0, 0))
        c.encode(qc, 0)                 # A = |T_L> = T|+_L>
        c.finish(qc, 1, [basis])
        out.append((qc, _meta(3, f"enc_{basis}", "enc", basis, 1, [basis], [0], IDEAL_P1_ROTATED)))
    for basis in ("X", "Y"):
        qc = c.new_bare(1)
        qc.h(0)
        qc.t(0)
        c.finish_bare(qc, [basis])
        out.append((qc, _meta(3, f"bare_{basis}", "bare", basis, 1, [basis], [0], IDEAL_P1_ROTATED)))
    return out


# ---------- lesson 4: teleporting the T state ----------

def _lesson4():
    out = []
    for basis in ("X", "Y"):
        qc = c.new_circuit(3)
        c.mark(qc, "prepare A")
        qc.h(c.data(0, 0))
        qc.t(c.data(0, 0))
        c.encode(qc, 0)                 # A = |T_L>
        c.mark(qc, "prepare B")
        qc.h(c.data(1, 0))
        c.encode(qc, 1)                 # B = |+_L>
        c.encode(qc, 2)                 # C = |0_L>
        c.transversal_cx(qc, 1, 2)      # logical Bell pair B,C
        c.transversal_cx(qc, 0, 1)      # Bell measurement part 1
        c.finish(qc, 3, ["X", "Z", basis])
        frame = [2, 0] if basis == "X" else [2, 0, 1]   # Z correction (from A) flips X and Y; X correction (from B) flips Y
        out.append((qc, _meta(4, f"enc_{basis}", "enc", basis, 3, ["X", "Z", basis], frame, IDEAL_P1_ROTATED)))
    for basis in ("X", "Y"):
        qc = c.new_bare(3)
        qc.h(0)
        qc.t(0)
        qc.h(1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        c.finish_bare(qc, ["X", "Z", basis])
        frame = [2, 0] if basis == "X" else [2, 0, 1]
        out.append((qc, _meta(4, f"bare_{basis}", "bare", basis, 3, ["X", "Z", basis], frame, IDEAL_P1_ROTATED)))
    return out


def build_all(settings):
    """Return (circuits, metadata) for the given settings dict."""
    inject = _check_injection(settings.get("inject"))
    pairs = _lesson1(inject) + _lesson2() + _lesson3() + _lesson4()
    circuits, meta = [], []
    for index, (qc, m) in enumerate(pairs):
        m["index"] = index
        m["nqubits"] = qc.num_qubits
        qc.name = f"L{m['lesson']}_{m['name']}"
        circuits.append(qc)
        meta.append(m)
    return circuits, meta
