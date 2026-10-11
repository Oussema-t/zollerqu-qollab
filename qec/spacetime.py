"""Layout of an encoded circuit as a spacetime block for the story view.

Plane coordinates: block b sits at x = b * PITCH; local qubit i at (col, row) = (i % 3, i // 3)
(row 0 on top). Data qubits only, no helpers. Time: integer layers. Gates inside a phase are
packed as early as possible; a new phase starts above every layer used so far, so phases never overlap.
"""
from qec import code913 as c

PITCH = 4.4
ROLES = ("transversal", "encode", "inject", "prepare", "readout", "measure")
CHECKS = {  # readout basis: (Pauli parts it sees, check supports, lookup table)
    "Z": ("XY", c.Z_STABS, c.Z_TABLE),
    "X": ("ZY", c.X_STABS, c.X_TABLE),
}


def _role(phase):
    for key in ROLES:
        if phase.startswith(key):
            return key
    return "other"


def _qubits(n):
    return [{"id": c.data(b, i), "kind": "data", "block": b, "local": i, "x": b * PITCH + i % 3, "y": i // 3}
            for b in range(n) for i in range(9)]


def _caught(pauli, i, basis):
    """Check indices that fire for a single Pauli on local qubit i, and the qubit the table corrects."""
    if basis not in CHECKS or pauli not in CHECKS[basis][0]:
        return [], None
    _, stabs, table = CHECKS[basis]
    syndrome = tuple(int(i in s) for s in stabs)
    return [k for k, s in enumerate(syndrome) if s], table.get(syndrome)


GATE = {"h": "H", "t": "T", "tdg": "T†", "s": "S", "sdg": "S†", "ry": "Ry", "x": "X", "y": "Y", "z": "Z", "cx": "CNOT"}


def _qname(q):
    b, i = divmod(q, 9)
    return f"q{i}", c.BLOCK_NAMES[b]


def _gates_on(ops):
    """'S† on q0, H on q1 and q2 of C': gates in op order, qubits grouped per gate."""
    by = {}
    for op in ops:
        by.setdefault(GATE.get(op["gate"], op["gate"]), []).append(_qname(op["q"][0])[0])
    parts = [f"{g} on {', '.join(qs[:-1]) + ' and ' + qs[-1] if len(qs) > 1 else qs[0]}" for g, qs in by.items()]
    return f"{', '.join(parts)} of {_qname(ops[0]['q'][0])[1]}"


def _kind(op):
    """Group key of one op for the layer caption."""
    role, gate, local = op["role"], op["gate"], op["q"][0] % 9
    if role == "encode":
        if gate == "cx" and local == 0:
            return "top row"
        if gate == "h" and local in (4, 5, 6, 7):
            return "sources"
        if gate == "cx" and local in (4, 5, 6, 7):
            return "fan"
    if role == "prepare" and local == 0 and gate in ("h", "t"):
        return f"prepare {gate}"
    if role in ("transversal", "readout", "measure", "inject"):
        return role
    return "other"


def _sentence(kind, ops):
    q, P = _qname(ops[0]["q"][0])
    if kind == "prepare h":
        return f"H on {q} of {P} puts it in |+⟩."
    if kind == "prepare t":
        return f"T on {q} of {P} adds a pi/4 phase to |1⟩: {q} now holds the T state."
    if kind == "top row":
        return " ".join(f"CNOT q0 → {_qname(o['q'][1])[0]} of {P} spreads q0's value along the top row, where logical X lives."
                        for o in ops)
    if kind == "sources":
        return f"H on {', '.join(_qname(o['q'][0])[0] for o in ops)} of {P} makes them the sources of the X checks."
    if kind == "fan":
        pairs = ", ".join(f"{_qname(o['q'][0])[0]} → {_qname(o['q'][1])[0]}" for o in ops)
        if len(ops) == 1:
            return f"CNOT {pairs} of {P} entangles the source with that qubit; once the whole fan-out is done, all eight checks start out satisfied."
        return f"CNOT {pairs} of {P} entangle each source with its target; once the whole fan-out is done, all eight checks start out satisfied."
    if kind == "transversal":
        G = _qname(ops[0]["q"][1])[1]
        n = {9: "Nine"}.get(len(ops), str(len(ops)))
        return f"{n} CNOTs at once, qi of {P} → qi of {G} for every i: one logical CNOT."
    if kind == "readout":
        if ops[0]["phase"].endswith("(Y)"):
            return f"{_gates_on(ops)}, so the measurement reads logical Y."
        if len(ops) == 9 and all(o["gate"] == "h" for o in ops):
            return f"H on all nine qubits of {P} so the measurement reads X."
        return f"{_gates_on(ops)} so the measurement reads X."
    if kind == "measure":
        return "Every qubit is measured; checks and the logical value are parities of these bits."
    if kind == "inject":
        p = ops[0]["gate"].upper()
        if p == "Z":
            return f"A Z error on {q}, put there on purpose. It cannot change a Z measurement, so this readout will not see it."
        return f"{'A' if p == 'Y' else 'An'} {p} error on {q}, put there on purpose. The qubit stays outlined red from here on."
    return " ".join(f"{GATE.get(o['gate'], o['gate'])} on {' → '.join(_qname(x)[0] for x in o['q'])} of {_qname(o['q'][0])[1]}."
                    for o in ops)


def _layer_text(ops):
    """One caption per layer that holds ops: one sentence per kind of op, in op order."""
    layers = {}
    for op in ops:
        groups = layers.setdefault(op["t"], {})
        groups.setdefault(_kind(op), []).append(op)
    return {str(t): " ".join(_sentence(k, g) for k, g in groups.items()) for t, groups in sorted(layers.items())}


def layout(qc, meta):
    n = meta["nblocks"]
    marks = sorted(qc.metadata.get("phases", []))
    ops, phases = [], []
    last = {}            # qubit -> last layer used
    floor = 0            # first layer the current phase may use
    current = None
    mi = 0
    for k, ins in enumerate(qc.data):
        while mi < len(marks) and marks[mi][0] <= k:
            name = marks[mi][1]
            mi += 1
            if name != current:
                if current is not None:
                    phases[-1]["t1"] = max(last.values(), default=-1)
                floor = max(last.values(), default=-1) + 1
                current = name
                phases.append({"name": name, "t0": floor, "t1": floor})
        if current is None:
            current = "prepare"
            phases.append({"name": current, "t0": 0, "t1": 0})
        name = ins.operation.name
        if name == "barrier":
            continue
        qs = [qc.find_bit(q).index for q in ins.qubits]
        t = max([floor] + [last[q] + 1 for q in qs if q in last])
        for q in qs:
            last[q] = t
        ops.append({"t": t, "gate": name, "q": qs, "phase": current, "role": _role(current)})
    if phases:
        phases[-1]["t1"] = max(last.values(), default=0)
    T = phases[-1]["t1"] if phases else 0

    blocks = [{"name": c.BLOCK_NAMES[b], "data": [c.data(b, i) for i in range(9)], "x0": b * PITCH, "y0": 0.0}
              for b in range(n)]

    inject = None
    for op in ops:
        if op["role"] == "inject":
            qid = op["q"][0]
            b, i = divmod(qid, 9)
            pauli = op["gate"].upper()
            caught, fix = _caught(pauli, i, meta["bases"][b])
            inject = {"q": qid, "t": op["t"], "pauli": pauli, "block": b, "local": i,
                      "caught": caught, "corrected": fix is not None, "corrected_qubit": fix}

    sheets = [{"type": typ, "block": b, "t0": 0, "t1": T, "from": None} for b in range(n) for typ in ("Z", "X")]
    for p in phases:
        if p["name"].startswith("transversal"):
            ctrl, targ = p["name"].split()[-1].split("→")
            cb, tb = c.BLOCK_NAMES.index(ctrl), c.BLOCK_NAMES.index(targ)
            sheets.append({"type": "X", "block": tb, "t0": p["t0"], "t1": T, "from": cb})
            sheets.append({"type": "Z", "block": cb, "t0": p["t0"], "t1": T, "from": tb})

    return {
        "lesson": meta["lesson"], "name": meta["name"], "bases": meta["bases"], "T": T,
        "pitch": PITCH, "qubits": _qubits(n), "blocks": blocks, "phases": phases, "ops": ops,
        "inject": inject, "sheets": sheets, "layer_text": _layer_text(ops),
    }
