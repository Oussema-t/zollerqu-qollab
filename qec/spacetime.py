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
        "inject": inject, "sheets": sheets,
    }
