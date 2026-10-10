"""Layout of an encoded circuit as a spacetime block for ui/spacetime.js.

Plane coordinates: block b sits at x = b * PITCH; local q0 (0,0), q1 (1,0), q2 (0,1), q3 (1,1);
the XXXX ancilla at the face centre (0.5, 0.5), the ZZZZ ancilla below the patch (0.5, 1.7).
Time: integer layers. Gates inside a phase are packed as early as possible; a new phase
starts above every layer used so far, so phases never overlap.
"""
from qec import code412 as c

PITCH = 2.2
LOCAL = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
CAUGHT = {  # which checks a single Pauli on local data qubit i flips
    "X": lambda i: [("z_top" if i in (0, 1) else "z_bottom"), "zzzz_anc"],
    "Z": lambda i: ["x_face"],
    "Y": lambda i: ["x_face", ("z_top" if i in (0, 1) else "z_bottom"), "zzzz_anc"],
}


def _role(phase):
    for key in ("transversal", "syndrome", "encode", "inject", "prepare", "readout", "measure"):
        if phase.startswith(key):
            return key
    return "other"


def _qubits(n):
    out = []
    for b in range(n):
        for i in range(4):
            x, y = LOCAL[i]
            out.append({"id": c.data(b, i), "kind": "data", "block": b, "local": i, "x": b * PITCH + x, "y": y})
    for b in range(n):
        out.append({"id": c.anc_x(n, b), "kind": "ax", "block": b, "local": None, "x": b * PITCH + 0.5, "y": 0.5})
        out.append({"id": c.anc_z(n, b), "kind": "az", "block": b, "local": None, "x": b * PITCH + 0.5, "y": 1.7})
    out.sort(key=lambda q: q["id"])
    return out


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

    blocks = [{"name": c.BLOCK_NAMES[b], "data": [c.data(b, i) for i in range(4)],
               "ax": c.anc_x(n, b), "az": c.anc_z(n, b), "x0": b * PITCH, "y0": 0.0} for b in range(n)]

    inject = None
    for op in ops:
        if op["role"] == "inject":
            qid = op["q"][0]
            b, i = divmod(qid, 4)
            inject = {"q": qid, "t": op["t"], "pauli": op["gate"].upper(), "block": b, "caught": CAUGHT[op["gate"].upper()](i)}

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
