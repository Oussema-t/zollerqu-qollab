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
