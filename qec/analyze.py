"""Turn measurement counts into the numbers the page shows."""
from qec import code913 as c

RECIPES = {
    # basis: (logical support, check supports, lookup table); Y has no checks and no correction
    "Z": (c.Z_L, c.Z_STABS, c.Z_TABLE),
    "X": (c.X_L, c.X_STABS, c.X_TABLE),
    "Y": ((0, 1, 2, 3, 6), (), None),      # Y_L = Y0 X1 X2 Z3 Z6 after the readout rotation
}


def bits_of(key):
    """Qiskit keys are little-endian; return bits indexed by qubit number."""
    return [int(ch) for ch in reversed(key.replace(" ", ""))]


def _parity(v, support):
    return sum(v[i] for i in support) % 2


def block_readout(bits, nblocks, b, basis):
    v = [bits[c.data(b, i)] for i in range(9)]
    logical, stabs, table = RECIPES[basis]
    raw = _parity(v, logical)
    syndrome = tuple(_parity(v, s) for s in stabs)
    if table is None or not any(syndrome):
        return {"logical": raw, "raw_logical": raw, "syndrome": syndrome, "status": "clean", "corrected_qubit": None}
    if syndrome not in table:
        return {"logical": raw, "raw_logical": raw, "syndrome": syndrome, "status": "discarded", "corrected_qubit": None}
    j = table[syndrome]
    return {"logical": raw ^ int(j in logical), "raw_logical": raw, "syndrome": syndrome,
            "status": "corrected", "corrected_qubit": j}


def analyze_one(m, counts):
    """Metrics for one circuit.

    Per shot, a value block is clean (checks quiet), corrected (syndrome in the lookup table) or
    discarded (any other syndrome). The shot is discarded if any value block is, corrected if any
    value block was corrected, clean otherwise. Frames (extra value blocks) and keep_if_zero use
    corrected logical bits for p1_corrected and p1_clean and raw bits for p1_all. n_kept counts
    clean and corrected shots that also pass keep_if_zero.
    """
    nblocks, bases = m["nblocks"], m["bases"]
    enc = m["kind"] == "enc"
    vb, kz = m["value_blocks"], m["keep_if_zero"]
    n = {"total": 0, "clean": 0, "corrected": 0, "discarded": 0}
    kept = {"corrected": [0, 0], "clean": [0, 0], "all": [0, 0]}   # [shots kept, shots reading 1]
    checks = [[0] * len(RECIPES[basis][1]) for basis in bases]
    corrections = [[0] * 9 for _ in range(nblocks)]

    def tally(which, logical, count):
        if all(logical[b] == 0 for b in kz):
            kept[which][0] += count
            kept[which][1] += count * (sum(logical[b] for b in vb) % 2)

    for key, count in counts.items():
        bits = bits_of(key)
        n["total"] += count
        if not enc:
            logical = bits[:nblocks]
            n["clean"] += count
            for which in kept:
                tally(which, logical, count)
            continue
        reads = [block_readout(bits, nblocks, b, bases[b]) for b in range(nblocks)]
        for b, r in enumerate(reads):
            for k, s in enumerate(r["syndrome"]):
                checks[b][k] += count * s
            if r["corrected_qubit"] is not None:
                corrections[b][r["corrected_qubit"]] += count
        status = {reads[b]["status"] for b in vb}
        cls = "discarded" if "discarded" in status else "corrected" if "corrected" in status else "clean"
        n[cls] += count
        tally("all", [r["raw_logical"] for r in reads], count)
        logical = [r["logical"] for r in reads]
        if cls != "discarded":
            tally("corrected", logical, count)
        if cls == "clean":
            tally("clean", logical, count)

    def frac(a, b):
        return (a / b) if b else None

    total = n["total"]
    p1 = {k: frac(ones, shots) for k, (shots, ones) in kept.items()}
    ideal = m["ideal_p1"]
    return {
        "index": m["index"], "lesson": m["lesson"], "name": m["name"], "kind": m["kind"], "label": m["label"],
        "bases": list(bases), "nblocks": nblocks, "ideal_p1": ideal,
        "n_total": total, "n_clean": n["clean"], "n_corrected": n["corrected"], "n_discarded": n["discarded"],
        "n_kept": kept["corrected"][0],
        "discard_fraction": frac(n["discarded"], total),
        "correct_fraction": frac(n["corrected"], total),
        "kept_fraction": frac(kept["corrected"][0], total),
        "p1_corrected": p1["corrected"], "p1_clean": p1["clean"], "p1_all": p1["all"],
        "deviation_corrected": None if p1["corrected"] is None else abs(p1["corrected"] - ideal),
        "deviation_clean": None if p1["clean"] is None else abs(p1["clean"] - ideal),
        "deviation_all": None if p1["all"] is None else abs(p1["all"] - ideal),
        "blocks": [{"basis": bases[b],
                    "checks": [x / total if total else 0 for x in checks[b]],
                    "corrections": [x / total if total else 0 for x in corrections[b]]}
                   for b in range(nblocks)] if enc else [],
    }


def analyze_all(meta, counts_list):
    """meta and counts_list are parallel lists, one entry per circuit."""
    return [analyze_one(m, dict(counts)) for m, counts in zip(meta, counts_list, strict=True)]
