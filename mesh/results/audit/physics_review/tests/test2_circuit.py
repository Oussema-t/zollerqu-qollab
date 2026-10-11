"""Lens item 2: build_qaoa_circuit (Qiskit Statevector) vs numpy simulator, random angles, random J/h, p=1..3, odd/even n."""
import numpy as np
from qiskit.quantum_info import Statevector, Operator
from extract import extract
ns = extract()
rng = np.random.default_rng(777)

def energies(J, h, n):
    k = np.arange(2**n)
    z = 1 - 2 * ((k[:, None] >> np.arange(n)[None, :]) & 1)          # z[k, i], x_i = (k >> i) & 1
    return np.einsum("ki,ij,kj->k", z, np.triu(J, 1), z) + z @ h

worst = {}
worst_amp = {}
count = 0
for n in range(1, 10):
    for with_fields in (False, True):
        for sparse in (False, True):
            for p in (1, 2, 3):
                for trial in range(3):
                    J = np.triu(rng.normal(size=(n, n)), 1)
                    if sparse:
                        J *= rng.random((n, n)) < 0.5
                    h = rng.normal(size=n) if with_fields else np.zeros(n)
                    if with_fields and sparse and n > 1:
                        h[rng.integers(n)] = 0.0
                    scale = max(np.abs(J).max() if n > 1 else 0.0, np.abs(h).max(), 1e-9)
                    Js, hs = J / scale, h / scale
                    E = energies(Js, hs, n)
                    gam, bet = rng.uniform(-3, 3, p), rng.uniform(-3, 3, p)
                    thetas = 2 * np.arcsin(np.sqrt(rng.uniform(0.02, 0.98, n)))
                    cases = {"standard": (dict(), dict()),
                             "mixer A": (dict(thetas=thetas), dict(thetas=thetas, mixer_axis=thetas)),
                             "mixer B": (dict(thetas=thetas, modified_mixer=True), dict(thetas=thetas, mixer_axis=-thetas))}
                    for name, (ckw, skw) in cases.items():
                        qc = ns["build_qaoa_circuit"](Js, hs, gam, bet, **ckw)
                        sv = Statevector(qc).data
                        ref = ns["qaoa_statevector"](E, gam, bet, n, **skw)
                        key = (name, "odd" if n % 2 else "even")
                        worst[key] = max(worst.get(key, 0), np.abs(np.abs(sv) ** 2 - np.abs(ref) ** 2).max())
                        # amplitudes up to ONE global phase
                        idx = np.argmax(np.abs(ref))
                        phase = sv[idx] / ref[idx]
                        worst_amp[key] = max(worst_amp.get(key, 0), np.abs(sv - phase * ref).max(), abs(abs(phase) - 1))
                        count += 1
print("circuits compared:", count)
for key in sorted(worst):
    print(f"{key}: max |dP| = {worst[key]:.3e}; max amplitude diff up to global phase = {worst_amp[key]:.3e}")

# little-endian count keys: Statevector.probabilities_dict key rightmost char = qubit 0
from qiskit import QuantumCircuit
qc = QuantumCircuit(4); qc.x(0)
print("probabilities_dict for X on qubit 0 of 4:", Statevector(qc).probabilities_dict())
print("index of that state in Statevector.probabilities():", int(np.argmax(Statevector(qc).probabilities())))

# round robin: every pair exactly once, each round disjoint, n = 2..40
for n in range(2, 41):
    rounds = ns["round_robin_pairs"](n)
    listed = [pr for r in rounds for pr in r]
    assert len(listed) == len(set(listed)) == n * (n - 1) // 2, n
    assert all(i < j for i, j in listed)
    for r in rounds:
        flat = [q for pr in r for q in pr]
        assert len(flat) == len(set(flat)), ("round not disjoint", n)
    assert len(rounds) == (n - 1 if n % 2 == 0 else n)
print("round_robin_pairs: all pairs once, rounds disjoint, n-1 (even) / n (odd) rounds for n=2..40: ok")
# depth of the dense layer
for n in (5, 6, 11, 12):
    J = np.triu(np.ones((n, n)), 1)
    qc = ns["build_qaoa_circuit"](J, np.zeros(n), [0.3], [0.2])
    ops = qc.count_ops()
    print(f"n={n}: rzz={ops.get('rzz')} expected {n*(n-1)//2}; depth={qc.depth()} (H + rounds + RX = {2 + (n - 1 if n % 2 == 0 else n)})")
