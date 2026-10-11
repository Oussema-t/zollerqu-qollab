"""Numerical sanity test for warm-start QAOA (Egger, Marecek, Woerner, Quantum 5, 479 (2021)).

All checks use the Qiskit gate definitions RX(t)=exp(-i t X/2), RY(t)=exp(-i t Y/2),
RZ(t)=exp(-i t Z/2), RZZ(t)=exp(-i t ZZ/2), and OUR convention:
minimise E(z)=sum_{i<j} J_ij z_i z_j + sum_i h_i z_i, z=1-2x,
U_C(g)=exp(-i g H), standard mixer exp(+i b sum X).
"""
import ast
import itertools

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector, SparsePauliOp

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
RX = lambda t: expm(-1j * t * X / 2)
RY = lambda t: expm(-1j * t * Y / 2)
RZ = lambda t: expm(-1j * t * Z / 2)
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
rng = np.random.default_rng(7)
results = []


def check(name, ok, extra=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")


def theta_of(c):
    return 2 * np.arcsin(np.sqrt(c))


def h_ws(c):  # paper Eq. (2)
    s = 2 * np.sqrt(c * (1 - c))
    return np.array([[2 * c - 1, -s], [-s, 1 - 2 * c]], dtype=complex)


def h_mod(c):  # Eq. (2) with off-diagonal elements multiplied by -1 (Sec. 2.3)
    s = 2 * np.sqrt(c * (1 - c))
    return np.array([[2 * c - 1, s], [s, 1 - 2 * c]], dtype=complex)


# ---------------------------------------------------------------- A. Eq. (2) mixer
errs = {k: 0.0 for k in "a1 a2 a3 a4 a5".split()}
for c in rng.uniform(0.01, 0.99, 25):
    th = theta_of(c)
    errs["a1"] = max(errs["a1"], np.abs(h_ws(c) - (-np.sin(th) * X - np.cos(th) * Z)).max())
    psi = RY(th) @ ket0
    errs["a3"] = max(errs["a3"], np.abs(h_ws(c) @ psi - (-1) * psi).max())
    errs["a4"] = max(errs["a4"], abs(abs(psi[1]) ** 2 - c))
    for b in rng.uniform(-3, 3, 5):
        errs["a2"] = max(errs["a2"], np.abs(expm(-1j * b * h_ws(c)) - RY(th) @ RZ(-2 * b) @ RY(-th)).max())
        errs["a5"] = max(errs["a5"], np.abs(RY(th) @ RZ(-2 * b) @ RY(-th) @ psi - np.exp(1j * b) * psi).max())
check("A1 Eq.(2) matrix == -sin(th) X - cos(th) Z, th=2 arcsin sqrt(c)", errs["a1"] < 1e-12, f"err={errs['a1']:.1e}")
check("A2 exp(-i b H_ws) == RY(th) RZ(-2b) RY(-th) (operator product, rightmost first)", errs["a2"] < 1e-12, f"err={errs['a2']:.1e}")
check("A3 RY(th)|0> is eigenvector of H_ws with eigenvalue -1 (ground state)", errs["a3"] < 1e-12, f"err={errs['a3']:.1e}")
check("A4 P(|1>) of RY(th)|0> equals c", errs["a4"] < 1e-12, f"err={errs['a4']:.1e}")
check("A5 mixer leaves initial state invariant up to phase exp(+i b)", errs["a5"] < 1e-12, f"err={errs['a5']:.1e}")
e = max(np.abs(RY(np.pi / 2) @ RZ(-2 * b) @ RY(-np.pi / 2) - expm(1j * b * X)).max() for b in rng.uniform(-3, 3, 10))
e2 = max(np.abs(RY(np.pi / 2) @ RZ(-2 * b) @ RY(-np.pi / 2) - RX(-2 * b)).max() for b in rng.uniform(-3, 3, 10))
check("A6 c=0.5: Eq.(2) mixer == exp(+i b X) == RX(-2b)  (OUR standard mixer)", e < 1e-12 and e2 < 1e-12, f"err={max(e, e2):.1e}")

# ---------------------------------------------------------------- B. modified (rounded) mixer
e = 0.0
for c in rng.uniform(0.01, 0.99, 25):
    th = theta_of(c)
    for b in rng.uniform(-3, 3, 5):
        e = max(e, np.abs(expm(-1j * b * h_mod(c)) - RY(-th) @ RZ(-2 * b) @ RY(th)).max())
check("B1 RY(-th) RZ(-2b) RY(th) == exp(-i b H'), H' = Eq.(2) with off-diagonals * -1 = +sin(th) X - cos(th) Z", e < 1e-12, f"err={e:.1e}")
e = max(np.abs(RY(-np.pi / 2) @ RZ(-2 * b) @ RY(np.pi / 2) - RX(2 * b)).max() for b in rng.uniform(-3, 3, 10))
check("B2 c=0.5: modified mixer == exp(-i b X) == RX(+2b)  (INVERSE of our standard mixer)", e < 1e-12, f"err={e:.1e}")
print("     variance of H' in the initial state RY(th)|0> (0 means eigenstate):")
for c in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    th = theta_of(c)
    psi = RY(th) @ ket0
    m = np.real(psi.conj() @ h_mod(c) @ psi)
    print(f"       c={c:4.2f}  <H'>={m:+.4f}  var={1 - m * m:.4f}   (Eq.2 mixer: <H>={np.real(psi.conj() @ h_ws(c) @ psi):+.4f})")
eps = 0.25
out_lo = RY(-theta_of(eps)) @ RZ(-np.pi) @ RY(2 * theta_of(eps)) @ ket0
out_hi = RY(-theta_of(1 - eps)) @ RZ(-np.pi) @ RY(2 * theta_of(1 - eps)) @ ket0
check("B3 Eq.(3), eps=0.25, beta=pi/2: c*=eps -> -i|1>", np.abs(out_lo - (-1j) * ket1).max() < 1e-12, f"state={np.round(out_lo, 6)}")
check("B4 Eq.(3), eps=0.25, beta=pi/2: c*=1-eps -> -i|0>", np.abs(out_hi - (-1j) * ket0).max() < 1e-12, f"state={np.round(out_hi, 6)}")
print("     flip probability P(bit flipped) at beta=pi/2 with the modified mixer, vs eps:")
for ep in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    o = RY(-theta_of(ep)) @ RZ(-np.pi) @ RY(2 * theta_of(ep)) @ ket0
    print(f"       eps={ep:4.2f}  P(|1>)={abs(o[1]) ** 2:.4f}")

# ---------------------------------------------------------------- C. actual Qiskit source
src = open("/private/tmp/claude-501/-Users-t/7c811c50-f516-4e09-a51a-78bd3f6d1c7e/scratchpad/ws/wsqaoa_main.py").read()
tree = ast.parse(src)
node = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WarmStartQAOAFactory"][0]
ns = {"np": np, "QuantumCircuit": QuantumCircuit, "Parameter": Parameter, "QiskitOptimizationError": ValueError}
exec("from __future__ import annotations\n" + ast.get_source_segment(src, node), ns)
Factory = ns["WarmStartQAOAFactory"]
fac = Factory(0.25)
iv = fac.create_initial_variables(np.array([0.0, 1.0, 0.4, 0.1, 0.97]))
check("C1 Qiskit create_initial_variables clips to [eps, 1-eps]", np.allclose(iv, [0.25, 0.75, 0.4, 0.25, 0.75]), f"-> {iv}")
e_paper = e_mod = e_init = 0.0
for c in rng.uniform(0.01, 0.99, 10):
    f0 = Factory(0.0)
    th = theta_of(c)
    init = Operator(f0.create_initial_state([c])).data @ ket0
    e_init = max(e_init, np.abs(init - RY(th) @ ket0).max())
    mix = f0.create_mixer([c])
    for b in rng.uniform(-3, 3, 4):
        u = Operator(mix.assign_parameters({mix.parameters[0]: b})).data
        e_paper = max(e_paper, np.abs(u - expm(-1j * b * h_ws(c))).max())
        e_mod = max(e_mod, np.abs(u - expm(-1j * b * h_mod(c))).max())
check("C2 Qiskit create_initial_state == RY(2 arcsin sqrt(c))|0>", e_init < 1e-12, f"err={e_init:.1e}")
check("C3 Qiskit create_mixer == exp(-i beta H_ws) of paper Eq.(2) (eigenstate mixer)", e_paper < 1e-12, f"err={e_paper:.1e}")
check("C4 Qiskit create_mixer is NOT the modified Sec.2.3 mixer", e_mod > 1e-2, f"distance={e_mod:.3f}")

# ---------------------------------------------------------------- D. Qiskit QAOAAnsatz conventions
from qiskit.circuit.library import QAOAAnsatz

n = 3
Jm = np.triu(rng.normal(size=(n, n)), 1)
hv = rng.normal(size=n)


def pauli_label(n, idx_to_char):  # qiskit label: leftmost char = highest qubit
    return "".join(idx_to_char.get(q, "I") for q in reversed(range(n)))


terms = [(pauli_label(n, {i: "Z", j: "Z"}), Jm[i, j]) for i in range(n) for j in range(i + 1, n)]
terms += [(pauli_label(n, {i: "Z"}), hv[i]) for i in range(n)]
Hc = SparsePauliOp.from_list(terms)
Hc_mat = Hc.to_matrix()
Xsum = SparsePauliOp.from_list([(pauli_label(n, {i: "X"}), 1.0) for i in range(n)]).to_matrix()
g, b = 0.37, 0.61
ans = QAOAAnsatz(Hc, reps=1)
pmap = {p: (b if p.name.startswith("β") else g) for p in ans.parameters}
sv = Statevector(ans.assign_parameters(pmap)).data
plus = np.ones(2 ** n) / np.sqrt(2 ** n)
ref_minus = expm(-1j * b * Xsum) @ expm(-1j * g * Hc_mat) @ plus
ref_plus = expm(+1j * b * Xsum) @ expm(-1j * g * Hc_mat) @ plus
ov_m, ov_p = abs(np.vdot(ref_minus, sv)), abs(np.vdot(ref_plus, sv))
check("D1 Qiskit DEFAULT QAOAAnsatz == exp(-i beta sum X) exp(-i gamma H_C)|+> (mixer +sum X, i.e. RX(+2 beta))", abs(ov_m - 1) < 1e-9, f"|<ref|sv>|={ov_m:.6f}; overlap with exp(+i beta sum X) version={ov_p:.6f}")

cs = np.array([0.25, 0.75, 0.6])
f0 = Factory(0.0)
ans_ws = QAOAAnsatz(Hc, reps=1, initial_state=f0.create_initial_state(list(cs)), mixer_operator=f0.create_mixer(list(cs)))
pmap = {p: (b if p.name.startswith("β") else g) for p in ans_ws.parameters}
sv = Statevector(ans_ws.assign_parameters(pmap)).data


def kron_all(ops):  # ops[i] acts on qubit i; qiskit little-endian => qubit 0 is the LAST tensor factor
    out = np.array([[1]], dtype=complex)
    for o in reversed(ops):
        out = np.kron(out, o)
    return out


init = kron_all([RY(theta_of(c)) for c in cs])[:, 0]
mix = kron_all([expm(-1j * b * h_ws(c)) for c in cs])
ref = mix @ expm(-1j * g * Hc_mat) @ init
check("D2 Qiskit QAOAAnsatz(initial_state, circuit mixer) == exp(-i beta H_ws) exp(-i gamma H_C) |phi*>  (paper layer)", abs(abs(np.vdot(ref, sv)) - 1) < 1e-9, f"|<ref|sv>|={abs(np.vdot(ref, sv)):.6f}")

# ---------------------------------------------------------------- E. OUR convention, gate level
def bits(n):
    return np.array(list(itertools.product([0, 1], repeat=n)))  # row k: x_0..x_{n-1}, x_0 most significant (our own order)


def energy_table(J, h):
    n = len(h)
    xs = bits(n)
    zs = 1 - 2 * xs
    return xs, np.einsum("ki,ij,kj->k", zs, J, zs) + zs @ h


def our_circuit(J, h, cs, layers, mixer):
    """layers: list of (gamma, beta). mixer in {'std','ws','mod'}. Returns probabilities indexed like bits(n)."""
    n = len(h)
    qc = QuantumCircuit(n)
    th = [theta_of(c) for c in cs]
    for i in range(n):
        qc.ry(th[i], i)
    for gam, bet in layers:
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    qc.rzz(2 * gam * J[i, j], i, j)
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * gam * h[i], i)
        for i in range(n):
            if mixer == "std":
                qc.rx(-2 * bet, i)
            elif mixer == "ws":      # paper Eq.(2): operator RY(th) RZ(-2b) RY(-th)
                qc.ry(-th[i], i); qc.rz(-2 * bet, i); qc.ry(th[i], i)
            elif mixer == "mod":     # paper Sec. 2.3: operator RY(-th) RZ(-2b) RY(th)
                qc.ry(th[i], i); qc.rz(-2 * bet, i); qc.ry(-th[i], i)
    p = Statevector(qc).probabilities()  # index = sum_i x_i 2^i (little endian)
    xs = bits(n)
    idx = (xs * (2 ** np.arange(n))).sum(axis=1)
    return p[idx]


n = 5
J = np.triu(rng.normal(size=(n, n)), 1)
h = rng.normal(size=n)
xs, E = energy_table(J, h)
xs0, E0 = energy_table(J, np.zeros(n))

# cross-check gate-level cost unitary against exact expm in OUR convention (std mixer, |+> start)
p_gate = our_circuit(J, h, [0.5] * n, [(0.3, 0.2), (0.7, 0.4)], "std")
Zs = [kron_all([Z if q == i else I2 for q in range(n)]) for i in range(n)]
Xs = [kron_all([X if q == i else I2 for q in range(n)]) for i in range(n)]
Hfull = sum(J[i, j] * Zs[i] @ Zs[j] for i in range(n) for j in range(i + 1, n)) + sum(h[i] * Zs[i] for i in range(n))
psi = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
for gam, bet in [(0.3, 0.2), (0.7, 0.4)]:
    psi = expm(1j * bet * sum(Xs)) @ expm(-1j * gam * Hfull) @ psi
idx = (bits(n) * (2 ** np.arange(n))).sum(axis=1)
check("E0 our gate recipe (rzz(2gJ), rz(2gh), rx(-2b)) == exp(+i b sum X) exp(-i g H) exactly; c=0.5 init == |+>", np.abs(p_gate - np.abs(psi[idx]) ** 2).max() < 1e-10)
p_ws_half = our_circuit(J, h, [0.5] * n, [(0.3, 0.2), (0.7, 0.4)], "ws")
check("E1 eps=0.5 (all c=0.5): WS circuit with Eq.(2) mixer == our standard QAOA at the same (gamma, beta)", np.abs(p_ws_half - p_gate).max() < 1e-10)
p_mod_half = our_circuit(J, h, [0.5] * n, [(0.3, 0.2), (0.7, 0.4)], "mod")
p_std_neg = our_circuit(J, h, [0.5] * n, [(0.3, -0.2), (0.7, -0.4)], "std")
check("E2 eps=0.5: MODIFIED mixer != our standard QAOA at same (gamma,beta); == standard QAOA with beta -> -beta", np.abs(p_mod_half - p_gate).max() > 1e-3 and np.abs(p_mod_half - p_std_neg).max() < 1e-10, f"diff_same={np.abs(p_mod_half - p_gate).max():.3f}")

x_round = np.array([0, 1, 1, 0, 1])
eps = 0.25
cs = np.where(x_round == 1, 1 - eps, eps)
k_round = int(np.where((xs == x_round).all(axis=1))[0][0])
k_comp = int(np.where((xs == 1 - x_round).all(axis=1))[0][0])
p_init = our_circuit(J, h, cs, [], "ws")
ok = all(np.abs(our_circuit(J, h, cs, [(0.0, bb)], "ws") - p_init).max() < 1e-12 for bb in [0.3, np.pi / 2, 1.9])
check("E3 Eq.(2) mixer, gamma=0: output distribution == initial product distribution for every beta (eigenstate)", ok, f"P(rounded)={p_init[k_round]:.5f} vs (1-eps)^n={(1 - eps) ** n:.5f}")
p = our_circuit(J, h, cs, [(0.0, np.pi / 2)], "mod")
check("E4 modified mixer, eps=0.25, (beta,gamma)=(pi/2,0): P(COMPLEMENT of rounded bitstring)=1", abs(p[k_comp] - 1) < 1e-10, f"P(rounded)={p[k_round]:.2e}, P(complement)={p[k_comp]:.6f}")
print(f"     with fields h!=0: E(rounded)={E[k_round]:+.4f}  E(complement)={E[k_comp]:+.4f}  (differ)")
print(f"     with h=0 (Z2 symmetric, MaxCut-like): E(rounded)={E0[k_round]:+.4f}  E(complement)={E0[k_comp]:+.4f}  (equal)")
check("E5 complement has the same energy only when all h_i=0", abs(E0[k_round] - E0[k_comp]) < 1e-12 and abs(E[k_round] - E[k_comp]) > 1e-6)
p = our_circuit(J, h, 1 - cs, [(0.0, np.pi / 2)], "mod")
check("E6 [own derivation] modified mixer built from the COMPLEMENTED warm start (c -> 1-c): P(rounded bitstring)=1 at (pi/2,0)", abs(p[k_round] - 1) < 1e-10, f"P(rounded)={p[k_round]:.6f}")

# Z-conjugation equivalence: modified mixer + init RY(th)  ==  Eq.(2) mixer + init RY(-th), same Z-basis probabilities
def our_circuit_neg_init(J, h, cs, layers):
    n = len(h)
    qc = QuantumCircuit(n)
    th = [theta_of(c) for c in cs]
    for i in range(n):
        qc.ry(-th[i], i)
    for gam, bet in layers:
        for i in range(n):
            for j in range(i + 1, n):
                qc.rzz(2 * gam * J[i, j], i, j)
        for i in range(n):
            qc.rz(2 * gam * h[i], i)
        for i in range(n):
            qc.ry(-th[i], i); qc.rz(-2 * bet, i); qc.ry(th[i], i)
    p = Statevector(qc).probabilities()
    return p[(bits(n) * (2 ** np.arange(n))).sum(axis=1)]


lay = [(0.41, 0.83), (0.9, 0.27)]
d = np.abs(our_circuit(J, h, cs, lay, "mod") - our_circuit_neg_init(J, h, cs, lay)).max()
check("E7 [own derivation] modified mixer + init RY(+th) has identical probabilities to Eq.(2) mixer + init RY(-th)", d < 1e-10, f"err={d:.1e}")

# symmetry E(gamma, beta) = E(-gamma, pi - beta) for all three mixers (all matrices real)
ok = True
for mixer in ["std", "ws", "mod"]:
    for _ in range(4):
        gam, bet = rng.uniform(0, 2), rng.uniform(0, np.pi)
        e1 = our_circuit(J, h, cs, [(gam, bet)], mixer) @ E
        e2 = our_circuit(J, h, cs, [(-gam, np.pi - bet)], mixer) @ E
        e3 = our_circuit(J, h, cs, [(gam, bet + np.pi)], mixer) @ E
        ok &= abs(e1 - e2) < 1e-9 and abs(e1 - e3) < 1e-9
check("E8 [own derivation] <E>(gamma,beta) == <E>(-gamma, pi-beta) and beta has period pi, for std/ws/mod mixers", ok)

# p=1 landscape on a Z2-symmetric instance: modified mixer attains E(rounded) at (pi/2, 0)
e_pt = our_circuit(J, np.zeros(n), cs, [(0.0, np.pi / 2)], "mod") @ E0
e_ws_best = min(our_circuit(J, np.zeros(n), cs, [(gam, bet)], "ws") @ E0 for gam in np.linspace(0, 3, 31) for bet in np.linspace(0, np.pi, 33))
e_mod_best = min(our_circuit(J, np.zeros(n), cs, [(gam, bet)], "mod") @ E0 for gam in np.linspace(0, 3, 31) for bet in np.linspace(0, np.pi, 33))
print(f"     h=0 instance, eps=0.25, p=1: E(rounded)={E0[k_round]:+.4f}, Emin={E0.min():+.4f}; modified mixer at (pi/2,0) -> {e_pt:+.4f}; grid-best modified={e_mod_best:+.4f}; grid-best Eq.(2) mixer={e_ws_best:+.4f}")
check("E9 modified mixer at (beta,gamma)=(pi/2,0) reproduces E(rounded) on the Z2-symmetric instance", abs(e_pt - E0[k_round]) < 1e-10)

print(f"\n{sum(results)}/{len(results)} checks passed")
