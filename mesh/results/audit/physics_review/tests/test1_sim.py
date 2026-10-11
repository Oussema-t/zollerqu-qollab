"""Lens item 1: simulator vs independent brute force (explicit Kronecker products / scipy expm)."""
import numpy as np, functools, itertools
from scipy.linalg import expm
from extract import extract
ns = extract()
rng = np.random.default_rng(20261010)
I2 = np.eye(2, dtype=complex); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0 + 0j, -1.0])

def kron_all(ops):
    """ops[q] acts on qubit q; qubit 0 least significant => kron(ops[n-1], ..., ops[0])."""
    return functools.reduce(np.kron, ops[::-1])

def op_on(q, op, n):
    return kron_all([op if k == q else I2 for k in range(n)])

def rand_state(n):
    s = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return s / np.linalg.norm(s)

def RY(t): return expm(-0.5j * t * Y)
def RZ(t): return expm(-0.5j * t * Z)
def RX(t): return expm(-0.5j * t * X)

report = {}
# --- kron_block ordering
err = 0
for n in range(1, 9):
    gates = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(n)]
    err = max(err, np.abs(ns["kron_block"](gates) - kron_all(gates)).max())
report["kron_block vs explicit kron (n=1..8, random 2x2)"] = err

# --- bit ordering sanity: a gate only on qubit q flips bit q of a basis state
for n in range(1, 9):
    for q in range(n):
        gates = [X if k == q else I2 for k in range(n)]
        for per_qubit in (False, True):
            for k in (0, 2**n - 1, int(rng.integers(2**n))):
                e = np.zeros(2**n, complex); e[k] = 1
                out = ns["apply_qubit_gates"](e, gates, per_qubit=per_qubit)
                assert np.argmax(np.abs(out)) == k ^ (1 << q) and abs(abs(out).max() - 1) < 1e-14, (n, q, k, per_qubit)
report["bit ordering (X on qubit q flips bit q), both paths, n=1..8"] = "ok"

# --- apply_qubit_gates (both paths) vs explicit Kronecker matrix, random NON-unitary gates (catches transposition errors)
err_b = err_q = 0
for n in range(1, 9):
    for trial in range(5):
        gates = np.array([rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(n)])
        s = rand_state(n)
        ref = kron_all(list(gates)) @ s
        err_b = max(err_b, np.abs(ns["apply_qubit_gates"](s, gates) - ref).max())
        err_q = max(err_q, np.abs(ns["apply_qubit_gates"](s, gates, per_qubit=True) - ref).max())
report["apply_qubit_gates block path vs kron (n=1..8)"] = err_b
report["apply_qubit_gates per-qubit path vs kron (n=1..8)"] = err_q

# --- n = 14, 15 (the per-qubit path the cell uses for n > 13) against an independent tensordot implementation
def tensor_apply(state, gates):
    n = len(gates)
    t = state.reshape([2] * n)                      # axis 0 = qubit n-1 ... axis n-1 = qubit 0
    for q in range(n):
        ax = n - 1 - q
        t = np.moveaxis(np.tensordot(gates[q], t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)
err14 = 0
for n in (13, 14, 15):
    gates = np.array([rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(n)]) / 1.5
    s = rand_state(n)
    ref = tensor_apply(s, gates)
    out = ns["apply_qubit_gates"](s, gates)        # default path: n<=13 blocks, n>13 per qubit
    err14 = max(err14, np.abs(out - ref).max() / np.abs(ref).max())
report["apply_qubit_gates default path n=13,14,15 vs tensordot (relative)"] = err14

# --- mixer_gates vs expm and vs the RY RZ RY products
err_e = err_a = err_b2 = err_std = 0
for trial in range(200):
    beta, theta = rng.uniform(-4, 4), rng.uniform(-4, 4)
    g = ns["mixer_gates"](beta, np.array([theta]))[0]
    err_e = max(err_e, np.abs(g - expm(1j * beta * (np.sin(theta) * X + np.cos(theta) * Z))).max())
    err_a = max(err_a, np.abs(g - RY(theta) @ RZ(-2 * beta) @ RY(-theta)).max())
    gB = ns["mixer_gates"](beta, np.array([-theta]))[0]
    err_b2 = max(err_b2, np.abs(gB - RY(-theta) @ RZ(-2 * beta) @ RY(theta)).max())
    err_b2 = max(err_b2, np.abs(gB - expm(1j * beta * (np.cos(theta) * Z - np.sin(theta) * X))).max())
    gS = ns["mixer_gates"](beta, np.array([np.pi / 2]))[0]
    err_std = max(err_std, np.abs(gS - RX(-2 * beta)).max(), np.abs(gS - expm(1j * beta * X)).max())
report["mixer_gates vs expm(+i beta (sin t X + cos t Z))"] = err_e
report["mixer A gate = RY(t) RZ(-2b) RY(-t)"] = err_a
report["mixer B gate = RY(-t) RZ(-2b) RY(t) = expm(+i b (cos t Z - sin t X))"] = err_b2
report["standard gate = RX(-2 beta) = expm(+i beta X)"] = err_std

# Egger Eq (2): H_ws = [[2c-1, -2 sqrt(c(1-c))],[.., 1-2c]]; mixer A = exp(-i beta H_ws); start is ground state (eigenvalue -1)
err_eq2 = err_gs = 0
for c in rng.uniform(0.01, 0.99, 50):
    theta = 2 * np.arcsin(np.sqrt(c)); beta = rng.uniform(-3, 3)
    H = np.array([[2 * c - 1, -2 * np.sqrt(c * (1 - c))], [-2 * np.sqrt(c * (1 - c)), 1 - 2 * c]], complex)
    err_eq2 = max(err_eq2, np.abs(ns["mixer_gates"](beta, np.array([theta]))[0] - expm(-1j * beta * H)).max())
    start = RY(theta) @ np.array([1, 0], complex)
    w, v = np.linalg.eigh(H)
    err_gs = max(err_gs, abs(w[0] + 1), 1 - abs(np.vdot(v[:, 0], start)))
report["mixer A = exp(-i beta H_ws(Eq.2))"] = err_eq2
report["RY(theta)|0> is ground state of Eq.(2) with eigenvalue -1"] = err_gs

# --- apply_uniform_x_mixer vs expm(i beta sum X)
err_u = 0
for n in range(1, 9):
    HX = sum(op_on(q, X, n) for q in range(n))
    for trial in range(3):
        beta = rng.uniform(-3, 3); s = rand_state(n)
        err_u = max(err_u, np.abs(ns["apply_uniform_x_mixer"](s, beta, n) - expm(1j * beta * HX) @ s).max())
report["apply_uniform_x_mixer vs expm(+i beta sum X) (n=1..8)"] = err_u
for n in (13, 14):   # block path at 13 vs per-qubit path at 14, against tensordot with RX(-2 beta)
    beta = 0.731; s = rand_state(n)
    ref = tensor_apply(s, np.array([RX(-2 * beta)] * n))
    report[f"apply_uniform_x_mixer n={n} vs tensordot RX(-2 beta)"] = np.abs(ns["apply_uniform_x_mixer"](s, beta, n) - ref).max()

# --- full qaoa_statevector vs expm of full Hamiltonians; H_C built from Kronecker products of Z (independent bit ordering)
def ising_ops(J, h, n):
    H = np.zeros((2**n, 2**n), complex)
    for i in range(n):
        H += h[i] * op_on(i, Z, n)
        for j in range(i + 1, n):
            H += J[i, j] * op_on(i, Z, n) @ op_on(j, Z, n)
    return H
errs = {"standard": 0, "mixer A": 0, "mixer B": 0}
phase_err = 0
for n in range(1, 9):
    for with_fields in (False, True):
        for p in (1, 2, 3):
            J = np.triu(rng.normal(size=(n, n)), 1); h = rng.normal(size=n) if with_fields else np.zeros(n)
            HC = ising_ops(J, h, n)
            E = np.real(np.diag(HC)).copy()
            # independent check of the convention x_i = (k >> i) & 1, z = 1 - 2x
            for k in range(2**n):
                z = 1 - 2 * ((k >> np.arange(n)) & 1)
                assert abs(E[k] - (z @ J @ z + h @ z)) < 1e-10
            gam, bet = rng.uniform(-2, 2, p), rng.uniform(-2, 2, p)
            thetas = rng.uniform(0.2, 2.9, n)
            HX = sum(op_on(q, X, n) for q in range(n))
            # standard
            psi = np.full(2**n, 2 ** (-n / 2), complex)
            for g, b in zip(gam, bet):
                psi = expm(1j * b * HX) @ (expm(-1j * g * HC) @ psi)
            out = ns["qaoa_statevector"](E, gam, bet, n)
            errs["standard"] = max(errs["standard"], np.abs(out - psi).max())
            # warm starts
            for name, sign in (("mixer A", 1.0), ("mixer B", -1.0)):
                HM = sum(np.sin(sign * thetas[q]) * op_on(q, X, n) + np.cos(sign * thetas[q]) * op_on(q, Z, n) for q in range(n))
                psi = kron_all([RY(thetas[q]) @ np.array([1, 0], complex) for q in range(n)])  # vector kron with same ordering
                for g, b in zip(gam, bet):
                    psi = expm(1j * b * HM) @ (expm(-1j * g * HC) @ psi)
                out = ns["qaoa_statevector"](E, gam, bet, n, thetas=thetas, mixer_axis=sign * thetas)
                errs[name] = max(errs[name], np.abs(out - psi).max())
for k, v in errs.items():
    report[f"qaoa_statevector ({k}) vs expm of full H, n=1..8, p=1..3, random angles incl. negative, with/without fields"] = v

# default mixer_axis=None for warm start => mixer A
n = 5; thetas = rng.uniform(0.2, 2.9, n); E = rng.normal(size=2**n)
a = ns["qaoa_statevector"](E, [0.3], [0.7], n, thetas=thetas)
b = ns["qaoa_statevector"](E, [0.3], [0.7], n, thetas=thetas, mixer_axis=thetas)
report["thetas given, mixer_axis None -> mixer A"] = np.abs(a - b).max()

# split_angles / view_probabilities ordering: [gammas, betas]
g, b = ns["split_angles"](np.array([1, 2, 3, 4, 5, 6.0]))
assert g.tolist() == [1, 2, 3] and b.tolist() == [4, 5, 6]
for k, v in report.items():
    print(f"{k}: {v if isinstance(v, str) else f'{v:.3e}'}")
