import math, numpy as np
from scipy.linalg import expm
rng = np.random.default_rng(20261010)

# (a) floating point of ceil(alpha*K)
for a, K in [(0.1, 1000), (0.25, 1000), (0.05, 1000), (0.01, 1000), (0.07, 100), (0.1, 8192), (0.1, 1024)]:
    print(f"alpha={a} K={K}: alpha*K={a*K!r} ceil={math.ceil(a*K)} robust={math.ceil(a*K - 1e-9)}")

# estimators
def cvar_paper(E, alpha):                      # Eq.(12) as intended: mean of ceil(alpha K) lowest
    E = np.sort(np.asarray(E, float)); m = math.ceil(alpha*len(E) - 1e-9)
    return E[:m].mean()
def cvar_notebook(probabilities, values, alpha):   # verbatim logic from authors' notebook
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]; vals = np.array(values)[sorted_indices]
    cvar = 0; total_prob = 0
    for i, (p, v) in enumerate(zip(probs, vals)):
        if p >= alpha - total_prob:
            p = alpha - total_prob
        total_prob += p; cvar += p*v
    return cvar/total_prob
def cvar_qiskit(measurements, alpha):          # verbatim logic from qiskit-algorithms
    sm = sorted(measurements, key=lambda x: x[1]); acc = 0.0; c = 0.0
    for probability, value in sm:
        c += value*min(probability, alpha-acc); acc += probability
        if acc >= alpha: break
    return c/alpha

# (c) gate identities with Qiskit definitions
X = np.array([[0,1],[1,0]],complex); Z = np.diag([1,-1]).astype(complex); I2 = np.eye(2)
RX = lambda t: expm(-1j*t*X/2); RZ = lambda t: expm(-1j*t*Z/2)
b, g = 0.37, 0.81
print("RX(-2b)==exp(+i b X):", np.allclose(RX(-2*b), expm(1j*b*X)), "| ==exp(-i b X):", np.allclose(RX(-2*b), expm(-1j*b*X)))
CX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)   # control = first tensor factor
U = CX @ np.kron(I2, RZ(-2*g)) @ CX
ZZ = np.kron(Z,Z)
print("CX.RZ(-2g).CX==exp(+i g ZZ):", np.allclose(U, expm(1j*g*ZZ)), "| ==exp(-i g ZZ):", np.allclose(U, expm(-1j*g*ZZ)))

# (d) QAOA on random Ising, our convention: U_C=exp(-i g H), U_M=exp(+i b sum X), start |+>
def ising_diag(n, J, h):
    idx = np.arange(2**n)
    bits = (idx[:,None] >> np.arange(n)[None,:]) & 1
    z = 1 - 2*bits
    return np.einsum('ki,ij,kj->k', z, np.triu(J,1), z) + z@h
def qaoa_probs(E, n, gammas, betas, mixer_sign=+1, cost_sign=-1):
    psi = np.full(2**n, 2**(-n/2), complex)
    for g_, b_ in zip(gammas, betas):
        psi = np.exp(cost_sign*1j*g_*E)*psi
        u = expm(mixer_sign*1j*b_*X)
        psi = psi.reshape([2]*n)
        for q in range(n):
            psi = np.moveaxis(np.tensordot(u, psi, axes=([1],[q])), 0, q)
        psi = psi.reshape(-1)
    return np.abs(psi)**2
n = 10
J = rng.normal(size=(n,n)); h = rng.normal(size=n)
E = ising_diag(n, J, h)
gam, bet = [0.21, 0.35], [0.42, 0.18]
p_ours  = qaoa_probs(E, n, gam, bet, mixer_sign=+1, cost_sign=-1)   # ours
p_eq67  = qaoa_probs(E, n, gam, bet, mixer_sign=-1, cost_sign=-1)   # paper Eq.(6),(7) at same numbers
p_eq67n = qaoa_probs(E, n, gam, [-x for x in bet], mixer_sign=-1, cost_sign=-1)  # paper Eq.(6),(7) with beta -> -beta
p_appA  = qaoa_probs(E, n, gam, bet, mixer_sign=+1, cost_sign=+1)   # paper App. A gates literally: RX(-2b), CX RZ(-2g) CX
p_conj  = qaoa_probs(E, n, gam, bet, mixer_sign=-1, cost_sign=+1)   # both signs flipped vs ours
print("sum p:", p_ours.sum())
print("max|ours - paperEq(6,7) same (g,b)|   =", np.abs(p_ours-p_eq67).max())
print("max|ours - paperEq(6,7) with b->-b|   =", np.abs(p_ours-p_eq67n).max())
print("max|ours - both-signs-flipped|        =", np.abs(p_ours-p_conj).max())
print("max|paperEq(6,7) - paperAppA literal| =", np.abs(p_eq67-p_appA).max())

# (b) estimator equivalence at alpha=0.1, K=1000 and at a non-integer alpha*K
for alpha, K in [(0.1,1000),(0.25,1000),(0.1,1024),(0.01,1000)]:
    s = rng.choice(2**n, size=K, p=p_ours); Es = E[s]
    vals, cnt = np.unique(s, return_counts=True)
    probs = cnt/K; v = E[vals]
    print(f"alpha={alpha} K={K}: Eq12-mean-of-lowest-ceil={cvar_paper(Es,alpha):.10f}  notebook={cvar_notebook(probs,v,alpha):.10f}  qiskit={cvar_qiskit(list(zip(probs,v)),alpha):.10f}  mean={Es.mean():.6f} min={Es.min():.6f}")

# (e) Monte Carlo spread of estimators for this fixed QAOA state
exact_mean = float(p_ours@E)
order = np.argsort(E); cp = np.cumsum(p_ours[order])
def exact_cvar(alpha):
    return cvar_qiskit(list(zip(p_ours, E)), alpha)
R = 4000
for K in [1000, 10000]:
    S = rng.choice(2**n, size=(R,K), p=p_ours); ES = np.sort(E[S], axis=1)
    line = f"K={K}: SE(mean)={ES.mean(1).std():.4f}"
    for alpha in [0.25, 0.1, 0.05, 0.01]:
        m = math.ceil(alpha*K-1e-9); est = ES[:,:m].mean(1)
        line += f" | a={alpha}: SE={est.std():.4f} bias={est.mean()-exact_cvar(alpha):+.4f}"
    print(line)
print("exact mean", exact_mean, "exact CVaR0.1", exact_cvar(0.1), "E_min", E.min(), "P(opt)", p_ours[np.argmin(E)])
