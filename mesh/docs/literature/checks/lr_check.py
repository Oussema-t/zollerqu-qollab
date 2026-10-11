# Sanity check of the LR-QAOA ramp direction in OUR convention (literature verification only).
# U_C = exp(-i gamma H), U_M = exp(+i beta sum X), start |+>^n, minimise H.
import numpy as np, itertools
rng = np.random.default_rng(0)
def run(n, p, dg, db, J, h, sign_g=+1, sign_b=+1, reverse=False):
    N = 2**n
    bits = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1)
    z = 1 - 2*bits
    E = np.einsum('ki,ij,kj->k', z, np.triu(J, 1), z) + z @ h
    scale = np.max(np.abs(J[np.triu_indices(n, 1)]))
    En = E/scale
    psi = np.ones(N, complex)/np.sqrt(N)
    for i in range(p):
        beta = (1 - i/p)*db; gamma = (i+1)/p*dg
        if reverse:
            beta, gamma = (i+1)/p*db, (1 - i/p)*dg
        psi = np.exp(-1j*sign_g*gamma*En)*psi
        # exp(+i beta X) on every qubit = cos(b) I + i sin(b) X
        t = psi.reshape([2]*n)
        b = sign_b*beta
        for q in range(n):
            t = np.cos(b)*t + 1j*np.sin(b)*np.flip(t, axis=q)
        psi = t.reshape(N)
    pr = np.abs(psi)**2
    gs = np.isclose(E, E.min())
    hs = np.isclose(E, E.max())
    return pr[gs].sum(), pr[hs].sum(), (pr*E).sum(), E.min(), E.max(), gs.sum()/N
n = 10
res = {}
for trial in range(5):
    J = rng.normal(size=(n, n)); J = np.triu(J, 1); h = rng.normal(size=n)
    for name, kw in [('ours(+g,+b,g up,b down)', {}), ('gamma sign flipped', dict(sign_g=-1)),
                     ('beta sign flipped', dict(sign_b=-1)), ('both flipped', dict(sign_g=-1, sign_b=-1)),
                     ('ramps reversed', dict(reverse=True))]:
        pg, ph, e, emin, emax, base = run(n, 30, 0.6, 0.3, J, h, **kw)
        res.setdefault(name, []).append((pg, ph, e, emin, emax, base))
for k, v in res.items():
    v = np.array(v)
    print(f"{k:28s} P(ground)={v[:,0].mean():.4f}  P(highest)={v[:,1].mean():.4f}  <E>={v[:,2].mean():+.3f}  Emin={v[:,3].mean():+.3f} Emax={v[:,4].mean():+.3f} uniform={v[:,5].mean():.5f}")
