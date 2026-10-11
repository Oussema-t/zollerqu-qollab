"""Lens items 3 (sign convention, adiabatic), 6 (CVaR), 7 (lr_angles, SPSA) - standalone, no notebook needed."""
import numpy as np
from extract import extract
ns = extract()
rng = np.random.default_rng(31415)

def energies(J, h, n):
    k = np.arange(2**n)
    z = 1 - 2 * ((k[:, None] >> np.arange(n)[None, :]) & 1)
    return np.einsum("ki,ij,kj->k", z, np.triu(J, 1), z) + z @ h

print("== item 7: lr_angles ==")
for p in (1, 2, 3, 5, 8):
    dg, db = 0.37, 0.91
    ang = ns["lr_angles"](p, dg, db)
    exp_g = [k / p * dg for k in range(1, p + 1)]
    exp_b = [(1 - (k - 1) / p) * db for k in range(1, p + 1)]
    # LR-QAOA paper indexing i = 0..p-1: beta_i = (1 - i/p) dbeta, gamma_i = ((i+1)/p) dgamma
    exp_g2 = [(i + 1) / p * dg for i in range(p)]; exp_b2 = [(1 - i / p) * db for i in range(p)]
    err = max(np.abs(ang[:p] - exp_g).max(), np.abs(ang[p:] - exp_b).max(), np.abs(ang[:p] - exp_g2).max(), np.abs(ang[p:] - exp_b2).max())
    print(f"p={p}: gammas={np.round(ang[:p], 4).tolist()} betas={np.round(ang[p:], 4).tolist()} max err vs formula {err:.1e}")

print("== item 3: adiabatic sign test on random Ising instances (independent energies) ==")
for n, with_fields in ((5, True), (6, False), (7, True), (8, True), (8, False)):
    res = []
    for trial in range(4):
        J = np.triu(rng.normal(size=(n, n)), 1); h = rng.normal(size=n) if with_fields else np.zeros(n)
        scale = max(np.abs(J).max(), np.abs(h).max())
        E = energies(J, h, n); Es = E / scale
        gs = np.flatnonzero(E <= E.min() + 1e-12); top = np.flatnonzero(E >= E.max() - 1e-12)
        row = []
        for p in (1, 4, 20, 100, 400):
            ang = ns["lr_angles"](p, 0.3, 0.3)
            pr = ns["qaoa_probabilities"](Es, ang[:p], ang[p:], n)
            pr_flip_beta = ns["qaoa_probabilities"](Es, ang[:p], -ang[p:], n)
            pr_flip_gamma = ns["qaoa_probabilities"](Es, -ang[:p], ang[p:], n)
            pr_flip_both = ns["qaoa_probabilities"](Es, -ang[:p], -ang[p:], n)
            row.append((p, pr[gs].sum(), pr @ E, pr_flip_beta[top].sum(), pr_flip_beta @ E,
                        np.abs(pr_flip_beta - pr_flip_gamma).max(), np.abs(pr_flip_both - pr).max()))
        res.append((E.min(), E.mean(), E.max(), row))
    for emin, emean, emax, row in res[:2]:
        print(f"n={n} fields={with_fields}: Emin={emin:.3f} Emean={emean:.3f} Emax={emax:.3f}")
        for p, pg, me, ptop, me2, sym1, sym2 in row:
            print(f"   p={p:4d}: P(ground)={pg:.4f} <E>={me:.3f} | beta->-beta: P(max)={ptop:.4f} <E>={me2:.3f} | "
                  f"|P(g,-b)-P(-g,b)|={sym1:.1e} |P(-g,-b)-P(g,b)|={sym2:.1e}")
    allp400 = [r[3][-1][1] for r in res]; allmax400 = [r[3][-1][3] for r in res]
    print(f"   n={n}: P(ground) at p=400 over 4 instances: {np.round(allp400, 4).tolist()}; flipped-beta P(max): {np.round(allmax400, 4).tolist()}")

print("== item 6: cvar_of_samples ==")
cv = ns["cvar_of_samples"]
print("hand: [5,1,4,2,3], 0.4 -> mean(1,2) =", cv([5, 1, 4, 2, 3], 0.4), "; 0.5 -> ceil(2.5)=3 -> mean(1,2,3) =", cv([5, 1, 4, 2, 3], 0.5),
      "; 1.0 ->", cv([5, 1, 4, 2, 3], 1.0), "; 0.2 ->", cv([5, 1, 4, 2, 3], 0.2), "; 0.1 -> ceil(0.5)=1 ->", cv([5, 1, 4, 2, 3], 0.1))
for K in (10, 100, 1000, 1024, 4096, 7, 13):
    s = rng.normal(size=K)
    for a in (0.1, 0.25, 0.3, 1 / 3, 0.7, 1.0):
        m = int(np.ceil(a * K - 1e-12))
        ref = np.sort(s)[:m].mean()
        assert abs(cv(s, a) - ref) < 1e-12, (K, a)
print("cvar matches mean of lowest ceil(alpha K) for K in 10,100,1000,1024,4096,7,13 and alpha in 0.1,.25,.3,1/3,.7,1")
# floating-point edge: 0.1 * 30 = 3.0000000000000004 -> should be 3, 0.07*100 = 7.000000000000001 -> 7
for a, K in ((0.1, 30), (0.07, 100), (0.3, 10), (0.6, 5), (0.1, 1000), (0.2, 15)):
    s = np.arange(K, dtype=float)
    used = round(2 * cv(s, a) + 1)        # mean of 0..m-1 = (m-1)/2
    print(f"   alpha={a}, K={K}: alpha*K={a*K!r} -> uses {used} samples (exact ceil = {int(np.ceil(round(a * K, 9)))})")

print("== item 7: SPSA vs Spall's update rule ==")
class Rec:
    def __init__(self, f): self.f, self.pts, self.n_evals = f, [], 0
    def __call__(self, x): self.pts.append(np.array(x, float)); self.n_evals += 1; return self.f(np.asarray(x))
ns["TASK04"] = dict(spsa=dict(alpha=0.602, gamma=0.101, stability_fraction=0.1, calibration_pairs=5))
f = lambda x: float(np.sum((x - np.array([0.3, -0.2, 0.5, 0.1])) ** 2 * np.array([1.0, 2.0, 0.5, 3.0])))
start = np.array([1.0, 1.0, -1.0, 0.5]); budget = 200; tuned = dict(perturbation=0.07, first_step=0.05)
rec = Rec(f)
final, msg = ns["spsa_minimize"](rec, start, budget, 123, tuned)
print(msg, "| evaluations used:", rec.n_evals, "of budget", budget)
pts = np.array(rec.pts)
cal = 5; iters = (budget - 2 * cal) // 2; A = 0.1 * iters
# calibration pairs: all centred on the start with size c
centres = (pts[0::2] + pts[1::2]) / 2; halfdiff = (pts[0::2] - pts[1::2]) / 2
assert np.allclose(centres[:cal], start), "calibration pairs must be centred on the start"
assert np.allclose(np.abs(halfdiff[:cal]), 0.07), "calibration perturbation size"
# independent re-implementation of Spall's recursion using the recorded perturbation directions
grads0 = [(f(start + 0.07 * np.sign(d)) - f(start - 0.07 * np.sign(d))) / (2 * 0.07) * np.sign(d) for d in halfdiff[:cal]]
a = 0.05 * (A + 1) ** 0.602 / np.mean([np.abs(g).mean() for g in grads0])
theta = start.copy(); worst_c = worst_theta = 0
for k in range(iters):
    ck = 0.07 / (k + 1) ** 0.101
    ak = a / (A + k + 1) ** 0.602
    d = np.sign(halfdiff[cal + k])
    worst_theta = max(worst_theta, np.abs(centres[cal + k] - theta).max())
    worst_c = max(worst_c, np.abs(np.abs(halfdiff[cal + k]) - ck).max())
    assert set(np.abs(d).tolist()) == {1.0}
    ghat = (f(theta + ck * d) - f(theta - ck * d)) / (2 * ck) * (1.0 / d)
    theta = theta - ak * ghat
print(f"max |centre_k - theta_k(Spall)| = {worst_theta:.2e}; max |c_k - c/(k+1)^0.101| = {worst_c:.2e}; |final - Spall final| = {np.abs(final - theta).max():.2e}")
print(f"first move size: a_0*|ghat_0| mean abs = {np.abs(centres[cal + 1] - centres[cal]).mean():.4f} (first_step = 0.05)")
print(f"f(start) = {f(start):.4f} -> f(final) = {f(final):.5f} (minimum 0)")
dirs = np.sign(halfdiff)
print("perturbation components: mean =", np.round(dirs.mean(axis=0), 3).tolist(), "(expected ~0), all +-1:", bool(np.all(np.abs(dirs) == 1)))
