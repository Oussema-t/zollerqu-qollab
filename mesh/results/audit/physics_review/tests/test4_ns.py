"""Lens items 4, 5, 6 on the REAL instances (notebook namespace) plus synthetic ones."""
import numpy as np, itertools, sys
from nsload import load_defs
ns = load_defs("small")
T = ns["TASK04"]
print("constants: EXACT_BRUTE_FORCE_MAX_QUBITS", ns["EXACT_BRUTE_FORCE_MAX_QUBITS"], "LOCAL_STATEVECTOR_MAX_QUBITS", ns["LOCAL_STATEVECTOR_MAX_QUBITS"],
      "IONQ_SIMULATOR_MAX_QUBITS", ns.get("IONQ_SIMULATOR_MAX_QUBITS"), "shots", ns["T4_SHOTS"], "layers", ns["T4_LAYERS"], "mu", T["mu"])
print("TARGET_SHARES", ns["TARGET_SHARES"], "DEVICE_SCENARIOS", ns["DEVICE_SCENARIOS"], "JOB_REFERENCE_HOURS", ns["JOB_REFERENCE_HOURS"])
print("RUN", {k: ns["RUN"][k] for k in ("region_counts", "seeds", "qaoa_layers", "shots")})
rng = np.random.default_rng(99)

def my_energies(J, h, const, n):
    k = np.arange(2**n)
    z = 1 - 2 * ((k[:, None] >> np.arange(n)[None, :]) & 1)
    return np.einsum("ki,ij,kj->k", z, J, z) + z @ h + const

insts = {}
for case in ns["BENCHMARK_CASES"]:
    for n in ns["RUN"]["region_counts"]:
        for seed in ns["RUN"]["seeds"]:
            for scen in ns["TARGET_SHARES"]:
                insts[(case, n, seed, scen)] = ns["t4_build_instance"](ns["region_cases"][(case, n, seed)], case, n, seed, scen)
print("instances built:", len(insts))

# ---------- A. model structure and scaling
worst = dict(lower=0, diag=0, scaled=0, escaled=0, e_vs_mine=0)
hzero = {}
for key, inst in insts.items():
    m, n = inst["model"], inst["n"]
    J, h = m["J"], m["h"]
    worst["lower"] = max(worst["lower"], np.abs(np.tril(J, 0)).max())
    worst["scaled"] = max(worst["scaled"], np.abs(m["J_scaled"] * m["scale"] - J).max(), np.abs(m["h_scaled"] * m["scale"] - h).max())
    E = my_energies(J, h, m["const"], n)
    worst["e_vs_mine"] = max(worst["e_vs_mine"], np.abs(E - inst["energies"]).max())
    worst["escaled"] = max(worst["escaled"], np.abs(my_energies(m["J_scaled"], m["h_scaled"], 0.0, n) - inst["energy_scaled"]).max())
    hzero[key[3]] = hzero.get(key[3], []) + [float(np.abs(h).max())]
    assert abs(max(np.abs(m["J_scaled"]).max(), np.abs(m["h_scaled"]).max()) - 1.0) < 1e-12
    assert inst["energies"].min() >= -1e-12
print("A. max |lower-tri/diag of J| =", worst["lower"], "; J_scaled*scale-J, h_scaled*scale-h:", worst["scaled"],
      "; energies vs my enumeration:", worst["e_vs_mine"], "; energy_scaled vs my enumeration of J_scaled,h_scaled:", worst["escaled"])
print("   max|h| per scenario:", {k: (min(v), max(v)) for k, v in hzero.items()})
print("   e_min range:", min(i["e_min"] for i in insts.values()), max(i["e_min"] for i in insts.values()),
      "| #optimal states:", sorted({len(i["optimal_states"]) for i in insts.values()}), "| #near states:", sorted({int(i["near_mask"].sum()) for i in insts.values()}))

# ---------- B. symmetry reduction on real instances
print("B. symmetry reduction")
for key, inst in insts.items():
    sym = np.allclose(inst["energies"], inst["energies"][::-1], atol=1e-9)
    if key[3] == "2 equal devices":
        assert sym
        red = ns["t4_reduced_view"](inst)
        n = inst["n"]
        J, h, c = inst["model"]["J"], inst["model"]["h"], inst["model"]["const"]
        # independent: energies of the states with x_0 = 0, by my own enumeration of the FULL model at index k' << 1
        full = my_energies(J, h, c, n)
        kk = np.arange(2 ** (n - 1))
        assert np.abs(full[kk << 1] - red["energies"]).max() < 1e-10
        # reduced scaled energies equal my enumeration of the reduced scaled J', h'
        assert np.abs(my_energies(red["J_scaled"], red["h_scaled"], 0.0, n - 1) - red["energy_scaled"]).max() < 1e-10
        assert np.array_equal(np.sort(red["optimal_states"] << 1), np.sort(inst["optimal_states"][inst["optimal_states"] % 2 == 0]))
        assert np.array_equal(red["near_mask"], inst["near_mask"][kk << 1]) and np.array_equal(red["balanced_mask"], inst["balanced_mask"][kk << 1])
        assert len(inst["optimal_states"]) == 2 * len(red["optimal_states"]) and inst["near_mask"].sum() == 2 * red["near_mask"].sum()
        # uniform bookkeeping as in run_qaoa_groups
        up_full = len(inst["optimal_states"]) / 2**n
        up_red = np.count_nonzero(inst["optimal_states"] % 2 == 0) / 2 ** (n - 1)
        un_full = inst["near_mask"].sum() / 2**n
        un_red = np.count_nonzero(np.flatnonzero(inst["near_mask"]) % 2 == 0) / 2 ** (n - 1)
        assert abs(up_red - len(red["optimal_states"]) / 2 ** (n - 1)) < 1e-15 and abs(un_red - red["near_mask"].mean()) < 1e-15
        assert abs(up_full - up_red) < 1e-15 and abs(un_full - un_red) < 1e-15
        scale_ratio = max(np.abs(J[1:, 1:]).max(), np.abs(h[1:] + J[0, 1:]).max()) / inst["model"]["scale"]
        if key[0] in ("1 control", "2 main"):
            print(f"   {key}: reduced n={red['n']}, scale_red/scale_full={scale_ratio:.4f}, e_max full {inst['e_max']:.4f} red {red['e_max']:.4f}, e_mean full {inst['e_mean']:.4f} red {red['e_mean']:.4f}, e_std full {inst['e_std']:.4f} red {red['e_std']:.4f}")
    else:
        assert not sym
print("   all equal-device instances: reduced energies = full energies at k<<1, optimal/near/balanced masks map, uniform bookkeeping equal: ok")
# synthetic: random J, h (with fields) -> the algebraic reduction formula for x_0 = 0 (general), and symmetric (h = 0)
worst_red = 0
for n in range(2, 10):
    for trial in range(10):
        J = np.triu(rng.normal(size=(n, n)), 1); h = rng.normal(size=n) * (trial % 2); const = rng.normal()
        full = my_energies(J, h, const, n)
        Jr, hr, cr = J[1:, 1:], h[1:] + J[0, 1:], const + h[0]
        red = my_energies(Jr, hr, cr, n - 1) if n > 1 else np.array([cr])
        worst_red = max(worst_red, np.abs(full[0::2] - red).max(), np.abs(full[np.arange(2 ** (n - 1)) << 1] - red).max())
        if trial % 2 == 0:   # symmetric: run the cell function on a synthetic inst
            e_min = full.min()
            inst = dict(model=dict(J=J, h=h, const=const), n=n, energies=full, e_min=float(e_min), optimal_states=np.flatnonzero(full <= e_min + 1e-9),
                        near_mask=full <= e_min + 0.05 * abs(e_min), balanced_mask=np.zeros(2**n, bool))
            r = ns["t4_reduced_view"](inst)
            assert np.abs(r["energies"] - red).max() < 1e-10 and abs(full[r["optimal_states"] << 1].min() - e_min) < 1e-10
print("   synthetic random J,h n=2..9: max |E_full(even states) - E_reduced(J',h',const')| =", worst_red)

# ---------- C. relaxation
print("C. relaxed_solution")
captured = {}
real_minimize = ns["minimize"]
def spy(fun, x0, **kw):
    captured["fun"] = fun
    res = real_minimize(fun, x0, **kw)
    captured["res"] = res
    return res
ns["minimize"] = spy
worst_bin = worst_grad = 0
for key, inst in list(insts.items()):
    c = ns["relaxed_solution"](inst)
    f = captured["fun"]; n = inst["n"]
    # binary points: objective equals the Ising energy (all 2^n states for n = 12)
    ks = rng.integers(0, 2**n, 300)
    for k in ks:
        x = ((k >> np.arange(n)) & 1).astype(float)
        worst_bin = max(worst_bin, abs(f(x)[0] - inst["energies"][k]))
    for _ in range(5):
        x = rng.uniform(0, 1, n); v, g = f(x)
        num = np.array([(f(x + 1e-6 * e)[0] - f(x - 1e-6 * e)[0]) / 2e-6 for e in np.eye(n)])
        worst_grad = max(worst_grad, np.abs(num - g).max() / max(1.0, np.abs(g).max()))
    inst["_relaxed"] = c
    inst["_relaxed_value"] = f(c)[0]
    inst["_relaxed_res"] = (captured["res"].nit, captured["res"].message)
print("   objective(binary x) - Ising energy, max over 300 random states x all instances:", worst_bin)
print("   analytic vs finite-difference gradient (relative):", worst_grad)
for key, inst in insts.items():
    c = inst["_relaxed"]
    print(f"   {key[0]:<16} {key[3][:9]:<9} alpha={inst['alpha']}: relaxed c min={c.min():.6f} max={c.max():.6f} (spread {c.max()-c.min():.2e}); objective={inst['_relaxed_value']:.3e}; "
          f"exact binary optimum={inst['e_min']:.4f}; L-BFGS nit={inst['_relaxed_res'][0]}")
# with mu > 0 (priority active): gradient + binary equality
T_mu = T["mu"]; T["mu"] = 0.7
cnt = 0
for case in ns["BENCHMARK_CASES"]:
    for scen in ns["TARGET_SHARES"]:
        reg = ns["region_cases"][(case, ns["RUN"]["region_counts"][0], ns["RUN"]["seeds"][0])]
        inst = ns["t4_build_instance"](reg, case, ns["RUN"]["region_counts"][0], ns["RUN"]["seeds"][0], scen)
        if inst["mu"] > 0:
            cnt += 1
            c = ns["relaxed_solution"](inst); f = captured["fun"]; n = inst["n"]
            wb = max(abs(f(((k >> np.arange(n)) & 1).astype(float))[0] - inst["energies"][k]) for k in rng.integers(0, 2**n, 300))
            x = rng.uniform(0, 1, n); v, g = f(x)
            num = np.array([(f(x + 1e-6 * e)[0] - f(x - 1e-6 * e)[0]) / 2e-6 for e in np.eye(n)])
            print(f"   mu=0.7 {case} {scen[:9]}: binary equality err {wb:.2e}; gradient rel err {np.abs(num - g).max() / max(1, np.abs(g).max()):.2e}; relaxed c spread {c.max() - c.min():.3f}")
T["mu"] = T_mu
print("   instances with active priority at mu=0.7:", cnt)
ns["minimize"] = real_minimize

# ---------- D. warm starts on the full instances
print("D. warm_start_setup")
I2 = np.eye(2)
for key in [k for k in insts if k[0] in ("1 control", "2 main", "6 three sources")]:
    inst = insts[key]; n = inst["n"]
    _, spectral_state, _ = ns["spectral_split"](inst)
    x_spec = ((spectral_state >> np.arange(n)) & 1)
    has_fields = bool(np.any(inst["model"]["h"] != 0))
    for flavour in ns["T4_WARM_FLAVOURS"]:
        thetas, axis, c, note = ns["warm_start_setup"](inst, flavour, spectral_state)
        start = ns["qaoa_statevector"](inst["energy_scaled"], [], [], n, thetas, axis)
        # start state marginals P(bit_i = 1) = c_i
        probs0 = np.abs(start) ** 2
        ks = np.arange(2**n)
        marg = np.array([probs0[((ks >> i) & 1) == 1].sum() for i in range(n)])
        e1 = np.abs(marg - c).max()
        # product-form: probs0[k] = prod_i c_i^{x_i} (1-c_i)^{1-x_i}
        bits = (ks[:, None] >> np.arange(n)[None, :]) & 1
        prod = np.prod(np.where(bits == 1, c, 1 - c), axis=1)
        e2 = np.abs(prod - probs0).max()
        out = f"   {key[0]:<16} {key[3][:9]:<9} {flavour:<26} c in [{c.min():.3f},{c.max():.3f}] marg err {e1:.1e} product err {e2:.1e}"
        if flavour.endswith("mixer A"):
            beta = 0.83
            evolved = ns["apply_qubit_gates"](start, ns["mixer_gates"](beta, axis))
            out += f" | mixer A eigenstate: |U start - e^(i n beta) start| = {np.abs(evolved - np.exp(1j * n * beta) * start).max():.1e}"
            # hence gamma = 0 leaves the distribution unchanged for any betas
            pr = ns["qaoa_probabilities"](inst["energy_scaled"], [0.0, 0.0], [0.4, 1.1], n, thetas, axis)
            out += f"; gamma=0 leaves P unchanged: {np.abs(pr - probs0).max():.1e}"
        else:
            for p in (1, 2):
                given = np.concatenate([np.zeros(p), [np.pi / 2], np.zeros(p - 1)])
                pr = ns["view_probabilities"](ns["t4_view"](inst, thetas, axis), given)
                k = int(np.argmax(pr))
                start_round = int(sum(int(ci > 0.5) << i for i, ci in enumerate(c)))
                comp = (2**n - 1) ^ start_round
                out += f" | p={p} recovery: max P={pr.max():.12f}, argmax==complement(start rounding): {k == comp}, ==spectral: {k == spectral_state}, ==complement(spectral): {k == ((2**n - 1) ^ spectral_state)}"
            out += f" | fields={has_fields} E(spectral)={inst['energies'][spectral_state]:.4f} E(complement)={inst['energies'][(2**n-1) ^ spectral_state]:.4f} E(recovered)={inst['energies'][k]:.4f} e_min={inst['e_min']:.4f}"
        print(out)

# ---------- E. distribution_metrics with a hand-computable view
print("E. distribution_metrics (hand example), shots =", ns["T4_SHOTS"])
view = dict(energies=np.array([1.0, 2.0, 1.0, 5.0]), optimal_states=np.array([0, 2]), near_mask=np.array([True, False, True, False]),
            balanced_mask=np.array([True, True, False, False]), e_min=1.0, e_max=5.0)
probs = np.array([0.1, 0.2, 0.3, 0.4])
m = ns["distribution_metrics"](probs, view)
S = ns["T4_SHOTS"]
expect = dict(p_opt=0.4, p_near=0.4, mean_cost=0.1 + 0.4 + 0.3 + 2.0, mean_ratio=(5 - 2.8) / 4, p_balanced=0.3, hit_opt=1 - 0.6**S, hit_near=1 - 0.6**S)
print("   got     ", {k: round(v, 6) for k, v in m.items()})
print("   expected", {k: round(v, 6) for k, v in expect.items()})
assert all(abs(m[k] - expect[k]) < 1e-12 for k in expect)
view2 = dict(view, near_mask=np.array([True, True, True, False]))
m2 = ns["distribution_metrics"](np.array([0.001, 0.002, 0.0, 0.997]), view2)
print("   small p: p_opt", m2["p_opt"], "hit_opt", m2["hit_opt"], "expected", 1 - (1 - 0.001) ** S, "| p_near", m2["p_near"], "hit_near", m2["hit_near"], "expected", 1 - (1 - 0.003) ** S)

# near_mask definition on real instances
for key in list(insts)[:4]:
    inst = insts[key]
    mine = inst["energies"] <= 1.05 * inst["e_min"] + 1e-9
    assert np.array_equal(mine, inst["near_mask"]), key
    bal = np.abs(inst["share_b"] - inst["alpha"]) <= 0.05
    assert np.array_equal(bal, inst["balanced_mask"])
print("   near_mask = (E <= 1.05 E_min), balanced_mask = |share_B - alpha| <= 0.05 on real instances: ok")

# ---------- F. modelled_outcome
print("F. modelled_outcome")
for key in [k for k in insts if k[0] == "2 main"]:
    inst = insts[key]; n = inst["n"]; hours = T["job_reference_hours"]; f_comm = T["comm_fraction_if_all_borders_cut"]
    (sa, sb) = inst["speeds"]; (act_a, idle_a), (act_b, idle_b) = inst["power"]
    worst_h = worst_k = worst_alt = 0
    for state in rng.integers(0, 2**n, 200):
        state = int(state)
        x = (state >> np.arange(n)) & 1
        reg = inst["regions"]
        work = np.asarray(reg["region_work"], float); W = work.sum()
        share_b = work[x == 1].sum() / W
        pairs = np.asarray(reg["region_pairs"]); cost = np.asarray(reg["region_boundary_cost"], float)
        cutfrac = cost[x[pairs[:, 0]] != x[pairs[:, 1]]].sum() / cost.sum()
        busy_a, busy_b = hours * (1 - share_b) / sa, hours * share_b / sb
        comm = f_comm * hours * cutfrac
        total = max(busy_a, busy_b) + comm
        kwh = (act_a * busy_a + idle_a * (total - busy_a) + act_b * busy_b + idle_b * (total - busy_b)) / 1000
        o = ns["modelled_outcome"](inst, state)
        worst_h = max(worst_h, abs(o["modelled_hours"] - total), abs(inst["modelled_hours_all"][state] - total))
        worst_k = max(worst_k, abs(o["modelled_kwh"] - kwh))
        assert abs(o["share_b"] - share_b) < 1e-12 and abs(o["cut_fraction"] - cutfrac) < 1e-12 and abs(o["load_b"] - work[x == 1].sum()) < 1e-9
        # non-enumerated code path
        inst2 = {k: v for k, v in inst.items() if k not in ("share_b", "cut", "energies")}
        o2 = ns["modelled_outcome"](inst2, state)
        worst_alt = max(worst_alt, abs(o2["modelled_hours"] - total), abs(o2["modelled_kwh"] - kwh))
        assert abs(ns["state_cost"](inst2, state) - inst["energies"][state]) < 1e-9
    print(f"   {key[3]}: speeds {inst['speeds']} power(active,idle) {inst['power']} | max |hours - mine| = {worst_h:.1e}, |kWh - mine| = {worst_k:.1e}, non-enumerated path = {worst_alt:.1e}")
    s = int(inst["optimal_states"][0]); o = ns["modelled_outcome"](inst, s)
    print("   optimum outcome:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()})
    # does the Ising optimum minimise modelled hours?
    best_rt = int(np.argmin(inst["modelled_hours_all"]))
    print(f"   runtime-optimal state {best_rt}: hours {inst['modelled_hours_all'][best_rt]:.3f}, Ising cost {inst['energies'][best_rt]:.4f} (rank {int((inst['energies'] < inst['energies'][best_rt] - 1e-12).sum())}); Ising optimum hours {inst['modelled_hours_all'][s]:.3f}")
# idle-power sanity: device finishing early idles until the end, hand numbers
fake = dict(speeds=(1.0, 4.0), power=((300.0, 100.0), (500.0, 50.0)), n=2, alpha=0.8,
            share_b=np.array([0.0, 0.4, 0.6, 1.0]), cut=np.array([0.0, 1.0, 1.0, 0.0]),
            regions=dict(region_pairs=np.array([[0, 1]])), model=dict(W=10.0))
o = ns["modelled_outcome"](fake, 1)
H = T["job_reference_hours"]; fc = T["comm_fraction_if_all_borders_cut"]
ba, bb = H * 0.6 / 1, H * 0.4 / 4; tot = max(ba, bb) + fc * H
print(f"   hand example: hours {o['modelled_hours']:.4f} (expected {tot:.4f}); kWh {o['modelled_kwh']:.4f} (expected {(300 * ba + 100 * (tot - ba) + 500 * bb + 50 * (tot - bb)) / 1000:.4f})")
