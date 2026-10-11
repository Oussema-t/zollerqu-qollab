import os
SMOKE = 'overrides=dict(eval_budget=40, random_starts=1, lr_grid_gamma=np.linspace(0.05, 1.5, 12), lr_grid_beta=np.linspace(0.1, 1.2, 6), classical_repeats=5, use_checkpoint=False, calibration_seeds=[101], sa_calibration_repeats=3)'
M = {
 "a_mixer_block_sign": [("return (hadamard * np.exp(1j * beta * weights)) @ hadamard", "return (hadamard * np.exp(-1j * beta * weights)) @ hadamard", 1)],
 "b_rx_plus": [("circuit.rx(float(-2 * beta), q)", "circuit.rx(float(2 * beta), q)", 1)],
 "c1_bits_to_state_rev": [("return int(np.dot(np.asarray(x, dtype=np.int64), 1 << np.arange(len(x), dtype=np.int64)))", "return int(np.dot(np.asarray(x, dtype=np.int64), 1 << np.arange(len(x) - 1, -1, -1, dtype=np.int64)))", 1)],
 "c2_pieces_rev": [('pieces = device_piece_counts(state_index_to_x(state, inst["n"]), inst["regions"]["region_pairs"])', 'pieces = device_piece_counts(state_index_to_x(state, inst["n"])[::-1], inst["regions"]["region_pairs"])', 1)],
 "c3_warm_x_rev": [('x_spectral = state_index_to_x(spectral_state, inst["n"]).astype(float)', 'x_spectral = state_index_to_x(spectral_state, inst["n"]).astype(float)[::-1]', 1)],
 "c4_spectral_only_rev": [("            state = bits_to_state(x)\n            energy = state_cost(inst, state)", "            state = bits_to_state(x[::-1])\n            energy = state_cost(inst, state)", 1)],
 "c5_warm_state_kron_rev": [("for q in range(n - 1, -1, -1):\n            state = np.kron(state", "for q in range(n):\n            state = np.kron(state", 1)],
 "c6_nonenum_outcome_rev": [('share_b = float(inst["model"]["a_tilde"] @ state_index_to_x(state, inst["n"]))', 'share_b = float(inst["model"]["a_tilde"] @ state_index_to_x(state, inst["n"])[::-1])', 1)],
 "d_lr_beta_ramps_up": [("(1.0 - (k - 1) / p) * delta_beta", "(k / p) * delta_beta", 1)],
 "d2_lr_gamma_ramps_down": [("k / p * delta_gamma, (1.0 - (k - 1) / p) * delta_beta", "(1.0 - (k - 1) / p) * delta_gamma, k / p * delta_beta", 1)],
 "e1_swap_AB_setup_only": [('return thetas, (-thetas if flavour.endswith("mixer B") else thetas), c, note', 'return thetas, (thetas if flavour.endswith("mixer B") else -thetas), c, note', 1)],
 "e2_swap_AB_consistent": [('return thetas, (-thetas if flavour.endswith("mixer B") else thetas), c, note', 'return thetas, (thetas if flavour.endswith("mixer B") else -thetas), c, note', 1),
                           ("sign = -1.0 if modified_mixer else 1.0", "sign = 1.0 if modified_mixer else -1.0", 1)],
 "f_h_red_no_J": [("h[1:] + J[0, 1:], const + h[0]", "h[1:], const + h[0]", 1)],
 "g1_sa_delta_sign": [("delta = -2.0 * z[i] * local[i]                         # energy change of flipping spin i", "delta = 2.0 * z[i] * local[i]", 1)],
 "g2_sa_accept_uphill": [("if delta <= 0 or rng.random() < np.exp(-delta / temperature):", "if delta >= 0 or rng.random() < np.exp(delta / temperature):", 1)],
 "g3_sa_local_update_wrong": [("            z[i] = -z[i]\n            local += 2.0 * z[i] * J_sym[:, i]\n            energy += delta\n            if energy < best_energy:", "            z[i] = -z[i]\n            energy += delta\n            if energy < best_energy:", 1)],
 "h_cvar_highest": [("return float(ordered[: int(np.ceil(fraction * ordered.size - 1e-9))].mean())", "return float(ordered[::-1][: int(np.ceil(fraction * ordered.size - 1e-9))].mean())", 1)],
 "i1_balance_no_4": [("energies = cut + lam * 4.0 * (alpha - share_b) ** 2 + mu * prio_slow", "energies = cut + lam * (alpha - share_b) ** 2 + mu * prio_slow", 1)],
 "i2_relax_no_4": [("gradient = 8.0 * lam * balance * model[\"a_tilde\"] - mu * model[\"p_tilde\"]", "gradient = 2.0 * lam * balance * model[\"a_tilde\"] - mu * model[\"p_tilde\"]", 1),
                   ("+ 4.0 * lam * balance**2 + mu * np.sum", "+ lam * balance**2 + mu * np.sum", 1)],
 "j_cost_phase_sign": [("state = state * (np.cos(phase) - 1j * np.sin(phase))", "state = state * (np.cos(phase) + 1j * np.sin(phase))", 1)],
 "k_rz_h_factor": [("circuit.rz(float(2 * gamma * h_scaled[q]), q)", "circuit.rz(float(gamma * h_scaled[q]), q)", 1)],
 "l_hit_near_shots": [("hit_near=float(1.0 - (1.0 - min(p_near, 1.0)) ** T4_SHOTS))", "hit_near=float(1.0 - (1.0 - min(p_near, 1.0)) ** (T4_SHOTS * 2)))", 1)],
 "m_near_factor_ignored": [('near_limit = TASK04["near_optimal_factor"] * e_min if e_min > T4_TOL else T4_TOL   # optimum 0: absolute tolerance', 'near_limit = 1.5 * e_min if e_min > T4_TOL else T4_TOL', 1)],
 "n_local_search_argmax": [("i = int(np.argmin(deltas))\n            if deltas[i] >= -1e-12:\n                break", "i = int(np.argmax(deltas))\n            if deltas[i] <= 1e-12:\n                break", 1)],
 "o_noise_uniform_swap": [("noisy_near = fidelity * run[\"p_near\"] + (1.0 - fidelity) * group[\"uniform_p_near\"]", "noisy_near = fidelity * run[\"p_near\"] + (1.0 - fidelity) * 1.0", 1)],
 "p_balanced_mask_eps": [('balanced_mask=imbalance <= TASK04["epsilon"]', 'balanced_mask=imbalance <= 5 * TASK04["epsilon"]', 1)],
 "q_runtime_model_min": [("total = max(busy_a, busy_b) + comm", "total = min(busy_a, busy_b) + comm", 1)],
 "r_reduced_scale": [("energy_scaled=(energies_red - const_red) / scale_red, thetas=None", "energy_scaled=(energies_red - const_red) / (4 * scale_red), thetas=None", 1)],
 "s_spsa_ascent": [("angles = angles - step * gradient_estimate(", "angles = angles + step * gradient_estimate(", 1)],
 "t_energy_scaled_no_const": [('energy_scaled=(energies - const) / model["scale"])', 'energy_scaled=(energies - const) / (2 * model["scale"]))', 1)],
}
os.makedirs("cfg", exist_ok=True)
for name, muts in M.items():
    open(f"cfg/mut_{name}.py", "w").write(
        f'CFG = dict(workdir="wd_mut_{name}", mode="small", run=dict(quantum_cases=["2 main", "9 priority"]), {SMOKE},\n           mutations={muts!r})\n')
print(len(M), "mutation configs written")
