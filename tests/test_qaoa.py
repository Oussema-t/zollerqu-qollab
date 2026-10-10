"""Task 4 tests: QAOA circuit conventions, the NumPy simulator against Qiskit, tuning helpers, varQITE."""
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from stocks import ising, qaoa, qaoa_sim, task3_report, varqite
from stocks import qaoa_tuning as tuning

MU = np.array([0.01, -0.02, 0.005, 0.0])
COV = np.array([[0.04, 0.01, 0.0, 0.005], [0.01, 0.09, -0.02, 0.0], [0.0, -0.02, 0.01, 0.002], [0.005, 0.0, 0.002, 0.03]])


def setUpModule():
    global SETUP12, SETUP29
    ctx = task3_report.load_context()
    SETUP12 = task3_report.size_setup(ctx, 12)
    SETUP29 = task3_report.size_setup(ctx, 29)


def small_model(mode="long_short", **kwargs):
    return ising.build_ising(MU, COV, mode, 1.0, **kwargs)


class CircuitConventionTests(unittest.TestCase):
    def test_cost_unitary_is_diagonal_phase_of_the_energy(self):
        """The rzz/rz layer must equal exp(-i gamma (E(z) - const) / scale) on every basis state."""
        model = small_model()
        gamma, scale = 0.37, qaoa_sim.cost_scale(model)
        circuit = QuantumCircuit(4)
        for i, j, coupling in model.pair_terms():
            circuit.rzz(2.0 * gamma * coupling / scale, i, j)
        for i in range(4):
            circuit.rz(2.0 * gamma * model.fields[i] / scale, i)
        unitary = Operator(circuit).data
        energies = model.energy(ising.all_states(4)) - model.const
        expected = np.exp(-1j * gamma * energies / scale)
        self.assertLess(np.abs(unitary - np.diag(expected)).max(), 1e-12)
        self.assertLess(np.abs(unitary - np.diag(np.diag(unitary))).max(), 1e-12)  # diagonal

    def test_mixer_direction_lowers_energy_only_with_opposite_signs(self):
        """Starting from |+..+>, the first-order energy change is proportional to -gamma*beta: same sign raises it."""
        model = small_model()
        sim = qaoa_sim.QaoaSimulator(model)
        uniform = sim.expectation(sim.probabilities([], []))
        self.assertLess(sim.expectation(sim.probabilities([0.3], [-0.3])), uniform)
        self.assertGreater(sim.expectation(sim.probabilities([0.3], [0.3])), uniform)

    def test_constant_changes_energy_not_probabilities(self):
        model = small_model()
        shifted = ising.build_ising(MU, COV, "long_short", 1.0)
        shifted.const += 5.0
        a, b = qaoa_sim.QaoaSimulator(model), qaoa_sim.QaoaSimulator(shifted)
        pa, pb = a.probabilities([0.4, 0.2], [-0.3, -0.1]), b.probabilities([0.4, 0.2], [-0.3, -0.1])
        np.testing.assert_allclose(pa, pb, atol=1e-14)
        self.assertAlmostEqual(b.expectation(pb) - a.expectation(pa), 5.0, places=12)

    def test_bit_order_matches_count_keys(self):
        """Index k of the probability vector is the state whose bitstring (qubit 0 rightmost) is bin(k)."""
        model = small_model()
        for k in (1, 6, 11):
            spins = ising.all_states(4)[k]
            key = ising.spins_to_bitstring(spins)
            self.assertEqual(int(key, 2), k)
            self.assertEqual(key[-1], "0" if spins[0] == 1 else "1")
        # a circuit that flips only qubit 2 (X) must put all probability on index 4
        circuit = QuantumCircuit(4)
        circuit.x(2)
        probabilities = np.asarray(Statevector.from_instruction(circuit).probabilities())
        self.assertEqual(int(np.argmax(probabilities)), 4)

    def test_builder_uses_retained_pairs_and_all_fields(self):
        model = task3_report.build_model(SETUP29, "long_short", 1.0, False)
        circuit = qaoa.build_circuit(model, [0.3, 0.2], [-0.2, -0.1])
        ops = qaoa.circuit_metrics(circuit)["gates"]
        self.assertEqual(ops["rzz"], 2 * 244)
        self.assertEqual(ops["rz"], 2 * 29)
        self.assertEqual(ops["rx"], 2 * 29)
        self.assertEqual(ops["h"], 29)
        self.assertNotIn("ry", ops)
        p0 = qaoa.build_circuit(model, [], [])
        self.assertEqual(qaoa.circuit_metrics(p0)["gates"], {"h": 29})
        self.assertEqual(p0.count_ops().get("measure"), 29)
        bare = qaoa.build_circuit(model, [], [], measure=False)
        self.assertNotIn("measure", bare.count_ops())

    def test_common_scale_and_floor(self):
        model = small_model()
        self.assertAlmostEqual(qaoa_sim.cost_scale(model), max(np.abs(model.couplings).max(), np.abs(model.fields).max()))
        tiny = ising.build_ising(np.zeros(3), np.eye(3) * 1e-20, "long_short", 1.0)
        self.assertEqual(qaoa_sim.cost_scale(tiny), 1.0)
        with self.assertRaises(ValueError):
            qaoa.build_circuit(model, [0.1], [0.1], scale=0.0)
        with self.assertRaises(ValueError):
            qaoa.build_circuit(model, [0.1, 0.2], [0.1])

    def test_lr_schedule(self):
        gammas, betas = qaoa.lr_schedule(4, 0.8, 0.4)
        np.testing.assert_allclose(gammas, [0.2, 0.4, 0.6, 0.8])
        np.testing.assert_allclose(betas, [0.4, 0.3, 0.2, 0.1])
        self.assertEqual(len(qaoa.lr_schedule(0)[0]), 0)
        with self.assertRaises(ValueError):
            qaoa.lr_schedule(-1)

    def test_layered_order_is_a_permutation_with_disjoint_rounds(self):
        model = task3_report.build_model(SETUP12, "long_short", 1.0, False)
        terms = model.pair_terms()
        ordered = qaoa.layered_pair_terms(terms)
        self.assertEqual(sorted(ordered), sorted(terms))
        busy, rounds = set(), 1
        for i, j, _ in ordered:
            if i in busy or j in busy:
                busy, rounds = set(), rounds + 1
            busy.update((i, j))
        self.assertLessEqual(rounds, 2 * 12)

    def test_large_circuit_is_never_simulated(self):
        model = task3_report.build_model(SETUP29, "long_short", 1.0, False)
        circuit = qaoa.build_circuit(model, [0.3], [-0.2])
        metrics = qaoa.circuit_metrics(circuit)
        self.assertEqual((metrics["qubits"], metrics["two_qubit_gates"]), (29, 244))
        self.assertEqual(metrics, qaoa.circuit_metrics(qaoa.build_circuit(model, [0.3], [-0.2])))  # reproducible
        with self.assertRaises(ValueError):
            qaoa.statevector_probabilities(circuit)
        with self.assertRaises(ValueError):
            qaoa_sim.QaoaSimulator(model)

    def test_routing_counts_more_gates_on_heavy_hex(self):
        from qiskit.transpiler import CouplingMap
        model = task3_report.build_model(SETUP12, "long_short", 1.0, False)
        circuit = qaoa.build_circuit(model, [0.3], [-0.2])
        flat = qaoa.routed_metrics(circuit)
        routed = qaoa.routed_metrics(circuit, CouplingMap.from_heavy_hex(3))
        self.assertEqual(flat["two_qubit_gates"], 66)
        self.assertGreater(routed["two_qubit_gates"], 66)
        self.assertEqual(routed, qaoa.routed_metrics(circuit, CouplingMap.from_heavy_hex(3)))  # same seed, same result


class SimulatorTests(unittest.TestCase):
    def test_uniform_at_p0_and_normalised(self):
        for mode in ising.MODES:
            sim = qaoa_sim.QaoaSimulator(task3_report.build_model(SETUP12, mode, 1.0, False))
            p0 = sim.probabilities([], [])
            np.testing.assert_allclose(p0, 1 / 4096, atol=1e-15)
            for p in (1, 2, 3):
                probabilities = sim.probabilities(np.full(p, 0.3), np.full(p, -0.2))
                self.assertTrue(np.all(np.isfinite(probabilities)))
                self.assertAlmostEqual(probabilities.sum(), 1.0, places=12)

    def test_matches_qiskit_statevector(self):
        rng = np.random.default_rng(7)
        for mode in ising.MODES:
            for balance in (False, True):
                model = task3_report.build_model(SETUP12, mode, 1.0, balance)
                sim = qaoa_sim.QaoaSimulator(model)
                for p in (1, 2, 3):
                    gammas, betas = rng.uniform(0, 1.5, p), rng.uniform(-1, 1, p)
                    mine = sim.probabilities(gammas, betas)
                    theirs = qaoa.statevector_probabilities(qaoa.build_circuit(model, gammas, betas))
                    self.assertLess(np.abs(mine - theirs).max(), 1e-12)
                    self.assertAlmostEqual(sim.expectation(mine), float(theirs @ sim.energies), places=12)

    def test_variants_match_qiskit(self):
        model = task3_report.build_model(SETUP12, "buy_or_skip", 1.0, False)
        sim = qaoa_sim.QaoaSimulator(model)
        gammas, betas, field_gammas, alphas = [0.5, 0.2], [-0.3, -0.1], [0.1, 0.4], [0.2, -0.3]
        mine = sim.probabilities(gammas, betas, field_gammas=field_gammas, alphas=alphas)
        theirs = qaoa.statevector_probabilities(qaoa.build_circuit(model, gammas, betas, field_gammas=field_gammas, alphas=alphas))
        self.assertLess(np.abs(mine - theirs).max(), 1e-12)

    def test_all_qubit_rotation_equals_single_qubit_gates(self):
        rng = np.random.default_rng(3)
        for n in (5, 12):
            amplitudes = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            slow_x, slow_y = amplitudes.copy(), amplitudes.copy()
            for q in range(n):
                slow_x = qaoa_sim.apply_rx(slow_x, n, q, 0.37)
                slow_y = qaoa_sim.apply_ry(slow_y, n, q, -0.21)
            self.assertLess(np.abs(qaoa_sim.apply_to_all(amplitudes, n, qaoa_sim.rx_block, 0.37) - slow_x).max(), 1e-12)
            self.assertLess(np.abs(qaoa_sim.apply_to_all(amplitudes, n, qaoa_sim.ry_block, -0.21) - slow_y).max(), 1e-12)

    def test_xy_mixer_preserves_the_number_of_ones(self):
        model = small_model()
        sim = qaoa_sim.QaoaSimulator(model)
        from stocks.task4_report import build_xy_circuit, dicke_state, ring_pairs
        initial = dicke_state(4, 2).astype(complex)
        self.assertAlmostEqual(np.sum(np.abs(initial) ** 2), 1.0)
        amplitudes = sim.amplitudes([0.4, 0.3], [0.2, -0.5], initial=initial, xy_pairs=ring_pairs(4))
        ones = np.array([bin(k).count("1") for k in range(16)])
        self.assertLess(np.abs(amplitudes[ones != 2]).max(), 1e-14)
        qiskit_probabilities = qaoa.statevector_probabilities(build_xy_circuit(model, [0.4], [0.2], ring_pairs(4), initial))
        mine = np.abs(sim.amplitudes([0.4], [0.2], initial=initial, xy_pairs=ring_pairs(4))) ** 2
        self.assertLess(np.abs(mine - qiskit_probabilities).max(), 1e-12)
        self.assertAlmostEqual(np.sum(np.abs(amplitudes) ** 2), 1.0, places=12)

    def test_distribution_metrics_and_cvar(self):
        energies = np.array([0.0, 1.0, 2.0, 3.0])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        m = qaoa_sim.distribution_metrics(probabilities, energies, 0.0, 3.0, 1e-12, shots=10)
        self.assertAlmostEqual(m["expectation"], 2.0)
        self.assertAlmostEqual(m["quality"], 1 / 3)
        self.assertAlmostEqual(m["p_optimum"], 0.1)
        self.assertAlmostEqual(m["p_optimum_in_shots"], 1 - 0.9 ** 10)
        self.assertEqual(m["most_probable_state"], 3)
        self.assertAlmostEqual(qaoa_sim.expected_best_of_shots(probabilities, energies, 1), 2.0)  # one shot: the mean
        # two shots: P(min = e_k) = P(both >= e_k) - P(both > e_k)
        two = 0.0 * (1 - 0.9 ** 2) + 1.0 * (0.9 ** 2 - 0.7 ** 2) + 2.0 * (0.7 ** 2 - 0.4 ** 2) + 3.0 * 0.4 ** 2
        self.assertAlmostEqual(qaoa_sim.expected_best_of_shots(probabilities, energies, 2), two)
        self.assertAlmostEqual(qaoa_sim.cvar(probabilities, energies, 0.1), 0.0)
        self.assertAlmostEqual(qaoa_sim.cvar(probabilities, energies, 0.3), (0.1 * 0.0 + 0.2 * 1.0) / 0.3)
        self.assertAlmostEqual(qaoa_sim.cvar(probabilities, energies, 1.0), 2.0)
        self.assertEqual(qaoa_sim.distribution_metrics(probabilities, np.ones(4), 1.0, 1.0, 1e-12)["quality"], 1.0)
        boltzmann = qaoa_sim.boltzmann(energies, 1.0)
        self.assertAlmostEqual(boltzmann.sum(), 1.0)
        self.assertTrue(np.all(np.diff(boltzmann) < 0))


class TuningTests(unittest.TestCase):
    def test_unpack_and_starts(self):
        x = tuning.lr_start("standard", 3, 0.6, -0.3)
        parts = tuning.unpack("standard", x, 3)
        np.testing.assert_allclose(parts["gammas"], [0.2, 0.4, 0.6])
        np.testing.assert_allclose(parts["betas"], [-0.3, -0.2, -0.1])
        self.assertEqual(len(tuning.lr_start("multi_angle", 2, 0.6, -0.3)), 6)
        self.assertEqual(len(tuning.lr_start("dc", 2, 0.6, -0.3)), 6)
        with self.assertRaises(ValueError):
            tuning.unpack("standard", x, 2)
        starts = tuning.starts_for("standard", 2, (0.4, -0.2))
        self.assertEqual(len(starts), 2 + tuning.N_RANDOM_STARTS)
        np.testing.assert_array_equal(starts["random_seed_4026"], tuning.starts_for("standard", 2, (0.4, -0.2))["random_seed_4026"])

    def test_evaluation_cap_and_best_point(self):
        sim = qaoa_sim.QaoaSimulator(small_model())
        for optimizer in tuning.OPTIMIZERS:
            record = tuning.optimize_angles(sim, "standard", 1, optimizer, [0.3, -0.2])
            self.assertLessEqual(record["evaluations"], tuning.evaluation_cap(1))
            self.assertIsNone(record["error"])
            self.assertIsNotNone(record["angles"])
            probabilities = tuning.probabilities_for(sim, record["angles"])
            self.assertAlmostEqual(sim.expectation(probabilities), record["objective_value"], places=12)
            self.assertLessEqual(record["objective_value"], sim.expectation(sim.probabilities([0.3], [-0.2])) + 1e-12)

    def test_unknown_optimizer_is_recorded_not_raised(self):
        record = tuning.optimize_angles(qaoa_sim.QaoaSimulator(small_model()), "standard", 1, "Nope", [0.3, -0.2])
        self.assertIn("ValueError", record["error"])
        self.assertIsNone(record["angles"])

    def test_grid_search_covers_both_signs_and_picks_lowest_expectation(self):
        model = small_model()
        sim = qaoa_sim.QaoaSimulator(model)
        energies = sim.energies
        optimal = energies <= energies.min() + 1e-12
        evaluate = lambda p: qaoa_sim.distribution_metrics(p, energies, energies.min(), energies.max(), 1e-12)
        grid = tuning.lr_grid_search(sim, optimal, evaluate, p_values=(1,), grid={"dgamma": [0.3, 0.6], "dbeta": [-0.3, 0.3]})
        entry = grid[1]
        values = np.array(entry["expectation"])
        self.assertEqual(entry["grid_points"], 4)
        self.assertEqual((entry["best_dgamma"], entry["best_dbeta"]), (0.3 if values[0].min() < values[1].min() else 0.6, -0.3))
        self.assertAlmostEqual(values.min(), entry["best"]["expectation"], places=12)
        self.assertLess(entry["best"]["expectation"], entry["recipe_as_written"]["expectation"])


class VarqiteTests(unittest.TestCase):
    def test_ansatz_matches_qiskit(self):
        rng = np.random.default_rng(5)
        model = small_model()
        for kind in ("A", "B"):
            ansatz = varqite.Ansatz(model, kind)
            theta = rng.uniform(-1, 1, ansatz.n_params)
            theirs = np.asarray(Statevector.from_instruction(ansatz.to_qiskit(theta, measure=False)).probabilities())
            self.assertLess(np.abs(ansatz.probabilities(theta) - theirs).max(), 1e-12)
            self.assertEqual(ansatz.two_qubit_gates(), len(model.pair_terms()))
        pairs = len(model.pair_terms())  # 4: two covariances of the small model are exactly 0
        self.assertEqual(pairs, 4)
        self.assertEqual(varqite.Ansatz(model, "A").n_params, pairs)
        self.assertEqual(varqite.Ansatz(model, "B").n_params, pairs + 1)
        with self.assertRaises(ValueError):
            varqite.Ansatz(model, "C")

    def test_pauli_terms_and_energy(self):
        model = small_model()
        ansatz = varqite.Ansatz(model, "A")
        terms = varqite.pauli_terms(model)
        values = varqite.term_values(ansatz, terms)
        rebuilt = sum(c * values[k] for k, (_, c) in enumerate(terms)) + model.const
        np.testing.assert_allclose(rebuilt, model.energy(ising.all_states(4)), atol=1e-14)

    def test_ansatz_b_is_stationary_at_zero(self):
        model = small_model()
        theta, record = varqite.train(model, "B", theta0=np.zeros(5), steps=3)
        self.assertLess(np.abs(theta).max(), 1e-12)  # cannot move from exactly zero
        self.assertEqual(record["start"].split(",")[0], "uniform(-0.1")  # the default start is random

    def test_training_lowers_energy_and_records_history(self):
        model = small_model()
        theta, record = varqite.train(model, "A", lr=0.4, steps=15)
        history = record["energy_per_step"]
        self.assertLess(record["best_energy"], history[0])
        self.assertEqual(record["evaluations"], record["steps_run"] * (1 + 2 * 4))  # 1 energy + 2 shifts per parameter
        self.assertEqual(len(theta), 4)
        self.assertIn("2404.16135", record["citation"])


if __name__ == "__main__":
    unittest.main()
