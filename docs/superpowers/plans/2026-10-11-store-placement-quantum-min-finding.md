# Store placement by quantum minimum finding: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. One implementer per task, explicit-path commits, no attribution lines.

**Goal:** A Qollab JS/Qiskit project for the Quantum Impact theme. Real census block populations and road travel times define a store-placement cost (travel, staffing, crowding, warehouse). The quantum subroutine is Dürr–Høyer minimum finding (Grover search with a tightening threshold). The 10 to 12 qubit demo shows the measured success probabilities landing on the theoretical curve and the optimum matching the exact classical answer; the scale argument is the proven quadratic query speedup, cited, not claimed.

**Theme text (verified Oct 10):** "Pick a public dataset, build a strong classical baseline, then show where a quantum subroutine makes a measurable difference." The measurable difference shown is oracle calls: N classical evaluations versus O(√N) quantum, both measured in the demo.

**Decisions made by the planner (Tushar can overrule):** county = Champaign County, Illinois; m = 10 candidate sites by default (1024 placements), m = 12 as a stretch after timing; the cost oracle is a diagonal lookup oracle built from the classical cost table (stated openly; at scale it would be a reversible arithmetic circuit); the warehouse choice is folded into the cost function; repo `D:\Projects\zollerqu-placement`, own GitHub repo plus a branch in Oussama's repo as for Unlocked.

**Deadline:** links to Oussama Sun Oct 11 14:00 CT. Build budget: about 6 hours of agent work plus 1 hour of Tushar on Qollab.

---

## Facts

| Fact | Source |
|---|---|
| TIGERweb layer 11 "Census Block Groups" (Census 2020 group) returns GEOID, CENTLAT, CENTLON, AREALAND, POP100, HU100 by plain GET, f=json, no key. Champaign County IL (STATE 17, COUNTY 019): 158 block groups, total POP100 205,865. Response carries `Access-Control-Allow-Origin: null`. | curl Oct 10, `https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/11/query?where=STATE%3D%2717%27+AND+COUNTY%3D%27019%27&outFields=GEOID,CENTLAT,CENTLON,AREALAND,POP100,HU100&returnGeometry=false&f=json` |
| Census API (`api.census.gov/data/2020/dec/pl`) redirects to a "Missing Key" page without a key. Not used. | curl Oct 10 |
| OSRM public demo table service returns durations and distances by plain GET, no key: `https://router.project-osrm.org/table/v1/driving/{lon,lat;...}?annotations=duration,distance`, HTTP 200. | curl Oct 10 |
| Dürr and Høyer, "A Quantum Algorithm for Finding the Minimum", arXiv quant-ph/9607014: finds the minimum of a table of size N in time O(c√N) with probability at least 1 − 1/2^c. | arXiv abstract, fetched Oct 10 |
| Montanaro, "Quantum speedup of branch-and-bound algorithms", Phys. Rev. Research 2, 013056 (2020): near-quadratic speedup of branch and bound in general. | arXiv 1906.10375 abstract, fetched Oct 10 |
| Grover success probability after j iterations with M marked of N: sin²((2j+1)θ), sin θ = √(M/N). | Boyer, Brassard, Høyer, Tapp 1998 (standard; the test asserts it numerically) |
| Qollab: local simulators allow unlimited `backend.run` per run; IonQ backends accept one job per run; multi-circuit jobs allowed; built-in simulator 24 qubits, Aria sims 25, Forte sims 29; GET-only network, servers must allow CORS; files up to 2 MB, 100 per project; SciPy and NumPy available in Pyodide. | qollab.xyz/learn/docs/runtime-environment and compute-backends, fetched Oct 10 |
| scipy.optimize.milp (HiGHS) exists locally (scipy 1.17.1). Availability inside Pyodide: not verified; brute force is the exact baseline regardless. | local python Oct 10 |
| Existing Unlocked repo gives the page skeleton: body-only index.html + main.js module, `qec.bridge` style Python entry points returning JSON, `dev/serve.py` harness with import map and Aer-backed fake backend, Playwright recorder. | D:\Projects\zollerqu-unlocked |

## Data model (`data/champaign.json`, built offline by `tools/build_data.py`)

```
{
  "county": "Champaign County, IL", "source": {"population": "TIGERweb Census 2020 Block Groups layer 11", "travel": "OSRM demo server, driving durations, fetched <date>"},
  "blocks": [{"geoid": "...", "lat": 40.08, "lon": -88.20, "pop": 840, "area_m2": 599342}, ...],        // 158
  "sites":  [{"id": 0, "name": "...", "lat": .., "lon": ..}, ...],                                       // 12 candidate store sites
  "warehouses": [{"id": 0, "lat": .., "lon": ..}, {"id": 1, ...}],                                       // 2
  "t_block_site": [[minutes]...],   // 158 x 12
  "t_site_site":  [[minutes]...],   // 12 x 12
  "t_wh_site":    [[minutes]...]    // 2 x 12
}
```

Sites: the centroids of the 12 block groups with the largest POP100 subject to pairwise OSRM time ≥ 4 minutes (greedy pick in population order). Warehouses: the two block-group centroids farthest from the population-weighted centre in opposite directions (east and west). Travel times from OSRM in chunks of at most 100 coordinates per request (demo server limit, to be confirmed by the tool; fall back to haversine distance at 30 km/h with a `"travel": "haversine"` marker if OSRM fails). File size well under 2 MB.

## Cost model (`qopt/cost.py`, pure NumPy, same code in Python and mirrored in JS for the sliders)

Placement S ⊆ sites, |S| = k. Parameters with defaults: k = 3; staff capacity c = 15,000 people per employee; staff cost 1.0 per employee; crowding radius τ = 6 min; crowding penalty λ = 20; warehouse trips weight 2.0 per minute.

- travel(S) = Σ_b pop_b · min_{i∈S} t_bi / Σ_b pop_b  (mean minutes per resident)
- staff(S) = Σ_{i∈S} ceil(assigned_pop_i / c) · staff cost, where each block is assigned to its nearest open site
- crowd(S) = λ · #{ {i,j} ⊆ S : t_ij < τ }
- warehouse(S) = min_w 2.0 · Σ_{i∈S} t_wi
- cost(S) = travel + staff + crowd + warehouse. Infeasible bitstrings (|S| ≠ k) get cost = +∞.

`cost_table(data, params) -> np.ndarray` of length 2^m indexed by the site bitstring (site i is bit i), with +∞ for infeasible. For m = 12 this is 4096 × 158 arithmetic, under 50 ms.

## Quantum part (`qopt/search.py`)

- Index register: m qubits, uniform superposition over all 2^m bitstrings (feasibility is part of the oracle, so the search space is "all placements").
- Threshold oracle O_T: phase −1 on x with cost(x) ≤ T. Built as `qiskit.circuit.library.DiagonalGate` on the vector (−1)^{[cost ≤ T]}. Gate count O(2^m); measured in Task 1.
- Grover iteration G = D · O_T with the standard diffusion (H, X, multi-controlled Z, X, H).
- `grover_circuit(cost, T, j)`: j iterations then measure. `success_theory(M, N, j) = sin²((2j+1)·asin(√(M/N)))`.
- Dürr–Høyer loop `find_min(cost, run, rng, c=...)`: threshold T = cost of a random feasible x; repeat: BBHT exponential search (pick j uniformly in [0, mstep), mstep grows by 6/5 up to √N) with oracle O_{T, strict} (cost < T); run one circuit with 1 shot... on simulators we use 100 shots and take the best sample; if a sample has cost < T, set T to it; stop after a run budget. Returns the trajectory: thresholds, iterations per round, cumulative oracle calls (= Σ j over rounds), final x.
- Everything above is plain Python on NumPy arrays plus Qiskit circuits; `run(circuits, shots)` is injected so tests use Aer and the page uses the Qollab backend.

Honesty line for the README and page: the oracle is a lookup table compiled from the classical cost table, because a reversible arithmetic cost circuit does not fit 29 qubits with 158 blocks. The theory counts oracle calls; the demo counts them too. The quadratic speedup is in the paper; wall clock is not claimed.

## Classical baselines (`qopt/classical.py`)

- `brute_force(cost)`: argmin, N evaluations (exact).
- `greedy(data, params, k)`: add the site with the best marginal cost, k times.
- `local_search(cost, m, k, restarts)`: swap neighbourhood from random feasible starts; returns best, and the number of distinct local minima found (shows the landscape).
- `milp(data, params, k)` (optional): p-median MIP via scipy.optimize.milp with the travel term only, reported as "exact for the travel part"; skipped if milp is unavailable in Pyodide.

## Page (`index.html`, `main.js`, `ui/`)

- Header, backend banner (reuse Unlocked's), data credit line.
- Map panel: SVG of block-group centroids (circle area ∝ population), candidate sites as squares, warehouses as triangles, chosen sites highlighted, assignment lines on hover.
- Controls: k (2..4), λ, τ, staff capacity, m (10 or 12). Changing a control recomputes the cost table and the classical answers instantly (in Python via the bridge; under 100 ms).
- "Solve classically": brute force, greedy, local search with local-minimum count, table of costs and oracle calls (N).
- "Run quantum search": one job with the theory-match sweep: fixed T chosen so that M = 1, 4 or 16 (three series), j = 0..J (J ≈ 1.2·π/4·√(N/M), capped at 12 circuits per series), 1000 shots each. Plot measured P(success) against sin²((2j+1)θ) per series. Then, on local simulators only, the Dürr–Høyer loop live, with the threshold trajectory animated on the map and the cumulative oracle-call counter next to N. On IonQ backends the loop is shown from the recorded take with a status note (one job per run).
- Result cards: best cost found (quantum) vs exact; oracle calls quantum vs N; max deviation of measured from theory (percentage points).
- "Load recorded take" with a take recorded locally on Aer (then replaced by a Forte simulator run by Tushar).
- Accessibility as in Unlocked: tabs/buttons keyboard reachable, aria-live status, reduced motion respected.

## Tasks

### Task 0 (planner, done): facts above.

### Task 1: Python core with tests (Opus)
Files: `qopt/cost.py`, `qopt/classical.py`, `qopt/search.py`, `qopt/bridge.py` (JSON entry points: `load_data`, `cost_table(params_json)`, `classical(params_json)`, `sweep_circuits(params_json, series_json)` returning circuits + meta, `sweep_analyze(meta_json, counts_json)`, `dh_round(...)` for the loop, `load_recorded`, `export_take`), `tests/`.
Tests (write first): cost of a hand-built 4-block, 3-site toy equals a hand computation; infeasible strings are +∞; brute force equals exhaustive Python loop; DiagonalGate oracle statevector phases equal (−1)^{[cost ≤ T]} for m = 6; Grover success probability at m = 8 matches sin² within 0.03 for M ∈ {1, 4, 16} and j = 0..6 (Aer statevector, 4000 shots); DH loop finds the brute-force optimum in at least 9 of 10 seeds at m = 8; cumulative oracle calls are reported. Timing print (not asserted): one Grover circuit at m = 10 and m = 12 with j = 10 on Aer, transpiled gate counts.
Commit: "Cost model, classical baselines and quantum minimum finding with tests".

### Task 2: Data tool (Sonnet, parallel with Task 1)
Files: `tools/build_data.py`, `data/champaign.json`, `tests/test_data.py` (schema, 158 blocks, totals, symmetric t_site_site diagonal 0, all times finite and positive, file under 2 MB).
Commit: "Champaign County block groups and travel times".

### Task 3: Dev harness and recorded take (Sonnet, after 1 and 2)
Copy `dev/serve.py` and `dev/bridge_shim.js` from Unlocked, retarget entry points; `tools/record_take.py` runs the sweep and the DH loop on Aer and writes `data/recorded_local.json`.
Commit: "Dev harness and local recorded take".

### Task 4: Page (Opus, after 1; mocks data until 2 lands)
Files: `index.html`, `main.js`, `main.css`, `ui/map.js`, `ui/plots.js`, `ui/cards.js`. Smoke test with Playwright against the harness: load take, run classical, run quantum sweep on the fake backend, no console errors, screenshots.
Commit: "Store placement page: map, controls, sweep plot, minimum-finding run".

### Task 5: Docs (planner)
README: problem, data and credits (Census Bureau TIGERweb, OSRM and OpenStreetMap contributors, Dürr–Høyer, Montanaro, BBHT), what the numbers mean, honesty notes (lookup oracle, quadratic speedup needs fault tolerance, demo sizes), how to run. Qollab project page text.

### Task 6: Review (fresh agent, 10 minutes cap), then Qollab (Tushar): upload, built-in simulator run, Forte 1 run for the take, publish, links to Oussama by 14:00 CT.

## Schedule (Oct 11, CT)
08:00 Tasks 1 and 2 in parallel (about 2 h). 10:00 Tasks 3 and 4 in parallel (about 2.5 h), docs alongside. 12:30 review and smoke test. 13:00 Tushar: Qollab upload and runs for both entries. 14:00 links to Oussama.

## Risks
- Gate count of the diagonal oracle at m = 12 (about 2^12 CNOT-level gates per oracle, times 2 per iteration, times up to 25 iterations) may be slow on the Qollab built-in simulator or rejected by the IonQ simulator compile step. Mitigation: m = 10 default, sweep capped at 12 circuits per series, measured in Task 1 before the page is built.
- OSRM demo server rate limits; fallback to haversine is in the tool.
- Pyodide milp unknown; brute force is the exact baseline anyway.
- Time: two entries, one person on Qollab tomorrow. The Unlocked upload comes first; this entry's Qollab step is last.
