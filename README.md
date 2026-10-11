# Store placement by quantum minimum finding

**What it is.** A Qollab page that picks where to open k stores in Champaign County, Illinois, from real census populations and road travel times, and finds the cheapest placement with Durr-Hoyer quantum minimum finding next to three classical baselines (exact brute force, greedy, local search).

**Why it matters, and for whom.** Facility location is the planner's problem behind store, clinic and warehouse networks, and the exact method scores every placement. The page counts the same unit on both sides: brute force scores N placements, the quantum loop scores about 7 sqrt N (65 of 65 recorded runs reached the exact minimum, table below). The speedup is proven in theory (Durr and Hoyer 1996); this page shows the circuits behaving as that theory predicts at the sizes a simulator allows, and claims nothing beyond that.

**How the quantum part works.** The register starts in the Dicke state over the C(m,4) feasible placements, Grover search marks every placement cheaper than a threshold, and the threshold tightens round by round (details under "Quantum part"). A theory-match sweep runs first: measured success probabilities on the sin^2((2j+1) theta) curves within 2.8 percentage points.

**How to run it.** On Qollab: Run, pick Built-in simulator (24q), press "Run quantum search"; "Load recorded take" shows a saved run at once. Local commands under "Running it". Demo video: `docs/demo.mp4` in the repo (public link added at submission).

A Qollab project for the Quantum Impact theme. Real census populations and road travel times define where to put k stores in Champaign County, Illinois. A strong classical baseline (exact brute force, greedy, local search) solves the small instance. The quantum subroutine is Durr-Hoyer minimum finding: Grover search over all placements with a threshold oracle that tightens round by round. The demo runs at 10 to 12 qubits over the N = C(m,4) feasible placements (210 or 495), the same search space the exact classical method scores, and shows three things: the measured Grover success probabilities land on the theory curve, the minimum found equals the exact answer, and the placements the quantum loop scores are counted next to the N the exact method scores, under one counting rule.

What this demo does not do: claim a speedup at this size. At 210 placements a laptop evaluates every one in a millisecond, and the oracle itself is compiled from that table. The point of the demo is that the circuits behave exactly as the theory predicts, so the proven scaling of the theory applies. See "What is proven" below.

## The problem

- 158 census block groups (2020 census) with population and centroid.
- 12 candidate store sites: the centroids of the most populous block groups, at least 4 driving minutes apart. The first m of them (m = 10 or 12) are the candidates, site i = bit i of the placement index. The search space, for both sides, is the set of feasible placements: the C(m,k) subsets of exactly k sites (210 at m = 10, k = 4).
- 2 warehouse candidates at the east and west edges of the county.
- Driving minutes between every block and site, site and site, warehouse and site (OSRM, OpenStreetMap roads).

Cost of a placement S with |S| = k (any other bit count is infeasible):

| term | formula | default weight |
|---|---|---|
| travel | population-weighted mean of the minutes from each block to its nearest open site | 1 per minute |
| staff | per open site, ceil(assigned population / capacity) employees | capacity 15,000, 3.0 per employee |
| crowding | penalty per pair of open sites closer than tau minutes | tau 10 min, 10 per pair |
| warehouse | best warehouse, sum of its minutes to the open sites | 0.1 per minute |

Defaults were chosen so the landscape is not trivial: with k = 4 and m = 10 the local search from 20 random starts finds 5 distinct local minima and greedy misses the optimum by 1.8 cost units. Every weight is a slider on the page and the classical answers update instantly.

## Classical baseline

- Brute force over the N = C(m, k) feasible placements: exact, N placements scored.
- Greedy: add the site with the best marginal cost, k times. It scores partial placements (1 to k sites), which are not entries of the table, so its count is reported in that unit and not compared.
- Local search: swap one open site for one closed site, steepest descent from 20 random feasible starts, reporting the number of distinct local minima reached and the placements scored (repeats included).

One counting rule for the comparison: placements scored. Brute force scores N. The quantum loop scores one placement for its random start, one per Grover iteration (each oracle call) and one per round for the measured sample.

## Quantum part

The register starts in the uniform superposition over the feasible placements only, the Dicke state |D(m,k)>, prepared by the deterministic circuit of Bartschi and Eidenbenz (O(km) gates, no ancilla; 246 CNOTs at m = 10). Grover's diffusion reflects about that state. So N = C(m,k) on the quantum side exactly as on the classical side. Durr and Hoyer's algorithm finds the minimum of an N-entry table with O(sqrt N) oracle calls. Each round runs a Grover search for an entry below the current threshold T (the oracle marks x with cost(x) < T), with the number of Grover iterations j chosen by the Boyer-Brassard-Hoyer-Tapp exponential schedule because the number of marked entries is unknown, and takes one measurement. If the measured placement is below T it becomes the new threshold. Placements scored are counted as the paper counts oracle calls: one for the random start, one per Grover iteration, one for scoring the measured sample, so a round costs j + 1. The loop stops after three rounds without improvement at the largest iteration budget (a project heuristic, not the paper's time bound) or after 120 rounds. Rounds, thresholds, the current best placement and the cumulative count are shown live on the page and on the map.

Before the loop, the page runs a theory-match sweep: three fixed thresholds that mark M = 1, 4 and 16 placements, Grover iterations j = 0, 1, ..., J, 1,000 shots each. The measured probability of sampling a marked placement is plotted against sin^2((2j+1) theta) with sin theta = sqrt(M/N). This is the check that the circuits do what the theory says. The sweep is one batched job, so it runs on IonQ backends too (one job per run); the Durr-Hoyer loop needs one short job per round and runs live only on local simulators, otherwise it is shown from the recorded take.

Scaling. The recorded take also contains a small study: the loop run from 20 random seeds at m = 6, 8 and 10 (and 5 seeds at m = 12 when the simulator allows), with one measurement per round. The page plots the mean number of oracle calls against sqrt N next to the line N. This is the check that the call count grows like the theory says, at the sizes we can simulate.

The oracle. The threshold oracle is a diagonal gate compiled from the classical cost table: phase -1 on every placement with cost below T. Before submission every circuit is unrolled to the gate set rz, sx, x, cx (translation only, no optimisation), because Qollab's built-in simulator accepts only standard gates; the CNOT counts above are the optimised level-1 numbers from tools/time_grover.py, the submitted level-0 circuits are larger. That is a lookup table, not an arithmetic circuit. A reversible circuit that computes the cost from 158 block populations does not fit in 29 qubits, which is the largest simulator Qollab offers. The theory and all counters on the page count oracle calls, which is the quantity the quadratic speedup is about; nothing here measures wall clock. On the shipped instance one Grover iteration (oracle, Dicke preparation twice, reflection) transpiles to 1,846 CNOTs at m = 10 and 5,286 at m = 12, so m = 12 is slow and marked as such.

## What is proven, and what is not

- Durr and Hoyer (1996): the minimum of an unsorted table of N entries is found with O(sqrt N) oracle calls, with probability at least 1 - 1/2^c in time O(c sqrt N). Any classical method needs Omega(N) calls in the worst case. The demo counts the calls on both sides.
- Montanaro (2020): branch-and-bound, the standard exact method for facility location and other mixed-integer programs, admits a near-quadratic quantum speedup in the size of the branch-and-bound tree.
- Not proven, and not claimed: any speedup on a 210-entry table on a simulator, or any wall-clock claim. The quadratic speedup needs a fault-tolerant machine and an arithmetic oracle; this page shows the algorithm behaving as predicted at the size we can simulate.

## Reading the page

1. Move the sliders. The map shows the exact best placement, the table shows the three classical methods, their costs and evaluations.
2. "Run quantum search" submits the sweep (one job). The plot shows measured points on the theory curves; the card gives the largest deviation in percentage points.
3. On a local simulator the Durr-Hoyer loop then runs round by round with one measurement per round. The map follows the current best placement, the counter shows placements scored so far against N, and the result card compares the final cost with the exact one. A single loop can stop short of the minimum (the paper guarantees success with probability at least 1/2 within its budget); the study card shows how often it reaches it.
4. "Load recorded take" shows a run recorded earlier (Aer locally; replaced by an IonQ Forte simulator run when available).

The shipped local take (Aer, m = 12, k = 4, N = C(12,4) = 495, defaults): the sweep's largest deviation from theory is 2.8 percentage points over 31 circuits at 1,000 shots. The seed-7 loop with one measurement per round reached the exact minimum (cost 62.73, sites 2, 3, 4, 7) scoring 170 placements (1 start + 128 Grover iterations + 41 rounds) against N = 495. Live runs on the page default to m = 10 (N = 210), where one Grover iteration is about three times cheaper; choose m = 12 to repeat the take's setting. The scaling study in the same take, one measurement per round, simulator seed = seed x 1000 + round:

| m | N = C(m,4) | sqrt N | seeds | mean placements scored (min..max) | reached the exact minimum |
|---|---|---|---|---|---|
| 6 | 15 | 3.9 | 20 | 27 (18..46) | 20 of 20 |
| 8 | 70 | 8.4 | 20 | 64 (39..96) | 20 of 20 |
| 10 | 210 | 14.5 | 20 | 108 (82..146) | 20 of 20 |
| 12 | 495 | 22.2 | 5 | 161 (120..207) | 5 of 5 |

Mean calls divided by sqrt N stay between 6.9 and 7.7 across the four sizes while N grows 33-fold. At m = 6 the loop scores more placements than N: the constant in front of sqrt N wins at tiny N, as the theory says it must. The stop rule (three idle rounds at the largest iteration budget) decides when to give up, so the exact-minimum fraction is a property of this implementation, not the paper's bound.

## Running it

Qollab: upload the project (index.html, main.js, main.css, ui/, qopt/, data/), pick a backend, open the page. The built-in simulator and the Aer and Braket local simulators run everything live; IonQ backends run the sweep live and show the loop from the take.

Locally:

```
pip install -r requirements-dev.txt
python -m pytest tests -q
python dev/serve.py 8765        # then open http://127.0.0.1:8765/
python tools/record_take.py     # regenerate data/recorded_local.json on Aer
python tools/build_data.py      # refetch data/champaign.json (TIGERweb + OSRM, no keys)
```

## Data and credits

- Population and centroids: US Census Bureau, TIGERweb, Census 2020 Block Groups (layer 11), Champaign County IL, 158 block groups, total population 205,865.
- Travel times: OSRM demo server (router.project-osrm.org), driving durations. Map data (c) OpenStreetMap contributors, ODbL.
- C. Durr, P. Hoyer, "A quantum algorithm for finding the minimum", arXiv:quant-ph/9607014 (1996).
- A. Bartschi, S. Eidenbenz, "Deterministic preparation of Dicke states", FCT 2019, arXiv:1904.07358.
- M. Boyer, G. Brassard, P. Hoyer, A. Tapp, "Tight bounds on quantum searching", Fortschritte der Physik 46, 493-506 (1998), arXiv:quant-ph/9605034.
- A. Montanaro, "Quantum speedup of branch-and-bound algorithms", Phys. Rev. Research 2, 013056 (2020).
- Built with Qiskit; local checks with Qiskit Aer.

MIT license. Team ZollerQu, Qollab x IonQ hackathon, October 2026.
