# Store placement by quantum minimum finding

**What it is.** A Qollab page that picks where to open k stores among m candidate locations in Champaign County, Illinois, from real census populations and road travel times, and finds the cheapest placement with Durr-Hoyer quantum minimum finding next to three classical baselines (exact brute force, greedy, local search).

**Why it matters, and for whom.** Facility location is the planner's problem behind store, clinic and warehouse networks, and the exact method scores every placement. The page counts the same unit on both sides: brute force scores all C(m,k) placements, the quantum loop scores about 7 times the square root of C(m,k) (27 of 27 recorded runs reached the exact minimum, table below). The speedup is proven in theory (Durr and Hoyer 1996); this page shows the circuits behaving as that theory predicts at the sizes a simulator allows, and claims nothing beyond that.

**How the quantum part works.** The site register starts in the Dicke state over the C(m,4) placements, a threshold oracle computes each placement's cost inside the circuit and marks every placement cheaper than the threshold, Grover search amplifies those, and the threshold tightens round by round (details under "Quantum part"). A theory-match sweep runs first: measured success probabilities on the sin^2((2j+1) theta) curves within 5.3 percentage points.

**How to run it.** On Qollab: Run, pick a backend, press "Run quantum search"; "Load recorded take" shows a saved run at once. The in-circuit oracle needs 20 qubits at m = 8, and Qollab's built-in simulator (BasicSimulator) takes about 138 s per Grover iteration there, so a live run is practical on the Aer or Braket local simulators and slow on the built-in one (numbers under "Quantum part"). Local commands under "Running it". Demo video: `docs/demo.mp4` in the repo (public link added at submission).

A Qollab project for the Quantum Impact theme. Real census populations and road travel times define where to open k stores among m candidate locations in Champaign County, Illinois. A strong classical baseline (exact brute force, greedy, local search) solves the small instance. The quantum subroutine is Durr-Hoyer minimum finding: Grover search over the placements with a threshold oracle that computes the cost in the circuit and tightens round by round. The demo runs on 20 to 22 qubits over the C(m,4) placements (70 at m = 8, 210 at m = 10), the same search space the exact classical method scores, and shows three things: the measured Grover success probabilities land on the theory curve, the minimum found equals the exact answer, and the placements the quantum loop scores are counted next to the C(m,k) the exact method scores, under one counting rule.

What this demo does not do: claim a speedup at this size. At 70 placements a laptop scores every one in a millisecond, and a simulated Grover iteration takes seconds. The point of the demo is that the circuits behave exactly as the theory predicts, so the proven scaling of the theory applies. See "What is proven" below.

## The problem

- 158 census block groups (2020 census) with population and centroid.
- 12 candidate store sites: the centroids of the most populous block groups, at least 4 driving minutes apart. The first m of them (m = 8, 10 or 12) are the m candidate locations, site i = bit i of the placement index. The search space, for both sides, is the C(m,k) placements of exactly k stores opened (70 at m = 8, k = 4).
- 2 warehouse candidates at the east and west edges of the county.
- Driving minutes between every block and site, site and site, warehouse and site (OSRM, OpenStreetMap roads).

The cost of a placement S of k stores opened is quadratic: one number per open site plus one number per open pair,

    cost(S) = sum over i in S of a_i  +  sum over pairs i < j in S of b_ij

| term | per open site a_i | per open pair b_ij | default weight |
|---|---|---|---|
| travel | minus G_i, the population-weighted minutes site i saves against the farthest of the m candidate locations | plus H_ij, the savings sites i and j share (counted twice otherwise) | 1 per minute |
| staff | ceil(population of the blocks nearest to i among the m candidate locations / capacity) employees | | capacity 15,000, 3.0 per employee |
| crowding | | penalty when the two sites are closer than tau minutes | tau 10 min, 10 per pair |
| warehouse | minutes from site i to its nearest warehouse | | 0.1 per minute |

The travel terms are the second-order inclusion-exclusion of "minutes to the nearest open site": exact for one or two open sites, an approximation for four. The constant part of the travel term (the mean minutes to the farthest candidate, 27.375 at m = 8) is the same for every placement and is reported, not searched over. Every coefficient is rounded to the nearest 1/8, so the costs on the page are exact multiples of 1/8 (11.375, 62.5), and the classical baselines and the quantum oracle add the same integers (eighths): their minimum agrees exactly.

At the default weights this landscape is easy for the heuristics: at m = 8 and m = 10 greedy finds the exact optimum and local search from 20 random starts sees a single local minimum. Every weight is a slider on the page and the classical answers update instantly.

## Classical baseline

- Brute force over the C(m,k) placements: exact, all C(m,k) placements scored.
- Greedy: add the site with the best marginal cost, k times. It scores partial placements (1 to k sites), so its count is reported in that unit and not compared.
- Local search: swap one open site for one closed site, steepest descent from 20 random starts, reporting the number of distinct local minima reached and the placements scored (repeats included).

One counting rule for the comparison: placements scored. Brute force scores all C(m,k). The quantum loop scores one placement for its random start, one per Grover iteration (each oracle call) and one per round for the measured sample.

## Quantum part

The site register starts in the uniform superposition over the C(m,k) placements only, the Dicke state |D(m,k)>, prepared by the deterministic circuit of Bartschi and Eidenbenz (O(km) gates, no ancilla). Grover's diffusion reflects about that state, so the quantum side searches exactly the C(m,k) placements the classical side scores. Durr and Hoyer's algorithm finds the minimum of C(m,k) values with O(sqrt C(m,k)) oracle calls. Each round runs a Grover search for a placement below the current threshold T (the oracle marks x with cost(x) < T), with the number of Grover iterations j chosen by the Boyer-Brassard-Hoyer-Tapp exponential schedule because the number of marked placements is unknown, and takes one measurement. If the measured placement is below T it becomes the new threshold. Placements scored are counted as the paper counts oracle calls: one for the random start, one per Grover iteration, one for scoring the measured sample, so a round costs j + 1. The loop stops after three rounds without improvement at the largest iteration budget (a project heuristic, not the paper's time bound) or after 120 rounds. Rounds, thresholds, the current best placement and the cumulative count are shown live on the page and on the map.

Before the loop, the page runs a theory-match sweep: three fixed thresholds that mark M = 1, 4 and 16 placements, Grover iterations j = 0, 1, ..., J, 1,000 shots each. The measured probability of sampling a marked placement is plotted against sin^2((2j+1) theta) with sin theta = sqrt(M / C(m,k)). This is the check that the circuits do what the theory says. The sweep is one batched job, so it runs on IonQ backends too (one job per run); the Durr-Hoyer loop needs one short job per round and runs live only on local simulators, otherwise it is shown from the recorded take.

Scaling. The recorded take also contains a small study: the loop run from random seeds at m = 6, 8 and 10 with one measurement per round. The page plots the mean number of placements scored against the square root of C(m,4) next to the line C(m,4). This is the check that the call count grows like the theory says, at the sizes we can simulate.

The oracle. The threshold oracle computes the cost of the placement in the circuit from the m per-site integers A_i and the per-pair integers B_ij, never from a list of costs. Registers: the m site qubits, an accumulator of n qubits holding a two's-complement integer (n from the coefficient bounds: 11 at m = 8 and m = 10), and one AND ancilla. Compute: Hadamards put the accumulator in the Fourier basis (the QFT of zero); each A_i is added with phase rotations controlled on site qubit i (Draper's adder, no carries); for each pair with B_ij nonzero, a Toffoli writes "site i and site j open" into the ancilla, B_ij is added controlled on it, and a second Toffoli clears it; the threshold is subtracted; an inverse QFT returns the accumulator to binary. It now holds cost - T in eighths, and its top bit is 1 exactly when the cost is below the threshold. A Z on that bit flips the phase, and the inverse of the compute step returns the accumulator and the ancilla to zero. That is m + n + 1 qubits: 20 at m = 8, 22 at m = 10, 25 at m = 12 (above the built-in simulator's 24). Before submission every circuit is unrolled to the gate set rz, sx, x, cx (translation only, no optimisation), because Qollab's built-in simulator accepts only standard gates. On the shipped instance one Grover iteration (oracle compute and uncompute, Dicke preparation twice, reflection) is 8,185 gates (2,716 CNOTs) at m = 8 and 12,153 gates (4,064 CNOTs) at m = 10 as submitted. Measured with tools/time_grover.py, one iteration takes 138 s at m = 8 and about 24 minutes (1,446 s) at m = 10 on qiskit's BasicSimulator (the built-in simulator), and 1.8 s and 6.7 s on Aer. The page therefore defaults to m = 8. The theory and all counters on the page count oracle calls, which is the quantity the quadratic speedup is about; nothing here measures wall clock.

## What is proven, and what is not

- Durr and Hoyer (1996): the minimum of an unsorted list of C(m,k) values is found with O(sqrt C(m,k)) oracle calls, with probability at least 1 - 1/2^c in time O(c sqrt C(m,k)). Any classical method needs a number of calls proportional to C(m,k) in the worst case. The demo counts the calls on both sides.
- Montanaro (2020): branch-and-bound, the standard exact method for facility location and other mixed-integer programs, admits a near-quadratic quantum speedup in the size of the branch-and-bound tree.
- The oracle here is an arithmetic circuit: it adds the per-site and per-pair integers of the quadratic cost in the circuit, with m + n + 1 qubits and a gate count that grows with the number of sites and pairs, not with the number of placements. The cost model is the quadratic approximation above, the same on both sides. Not proven, and not claimed: any speedup on 70 placements on a simulator, or any wall-clock claim. The quadratic speedup needs a fault-tolerant machine; this page shows the algorithm behaving as predicted at the size we can simulate.

## Reading the page

1. Move the sliders. The map shows the exact best placement, the table shows the three classical methods, their costs and evaluations.
2. "Run quantum search" submits the sweep (one job). The plot shows measured points on the theory curves; the card gives the largest deviation in percentage points.
3. On a local simulator the Durr-Hoyer loop then runs round by round with one measurement per round. The map follows the current best placement, the counter shows placements scored so far against the C(m,k) brute force scores, and the result card compares the final cost with the exact one. A single loop can stop short of the minimum (the paper guarantees success with probability at least 1/2 within its budget); the study card shows how often it reaches it.
4. "Load recorded take" shows a run recorded earlier (Aer locally; replaced by an IonQ Forte simulator run when available).

The shipped local take (Aer, m = 8, k = 4, C(8,4) = 70 placements, default weights): the sweep's largest deviation from theory is 5.3 percentage points over 17 circuits at 1,000 shots, at M = 16, j = 2 (41.8% measured against 36.5%); the exact statevector of that same circuit gives 36.5%, so the gap is sampling noise at about 3.5 standard errors, and the M = 1 and M = 4 series stay within 1.1 and 2.4 points. The seed-7 loop with one measurement per round reached the exact minimum (cost 11.375, sites 2, 3, 4, 6) scoring 46 placements (1 start + 25 Grover iterations + 20 rounds) against the 70 brute force scores. The scaling study in the same take, one measurement per round, simulator seed = seed x 1000 + round; fewer seeds at m = 8 and m = 10 because a simulated Grover iteration with the in-circuit oracle takes seconds:

| m | C(m,4) placements | sqrt C(m,4) | seeds | mean placements scored (min..max) | reached the exact minimum |
|---|---|---|---|---|---|
| 6 | 15 | 3.9 | 20 | 27.5 (17..36) | 20 of 20 |
| 8 | 70 | 8.4 | 5 | 56 (51..61) | 5 of 5 |
| 10 | 210 | 14.5 | 2 | 104 (87..121) | 2 of 2 |

Mean placements scored divided by the square root of C(m,4) stay between 6.7 and 7.2 across the three sizes while C(m,4) grows 14-fold. At m = 6 the loop scores more placements than the 15 brute force scores: the constant in front of the square root wins at tiny sizes, as the theory says it must. The stop rule (three idle rounds at the largest iteration budget) decides when to give up, so the exact-minimum fraction is a property of this implementation, not the paper's bound. The study is small (5 seeds at m = 8, 2 at m = 10); recording it took 21 minutes on Aer.

## Running it

Qollab: upload the project (index.html, main.js, main.css, ui/, qopt/, data/), pick a backend, open the page. The Aer and Braket local simulators run everything live at m = 8; the built-in simulator runs it too but slowly (see "Quantum part"); IonQ backends run the sweep live and show the loop from the take.

Locally:

```
pip install -r requirements-dev.txt
python -m pytest tests -q
python dev/serve.py 8765        # then open http://127.0.0.1:8765/
python tools/record_take.py     # regenerate data/recorded_local.json on Aer
python tools/time_grover.py     # qubits, gates and seconds per Grover iteration
python tools/build_data.py      # refetch data/champaign.json (TIGERweb + OSRM, no keys)
```

## Data and credits

- Population and centroids: US Census Bureau, TIGERweb, Census 2020 Block Groups (layer 11), Champaign County IL, 158 block groups, total population 205,865.
- Travel times: OSRM demo server (router.project-osrm.org), driving durations. Map data (c) OpenStreetMap contributors, ODbL.
- C. Durr, P. Hoyer, "A quantum algorithm for finding the minimum", arXiv:quant-ph/9607014 (1996).
- A. Bartschi, S. Eidenbenz, "Deterministic preparation of Dicke states", FCT 2019, arXiv:1904.07358.
- M. Boyer, G. Brassard, P. Hoyer, A. Tapp, "Tight bounds on quantum searching", Fortschritte der Physik 46, 493-506 (1998), arXiv:quant-ph/9605034.
- T. G. Draper, "Addition on a quantum computer", arXiv:quant-ph/0008033 (2000).
- A. Montanaro, "Quantum speedup of branch-and-bound algorithms", Phys. Rev. Research 2, 013056 (2020).
- Built with Qiskit; local checks with Qiskit Aer.

MIT license. Team ZollerQu, Qollab x IonQ hackathon, October 2026.
