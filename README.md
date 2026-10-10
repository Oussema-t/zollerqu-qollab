# Store placement by quantum minimum finding

A Qollab project for the Quantum Impact theme. Real census populations and road travel times define where to put k stores in Champaign County, Illinois. A strong classical baseline (exact brute force, greedy, local search) solves the small instance. The quantum subroutine is Durr-Hoyer minimum finding: Grover search over all placements with a threshold oracle that tightens round by round. The demo runs at 10 qubits (1,024 placements) and shows three things: the measured Grover success probabilities land on the theory curve, the minimum found equals the exact answer, and the number of oracle calls used is counted next to the N evaluations the exact classical method needs.

What this demo does not do: claim a speedup at this size. At 1,024 placements a laptop evaluates every one in a millisecond. The point of the demo is that the circuits behave exactly as the theory predicts, so the proven scaling of the theory applies. See "What is proven" below.

## The problem

- 158 census block groups (2020 census) with population and centroid.
- 12 candidate store sites: the centroids of the most populous block groups, at least 4 driving minutes apart. The first m of them (m = 10 or 12) form the search space, site i = bit i of the placement index.
- 2 warehouse candidates at the east and west edges of the county.
- Driving minutes between every block and site, site and site, warehouse and site (OSRM, OpenStreetMap roads).

Cost of a placement S with |S| = k (any other bit count is infeasible):

| term | formula | default weight |
|---|---|---|
| travel | population-weighted mean of the minutes from each block to its nearest open site | 1 per minute |
| staff | per open site, ceil(assigned population / capacity) employees | capacity 15,000, 3.0 per employee |
| crowding | penalty per pair of open sites closer than tau minutes | tau 10 min, 10 per pair |
| warehouse | best warehouse, sum of its minutes to the open sites | 0.1 per minute |

Defaults were chosen so the landscape is not trivial: with k = 4 and m = 10 the local search from 30 random starts finds 5 distinct local minima and greedy misses the optimum by 1.8 cost units. Every weight is a slider on the page and the classical answers update instantly.

## Classical baseline

- Brute force over all C(m, k) feasible placements: exact, N = 2^m oracle evaluations in the same counting as the quantum part (the oracle also sees infeasible strings).
- Greedy: add the site with the best marginal cost, k times.
- Local search: swap one open site for one closed site, steepest descent from random feasible starts, reporting the number of distinct local minima reached.

## Quantum part

Durr and Hoyer's algorithm finds the minimum of an N-entry table with O(sqrt N) oracle calls. Each round runs a Grover search for an entry below the current threshold T (the oracle marks x with cost(x) < T), with the number of Grover iterations chosen by the Boyer-Brassard-Hoyer-Tapp exponential schedule because the number of marked entries is unknown. A sample below T becomes the new threshold. The rounds, thresholds and cumulative oracle calls are shown live on the page and on the map.

Before the loop, the page runs a theory-match sweep: three fixed thresholds that mark M = 1, 4 and 16 placements, Grover iterations j = 0, 1, ..., J, 1,000 shots each. The measured probability of sampling a marked placement is plotted against sin^2((2j+1) theta) with sin theta = sqrt(M/N). This is the check that the circuits do what the theory says. The sweep is one batched job, so it runs on IonQ backends too (one job per run); the Durr-Hoyer loop needs many short jobs and runs live only on local simulators, otherwise it is shown from the recorded take.

The oracle. The threshold oracle is a diagonal gate compiled from the classical cost table: phase -1 on every placement with cost below T. That is a lookup table, not an arithmetic circuit. A reversible circuit that computes the cost from 158 block populations does not fit in 29 qubits, which is the largest simulator Qollab offers. The theory and all counters on the page count oracle calls, which is the quantity the quadratic speedup is about; nothing here measures wall clock. The compiled oracle costs about 1,350 CNOTs per Grover iteration at m = 10 and 4,660 at m = 12, so m = 12 is slow and marked as such.

## What is proven, and what is not

- Durr and Hoyer (1996): the minimum of an unsorted table of N entries is found with O(sqrt N) oracle calls, with probability at least 1 - 1/2^c in time O(c sqrt N). Any classical method needs Omega(N) calls in the worst case. The demo counts the calls on both sides.
- Montanaro (2020): branch-and-bound, the standard exact method for facility location and other mixed-integer programs, admits a near-quadratic quantum speedup in the size of the branch-and-bound tree.
- Not proven, and not claimed: any speedup on a 1,024-entry table on a simulator, or any wall-clock claim. The quadratic speedup needs a fault-tolerant machine and an arithmetic oracle; this page shows the algorithm behaving as predicted at the size we can simulate.

## Reading the page

1. Move the sliders. The map shows the exact best placement, the table shows the three classical methods, their costs and evaluations.
2. "Run quantum search" submits the sweep (one job). The plot shows measured points on the theory curves; the card gives the largest deviation in percentage points.
3. On a local simulator the Durr-Hoyer loop then runs round by round. The map follows the current best placement, the counter shows oracle calls so far against N, and the result card compares the final cost with the exact one.
4. "Load recorded take" shows a run recorded earlier (Aer locally; replaced by an IonQ Forte simulator run when available).

The shipped local take (Aer, m = 10, defaults, seed 7): the sweep's largest deviation from theory is 2.5 percentage points over 33 circuits at 1,000 shots; the Durr-Hoyer loop reached the exact minimum (cost 62.73, sites 2, 3, 4, 7) in 27 rounds with 107 oracle calls against N = 1,024.

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
- M. Boyer, G. Brassard, P. Hoyer, A. Tapp, "Tight bounds on quantum searching", Fortschritte der Physik 46, 493-506 (1998), arXiv:quant-ph/9605034.
- A. Montanaro, "Quantum speedup of branch-and-bound algorithms", Phys. Rev. Research 2, 013056 (2020).
- Built with Qiskit; local checks with Qiskit Aer.

MIT license. Team ZollerQu, Qollab x IonQ hackathon, October 2026.
