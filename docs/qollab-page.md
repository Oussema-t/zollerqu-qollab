# Qollab project page text

**Title:** Store placement by quantum minimum finding

**One line:** Real census populations and road times decide where k stores go; Durr-Hoyer minimum finding searches the placements with Grover's algorithm, and the measured success probabilities land on the theory curve.

**Description:**
Pick a public dataset: 158 census block groups of Champaign County, Illinois (US Census Bureau TIGERweb, 2020) and OSRM driving minutes between blocks, 12 candidate store sites and 2 warehouses. Build a strong classical baseline: exact brute force, greedy and local search over a cost that mixes travel time, staffing, crowding and warehouse trips, all adjustable with sliders. Then the quantum subroutine: Durr-Hoyer minimum finding, a Grover search with a threshold that tightens every round, on 10 qubits live (N = C(10,4) = 210 feasible placements) and 12 qubits in the shipped take (N = 495), always the same search space the exact classical method scores (the register starts in a Dicke state).

The page shows three things. A theory-match sweep: for 1, 4 and 16 marked placements the measured probability of sampling a marked one follows sin^2((2j+1) theta) as the iteration count j grows. The loop: thresholds, rounds and cumulative oracle calls animate on the map until the minimum equals the exact answer. The count: one unit on both sides, placements scored (start, Grover iterations, one measurement per round): 170 next to the N = 495 that brute force scores in the shipped take; a 20-seed study at m = 6, 8, 10, 12 shows the mean count growing like 7 times sqrt N while N grows 33-fold.

What is proven: Durr and Hoyer (1996) find the minimum of N entries with O(sqrt N) oracle calls, against Omega(N) classically; Montanaro (2020) gives a near-quadratic speedup for branch-and-bound, the standard exact method for facility location. What is not claimed: any speedup at 210 placements, or wall clock. The oracle is a lookup table compiled from the classical costs, stated openly on the page, because an arithmetic cost circuit over 158 blocks does not fit in 29 qubits.

Backends: built-in, Aer and Braket simulators run everything live. IonQ backends accept one job per run, so they run the sweep live and show the loop from the recorded take.

**Tags:** optimization, Grover, minimum finding, facility location, census data, Quantum Impact
