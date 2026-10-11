# Qollab project page text

**Title:** Store placement by quantum minimum finding

**One line:** Real census populations and road times decide where k stores go; Durr-Hoyer minimum finding searches the placements with Grover's algorithm, and the measured success probabilities land on the theory curve.

**Description:**
Pick a public dataset: 158 census block groups of Champaign County, Illinois (US Census Bureau TIGERweb, 2020) and OSRM driving minutes between blocks, 12 candidate store sites and 2 warehouses. Build a strong classical baseline: exact brute force, greedy and local search over a cost that mixes travel time, staffing, crowding and warehouse trips, all adjustable with sliders; the cost is one number per open site plus one per open pair, in multiples of 1/8. Then the quantum subroutine: Durr-Hoyer minimum finding, a Grover search with a threshold that tightens every round, over the C(m,k) placements of k stores opened among m candidate locations (C(8,4) = 70 by default), always the same search space the exact classical method scores (the site register starts in a Dicke state). The oracle computes each placement's cost inside the circuit: it adds each open site's cost and each open pair's penalty into an arithmetic register and flips the phase on the sign bit after subtracting the threshold; 20 qubits at m = 8.

The page shows three things. A theory-match sweep: for 1, 4 and 16 marked placements the measured probability of sampling a marked one follows sin^2((2j+1) theta) as the iteration count j grows. The loop: thresholds, rounds and cumulative oracle calls animate on the map until the minimum equals the exact answer. The count: one unit on both sides, placements scored (start, Grover iterations, one measurement per round): 46 next to the 70 that brute force scores in the shipped take; a study at m = 6, 8 and 10 shows the mean count growing like 7 times the square root of C(m,4) while C(m,4) grows 14-fold.

What is proven: Durr and Hoyer (1996) find the minimum of C(m,k) values with O(sqrt C(m,k)) oracle calls, against a number proportional to C(m,k) classically; Montanaro (2020) gives a near-quadratic speedup for branch-and-bound, the standard exact method for facility location. What is not claimed: any speedup at 70 placements, or wall clock. The oracle is an arithmetic circuit built from the per-site and per-pair numbers, never from a list of placement costs; the cost model is a quadratic approximation shared by both sides.

Backends: the Aer and Braket local simulators run everything live at m = 8 (about 2 s per Grover iteration on Aer); the built-in simulator runs it too, slowly (about 2 minutes per iteration at m = 8). IonQ backends accept one job per run, so they run the sweep live and show the loop from the recorded take.

**Tags:** optimization, Grover, minimum finding, facility location, census data, Quantum Impact
