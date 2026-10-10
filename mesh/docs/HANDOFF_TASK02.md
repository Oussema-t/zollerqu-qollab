# Handoff for writing the Task 02 prompt (ZollerQu mesh project)

## Short version
- **Project:** split a simulation mesh between two compute devices of different speed (target work share α, e.g. 0.8 for a GPU assumed 4× faster), written as a cut + balance Ising problem and solved with LR-QAOA on IonQ (Qollab, Forte 1 noise-model simulator). The PDE solver stays classical. No speed-up, energy or "beats METIS" claims.
- **Done:** run-settings cell (small/large benchmark switch), Task 01 (L-plate mesh + cell graph), Task 01b (four benchmark mesh cases). All checks pass; the notebook runs top to bottom in about 35 s.
- **Next, Task 02:** group the triangles of each mesh case into n regions (one region = one qubit later), with region work and region-boundary weights, every region one connected piece, plus plots. No Ising, partitioning or QAOA yet.

## 1. Workspace and workflow
- Folder `~/Downloads/zollerqu_mesh_workspace/`, notebook `mesh_idea_ZollerQu.ipynb`, outputs in `results/`.
- One task per code cell, wrapped in markers `# TASK_XX_IMPLEMENTATION_CELL_START` / `# TASK_XX_IMPLEMENTATION_CELL_END`, with a Markdown description cell (goal, why, acceptance checks, scope limit) right before it.
- Claude Code inserts the new description cell and code cell after the last cell, runs the notebook for real, reports the actual printed output, fixes failures without weakening checks, and stops after the task. Each cell ends with a printed block headed `TASK 0X RESULT`.
- Kernel: project venv, Python 3.12, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2. **networkx is not installed** (Task 02 does not need it; Task 04's Kernighan-Lin will).
- Use only numpy, scipy and matplotlib, so the code can later move into `mesh/coarsen.py` and run on Qollab (Pyodide).

## 2. Current notebook cells
| # | Cell |
|---|---|
| 0 | Intro / guardrails (Markdown) |
| 1–2 | Run settings: description + code (`RUN_SETTINGS_CELL_START/END`) |
| 3–5 | Task 01: description, code, review checklist |
| 6–7 | Task 01b: description, code |

## 3. Variables available to Task 02 (do not rename or overwrite)
**Run settings (cell 2):**
- `RUN_MODE` ("small" | "large"), `RUN = RUN_MODES[RUN_MODE]` with keys `region_counts`, `seeds`, `qaoa_layers`, `shots`, `quantum_cases`.
  - small: `region_counts=[12]`, `seeds=[7]`
  - large: `region_counts=[12, 16, 20, 24]`, `seeds=[7, 8, 9]` (24 = labelled stress test)
- `BENCHMARK_CASES = ["1 control", "2 main", "3 hard", "4 structural"]`
- `DEVICE_SCENARIOS`, `TARGET_SHARES = {"2 equal devices": 0.5, "CPU + GPU (GPU 4x)": 0.8}` (speeds are assumptions)
- `RUN_THREE_DEVICE_STRETCH = False`, `THREE_DEVICE_SCENARIO`
- `IONQ_SIMULATOR_MAX_QUBITS = 29`, `LOCAL_STATEVECTOR_MAX_QUBITS = 20`, `EXACT_BRUTE_FORCE_MAX_QUBITS = 22`, `STRESS_TEST_FROM_QUBITS = 24`

**Task 01 (cell 4), main mesh, seed 7:** `points` (897×2), `triangles` (1670×3, counter-clockwise), `cell_areas`, `cell_centroids`, `adjacency_pairs` (2444×2 cell pairs sharing a full edge), `adjacency_weights` (shared-edge length), `adjacency_matrix` (symmetric weighted CSR).

**Task 01b (cell 7):**
- `mesh_cases[name]` for each name in `BENCHMARK_CASES` is a dict with `points`, `triangles`, `cell_areas`, `cell_centroids`, `adjacency_pairs`, `adjacency_weights`, `adjacency_matrix`, `n_components`, `n_isolated`, `euler`, `exact_area`, `boundary_length`, `exact_perimeter`, `refinement`, `min_angle`, `fingerprint`.
- `MESH_CASES[name]` is the case spec (hull, cutouts, loops, n_holes, h_min, h_max, features).
- `build_case_mesh(case_spec, seed) -> (points, triangles)` and `check_case_mesh(name, case_spec, points, triangles) -> dict` (same keys as above) build any case for another seed.
- Case 2 is bit-identical to the Task 01 mesh.

| Case | Domain | Refinement | Cells | Adjacency edges |
|---|---|---|---|---|
| 1 control | rectangle 2×1 | none (uniform) | 1784 | 2608 |
| 2 main | L-plate ([0,2]² minus [1,2]²) | heat source, strength 1.0 at (0.5, 0.5) | 1670 | 2444 |
| 3 hard | L-plate | sources 1.0 at (0.5, 0.5) and 0.5 at (1.55, 0.5) | 1802 | 2641 |
| 4 structural | rectangle 2×1 with a hole (r = 0.2 at (1, 0.5), meshed as a 70-gon) | around the hole edge | 2166 | 3160 |

## 4. Task 02 requirements (from the project spec + review)
- **Input:** every case in `BENCHMARK_CASES`, for every `n` in `RUN["region_counts"]` and every seed in `RUN["seeds"]`. Seed 7 uses the existing `mesh_cases`. For other seeds, rebuild the mesh with `build_case_mesh` and check it with `check_case_mesh`. Recommended: use the same seed for the mesh and for k-means.
- **Method:** k-means on cell centroids with `scipy.cluster.vq.kmeans2(data, k, minit="++", rng=np.random.default_rng(seed))`. In SciPy 1.18 the argument is `rng`; avoid the old `seed=` argument.
- **Empty clusters:** kmeans2 can return empty clusters. Handle them explicitly, e.g. re-run with the next rng draw or split the largest region, so that exactly n non-empty regions result. Report what was done.
- **Connectivity repair:** a k-means cluster can fall into two or more pieces, e.g. around the L's inner corner or around the hole. Find pieces with `scipy.sparse.csgraph.connected_components` on each region's cell subgraph. Merge every non-largest piece into the neighbouring region with which it shares the most cell edges. Repeat until every region is one piece. Count and report repairs; never hide them.
- **Cell work = 1 per triangle**, so region work `a_i` = number of triangles. The uneven work comes from mesh density. This keeps Kernighan-Lin refinement valid later, because KL preserves cell counts, not weighted work.
- **Region boundary weight** `w_ij` = number of shared triangle edges between regions i and j (the communication count). Also store the summed shared-edge length as an alternative.

**Outputs for later tasks:** a dict `region_cases[(case_name, n, seed)]` containing:
- `cell_region` (cell → region index 0..n-1)
- `region_work` (n,)
- `region_pairs` (m×2, i < j)
- `region_boundary_count` (m,)
- `region_boundary_length` (m,)
- `region_centroids` (n×2)
- `n_repaired_pieces`
- `n_empty_fixed`
- the mesh arrays used (or a reference to them)

**Acceptance checks (assertions):**
- exactly n non-empty regions; every cell in exactly one region
- every region one connected piece; the region graph connected
- sum of `region_work` = number of cells
- sum of `region_boundary_count` + edges inside regions = total cell-adjacency edges
- no self-pairs, i < j
- reproducible: same seed gives the same `cell_region`

**Diagnostics per (case, n, seed):**
- region work min / mean / max and max/mean ratio
- number of region adjacencies and the fraction of all n(n−1)/2 pairs that are adjacent (the balance term will couple all pairs anyway)
- total cut edges if every region boundary were cut
- repairs and empty-cluster fixes

**Plot:**
- One figure per n with a panel per case (seed 7): cells coloured by region, region borders drawn in black, region index written at each region centroid (colour alone cannot separate 12–24 regions).
- Save as `results/regions_n{n}_seed7.png`.
- Small mode gives one figure (n = 12); large mode gives four.

**Scope limit:** coarsening only. No Ising/QUBO, partitioning, baselines, performance model, QAOA or IonQ. Do not edit earlier cells.

## 5. Decisions already made (keep consistent)
- Two devices for the main results; three device types only as a switchable stretch goal.
- Device speeds are assumptions; Phase 2 (after the hackathon) measures them on the team's HPC CPUs and GPUs.
- Qubits = regions. Main results at 12–16, recorded runs up to 20, 24 only as a stress test, hard cap 29.
- "Stronger source → finer mesh" is a stated modelling assumption. Case 4 computes no stress; it is only meshed the way a stress solver would need.

## 6. For later tasks (context only, not Task 02)
- **QAOA sign:** with `rzz(2γJ)`, `rz(2γh)`, `rx(2β)` and γ, β > 0, the circuit pushes towards the maximum of H. Use `rx(-2β)` and verify that the noiseless mean energy is below the random-guess mean.
- **Scaling:** scale J and h by the same factor (the max of all |J| and |h|).
- **Define before results:** approximation ratio r = (E_max − E)/(E_max − E_min) from brute force, a fixed "balanced" tolerance ε, and P(opt) counting both degenerate optima at α = 0.5.
- **Gate counts:** check that `rzz` survives transpilation for the Qollab IonQ backend. For the heavy-hex comparison, include `rzz` in the basis and build the coupling map explicitly, so only SWAP overhead is compared.
- **Prior art to cite:**
  - IonQ+Ansys arXiv 2503.13128 and IonQ+Synopsys arXiv 2603.15515 partition graphs to reduce fill-in in sparse direct solvers, with balanced splits only and no open code. The 14.6% figure comes mostly from simulation; hardware was used only at 36 qubits.
  - arXiv 2408.04097 (quantum annealing for HPC load balancing).
  - Our angle: open, reproducible, unequal device shares, measured gate counts.

## 7. Timeline
Behind the spec schedule (Tasks 1–4 were due Sat 19:00 Germany time). Keep Task 02 compact. The priority after Task 04 is the first live IonQ run (Tasks 05–06).
