# Claude Code Prompt — Task 01 only: Generate the PDE mesh and cell graph

You are working in the project folder `zollerqu_mesh_workspace`. Read `README.md` and `mesh_partitioning.ipynb` first.

## Your task
Implement **only Task 01** in the code cell marked `TASK_01_IMPLEMENTATION_CELL_START` / `TASK_01_IMPLEMENTATION_CELL_END` in `mesh_partitioning.ipynb`. Do not implement later tasks and do not restructure the project.

Build a deterministic triangular mesh and its cell-adjacency graph for the mesh-partitioning hackathon project.

### Requirements
1. Model a 2D **L-shaped plate** for a heat-transfer example. Define a heat-source location and create a non-uniform distribution of mesh points with a higher density near the source.
2. Use `numpy` and `scipy.spatial.Delaunay`; use a fixed seed (`7`) so repeated runs reproduce the same mesh.
3. Generate approximately **1,000–3,000 triangles**.
4. Keep triangles strictly inside the actual L-shaped domain. Do not rely only on a triangle-centroid test: check geometry robustly enough to avoid triangles spanning the missing corner of the L-shape. If needed, add boundary points and filter triangles by vertices, edge midpoints, and centroid; explain the chosen validation briefly in a code comment.
5. Treat each triangle as one computational cell. Two triangles are adjacent when they share a complete edge. Build an adjacency representation without requiring NetworkX in this first cell. Store the shared-edge length as the edge weight (also note that unit weights could be used later).
6. Check for isolated cells and graph connectivity. Print useful, truthful diagnostics: fixed seed, point count, triangle count, adjacency-edge count, connected-component count, isolated-cell count, and min/mean/max triangle area.
7. Plot the mesh and visibly mark the heat-source location. Save the figure to `results/mesh_seed7.png` and display it in the notebook. Make the plot readable, with equal axis scaling and labels.
8. Keep the code self-contained in this one notebook code cell except for imports. Create the `results/` directory if it does not exist. Use only NumPy, SciPy, Matplotlib, and Python standard-library modules for this task.
9. Use descriptive variable names and short comments. Add explicit assertions for key invariants (positive triangle areas, no retained triangle outside the domain, expected triangle-count range, and no isolated cells if the chosen mesh design guarantees connectivity). If filtering causes the mesh count or connectivity checks to fail, adjust the point-generation strategy rather than silently weakening the checks.
10. At the end, print a compact summary headed `TASK 01 RESULT` so a person can immediately see if the cell succeeded.

### Constraints / project scope
- This project partitions a mesh for a **classical PDE solver**. The quantum algorithm, if later added, optimizes the allocation only; it does not solve the PDE.
- This task is just mesh + graph generation. Do not implement QAOA, Ising/QUBO, partitioning, a PDE time solver, HPC access, performance or energy claims.
- Do not invent benchmark results. Only print metrics computed by the code.
- Do not edit shared project files outside this project folder.

### Required workflow
1. Inspect the notebook structure and locate the exact marked Task 01 code cell.
2. Replace only the placeholder inside that cell; preserve the preceding task-description Markdown cell.
3. Validate syntax and execute the cell if notebook execution is available. If you cannot execute it, say so explicitly and provide the reason; do not claim it passed.
4. Check that the plot file is generated and report the actual printed summary.
5. Stop after Task 01. Do not start Task 02.
