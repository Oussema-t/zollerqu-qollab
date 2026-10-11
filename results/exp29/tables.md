### Baseline: the ramp tuned at 12 assets, as in the paper

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| ramp tuned at 12 (as used in the paper), p=1 | 1 | 406 | 0 | 3.62e-05 | 3.6 % | 30.4 % | 0.9650 |
| ramp tuned at 12 (as used in the paper), p=2 | 2 | 812 | 0 | 6.50e-05 | 6.3 % | 47.8 % | 0.9730 |
| ramp tuned at 12 (as used in the paper), p=3 | 3 | 1,218 | 0 | 6.83e-05 | 6.6 % | 49.5 % | 0.9733 |
| ramp tuned at 12 (as used in the paper), p=4 | 4 | 1,624 | 0 | 1.04e-04 | 9.8 % | 64.5 % | 0.9762 |
| ramp tuned at 12 (as used in the paper), p=5 | 5 | 2,030 | 0 | 7.99e-05 | 7.7 % | 55.0 % | 0.9736 |

### Factor 1: ramp re-tuned at 29 assets (3 layers)

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| ramp grid at 29, ORACLE best P(opt) (2.4, -0.45) | 3 | 1,218 | 30 | 1.97e-04 | 17.9 % | 86.1 % | 0.9692 |
| ramp grid at 29, lowest CVaR 0.1 (2.4, -0.45) | 3 | 1,218 | 30 | 1.97e-04 | 17.9 % | 86.1 % | 0.9692 |
| ramp grid at 29, lowest expectation (1.7, -0.6) | 3 | 1,218 | 30 | 1.04e-04 | 9.9 % | 64.7 % | 0.9748 |

### Factor 2: optimiser over all 6 angles (3 layers, 40 circuits, start = ramp tuned at 12)

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| COBYLA, ramp tuned at 12, objective expectation, seed 2026 | 3 | 1,218 | 40 | 1.40e-04 | 13.1 % | 75.3 % | 0.9782 |
| L-BFGS-B, ramp tuned at 12, objective expectation, seed 2026 | 3 | 1,218 | 40 | 1.27e-04 | 11.9 % | 71.8 % | 0.9765 |
| Nelder-Mead, ramp tuned at 12, objective expectation, seed 2026 | 3 | 1,218 | 40 | 1.22e-04 | 11.5 % | 70.5 % | 0.9774 |
| SPSA, ramp tuned at 12, objective expectation, seed 2026 | 3 | 1,218 | 39 | 9.68e-05 | 9.2 % | 62.0 % | 0.9740 |
| SPSA, ramp tuned at 12, objective expectation, seed 2027 | 3 | 1,218 | 39 | 3.16e-05 | 3.1 % | 27.1 % | 0.9655 |

### Factor 3: tuning objective

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| COBYLA, ramp tuned at 12, objective cvar_0.1, seed 2026 | 3 | 1,218 | 40 | 1.67e-04 | 15.3 % | 81.1 % | 0.9748 |

### Factor 4: finite shots per tuning circuit

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| COBYLA, ramp tuned at 12, objective shots_1000, seed 2026 | 3 | 1,218 | 38 | 1.16e-04 | 11.0 % | 68.7 % | 0.9762 |
| COBYLA, ramp tuned at 12, objective shots_1000, seed 2027 | 3 | 1,218 | 40 | 1.15e-04 | 10.9 % | 68.5 % | 0.9754 |
| COBYLA, ramp tuned at 12, objective shots_10000, seed 2026 | 3 | 1,218 | 40 | 1.11e-04 | 10.5 % | 67.0 % | 0.9765 |

### Factor 5: initialisation (random start instead of the ramp)

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| COBYLA, random start, seed 1, objective expectation, seed 2026 | 3 | 1,218 | 40 | 7.37e-05 | 7.1 % | 52.2 % | 0.9807 |
| COBYLA, random start, seed 2, objective expectation, seed 2026 | 3 | 1,218 | 40 | 8.00e-05 | 7.7 % | 55.1 % | 0.9727 |

### Factor 6: depth (ramp tuned at 12 assets, no tuning at 29)

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| ramp tuned at 12 on the wide grid (dgamma up to 3.0), p=5 | 5 | 2,030 | 0 | 1.02e-04 | 9.7 % | 64.1 % | 0.9761 |
| ramp tuned at 12 on the original grid (dgamma up to 1.5), p=6 | 6 | 2,436 | 0 | 6.12e-05 | 5.9 % | 45.8 % | 0.9723 |
| ramp tuned at 12 on the wide grid (dgamma up to 3.0), p=6 | 6 | 2,436 | 0 | 2.08e-04 | 18.8 % | 87.5 % | 0.9804 |
| ramp tuned at 12 on the original grid (dgamma up to 1.5), p=8 | 8 | 3,248 | 0 | 1.12e-04 | 10.6 % | 67.4 % | 0.9772 |
| ramp tuned at 12 on the wide grid (dgamma up to 3.0), p=8 | 8 | 3,248 | 0 | 4.66e-04 | 37.2 % | 99.1 % | 0.9838 |
| ramp tuned at 12 on the original grid (dgamma up to 1.5), p=10 | 10 | 4,060 | 0 | 1.93e-04 | 17.6 % | 85.5 % | 0.9804 |
| ramp tuned at 12 on the wide grid (dgamma up to 3.0), p=10 | 10 | 4,060 | 0 | 6.88e-04 | 49.8 % | 99.90 % | 0.9851 |
| ramp tuned at 12 on the original grid (dgamma up to 1.5), p=12 | 12 | 4,872 | 0 | 3.03e-04 | 26.2 % | 95.2 % | 0.9827 |
| ramp tuned at 12 on the wide grid (dgamma up to 3.0), p=12 | 12 | 4,872 | 0 | 9.70e-04 | 62.1 % | > 99.99 % | 0.9871 |
| ramp tuned at 12 on the deep grid, p=16 | 16 | 6,496 | 0 | 2.09e-03 | 87.7 % | > 99.99 % | 0.9894 |
| ramp tuned at 12 on the original grid (dgamma up to 1.5), p=16 | 16 | 6,496 | 0 | 6.03e-04 | 45.3 % | 99.76 % | 0.9854 |
| ramp tuned at 12 on the wide grid (dgamma up to 3.0), p=16 | 16 | 6,496 | 0 | 2.29e-03 | 89.9 % | > 99.99 % | 0.9896 |
| ramp tuned at 12 on the deep grid, p=20 | 20 | 8,120 | 0 | 3.43e-03 | 96.8 % | 100 % | 0.9908 |
| ramp tuned at 12 on the deep grid, p=24 | 24 | 9,744 | 0 | 5.42e-03 | 99.56 % | 100 % | 0.9926 |
| ramp tuned at 12 on the deep grid, p=32 | 32 | 12,992 | 0 | 1.03e-02 | > 99.99 % | 100 % | 0.9947 |

### Combination: 16 layers + ramp re-tuned at 29

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| ramp grid at 29, ORACLE best P(opt) (3.2, -0.5) | 16 | 6,496 | 9 | 3.06e-03 | 95.3 % | 100 % | 0.9893 |
| ramp grid at 29, ORACLE best P(opt) (4.0, -0.5) (picked over both grids) | 16 | 6,496 | 18 | 5.32e-03 | 99.52 % | 100 % | 0.9892 |
| ramp grid at 29, lowest CVaR 0.1 (3.2, -0.5) | 16 | 6,496 | 9 | 3.06e-03 | 95.3 % | 100 % | 0.9893 |
| ramp grid at 29, lowest CVaR 0.1 (4.0, -0.5) (picked over both grids) | 16 | 6,496 | 18 | 5.32e-03 | 99.52 % | 100 % | 0.9892 |
| ramp grid at 29, lowest expectation (3.2, -0.6) | 16 | 6,496 | 9 | 2.94e-03 | 94.8 % | 100 % | 0.9902 |
| ramp grid at 29, lowest expectation (3.6, -0.7) (picked over both grids) | 16 | 6,496 | 18 | 3.32e-03 | 96.4 % | 100 % | 0.9907 |

### Robustness: the long/short problem (ramp tuned at 12, no tuning at 29)

| Circuit | Layers | Two-qubit gates | 29-qubit circuits spent on tuning | P(best) per shot | Best found in 1,000 shots | in 10,000 | Quality |
|---|---|---|---|---|---|---|---|
| long/short, ramp tuned at 12, p=3 [long_short] | 3 | 1,218 | 0 | 7.74e-07 | 0.1 % | 0.8 % | 0.9464 |
| long/short, ramp tuned at 12, p=8 [long_short] | 8 | 3,248 | 0 | 3.66e-05 | 3.6 % | 30.7 % | 0.9703 |
| long/short, ramp tuned at 12, p=16 [long_short] | 16 | 6,496 | 0 | 4.67e-04 | 37.3 % | 99.1 % | 0.9826 |
| long/short, ramp tuned at 12, p=24 [long_short] | 24 | 9,744 | 0 | 3.92e-03 | 98.0 % | 100 % | 0.9889 |
| long/short, ramp tuned at 12, p=32 [long_short] | 32 | 12,992 | 0 | 1.11e-02 | > 99.99 % | 100 % | 0.9914 |

### Classical baselines, buy or skip (200 runs per budget)

| Budget (evaluations) | Random sampling | Greedy descent (restarted) | Simulated annealing (one run) |
|---|---|---|---|
| 100 | 0.0 % | 0.0 % | 1.5 % |
| 1,000 | 0.0 % | 61.0 % | 61.0 % |
| 10,000 | 0.0 % | 100 % | 100 % |
| 100,000 | 0.0 % | 100 % | 100 % |

### Classical baselines, long/short (200 runs per budget)

| Budget (evaluations) | Random sampling | Greedy descent (restarted) | Simulated annealing (one run) |
|---|---|---|---|
| 100 | 0.0 % | 0.0 % | 0.0 % |
| 1,000 | 0.0 % | 1.5 % | 3.5 % |
| 10,000 | 0.0 % | 19.5 % | 42.0 % |
| 100,000 | 0.0 % | 89.5 % | 98.0 % |

