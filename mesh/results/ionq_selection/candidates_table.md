# IonQ circuit candidates (case 2, 12 qubits, seed 7)

Existing 12-qubit results only (no new training). Columns:
- **P(opt)** and **P(≤5 %)**: noiseless probability per shot of the optimum and of a split within 5 % of the optimum.
- **2q**: two-qubit gates measured by transpiling to IonQ Forte 1 (QIS gate set as submitted; rzz and cx = 1 native gate each).
- **native ZZ**: qiskit-ionq's local native compile, which splits each rzz into 2 ZZ (upper bound; IonQ's own compiler maps rzz to one ZZ).
- **crude**: noiseless P(opt) × 0.995^(2q) (*crude estimate*).
- **Aer**: density matrix with 0.5 % two-qubit and 0.05 % one-qubit depolarizing noise and 0.5 % readout flips (*approximate noise model, not IonQ*; shortlist only).
- **shots to solution** = 1 / P(opt) (noiseless / crude).
- ★ = in the batch, ◆ = fallback, † = starts from a classical answer (warm start). No circuit uses a generic state preparation; the starts are H on every qubit or one RY per qubit.

## 2 equal devices

Classical samples to solution (median over 100 runs, evaluations until the optimum first appears): simulated annealing **1,181**, local search **1,111**. Random guessing needs 1 / 4.88e-04 = 2,048 samples; the target-share sampler needs 2,048.

| | circuit | 2q | depth (2q depth) | native ZZ | P(opt) | P(≤5 %) | crude P(opt) | Aer P(opt) | shots to solution (noiseless / crude) |
|---|---|---|---|---|---|---|---|---|---|
| ★◆ | varQITE ZY (paper, all pairs) | 66 | 34 (21) | 132 | 0.7501 | 0.7501 | 0.5388 | 0.5492 | 1 / 2 |
| ★ | varQITE A (all pairs) | 66 | 33 (21) | 66 | 0.7057 | 0.7057 | 0.5070 | 0.4943 | 1 / 2 |
| † | warm-start QAOA (rounded spectral, mixer A) + COBYLA, p = 2 | 132 | 29 (22) | 264 | 0.0124 | 0.0124 | 0.0064 | 0.0086 | 81 / 156 |
| † | warm-start QAOA (rounded spectral, mixer A) + COBYLA, p = 1 | 66 | 15 (11) | 132 | 0.0084 | 0.0084 | 0.0060 | 0.0070 | 119 / 166 |
| † | warm-start QAOA (rounded spectral, mixer A), fixed: LR-QAOA angles, p = 1 | 66 | 15 (11) | 132 | 0.0069 | 0.0069 | 0.0049 | 0.0058 | 146 / 203 |
| † | warm-start QAOA (rounded spectral, mixer A), fixed: LR-QAOA angles, p = 2 | 132 | 29 (22) | 264 | 0.0075 | 0.0075 | 0.0038 | 0.0055 | 134 / 260 |
| † | warm-start QAOA (rounded spectral, mixer A), initial state only, p = 0 | 0 | 1 (0) | 0 | 0.0035 | 0.0035 | 0.0035 | – | 284 / 284 |
| † | warm-start QAOA (rounded spectral, mixer B), initial state only, p = 0 | 0 | 1 (0) | 0 | 0.0035 | 0.0035 | 0.0035 | – | 284 / 284 |
|  | symmetry-reduced QAOA + COBYLA, p = 2 | 110 | 27 (22) | 220 | 0.0022 | 0.0022 | 0.0013 | – | 460 / 799 |
|  | QAOA + L-BFGS-B, p = 2 | 132 | 25 (22) | 264 | 0.0023 | 0.0023 | 0.0012 | – | 435 / 844 |
|  | QAOA + Nelder-Mead, p = 2 | 132 | 25 (22) | 264 | 0.0023 | 0.0023 | 0.0012 | – | 435 / 844 |
|  | QAOA + Powell, p = 2 | 132 | 25 (22) | 264 | 0.0023 | 0.0023 | 0.0012 | – | 441 / 855 |
|  | CVaR-QAOA + SPSA, p = 1 | 66 | 13 (11) | 132 | 0.0016 | 0.0016 | 0.0012 | – | 614 / 855 |
|  | CVaR-QAOA + COBYLA, p = 1 | 66 | 13 (11) | 132 | 0.0016 | 0.0016 | 0.0012 | – | 615 / 856 |
| † | warm-start QAOA (relaxation, mixer A) + COBYLA, p = 2 | 132 | 25 (22) | 264 | 0.0022 | 0.0022 | 0.0011 | – | 452 / 875 |
|  | QAOA + COBYLA, p = 2 | 132 | 25 (22) | 264 | 0.0022 | 0.0022 | 0.0011 | – | 452 / 875 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 2 | 132 | 25 (22) | 264 | 0.0021 | 0.0021 | 0.0011 | – | 475 / 922 |
|  | symmetry-reduced QAOA + COBYLA, p = 1 | 55 | 14 (11) | 110 | 0.0014 | 0.0014 | 0.0011 | – | 710 / 936 |
|  | QAOA + COBYLA, p = 1 | 66 | 13 (11) | 132 | 0.0015 | 0.0015 | 0.0010 | – | 689 / 960 |
|  | QAOA + L-BFGS-B, p = 1 | 66 | 13 (11) | 132 | 0.0015 | 0.0015 | 0.0010 | – | 690 / 960 |
|  | QAOA + Powell, p = 1 | 66 | 13 (11) | 132 | 0.0015 | 0.0015 | 0.0010 | – | 690 / 960 |
|  | QAOA + Nelder-Mead, p = 1 | 66 | 13 (11) | 132 | 0.0015 | 0.0015 | 0.0010 | – | 690 / 960 |
| † | warm-start QAOA (relaxation, mixer A) + COBYLA, p = 1 | 66 | 13 (11) | 132 | 0.0014 | 0.0014 | 0.0010 | – | 690 / 960 |
|  | QAOA + SPSA, p = 1 | 66 | 13 (11) | 132 | 0.0014 | 0.0014 | 0.0010 | – | 691 / 962 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 3 | 198 | 37 (33) | 396 | 0.0027 | 0.0027 | 0.0010 | – | 368 / 994 |
|  | CVaR-QAOA + SPSA, p = 2 | 132 | 25 (22) | 264 | 0.0019 | 0.0019 | 0.0010 | – | 513 / 995 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 4 | 264 (>200) | 49 (44) | 528 | 0.0035 | 0.0035 | 9.3e-04 | – | 285 / 1,071 |
|  | symmetry-reduced QAOA, fixed schedule, p = 1 | 55 | 14 (11) | 110 | 0.0012 | 0.0012 | 9.3e-04 | – | 819 / 1,079 |
| † | warm-start QAOA (relaxation, mixer A), fixed: LR-QAOA angles, p = 1 | 66 | 13 (11) | 132 | 0.0013 | 0.0013 | 9.2e-04 | – | 778 / 1,083 |
|  | LR-QAOA, fixed schedule, p = 1 | 66 | 13 (11) | 132 | 0.0013 | 0.0013 | 9.2e-04 | – | 778 / 1,083 |
|  | QAOA + SPSA, p = 2 | 132 | 25 (22) | 264 | 0.0017 | 0.0017 | 8.7e-04 | – | 593 / 1,149 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 5 | 330 (>200) | 61 (55) | 660 | 0.0044 | 0.0044 | 8.5e-04 | – | 226 / 1,181 |
|  | CVaR-QAOA + COBYLA, p = 2 | 132 | 25 (22) | 264 | 0.0016 | 0.0016 | 8.4e-04 | – | 613 / 1,187 |
|  | symmetry-reduced QAOA, fixed schedule, p = 2 | 110 | 27 (22) | 220 | 0.0014 | 0.0014 | 7.8e-04 | – | 741 / 1,285 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 1 | 66 | 13 (11) | 132 | 0.0010 | 0.0010 | 7.3e-04 | – | 979 / 1,363 |
| † | warm-start QAOA (relaxation, mixer A), fixed: LR-QAOA angles, p = 2 | 132 | 25 (22) | 264 | 0.0014 | 0.0014 | 7.3e-04 | – | 705 / 1,366 |
|  | LR-QAOA, fixed schedule, p = 2 | 132 | 25 (22) | 264 | 0.0014 | 0.0014 | 7.3e-04 | – | 705 / 1,366 |
|  | LR-QAOA, fixed schedule, p = 3 | 198 | 37 (33) | 396 | 0.0016 | 0.0016 | 6.0e-04 | – | 613 / 1,655 |
| † | warm-start QAOA (relaxation, mixer A), initial state only, p = 0 | 0 | 1 (0) | 0 | 4.9e-04 | 4.9e-04 | 4.9e-04 | – | 2,048 / 2,048 |
| ★ | p = 0 control (uniform) | 0 | 1 (0) | 0 | 4.9e-04 | 4.9e-04 | 4.9e-04 | 4.9e-04 | 2,048 / 2,048 |
|  | LR-QAOA, fixed schedule, p = 4 | 264 (>200) | 49 (44) | 528 | 0.0018 | 0.0018 | 4.8e-04 | – | 558 / 2,096 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 2 | 132 | 25 (22) | 264 | 9.1e-04 | 9.1e-04 | 4.7e-04 | – | 1,097 / 2,126 |
|  | varQITE B (IonQ-native, graph edges) | 22 | 14 (12) | 44 | 4.0e-04 | 4.0e-04 | 3.6e-04 | – | 2,484 / 2,773 |
|  | LR-QAOA, fixed schedule, p = 5 | 330 (>200) | 61 (55) | 660 | 0.0019 | 0.0019 | 3.6e-04 | – | 532 / 2,780 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 3 | 198 | 37 (33) | 396 | 3.4e-04 | 3.4e-04 | 1.2e-04 | – | 2,968 / 8,007 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 4 | 264 (>200) | 49 (44) | 528 | 9.0e-05 | 9.0e-05 | 2.4e-05 | – | 11,090 / 41,653 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 5 | 330 (>200) | 61 (55) | 660 | 1.2e-04 | 1.2e-04 | 2.3e-05 | – | 8,375 / 43,790 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 1 | 66 | 13 (11) | 132 | 5.9e-06 | 5.9e-06 | 4.2e-06 | – | 169,827 / 236,420 |
|  | varQITE A (graph edges) | 22 | 20 (12) | 22 | 1.9e-14 | 1.9e-14 | 1.7e-14 | – | 53,120,219,148,471 / 59,313,299,208,313 |
|  | varQITE A+ (graph edges, ry layer) | 22 | 20 (12) | 22 | 1.8e-15 | 1.8e-15 | 1.6e-15 | – | 545,216,690,494,941 / 608,781,387,108,102 |
| † | warm-start QAOA (rounded spectral, mixer B), fixed: recovery angles, p = 1 | 66 | 14 (11) | 132 | 3.0e-34 | 3.0e-34 | 2.2e-34 | – | 3,320,222,702,044,273,603,673,706,108,813,312 / 4,622,148,406,064,330,596,282,265,199,706,112 |
| † | warm-start QAOA (rounded spectral, mixer B) + COBYLA, p = 1 | 66 | 14 (11) | 132 | 3.0e-34 | 3.0e-34 | 2.2e-34 | – | 3,320,222,702,044,273,603,673,706,108,813,312 / 4,622,148,406,064,330,596,282,265,199,706,112 |
| † | warm-start QAOA (rounded spectral, mixer B), fixed: recovery angles, p = 2 | 132 | 25 (22) | 264 | 3.0e-34 | 3.0e-34 | 1.6e-34 | – | 3,320,222,702,044,273,603,673,706,108,813,312 / 6,434,585,208,555,130,389,692,406,722,199,552 |
| † | warm-start QAOA (rounded spectral, mixer B) + COBYLA, p = 2 | 132 | 25 (22) | 264 | 3.0e-34 | 3.0e-34 | 1.6e-34 | – | 3,320,222,702,044,273,603,673,706,108,813,312 / 6,434,585,208,555,130,389,692,406,722,199,552 |

## CPU + GPU (GPU 4x)

Classical samples to solution (median over 100 runs, evaluations until the optimum first appears): simulated annealing **338**, local search **122**. Random guessing needs 1 / 2.44e-04 = 4,096 samples; the target-share sampler needs 931.

| | circuit | 2q | depth (2q depth) | native ZZ | P(opt) | P(≤5 %) | crude P(opt) | Aer P(opt) | shots to solution (noiseless / crude) |
|---|---|---|---|---|---|---|---|---|---|
| ★† | warm-start QAOA (rounded spectral, mixer B), fixed: recovery angles, p = 1 | 66 | 14 (11) | 132 | 1.0000 | 1.0000 | 0.7183 | 0.7298 | 1 / 1 |
| † | warm-start QAOA (rounded spectral, mixer B) + COBYLA, p = 1 | 66 | 14 (11) | 132 | 1.0000 | 1.0000 | 0.7183 | 0.7298 | 1 / 1 |
| † | warm-start QAOA (rounded spectral, mixer B), fixed: recovery angles, p = 2 | 132 | 25 (22) | 264 | 1.0000 | 1.0000 | 0.5160 | 0.5721 | 1 / 2 |
| † | warm-start QAOA (rounded spectral, mixer B) + COBYLA, p = 2 | 132 | 29 (22) | 264 | 1.0000 | 1.0000 | 0.5160 | 0.5655 | 1 / 2 |
| † | warm-start QAOA (rounded spectral, mixer A), fixed: LR-QAOA angles, p = 1 | 66 | 15 (11) | 132 | 0.1303 | 0.1304 | 0.0936 | 0.0993 | 8 / 11 |
| † | warm-start QAOA (rounded spectral, mixer A) + COBYLA, p = 2 | 132 | 29 (22) | 264 | 0.1739 | 0.1740 | 0.0897 | 0.1054 | 6 / 11 |
| † | warm-start QAOA (rounded spectral, mixer A) + COBYLA, p = 1 | 66 | 15 (11) | 132 | 0.1199 | 0.1201 | 0.0861 | – | 8 / 12 |
| † | warm-start QAOA (rounded spectral, mixer A), fixed: LR-QAOA angles, p = 2 | 132 | 29 (22) | 264 | 0.1431 | 0.1432 | 0.0738 | – | 7 / 14 |
| † | warm-start QAOA (rounded spectral, mixer A), initial state only, p = 0 | 0 | 1 (0) | 0 | 0.0317 | 0.0317 | 0.0317 | – | 32 / 32 |
| ★ | CVaR-QAOA + SPSA, p = 2 | 132 | 27 (22) | 264 | 0.0058 | 0.0112 | 0.0030 | 0.0047 | 173 / 335 |
|  | QAOA + Powell, p = 2 | 132 | 27 (22) | 264 | 0.0054 | 0.0104 | 0.0028 | – | 187 / 362 |
|  | QAOA + L-BFGS-B, p = 2 | 132 | 27 (22) | 264 | 0.0054 | 0.0104 | 0.0028 | – | 187 / 362 |
|  | QAOA + Nelder-Mead, p = 2 | 132 | 27 (22) | 264 | 0.0054 | 0.0104 | 0.0028 | – | 187 / 362 |
|  | QAOA + COBYLA, p = 2 | 132 | 27 (22) | 264 | 0.0053 | 0.0104 | 0.0028 | – | 187 / 363 |
|  | QAOA + SPSA, p = 2 | 132 | 27 (22) | 264 | 0.0051 | 0.0100 | 0.0026 | – | 195 / 377 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 2 | 132 | 27 (22) | 264 | 0.0049 | 0.0096 | 0.0025 | – | 204 / 395 |
|  | CVaR-QAOA + SPSA, p = 1 | 66 | 14 (11) | 132 | 0.0034 | 0.0067 | 0.0025 | – | 293 / 408 |
|  | QAOA + L-BFGS-B, p = 1 | 66 | 14 (11) | 132 | 0.0033 | 0.0065 | 0.0024 | – | 300 / 417 |
|  | QAOA + Powell, p = 1 | 66 | 14 (11) | 132 | 0.0033 | 0.0065 | 0.0024 | – | 300 / 417 |
|  | QAOA + COBYLA, p = 1 | 66 | 14 (11) | 132 | 0.0033 | 0.0065 | 0.0024 | – | 300 / 417 |
|  | QAOA + Nelder-Mead, p = 1 | 66 | 14 (11) | 132 | 0.0033 | 0.0065 | 0.0024 | – | 300 / 417 |
|  | QAOA + SPSA, p = 1 | 66 | 14 (11) | 132 | 0.0033 | 0.0065 | 0.0024 | – | 303 / 421 |
|  | LR-QAOA, fixed schedule, p = 2 | 132 | 27 (22) | 264 | 0.0046 | 0.0089 | 0.0024 | – | 218 / 423 |
|  | CVaR-QAOA + COBYLA, p = 2 | 132 | 27 (22) | 264 | 0.0046 | 0.0089 | 0.0024 | – | 219 / 425 |
|  | CVaR-QAOA + COBYLA, p = 1 | 66 | 14 (11) | 132 | 0.0031 | 0.0061 | 0.0022 | – | 322 / 449 |
|  | LR-QAOA, fixed schedule, p = 1 | 66 | 14 (11) | 132 | 0.0031 | 0.0060 | 0.0022 | – | 326 / 454 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 3 | 198 | 40 (33) | 396 | 0.0058 | 0.0114 | 0.0022 | – | 171 / 463 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 1 | 66 | 14 (11) | 132 | 0.0030 | 0.0058 | 0.0021 | – | 336 / 468 |
|  | LR-QAOA, fixed schedule, p = 3 | 198 | 40 (33) | 396 | 0.0053 | 0.0102 | 0.0019 | – | 190 / 513 |
| † | warm-start QAOA (relaxation, mixer A) + COBYLA, p = 1 | 66 | 15 (11) | 132 | 0.0026 | 0.0051 | 0.0018 | – | 392 / 545 |
| † | warm-start QAOA (relaxation, mixer A) + COBYLA, p = 2 | 132 | 29 (22) | 264 | 0.0035 | 0.0070 | 0.0018 | – | 285 / 552 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 4 | 264 (>200) | 53 (44) | 528 | 0.0065 | 0.0127 | 0.0017 | – | 153 / 574 |
|  | LR-QAOA, fixed schedule, p = 4 | 264 (>200) | 53 (44) | 528 | 0.0058 | 0.0112 | 0.0015 | – | 172 / 647 |
|  | LR-QAOA (success-calibrated), fixed schedule, p = 5 | 330 (>200) | 66 (55) | 660 | 0.0074 | 0.0142 | 0.0014 | – | 136 / 710 |
| † | warm-start QAOA (relaxation, mixer A), fixed: LR-QAOA angles, p = 1 | 66 | 15 (11) | 132 | 0.0017 | 0.0034 | 0.0012 | – | 581 / 808 |
|  | LR-QAOA, fixed schedule, p = 5 | 330 (>200) | 66 (55) | 660 | 0.0061 | 0.0118 | 0.0012 | – | 165 / 864 |
| † | warm-start QAOA (relaxation, mixer A), initial state only, p = 0 | 0 | 1 (0) | 0 | 0.0011 | 0.0021 | 0.0011 | – | 931 / 931 |
| † | warm-start QAOA (relaxation, mixer A), fixed: LR-QAOA angles, p = 2 | 132 | 29 (22) | 264 | 0.0020 | 0.0039 | 0.0010 | – | 503 / 975 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 1 | 66 | 14 (11) | 132 | 0.0013 | 0.0025 | 9.1e-04 | – | 789 / 1,098 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 4 | 264 (>200) | 53 (44) | 528 | 0.0034 | 0.0067 | 9.1e-04 | – | 293 / 1,101 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 3 | 198 | 40 (33) | 396 | 0.0024 | 0.0048 | 8.9e-04 | – | 415 / 1,119 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 5 | 330 (>200) | 66 (55) | 660 | 0.0044 | 0.0086 | 8.4e-04 | – | 229 / 1,196 |
|  | LR-QAOA (paper default 0.6/0.3), fixed schedule, p = 2 | 132 | 27 (22) | 264 | 0.0016 | 0.0032 | 8.3e-04 | – | 618 / 1,198 |
| ★ | p = 0 control (uniform) | 0 | 1 (0) | 0 | 2.4e-04 | 4.9e-04 | 2.4e-04 | 2.4e-04 | 4,096 / 4,096 |
|  | varQITE B (IonQ-native, graph edges) | 22 | 14 (12) | 44 | 1.8e-05 | 2.3e-05 | 1.6e-05 | – | 55,718 / 62,214 |
| † | warm-start QAOA (rounded spectral, mixer B), initial state only, p = 0 | 0 | 1 (0) | 0 | 6.0e-08 | 4.4e-05 | 6.0e-08 | – | 16,777,216 / 16,777,216 |
|  | varQITE ZY (paper, all pairs) | 66 | 34 (21) | 132 | 4.0e-13 | 6.3e-08 | 2.9e-13 | 1.2e-05 | 2,488,318,888,147 / 3,464,038,474,150 |
|  | varQITE A+ (graph edges, ry layer) | 22 | 19 (12) | 22 | 2.6e-19 | 2.7e-17 | 2.3e-19 | – | 3,836,537,347,280,879,616 / 4,283,824,337,529,818,112 |
|  | varQITE A (graph edges) | 22 | 18 (12) | 22 | 1.8e-20 | 4.1e-16 | 1.6e-20 | – | 55,713,740,616,434,565,120 / 62,209,189,272,368,275,456 |
|  | varQITE A (all pairs) | 66 | 33 (21) | 66 | 1.8e-21 | 1.6e-20 | 1.3e-21 | – | 543,631,891,087,733,358,592 / 756,800,824,634,382,745,600 |

