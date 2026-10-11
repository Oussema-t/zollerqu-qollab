# ZollerQu stock project: summary of all experiments and results (as of Sun 11 Oct 2026, 01:00 Germany)

Qollab x IonQ hackathon, team ZollerQu (2 people). Repo: github.com/Oussema-t/zollerqu-qollab, branch `stocks`,
HEAD ccbb522. The repo stays PRIVATE; the Qollab project will be self-contained. Nothing has run on IonQ yet.

## 1. Problem
Split a set of stocks into two groups so that the portfolio has low risk and decent expected return.
- Mode A "long/short": each asset is bought (+1) or bet against (-1); equal weights +-1/n.
- Mode B "buy or skip": each asset is bought (1) or left in cash (0); weight 1/n per bought asset.
- Score to minimise (training data only): J = V_p/V_ref - eta * E_p/sigma_ref, with V_p = portfolio variance
  proxy, E_p = expected-return proxy (mean weekly log return), V_ref = mean asset variance. eta in {0.25, 0.5, 1, 2};
  eta = 1 is the default.
- Balance switch (equal buys and sells): off by default; on = exactly n/2 on each side.
- One asset = one qubit. Sizes: 12 (exact answer known, used for all tuning) and 29 (IonQ limit).

## 2. Data (Task 1 and 2)
- Kaggle "Huge Stock Market Dataset" (CC0). 29 named US assets in 10 groups + SPY as market reference
  (not a qubit): Tech AAPL MSFT INTC CSCO XLK; Gold&silver GLD SLV GDX NEM; Energy XOM CVX SLB USO;
  Banks JPM BAC GS XLF; Bonds TLT IEF SHY; Healthcare JNJ PFE XLV; Staples PG KO WMT; XLU; VNQ; FCX.
- Weekly closes 2006-05-26 to 2017-11-10 (599 weeks), 598 weekly log returns.
- Chronological split: train = first 418 weeks (2006-06-02 to 2014-05-30, includes the 2008 crash),
  test = last 180 weeks (2014-06-06 to 2017-11-10). Nothing was chosen with test data.
- 12-asset selection (round-robin over groups): AAPL GLD XOM JPM TLT JNJ PG XLU VNQ FCX MSFT SLV.
- Relations between assets = Pearson correlation of training weekly returns. Bonds are the only group
  negatively correlated with the rest (mean about -0.22; TLT-XLK -0.45). Gold is near 0 (+0.10 to +0.14).
- Links kept in the graph: all 66 at 12 assets; the strongest 60 % by |correlation| (244 of 406) at 29 assets.
  (Ranking by |covariance| was tried first and dropped: it removed every bond link.)
- Also validated earlier: OR-Library Hang Seng data (31 anonymised stocks, MIT) as a second dataset.

## 3. Classical part (Task 3)
- Score -> Ising energy E(s) = sum J_ij s_i s_j + sum h_i s_i + const (exact, verified to 1e-16 with all links;
  thinned 29-asset model is a surrogate: up to 0.013 off, correlation 0.975 with the true score).
- Answer key at 12 assets: all 4,096 portfolios enumerated for every setting (exactly one optimum each).
- Classical solvers (own NumPy code): random sampling, greedy single-flip/swap descent, simulated annealing.
  Hit rate of the optimum per try, 12 assets, eta 1, balance off: random 0.0000-0.0006, greedy 0.07 (A) / 1.00 (B),
  annealing 0.26 (A) / 1.00 (B). Averaged over all 16 settings: random 0.0006, greedy 0.64, annealing 0.72.
- Balance by penalty lambda*(sum s)^2 verified by enumeration at 12 assets; lambda about 0.07-0.10 -> penalty coupling
  is 12x (A) / 5x (B) the largest objective term. At 29 assets balance is a hard constraint in the solvers.
- Held-out test (180 weeks, simple returns, no costs): all 32 settings were calmer than SPY (weekly volatility
  0.00-0.91 % vs 1.57 %) and calmer than random portfolios; 14 of 32 earned more than a random portfolio;
  NONE earned more than SPY (best +0.165 %/week vs +0.214 %). Conclusion: the method reduces risk, not profit.
- Mode B with balance off buys almost nothing at low eta (cash has zero risk); it needs balance on or high eta.

## 4. Quantum part (Task 4), all on a classical statevector simulator, 12 qubits
Circuit per layer: H on all qubits; rzz(2*gamma*J_ij/scale) per link; rz(2*gamma*h_i/scale) per asset; rx(2*beta) mixer.
scale = max(|J|,|h|). Own NumPy simulator cross-checked against Qiskit Statevector (max difference 3e-16).
Heavy runs were done on a Hetzner 8-core server in 121 s.

### 4.1 KEY FINDING: sign of the ramp
With mixer rx(+2*beta), lowering the energy needs gamma and beta of OPPOSITE sign. The LR-QAOA recipe as written
(dgamma=0.6, dbeta=0.3, same sign) steers to the WORST portfolios: quality 0.45 at p=5 vs 0.81 for random.
With dbeta<0 it works. All tuned ramps below have beta<0.

### 4.2 Main results (P = probability that ONE shot is the exact best portfolio; uniform = 0.00024)
| Method (12 qubits, eta 1, balance off)   | Mode A P(opt)/shot | Mode B P(opt)/shot |
|------------------------------------------|-------------------:|-------------------:|
| p=0 control (random)                     | 0.0002             | 0.0002             |
| LR-QAOA p=1 / 2 / 3 / 4 / 5              | 0.0011 / 0.0020 / 0.0044 / 0.0058 / **0.0073** (30x) | 0.0114 / 0.0155 / 0.0181 / **0.0207** (85x) / 0.0205 |
| Standard QAOA (free angles) p=1 / 2 / 3  | 0.0010 / 0.0023 / 0.0042 | 0.0114 / 0.0122 / 0.0041 |
| CVaR-QAOA (alpha=0.1) p=3                | 0.0066             | **0.0577** (236x)  |
| Multi-angle p=3                          | 0.0054             | 0.0057             |
| Counterdiabatic (DC) p=3                 | 0.0067             | 0.0000 (quality 0.985 but never the optimum) |
| VarQITE (McLachlan) angle training p=3   | 0.0042             | 0.0258             |
| varQITE Pauli-update (arXiv 2404.16135)  | <0.00001 (quality 0.880) | <0.00001 (quality 0.871) |
Quality = (E_max - E)/(E_max - E_min) of the sampled distribution (1 = always the best). LR-QAOA p=5: 0.950 (A);
p=4: 0.973 (B). Chance to see the optimum at least once in 1000 shots: 0.999 (A), 1.000 (B).
Most likely single portfolio: mode A = exactly the optimum (rank 1 of 4096: long AAPL MSFT GLD SLV JPM TLT PG XLU,
short XOM JNJ VNQ FCX); mode B = "buy TLT only" (optimum "buy AAPL+TLT" is rank 6).
Tuned ramps: A (0.3,-0.3) (0.5,-0.3) (0.7,-0.4) (0.7,-0.4) (0.7,-0.4) for p=1..5; B (1.0,-0.7) (1.4,-0.6)
(1.4,-0.6) (1.4,-0.5) (1.3,-0.5).

### 4.3 Optimisers (same 5 starts, cap 200*p circuit evaluations)
COBYLA, Nelder-Mead, Powell, L-BFGS-B, own SPSA. Mean quality over starts and p: A: Nelder-Mead 0.895, COBYLA 0.890,
SPSA 0.887, L-BFGS-B 0.884, Powell 0.857; B: L-BFGS-B 0.969, COBYLA 0.968, Nelder-Mead 0.968, SPSA 0.965, Powell 0.950.
The starting point matters more than the optimiser: from the tuned ramp all reach about the same quality; random
starts often get stuck. L-BFGS-B uses the fewest evaluations (24-98).

### 4.4 Balance in the circuit (P of the best BALANCED portfolio per shot, share of balanced shots)
| 12 qubits, p=2        | Mode A           | Mode B           |
|-----------------------|------------------|------------------|
| penalty in the model  | 0.0011 (97 % balanced) | 0.0011 (97 %) |
| post-selection        | 0.0019 (30 %)    | 0.0002-0.0009 (1 %) |
| XY mixer + Dicke start| **0.0192** (100 %) | **0.0068** (100 %) |
Penalty circuits mostly learn the constraint (their grids hit the small-dgamma edge). XY mixer: 12 XX+YY gates per
layer on a ring; Dicke state prepared generically (no efficient gate count claimed).

### 4.5 varQITE (IonQ+ORNL method, implemented from the description, no code reused)
Ansatz A (ry+cx per link, 66 params) and B (rzz per link + rx layer, 67 params). Pauli-term update rule with
parameter-shift Jacobian, least squares rcond 1e-2, 60 steps, lr in {0.1, 0.4, 0.8}, 7,980 simulated circuits per run.
Best: ansatz A, lr 0.8: energy 0.0833->0.0576 (A), 0.0775->0.0327 (B); quality 0.880/0.871; optimum probability <1e-5.
Facts: ansatz B at theta=0 is a stationary point (all derivatives exactly 0) so it needs a random start; ansatz A learns
slowly at lr <= 0.4. Weaker than LR-QAOA at this budget with the same 66 two-qubit gates.
Exact imaginary time on a diagonal cost = classical Boltzmann sampling (reference only): it passes the best
LR-QAOA value at tau=80 (A) / 160 (B).

### 4.6 Gate counts (one layer; measured with Qiskit transpiler, level 1)
| qubits | links | two-qubit gates all-to-all (ions) | two-qubit depth | on heavy-hex grid (SWAPs) |
|-------:|------:|----------------------------------:|----------------:|--------------------------:|
| 12     | 66    | 66                                | 15              | 285 (4.3x)                |
| 29     | 244   | 244                               | 29              | 1,576 (6.5x)              |
29-qubit circuits were built and counted only, never simulated. p=3 at 29 qubits = 732 two-qubit gates.

### 4.7 Transfer of the 12-qubit angles (quality with the 12-asset ramp, p=2)
12 assets 0.916 -> 16 assets 0.905 (fresh grid at 16: 0.910) -> 20 assets 0.919 (A); B: 0.969 -> 0.960 -> 0.968.
At 20 assets thinned to 60 %: 0.882 (A), 0.963 (B). Rule for 29 qubits: use the 12-asset ramp for the same mode
and p, untuned.

### 4.8 Deferred / not done
Warm start, variational real-time evolution, recursive QAOA, any optimiser loop on the backend.
No run on IonQ yet. README, credits, video and the Qollab main.py are still to do.

## 5. Honest conclusions
1. Under the SAME budget of 1,000 evaluations the tuned circuits find the optimum in 99-100 % of runs, restarted
   greedy in 82 % (A) / 100 % (B), a 1,000-step annealer in 47 % / 100 %, random sampling in 22 %; but the circuits
   first cost 300-10,700 offline simulated circuits to tune. Per shot they are 30-236x better than uniform, and in
   mode A their most likely portfolio is the optimum.
2. No quantum advantage is claimed: the tuning is impossible beyond ~20 qubits without hardware shots, and greedy
   with 1,000 evaluations at 12 qubits already covers a quarter of the search space.
3. The published ramp recipe has a sign problem for this convention; this is a reproducible finding.
4. The XY mixer is the right way to enforce "equal buys and sells"; penalties are not.
5. varQITE (Pauli-update) underperforms at this budget; reported, not hidden.
6. Financially: all optimised portfolios lowered risk out of sample; none beat the market's return.
7. Everything is a simulation; the IonQ noise-model run (Task 5) will show how much survives noise
   (66 two-qubit gates per layer at 12 qubits; 244 at 29).

## 6. Open questions to discuss
- Which circuits to put in the single IonQ batch (p=0 control, LR-QAOA p=1..3, CVaR p=3, XY p=2, varQITE final)?
- Expected noise impact: ~0.5 % error per two-qubit gate x 66-244 gates per layer.
- Post-selection vs XY mixer at 29 qubits (Dicke preparation cost).
- How to present "no advantage" positively on the project page (sampler quality, IonQ connectivity benefit).
