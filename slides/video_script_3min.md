# 3-minute video script (ZollerQu, stocks)

260 words. At a calm 115 words per minute plus 10 s of clicking in the demo, this takes about **2:25**, leaving about 34 s spare before the 3:00 limit.

Read only these lines; do not read the slides aloud. Move to the next slide as soon as the last word is said. Checkpoint: if you are past **1:30** when slide 8 starts, speak a little faster.

| slide | title | start | time | words |
|---|---|---|---|---|
| 1 | Portfolio Selection with QAOA | 0:00 | 8 s | 15 |
| 2 | Which assets should a fund buy? | 0:07 | 13 s | 25 |
| 3 | One qubit per asset, one Ising energy | 0:20 | 13 s | 25 |
| 4 | QAOA circuits, trained offline, one IonQ job | 0:33 | 11 s | 21 |
| 5 | 8 circuit types, 5 settings that mattered | 0:44 | 8 s | 16 |
| 6 | All-to-all links: 4.3× fewer two-qubit gates | 0:53 | 11 s | 21 |
| 7 | IonQ noise model: 29× random guessing | 1:04 | 14 s | 27 |
| 8 | 29 qubits: more layers help, only without noise | 1:18 | 16 s | 31 |
| 9 | Iceberg catches errors, but doubles the gates | 1:34 | 16 s | 30 |
| 10 | Try it on Qollab: one click, one IonQ job | 1:50 | 29 s | 36 |
| 11 | Thank you for your attention | 2:18 | 7 s | 13 |

## Slide 1: Portfolio Selection with QAOA  ·  start at 0:00

Hi, we are team ZollerQu. We pick stocks with QAOA on the IonQ noise model.

## Slide 2: Which assets should a fund buy?  ·  start at 0:07

Funds want low risk and good return. Risk comes from assets that fall together. Twelve assets give four thousand portfolios, twenty-nine give half a billion.

## Slide 3: One qubit per asset, one Ising energy  ·  start at 0:20

Each asset is one qubit: zero means buy, one means skip. Risk minus return becomes an Ising energy. The exact best: Apple plus US bonds.

## Slide 4: QAOA circuits, trained offline, one IonQ job  ·  start at 0:33

We trained fifty-eight circuits offline. Best at twelve qubits: CVaR-QAOA, three layers. At twenty-nine qubits: linear-ramp QAOA, with the angles reused.

## Slide 5: 8 circuit types, 5 settings that mattered  ·  start at 0:44

We compared eight circuit types. The mixer sign, the CVaR objective and more layers mattered most.

## Slide 6: All-to-all links: 4.3× fewer two-qubit gates  ·  start at 0:53

Trapped ions connect every pair of qubits, so one layer needs sixty-six two-qubit gates. A heavy-hex chip needs two hundred eighty-five.

## Slide 7: IonQ noise model: 29× random guessing  ·  start at 1:04

Main result: on the IonQ noise model, seven in a thousand shots were the exact best portfolio, twenty-nine times better than random. Without noise it is fifty-eight.

## Slide 8: 29 qubits: more layers help, only without noise  ·  start at 1:18

With twenty-nine assets, three layers find the best in seven percent of runs, classical search in sixty-one. Sixteen layers reach ninety percent, but only without noise. No quantum advantage is claimed.

## Slide 9: Iceberg catches errors, but doubles the gates  ·  start at 1:34

We also tried Iceberg error detection. On the IonQ noise model it flagged three in four shots. The kept shots were not clearly better, because the code doubles the gates.

## Slide 10: Try it on Qollab: one click, one IonQ job  ·  start at 1:50

*Before recording: run once on Qollab so a finished result is already on screen. In the video, open main.py, open Run → Select QPU → IonQ Forte 1 (29q) noise model, then scroll the finished output. Do not wait for a new job (about 30 s).*

[SWITCH TO QOLLAB] This is our project. The code is in main.py. We press Run, pick the IonQ Forte 1 noise model, and get results and plots. [BACK TO SLIDES] Classical methods still match us. Next: more Iceberg layers and shorter circuits.

## Slide 11: Thank you for your attention  ·  start at 2:18

Thank you for your attention. This effort is supported by Qollab and IonQ.
