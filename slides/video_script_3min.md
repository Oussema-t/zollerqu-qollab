# 3-minute video script (ZollerQu, stocks)

275 words. At a calm 115 words per minute plus 10 s of clicking in the demo, this takes about **2:33**, leaving about 27 s spare before the 3:00 limit.

Read only these lines; do not read the slides aloud. Move to the next slide as soon as the last word is said. Checkpoint: if you are past **1:30** when slide 8 starts, speak a little faster.

| slide | title | start | time | words |
|---|---|---|---|---|
| 1 | Portfolio Selection with QAOA | 0:00 | 9 s | 18 |
| 2 | Which assets should a fund buy? | 0:09 | 15 s | 28 |
| 3 | One qubit per asset, one Ising energy | 0:24 | 15 s | 28 |
| 4 | QAOA circuits, trained offline, one IonQ job | 0:38 | 13 s | 24 |
| 5 | 8 circuit types, 5 settings that mattered | 0:51 | 9 s | 18 |
| 6 | All-to-all links: 4.3× fewer two-qubit gates | 1:00 | 13 s | 25 |
| 7 | IonQ noise model: 29× random guessing | 1:13 | 15 s | 29 |
| 8 | 29 qubits: more layers help, only without noise | 1:28 | 16 s | 31 |
| 9 | Iceberg catches errors, but doubles the gates | 1:44 | 15 s | 28 |
| 10 | Try it on Qollab: one click, one IonQ job | 1:59 | 27 s | 33 |
| 11 | Thank you for your attention | 2:26 | 7 s | 13 |

## Slide 1: Portfolio Selection with QAOA  ·  start at 0:00

Hi, we're team ZollerQu. Which stocks should you buy? We answer that with QAOA, on IonQ's noise model.

## Slide 2: Which assets should a fund buy?  ·  start at 0:09

Every fund wants low risk and good returns, but some assets fall together. And the choices explode: four thousand portfolios for twelve assets, half a billion for twenty-nine.

## Slide 3: One qubit per asset, one Ising energy  ·  start at 0:24

Each asset gets one qubit: zero means buy, one means skip. Risk minus return becomes an Ising energy, and the lowest energy wins: here, Apple plus US bonds.

## Slide 4: QAOA circuits, trained offline, one IonQ job  ·  start at 0:38

We trained fifty-eight circuits offline. At twelve qubits, CVaR-QAOA with three layers won. At twenty-nine, we use linear-ramp QAOA with angles tuned at twelve.

## Slide 5: 8 circuit types, 5 settings that mattered  ·  start at 0:51

Out of eight circuit types, three things mattered most: the mixer sign, the CVaR objective, and more layers.

## Slide 6: All-to-all links: 4.3× fewer two-qubit gates  ·  start at 1:00

Why IonQ? Trapped ions connect any two qubits directly, so one layer needs just sixty-six two-qubit gates, versus two hundred eighty-five on a heavy-hex chip.

## Slide 7: IonQ noise model: 29× random guessing  ·  start at 1:13

Our main result: on the IonQ noise model, seven in a thousand shots hit the exact best portfolio. That's twenty-nine times more often than random. Without noise, it's fifty-eight.

## Slide 8: 29 qubits: more layers help, only without noise  ·  start at 1:28

At twenty-nine assets, three layers find the best in seven percent of runs; classical search gets sixty-one. Sixteen layers reach ninety percent, but only without noise. So no quantum advantage here.

## Slide 9: Iceberg catches errors, but doubles the gates  ·  start at 1:44

We also tried Iceberg error detection. It caught errors in three out of four shots, but the kept shots weren't clearly better, because the code doubles the gates.

## Slide 10: Try it on Qollab: one click, one IonQ job  ·  start at 1:59

*Before recording: run once on Qollab so a finished result is already on screen. In the video, open main.py, open Run → Select QPU → IonQ Forte 1 (29q) noise model, then scroll the finished output. Do not wait for a new job (about 30 s).*

[SWITCH TO QOLLAB] Here's our project; the code is in main.py. We press Run, pick the Forte 1 noise model, and get results and plots. [BACK TO SLIDES] Honestly, classical methods still keep up. Next: Iceberg on deeper circuits.

## Slide 11: Thank you for your attention  ·  start at 2:26

Thank you for your attention! This effort is supported by Qollab and IonQ.
