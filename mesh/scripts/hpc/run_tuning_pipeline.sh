#!/bin/bash
# Runs ON the server (started by scripts/hpc/launch_tuning.sh): the QAOA tuning pipeline.
#   0. 12 qubits, 18 calibration meshes: scripts/qaoa_tuning.py large, one process per scenario and depth, then merged;
#   1. freeze the calibration-mesh regions at 16 and 20 qubits (once, so parallel runs never write the same file);
#   2. scripts/qaoa_scaling.py at n = 16 and 20, both scenarios, depths split over 16 single-threaded processes;
#   3. --choose: settings for 24 and 29 qubits from the n = 20 calibration scores only;
#   4. scripts/run_29_qubits.py at n = 24 and 29 with those frozen settings (4 instances in parallel).
# Logs: ~/mesh/logs; results: ~/mesh/results/qaoa_tuning; finished when ~/mesh/results/qaoa_tuning/DONE exists.
set -u
cd ~/mesh
PY=~/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
for s in 0 1; do
  for p in 8 6 5 4 3 2 1; do
    $PY scripts/qaoa_tuning.py large $s $p > logs/tuning_n12_s${s}_p$p.log 2>&1 &
    sleep 10                                    # stagger: every notebook load rewrites data/frozen_regions.npz
  done
done
wait
$PY scripts/qaoa_tuning.py large merge > logs/tuning_n12_merge.log 2>&1
for n in 16 20; do $PY scripts/qaoa_scaling.py --freeze $n > logs/freeze_n$n.log 2>&1; done
for n in 20 16; do
  for s in 0 1; do
    for group in 8 6,1 5,2 4,3; do
      $PY scripts/qaoa_scaling.py $n $s $group > logs/scaling_n${n}_s${s}_p${group/,/_}.log 2>&1 &
      sleep 10                                  # stagger: every notebook load rewrites data/frozen_regions.npz
    done
  done
done
wait
$PY scripts/qaoa_scaling.py --choose > logs/choose.log 2>&1
for n in 24 29; do
  Q29_CHOICE=scripts/q29_choice_tuned.json Q29_OUT=results/qaoa_tuning/hpc $PY scripts/run_29_qubits.py $n 7,8,9 4 > logs/run_n$n.log 2>&1
done
touch results/qaoa_tuning/DONE
