#!/bin/zsh
# usage: run_batch.sh <max_parallel> cfg1 cfg2 ...
cd /private/tmp/claude-501/-Users-t/7c811c50-f516-4e09-a51a-78bd3f6d1c7e/scratchpad/review_bugs
maxp=$1; shift
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for c in "$@"; do
  while [ $(jobs -r | wc -l) -ge $maxp ]; do sleep 2; done
  name=$(basename $c .py)
  ~/zollerqu-qollab/venv/bin/python -W ignore runner.py $c > logs4/$name.log 2>&1 &
done
wait
echo BATCH_DONE
