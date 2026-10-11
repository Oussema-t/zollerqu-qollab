#!/bin/zsh
# usage: cmp.sh <wd_mut> [maxlines]  -> diff of the printed TASK 04 RESULT block against the unmutated smoke baseline (wall-clock line removed)
cd /private/tmp/claude-501/-Users-t/7c811c50-f516-4e09-a51a-78bd3f6d1c7e/scratchpad/review_bugs
b=wd_base/stdout.txt; m=$1/stdout.txt
n=$(diff <(grep -v "measured wall-clock" $b) <(grep -v "measured wall-clock" $m) | grep -c '^>')
echo "--- $1: $n differing output lines (of $(wc -l < $m))"
diff <(grep -v "measured wall-clock" $b) <(grep -v "measured wall-clock" $m) | grep '^[<>]' | cut -c1-${3:-235} | head -${2:-8}
