#!/bin/zsh
# Safety net for scripts/hpc/launch_29_qubits.sh (run it detached: nohup ... &): for every server named, pull
# ~/mesh/results/hpc_n29 and the log as soon as DONE appears, then DELETE the server (billing stops only on delete).
# At the hard deadline (unix time) it pulls whatever exists and deletes everything, finished or not.
# usage: nohup scripts/hpc/watch_and_delete.sh <deadline unix time> <server name> [<server name> ...] > /dev/null 2>&1 &
# Q29_OUT (default results): the remote parent folder of hpc_n29, as given to launch_29_qubits.sh.
# DONE_FILE / PULL (paths under ~/mesh) override what marks the end and what is copied back, e.g. for launch_tuning.sh:
#   DONE_FILE=results/qaoa_tuning/DONE PULL="results/qaoa_tuning logs" nohup scripts/hpc/watch_and_delete.sh <deadline> zq-tuning &
MESH=$(cd "$(dirname "$0")/../.." && pwd)
STATE=$MESH/scripts/hpc/state
DEADLINE=$1; shift
RESULTS=${Q29_OUT:-results}
DONE_FILE=${DONE_FILE:-$RESULTS/hpc_n29/DONE}
PULL=(${=PULL:-$RESULTS/hpc_n29 run29.log})
TOK=$(cat ${HCLOUD_TOKEN_FILE:-$HOME/.config/classiq_hetzner})
typeset -A left
for name in "$@"; do left[$name]=1; done
pull_and_delete() {
  local name=$1 info=$(cat $STATE/$1.server); local sid=${info% *} ip=${info#* }
  mkdir -p $STATE/results_$name
  for item in $PULL; do
    scp -o BatchMode=yes -o ConnectTimeout=20 -r root@$ip:~/mesh/$item $STATE/results_$name/ >> $STATE/watcher.log 2>&1
  done
  curl -s -X DELETE -H "Authorization: Bearer $TOK" "https://api.hetzner.cloud/v1/servers/$sid" > $STATE/delete_$name.json
  echo "$(date) pulled and deleted $name ($sid)" >> $STATE/watcher.log
  unset "left[$name]"
}
while (( ${#left} > 0 )); do
  for name in ${(k)left}; do
    info=$(cat $STATE/$name.server); ip=${info#* }
    if ssh -o BatchMode=yes -o ConnectTimeout=10 root@$ip "test -f ~/mesh/$DONE_FILE" 2>/dev/null || [ $(date +%s) -ge $DEADLINE ]; then
      pull_and_delete $name
    fi
  done
  (( ${#left} > 0 )) && sleep 30
done
curl -s -H "Authorization: Bearer $TOK" https://api.hetzner.cloud/v1/servers > $STATE/servers_after.json
echo "$(date) watcher finished; servers left: $(python3 -c "import json; print(len(json.load(open('$STATE/servers_after.json'))['servers']))")" >> $STATE/watcher.log
