#!/bin/zsh
# Create one Hetzner Cloud server, install the Python stack, upload the notebook, the frozen data, the 12-qubit tuning results and the
# tuning scripts, and start scripts/hpc/run_tuning_pipeline.sh (16 / 20-qubit re-optimisation, then frozen settings at 24 and 29 qubits).
# usage: scripts/hpc/launch_tuning.sh <server name> <server type> [location]      e.g. scripts/hpc/launch_tuning.sh zq-tuning ccx33 fsn1
# A server bills until it is DELETED: always start the watcher right after (it copies the results back, then deletes the server):
#   DONE_FILE=results/qaoa_tuning/DONE PULL="results/qaoa_tuning logs" nohup scripts/hpc/watch_and_delete.sh <deadline> <name> &
# Token: file in $HCLOUD_TOKEN_FILE (never committed); SSH key name: $HCLOUD_SSH_KEY.
NAME=$1; TYPE=$2; LOC=${3:-fsn1}
MESH=$(cd "$(dirname "$0")/../.." && pwd)
STATE=$MESH/scripts/hpc/state; mkdir -p $STATE
TOK=$(cat ${HCLOUD_TOKEN_FILE:-$HOME/.config/classiq_hetzner})
KEY=${HCLOUD_SSH_KEY:-mac}
reply=$(curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" https://api.hetzner.cloud/v1/servers \
  -d "{\"name\":\"$NAME\",\"server_type\":\"$TYPE\",\"location\":\"$LOC\",\"image\":\"ubuntu-24.04\",\"ssh_keys\":[\"$KEY\"]}")
info=$(echo "$reply" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['server']['id'], d['server']['public_net']['ipv4']['ip']) if 'server' in d else print('ERROR', d.get('error'))")
echo "$info" > $STATE/$NAME.server
echo "$NAME $TYPE: $info"
[[ $info == ERROR* ]] && exit 1
IP=${info#* }
ssh-keygen -R $IP > /dev/null 2>&1
for i in {1..40}; do
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes root@$IP "echo ok" > /dev/null 2>&1 && break
  sleep 6
done
ssh -o BatchMode=yes root@$IP "apt-get update -qq && apt-get install -y -qq python3-venv python3-pip > /dev/null && python3 -m venv ~/venv && ~/venv/bin/pip install -q numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2 qiskit==2.5.2 nbformat pandas && mkdir -p ~/mesh && echo SETUP_DONE" 2>&1 | tail -1
tar -C $MESH -czf $STATE/bundle_tuning.tgz mesh_idea_ZollerQu.ipynb data scripts/qaoa_tuning.py scripts/qaoa_scaling.py scripts/run_29_qubits.py \
  scripts/q29_choice.json scripts/hpc/run_tuning_pipeline.sh results/task04_checkpoint_small.json.gz \
  $(cd $MESH && ls results/qaoa_tuning/frozen_regions_calibration*.npz)
scp -o BatchMode=yes $STATE/bundle_tuning.tgz root@$IP:~/mesh/ > /dev/null && ssh -o BatchMode=yes root@$IP "cd ~/mesh && tar -xzf bundle_tuning.tgz 2>/dev/null; nproc; free -g | sed -n 2p"
# launch in its OWN ssh call (a pkill/relaunch in one command line would match and kill its own shell)
ssh -o BatchMode=yes root@$IP "cd ~/mesh && nohup bash scripts/hpc/run_tuning_pipeline.sh > pipeline.log 2>&1 < /dev/null & echo STARTED"
