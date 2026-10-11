#!/bin/zsh
# Create one Hetzner Cloud server, install the Python stack, upload the notebook + frozen data + 29-qubit script and start
# scripts/run_29_qubits.py (4 instances in parallel; ~7 GB each, so a 32 GB server is enough).
# usage: scripts/hpc/launch_29_qubits.sh <server name> <server type> <seeds, e.g. 7,9> [location]
#   e.g. scripts/hpc/launch_29_qubits.sh mesh29-s7 ccx33 7,9 fsn1 ; scripts/hpc/launch_29_qubits.sh mesh29-s8 cpx62 8 nbg1
# Writes <name>.server ("<id> <ip>") into scripts/hpc/state/. A server bills until it is DELETED: always run
# scripts/hpc/watch_and_delete.sh next to it. Token: file in $HCLOUD_TOKEN_FILE (never committed); SSH key name: $HCLOUD_SSH_KEY.
# Used on 2026-10-10 with ccx33 (fsn1) + cpx62 (nbg1); our account allowed 8 dedicated + 16 shared cores at a time.
# Optional: Q29_CHOICE (settings file, default scripts/q29_choice.json) and Q29_OUT (parent of hpc_n29, default results) are passed on;
#   e.g. Q29_CHOICE=scripts/q29_choice_tuned.json Q29_OUT=results/qaoa_tuning/hpc scripts/hpc/launch_29_qubits.sh zq-tuned ccx33 7,8,9
NAME=$1; TYPE=$2; SEEDS=$3; LOC=${4:-fsn1}
MESH=$(cd "$(dirname "$0")/../.." && pwd)
STATE=$MESH/scripts/hpc/state; mkdir -p $STATE
TOK=$(cat ${HCLOUD_TOKEN_FILE:-$HOME/.config/classiq_hetzner})
KEY=${HCLOUD_SSH_KEY:-mac}
reply=$(curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" https://api.hetzner.cloud/v1/servers \
  -d "{\"name\":\"$NAME\",\"server_type\":\"$TYPE\",\"location\":\"$LOC\",\"image\":\"ubuntu-24.04\",\"ssh_keys\":[\"$KEY\"]}")
info=$(echo "$reply" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['server']['id'], d['server']['public_net']['ipv4']['ip']) if 'server' in d else print('ERROR', d.get('error'))")
echo "$info" > $STATE/$NAME.server
echo "$NAME $TYPE seeds $SEEDS: $info"
[[ $info == ERROR* ]] && exit 1
IP=${info#* }
ssh-keygen -R $IP > /dev/null 2>&1
for i in {1..40}; do
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes root@$IP "echo ok" > /dev/null 2>&1 && break
  sleep 6
done
ssh -o BatchMode=yes root@$IP "apt-get update -qq && apt-get install -y -qq python3-venv python3-pip > /dev/null && python3 -m venv ~/venv && ~/venv/bin/pip install -q numpy==2.5.3 scipy==1.18.1 matplotlib==3.11.2 qiskit==2.5.2 nbformat pandas && mkdir -p ~/mesh && echo SETUP_DONE" 2>&1 | tail -1
CHOICE=${Q29_CHOICE:-scripts/q29_choice.json}
tar -C $MESH -czf $STATE/bundle.tgz mesh_idea_ZollerQu.ipynb data scripts/run_29_qubits.py scripts/q29_choice.json $CHOICE
scp -o BatchMode=yes $STATE/bundle.tgz root@$IP:~/mesh/ > /dev/null && ssh -o BatchMode=yes root@$IP "cd ~/mesh && tar -xzf bundle.tgz 2>/dev/null; nproc; free -g | sed -n 2p"
# launch in its OWN ssh call (a pkill/relaunch in one command line would match and kill its own shell)
ssh -o BatchMode=yes root@$IP "cd ~/mesh && Q29_CHOICE=$CHOICE Q29_OUT=${Q29_OUT:-} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup ~/venv/bin/python scripts/run_29_qubits.py 29 $SEEDS 4 > run29.log 2>&1 < /dev/null & echo STARTED"
