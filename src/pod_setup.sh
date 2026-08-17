#!/bin/bash
# Wait for pod SSH, upload, install, launch extraction. Usage: pod_setup.sh <ip> <port>
set -e
IP=$1; PORT=$2
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT root@$IP"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

for i in $(seq 1 60); do
  if $SSH "echo pod-up" 2>/dev/null | grep -q pod-up; then break; fi
  sleep 20
done
$SSH "echo pod-up" | grep -q pod-up || { echo "SSH never came up"; exit 1; }
echo "SSH up. Uploading..."

scp -o StrictHostKeyChecking=no -P $PORT "$DIR/src/pod_extract.py" "$DIR/data/probe_manifest.jsonl" root@$IP:/workspace/
echo "Installing deps..."
$SSH "pip install -q -U 'transformers>=4.50' accelerate sentencepiece 2>&1 | tail -1"
echo "Launching extraction (nohup)..."
$SSH "cd /workspace && nohup python pod_extract.py --manifest probe_manifest.jsonl --out /workspace/out --batch 8 > extract.log 2>&1 & echo started"
echo "LAUNCHED"
