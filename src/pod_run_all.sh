#!/bin/bash
# Full pod lifecycle for probe extraction: wait for ssh -> upload -> install -> run -> watch
# -> download -> terminate. Usage: pod_run_all.sh <pod_id>
set -u
POD_ID=$1
DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$DIR/../../../.env"

get_ssh() {
  curl -s --request POST --header "content-type: application/json" \
    --url "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    --data "{\"query\":\"query { pod(input:{podId:\\\"$POD_ID\\\"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }\"}" |
  python3 -c "
import json,sys
r=json.load(sys.stdin)
rt=(r['data']['pod'] or {}).get('runtime')
if rt and rt.get('ports'):
    for p in rt['ports']:
        if p['privatePort']==22 and p['isIpPublic']: print(p['ip'], p['publicPort']); break
" 2>/dev/null
}

echo "waiting for ssh port..."
for i in $(seq 1 90); do
  INFO=$(get_ssh); [ -n "$INFO" ] && break; sleep 10
done
IP=$(echo $INFO | cut -d' ' -f1); PORT=$(echo $INFO | cut -d' ' -f2)
echo "endpoint: $IP:$PORT"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT root@$IP"

for i in $(seq 1 90); do
  $SSH "echo pod-up" </dev/null 2>/dev/null | grep -q pod-up && break; sleep 15
done
$SSH "echo pod-up" </dev/null | grep -q pod-up || { echo "FATAL: ssh never up"; exit 1; }

echo "uploading..."
scp -o StrictHostKeyChecking=no -P $PORT "$DIR/src/pod_extract.py" "$DIR/data/probe_manifest.jsonl" root@$IP:/workspace/ </dev/null
echo "installing..."
$SSH "pip install -q -U 'transformers>=4.50' accelerate sentencepiece 2>&1 | tail -1" </dev/null
echo "launching..."
$SSH "cd /workspace && (HF_TOKEN=$HUGGINGFACE_API_KEY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python pod_extract.py --manifest probe_manifest.jsonl --out /workspace/out > extract.log 2>&1 &) && echo launched" </dev/null

echo "watching..."
for i in $(seq 1 240); do
  STATUS=$($SSH "grep -E 'DONE|Traceback|OutOfMemoryError' /workspace/extract.log 2>/dev/null | tail -1" </dev/null 2>/dev/null)
  if echo "$STATUS" | grep -q DONE; then echo "EXTRACTION DONE"; break; fi
  if echo "$STATUS" | grep -qE 'Traceback|OutOfMemoryError'; then
    echo "ERROR on pod:"; $SSH "tail -20 /workspace/extract.log" </dev/null; exit 2
  fi
  sleep 30
done

echo "downloading artifacts..."
mkdir -p "$DIR/runs/pod_out"
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" root@$IP:/workspace/out/ "$DIR/runs/pod_out/" </dev/null
ls -la "$DIR/runs/pod_out/"

echo "terminating pod $POD_ID..."
curl -s --request POST --header "content-type: application/json" \
  --url "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
  --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD_ID\\\"}) }\"}"
echo "ALL DONE"
