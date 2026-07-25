#!/bin/bash
# Autonomous P8 cloud-leg runner: bootstrap -> bench matrix -> serve results.
# Started as the pod's docker command; needs GIST_BASE env (raw gist base URL).
set -uxo pipefail
mkdir -p /workspace/p8/data /workspace/p8/logs
cd /workspace/p8

# serve results dir immediately on the pod's exposed HTTP port (19123):
# progress + final CSVs both visible from outside
(cd /workspace/p8 && python -m http.server 19123 >/dev/null 2>&1 &)

curl -fsSL "$GIST_BASE/bootstrap_runpod.sh" -o bootstrap.sh
curl -fsSL "$GIST_BASE/harness.py" -o harness.py
bash bootstrap.sh > logs/bootstrap.log 2>&1
echo "bootstrap rc=$?" > status.txt

# harness expects models/ + data/ next to itself
ln -sf /workspace/p8/models models 2>/dev/null || true
export P8_NATIVE=1
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr " " "-"); export GPU_NAME
# RUN_ENGINES: space-separated engine filter (default: full matrix + offload)
if [ -n "${RUN_ENGINES:-}" ]; then
  for e in $RUN_ENGINES; do
    python harness.py run --engine "$e" >> logs/matrix.log 2>&1
    echo "matrix($e) rc=$?" >> status.txt
  done
else
  python harness.py run > logs/matrix.log 2>&1
  echo "matrix rc=$?" >> status.txt
  python harness.py offload > logs/offload.log 2>&1
  echo "offload rc=$?" >> status.txt
fi
echo "ALL DONE" >> status.txt
sleep infinity
