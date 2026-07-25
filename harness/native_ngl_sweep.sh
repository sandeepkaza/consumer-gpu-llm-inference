#!/usr/bin/env bash
# Paper 8 RQ3 native-Linux replication: does the offload inverted-U reproduce
export LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH:-}
export PATH=/app:$PATH
# off WSL2, or does native CUDA OOM instead of paging?
#
# On the 4GB WSL2 floor, -ngl 99 on an overflowing model PAGED (UVM) and ran
# slow -> inverted-U. Native CUDA malloc does NOT page by default; it OOMs.
# This sweep runs a model that overflows a 24GB card across -ngl 0..all and
# records tokens/s (or OOM) at each point. Two possible outcomes, both honest:
#   (a) high-ngl OOMs  -> the "all-on-GPU is slow" arm is WSL2-specific
#   (b) high-ngl pages -> inverted-U reproduces on native Linux
set -uo pipefail

# ~34GB Q8 of a 32B model overflows a 24GB card (weights alone > VRAM).
HF_REPO="bartowski/Qwen2.5-32B-Instruct-GGUF"
HF_QUANT="Q8_0"
OUT=/root/native_sweep.csv

echo "== gpu / build =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CLI=$(command -v llama-cli || echo /app/llama-cli || find / -name llama-cli -type f 2>/dev/null | head -1)
BENCH=$(command -v llama-bench || echo /app/llama-bench || find / -name llama-bench -type f 2>/dev/null | head -1)
echo "cli=$CLI  bench=$BENCH"

echo "== downloading model via -hf (handles splits, caches) =="
# one tiny run to trigger the download into the HF cache
"$CLI" -hf "${HF_REPO}:${HF_QUANT}" -ngl 0 -n 1 -p "hi" --no-warmup 2>/tmp/dl.err >/dev/null || {
  echo "hf pull failed, tail:"; tail -5 /tmp/dl.err; exit 1; }
# HF hub caches blobs by hash; the real name lives in snapshots/ symlinks.
MODEL=$(find /root/.cache/huggingface -path "*snapshots*" -name "*Q8_0*.gguf" 2>/dev/null | grep -vi "of-0000[2-9]" | head -1)
[ -z "$MODEL" ] && MODEL=$(find / -path "*snapshots*" -name "*Qwen2.5-32B*Q8_0*.gguf" 2>/dev/null | head -1)
[ -z "$MODEL" ] && MODEL=$(find / -name "*Qwen2.5-32B*Q8_0*.gguf" 2>/dev/null | head -1)
echo "model: $MODEL ($(du -h "$MODEL" 2>/dev/null | cut -f1))"
[ -f "$MODEL" ] || { echo "MODEL NOT FOUND after pull"; exit 1; }

echo "ngl,tps,note" > "$OUT"
# 64-layer model; sweep across the fit boundary (~24GB holds part of a 34GB model)
for NGL in 0 12 24 36 44 48 52 56 64; do
  echo "== ngl=$NGL =="
  # -n 128 gen tokens, -p 0 (no prompt), 2 reps; capture tps or failure
  RES=$(timeout 400 "$BENCH" -m "$MODEL" -ngl "$NGL" -p 0 -n 128 -r 2 2>/tmp/bench.err)
  RC=$?
  if [ $RC -ne 0 ]; then
    ERR=$(grep -iE "out of memory|cudaMalloc|failed to allocate|error" /tmp/bench.err | head -1 | tr ',' ';')
    echo "$NGL,,OOM/err: ${ERR:-rc=$RC}" >> "$OUT"
    echo "  ngl=$NGL FAILED: ${ERR:-rc=$RC}"
  else
    # llama-bench prints a markdown table; extract the tg (generation) t/s
    TPS=$(echo "$RES" | grep -iE "tg128|tg " | grep -oE "[0-9]+\.[0-9]+ ± " | head -1 | grep -oE "[0-9]+\.[0-9]+")
    [ -z "$TPS" ] && TPS=$(echo "$RES" | grep -oE "[0-9]+\.[0-9]+ ± [0-9]+\.[0-9]+" | tail -1 | grep -oE "^[0-9]+\.[0-9]+")
    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "$NGL,$TPS,vram_mb=$VRAM" >> "$OUT"
    echo "  ngl=$NGL tps=$TPS vram=${VRAM}MB"
  fi
done
echo "SWEEP DONE"
cat "$OUT"
