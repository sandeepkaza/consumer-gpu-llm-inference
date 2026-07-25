#!/bin/bash
# P8 cloud-leg bootstrap for a RunPod pod (run as root inside the pod).
# Pins the SAME engine versions as the local docker leg:
#   llama.cpp b5343 (CUDA binary), ollama 0.9.6, vllm 0.9.2
set -euxo pipefail

cd /workspace
mkdir -p /workspace/p8/models /workspace/p8/data /workspace/p8/logs

# --- llama.cpp b5343: build from source with CUDA (no Linux CUDA binaries
# published; pod image has nvcc 12.4.1) ---
if [ ! -x /opt/llama/llama-server ]; then
  apt-get update -qq && apt-get install -y -qq cmake ninja-build git curl unzip
  # b5343 git tag pruned upstream; 62d4250e = commit inside the
  # server-cuda-b5343 image (verified via llama-server --version)
  curl -fL https://github.com/ggml-org/llama.cpp/archive/62d4250e.tar.gz | tar xz -C /tmp
  mv /tmp/llama.cpp-62d4250e* /tmp/llama.cpp
  cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -G Ninja \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF
  cmake --build /tmp/llama.cpp/build --target llama-server -j"$(nproc)"
  mkdir -p /opt/llama
  cp /tmp/llama.cpp/build/bin/llama-server /opt/llama/llama-server
  chmod +x /opt/llama/llama-server
fi

# --- ollama 0.9.6 ---
if ! command -v ollama; then
  curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.9.6 sh
fi

# --- vllm 0.9.2 ---
# transformers>=4.54 double-registers 'aimv2' against vllm 0.9.2 — pin both
pip show vllm 2>/dev/null | grep -q "Version: 0.9.2" || pip install -q vllm==0.9.2 "transformers==4.53.2"

# --- models (same pinned GGUFs) ---
cd /workspace/p8/models
for f in \
  "bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
  "bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q5_K_M.gguf" \
  "bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q8_0.gguf" \
  "bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q2_K.gguf" \
  "bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q4_K_M.gguf" \
  "bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q8_0.gguf" \
  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q2_K.gguf" \
  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"; do
  repo=${f% *}; file=${f#* }
  [ -f "$file" ] || curl -fL -O "https://huggingface.co/$repo/resolve/main/$file"
done

echo "BOOTSTRAP DONE"
