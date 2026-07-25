# Paper 8 — "LLM Inference at the Consumer Floor: llama.cpp, vLLM, and Ollama on a 4 GB GPU, Measured"

**Genre:** inference-engine benchmark + FinOps analysis.
**Venue path:** UCC / edge-AI venues → IEEE Access.
**Why intense/viral:** everyone benchmarks H100s; nobody rigorously measures
the consumer floor where most hobbyist/edge deployment actually happens.
Local RTX 3050 (4 GB, CUDA 12.5, WSL2 passthrough verified) = the testbed —
$0, no quota, and the constraint IS the story.

## RQs
- RQ1: Engine comparison — llama.cpp (server), Ollama, vLLM (if it fits) on
  identical models/quantizations: tokens/s decode, time-to-first-token,
  concurrent-request behavior (batching), VRAM headroom, pinned versions.
- RQ2: Quantization ladder — same model at Q2/Q4/Q5/Q8: throughput vs a
  fixed quality probe set (perplexity + small task suite; quality measured,
  not assumed) — where the knee is on 4 GB.
- RQ3: Offload boundary — layers-on-GPU sweep (partial CPU offload): the
  throughput cliff as models exceed VRAM; hybrid CPU+GPU reality on consumer
  hardware.
- RQ4 (FinOps headline): $/Mtok — local (measured wall power via nvidia-smi
  draw × electricity rate, hardware amortization stated) vs cloud API prices
  vs cloud GPU rental: when does the consumer floor actually save money?

## Models
2–4 open models that fit 4 GB quantized (e.g., 1B–8B class), exact
checkpoints pinned and published.

## Cost
$0 cloud. Electricity noise. GPU-quota-free by design.
(Optional later leg if user files GPU quota requests: same harness on one
cloud T4/L4 for the consumer-vs-datacenter bridge.)

## Status
- [ ] WSL2 CUDA + engines install; harness (prompt set, timing, power log)
- [ ] Runs: engines × models × quant × concurrency
- [ ] results.md → user validates → 10pp draft
