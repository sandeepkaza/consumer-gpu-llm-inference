# Paper 8 — Results (validation source)

**Status: ALL LEGS COMPLETE 2026-07-19** (local 53 rows + offload 24 +
quality 8; cloud llama.cpp 24 + ollama 24 + vllm 3; total cloud spend
$2.44, all pods terminated). All numbers from `analyze.py` → `analysis.txt`, raw CSVs in
`data/` (`runs.csv` 53 rows local, `cloud-3090/` + `cloud-3090-v5/` cloud,
`offload.csv` 24, `quality.csv` 8). USER MUST VALIDATE every number and the
RQ4 price assumptions before drafting.

## Testbeds
- **Floor**: RTX 3050 Ti Laptop 4GB (sm_86), driver 555.97, Docker Desktop
  WSL2; engines as pinned containers: llama.cpp `server-cuda-b5343` (commit
  62d4250e), Ollama 0.9.6, vLLM v0.6.3 (v0.9.2 image refused by toolkit
  gate — Paper 9 finding, documented skew).
- **Ceiling**: RunPod secure RTX 3090 24GB, $0.46/hr; same engine versions
  native (llama.cpp built from 62d4250e sm86+89; ollama 0.9.6; vllm 0.9.2
  pip + transformers 4.53.2). Harness identical (`harness.py`, P8_NATIVE=1).
- Workload: 12 fixed prompts, 256 max tokens, temp 0, streaming; conc 1/2/4;
  nvidia-smi 0.5s sampler (power/VRAM/util).

## RQ1 — Engine comparison (conc=1 decode tok/s, local 4GB)

| model | llama.cpp | Ollama | vLLM |
|---|---|---|---|
| 1B Q4 | 90.8 | 90.6 | — |
| 1B Q8 | 74.7 | 50.2 | — |
| 3B Q4 | 38.1 | 32.0 | — |
| 3B Q8 | 31.0 | **12.8** | — |
| 8B Q2 | 18.6 | **9.1** | — |
| 8B Q4 | 7.2 | **FAIL** | — |
| Qwen1.5B fp16 | — | — | 38.9 |

- Engines tie when the model fits comfortably; **Ollama collapses at the
  VRAM edge** (self-caps ~2.7GB, spills to CPU: −59% at 3B Q8, −51% at
  8B Q2, power drops to 11-14W = GPU idling) and **fails to serve 8B Q4
  at all** (runner load timeout, even with OLLAMA_LOAD_TIMEOUT=20m) where
  llama.cpp sustains 7.2 tok/s.
- Ollama TTFT 3-11× llama.cpp (0.10-1.04s vs 0.03-0.10s); Ollama lazy-load
  (ready ~5s) vs llama.cpp eager (66-374s to first readiness).
- vLLM (v0.6.3) runs fp16-only-class models on 4GB: 38.9 tok/s single,
  **agg 130 tok/s @ conc4 (3.4× scaling)** vs ~2.2× for llama.cpp/Ollama —
  continuous batching is real even at the floor.
- Concurrency: llama.cpp 1B Q4 agg 90→207 tok/s (c1→c4); per-stream decode
  degrades gracefully (91→52).

## RQ2 — Quantization ladder (wikitext-2-raw ppl, 60 chunks, llama.cpp;
comparable within model family only — tokenizers differ across families)

| model | Q2_K | Q4_K_M | Q5_K_M | Q8_0 |
|---|---|---|---|---|
| Llama-3.2-1B | n/p | 15.28 | 14.87 | 14.76 |
| Qwen2.5-3B | 11.12 | 8.69 | — | 8.46 |
| Llama-3.1-8B | 10.03 | 7.62 | — | — |

- **Q4_K_M is the knee**: Q2→Q4 = −22% ppl (Qwen 3B), Q4→Q8 = −2.7% only.
- **Within Llama family: 8B-Q2 (10.03) beats the best 1B (14.76)** —
  parameter count dominates quantization damage down to Q2 on 4GB.
- Best-quality config that runs: 8B Q4 (7.62) at 7-13 tok/s (offloaded).

## RQ3 — Offload boundary (8B Q4 = 4.6GB on 4GB VRAM, llama.cpp -ngl sweep)

ngl:  0 → 7.2 tps | 16 → 8.2 | 20 → 8.8 | **24 → 13.2 (peak)** |
28 → 12.8 | 32 → 8.6 | 99 → 7.1 tps

- **Inverted-U**: partial offload (24/32 layers) is 1.85× faster than both
  pure CPU AND "all layers on GPU" — full offload oversubscribes VRAM
  (UVM/WSL2 paging) back down to CPU speed at 98% GPU util (busy ≠ fast).
- TTFT moves opposite: monotonic 1.58s → 0.19s toward full offload. The
  right ngl depends on whether you optimize decode or first token.
- Default `-ngl 99`-style "everything on GPU" is the WRONG default at the
  boundary.

## RQ4 — $/Mtok (assumptions: $0.15/kWh; 3090 $0.46/hr; llama.cpp conc=4)

| config | local elec | + amortization* | 3090 rental | API ref |
|---|---|---|---|---|
| 1B Q4/Q8 | $0.006 | ~$0.03 | $0.14-0.15 | $0.40-0.60 out |
| 3B Q4/Q8 | $0.013-0.015 | ~$0.06 | $0.32-0.33 | (4o-mini $0.60, |
| 8B Q2/Q4 | $0.027-0.051 | ~$0.15-0.24 | $0.47-0.50 | Flash $0.40) |

\* amortization: $400 GPU-hardware share / 3y continuous — user validates
assumption; API prices July 2026, MUST be re-verified before publish.

- Local electricity-only cost is ~100× under API list prices; even with
  hardware amortization ~4-20×. 3090 rental lands 2-4× under APIs.
- **Energy efficiency inversion**: 3050 floor delivers 2.3-2.5× more
  tokens/joule than the 3090 on models COMFORTABLY inside 4GB — 1B/3B, all
  ≤3.7GB (1B Q4: 3.09 vs 1.26 tok/J; 1B Q5 is the low end at 2.33×). The
  advantage narrows at the edge (8B Q2, 3.8GB: only 1.91×, 0.62 vs 0.33)
  and the ceiling only wins tok/J at 8B Q4 (0.34 vs 0.25) where the floor
  thrashes. Efficiency advantage is a function of HEADROOM, not just model
  size. Small-model inference is MOST energy-efficient on the consumer floor.
- Floor→ceiling speedup: uniform ~4.3-4.6× for models comfortably inside
  VRAM; 5.98× at 8B Q2 (edge); 16× at 8B Q4 (the cliff); 3090 ollama serves
  8B Q4 at 112.8 tok/s (7.9GB VRAM) — the local failure is purely a
  VRAM-boundary artifact.

## Incidents (paper's own evidence of the compat tax — cross-cites Paper 9)
- vllm/vllm-openai:v0.9.2 (CUDA 12.8 base) refused by container toolkit on
  driver 555.97 → pinned v0.6.3 locally.
- Current llama.cpp server-cuda images (≥12.8 base) equally refused → pinned
  b5343 (12.4 base); its git tag is pruned upstream — pinned by commit
  62d4250e recovered from the image binary.
- vLLM 0.9.2 + transformers ≥4.54: AutoConfig 'aimv2' double-registration
  crash → transformers pinned 4.53.2.
- First cloud build with CMAKE_CUDA_ARCHITECTURES=89 on a 3090 (sm_86):
  silent server death — Paper 9's arch-not-compiled class, self-inflicted.

## Threats
- WSL2 UVM paging shapes the RQ3 inverted-U; native-Linux replication
  advised (shape expected to persist, magnitudes may differ).
- Laptop 3050 power-limited (~30W cap observed) — tok/J favors it partly
  because of the mobile power envelope; state as "mobile Ampere floor".
- decode tok/s counted as SSE chunks/s ≈ tokens/s (chunk≈token for these
  servers); stated in method.
- Single prompt set (12 prompts, 256 out); no long-context axis.
- Cloud vLLM (3090, Qwen1.5B fp16, 22.6GB VRAM util): 174 tok/s single,
  agg 600 tok/s @ conc4 (3.5× scaling), TTFT 13ms — vs the same model on
  the 4GB floor: 39 tok/s, agg 130. 4.5× single-stream, 4.6× aggregate.

## RQ3 NATIVE-LINUX REPLICATION (added 2026-07-24) — inverted-U is WSL2-specific

Native Linux (vast.ai RTX 3090 24GB, driver 580.95, llama-bench, Qwen2.5-32B-Q8
33GB model overflowing 24GB card). ngl sweep (data/native-linux/native_sweep.csv):

| -ngl | tps | note |
|---|---|---|
| 0 | 1.34 | pure CPU |
| 12 | 1.61 | |
| 24 | 2.09 | |
| 36 | 3.21 | |
| 44 | 4.32 | PEAK = max layers that fit 24GB |
| 48 | FAIL | model failed to load (VRAM alloc exceeds 24GB) |
| 52,56,64 | FAIL | same — hard OOM at load |

**FINDING: the WSL2 inverted-U does NOT reproduce on native Linux.** tps rises
MONOTONICALLY with -ngl to the fit ceiling, then hard-FAILS (model won't load) —
native CUDA does not oversubscribe device memory, so it returns an allocation
failure rather than paging. WSL2's UVM paged the over-allocation (all-on-GPU
ran, slow, 98% util); native CUDA fails the alloc (all-on-GPU won't run at all).
So the "busy is not fast" / inverted-U finding is WSL2-SPECIFIC. Corrects the
prior speculation ("shape should persist") — measurement refuted it.
Practical rule differs by platform: WSL2 → sweep -ngl, avoid all-on-GPU trap;
native Linux → use largest -ngl that loads (no over-offload penalty, just a
fit ceiling). Caveat: different model/card than the 4GB floor; tests the
paging-vs-OOM MECHANISM (OS/runtime property), not the floor's exact magnitudes.
Cost: ~$0.30 vast (3090 ~1h incl 33GB download). VM destroyed.
Bonus: first provision host (cuda_max_good=12.0) REFUSED the llama.cpp image
with `nvidia-container-cli: unsatisfied condition: cuda>=12.8` — Paper 9's RQ2
gate finding hit live in the wild.
