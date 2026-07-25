"""P8 bench harness: llama.cpp / Ollama / vLLM on RTX 3050 Ti 4GB (and cloud leg).

All engines run as pinned Docker containers exposing an OpenAI-compatible
endpoint; one streaming client measures TTFT + decode rate uniformly while a
sampler logs GPU power/VRAM via nvidia-smi.

Usage:
  python harness.py pull-models          # download pinned GGUFs -> models/
  python harness.py run                  # full matrix -> data/runs.csv
  python harness.py run --engine llamacpp --model l31-8b-q4  # subset
  python harness.py offload              # RQ3: -ngl sweep on 8B Q4
  python harness.py quality              # RQ2: perplexity + quiz per quant

Cloud leg: same script on a RunPod pod (GPU_NAME env overrides testbed label).
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
MODELS = HERE / "models"
DATA = HERE / "data"
LOGS = HERE / "logs"
for d in (MODELS, DATA, LOGS):
    d.mkdir(exist_ok=True)

TESTBED = os.environ.get("GPU_NAME", "RTX-3050Ti-Laptop-4GB")
# P8_NATIVE=1: engines as local subprocesses (RunPod pods — no docker-in-docker).
# Engine versions must match the docker pins: llama.cpp b5343, ollama 0.9.6,
# vllm 0.9.2 (bootstrap_runpod.sh installs exactly these).
NATIVE = os.environ.get("P8_NATIVE") == "1"
NATIVE_PROCS = []

# ---- pinned engine images -------------------------------------------------
# newest server-cuda build on a CUDA <=12.5 base (current builds use >=12.8
# bases, refused by the container-toolkit gate on driver 555.97 — see Paper 9)
LLAMACPP_IMG = "ghcr.io/ggml-org/llama.cpp:server-cuda-b5343"
OLLAMA_IMG = "ollama/ollama:0.9.6"
# v0.9.2 image is CUDA-12.8-based -> refused by container-toolkit gate on
# driver 555.97 (Paper 9); v0.6.3 = newest CUDA<=12.5-based image. Cloud leg
# (native pip) runs 0.9.2 — version skew documented in results.
VLLM_IMG = "vllm/vllm-openai:v0.6.3"

# ---- pinned models (HF repo, file, ~GGUF GB) ------------------------------
GGUFS = {
    # 1B: no Q2_K published; quant ladder lives on the 3B (Q2/Q4/Q8)
    "l32-1b-q4":  ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "l32-1b-q5":  ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q5_K_M.gguf"),
    "l32-1b-q8":  ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
    "qw25-3b-q2": ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q2_K.gguf"),
    "qw25-3b-q4": ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
    "qw25-3b-q8": ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q8_0.gguf"),
    "l31-8b-q2":  ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q2_K.gguf"),
    "l31-8b-q4":  ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
}
# vLLM can't load GGUF single-file for all archs reliably on 4GB; vLLM leg uses
# the smallest HF fp16/AWQ that fits — measured separately.
VLLM_MODELS = {
    # ungated (meta-llama needs HF auth); 1.5B fp16 ~3GB weights — tight on
    # 4GB with KV cache; if OOM, drop to Qwen2.5-0.5B-Instruct
    "qw25-15b-fp16": "Qwen/Qwen2.5-1.5B-Instruct",
}

PORT = 18080
N_PROMPTS = 12
MAX_TOKENS = 256
CONCURRENCY = (1, 2, 4)

PROMPTS = [
    "Explain the difference between a mutex and a semaphore with a concrete example.",
    "Write a Python function that merges two sorted lists in O(n) time.",
    "Summarize the causes of the 2008 financial crisis in three paragraphs.",
    "What are the tradeoffs between microservices and monoliths for a 5-person startup?",
    "Describe how DNS resolution works from typing a URL to receiving an IP.",
    "Write a short story opening about a lighthouse keeper who finds a message in a bottle.",
    "Explain gradient descent to a high-school student using a hiking analogy.",
    "List and explain five HTTP status codes every backend developer should know.",
    "How does a B-tree differ from a binary search tree, and why do databases prefer B-trees?",
    "Draft a polite email declining a job offer while keeping the relationship warm.",
    "Explain what happens during a TCP three-way handshake and why each step exists.",
    "Compare SQL and NoSQL databases for storing e-commerce order history.",
]


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


# ---- engine lifecycle -----------------------------------------------------

def stop_engine():
    if NATIVE:
        for p in NATIVE_PROCS:
            try:
                p.terminate()
            except OSError:
                pass
        NATIVE_PROCS.clear()
        # precise patterns — a bare "ollama"/"vllm" would match this very
        # process's cmdline when run with --engine and SIGTERM the harness
        for pat in ("llama-server", "ollama serve", "vllm.entrypoints"):
            subprocess.run(["pkill", "-f", pat], capture_output=True)
        # wait until the bench port is actually released (llama-server can
        # take >3s to drain; ollama then fails to bind)
        for _ in range(30):
            r = subprocess.run(["ss", "-ltn"], capture_output=True, text=True)
            if f":{PORT}" not in (r.stdout or ""):
                break
            time.sleep(2)
        else:
            for pat in ("llama-server", "ollama serve", "vllm.entrypoints"):
                subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
            time.sleep(3)
    else:
        sh(["docker", "rm", "-f", "p8bench"])


def _spawn(args, env=None):
    e = dict(os.environ, **(env or {}))
    log = open(LOGS / f"native_{Path(args[0]).name}.log", "ab")
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=e)
    NATIVE_PROCS.append(p)
    return p


def start_llamacpp(model_key, ngl=99):
    f = GGUFS[model_key][1]
    if NATIVE:
        _spawn(["/opt/llama/llama-server", "-m", str(MODELS / f),
                "-ngl", str(ngl), "-c", "4096", "--host", "127.0.0.1",
                "--port", str(PORT), "--parallel", "4"])
    else:
        args = ["docker", "run", "-d", "--name", "p8bench", "--gpus", "all",
                "-p", f"{PORT}:8080", "-v", f"{MODELS}:/models",
                LLAMACPP_IMG, "-m", f"/models/{f}", "-ngl", str(ngl),
                "-c", "4096", "--host", "0.0.0.0", "--port", "8080",
                "--parallel", "4"]
        r = sh(args)
        if r.returncode != 0:
            raise RuntimeError(f"llamacpp start: {r.stderr}")
    return f"http://localhost:{PORT}/v1/chat/completions", "default"


def start_ollama(model_key):
    f = GGUFS[model_key][1]
    if NATIVE:
        _spawn(["ollama", "serve"],
               env={"OLLAMA_HOST": f"127.0.0.1:{PORT}",
                    "OLLAMA_NUM_PARALLEL": "4"})
        time.sleep(5)
        mf = MODELS / "Modelfile"
        mf.write_text(f"FROM {MODELS / f}\n", encoding="utf-8")
        r = subprocess.run(["ollama", "create", "bench", "-f", str(mf)],
                           capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{PORT}"))
        if r.returncode != 0:
            raise RuntimeError(f"ollama create: {r.stderr}")
    else:
        r = sh(["docker", "run", "-d", "--name", "p8bench", "--gpus", "all",
                "-p", f"{PORT}:11434", "-v", f"{MODELS}:/models",
                "-e", "OLLAMA_NUM_PARALLEL=4",
                "-e", "OLLAMA_LOAD_TIMEOUT=20m", OLLAMA_IMG])
        if r.returncode != 0:
            raise RuntimeError(f"ollama start: {r.stderr}")
        time.sleep(5)
        modfile = f"FROM /models/{f}"
        sh(["docker", "exec", "p8bench", "sh", "-c",
            f"echo '{modfile}' > /tmp/Modelfile"])
        r = sh(["docker", "exec", "p8bench", "ollama", "create", "bench",
                "-f", "/tmp/Modelfile"], timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"ollama create: {r.stderr}")
    return f"http://localhost:{PORT}/v1/chat/completions", "bench"


def start_vllm(model_key):
    hf = VLLM_MODELS[model_key]
    tok = os.environ.get("HF_TOKEN", "")
    if NATIVE:
        _spawn(["python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", hf, "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.90", "--dtype", "half",
                "--port", str(PORT)])
    else:
        r = sh(["docker", "run", "-d", "--name", "p8bench", "--gpus", "all",
                "-p", f"{PORT}:8000", "-e", f"HUGGING_FACE_HUB_TOKEN={tok}",
                "-v", f"{MODELS / 'hf-cache'}:/root/.cache/huggingface",
                VLLM_IMG, "--model", hf, "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.90", "--dtype", "half"])
        if r.returncode != 0:
            raise RuntimeError(f"vllm start: {r.stderr}")
    return f"http://localhost:{PORT}/v1/chat/completions", hf


def wait_ready(url, model, timeout=1800):
    """Poll with a 1-token request until the server answers.

    Must use the real model name: Ollama's OpenAI endpoint 404s unknown
    models, so a dummy name never reads as ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            body = json.dumps({"model": model, "max_tokens": 1,
                               "messages": [{"role": "user", "content": "hi"}]})
            req = urllib.request.Request(url, data=body.encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
            return time.time() - t0
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"engine not ready after {timeout}s")


# ---- GPU sampler ----------------------------------------------------------

class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            r = sh(["nvidia-smi", "--query-gpu=power.draw,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"])
            if r.returncode == 0 and r.stdout.strip():
                try:
                    p, m, u = [float(x) for x in r.stdout.strip().split(",")]
                    self.samples.append((p, m, u))
                except ValueError:
                    pass
            time.sleep(0.5)

    def result(self):
        self.stop_flag.set()
        if not self.samples:
            return {}
        return {
            "power_mean_w": round(statistics.mean(s[0] for s in self.samples), 1),
            "power_max_w": round(max(s[0] for s in self.samples), 1),
            "vram_max_mb": round(max(s[1] for s in self.samples)),
            "util_mean": round(statistics.mean(s[2] for s in self.samples), 1),
        }


# ---- streaming client -----------------------------------------------------

def one_request(url, model, prompt):
    """Streamed chat completion; returns ttft, decode tok/s, n_tokens."""
    body = json.dumps({"model": model, "stream": True, "max_tokens": MAX_TOKENS,
                       "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]})
    req = urllib.request.Request(url, data=body.encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            delta = d.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                n += 1
    total = time.time() - t0
    decode_s = total - (ttft or 0)
    return {"ttft_s": ttft, "n_chunks": n,
            "decode_tps": n / decode_s if decode_s > 0 and n else 0,
            "total_s": total}


def bench_round(url, model, conc):
    """N_PROMPTS requests at given concurrency; per-request metrics."""
    results = []
    lock = threading.Lock()
    prompts = list(PROMPTS[:N_PROMPTS])

    def worker():
        while True:
            with lock:
                if not prompts:
                    return
                p = prompts.pop()
            try:
                results.append(one_request(url, model, p))
            except Exception as e:
                results.append({"error": str(e)[:200]})

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    good = [r for r in results if "error" not in r]
    agg = {
        "conc": conc, "n_ok": len(good), "n_err": len(results) - len(good),
        "wall_s": round(wall, 1),
        "agg_tps": round(sum(r["n_chunks"] for r in good) / wall, 2) if wall else 0,
    }
    if good:
        agg["ttft_med_s"] = round(statistics.median(r["ttft_s"] for r in good if r["ttft_s"]), 3)
        agg["decode_tps_med"] = round(statistics.median(r["decode_tps"] for r in good), 2)
    return agg


# ---- matrix runner --------------------------------------------------------

FIELDS = ["testbed", "engine", "engine_ver", "model", "ngl", "conc", "n_ok",
          "n_err", "load_s", "ttft_med_s", "decode_tps_med", "agg_tps",
          "wall_s", "power_mean_w", "power_max_w", "vram_max_mb", "util_mean",
          "ts"]


def append_row(row, out_name="runs.csv"):
    out = DATA / out_name
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def run_case(engine, model_key, ngl=99, out_name="runs.csv"):
    print(f"=== {engine} {model_key} ngl={ngl}", flush=True)
    stop_engine()
    try:
        if engine == "llamacpp":
            url, mname = start_llamacpp(model_key, ngl)
            ver = LLAMACPP_IMG
        elif engine == "ollama":
            url, mname = start_ollama(model_key)
            ver = OLLAMA_IMG
        elif engine == "vllm":
            url, mname = start_vllm(model_key)
            ver = VLLM_IMG
        load_s = wait_ready(url, mname)
        print(f"  ready in {load_s:.0f}s", flush=True)
        for conc in CONCURRENCY:
            sampler = GpuSampler()
            sampler.start()
            agg = bench_round(url, mname, conc)
            agg.update(sampler.result())
            row = {"testbed": TESTBED, "engine": engine, "engine_ver": ver,
                   "model": model_key, "ngl": ngl, "load_s": round(load_s, 1),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **agg}
            append_row(row, out_name)
            print(f"  conc={conc}: {agg}", flush=True)
    except Exception as e:
        print(f"  FAIL: {e}", flush=True)
        detail = ""
        if not NATIVE:
            logs = sh(["docker", "logs", "p8bench", "--tail", "50"])
            detail = f"{logs.stdout}\n{logs.stderr}"
        (LOGS / f"fail_{engine}_{model_key}.log").write_text(
            f"{e}\n---\n{detail}", encoding="utf-8")
        append_row({"testbed": TESTBED, "engine": engine, "model": model_key,
                    "ngl": ngl, "n_ok": 0, "n_err": -1,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, out_name)
    finally:
        stop_engine()


def pull_models():
    for key, (repo, fname) in GGUFS.items():
        dest = MODELS / fname
        if dest.exists():
            print(f"have {fname}", flush=True)
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        print(f"pull {key}: {fname}", flush=True)
        urllib.request.urlretrieve(url, dest)
    print("models ready", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull-models", "run", "offload", "quality"])
    ap.add_argument("--engine")
    ap.add_argument("--model")
    a = ap.parse_args()

    if a.cmd == "pull-models":
        pull_models()
    elif a.cmd == "run":
        cases = []
        for mk in GGUFS:
            cases.append(("llamacpp", mk))
            cases.append(("ollama", mk))
        for mk in VLLM_MODELS:
            cases.append(("vllm", mk))
        done = set()
        runs = DATA / "runs.csv"
        if runs.exists():
            with open(runs, encoding="utf-8") as f:
                done = {(r["engine"], r["model"]) for r in csv.DictReader(f)
                        if r.get("n_ok") and int(r["n_ok"]) > 0}
        for eng, mk in cases:
            if a.engine and eng != a.engine:
                continue
            if a.model and mk != a.model:
                continue
            if (eng, mk) in done:
                print(f"skip (done): {eng} {mk}", flush=True)
                continue
            run_case(eng, mk)
    elif a.cmd == "offload":
        # RQ3: 8B Q4 (~4.9GB > VRAM) layer sweep; 8B has 32 layers + embed
        for ngl in (0, 8, 16, 20, 24, 28, 32, 99):
            run_case("llamacpp", "l31-8b-q4", ngl=ngl, out_name="offload.csv")
    elif a.cmd == "quality":
        print("quality probe: see quality.py (perplexity via llama-perplexity)", flush=True)


if __name__ == "__main__":
    main()
