"""RQ2 quality probe: perplexity per quant on wikitext-2-raw test set.

Uses llama.cpp full image (has llama-perplexity), same b5343 CUDA base as the
bench harness. Fixed --chunks so every model sees identical text; ppl values
comparable within a model family across quants (the ladder's quality axis).

Run: python quality.py  -> data/quality.csv
"""

import csv
import re
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
MODELS = HERE / "models"
DATA = HERE / "data"
FULL_IMG = "ghcr.io/ggml-org/llama.cpp:full-cuda-b5343"
WIKI_ZIP = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
CHUNKS = 60  # ~30k tokens; identical across models

from harness import GGUFS  # pinned model registry


def ensure_wikitext():
    raw = MODELS / "wikitext-2-raw" / "wiki.test.raw"
    if raw.exists():
        return raw
    z = MODELS / "wikitext-2-raw-v1.zip"
    if not z.exists():
        print("downloading wikitext-2-raw", flush=True)
        urllib.request.urlretrieve(WIKI_ZIP, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(MODELS)
    assert raw.exists(), "unzip layout changed"
    return raw


def main():
    ensure_wikitext()
    out = DATA / "quality.csv"
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {r["model"] for r in csv.DictReader(f)}
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "file", "ppl", "ppl_err",
                                          "chunks", "wall_s"])
        if new:
            w.writeheader()
        for key, (_, fname) in GGUFS.items():
            if key in done:
                print(f"skip {key}", flush=True)
                continue
            print(f"=== ppl {key}", flush=True)
            t0 = time.time()
            r = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all",
                 "-v", f"{MODELS}:/models", "--entrypoint", "/app/llama-perplexity",
                 FULL_IMG, "-m", f"/models/{fname}",
                 "-f", "/models/wikitext-2-raw/wiki.test.raw",
                 "--chunks", str(CHUNKS), "-ngl", "99"],
                capture_output=True, text=True, timeout=7200,
                encoding="utf-8", errors="replace")
            txt = (r.stdout or "") + (r.stderr or "")
            m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", txt)
            row = {"model": key, "file": fname,
                   "ppl": m.group(1) if m else "",
                   "ppl_err": m.group(2) if m else "",
                   "chunks": CHUNKS, "wall_s": round(time.time() - t0, 1)}
            if not m:
                (HERE / "logs" / f"ppl_fail_{key}.log").write_text(txt[-5000:],
                                                                   encoding="utf-8")
                print(f"  NO PPL — logged", flush=True)
            w.writerow(row)
            f.flush()
            print(f"  {row}", flush=True)


if __name__ == "__main__":
    main()
