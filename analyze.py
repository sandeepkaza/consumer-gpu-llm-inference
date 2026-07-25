"""P8 analysis: runs.csv (+cloud legs) + offload.csv + quality.csv -> stats
for results.md. Every cited number traces to a line here.

$/Mtok model (RQ4):
  local: measured watts * electricity $0.15/kWh + hardware amortization
         ($400 laptop-GPU share / 3y / duty) — assumptions stated, user
         adjusts.
  cloud GPU: pod $/hr / measured agg tok/s.
  APIs (July 2026 public prices, verify before publish): GPT-4o-mini out
  $0.60/Mtok, Claude Haiku 4.5 out $5/Mtok [PLACEHOLDER — user validates],
  Gemini 2.0 Flash out $0.40/Mtok.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
KWH = 0.15  # $/kWh assumption


def load(p):
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


local = load(DATA / "runs.csv")
cloud = load(DATA / "cloud-3090" / "runs.csv") + load(DATA / "cloud-3090-v5" / "runs.csv")
cloud6 = load(DATA / "cloud-3090-v6" / "runs.csv")
cloud += cloud6
offload = load(DATA / "offload.csv")
quality = load(DATA / "quality.csv")


def good(rows):
    return [r for r in rows if r.get("n_ok") and r["n_ok"].isdigit() and int(r["n_ok"]) > 0]


print("=" * 72)
print("LOCAL (RTX 3050 Ti 4GB) — engine x model, conc=1 decode tok/s")
print("=" * 72)
by = {}
for r in good(local):
    if r["conc"] == "1":
        by[(r["model"], r["engine"])] = r
models = sorted({m for m, _ in by})
print(f"{'model':<14}{'llamacpp':>10}{'ollama':>10}{'vllm':>10}")
for m in models:
    row = [m]
    for e in ("llamacpp", "ollama", "vllm"):
        r = by.get((m, e))
        row.append(r["decode_tps_med"] if r else "-")
    print(f"{row[0]:<14}{row[1]:>10}{row[2]:>10}{row[3]:>10}")

fails = [r for r in local if r.get("n_err") == "-1"]
print(f"\nfailed cases: {[(r['engine'], r['model']) for r in fails]}")

print()
print("=" * 72)
print("CLOUD (RTX 3090 24GB) vs LOCAL — llamacpp conc=1")
print("=" * 72)
cby = {}
for r in good(cloud):
    if r["conc"] == "1":
        cby[(r["model"], r["engine"])] = r
print(f"{'model':<14}{'3050 tps':>10}{'3090 tps':>10}{'ratio':>7}"
      f"{'3050 tok/J':>11}{'3090 tok/J':>11}")
for m in models:
    l = by.get((m, "llamacpp"))
    c = cby.get((m, "llamacpp"))
    if not (l and c):
        continue
    lt, ct = float(l["decode_tps_med"]), float(c["decode_tps_med"])
    lw, cw = float(l["power_mean_w"] or 1), float(c["power_mean_w"] or 1)
    print(f"{m:<14}{lt:>10.1f}{ct:>10.1f}{ct/lt:>7.1f}"
          f"{lt/lw:>11.2f}{ct/cw:>11.2f}")

print("\ncloud ollama conc=1 (incl. 8B-Q4 that local ollama cannot serve):")
for m in models:
    c = cby.get((m, "ollama"))
    if c:
        print(f"  {m}: {c['decode_tps_med']} tps, {c['vram_max_mb']} MB")
cv = [r for r in good(cloud) if r["engine"] == "vllm"]
for r in cv:
    print(f"  vllm {r['model']} conc={r['conc']}: {r['decode_tps_med']} tps, agg {r['agg_tps']}")

print()
print("=" * 72)
print("RQ3 OFFLOAD (8B-Q4, 4.6GB model on 4GB VRAM), conc=1")
print("=" * 72)
for r in offload:
    if r["conc"] == "1":
        print(f"  ngl={r['ngl']:>3}: {r['decode_tps_med']:>6} tps, "
              f"TTFT {r['ttft_med_s']}s, {r['power_mean_w']}W, "
              f"{r['vram_max_mb']}MB, util {r['util_mean']}%")

print()
print("=" * 72)
print("RQ2 QUALITY (wikitext-2 ppl, 60 chunks; comparable within family only)")
print("=" * 72)
for r in quality:
    print(f"  {r['model']:<12} ppl {r['ppl']} +/- {r['ppl_err']}")

print()
print("=" * 72)
print("RQ4 $/Mtok (assumptions: $%.2f/kWh; 3090 secure $0.46/hr)" % KWH)
print("=" * 72)
print("local (electricity only) + cloud rental, per model, llamacpp conc=4 agg:")
for m in models:
    lr = next((r for r in good(local)
               if r["model"] == m and r["engine"] == "llamacpp" and r["conc"] == "4"), None)
    cr = next((r for r in good(cloud)
               if r["model"] == m and r["engine"] == "llamacpp" and r["conc"] == "4"), None)
    if lr:
        tps = float(lr["agg_tps"])
        watts = float(lr["power_mean_w"] or 0)
        # $/Mtok = (watts/1000 * KWH) per hour / (tps*3600 tok per hour) * 1e6
        el = (watts / 1000 * KWH) / (tps * 3600) * 1e6 if tps else 0
        line = f"  {m:<14} local elec ${el:.3f}/Mtok ({tps:.0f} agg tps @ {watts:.0f}W)"
        if cr:
            ctps = float(cr["agg_tps"])
            cc = 0.46 / (ctps * 3600) * 1e6
            line += f" | 3090 rental ${cc:.2f}/Mtok ({ctps:.0f} agg tps)"
        print(line)
print("\nAPI reference (verify current prices): GPT-4o-mini $0.60/Mtok out, "
      "Gemini-2.0-Flash $0.40/Mtok out")
