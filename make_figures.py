"""Paper 8 figures (vector PDF, IEEE single-column 3.5in).

Reads analysis.txt-derived constants and data/*.csv. Six figures:
Fig 1 fig_engine.pdf   -- engine x model decode tok/s, local 4GB (Ollama VRAM-edge collapse)
Fig 2 fig_quant.pdf    -- quantization ppl ladder within family (Q4 knee; 8B-Q2 < best 1B)
Fig 3 fig_offload.pdf  -- ngl sweep inverted-U (decode vs TTFT vs util)
Fig 4 fig_cost.pdf     -- $/Mtok local elec vs 3090 rental vs API refs (log)
Fig 5 fig_energy.pdf   -- tok/J floor vs ceiling (energy inversion)
Run: python make_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LC, OL, VL = "#0072B2", "#D55E00", "#009E73"  # llama.cpp, ollama, vllm
FLOOR, CEIL = "#0072B2", "#CC79A7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "font.family": "serif", "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "pdf.fonttype": 42,
})

# ---- Fig 1: engine x model (local 4GB), conc=1 decode tok/s ----
models = ["1B Q4", "1B Q8", "3B Q4", "3B Q8", "8B Q2", "8B Q4"]
lcpp = [90.75, 74.66, 38.14, 30.96, 18.57, 7.18]
olla = [90.64, 50.16, 31.98, 12.83, 9.13, None]  # 8B Q4 = fail
fig, ax = plt.subplots(figsize=(3.5, 2.3))
x = range(len(models)); w = 0.38
ax.bar([i - w/2 for i in x], lcpp, w, color=LC, label="llama.cpp")
ol_plot = [v if v is not None else 0 for v in olla]
ax.bar([i + w/2 for i in x], ol_plot, w, color=OL, hatch="///", edgecolor="white",
       linewidth=0.3, label="Ollama")
ax.annotate("FAIL", (5 + w/2, 1.5), ha="center", fontsize=6, color=OL, rotation=90, va="bottom")
for i, (l, o) in enumerate(zip(lcpp, olla)):
    if o and o < l * 0.8:
        ax.annotate(f"-{100*(1-o/l):.0f}%", (i + w/2, o), ha="center", va="bottom",
                    fontsize=5.6, color=OL)
ax.set_xticks(list(x), models, fontsize=6.6)
ax.set_ylabel("Decode tok/s (conc=1)")
ax.set_title("Ollama collapses at the VRAM edge", fontsize=6.8)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linewidth=0.3, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("fig_engine.pdf"); print("fig_engine.pdf")

# ---- Fig 2: quantization ppl ladder ----
fig, ax = plt.subplots(figsize=(3.5, 2.2))
qorder = {"Q2": 2, "Q4": 4, "Q5": 5, "Q8": 8}
fams = {
    "Llama 1B": ([4, 5, 8], [15.28, 14.87, 14.76], "o-", LC),
    "Qwen 3B": ([2, 4, 8], [11.12, 8.69, 8.46], "s-", OL),
    "Llama 8B": ([2, 4], [10.03, 7.62], "^-", VL),
}
for lbl, (xs, ys, style, c) in fams.items():
    ax.plot(xs, ys, style, color=c, linewidth=1.4, label=lbl, markersize=4)
ax.axhline(14.76, color=LC, linewidth=0.5, linestyle=":")
ax.annotate("best 1B (14.76)", (5.5, 14.9), fontsize=5.8, color=MUTED)
ax.annotate("8B-Q2 (10.03)\nbeats best 1B", (2.1, 10.4), fontsize=5.8, color=VL)
ax.set_xticks([2, 4, 5, 8], ["Q2", "Q4", "Q5", "Q8"])
ax.set_xlabel("Quantization (bits, coarse)")
ax.set_ylabel("wikitext-2 perplexity (lower better)")
ax.set_title("Q4 is the knee; params beat precision", fontsize=6.8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, linewidth=0.3, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("fig_quant.pdf"); print("fig_quant.pdf")

# ---- Fig 3: offload inverted-U ----
ngl = [0, 8, 16, 20, 24, 28, 32, 99]
tps = [7.18, 6.24, 8.2, 8.82, 13.23, 12.84, 8.62, 7.14]
ttft = [1.58, 1.26, 0.918, 0.693, 0.453, 0.307, 0.189, 0.208]
util = [0.0, 28.6, 32.0, 34.3, 35.9, 56.1, 84.0, 98.1]
fig, ax = plt.subplots(figsize=(3.5, 2.3))
xi = range(len(ngl))
ax.plot(xi, tps, "o-", color=LC, linewidth=1.6, label="decode tok/s", zorder=3)
ax.set_xticks(list(xi), [str(n) for n in ngl])
ax.set_xlabel("GPU layers offloaded (-ngl)")
ax.set_ylabel("Decode tok/s", color=LC)
ax.axvline(4, color=MUTED, linewidth=0.5, linestyle=":")
ax.annotate("peak 13.2\n@ngl=24\n(1.85x)", (4, 13.4), fontsize=6, color=LC, ha="center", va="bottom")
ax.annotate("util 98%\nbut 7.1 tps\n(busy≠fast)", (7, 7.5), fontsize=5.8, color=MUTED, ha="center", va="bottom")
ax2 = ax.twinx()
ax2.plot(xi, ttft, "s--", color=OL, linewidth=1.1, markersize=3, label="TTFT (s)")
ax2.set_ylabel("TTFT (s)", color=OL)
ax2.spines[["top"]].set_visible(False)
ax.spines[["top"]].set_visible(False)
ax.grid(True, linewidth=0.3, color=GRID); ax.set_axisbelow(True)
l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, frameon=False, loc="upper center", fontsize=6.3)
fig.tight_layout(); fig.savefig("fig_offload.pdf"); print("fig_offload.pdf")

# ---- Fig 4: $/Mtok ----
cfgs = ["1B Q4", "3B Q4", "8B Q2", "8B Q4"]
local = [0.006, 0.015, 0.027, 0.051]
rental = [0.15, 0.32, 0.50, 0.47]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
x = range(len(cfgs)); w = 0.38
ax.bar([i - w/2 for i in x], local, w, color=FLOOR, label="local (elec only)")
ax.bar([i + w/2 for i in x], rental, w, color=CEIL, hatch="///", edgecolor="white",
       linewidth=0.3, label="3090 rental")
ax.axhspan(0.40, 0.60, color=MUTED, alpha=0.12)
ax.axhline(0.40, color=MUTED, linewidth=0.5, linestyle=":")
ax.annotate("API list ($0.40-0.60/Mtok out)", (0.5, 0.62), fontsize=5.8, color=MUTED)
ax.set_yscale("log")
ax.set_xticks(list(x), cfgs)
ax.set_ylabel("\\$/Mtok output (log)")
ax.set_title("local elec ~100x under API; rental 2-4x under", fontsize=6.6)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linewidth=0.3, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("fig_cost.pdf"); print("fig_cost.pdf")

# ---- Fig 5: energy inversion tok/J ----
emods = ["1B Q4", "1B Q8", "3B Q4", "3B Q8", "8B Q2", "8B Q4"]
floor_j = [3.09, 2.53, 1.28, 1.04, 0.62, 0.25]
ceil_j = [1.26, 1.03, 0.52, 0.42, 0.33, 0.34]
fig, ax = plt.subplots(figsize=(3.5, 2.2))
x = range(len(emods)); w = 0.38
ax.bar([i - w/2 for i in x], floor_j, w, color=FLOOR, label="3050 4GB floor")
ax.bar([i + w/2 for i in x], ceil_j, w, color=CEIL, hatch="///", edgecolor="white",
       linewidth=0.3, label="3090 ceiling")
ax.set_xticks(list(x), emods, fontsize=6.6)
ax.set_ylabel("Tokens / joule")
ax.annotate("floor 2.4x more\nefficient (fits)", (0.5, 2.6), fontsize=5.8, color=FLOOR)
ax.annotate("inverts at\n8B-Q4 (thrash)", (5, 0.5), fontsize=5.6, color=MUTED, ha="center")
ax.set_title("energy-efficiency inversion", fontsize=6.8)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linewidth=0.3, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("fig_energy.pdf"); print("fig_energy.pdf")

print("\n[check] all 5 figures written")
