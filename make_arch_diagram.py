"""fig_rig.png -- Paper 8 floor/ceiling inference-measurement rig.

One harness, one prompt set, one engine matrix, two testbeds. Requires
graphviz `dot`. Run: python make_arch_diagram.py
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage
from diagrams.onprem.compute import Server

graph_attr = {"dpi": "300", "fontsize": "13", "pad": "0.2", "splines": "spline",
              "nodesep": "0.35", "ranksep": "0.6"}
node_attr = {"fontsize": "11"}

with Diagram("", filename="fig_rig", outformat="png", show=False,
             graph_attr=graph_attr, node_attr=node_attr, direction="LR"):

    harness = Python("harness.py\n12 prompts, 256 tok, temp 0\nstreaming; conc 1/2/4\nnvidia-smi 0.5s sampler")

    with Cluster("engines (pinned containers / native)"):
        eng = Storage("llama.cpp b5343 (62d4250e)\nOllama 0.9.6\nvLLM 0.6.3 local / 0.9.2 cloud")

    with Cluster("FLOOR: RTX 3050 Ti Laptop 4GB (sm86)"):
        floor = Server("driver 555.97, WSL2 Docker\nmodels 1B/3B/8B x Q2..Q8\noffload -ngl sweep")

    with Cluster("CEILING: RunPod RTX 3090 24GB"):
        ceil = Server("$0.46/hr, native\nsame engine versions\nsame harness (P8_NATIVE=1)")

    out = Storage("runs.csv / offload.csv\nquality.csv / cloud-3090*\n-> analyze.py -> analysis.txt")

    harness >> eng
    eng >> floor
    eng >> ceil
    floor >> Edge(label="tok/s, TTFT, W, VRAM, ppl") >> out
    ceil >> out
