# consumer-gpu-inference-study

Companion artifact for the paper **"Inference at the Edge of VRAM: A Floor-and-Ceiling Measurement of Consumer-GPU LLM Serving"**.

Floor-and-ceiling measurement of consumer-GPU LLM serving across the VRAM edge: engine collapse, quantization, offload, and energy, plus a native-Linux offload replication.

## What's here

`data/` — per-run CSVs (local floor runs, offload sweep, quality, cloud 3090 ceiling). `data/native-linux/native_sweep.csv` — the native-Linux offload replication. `harness/` — the identical measurement harness for both testbeds.

## Reproduce

Requires Python 3 (plus, per paper, Docker/kind/a GPU host — see the paper's
Method section). Every number in the paper is regenerated from the committed
data by:

```bash
cd harness && python3 make_figures.py   # figures from analysis constants + data/
```

Each stage writes its intermediate data, so the path from raw measurement to
every reported figure is inspectable without re-running the measurement.

## Citation

See `CITATION.cff`. A DOI will be added after Zenodo archival. Venue path:
UCC -> IEEE Access.

## License

MIT (`LICENSE`). Author: Venkata Sandeep Chowdary Kaza <sandeep.kaza@gmail.com>, ORCID 0009-0001-9454-8995.

## AI-assistance disclosure

AI-assisted scripting and editing (Anthropic Claude) was employed under the
author's direction; the author defined the research questions, designed the
method, ran the measurements, and validated every reported number.
