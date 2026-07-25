# Paper 8 — "The ARM Dividend, Audited: Price-Performance of ARM vs x86 Managed Kubernetes Across Four Clouds"

**Genre:** measurement + FinOps analysis. **Venue path:** IEEE CLOUD → UCC → IEEE Access.
**Gap (web-verified):** vendor/blog claims (+20–40% price-perf) dominate; no
independent, multi-cloud, microservice-suite measurement. Unique angle we can
execute: ARM on all FOUR clouds in hand — AWS Graviton (EKS), Azure
Cobalt/Ampere (AKS), OCI Ampere A1 (OKE), GCP Axion/Tau T2A (GKE) — vs same-cloud x86 baselines.

## RQs
- RQ1: Throughput/latency — boutique under fixed load: RPS at SLO, p95, per
  arch per cloud (same node vCPU/RAM shape, same K8s version).
- RQ2: Price-performance — cost per 1k requests at SLO from real on-demand
  pricing; spot variant where offered.
- RQ3: Migration friction — multi-arch image availability across the app's
  dependency images, build changes needed, scheduling pitfalls (taints/nodeSelector),
  measured as a findings catalog (ops-findings pattern).
- RQ4: Energy/efficiency proxy if exposed (CPU util at equal RPS) — honest
  scope: no power meters, report utilization only.

## Testbed
Per cloud: 2 nodegroups (ARM + x86, matched shape), boutique + load
generator at fixed RPS steps; ≥5 trials per point. Same-day teardown.

## Cost
Highest of the slate: 6 nodegroups × hours. Target <$60; sequenced last.

## Status
- [ ] Harness: load driver + SLO gate + per-arch scheduling
- [ ] Runs: 3 clouds × 2 arch × load steps
- [ ] results.md → user validates → 10pp draft
