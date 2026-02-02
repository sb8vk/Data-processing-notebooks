# Collection of executable notebooks exploring data science & ML.

---

## Overview 
This repository contains hands-on guides, reproducible notebooks, and short technical notes for myself as I explore GPU-accelerated data science. 

> Goal: exploring how to make smarter tool choices and understand where GPUs can materially improve workload performance.

---

## What you’ll find here 
- `notebooks/` — Executable, small-focus notebooks that demonstrate end-to-end patterns (data loading → transform → model/training/eval).
- `docs/` — Long-form guides and how-tos (optimizations, profiling, memory management).
- `benchmarks/` — Scripts and raw results (CSV / Apache Arrow) and scripts for reproducing them.
- `envs/` — `requirements.txt` / `environment.yml` / `Dockerfile` used for experiments.

---

## Quick start 
1. Check GPU and drivers:

```bash
# Verify driver & basic GPU health
nvidia-smi
# Confirm CUDA toolkit on PATH
nvcc --version
```
