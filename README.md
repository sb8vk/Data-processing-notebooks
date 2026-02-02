# Collection of executable notebooks focused on GPU-accelerated data science & ML.

---

## Overview 
This repository contains hands-on guides, reproducible notebooks, and short technical notes that show developers how to get real results with GPU-accelerated tools (PyTorch, TensorFlow, RAPIDS, CUDA, etc.). Content favors concrete examples, clear tradeoffs, and measured benchmarks over abstract feature lists. Expect code snippets, environment recipes, and reproducible benchmark artifacts (CSV / Apache Arrow) so readers can verify results on their hardware.

> Goal: exploring how to make smarter tool choices and understand where GPUs (CUDA cores, Tensor Cores, GPU memory) materially improve workload performance.

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

2. Recommended environment (conda):

```bash
conda create -n writing python=3.10 -y
conda activate writing
pip install -r envs/requirements.txt
```

3. Run notebooks locally:

```bash
jupyter lab --notebook-dir notebooks
# Or run a single notebook reproducibly with papermill
pip install papermill
papermill notebooks/01-example.ipynb -p BATCH_SIZE 512 output.ipynb
```

4. If you use containers (recommended for consistent CUDA / driver stacks):

```bash
# Build (uses NVIDIA container toolkit on host)
docker build -t sb8vk/writing:latest .
# Run with GPU access
docker run --gpus all -it --rm -p 8888:8888 sb8vk/writing:latest
```

---

## Notebook & content conventions 
- Filenames: `NN-short-title.ipynb` (e.g., `02-gpu-memory.ipynb`).
- Top cell should include: purpose, expected runtime, hardware used, and a short list of outputs (plots, CSV/Arrow files).
- Include an `environment.yml` or `requirements.txt` and exact CUDA/driver versions used for each benchmark.
- Prefer small notebooks (one main idea per notebook). If a notebook runs long, include a smoke-test cell that runs quickly on CPU for CI.
- Use `Apache Arrow` or compressed CSV for benchmark outputs to make downstream analysis fast and portable.

---

## Practical examples (short snippets) 🔧
- Check GPU availability and memory with PyTorch:

```python
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('Total GPU memory (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)
```

- Quick micro-benchmark: matrix multiply throughput (useful to sanity-check Tensor Core performance)

```python
import torch, time
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
start = time.time()
for _ in range(10):
    torch.matmul(x, x)
torch.cuda.synchronize()
print('Elapsed (s):', time.time()-start)
```

Measure and report: median runtime over N runs, peak GPU memory (via `nvidia-smi`) and throughput in GFLOPS/TFLOPS when appropriate.

---

## Benchmarking & reproducibility 
- Always record: GPU model, CUDA version, driver version, and batch sizes / input shapes.
- Use controlled runs: run at least 5-10 repeats, discard warm-up iterations, report median and interquartile range.
- Share raw results as `Apache Arrow` or `CSV` and provide a script to reproduce plots and tables.
- Be transparent about tradeoffs: e.g., mixed precision often increases throughput and reduces GPU memory, but may need careful loss scaling on some models.

---

## Performance tradeoffs & when to use GPUs 
- Best fit: large, vectorized work (dense linear algebra, batch model inference/training, data-parallel transforms). GPUs shine when you can keep a high ratio of compute to data-transfer.
- Watch for: small batch sizes, heavy Python loop overhead, or IO-bound pipelines—these can hide GPU benefits due to PCIe/PCIe4.0 latency and CPU bottlenecks.
- Alternatives: for single-threaded or low-data scenarios, modern CPUs or specialized accelerators may be more cost-effective; for out-of-core tabular workloads, Dask and Apache Arrow can reduce data movement costs.

---

## Testing & CI 
- Use `nbval` + `pytest` for notebook correctness; use smoke tests that run quickly on CPU in CI.
- GPU-enabled CI: prefer self-hosted runners with available GPUs or use cloud CI that exposes GPUs. When GPUs aren't available, validate logic with CPU fallbacks and mock metrics.
- Example CI jobs:
  - Linting (black, ruff)
  - Notebook smoke tests (CPU)
  - Full benchmark reproducibility job (self-hosted GPU runner)
