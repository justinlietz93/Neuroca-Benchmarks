# Installation Guide

This guide provides instructions for setting up the Neuroca vs. Agno Benchmark Suite.

## Prerequisites

- Python 3.9 – 3.12
- pip / venv
- matplotlib, numpy, psutil (for visualization and metrics)

## Step 1: Create a Fresh Workspace

```bash
mkdir agno_vs_neuroca_bench && cd $_
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate
python -m pip install --upgrade pip
```

## Step 2: Install Both Memory Engines

```bash
# Agno (legacy) – pinned to last stable tag
pip install agno==0.4.3

# Neuroca – from TestPyPI or prod once released
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple neuroca==0.1.0
```

## Step 3: Clone the Repository

```bash
git clone https://github.com/yourusername/neuroca-agno-benchmarks.git
cd neuroca-agno-benchmarks
```

## Step 4: Install Dependencies

```bash
pip install -r agno_vs_neuroca_bench/requirements.txt
```

## Step 5: Generate Test Dataset

```bash
python agno_vs_neuroca_bench/seed_dataset.py
```

This will create a `dataset.jsonl` file with 10,000 records for benchmarking.

## Step 6: Install Benchmark-Specific Dependencies

For visualizations and memory tracking:
```bash
pip install matplotlib numpy psutil lorem
```

For token counting:
```bash
pip install tiktoken
```

## Troubleshooting

### ImportError — faiss
Neuroca's in-memory vector backend falls back to scikit-learn if FAISS is missing but will be slower.
- Linux/macOS: `pip install faiss-cpu`
- Windows: Install appropriate wheel from PyPI

### Tokeniser missing
Install tiktoken or switch to openai_tokenizer alternative.

### Windows IOError with long paths
Run in C:\work\bench (shorter path) or enable LongPathsEnabled registry key.
