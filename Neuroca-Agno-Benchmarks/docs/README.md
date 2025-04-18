# Neuroca vs. Agno Benchmark Documentation

This directory contains detailed documentation for the Neuroca vs. Agno Memory Systems Benchmark Suite.

## Documentation Structure

### Overview Documents
- [Benchmark Summary](BenchmarkSummary.md) - Comprehensive overview of all benchmarks
- [Replication Guide](REPLICATION.md) - Step-by-step instructions for reproducing results

### Individual Benchmark Documentation
- [Basic Memory Operations Benchmark](BasicMemoryBenchmark.md) - Core performance metrics
- [LLM Integration Benchmark](LLMBenchmark.md) - Testing memory-augmented LLMs
- [Memory Pressure & Maintenance Benchmark](MemoryPressureBenchmark.md) - Long-term stability testing

### Architecture & Design
- [Benchmark Architecture](images/benchmark_architecture.md) - Layered benchmark approach visualization

## Benchmark Results

The actual benchmark results are stored in the `../agno_vs_neuroca_bench/benchmark_results/` directory:

- [Executive Summary](../agno_vs_neuroca_bench/benchmark_results/executive_summary.md) - High-level findings
- [Memory Pressure Report](../agno_vs_neuroca_bench/benchmark_results/memory_pressure_report.md) - Detailed analysis of stability
- [Unified Benchmark Report](../agno_vs_neuroca_bench/benchmark_results/unified_benchmark_report.md) - Complete results

## Visualizing the Architecture Diagram

The benchmark architecture diagram is provided as a Mermaid diagram. To view it:

1. In GitHub, the diagram will render automatically in the markdown file
2. Locally, you can use a Mermaid viewer extension in VS Code
3. Online, you can use the [Mermaid Live Editor](https://mermaid-js.github.io/mermaid-live-editor/)

## Navigating Benchmark Implementation

The actual benchmark implementations can be found in the `../agno_vs_neuroca_bench/` directory:

- `bench.py` - Basic memory operations benchmark
- `llm_benchmark.py` - LLM integration benchmark 
- `conversation_benchmark.py` - Conversation context benchmark
- `memory_retention_benchmark.py` - Information retention benchmark
- `task_oriented_benchmark.py` - Task performance benchmark
- `memory_pressure_benchmark.py` - Long-term stability benchmark
- `run_all_benchmarks.py` - Unified benchmark runner

## Getting Started

If you're new to this project, we recommend following this sequence:

1. Read the [Benchmark Summary](BenchmarkSummary.md) for a high-level overview
2. Review the [Executive Summary](../agno_vs_neuroca_bench/benchmark_results/executive_summary.md) for key findings
3. Follow the [Installation Guide](../INSTALLATION.md) to set up your environment
4. Use the [Replication Guide](REPLICATION.md) to reproduce the results
