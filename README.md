# Neuroca vs. Agno Memory Systems Benchmark Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A comprehensive benchmark suite for comparing Neuroca and Agno memory systems across multiple performance dimensions with special emphasis on long-term stability and self-maintenance capabilities.

## Overview

This repository contains a suite of benchmarks designed to evaluate and compare the performance characteristics of Neuroca and Agno memory systems. The benchmarks cover various aspects including basic operations, LLM integration, memory retention, task performance, and long-term stability under memory pressure.

![Memory Pressure Benchmark](Neuroca-Benchmarks\Neuroca-Agno-Benchmarks\agno_vs_neuroca_bench\benchmark_results\neuroca_pressure_metrics.png)

## Key Features

- **Multi-dimensional Analysis**: Evaluates memory systems across 6 key performance dimensions
- **Self-Maintenance Testing**: Specifically tests memory systems' ability to operate indefinitely
- **Comprehensive Metrics**: Measures speed, accuracy, resource usage, and stability
- **Reproducible Results**: Well-documented process for replicating benchmark results
- **Configurable Parameters**: Customize benchmark parameters to test different scenarios

## Benchmark Suite Components

The benchmark suite includes the following components:

1. **Basic Memory Operations Benchmark**: Measures raw performance (insertion speed, query latency, memory usage)
2. **LLM Integration Benchmark**: Evaluates context quality and token usage when augmenting LLMs
3. **Conversation Context Benchmark**: Tests memory retrieval in conversational contexts
4. **Memory Retention Benchmark**: Measures ability to recall important facts amid noise
5. **Task-Oriented Benchmark**: Evaluates effectiveness in QA scenarios
6. **Memory Pressure & Maintenance Benchmark**: Tests long-term stability under continuous load

## Key Findings

Our benchmarks revealed significant differences between the memory systems:

| Benchmark | Neuroca | Agno | Key Difference |
|-----------|---------|------|----------------|
| Basic Operations | ~0ms retrieval latency | 15-35ms retrieval latency | Neuroca's multi-tiered architecture enables near-instant queries |
| LLM Integration | 2,651 input tokens | 1,846 input tokens | Neuroca 43.6% more context |
| Memory Retention | 100% recall | 80% recall | Neuroca better important fact retention |
| Task-Oriented | 62% accuracy | 58% accuracy | Neuroca slightly better question answering |
| Memory Pressure | 5,000+ records, 0 errors | 0 records, immediate error | Neuroca's self-maintenance critical |

For detailed results, see the [Executive Summary](agno_vs_neuroca_bench/benchmark_results/executive_summary.md).

## Why Choose Neuroca?

- **Permanent Memory**: Successfully managed 5,000+ records under continuous load through autonomous self-maintenance, while other systems fail or degrade over time.
- **Reliable Recall**: Achieved 100% recall accuracy in noisy environments compared to 80% with alternatives.
- **Fast Queries**: Near-instant (0ms) retrieval latency versus 15–35 ms with other frameworks, enabling context-window-less operation.
- **Context-Window-Less Operation**: Dynamic access to relevant memories far beyond fixed context window limits in LLMs.

## Future Directions

This benchmark suite will be extended to include additional AI memory and retrieval frameworks, assessing:

- Scalability under varying workloads
- Integration with diverse LLMs and vector stores
- Real-world application performance and cost-efficiency
- Advanced self-maintenance and health monitoring capabilities

## Getting Started

See the [Installation Guide](INSTALLATION.md) for setup instructions.

To run all benchmarks:

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --all
```

See the [Replication Guide](docs/REPLICATION.md) for detailed step-by-step instructions.

## Documentation

- [Installation Guide](INSTALLATION.md): How to set up the benchmark environment
- [Benchmark Documentation](docs/): Detailed documentation for each benchmark
- [Replication Guide](docs/REPLICATION.md): Step-by-step instructions for reproducing results
- [Results Analysis](agno_vs_neuroca_bench/benchmark_results/): Comprehensive benchmark results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
