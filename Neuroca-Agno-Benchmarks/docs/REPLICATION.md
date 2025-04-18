# Benchmark Replication Guide

This guide provides detailed, step-by-step instructions for reproducing the benchmark results presented in our comparison of Neuroca and Agno memory systems.

## Prerequisites

Before beginning, ensure you've completed all the steps in the [Installation Guide](../INSTALLATION.md).

## Running the Complete Benchmark Suite

For a comprehensive evaluation of all benchmarks with default parameters:

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --all
```

This will execute all benchmarks sequentially and produce a unified report in `benchmark_results/unified_benchmark_report.md`.

## Running Individual Benchmarks

### Basic Memory Operations Benchmark

This benchmark evaluates raw performance metrics like insertion speed, query latency, and memory usage.

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --basic
```

This benchmark:
1. Creates 10,000 random text records
2. Measures time to insert all records into each memory system
3. Performs 1,000 sample queries and measures response times
4. Records memory consumption

### LLM Integration Benchmark

This benchmark evaluates how effectively each memory system augments LLMs with relevant context.

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --llm
```

The benchmark:
1. Stores a corpus of related information in each memory system
2. Simulates LLM queries requiring context from stored information
3. Measures retrieval time, relevance of retrieved content, and token usage

### Conversation Context Benchmark

This benchmark tests how well each memory system handles conversational memory.

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --conversation
```

The benchmark:
1. Simulates a multi-turn conversation
2. Stores each exchange in the memory system
3. Measures retrieval accuracy when recalling previous exchanges
4. Evaluates context window usage efficiency

### Memory Retention Benchmark

This benchmark tests how well each system retains important information amid noise.

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --retention
```

The benchmark:
1. Stores a mixture of high-importance and low-importance facts
2. Floods the system with additional "noise" information
3. Tests recall accuracy for the original high-importance information

### Task-Oriented Benchmark

This benchmark evaluates effectiveness in question-answering scenarios.

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --task --dataset-size 500 --query-count 50
```

The benchmark:
1. Stores a dataset of question-answer pairs with relevant context
2. Queries for answers to specific questions
3. Measures accuracy, retrieval time, and relevance of retrieved context

### Memory Pressure & Maintenance Benchmark

This crucial benchmark tests long-term stability under continuous memory pressure.

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --pressure --max-records 20000 --batch-size 1000
```

The benchmark:
1. Progressively adds batches of records until reaching saturation or max limit
2. Monitors memory usage, query latency, and system health
3. Detects and logs self-maintenance events
4. Tests recovery capabilities after saturation

## Customizing Benchmark Parameters

Each benchmark accepts specific parameters to customize the test:

### Memory Pressure Benchmark Parameters

- `--max-records`: Maximum number of records to add (default: 20000)
- `--batch-size`: Number of records to add in each batch (default: 1000)

### Task-Oriented Benchmark Parameters

- `--dataset-size`: Number of QA pairs to use (default: 500)
- `--query-count`: Number of queries to perform (default: 50)

## Interpreting Results

After running the benchmarks, results will be saved to the `benchmark_results/` directory:

- `unified_benchmark_report.md`: Comprehensive report of all benchmark results
- `memory_pressure_report.md`: Detailed analysis of memory pressure benchmark
- `executive_summary.md`: High-level summary of key findings
- Various PNG files: Visualizations of performance metrics

For visualizations of memory pressure performance:
- `neuroca_pressure_metrics.png`: Shows Neuroca's performance under load
- `agno_pressure_metrics.png`: Shows Agno's performance under load

## Expected Results

When replicating these benchmarks, you should observe:

1. Neuroca demonstrating faster query times but slightly slower insertion
2. Neuroca maintaining better stability under memory pressure
3. Neuroca performing automatic maintenance when reaching capacity
4. Agno potentially failing under extreme memory pressure

Your specific results may vary based on hardware, but the relative performance trends should remain consistent.

## Troubleshooting

If you encounter issues during replication:

1. **Memory errors**: Try reducing `--max-records` or `--dataset-size`
2. **Performance inconsistencies**: Ensure no other resource-intensive processes are running
3. **Import errors**: Verify all dependencies are installed correctly

For detailed logs, check:
- `unified_benchmark.log`
- `memory_pressure.log`
- Other benchmark-specific log files

## Validating Your Results

To validate that your results are consistent with our findings:

1. Compare your unified report with the reference report in the repository
2. Visualize performance metrics using the generated PNG files
3. Verify that Neuroca's self-maintenance capabilities activate under pressure

If significant deviations are observed, please check your environment configuration and dependency versions.
