# Memory Pressure & Maintenance Benchmark

## Overview

The Memory Pressure & Maintenance Benchmark is designed to evaluate how memory systems perform under continuous load and progressive growth. This benchmark is particularly important for assessing long-term stability and the self-maintenance capabilities that enable "indefinite" operation.

Key focus areas:
- **Progressive Memory Growth**: How systems handle continually increasing data
- **Self-Maintenance Detection**: Ability to perform automatic maintenance
- **Health Monitoring**: Stability of core metrics under pressure
- **Recovery Capabilities**: Resilience after reaching capacity limits

## Implementation Details

The benchmark is implemented in `memory_pressure_benchmark.py` and follows these steps:

1. **Setup Phase**:
   - Initialize both memory systems with identical configurations
   - Define health metrics and maintenance event detection thresholds
   - Set up metric tracking (memory usage, query latency, health score)

2. **Progressive Loading Phase**:
   - Add records in batches (default: 1,000 records per batch)
   - Use varying importance distributions to test prioritization
   - Monitor performance metrics after each batch
   - Continue until system saturation or maximum record count reached

3. **Maintenance Detection**:
   - Monitor for automatic self-maintenance events
   - Explicitly trigger maintenance at intervals to test system response
   - Log all maintenance activities and their effects

4. **Recovery Testing**:
   - If saturation is detected, test emergency recovery
   - Measure performance before and after recovery
   - Evaluate system stability post-recovery

## Key Metrics

The benchmark captures the following metrics:

| Metric | Description | Unit |
|--------|-------------|------|
| Max Records | Maximum records stored before failure/saturation | count |
| Final Memory | Memory usage at end of test | MB |
| Memory per Record | Average memory consumed per record | bytes |
| Final Query Latency | Query response time at maximum load | ms |
| Health Score | System health metric (0.0-1.0) | score |
| Maintenance Events | Number of detected maintenance operations | count |
| Errors Encountered | Number of errors during testing | count |

## Technical Implementation

### Memory Pressure Test Process

```python
def run_benchmark(self, record_batch_size=1000):
    # Initial metrics
    initial_metrics = self.measure_performance()
    
    # Progressive load phase
    while total_records < self.max_records and not saturation_detected:
        # Generate batch with varying importance distributions
        records = [self.generate_record() for _ in range(record_batch_size)]
        
        # Measure before batch
        before_metrics = self.measure_performance()
        
        # Store batch and update metrics
        try:
            batch_start = time.time()
            store_records(records)
            total_records += record_batch_size
            
            # Measure after batch
            after_metrics = self.measure_performance()
            
            # Detect maintenance events
            maintenance_event = self.detect_maintenance_events(before_metrics, after_metrics)
            
            # Check for saturation
            saturation_detection_logic()
            
            # Trigger explicit maintenance at intervals
            if total_records % maintenance_interval == 0:
                self.memory.perform_maintenance()
                
        except Exception as e:
            # Log errors and break on failure
            self.errors.append({"time": time.time(), "error": str(e)})
            break
            
    # Test recovery if system saturated
    if saturation_detected:
        test_emergency_recovery()
    
    return results
```

### Self-Maintenance Detection

The benchmark detects maintenance events by looking for significant changes in key metrics:

```python
def detect_maintenance_events(self, before_metrics, after_metrics):
    """Detect if a maintenance event occurred based on metrics changes"""
    maintenance_event = False
    reason = []
    
    # Memory reduction (garbage collection or compaction)
    if after_metrics["memory_usage"] < before_metrics["memory_usage"] * 0.9:
        maintenance_event = True
        reason.append("memory_reduction")
    
    # Speed improvement (index optimization)
    if before_metrics["latencies"] and after_metrics["latencies"]:
        before_avg = sum(before_metrics["latencies"]) / max(1, len(before_metrics["latencies"]))
        after_avg = sum(after_metrics["latencies"]) / max(1, len(after_metrics["latencies"]))
        
        if before_avg > 0.0001 and after_avg < before_avg * 0.8:  # 20% or more improvement
            maintenance_event = True
            reason.append("speed_improvement")
    
    # Health score improvement
    if after_metrics["health_score"] > before_metrics["health_score"] + 0.1:
        maintenance_event = True
        reason.append("health_improvement")
    
    if maintenance_event:
        return {"time": time.time(), "reasons": reason}
    return None
```

## Running the Benchmark

To run just this benchmark:

```bash
cd agno_vs_neuroca_bench
python run_all_benchmarks.py --pressure --max-records 20000 --batch-size 1000
```

Parameters:
- `--max-records`: Maximum number of records to add (default: 20000)
- `--batch-size`: Number of records to add in each batch (default: 1000)

Results are saved to:
- `benchmark_results/memory_pressure_report.md`: Detailed analysis
- `benchmark_results/neuroca_pressure_metrics.png`: Visual performance graphs
- `benchmark_results/agno_pressure_metrics.png`: Visual performance graphs

## Expected Results

When running this benchmark, you should observe:

1. **Neuroca**:
   - Successfully handles thousands of records with stable performance
   - Automatically performs maintenance when approaching capacity
   - Maintains good health metrics even under pressure
   - Selectively removes lower-importance memories during maintenance
   - Recovers successfully from high-pressure states

2. **Agno**:
   - May encounter errors under high memory pressure
   - Lacks explicit maintenance capabilities
   - May show degrading performance as record count increases

## Result Interpretation

The results highlight fundamental architectural differences:

1. **Neuroca's Architecture**:
   - Multi-tiered memory system with importance-based management
   - Automatic garbage collection of low-importance memories
   - Health monitoring systems that maintain stability
   - Recovery mechanisms for high-pressure situations
   
2. **Agno's Architecture**:
   - Simpler memory structure without explicit maintenance
   - Potentially limited capacity under continuous growth
   - May require manual intervention for long-term stability

These differences make Neuroca better suited for applications requiring indefinite operation with continuous memory growth.

## Example Results

From a sample run with 5,000 records:

```
===== Memory Pressure Benchmark Comparison =====

Metric                         Neuroca         Agno            Difference
---------------------------------------------------------------------------
Max Records                    5,000           0               +500000.0%
Final Memory (MB)              101.4           122.8           -17.4%
Final Query Latency (ms)       5.18            6.41            -19.2%
Final Health Score             0.74            0.70            +0.04
Maintenance Events             1               0               +1
Errors Encountered             0               1               -1
```

The above results demonstrate Neuroca's ability to handle continuous growth with self-maintenance, while Agno encountered errors preventing successful operation under the same conditions.

## Visualizations

The benchmark generates three visualization graphs:

1. **Memory Usage vs. Records**: Shows memory consumption growth pattern
2. **Query Latency vs. Records**: Shows response time under increasing load
3. **Health Score vs. Records**: Shows stability metrics over time

Red vertical lines on the graphs indicate detected maintenance events.

![Neuroca Memory Pressure](../agno_vs_neuroca_bench/benchmark_results/neuroca_pressure_metrics.png)
