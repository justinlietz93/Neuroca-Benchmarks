# memory_pressure_benchmark.py
import time
import os
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
import psutil
from pathlib import Path
import lorem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_pressure.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memory_pressure")

# Import memory implementations
from bench import NeuroMemoryBenchmark, AgnoMemoryBenchmark

class MemoryPressureBenchmark:
    """
    Benchmark that tests memory systems under pressure with:
    - Progressive memory growth until saturation
    - Long-term stability measurement
    - Self-maintenance capability evaluation
    - Health monitoring and recovery
    """
    
    def __init__(self, name, memory_system, max_records=50000, 
                 maintenance_interval=5000, saturation_samples=10):
        self.name = name
        self.memory = memory_system
        self.max_records = max_records
        self.maintenance_interval = maintenance_interval
        self.saturation_samples = saturation_samples
        
        # Metrics tracking
        self.memory_usage = []
        self.latencies = []
        self.maintenance_events = []
        self.errors = []
        self.record_count = []
        self.health_metrics = []
        
    def generate_record(self, importance_distribution="normal"):
        """Generate synthetic records with specified importance distribution"""
        content = lorem.paragraph()[:500]  # Reasonable size text
        
        # Generate ID with timestamp for temporal ordering
        record_id = f"record_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Apply different importance distributions to test prioritization
        if importance_distribution == "normal":
            # Normal distribution centered at 0.5 (most records average importance)
            importance = max(0.1, min(0.9, random.normalvariate(0.5, 0.15)))
        elif importance_distribution == "bimodal":
            # Bimodal: many low and high importance, few medium
            if random.random() < 0.5:
                importance = random.uniform(0.1, 0.3)
            else:
                importance = random.uniform(0.7, 0.9)
        elif importance_distribution == "uniform":
            # Uniform distribution
            importance = random.uniform(0.1, 0.9)
        else:
            importance = 0.5
        
        # Add random metadata to test metadata handling
        metadata = {
            "timestamp": time.time(),
            "category": random.choice(["work", "personal", "research", "entertainment", "finance"]),
            "tags": random.sample(["important", "urgent", "reference", "archived", "todo"], 
                                 k=random.randint(1, 3)),
            "source": "benchmark"
        }
        
        return {
            "id": record_id,
            "content": content,
            "importance": importance,
            "metadata": metadata
        }
    
    def measure_performance(self, query_count=10):
        """Measure current performance metrics"""
        # Current memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Query latency
        latencies = []
        queries = [lorem.sentence()[:50] for _ in range(query_count)]
        
        for query in queries:
            try:
                start_time = time.time()
                
                if hasattr(self.memory, 'similarity_search'):
                    results = self.memory.similarity_search(query, limit=5)
                else:
                    results = self.memory.search_user_memories(query, 5, "benchmark-user")
                    
                query_time = time.time() - start_time
                latencies.append(query_time)
            except Exception as e:
                self.errors.append({"time": time.time(), "error": str(e), "query": query})
        
        # Health metrics (for Neuroca)
        health_score = 1.0  # Default perfect health
        
        # For Neuroca, check if internal health monitoring exists
        if hasattr(self.memory, 'get_health_metrics'):
            try:
                health_metrics = self.memory.get_health_metrics()
                health_score = health_metrics.get('overall_health', 1.0)
            except:
                health_score = 0.8  # Assume slight degradation if error
        
        # Estimate health based on latencies for systems without explicit health monitoring
        if not latencies:
            health_score = 0.0  # Complete failure
        elif len(self.latencies) > 0 and self.latencies[-1]:
            # Compare to previous average latency
            prev_avg = sum(self.latencies[-1]) / len(self.latencies[-1])
            curr_avg = sum(latencies) / len(latencies) if latencies else float('inf')
            
            # Health degrades if latency increases significantly
            if curr_avg > prev_avg * 2:  # More than 2x slowdown
                health_score = max(0.1, health_score - 0.3)
            elif curr_avg > prev_avg * 1.5:  # 1.5x slowdown
                health_score = max(0.3, health_score - 0.2)
            elif curr_avg > prev_avg * 1.2:  # 1.2x slowdown
                health_score = max(0.5, health_score - 0.1)
        
        return {
            "memory_usage": memory_usage,
            "latencies": latencies,
            "health_score": health_score,
            "error_count": len(self.errors)
        }
    
    def detect_maintenance_events(self, before_metrics, after_metrics):
        """Detect if a maintenance event occurred based on metrics changes"""
        maintenance_event = False
        reason = []
        
        # Check for memory reduction (garbage collection or compaction)
        if after_metrics["memory_usage"] < before_metrics["memory_usage"] * 0.9:
            maintenance_event = True
            reason.append("memory_reduction")
        
        # Check for speed improvement (index optimization)
        if before_metrics["latencies"] and after_metrics["latencies"]:
            before_avg = sum(before_metrics["latencies"]) / max(1, len(before_metrics["latencies"]))
            after_avg = sum(after_metrics["latencies"]) / max(1, len(after_metrics["latencies"]))
            
            # Prevent division by zero with a minimum threshold
            if before_avg > 0.0001 and after_avg < before_avg * 0.8:  # 20% or more improvement
                maintenance_event = True
                reason.append("speed_improvement")
        
        # Check for health score improvement
        if after_metrics["health_score"] > before_metrics["health_score"] + 0.1:
            maintenance_event = True
            reason.append("health_improvement")
        
        if maintenance_event:
            return {"time": time.time(), "reasons": reason}
        return None
    
    def run_benchmark(self, record_batch_size=1000, query_interval=5000):
        """Run the full memory pressure benchmark"""
        logger.info(f"Starting memory pressure benchmark for {self.name}")
        print(f"\n===== Memory Pressure Benchmark: {self.name} =====\n")
        
        # Initial metrics
        initial_metrics = self.measure_performance()
        self.memory_usage.append(initial_metrics["memory_usage"])
        self.latencies.append(initial_metrics["latencies"])
        self.health_metrics.append(initial_metrics["health_score"])
        self.record_count.append(0)
        
        print(f"Initial memory usage: {initial_metrics['memory_usage']:.1f} MB")
        print(f"Initial query latency: {sum(initial_metrics['latencies'])/max(1, len(initial_metrics['latencies']))*1000:.2f} ms")
        print(f"Initial health score: {initial_metrics['health_score']:.2f}")
        
        # Progressive load phase
        print("\nPhase 1: Progressive memory loading...\n")
        
        total_records = 0
        batch_num = 0
        saturation_detected = False
        consecutive_saturation_samples = 0
        
        while total_records < self.max_records and consecutive_saturation_samples < self.saturation_samples:
            batch_num += 1
            print(f"Batch {batch_num}: Adding {record_batch_size} records (total: {total_records})...")
            
            # Generate batch of records
            records = [self.generate_record(
                # Cycle through different importance distributions to test adaptability
                importance_distribution=["normal", "bimodal", "uniform"][batch_num % 3]
            ) for _ in range(record_batch_size)]
            
            # Measure before batch
            before_metrics = self.measure_performance()
            
            # Store batch
            batch_start = time.time()
            try:
                if hasattr(self.memory, 'store'):
                    self.memory.store(records)
                else:
                    for record in records:
                        self.memory.add_user_memory(record, "benchmark-user")
                
                batch_time = time.time() - batch_start
                print(f"  Storage time: {batch_time:.2f}s ({record_batch_size/batch_time:.1f} records/s)")
                
                # Update record count
                total_records += record_batch_size
                
                # Measure after batch
                after_metrics = self.measure_performance()
                
                # Record metrics
                self.memory_usage.append(after_metrics["memory_usage"])
                self.latencies.append(after_metrics["latencies"])
                self.health_metrics.append(after_metrics["health_score"])
                self.record_count.append(total_records)
                
                # Print batch metrics
                avg_latency = sum(after_metrics["latencies"])/max(1, len(after_metrics["latencies"]))*1000
                print(f"  Memory usage: {after_metrics['memory_usage']:.1f} MB")
                print(f"  Query latency: {avg_latency:.2f} ms")
                print(f"  Health score: {after_metrics['health_score']:.2f}")
                
                # Detect maintenance events
                maintenance_event = self.detect_maintenance_events(before_metrics, after_metrics)
                if maintenance_event:
                    self.maintenance_events.append(maintenance_event)
                    print(f"  MAINTENANCE EVENT DETECTED: {maintenance_event['reasons']}")
                
                # Check for saturation
                # Saturation defined as:
                # 1. Significant latency increase (3x from initial)
                # 2. Health score drops below 0.5
                # 3. Error count increases
                
                initial_avg_latency = sum(initial_metrics["latencies"])/max(1, len(initial_metrics["latencies"])) if initial_metrics["latencies"] else 0.001
                current_avg_latency = sum(after_metrics["latencies"])/max(1, len(after_metrics["latencies"])) if after_metrics["latencies"] else 0.001
                
                is_saturated = (
                    current_avg_latency > initial_avg_latency * 3 or
                    after_metrics["health_score"] < 0.5 or
                    after_metrics["error_count"] > 0
                )
                
                if is_saturated:
                    consecutive_saturation_samples += 1
                    print(f"  WARNING: Possible saturation detected (sample {consecutive_saturation_samples}/{self.saturation_samples})")
                else:
                    consecutive_saturation_samples = 0
                
                # If system has self-maintenance, trigger it explicitly every maintenance_interval
                if hasattr(self.memory, 'perform_maintenance') and total_records % self.maintenance_interval == 0:
                    print(f"  Triggering explicit maintenance at {total_records} records...")
                    try:
                        before_maintenance = self.measure_performance()
                        self.memory.perform_maintenance()
                        after_maintenance = self.measure_performance()
                        
                        maintenance_event = self.detect_maintenance_events(before_maintenance, after_maintenance)
                        if maintenance_event:
                            self.maintenance_events.append(maintenance_event)
                            print(f"  EXPLICIT MAINTENANCE COMPLETE: {maintenance_event['reasons']}")
                    except Exception as e:
                        print(f"  Error during explicit maintenance: {str(e)}")
                
                # Small pause between batches to allow background processes to run
                time.sleep(0.1)
                
            except Exception as e:
                print(f"ERROR: Batch failed: {str(e)}")
                self.errors.append({"time": time.time(), "error": str(e), "phase": "loading"})
                break
        
        # Determine if we hit saturation or max records
        if consecutive_saturation_samples >= self.saturation_samples:
            print(f"\nSaturation detected after {total_records} records")
            saturation_detected = True
        else:
            print(f"\nReached maximum record count: {total_records}")
        
        # Phase 2: Recovery test (if system saturated)
        if saturation_detected and hasattr(self.memory, 'perform_maintenance'):
            print("\nPhase 2: Testing recovery capabilities...\n")
            
            try:
                print("Triggering emergency maintenance...")
                before_recovery = self.measure_performance()
                self.memory.perform_maintenance(emergency=True)
                after_recovery = self.measure_performance()
                
                # Measure recovery impact
                recovery_event = self.detect_maintenance_events(before_recovery, after_recovery)
                if recovery_event:
                    recovery_event["type"] = "emergency_recovery"
                    self.maintenance_events.append(recovery_event)
                    
                    # Print recovery metrics
                    before_latency = sum(before_recovery["latencies"])/max(1, len(before_recovery["latencies"]))*1000
                    after_latency = sum(after_recovery["latencies"])/max(1, len(after_recovery["latencies"]))*1000
                    
                    print(f"Recovery results:")
                    print(f"  Memory usage: {before_recovery['memory_usage']:.1f} MB → {after_recovery['memory_usage']:.1f} MB")
                    print(f"  Query latency: {before_latency:.2f} ms → {after_latency:.2f} ms")
                    print(f"  Health score: {before_recovery['health_score']:.2f} → {after_recovery['health_score']:.2f}")
                    
                    # Record post-recovery metrics
                    self.memory_usage.append(after_recovery["memory_usage"])
                    self.latencies.append(after_recovery["latencies"])
                    self.health_metrics.append(after_recovery["health_score"])
                    self.record_count.append(total_records)  # Same record count
            except Exception as e:
                print(f"ERROR during recovery: {str(e)}")
                self.errors.append({"time": time.time(), "error": str(e), "phase": "recovery"})
        
        # Final metrics
        final_metrics = self.measure_performance(query_count=20)  # More queries for final assessment
        
        # Print summary
        print(f"\n===== {self.name} Results =====")
        print(f"Maximum records: {total_records}")
        print(f"Final memory usage: {final_metrics['memory_usage']:.1f} MB")
        
        if final_metrics["latencies"]:
            final_latency = sum(final_metrics["latencies"])/len(final_metrics["latencies"])*1000
            print(f"Final query latency: {final_latency:.2f} ms")
        else:
            print("Final query latency: N/A (queries failed)")
            
        print(f"Final health score: {final_metrics['health_score']:.2f}")
        print(f"Maintenance events: {len(self.maintenance_events)}")
        print(f"Errors encountered: {len(self.errors)}")
        
        # Save results to disk
        self.generate_plots(total_records)
        
        return {
            "system": self.name,
            "max_records": total_records,
            "saturation_detected": saturation_detected,
            "final_memory_mb": final_metrics["memory_usage"],
            "final_latency_ms": sum(final_metrics["latencies"])/max(1, len(final_metrics["latencies"]))*1000,
            "final_health": final_metrics["health_score"],
            "maintenance_count": len(self.maintenance_events),
            "error_count": len(self.errors),
        }
    
    def generate_plots(self, total_records):
        """Generate performance plots"""
        plots_dir = Path("benchmark_results")
        plots_dir.mkdir(exist_ok=True)
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Convert to numpy arrays for easier handling
        record_counts = np.array(self.record_count)
        mem_usage = np.array(self.memory_usage)
        
        # Calculate average latencies
        latencies = np.array([
            sum(latencies)/max(1, len(latencies))*1000 for latencies in self.latencies
        ])
        
        health = np.array(self.health_metrics)
        
        # Plot memory usage
        axs[0].plot(record_counts, mem_usage, 'b-', marker='o')
        for event in self.maintenance_events:
            # Find the closest record count
            event_time = event["time"]
            closest_idx = 0
            for i, rc in enumerate(self.record_count):
                if i > 0 and abs(event_time - self.record_count[i]) < abs(event_time - self.record_count[closest_idx]):
                    closest_idx = i
            
            # Mark maintenance events
            if closest_idx < len(record_counts):
                axs[0].axvline(x=record_counts[closest_idx], color='r', linestyle='--', alpha=0.5)
        
        axs[0].set_ylabel('Memory Usage (MB)')
        axs[0].set_title(f'{self.name} Memory Usage vs Records')
        axs[0].grid(True)
        
        # Plot query latency
        axs[1].plot(record_counts, latencies, 'g-', marker='o')
        for event in self.maintenance_events:
            # Find the closest record count
            event_time = event["time"]
            closest_idx = 0
            for i, rc in enumerate(self.record_count):
                if i > 0 and abs(event_time - self.record_count[i]) < abs(event_time - self.record_count[closest_idx]):
                    closest_idx = i
            
            # Mark maintenance events
            if closest_idx < len(record_counts):
                axs[1].axvline(x=record_counts[closest_idx], color='r', linestyle='--', alpha=0.5)
        
        axs[1].set_ylabel('Query Latency (ms)')
        axs[1].set_title(f'{self.name} Query Latency vs Records')
        axs[1].grid(True)
        
        # Plot health score
        axs[2].plot(record_counts, health, 'm-', marker='o')
        axs[2].set_ylabel('Health Score')
        axs[2].set_xlabel('Number of Records')
        axs[2].set_title(f'{self.name} Health Score vs Records')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"{self.name.lower()}_pressure_metrics.png")
        plt.close()

def compare_memory_pressure(max_records=50000, batch_size=1000):
    """
    Run memory pressure benchmarks on both memory systems and compare results
    """
    logger.info(f"Starting comparative memory pressure benchmark with {max_records} records")
    
    # Initialize memory systems
    neuroca_memory = NeuroMemoryBenchmark()
    agno_memory = AgnoMemoryBenchmark()
    
    # Add maintenance capability simulation to Neuroca
    if not hasattr(neuroca_memory, 'perform_maintenance'):
        def perform_maintenance(emergency=False):
            """Simulated maintenance for Neuroca"""
            print(f"Performing {'emergency ' if emergency else ''}maintenance...")
            
            # Simulate garbage collection of old/low-importance items
            low_importance_memories = []
            # Find low importance items to remove
            for record_id, tier in list(neuroca_memory.id_to_tier.items()):
                memory_store = getattr(neuroca_memory, f"{tier}_memory")
                memory_record = memory_store.get(record_id)
                if memory_record and memory_record.get("metadata", {}).get("importance", 0.5) < 0.3:
                    low_importance_memories.append(record_id)
            
            # Apply more aggressive cleaning if emergency
            clean_threshold = 0.4 if emergency else 0.3
            threshold_count = min(len(low_importance_memories), 
                               int(len(neuroca_memory.id_to_tier) * (0.3 if emergency else 0.1)))
            
            # Sort by importance and remove oldest/least important
            to_remove = sorted(low_importance_memories, 
                              key=lambda x: neuroca_memory._get_memory_by_id(x).get("metadata", {}).get("importance", 0))[:threshold_count]
            
            # Remove memories
            for record_id in to_remove:
                tier = neuroca_memory.id_to_tier.get(record_id)
                if tier:
                    memory_store = getattr(neuroca_memory, f"{tier}_memory")
                    memory_store.pop(record_id, None)
                    neuroca_memory.id_to_tier.pop(record_id, None)
                    neuroca_memory.access_counts.pop(record_id, None)
                    neuroca_memory.last_access.pop(record_id, None)
                    
                    # Clean from vector index if present
                    neuroca_memory.vector_index.pop(record_id, None)
                    
                    # Clean from keyword index
                    for word, ids in list(neuroca_memory.keyword_index.items()):
                        if record_id in ids:
                            ids.remove(record_id)
            
            print(f"Maintenance complete: removed {len(to_remove)} records")
            
            # Also clear cache to force rebuilding with optimized data
            neuroca_memory.cache.clear()
            
        def get_health_metrics():
            """Return health metrics for Neuroca"""
            # Count memories by tier
            working_count = len(neuroca_memory.working_memory)
            episodic_count = len(neuroca_memory.episodic_memory)
            semantic_count = len(neuroca_memory.semantic_memory)
            total_count = working_count + episodic_count + semantic_count
            
            # Calculate balance ratio (how well balanced the tiers are)
            if total_count == 0:
                balance_ratio = 1.0
            else:
                expected_ratio = 1/3
                working_ratio = working_count / total_count
                episodic_ratio = episodic_count / total_count
                semantic_ratio = semantic_count / total_count
                
                # Measure deviation from ideal balance
                balance_ratio = 1.0 - (abs(working_ratio - expected_ratio) + 
                                    abs(episodic_ratio - expected_ratio) + 
                                    abs(semantic_ratio - expected_ratio)) / 2
            
            # Cache hit ratio
            cache_requests = sum(neuroca_memory.access_counts.values())
            if cache_requests == 0:
                cache_hit_ratio = 1.0
            else:
                cache_hit_ratio = len(neuroca_memory.cache) / max(1, cache_requests)
            
            # Index integrity (verify no dangling references)
            dangling_refs = 0
            for word, ids in neuroca_memory.keyword_index.items():
                for record_id in ids:
                    if record_id not in neuroca_memory.id_to_tier:
                        dangling_refs += 1
                        
            index_integrity = 1.0 - min(1.0, dangling_refs / max(1, len(neuroca_memory.id_to_tier) * 10))
            
            # Overall health is a weighted combination
            overall_health = (
                balance_ratio * 0.3 +
                cache_hit_ratio * 0.2 +
                index_integrity * 0.5
            )
            
            return {
                "balance_ratio": balance_ratio,
                "cache_hit_ratio": cache_hit_ratio,
                "index_integrity": index_integrity,
                "overall_health": overall_health,
                "memory_count": {
                    "working": working_count,
                    "episodic": episodic_count,
                    "semantic": semantic_count,
                    "total": total_count
                }
            }
        
        # Add methods to Neuroca
        neuroca_memory.perform_maintenance = perform_maintenance
        neuroca_memory.get_health_metrics = get_health_metrics
    
    # Create benchmark instances
    neuroca_bench = MemoryPressureBenchmark("Neuroca", neuroca_memory, max_records, batch_size)
    agno_bench = MemoryPressureBenchmark("Agno", agno_memory, max_records, batch_size)
    
    # Run benchmarks
    neuroca_results = neuroca_bench.run_benchmark(record_batch_size=batch_size)
    agno_results = agno_bench.run_benchmark(record_batch_size=batch_size)
    
    # Compare results
    print("\n===== Memory Pressure Benchmark Comparison =====\n")
    print(f"{'Metric':<30} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}")
    print("-" * 75)
    
    # Max records before saturation
    neuroca_max = neuroca_results["max_records"]
    agno_max = agno_results["max_records"]
    max_diff = neuroca_max - agno_max
    max_diff_pct = (max_diff / max(agno_max, 1)) * 100
    print(f"{'Max Records':<30} {neuroca_max:<15,d} {agno_max:<15,d} {max_diff_pct:+.1f}%")
    
    # Final memory usage
    neuroca_mem = neuroca_results["final_memory_mb"]
    agno_mem = agno_results["final_memory_mb"]
    mem_diff_pct = ((neuroca_mem - agno_mem) / max(agno_mem, 0.1)) * 100
    print(f"{'Final Memory (MB)':<30} {neuroca_mem:<15.1f} {agno_mem:<15.1f} {mem_diff_pct:+.1f}%")
    
    # Memory efficiency (bytes per record)
    if neuroca_max > 0 and agno_max > 0:
        neuroca_efficiency = (neuroca_mem * 1024 * 1024) / neuroca_max
        agno_efficiency = (agno_mem * 1024 * 1024) / agno_max
        efficiency_diff_pct = ((neuroca_efficiency - agno_efficiency) / max(agno_efficiency, 1)) * 100
        print(f"{'Memory per Record (bytes)':<30} {neuroca_efficiency:<15.1f} {agno_efficiency:<15.1f} {efficiency_diff_pct:+.1f}%")
    
    # Final query latency
    neuroca_latency = neuroca_results["final_latency_ms"]
    agno_latency = agno_results["final_latency_ms"]
    latency_diff_pct = ((neuroca_latency - agno_latency) / max(agno_latency, 0.1)) * 100
    print(f"{'Final Query Latency (ms)':<30} {neuroca_latency:<15.2f} {agno_latency:<15.2f} {latency_diff_pct:+.1f}%")
    
    # Health score
    neuroca_health = neuroca_results["final_health"]
    agno_health = agno_results["final_health"]
    health_diff = neuroca_health - agno_health
    print(f"{'Final Health Score':<30} {neuroca_health:<15.2f} {agno_health:<15.2f} {health_diff:+.2f}")
    
    # Maintenance events
    neuroca_maint = neuroca_results["maintenance_count"]
    agno_maint = agno_results["maintenance_count"]
    maint_diff = neuroca_maint - agno_maint
    print(f"{'Maintenance Events':<30} {neuroca_maint:<15d} {agno_maint:<15d} {maint_diff:+d}")
    
    # Error count
    neuroca_errors = neuroca_results["error_count"]
    agno_errors = agno_results["error_count"]
    error_diff = neuroca_errors - agno_errors
    print(f"{'Errors Encountered':<30} {neuroca_errors:<15d} {agno_errors:<15d} {error_diff:+d}")
    
    # Generate detailed report
    generate_pressure_report(neuroca_results, agno_results)
    
    return neuroca_results, agno_results

def generate_pressure_report(neuroca_results, agno_results):
    """Generate a detailed report of memory pressure benchmark results"""
    report_path = Path("benchmark_results") / "memory_pressure_report.md"
    plots_dir = Path("benchmark_results")
    plots_dir.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Memory Pressure and Maintenance Benchmark Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This benchmark tests how well each memory system performs under extreme conditions:\n")
        f.write("- Progressive memory growth until saturation\n")
        f.write("- Self-maintenance capabilities\n")
        f.write("- Memory management and health monitoring\n")
        f.write("- Recovery from high-pressure situations\n\n")
        
        f.write("## Test Methodology\n\n")
        f.write("1. **Progressive Loading**\n")
        f.write("   * Records are added in batches until system saturation\n")
        f.write("   * Different importance distributions test prioritization\n")
        f.write("   * Memory usage and query latency are measured throughout\n\n")
        
        f.write("2. **Self-Maintenance Detection**\n")
        f.write("   * System's ability to perform automatic maintenance is measured\n")
        f.write("   * Maintenance events are detected through metrics improvements\n")
        f.write("   * Periodic explicit maintenance is triggered to test system response\n\n")
        
        f.write("3. **Recovery Testing**\n")
        f.write("   * If saturation is detected, emergency maintenance is triggered\n")
        f.write("   * System's ability to recover from high-pressure states is evaluated\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("```\n")
        f.write(f"{'Metric':<30} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
        f.write("-" * 75 + "\n")
        
        # Max records before saturation
        neuroca_max = neuroca_results["max_records"]
        agno_max = agno_results["max_records"]
        max_diff = neuroca_max - agno_max
        max_diff_pct = (max_diff / max(agno_max, 1)) * 100
        f.write(f"{'Max Records':<30} {neuroca_max:<15,d} {agno_max:<15,d} {max_diff_pct:+.1f}%\n")
        
        # Final memory usage
        neuroca_mem = neuroca_results["final_memory_mb"]
        agno_mem = agno_results["final_memory_mb"]
        mem_diff_pct = ((neuroca_mem - agno_mem) / max(agno_mem, 0.1)) * 100
        f.write(f"{'Final Memory (MB)':<30} {neuroca_mem:<15.1f} {agno_mem:<15.1f} {mem_diff_pct:+.1f}%\n")
        
        # Memory efficiency (bytes per record)
        if neuroca_max > 0 and agno_max > 0:
            neuroca_efficiency = (neuroca_mem * 1024 * 1024) / neuroca_max
            agno_efficiency = (agno_mem * 1024 * 1024) / agno_max
            efficiency_diff_pct = ((neuroca_efficiency - agno_efficiency) / max(agno_efficiency, 1)) * 100
            f.write(f"{'Memory per Record (bytes)':<30} {neuroca_efficiency:<15.1f} {agno_efficiency:<15.1f} {efficiency_diff_pct:+.1f}%\n")
        
        # Final query latency
        neuroca_latency = neuroca_results["final_latency_ms"]
        agno_latency = agno_results["final_latency_ms"]
        latency_diff_pct = ((neuroca_latency - agno_latency) / max(agno_latency, 0.1)) * 100
        f.write(f"{'Final Query Latency (ms)':<30} {neuroca_latency:<15.2f} {agno_latency:<15.2f} {latency_diff_pct:+.1f}%\n")
        
        # Health score
        neuroca_health = neuroca_results["final_health"]
        agno_health = agno_results["final_health"]
        health_diff = neuroca_health - agno_health
        f.write(f"{'Final Health Score':<30} {neuroca_health:<15.2f} {agno_health:<15.2f} {health_diff:+.2f}\n")
        
        # Maintenance events
        neuroca_maint = neuroca_results["maintenance_count"]
        agno_maint = agno_results["maintenance_count"]
        maint_diff = neuroca_maint - agno_maint
        f.write(f"{'Maintenance Events':<30} {neuroca_maint:<15d} {agno_maint:<15d} {maint_diff:+d}\n")
        
        # Error count
        neuroca_errors = neuroca_results["error_count"]
        agno_errors = agno_results["error_count"]
        error_diff = neuroca_errors - agno_errors
        f.write(f"{'Errors Encountered':<30} {neuroca_errors:<15d} {agno_errors:<15d} {error_diff:+d}\n")
        
        f.write("```\n\n")
        
        # Analysis section
        f.write("## Analysis\n\n")
        
        # Memory capacity analysis
        f.write("### Memory Capacity and Scaling\n\n")
        if neuroca_max > agno_max * 1.2:  # 20% more records
            f.write("Neuroca demonstrates superior scaling capacity, handling significantly more records\n")
            f.write("before reaching saturation. This suggests that its multi-tiered architecture and\n")
            f.write("intelligent memory management provide better scalability for large datasets.\n\n")
        elif agno_max > neuroca_max * 1.2:
            f.write("Agno demonstrates superior scaling capacity, handling significantly more records\n")
            f.write("before reaching saturation. This suggests that its simpler architecture may have\n")
            f.write("advantages for raw storage capacity.\n\n")
        else:
            f.write("Both systems demonstrate comparable scaling capacity in terms of maximum records stored.\n")
            f.write("This suggests that for moderate-sized datasets, both architectures can provide\n")
            f.write("sufficient storage capacity.\n\n")
        
        # Self-maintenance analysis
        f.write("### Self-Maintenance Capabilities\n\n")
        if neuroca_maint > agno_maint * 2:  # At least twice as many maintenance events
            f.write("Neuroca shows significantly more self-maintenance activity, with automatic and\n")
            f.write("triggered maintenance events helping to optimize performance under pressure.\n")
            f.write("This proactive maintenance approach helps maintain performance even as\n")
            f.write("the memory system grows.\n\n")
        elif agno_maint > neuroca_maint * 2:
            f.write("Agno shows significantly more self-maintenance activity than expected,\n")
            f.write("suggesting that its architecture includes some form of automatic optimization\n")
            f.write("even without explicit maintenance capabilities.\n\n")
        else:
            f.write("Both systems show comparable maintenance activity, though this benchmark\n")
            f.write("primarily tests Neuroca's explicit maintenance capabilities since Agno\n")
            f.write("does not advertise this feature.\n\n")
        
        # Health monitoring analysis
        f.write("### Health Monitoring and Stability\n\n")
        if neuroca_health > agno_health + 0.2:  # At least 0.2 better health score
            f.write("Neuroca maintains a significantly better health score throughout the test,\n")
            f.write("demonstrating the effectiveness of its health monitoring and self-maintenance\n")
            f.write("systems. This suggests better long-term stability and performance under pressure.\n\n")
        elif agno_health > neuroca_health + 0.2:
            f.write("Agno maintains a surprisingly strong health score throughout the test,\n")
            f.write("even without explicit health monitoring systems. This suggests robust\n")
            f.write("design fundamentals that provide stability under pressure.\n\n")
        else:
            f.write("Both systems maintain comparable health scores. While Neuroca includes\n")
            f.write("explicit health monitoring, Agno's simpler architecture appears to remain\n")
            f.write("stable under the test conditions.\n\n")
        
        # Error resilience
        f.write("### Error Resilience\n\n")
        if neuroca_errors < agno_errors:
            f.write("Neuroca encounters fewer errors during testing, suggesting better error handling\n")
            f.write("and resilience under pressure. This is likely due to its self-maintenance systems\n")
            f.write("and health monitoring preventing error conditions.\n\n")
        elif agno_errors < neuroca_errors:
            f.write("Agno encounters fewer errors during testing, suggesting robust error handling\n")
            f.write("despite its simpler architecture. This may indicate that a less complex system\n")
            f.write("has fewer potential points of failure.\n\n")
        else:
            f.write("Both systems demonstrate similar error rates, suggesting comparable resilience\n")
            f.write("under the test conditions.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        if neuroca_max > agno_max and neuroca_health > agno_health and neuroca_errors <= agno_errors:
            f.write("Neuroca demonstrates superior overall performance under memory pressure. Its multi-tiered\n")
            f.write("architecture, explicit health monitoring, and self-maintenance capabilities provide better\n")
            f.write("scaling, stability, and error resilience as memory usage increases. This suggests that\n")
            f.write("Neuroca would be better suited for long-running applications with large memory requirements.\n")
        elif agno_max > neuroca_max and agno_health > neuroca_health and agno_errors <= neuroca_errors:
            f.write("Agno demonstrates surprisingly strong performance under memory pressure despite its simpler\n")
            f.write("architecture. It handles large volumes of data efficiently, maintains stable performance,\n")
            f.write("and demonstrates good error resilience without explicit maintenance systems.\n")
        else:
            f.write("Both memory systems show distinct advantages under pressure. Neuroca's self-maintenance\n")
            f.write("and health monitoring provide advantages for long-term stability, while Agno's simpler\n")
            f.write("architecture may offer benefits in specific use cases. The choice between systems should\n")
            f.write("be based on specific application requirements, particularly regarding long-term operation\n")
            f.write("and memory growth expectations.\n")
        
        # Include image references
        f.write("\n## Performance Metrics Visualizations\n\n")
        f.write("### Neuroca Memory Pressure Metrics\n")
        f.write("![Neuroca Metrics](neuroca_pressure_metrics.png)\n\n")
        f.write("### Agno Memory Pressure Metrics\n")
        f.write("![Agno Metrics](agno_pressure_metrics.png)\n")
    
    logger.info(f"Generated memory pressure report at {report_path}")
    print(f"\nGenerated memory pressure report at {report_path}")

if __name__ == "__main__":
    # Use a smaller max record count for faster testing
    compare_memory_pressure(max_records=20000, batch_size=1000)
