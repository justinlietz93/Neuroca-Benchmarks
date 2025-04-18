# run_all_benchmarks.py
import time
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("unified_benchmark")

# Import individual benchmarks
from bench import compare_benchmarks
from llm_benchmark import compare_llm_benchmarks
from conversation_benchmark import compare_conversation_systems
from memory_retention_benchmark import compare_memory_retention 
from task_oriented_benchmark import compare_task_benchmarks
from memory_pressure_benchmark import compare_memory_pressure
from multi_agent_benchmark import compare_multi_agent_benchmarks

def run_all_benchmarks(args):
    """Run all benchmarks and compile results"""
    logger.info("Starting unified benchmark suite")
    
    start_time = time.time()
    results = {}
    
    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    # Define benchmark functions and their parameters
    benchmarks = [
        {
            "name": "Basic Memory Operations",
            "function": compare_benchmarks,
            "enabled": args.basic,
            "args": {}
        },
        {
            "name": "LLM Integration",
            "function": compare_llm_benchmarks,
            "enabled": args.llm,
            "args": {}
        },
        {
            "name": "Conversation Context",
            "function": compare_conversation_systems,
            "enabled": args.conversation,
            "args": {}
        },
        {
            "name": "Memory Retention",
            "function": compare_memory_retention,
            "enabled": args.retention,
            "args": {}
        },
        {
            "name": "Task-Oriented",
            "function": compare_task_benchmarks,
            "enabled": args.task,
            "args": {"dataset_size": args.dataset_size, "query_count": args.query_count}
        },
        {
            "name": "Memory Pressure & Maintenance",
            "function": compare_memory_pressure,
            "enabled": args.pressure,
            "args": {"max_records": args.max_records, "batch_size": args.batch_size}
        },
        {
            "name": "Multi-Agent Collaboration",
            "function": compare_multi_agent_benchmarks,
            "enabled": args.multi_agent,
            "args": {"num_agents": args.agents, "simulation_time": args.sim_time, "task_count": args.tasks, "runs": args.runs}
        }
    ]
    
    # Run each enabled benchmark
    for benchmark in benchmarks:
        if benchmark["enabled"]:
            logger.info(f"Running {benchmark['name']} benchmark")
            print(f"\n{'='*20} RUNNING {benchmark['name'].upper()} BENCHMARK {'='*20}\n")
            
            try:
                benchmark_start = time.time()
                benchmark_results = benchmark["function"](**benchmark["args"])
                benchmark_time = time.time() - benchmark_start
                
                results[benchmark["name"]] = {
                    "results": benchmark_results,
                    "time": benchmark_time
                }
                
                logger.info(f"Completed {benchmark['name']} benchmark in {benchmark_time:.2f}s")
            except Exception as e:
                logger.error(f"Error in {benchmark['name']} benchmark: {str(e)}")
                print(f"Error in {benchmark['name']} benchmark: {str(e)}")
    
    # Generate unified report
    generate_unified_report(results, results_dir)
    
    total_time = time.time() - start_time
    logger.info(f"All benchmarks completed in {total_time:.2f}s")
    print(f"\nAll benchmarks completed in {total_time:.2f}s")
    print(f"Results saved to {results_dir}")

def generate_unified_report(results, results_dir):
    """Generate a unified report of all benchmark results"""
    report_path = results_dir / "unified_benchmark_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Unified Benchmark Report: Neuroca vs Agno\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compiles the results of multiple benchmarks designed to\n")
        f.write("compare the performance characteristics of the Neuroca and Agno memory systems.\n")
        f.write("Each benchmark tests different aspects of memory functionality in various scenarios.\n\n")
        
        f.write("## Benchmark Summary\n\n")
        
        if not results:
            f.write("No benchmarks were run.\n\n")
        else:
            f.write("The following benchmarks were executed:\n\n")
            for name in results.keys():
                f.write(f"- {name}\n")
            f.write("\n")
        
        # Include detailed results from each benchmark
        for name, data in results.items():
            f.write(f"## {name} Benchmark\n\n")
            
            # Add benchmark-specific results
            if name == "Basic Memory Operations":
                write_basic_benchmark_results(f, data["results"])
            elif name == "LLM Integration":
                write_llm_benchmark_results(f, data["results"])
            elif name == "Conversation Context":
                write_conversation_benchmark_results(f, data["results"])
            elif name == "Memory Retention":
                write_memory_retention_results(f, data["results"])
            elif name == "Task-Oriented":
                write_task_benchmark_results(f, data["results"])
            elif name == "Memory Pressure & Maintenance":
                write_memory_pressure_results(f, data["results"])
            elif name == "Multi-Agent Collaboration":
                write_multi_agent_results(f, data["results"])
            
            f.write(f"Benchmark completed in {data['time']:.2f} seconds.\n\n")
        
        # Conclusion section
        f.write("## Conclusion\n\n")
        f.write("The benchmarks provide a comprehensive view of how Neuroca and Agno perform\n")
        f.write("across different memory-related tasks. Key findings include:\n\n")
        
        if "Basic Memory Operations" in results:
            neuroca_results, agno_results = results["Basic Memory Operations"]["results"]
            neuroca_insert = neuroca_results.get("insert_time", 0)
            agno_insert = agno_results.get("insert_time", 0)
            neuroca_query = neuroca_results.get("p50_ms", 0)
            agno_query = agno_results.get("p50_ms", 0)
            
            insert_diff = "faster" if neuroca_insert < agno_insert else "slower"
            query_diff = "faster" if neuroca_query < agno_query else "slower"
            
            f.write(f"- **Memory Operations**: Neuroca is {insert_diff} at inserting data and {query_diff} at querying data.\n")
        
        if "Task-Oriented" in results:
            neuroca_results, agno_results = results["Task-Oriented"]["results"]
            if neuroca_results["accuracy"] > agno_results["accuracy"]:
                f.write("- **Task Accuracy**: Neuroca achieved higher accuracy in task-oriented retrieval.\n")
            elif neuroca_results["accuracy"] < agno_results["accuracy"]:
                f.write("- **Task Accuracy**: Agno achieved higher accuracy in task-oriented retrieval.\n")
            else:
                f.write("- **Task Accuracy**: Both systems achieved similar accuracy in task-oriented retrieval.\n")
        
        if "Memory Retention" in results:
            neuroca_results, agno_results = results["Memory Retention"]["results"]
            if neuroca_results["accuracy"] > agno_results["accuracy"]:
                f.write("- **Memory Retention**: Neuroca better retained important information among noise.\n")
            elif neuroca_results["accuracy"] < agno_results["accuracy"]:
                f.write("- **Memory Retention**: Agno better retained important information among noise.\n")
            else:
                f.write("- **Memory Retention**: Both systems showed similar memory retention capabilities.\n")
        
        f.write("\nBased on these benchmarks, the most suitable memory system would depend on specific use case priorities:\n\n")
        
        f.write("- For applications prioritizing retrieval speed: Consider the system with faster query times.\n")
        f.write("- For applications with large datasets: Consider memory efficiency and scaling characteristics.\n")
        f.write("- For applications requiring high accuracy: Focus on the accuracy metrics from task-oriented benchmarks.\n")
        
        f.write("\nNote that these benchmarks use simulated implementations of both memory systems. \n")
        f.write("Results with the actual implementations may vary, especially regarding accuracy metrics.\n")
    
    logger.info(f"Unified report generated at {report_path}")

def write_basic_benchmark_results(f, results):
    """Write basic memory operations benchmark results to report"""
    neuroca_results, agno_results = results
    
    f.write("### Memory Operation Performance\n\n")
    f.write("```\n")
    f.write(f"{'Metric':<25} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
    f.write("-" * 70 + "\n")
    
    # Insert time
    neuroca_insert = neuroca_results.get("insert_time", 0)
    agno_insert = agno_results.get("insert_time", 0)
    insert_diff_pct = ((neuroca_insert - agno_insert) / max(agno_insert, 0.001)) * 100 if agno_insert else 0
    f.write(f"{'Insert Time (s)':<25} {neuroca_insert:<15.2f} {agno_insert:<15.2f} {insert_diff_pct:+.1f}%\n")
    
    # Memory usage
    neuroca_mem = neuroca_results.get("memory_mb", 0)
    agno_mem = agno_results.get("memory_mb", 0)
    mem_diff_pct = ((neuroca_mem - agno_mem) / max(agno_mem, 0.001)) * 100 if agno_mem else 0
    f.write(f"{'Memory Usage (MB)':<25} {neuroca_mem:<15.1f} {agno_mem:<15.1f} {mem_diff_pct:+.1f}%\n")
    
    # Query latency metrics
    for metric in ["p50_ms", "p95_ms", "p99_ms"]:
        neuroca_val = neuroca_results.get(metric, 0)
        agno_val = agno_results.get(metric, 0)
        val_diff_pct = ((neuroca_val - agno_val) / max(agno_val, 0.001)) * 100 if agno_val else 0
        
        metric_name = metric.replace("_ms", "").upper() + " Query (ms)"
        f.write(f"{metric_name:<25} {neuroca_val:<15.2f} {agno_val:<15.2f} {val_diff_pct:+.1f}%\n")
    
    f.write("```\n\n")
    
    # Analysis
    if neuroca_insert > agno_insert * 1.2:  # 20% slower
        f.write("Neuroca's insertion operations are significantly slower than Agno's, which is expected due to\n")
        f.write("the computation of vector embeddings during storage. This represents a tradeoff between\n")
        f.write("insertion speed and potential query performance.\n\n")
    else:
        f.write("Both systems show comparable insertion performance despite the theoretical overhead\n")
        f.write("of vector embedding computation in Neuroca.\n\n")
    
    p50_diff_pct = ((neuroca_results.get("p50_ms", 0) - agno_results.get("p50_ms", 0)) / 
                    max(agno_results.get("p50_ms", 0), 0.001)) * 100
    
    if p50_diff_pct < -15:  # 15% faster
        f.write("Neuroca demonstrates significantly faster query performance, which validates the\n")
        f.write("benefit of pre-computed vector embeddings for efficient similarity search.\n\n")
    elif p50_diff_pct > 15:  # 15% slower
        f.write("Agno demonstrates faster query performance, suggesting its retrieval approach\n")
        f.write("may be more efficient for certain types of queries.\n\n")
    else:
        f.write("Both systems demonstrate similar query performance within this benchmark context.\n\n")

def write_llm_benchmark_results(f, results):
    """Write LLM integration benchmark results to report"""
    neuroca_stats, agno_stats = results
    
    f.write("### LLM Augmentation Performance\n\n")
    f.write("```\n")
    f.write(f"{'Metric':<25} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
    f.write("-" * 70 + "\n")
    
    # Storage time
    neuroca_store = neuroca_stats.get("storage_time", 0)
    agno_store = agno_stats.get("storage_time", 0)
    store_diff_pct = ((neuroca_store - agno_store) / max(agno_store, 0.001)) * 100
    f.write(f"{'Storage Time (s)':<25} {neuroca_store:<15.2f} {agno_store:<15.2f} {store_diff_pct:+.1f}%\n")
    
    # Retrieval time
    neuroca_retrieve = neuroca_stats.get("avg_retrieval_ms", 0)
    agno_retrieve = agno_stats.get("avg_retrieval_ms", 0)
    retrieve_diff_pct = ((neuroca_retrieve - agno_retrieve) / max(agno_retrieve, 0.001)) * 100
    f.write(f"{'Retrieval Time (ms)':<25} {neuroca_retrieve:<15.1f} {agno_retrieve:<15.1f} {retrieve_diff_pct:+.1f}%\n")
    
    # Token metrics
    for metric in ["avg_input_tokens", "avg_output_tokens", "avg_total_tokens"]:
        neuroca_val = neuroca_stats.get(metric, 0)
        agno_val = agno_stats.get(metric, 0)
        val_diff_pct = ((neuroca_val - agno_val) / max(agno_val, 0.001)) * 100
        
        metric_name = metric.replace("avg_", "").replace("_", " ").title()
        f.write(f"{metric_name:<25} {neuroca_val:<15.1f} {agno_val:<15.1f} {val_diff_pct:+.1f}%\n")
    
    f.write("```\n\n")
    
    # Analysis
    token_diff_pct = ((neuroca_stats.get("avg_total_tokens", 0) - agno_stats.get("avg_total_tokens", 0)) / 
                      max(agno_stats.get("avg_total_tokens", 0), 0.001)) * 100
    
    if abs(token_diff_pct) < 10:
        f.write("Both memory systems use a similar number of tokens when augmenting LLMs.\n")
        f.write("This suggests comparable efficiency in context window utilization.\n\n")
    else:
        token_comparison = "more" if token_diff_pct > 0 else "fewer"
        f.write(f"Neuroca uses {abs(token_diff_pct):.1f}% {token_comparison} tokens when augmenting LLMs.\n")
        if token_comparison == "fewer":
            f.write("This improved token efficiency could translate to cost savings in production deployments\n")
            f.write("where LLM API usage is charged based on token count.\n\n")
        else:
            f.write("This higher token usage could lead to increased costs in production deployments,\n")
            f.write("though it might be justified if the additional tokens provide better context.\n\n")

def write_conversation_benchmark_results(f, results):
    """Write conversation benchmark results to report"""
    neuroca_metrics, agno_metrics = results
    
    f.write("### Conversation Memory Performance\n\n")
    f.write("```\n")
    f.write(f"{'Metric':<25} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
    f.write("-" * 70 + "\n")
    
    # Common metrics
    metrics = [
        ("Store time (ms)", "avg_store_ms"),
        ("Retrieve time (ms)", "avg_retrieve_ms"),
        ("Total time (ms)", "avg_total_ms"),
        ("Tokens per turn", "avg_total_tokens"),
        ("Final memory count", "final_memory_count")
    ]
    
    for label, key in metrics:
        neuroca_val = neuroca_metrics.get(key, 0)
        agno_val = agno_metrics.get(key, 0)
        
        # Calculate difference
        if agno_val != 0:
            diff_pct = ((neuroca_val - agno_val) / agno_val) * 100
            diff_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = "N/A"
            
        f.write(f"{label:<25} {neuroca_val:<15.1f} {agno_val:<15.1f} {diff_str:<15}\n")
    
    f.write("```\n\n")
    
    # Analysis
    store_diff_pct = ((neuroca_metrics.get("avg_store_ms", 0) - agno_metrics.get("avg_store_ms", 0)) / 
                     max(agno_metrics.get("avg_store_ms", 0), 0.1)) * 100
                      
    if store_diff_pct > 20:
        f.write("Neuroca's storage operations are slower than Agno's in conversational contexts. This is expected\n")
        f.write("as Neuroca computes vector embeddings during storage, which adds computational overhead but\n")
        f.write("enables more sophisticated semantic retrieval.\n\n")
    else:
        f.write("Both systems show comparable storage performance in conversational contexts.\n")
        f.write("This suggests efficient storage implementations in both memory systems.\n\n")
        
    retrieve_diff_pct = ((neuroca_metrics.get("avg_retrieve_ms", 0) - agno_metrics.get("avg_retrieve_ms", 0)) / 
                        max(agno_metrics.get("avg_retrieve_ms", 0), 0.1)) * 100
                            
    retrieve_comparison = "faster" if retrieve_diff_pct < 0 else "slower"
    
    if abs(retrieve_diff_pct) < 15:
        f.write("Both systems demonstrate similar retrieval performance in conversational contexts.\n")
        f.write("The retrieval times are close enough that other factors like network latency\n")
        f.write("would likely have a larger impact in real-world scenarios.\n\n")
    else:
        f.write(f"Neuroca's retrieval is {abs(retrieve_diff_pct):.1f}% {retrieve_comparison} than Agno's in\n")
        f.write("conversational contexts. This could impact the responsiveness of conversational agents.\n\n")

def write_memory_retention_results(f, results):
    """Write memory retention benchmark results to report"""
    neuroca_results, agno_results = results
    
    f.write("### Memory Retention Performance\n\n")
    f.write("```\n")
    f.write(f"{'Metric':<25} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
    f.write("-" * 70 + "\n")
    
    # Accuracy
    neuroca_accuracy = neuroca_results.get("accuracy", 0) * 100
    agno_accuracy = agno_results.get("accuracy", 0) * 100
    acc_diff = neuroca_accuracy - agno_accuracy
    f.write(f"{'Recall Accuracy (%)':<25} {neuroca_accuracy:<15.1f} {agno_accuracy:<15.1f} {acc_diff:+.1f}\n")
    
    # Average storage time
    neuroca_store = neuroca_results.get("avg_store_time_ms", 0)
    agno_store = agno_results.get("avg_store_time_ms", 0)
    store_diff_pct = ((neuroca_store - agno_store) / max(agno_store, 0.001)) * 100
    f.write(f"{'Storage Time (ms)':<25} {neuroca_store:<15.2f} {agno_store:<15.2f} {store_diff_pct:+.1f}%\n")
    
    # Average retrieval time
    neuroca_retrieve = neuroca_results.get("avg_retrieval_time_ms", 0)
    agno_retrieve = agno_results.get("avg_retrieval_time_ms", 0)
    retrieve_diff_pct = ((neuroca_retrieve - agno_retrieve) / max(agno_retrieve, 0.001)) * 100
    f.write(f"{'Retrieval Time (ms)':<25} {neuroca_retrieve:<15.2f} {agno_retrieve:<15.2f} {retrieve_diff_pct:+.1f}%\n")
    
    # Correct answers
    neuroca_correct = neuroca_results.get("correct_answers", 0)
    agno_correct = agno_results.get("correct_answers", 0)
    total_questions = neuroca_results.get("total_questions", 0)
    f.write(f"{'Correct Answers':<25} {neuroca_correct}/{total_questions}{' '*7} {agno_correct}/{total_questions}{' '*7} {neuroca_correct-agno_correct:+}\n")
    
    f.write("```\n\n")
    
    # Analysis
    if abs(acc_diff) < 10:
        f.write("Both memory systems demonstrate similar accuracy in recalling key facts amongst noise.\n")
        f.write("This suggests that with proper importance scoring, both systems can effectively\n")
        f.write("prioritize critical information retention.\n\n")
    else:
        better_system = "Neuroca" if acc_diff > 0 else "Agno"
        f.write(f"{better_system} shows significantly better accuracy in retrieving key facts amongst noise.\n")
        if better_system == "Neuroca":
            f.write("This suggests that Neuroca's vector-based similarity search may provide\n")
            f.write("an advantage in distinguishing important content from noise.\n\n")
        else:
            f.write("This suggests that Agno's approach to memory storage and retrieval\n")
            f.write("may be more effective at maintaining distinct representations of important facts.\n\n")

def write_task_benchmark_results(f, results):
    """Write task-oriented benchmark results to report"""
    neuroca_results, agno_results = results
    
    f.write("### Task-Oriented Performance\n\n")
    f.write("```\n")
    f.write(f"{'Metric':<25} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
    f.write("-" * 70 + "\n")
    
    # Accuracy
    neuroca_accuracy = neuroca_results.get("accuracy", 0) * 100
    agno_accuracy = agno_results.get("accuracy", 0) * 100
    acc_diff = neuroca_accuracy - agno_accuracy
    f.write(f"{'Accuracy (%)':<25} {neuroca_accuracy:<15.1f} {agno_accuracy:<15.1f} {acc_diff:+.1f}\n")
    
    # Storage time
    neuroca_store = neuroca_results.get("storage_time", 0)
    agno_store = agno_results.get("storage_time", 0)
    store_diff_pct = ((neuroca_store - agno_store) / max(agno_store, 0.001)) * 100
    f.write(f"{'Storage Time (s)':<25} {neuroca_store:<15.2f} {agno_store:<15.2f} {store_diff_pct:+.1f}%\n")
    
    # Retrieval time
    neuroca_retrieve = neuroca_results.get("avg_retrieval_time_ms", 0)
    agno_retrieve = agno_results.get("avg_retrieval_time_ms", 0)
    retrieve_diff_pct = ((neuroca_retrieve - agno_retrieve) / max(agno_retrieve, 0.001)) * 100
    f.write(f"{'Retrieval Time (ms)':<25} {neuroca_retrieve:<15.2f} {agno_retrieve:<15.2f} {retrieve_diff_pct:+.1f}%\n")
    
    # P95 retrieval time
    neuroca_p95 = neuroca_results.get("p95_retrieval_time_ms", 0)
    agno_p95 = agno_results.get("p95_retrieval_time_ms", 0)
    p95_diff_pct = ((neuroca_p95 - agno_p95) / max(agno_p95, 0.001)) * 100
    f.write(f"{'P95 Retrieval (ms)':<25} {neuroca_p95:<15.2f} {agno_p95:<15.2f} {p95_diff_pct:+.1f}%\n")
    
    # Combined time
    neuroca_combined = neuroca_results.get("avg_combined_time_ms", 0)
    agno_combined = agno_results.get("avg_combined_time_ms", 0)
    combined_diff_pct = ((neuroca_combined - agno_combined) / max(agno_combined, 0.001)) * 100
    f.write(f"{'Combined Time (ms)':<25} {neuroca_combined:<15.2f} {agno_combined:<15.2f} {combined_diff_pct:+.1f}%\n")
    
    # Memory usage
    neuroca_mem = neuroca_results.get("memory_increase_mb", 0)
    agno_mem = agno_results.get("memory_increase_mb", 0)
    mem_diff_pct = ((neuroca_mem - agno_mem) / max(agno_mem, 0.001)) * 100
    f.write(f"{'Memory Increase (MB)':<25} {neuroca_mem:<15.1f} {agno_mem:<15.1f} {mem_diff_pct:+.1f}%\n")
    
    f.write("```\n\n")
    
    # Analysis
    if abs(acc_diff) < 10:
        f.write("Both memory systems demonstrate similar accuracy in task-oriented retrieval.\n")
        f.write("This suggests that both systems effectively index and retrieve information based on\n")
        f.write("semantic similarity in this particular benchmark context.\n\n")
    else:
        better_system = "Neuroca" if acc_diff > 0 else "Agno"
        f.write(f"{better_system} achieves significantly better accuracy in task-oriented retrieval.\n")
        if better_system == "Neuroca":
            f.write("This suggests that Neuroca's vector-based similarity search may provide\n")
            f.write("an advantage in mapping questions to relevant contexts that contain the answers.\n\n")
        else:
            f.write("This suggests that Agno's approach to memory storage and retrieval\n")
            f.write("may be more effective at maintaining the relationships between questions and answers.\n\n")

def write_memory_pressure_results(f, results):
    """Write memory pressure & maintenance benchmark results to report"""
    neuroca_results, agno_results = results
    
    f.write("### Memory Pressure & Maintenance Performance\n\n")
    f.write("```\n")
    f.write(f"{'Metric':<25} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
    f.write("-" * 70 + "\n")
    
    # Max records
    neuroca_max = neuroca_results.get("max_records", 0)
    agno_max = agno_results.get("max_records", 0)
    max_diff_pct = ((neuroca_max - agno_max) / max(agno_max, 1)) * 100
    f.write(f"{'Max Records':<25} {neuroca_max:<15,d} {agno_max:<15,d} {max_diff_pct:+.1f}%\n")
    
    # Memory usage
    neuroca_mem = neuroca_results.get("final_memory_mb", 0)
    agno_mem = agno_results.get("final_memory_mb", 0)
    mem_diff_pct = ((neuroca_mem - agno_mem) / max(agno_mem, 0.1)) * 100
    f.write(f"{'Final Memory (MB)':<25} {neuroca_mem:<15.1f} {agno_mem:<15.1f} {mem_diff_pct:+.1f}%\n")
    
    # Memory per record
    if neuroca_max > 0 and agno_max > 0:
        neuroca_efficiency = (neuroca_mem * 1024 * 1024) / neuroca_max
        agno_efficiency = (agno_mem * 1024 * 1024) / agno_max
        efficiency_diff_pct = ((neuroca_efficiency - agno_efficiency) / max(agno_efficiency, 1)) * 100
        f.write(f"{'Memory/Record (bytes)':<25} {neuroca_efficiency:<15.1f} {agno_efficiency:<15.1f} {efficiency_diff_pct:+.1f}%\n")
    
    # Final query latency
    neuroca_latency = neuroca_results.get("final_latency_ms", 0)
    agno_latency = agno_results.get("final_latency_ms", 0)
    latency_diff_pct = ((neuroca_latency - agno_latency) / max(agno_latency, 0.1)) * 100
    f.write(f"{'Final Query Latency (ms)':<25} {neuroca_latency:<15.2f} {agno_latency:<15.2f} {latency_diff_pct:+.1f}%\n")
    
    # Health score
    neuroca_health = neuroca_results.get("final_health", 0)
    agno_health = agno_results.get("final_health", 0)
    health_diff = neuroca_health - agno_health
    f.write(f"{'Final Health Score':<25} {neuroca_health:<15.2f} {agno_health:<15.2f} {health_diff:+.2f}\n")
    
    # Maintenance events
    neuroca_maint = neuroca_results.get("maintenance_count", 0)
    agno_maint = agno_results.get("maintenance_count", 0)
    maint_diff = neuroca_maint - agno_maint
    f.write(f"{'Maintenance Events':<25} {neuroca_maint:<15d} {agno_maint:<15d} {maint_diff:+d}\n")
    
    # Error count
    neuroca_errors = neuroca_results.get("error_count", 0)
    agno_errors = agno_results.get("error_count", 0)
    error_diff = neuroca_errors - agno_errors
    f.write(f"{'Errors Encountered':<25} {neuroca_errors:<15d} {agno_errors:<15d} {error_diff:+d}\n")
    
    f.write("```\n\n")
    
    # Analysis
    # Capacity analysis
    if neuroca_max > agno_max * 1.2:  # 20% more capacity
        f.write("Neuroca demonstrates superior scaling capacity, storing significantly more records before\n")
        f.write("reaching saturation. This suggests that its multi-tiered architecture and automatic\n")
        f.write("maintenance capabilities enable it to handle larger datasets efficiently.\n\n")
    elif agno_max > neuroca_max * 1.2:
        f.write("Agno demonstrates superior scaling capacity, storing significantly more records before\n")
        f.write("reaching saturation. This suggests that its simpler architecture may have fundamental\n")
        f.write("efficiency advantages for raw storage volume.\n\n")
    else:
        f.write("Both systems demonstrate comparable scaling capacity in terms of maximum records stored.\n")
        f.write("This suggests that for moderate-sized datasets, either architecture provides\n")
        f.write("sufficient storage capacity.\n\n")
    
    # Self-maintenance analysis
    if neuroca_maint > 0 and agno_maint == 0:
        f.write("Neuroca's self-maintenance capabilities were actively triggered during the benchmark,\n")
        f.write("allowing it to optimize its internal state as the memory system grew. This automatic\n")
        f.write("maintenance is crucial for long-running applications where memory management becomes\n")
        f.write("increasingly important over time.\n\n")
    elif neuroca_maint > agno_maint and agno_maint > 0:
        f.write("Both systems demonstrated self-maintenance capabilities, with Neuroca performing\n")
        f.write(f"more maintenance operations ({neuroca_maint} vs {agno_maint}). This proactive maintenance\n")
        f.write("approach helps maintain performance even as the memory systems grow.\n\n")
    elif agno_maint > neuroca_maint:
        f.write("Surprisingly, Agno demonstrated more self-maintenance activity than Neuroca,\n")
        f.write("suggesting it includes some form of automatic optimization even without\n")
        f.write("explicit maintenance capabilities in its documentation.\n\n")
    
    # Health and stability
    if neuroca_health > agno_health + 0.2:  # At least 0.2 better health
        f.write("Neuroca maintained a significantly better health score throughout the test,\n")
        f.write("demonstrating the effectiveness of its health monitoring and self-maintenance\n")
        f.write("systems. This suggests better long-term stability and performance under pressure.\n\n")
    elif agno_health > neuroca_health + 0.2:
        f.write("Agno maintained a surprisingly strong health score throughout the test,\n")
        f.write("even without explicit health monitoring systems. This suggests robust\n")
        f.write("design fundamentals that provide stability under pressure.\n\n")
    else:
        f.write("Both systems maintained comparable health scores. While Neuroca includes\n")
        f.write("explicit health monitoring, Agno's simpler architecture appeared to remain\n")
        f.write("stable under the test conditions.\n\n")

def write_multi_agent_results(f, results):
    """Write multi-agent benchmark results to report"""
    neuroca_results, agno_results = results
    
    f.write("### Multi-Agent Collaboration Performance\n\n")
    f.write("```\n")
    f.write(f"{'Metric':<30} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
    f.write("-" * 75 + "\n")
    
    # Task completion
    n_completion = neuroca_results["task_completion"]["completion_percentage"]
    a_completion = agno_results["task_completion"]["completion_percentage"]
    completion_diff = n_completion - a_completion
    f.write(f"{'Task Completion (%)':<30} {n_completion:<15.1f} {a_completion:<15.1f} {completion_diff:+.1f}\n")
    
    # Task duration
    n_duration = neuroca_results["task_completion"]["avg_task_duration"]
    a_duration = agno_results["task_completion"]["avg_task_duration"]
    duration_diff_pct = ((n_duration - a_duration) / max(a_duration, 0.001)) * 100
    f.write(f"{'Avg Task Duration (ms)':<30} {n_duration:<15.1f} {a_duration:<15.1f} {duration_diff_pct:+.1f}%\n")
    
    # Communication
    n_comm_time = neuroca_results["communication"]["avg_total_time_ms"]
    a_comm_time = agno_results["communication"]["avg_total_time_ms"]
    comm_diff_pct = ((n_comm_time - a_comm_time) / max(a_comm_time, 0.001)) * 100
    f.write(f"{'Communication Time (ms)':<30} {n_comm_time:<15.1f} {a_comm_time:<15.1f} {comm_diff_pct:+.1f}%\n")
    
    # Message count
    n_messages = neuroca_results["communication"]["total_messages"]
    a_messages = agno_results["communication"]["total_messages"]
    message_diff_pct = ((n_messages - a_messages) / max(a_messages, 1)) * 100
    f.write(f"{'Total Messages':<30} {n_messages:<15.1f} {a_messages:<15.1f} {message_diff_pct:+.1f}%\n")
    
    # Memory usage
    n_memory = neuroca_results["resource_usage"]["avg_memory_mb"]
    a_memory = agno_results["resource_usage"]["avg_memory_mb"]
    memory_diff_pct = ((n_memory - a_memory) / max(a_memory, 0.1)) * 100
    f.write(f"{'Memory Usage (MB)':<30} {n_memory:<15.1f} {a_memory:<15.1f} {memory_diff_pct:+.1f}%\n")
    
    f.write("```\n\n")
    
    # Analysis
    if abs(completion_diff) < 5:
        f.write("Both memory systems achieved similar task completion rates in the multi-agent simulation.\n\n")
    else:
        better_system = "Neuroca" if completion_diff > 0 else "Agno"
        f.write(f"{better_system} achieved a higher task completion rate, suggesting better support for multi-agent coordination.\n\n")
        
    if abs(duration_diff_pct) < 10:
        f.write("Task processing times were comparable between the two systems.\n\n")
    else:
        faster_system = "Neuroca" if n_duration < a_duration else "Agno"
        f.write(f"{faster_system}'s memory system enabled faster task processing for the agents.\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unified benchmark suite for Neuroca vs Agno")
    
    # Benchmark selection
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--basic", action="store_true", help="Run basic memory operations benchmark")
    parser.add_argument("--llm", action="store_true", help="Run LLM integration benchmark")
    parser.add_argument("--conversation", action="store_true", help="Run conversation context benchmark")
    parser.add_argument("--retention", action="store_true", help="Run memory retention benchmark")
    parser.add_argument("--task", action="store_true", help="Run task-oriented benchmark")
    parser.add_argument("--pressure", action="store_true", help="Run memory pressure & maintenance benchmark")
    parser.add_argument("--multi-agent", action="store_true", help="Run multi-agent collaboration benchmark")
    
    # Benchmark parameters
    parser.add_argument("--dataset-size", type=int, default=500, help="Dataset size for task benchmark")
    parser.add_argument("--query-count", type=int, default=50, help="Query count for task benchmark")
    parser.add_argument("--max-records", type=int, default=20000, help="Maximum records for memory pressure benchmark")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for memory pressure benchmark")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents for multi-agent benchmark")
    parser.add_argument("--sim-time", type=float, default=10, help="Simulation time (s) for multi-agent benchmark")
    parser.add_argument("--tasks", type=int, default=20, help="Number of tasks for multi-agent benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for multi-agent benchmark")
    
    args = parser.parse_args()
    
    # If no specific benchmark is selected, run all
    if not (args.basic or args.llm or args.conversation or args.retention or args.task or args.pressure or args.multi_agent):
        args.all = True
    
    # If --all is selected, enable all benchmarks
    if args.all:
        args.basic = True
        args.llm = True
        args.conversation = True
        args.retention = True
        args.task = True
        args.pressure = True
        args.multi_agent = True
    
    run_all_benchmarks(args)
