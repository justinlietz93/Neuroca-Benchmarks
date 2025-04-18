# benchmark.py
import time
import random
import logging
import os
import psutil
import statistics
from pathlib import Path
import json

# Import local modules
from .task import Task, TaskManager
from .message_board import MessageBoard
from .agent import Agent

# Import memory implementations
from bench import NeuroMemoryBenchmark, AgnoMemoryBenchmark

# Configure logging
logger = logging.getLogger("multi_agent_benchmark")

def create_task_network(num_tasks=20, max_dependencies=2):
    """Create a network of interconnected tasks"""
    tasks = []
    
    # Create initial tasks without dependencies
    for i in range(5):
        task_id = f"task_{i}"
        tasks.append(Task(task_id, complexity=random.uniform(0.5, 2.0)))
    
    # Create tasks with dependencies
    for i in range(5, num_tasks):
        task_id = f"task_{i}"
        
        # Randomly select dependencies from existing tasks
        num_deps = min(random.randint(0, max_dependencies), i)
        deps = random.sample([t.task_id for t in tasks], num_deps) if num_deps > 0 else []
        
        tasks.append(Task(task_id, complexity=random.uniform(0.5, 2.0), dependencies=deps))
    
    return tasks

def run_multi_agent_benchmark(memory_system_type, num_agents=5, simulation_time=10, task_count=20):
    """
    Run a multi-agent benchmark with the specified memory system
    
    Args:
        memory_system_type (str): "neuroca" or "agno"
        num_agents (int): Number of agents to create
        simulation_time (float): Simulation time in seconds
        task_count (int): Number of tasks to create
        
    Returns:
        dict: Benchmark metrics
    """
    # Create components
    message_board = MessageBoard()
    task_manager = TaskManager()
    
    # Create task network
    tasks = create_task_network(task_count, max_dependencies=2)
    for task in tasks:
        task_manager.add_task(task)
    
    # Create agents with different roles
    agents = []
    
    for i in range(num_agents):
        # Assign roles
        if i == 0:
            role = "planner"  # First agent is a planner
        elif i == 1:
            role = "coordinator"  # Second agent is a coordinator
        else:
            role = "worker"  # All others are workers
        
        # Create appropriate memory system
        if memory_system_type.lower() == "neuroca":
            memory = NeuroMemoryBenchmark()
        else:
            memory = AgnoMemoryBenchmark()
        
        # Create agent
        agent = Agent(
            f"agent_{i}", 
            memory, 
            message_board, 
            task_manager,
            role=role,
            speed_factor=random.uniform(0.8, 1.2),  # Slight variation in processing speed
            error_rate=0.05  # 5% chance of task failure
        )
        
        agents.append(agent)
    
    # Start agents
    for agent in agents:
        agent.start()
    
    # Run simulation for specified time
    start_time = time.time()
    
    # Measure resource usage at intervals
    memory_usage_samples = []
    cpu_usage_samples = []
    task_completion_samples = []
    message_count_samples = []
    
    sampling_interval = min(1.0, simulation_time / 10)  # Take at least 10 samples
    
    while time.time() - start_time < simulation_time:
        # Record resource usage
        process = psutil.Process(os.getpid())
        memory_usage_samples.append(process.memory_info().rss / (1024 * 1024))  # MB
        cpu_usage_samples.append(process.cpu_percent(interval=None))
        
        # Record task completion
        task_stats = task_manager.get_stats()
        task_completion_samples.append(task_stats["completion_percentage"])
        
        # Record message count
        message_stats = message_board.get_stats()
        message_count_samples.append(message_stats["message_count"])
        
        # Sleep for sampling interval
        time.sleep(sampling_interval)
    
    # Calculate benchmark duration
    benchmark_duration = time.time() - start_time
    
    # Stop all agents
    for agent in agents:
        agent.stop()
    
    # All agents should join within a reasonable time
    for agent in agents:
        agent.join(timeout=1.0)
    
    # Collect final statistics
    final_task_stats = task_manager.get_stats()
    final_message_stats = message_board.get_stats()
    
    # Calculate resource usage statistics
    avg_memory_usage = statistics.mean(memory_usage_samples)
    max_memory_usage = max(memory_usage_samples)
    avg_cpu_usage = statistics.mean(cpu_usage_samples)
    max_cpu_usage = max(cpu_usage_samples)
    
    # Check if all tasks were processed
    all_tasks_completed = task_manager.all_tasks_completed()
    
    # Calculate agent-specific statistics
    agent_stats = []
    for agent in agents:
        agent_stats.append({
            "agent_id": agent.agent_id,
            "role": agent.role,
            "completed_tasks": len(agent.completed_tasks),
            "failed_tasks": len(agent.failed_tasks),
            "knowledge_items": len(agent.knowledge)
        })
    
    # Compile benchmark results
    benchmark_results = {
        "memory_system": memory_system_type,
        "num_agents": num_agents,
        "simulation_time": simulation_time,
        "task_count": task_count,
        "benchmark_duration": benchmark_duration,
        "task_completion": {
            "total_tasks": final_task_stats["total_tasks"],
            "completed_tasks": final_task_stats["completed_tasks"],
            "failed_tasks": final_task_stats["failed_tasks"],
            "completion_percentage": final_task_stats["completion_percentage"],
            "avg_task_duration": final_task_stats["avg_task_duration"] * 1000,  # Convert to ms
            "all_tasks_completed": all_tasks_completed
        },
        "communication": {
            "total_messages": final_message_stats["message_count"],
            "avg_wait_time_ms": final_message_stats["avg_wait_time"],
            "avg_process_time_ms": final_message_stats["avg_process_time"],
            "avg_total_time_ms": final_message_stats["avg_total_time"],
            "p95_total_time_ms": final_message_stats["p95_total_time"]
        },
        "resource_usage": {
            "avg_memory_mb": avg_memory_usage,
            "max_memory_mb": max_memory_usage,
            "avg_cpu_percent": avg_cpu_usage,
            "max_cpu_percent": max_cpu_usage
        },
        "agent_performance": agent_stats
    }
    
    return benchmark_results

def average_benchmark_results(results):
    """Average multiple benchmark results into a single result"""
    if not results:
        return {}
        
    avg_result = {
        "memory_system": results[0]["memory_system"],
        "num_agents": results[0]["num_agents"],
        "simulation_time": results[0]["simulation_time"],
        "task_count": results[0]["task_count"],
        "benchmark_duration": statistics.mean([r["benchmark_duration"] for r in results]),
        "task_completion": {
            "total_tasks": statistics.mean([r["task_completion"]["total_tasks"] for r in results]),
            "completed_tasks": statistics.mean([r["task_completion"]["completed_tasks"] for r in results]),
            "failed_tasks": statistics.mean([r["task_completion"]["failed_tasks"] for r in results]),
            "completion_percentage": statistics.mean([r["task_completion"]["completion_percentage"] for r in results]),
            "avg_task_duration": statistics.mean([r["task_completion"]["avg_task_duration"] for r in results]),
            "all_tasks_completed": all(r["task_completion"]["all_tasks_completed"] for r in results)
        },
        "communication": {
            "total_messages": statistics.mean([r["communication"]["total_messages"] for r in results]),
            "avg_wait_time_ms": statistics.mean([r["communication"]["avg_wait_time_ms"] for r in results]),
            "avg_process_time_ms": statistics.mean([r["communication"]["avg_process_time_ms"] for r in results]),
            "avg_total_time_ms": statistics.mean([r["communication"]["avg_total_time_ms"] for r in results]),
            "p95_total_time_ms": statistics.mean([r["communication"]["p95_total_time_ms"] for r in results])
        },
        "resource_usage": {
            "avg_memory_mb": statistics.mean([r["resource_usage"]["avg_memory_mb"] for r in results]),
            "max_memory_mb": statistics.mean([r["resource_usage"]["max_memory_mb"] for r in results]),
            "avg_cpu_percent": statistics.mean([r["resource_usage"]["avg_cpu_percent"] for r in results]),
            "max_cpu_percent": statistics.mean([r["resource_usage"]["max_cpu_percent"] for r in results])
        }
    }
    
    return avg_result

def generate_multi_agent_report(neuroca_results, agno_results):
    """Generate a comprehensive report comparing the multi-agent benchmark results"""
    report_path = Path("multi_agent_benchmark_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Multi-Agent System Benchmark Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This benchmark simulates a multi-agent system where agents with different roles (planner, coordinator, worker) \n")
        f.write("collaborate to complete tasks. Each agent has its own memory system, allowing it to retain and retrieve \n")
        f.write("information. The benchmark tests how well different memory systems (Neuroca and Agno) support multi-agent collaboration.\n\n")
        
        f.write("## Benchmark Parameters\n\n")
        f.write(f"- Number of agents: {neuroca_results['num_agents']}\n")
        f.write(f"- Simulation time: {neuroca_results['simulation_time']} seconds\n")
        f.write(f"- Task count: {neuroca_results['task_count']}\n\n")
        
        f.write("## Results Summary\n\n")
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
        f.write("## Analysis\n\n")
        
        # Task completion analysis
        f.write("### Task Completion\n\n")
        if abs(completion_diff) < 5:
            f.write("Both memory systems achieved similar task completion rates, suggesting that \n")
            f.write("for this type of multi-agent coordination, the choice of memory system \n")
            f.write("does not significantly impact the ability of agents to complete tasks.\n\n")
        else:
            better_system = "Neuroca" if completion_diff > 0 else "Agno"
            f.write(f"{better_system} achieved a higher task completion rate, indicating that its \n")
            f.write(f"memory system may better support the information retrieval and storage \n")
            f.write(f"patterns needed for effective multi-agent coordination.\n\n")
        
        # Task duration analysis
        f.write("### Task Processing Efficiency\n\n")
        if abs(duration_diff_pct) < 10:
            f.write("Both memory systems supported similar task processing times, indicating \n")
            f.write("comparable efficiency in retrieving task-related information from memory.\n\n")
        else:
            faster_system = "Neuroca" if n_duration < a_duration else "Agno"
            f.write(f"{faster_system}'s memory system enabled faster task processing, which could be \n")
            f.write(f"significant in time-sensitive multi-agent operations where quick decision-making \n")
            f.write(f"based on stored information is critical.\n\n")
        
        # Communication analysis
        f.write("### Agent Communication\n\n")
        if abs(comm_diff_pct) < 10 and abs(message_diff_pct) < 10:
            f.write("Both memory systems supported similar communication patterns between agents, \n")
            f.write("with comparable message processing times and message counts. This suggests \n")
            f.write("that both systems can effectively support the collaborative aspects of \n")
            f.write("multi-agent systems.\n\n")
        else:
            if n_comm_time < a_comm_time:
                f.write("Neuroca's memory system enabled faster message processing, which could \n")
                f.write("lead to more responsive agent interactions in time-sensitive scenarios.\n\n")
            elif a_comm_time < n_comm_time:
                f.write("Agno's memory system demonstrated faster message processing, potentially \n")
                f.write("offering advantages in scenarios requiring rapid agent communication.\n\n")
                
            if abs(message_diff_pct) > 10:
                more_msgs = "Neuroca" if n_messages > a_messages else "Agno"
                f.write(f"The {more_msgs} memory system facilitated more message exchanges between agents. \n")
                f.write(f"This could indicate either better collaboration or potential inefficiencies \n")
                f.write(f"in how agents share information, depending on the specific use case.\n\n")
        
        # Resource usage analysis
        f.write("### Resource Efficiency\n\n")
        if abs(memory_diff_pct) < 10:
            f.write("Both memory systems exhibited similar memory usage, suggesting comparable \n")
            f.write("efficiency in storing and managing agent knowledge and communication history.\n\n")
        else:
            efficient_system = "Neuroca" if n_memory < a_memory else "Agno"
            f.write(f"{efficient_system}'s memory system demonstrated better memory efficiency, which \n")
            f.write(f"could be important for scalability in larger multi-agent deployments or on \n")
            f.write(f"memory-constrained systems.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        if completion_diff > 5 and n_duration < a_duration:
            f.write("Neuroca's memory system appears to offer advantages for multi-agent systems, \n")
            f.write("with higher task completion rates and faster task processing times.\n")
        elif completion_diff < -5 and a_duration < n_duration:
            f.write("Agno's memory system demonstrates better performance in multi-agent scenarios, \n")
            f.write("achieving higher task completion rates with faster processing times.\n")
        elif n_memory < a_memory * 0.8:
            f.write("Neuroca's memory system shows significant memory efficiency advantages, which \n")
            f.write("would be particularly valuable in large-scale multi-agent deployments.\n")
        elif a_memory < n_memory * 0.8:
            f.write("Agno's memory system offers notable memory efficiency benefits, making it \n")
            f.write("potentially more suitable for resource-constrained multi-agent applications.\n")
        else:
            f.write("Both memory systems performed comparably across most metrics in this multi-agent \n")
            f.write("benchmark. The choice between Neuroca and Agno for multi-agent systems may depend \n")
            f.write("more on specific application requirements or integration considerations rather \n")
            f.write("than raw performance differences.\n")
    
    logger.info(f"Generated multi-agent benchmark report at {report_path}")
    print(f"\nGenerated multi-agent benchmark report at {report_path}")

def compare_multi_agent_benchmarks(num_agents=5, simulation_time=10, task_count=20, runs=3):
    """
    Run and compare multi-agent benchmarks for both memory systems
    
    Args:
        num_agents (int): Number of agents per benchmark
        simulation_time (float): Simulation time in seconds
        task_count (int): Number of tasks per benchmark
        runs (int): Number of benchmark runs to average results
        
    Returns:
        tuple: (neuroca_results, agno_results)
    """
    logger.info(f"Starting multi-agent benchmarks with {num_agents} agents, {simulation_time}s simulation, {task_count} tasks")
    
    # Run Neuroca benchmarks
    neuroca_results = []
    print(f"\n===== Running {runs} Neuroca Multi-Agent Benchmarks =====\n")
    
    for i in range(runs):
        print(f"Run {i+1}/{runs}...")
        result = run_multi_agent_benchmark("neuroca", num_agents, simulation_time, task_count)
        neuroca_results.append(result)
        print(f"Completed: {result['task_completion']['completion_percentage']:.1f}% of tasks, {result['communication']['total_messages']} messages exchanged\n")
    
    # Run Agno benchmarks
    agno_results = []
    print(f"\n===== Running {runs} Agno Multi-Agent Benchmarks =====\n")
    
    for i in range(runs):
        print(f"Run {i+1}/{runs}...")
        result = run_multi_agent_benchmark("agno", num_agents, simulation_time, task_count)
        agno_results.append(result)
        print(f"Completed: {result['task_completion']['completion_percentage']:.1f}% of tasks, {result['communication']['total_messages']} messages exchanged\n")
    
    # Average results
    neuroca_avg = average_benchmark_results(neuroca_results)
    agno_avg = average_benchmark_results(agno_results)
    
    # Print comparison
    print("\n===== Multi-Agent Benchmark Comparison =====\n")
    print(f"{'Metric':<30} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}")
    print("-" * 75)
    
    # Task completion
    n_completion = neuroca_avg["task_completion"]["completion_percentage"]
    a_completion = agno_avg["task_completion"]["completion_percentage"]
    completion_diff = n_completion - a_completion
    print(f"{'Task Completion (%)':<30} {n_completion:<15.1f} {a_completion:<15.1f} {completion_diff:+.1f}")
    
    # Task duration
    n_duration = neuroca_avg["task_completion"]["avg_task_duration"]
    a_duration = agno_avg["task_completion"]["avg_task_duration"]
    duration_diff_pct = ((n_duration - a_duration) / max(a_duration, 0.001)) * 100
    print(f"{'Avg Task Duration (ms)':<30} {n_duration:<15.1f} {a_duration:<15.1f} {duration_diff_pct:+.1f}%")
    
    # Communication
    n_comm_time = neuroca_avg["communication"]["avg_total_time_ms"]
    a_comm_time = agno_avg["communication"]["avg_total_time_ms"]
    comm_diff_pct = ((n_comm_time - a_comm_time) / max(a_comm_time, 0.001)) * 100
    print(f"{'Communication Time (ms)':<30} {n_comm_time:<15.1f} {a_comm_time:<15.1f} {comm_diff_pct:+.1f}%")
    
    # Message count
    n_messages = neuroca_avg["communication"]["total_messages"]
    a_messages = agno_avg["communication"]["total_messages"]
    message_diff_pct = ((n_messages - a_messages) / max(a_messages, 1)) * 100
    print(f"{'Total Messages':<30} {n_messages:<15.1f} {a_messages:<15.1f} {message_diff_pct:+.1f}%")
    
    # Memory usage
    n_memory = neuroca_avg["resource_usage"]["avg_memory_mb"]
    a_memory = agno_avg["resource_usage"]["avg_memory_mb"]
    memory_diff_pct = ((n_memory - a_memory) / max(a_memory, 0.1)) * 100
    print(f"{'Memory Usage (MB)':<30} {n_memory:<15.1f} {a_memory:<15.1f} {memory_diff_pct:+.1f}%")
    
    # Generate detailed report
    generate_multi_agent_report(neuroca_avg, agno_avg)
    
    return neuroca_avg, agno_avg
