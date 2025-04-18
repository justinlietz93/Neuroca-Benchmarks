# multi_agent_benchmark.py - Main facade for multi-agent benchmarks
from multi_agent_benchmark import compare_multi_agent_benchmarks

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multi-agent benchmarks for Neuroca and Agno memory systems")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents to create")
    parser.add_argument("--time", type=float, default=30, help="Simulation time in seconds")
    parser.add_argument("--tasks", type=int, default=30, help="Number of tasks to create")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs to average results")
    
    args = parser.parse_args()
    
    compare_multi_agent_benchmarks(
        num_agents=args.agents,
        simulation_time=args.time,
        task_count=args.tasks,
        runs=args.runs
    )
