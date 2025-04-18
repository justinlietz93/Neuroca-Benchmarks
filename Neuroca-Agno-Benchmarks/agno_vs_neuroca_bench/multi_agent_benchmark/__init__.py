# multi_agent_benchmark/__init__.py
from .benchmark import compare_multi_agent_benchmarks, run_multi_agent_benchmark
from .agent import Agent
from .message_board import MessageBoard
from .task import Task, TaskManager

__all__ = [
    'compare_multi_agent_benchmarks',
    'run_multi_agent_benchmark',
    'Agent',
    'MessageBoard',
    'Task',
    'TaskManager'
]
