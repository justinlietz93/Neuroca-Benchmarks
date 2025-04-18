# task.py
import threading
import time
import statistics

class Task:
    """Represents a task to be performed by an agent"""
    
    def __init__(self, task_id, complexity=1, dependencies=None):
        self.task_id = task_id
        self.complexity = complexity  # Affects processing time
        self.dependencies = dependencies or []  # List of task IDs this task depends on
        self.status = "pending"  # pending, in_progress, completed, failed
        self.assigned_to = None
        self.result = None
        self.start_time = None
        self.end_time = None
        
    def to_dict(self):
        """Convert task to dictionary for reporting"""
        return {
            "task_id": self.task_id,
            "complexity": self.complexity,
            "dependencies": self.dependencies,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "result": self.result,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": (self.end_time - self.start_time) if self.start_time and self.end_time else None
        }

class TaskManager:
    """Manages tasks and task dependencies"""
    
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
        
    def add_task(self, task):
        """Add a task to the manager"""
        with self.lock:
            self.tasks[task.task_id] = task
            
    def get_task(self, task_id):
        """Get a task by ID"""
        with self.lock:
            return self.tasks.get(task_id)
            
    def get_available_tasks(self):
        """Get tasks that are ready to be worked on"""
        with self.lock:
            available_tasks = []
            for task_id, task in self.tasks.items():
                if task.status == "pending" and task.assigned_to is None:
                    # Check if all dependencies are completed
                    deps_completed = all(
                        self.tasks.get(dep).status == "completed"
                        for dep in task.dependencies
                        if dep in self.tasks
                    )
                    if deps_completed:
                        available_tasks.append(task)
            return available_tasks
            
    def assign_task(self, task_id, agent_id):
        """Assign a task to an agent"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == "pending" and task.assigned_to is None:
                task.assigned_to = agent_id
                task.status = "in_progress"
                task.start_time = time.time()
                return True
            return False
            
    def complete_task(self, task_id, result):
        """Mark a task as completed"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == "in_progress":
                task.status = "completed"
                task.result = result
                task.end_time = time.time()
                return True
            return False
            
    def fail_task(self, task_id, reason):
        """Mark a task as failed"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task and task.status == "in_progress":
                task.status = "failed"
                task.result = {"error": reason}
                task.end_time = time.time()
                return True
            return False
            
    def all_tasks_completed(self):
        """Check if all tasks are completed"""
        with self.lock:
            return all(task.status == "completed" for task in self.tasks.values())
            
    def get_stats(self):
        """Get statistics about task completion"""
        with self.lock:
            tasks = list(self.tasks.values())
            if not tasks:
                return {
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "pending_tasks": 0,
                    "in_progress_tasks": 0,
                    "completion_percentage": 0,
                    "avg_task_duration": 0
                }
                
            completed = [t for t in tasks if t.status == "completed"]
            failed = [t for t in tasks if t.status == "failed"]
            pending = [t for t in tasks if t.status == "pending"]
            in_progress = [t for t in tasks if t.status == "in_progress"]
            
            durations = []
            for task in completed:
                if task.start_time and task.end_time:
                    durations.append(task.end_time - task.start_time)
            
            return {
                "total_tasks": len(tasks),
                "completed_tasks": len(completed),
                "failed_tasks": len(failed),
                "pending_tasks": len(pending),
                "in_progress_tasks": len(in_progress),
                "completion_percentage": (len(completed) / len(tasks)) * 100 if tasks else 0,
                "avg_task_duration": statistics.mean(durations) if durations else 0
            }
