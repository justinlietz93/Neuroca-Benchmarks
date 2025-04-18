# agent.py
import threading
import time
import random
from types import SimpleNamespace

# Import memory implementations 
from bench import NeuroMemoryBenchmark, AgnoMemoryBenchmark

class Agent(threading.Thread):
    """Simulates an agent with memory and reasoning capabilities"""
    
    def __init__(self, agent_id, memory_system, message_board, task_manager, 
                 role="worker", speed_factor=1.0, error_rate=0.05):
        super().__init__()
        self.agent_id = agent_id
        self.memory = memory_system
        self.message_board = message_board
        self.task_manager = task_manager
        self.role = role  # worker, planner, coordinator
        self.speed_factor = speed_factor  # Affects processing time
        self.error_rate = error_rate  # Probability of task failure
        self.running = True
        self.daemon = True
        self.completed_tasks = []
        self.failed_tasks = []
        self.knowledge = {}  # Agent's private knowledge
        
    def store_in_memory(self, content, metadata=None, importance=0.7):
        """Store information in the agent's memory"""
        if isinstance(self.memory, NeuroMemoryBenchmark):
            # For Neuroca memory implementation
            record = {
                "id": f"mem_{time.time()}_{random.randint(1000, 9999)}",
                "content": content,
                "metadata": metadata or {"source": "agent", "agent_id": self.agent_id},
                "importance": importance
            }
            self.memory.store([record])
        else:
            # For Agno memory implementation
            memory_obj = SimpleNamespace(
                memory=content,
                topics=[],
                metadata=metadata or {"source": "agent", "agent_id": self.agent_id}
            )
            self.memory.add_user_memory(memory_obj, self.agent_id)
    
    def query_memory(self, query):
        """Retrieve information from the agent's memory"""
        if isinstance(self.memory, NeuroMemoryBenchmark):
            return self.memory.similarity_search(query, limit=5)
        else:
            return self.memory.search_user_memories(query, 5, self.agent_id)
    
    def process_message(self, message):
        """Process a message from the message board"""
        # Store the message in memory
        self.store_in_memory(
            f"Message from {message['sender']}: {message['content']}",
            metadata={"type": "message", "sender": message["sender"], "timestamp": message["timestamp"]},
            importance=0.6
        )
        
        # Different roles have different processing logic
        if self.role == "planner":
            # Planners analyze tasks and create subtasks
            if "new_task" in message["content"]:
                # Simulate planning by creating subtasks
                self.plan_task(message["content"])
        elif self.role == "coordinator":
            # Coordinators assign tasks to workers
            if "available_tasks" in message["content"]:
                # Assign available tasks to workers
                self.coordinate_tasks()
        elif self.role == "worker":
            # Workers execute assigned tasks
            if "assigned_task" in message["content"] and self.agent_id in message["content"]:
                # Extract task ID
                task_id = message["content"].split("task_id:")[1].strip()
                self.execute_task(task_id)
    
    def plan_task(self, task_description):
        """Break down a task into subtasks (for planner role)"""
        # Simulate planning by creating subtasks
        from .task import Task
        
        task_id = f"task_{time.time()}_{random.randint(1000, 9999)}"
        
        # Create a main task
        main_task = Task(task_id, complexity=random.uniform(1.0, 3.0))
        self.task_manager.add_task(main_task)
        
        # Create 2-5 subtasks with dependencies
        num_subtasks = random.randint(2, 5)
        subtask_ids = []
        
        for i in range(num_subtasks):
            subtask_id = f"{task_id}_sub{i}"
            # Some subtasks depend on previous subtasks
            dependencies = [subtask_ids[-1]] if subtask_ids and random.random() > 0.5 else []
            
            subtask = Task(
                subtask_id,
                complexity=random.uniform(0.5, 2.0),
                dependencies=dependencies
            )
            subtask_ids.append(subtask_id)
            self.task_manager.add_task(subtask)
        
        # Post a message about available tasks
        self.message_board.post_message(
            self.agent_id,
            f"available_tasks: {', '.join(subtask_ids)}"
        )
        
        # Store the plan in memory
        self.store_in_memory(
            f"Planned task {task_id} with subtasks: {', '.join(subtask_ids)}",
            metadata={"type": "plan", "task_id": task_id, "subtasks": subtask_ids},
            importance=0.8
        )
    
    def coordinate_tasks(self):
        """Assign tasks to workers (for coordinator role)"""
        # Get available tasks
        available_tasks = self.task_manager.get_available_tasks()
        
        # Assign tasks to random workers
        for task in available_tasks:
            # Choose a worker (agent IDs 2 and up are workers in our simulation)
            worker_id = f"agent_{random.randint(2, 5)}"
            
            if self.task_manager.assign_task(task.task_id, worker_id):
                # Post a message about the assignment
                self.message_board.post_message(
                    self.agent_id,
                    f"assigned_task to {worker_id} task_id: {task.task_id}"
                )
                
                # Store the assignment in memory
                self.store_in_memory(
                    f"Assigned task {task.task_id} to worker {worker_id}",
                    metadata={"type": "assignment", "task_id": task.task_id, "worker": worker_id},
                    importance=0.7
                )
    
    def execute_task(self, task_id):
        """Execute an assigned task (for worker role)"""
        task = self.task_manager.get_task(task_id)
        
        if not task or task.assigned_to != self.agent_id:
            return
            
        # Simulate task execution time based on complexity and speed factor
        execution_time = task.complexity / self.speed_factor
        time.sleep(execution_time)
        
        # Determine if task fails based on error rate
        if random.random() < self.error_rate:
            # Task failed
            self.task_manager.fail_task(task_id, "Random execution error")
            self.failed_tasks.append(task_id)
            
            # Store the failure in memory
            self.store_in_memory(
                f"Failed to execute task {task_id}: Random execution error",
                metadata={"type": "task_failure", "task_id": task_id},
                importance=0.9  # High importance for failures
            )
            
            # Notify coordinator
            self.message_board.post_message(
                self.agent_id,
                f"task_failed: {task_id} reason: Random execution error"
            )
        else:
            # Task succeeded
            result = {"status": "success", "data": f"Result of task {task_id} by {self.agent_id}"}
            self.task_manager.complete_task(task_id, result)
            self.completed_tasks.append(task_id)
            
            # Store the result in memory
            self.store_in_memory(
                f"Completed task {task_id} with result: {result}",
                metadata={"type": "task_completion", "task_id": task_id, "result": result},
                importance=0.8
            )
            
            # Share knowledge with other agents
            self.message_board.post_message(
                self.agent_id,
                f"task_completed: {task_id} knowledge: Task {task_id} yielded important information"
            )
            
            # Add to agent's knowledge
            self.knowledge[task_id] = result
    
    def run(self):
        """Main agent loop"""
        while self.running:
            # Check for new messages
            message = self.message_board.get_message()
            
            if message and not message["processed"]:
                process_start = time.time()
                
                # Process the message
                self.process_message(message)
                
                # Mark message as processed
                message["processed"] = True
                
                process_end = time.time()
                
                # Record processing statistics
                self.message_board.record_processing(
                    message, self.agent_id, process_start, process_end
                )
            
            # If agent is a planner, occasionally create new tasks
            if self.role == "planner" and random.random() < 0.1:
                self.message_board.post_message(
                    self.agent_id,
                    f"new_task: Complex task requiring planning {random.randint(1000, 9999)}"
                )
            
            # Workers actively look for tasks
            if self.role == "worker":
                # Check if there are any tasks assigned to this agent
                available_tasks = self.task_manager.get_available_tasks()
                assigned_tasks = [t for t in available_tasks if t.assigned_to == self.agent_id]
                
                for task in assigned_tasks:
                    self.execute_task(task.task_id)
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.01)
    
    def stop(self):
        """Stop the agent"""
        self.running = False
