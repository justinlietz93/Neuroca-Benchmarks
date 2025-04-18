# message_board.py
import queue
import threading
import time
import statistics

class MessageBoard:
    """Simulates a shared message board for agent communication"""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.message_stats = []
        self.lock = threading.Lock()
        
    def post_message(self, sender, message, timestamp=None):
        """Post a message to the board"""
        if timestamp is None:
            timestamp = time.time()
            
        message_obj = {
            "sender": sender,
            "content": message,
            "timestamp": timestamp,
            "processed": False
        }
        
        self.queue.put(message_obj)
        return timestamp
        
    def get_message(self, timeout=0.1):
        """Get the next message from the board"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def record_processing(self, message, receiver, process_start, process_end):
        """Record stats about message processing"""
        with self.lock:
            self.message_stats.append({
                "sender": message["sender"],
                "receiver": receiver,
                "content": message["content"],
                "timestamp": message["timestamp"],
                "process_start": process_start,
                "process_end": process_end,
                "wait_time": process_start - message["timestamp"],
                "process_time": process_end - process_start,
                "total_time": process_end - message["timestamp"]
            })
            
    def get_stats(self):
        """Get communication statistics"""
        with self.lock:
            if not self.message_stats:
                return {
                    "message_count": 0,
                    "avg_wait_time": 0,
                    "avg_process_time": 0,
                    "avg_total_time": 0,
                    "p95_total_time": 0
                }
                
            wait_times = [stat["wait_time"] for stat in self.message_stats]
            process_times = [stat["process_time"] for stat in self.message_stats]
            total_times = [stat["total_time"] for stat in self.message_stats]
            
            return {
                "message_count": len(self.message_stats),
                "avg_wait_time": statistics.mean(wait_times) * 1000,  # Convert to ms
                "avg_process_time": statistics.mean(process_times) * 1000,  # Convert to ms
                "avg_total_time": statistics.mean(total_times) * 1000,  # Convert to ms
                "p95_total_time": statistics.quantiles(total_times, n=20)[-1] * 1000 if len(total_times) >= 20 else 0
            }
