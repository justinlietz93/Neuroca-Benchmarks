# conversation_benchmark.py
import json, time, statistics, random, logging
from pathlib import Path
from types import SimpleNamespace
import textwrap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conversation_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("conversation_benchmark")

# Import memory implementations from the benchmark file
from bench import NeuroMemoryBenchmark, AgnoMemoryBenchmark

class MockAgent:
    """Simulates an agent using a memory system in a conversation"""
    
    def __init__(self, name, memory_system, user_id="test-user"):
        self.name = name
        self.memory = memory_system
        self.user_id = user_id
        self.message_history = []
        
    def count_tokens(self, text):
        """Simple token counter approximation"""
        return len(text) // 4
        
    def process_user_message(self, message):
        """Process a user message and generate a response"""
        start_time = time.time()
        
        # Add message to history
        self.message_history.append({"role": "user", "content": message})
        
        # Step 1: Store the user message in memory
        if isinstance(self.memory, NeuroMemoryBenchmark):
            # For Neuroca memory implementation
            memory_record = {
                "id": f"msg_{len(self.message_history)}",
                "content": message,
                "metadata": {"type": "user_message", "timestamp": time.time()},
                "importance": 0.7  # Assign moderate importance
            }
            store_start = time.time()
            self.memory.store([memory_record])
            store_time = time.time() - store_start
        else:
            # For Agno memory implementation
            memory_obj = SimpleNamespace(
                memory=message,
                topics=[],
                metadata={"type": "user_message", "timestamp": time.time()}
            )
            store_start = time.time()
            self.memory.add_user_memory(memory_obj, self.user_id)
            store_time = time.time() - store_start
            
        # Step 2: Retrieve relevant memories for this message
        query = message
        if len(query) > 20:
            # Use first few words as query if message is long
            query = " ".join(message.split()[:5])
            
        retrieve_start = time.time()
        if isinstance(self.memory, NeuroMemoryBenchmark):
            relevant_memories = self.memory.similarity_search(query, limit=3)
        else:
            relevant_memories = self.memory.search_user_memories(query, 3, self.user_id)
        retrieve_time = time.time() - retrieve_start
            
        # Step 3: Generate a response using the memories
        response = self.generate_response(message, relevant_memories)
        
        # Step 4: Store the agent's response in memory
        if isinstance(self.memory, NeuroMemoryBenchmark):
            response_record = {
                "id": f"resp_{len(self.message_history)}",
                "content": response,
                "metadata": {"type": "assistant_message", "timestamp": time.time()},
                "importance": 0.6
            }
            self.memory.store([response_record])
        else:
            response_obj = SimpleNamespace(
                memory=response,
                topics=[],
                metadata={"type": "assistant_message", "timestamp": time.time()}
            )
            self.memory.add_user_memory(response_obj, self.user_id)
            
        # Add response to history
        self.message_history.append({"role": "assistant", "content": response})
        
        total_time = time.time() - start_time
        
        # Calculate token usage
        message_tokens = self.count_tokens(message)
        memory_tokens = sum(self.count_tokens(m[1]["content"] if isinstance(m, tuple) else m["content"]) 
                           for m in relevant_memories)
        response_tokens = self.count_tokens(response)
        
        metrics = {
            "store_time": store_time,
            "retrieve_time": retrieve_time,
            "total_time": total_time,
            "message_tokens": message_tokens,
            "memory_tokens": memory_tokens,
            "response_tokens": response_tokens,
            "total_tokens": message_tokens + memory_tokens + response_tokens,
            "num_memories": len(relevant_memories)
        }
        
        return response, metrics
        
    def generate_response(self, message, memories):
        """Generate a simple mock response based on message and memories"""
        # Create a simple response
        if not memories:
            return f"I don't recall our previous conversation about that. Could you tell me more about {message.split()[0] if message else ''}?"
            
        response_parts = ["Based on what I remember:"]
        
        for i, memory in enumerate(memories):
            if isinstance(memory, tuple):
                # Handle tuple format from Neuroca
                key, content = memory
                if isinstance(content, dict) and "content" in content:
                    memory_content = content['content']
                    response_parts.append(f"- Memory {i+1}: {memory_content[:50]}...")
            else:
                # Handle dict format from Agno
                if isinstance(memory, dict) and "content" in memory:
                    memory_content = memory['content']
                    response_parts.append(f"- Memory {i+1}: {memory_content[:50]}...")
        
        # Add response to message
        message_words = message.split()
        if message_words:
            response_parts.append(f"\nRegarding your question about {message_words[0]}, I think...")
            
        return "\n".join(response_parts)
        
    def get_memory_count(self):
        """Get the count of memories stored"""
        if isinstance(self.memory, NeuroMemoryBenchmark):
            return len(self.memory.memory_store)
        else:
            return len(self.memory.memories)

def run_conversation_benchmark(system_type="neuroca"):
    """Run a simulated conversation benchmark"""
    logger.info(f"Starting conversation benchmark with {system_type}")
    
    # Initialize the memory system
    if system_type.lower() == "neuroca":
        memory_system = NeuroMemoryBenchmark()
        agent_name = "Neuroca Agent"
    else:
        memory_system = AgnoMemoryBenchmark()
        agent_name = "Agno Agent"
        
    # Create the agent
    agent = MockAgent(agent_name, memory_system)
    
    # Sample conversation turns
    conversation = [
        "Hello, my name is Peter and I like gardening.",
        "I especially enjoy growing carrots and lettuce.",
        "Do you have any recommendations for my garden?",
        "I also have a pet rabbit named Floppy.",
        "What vegetables would be good for my rabbit?",
        "Can you remind me what vegetables I said I like to grow?",
        "Who did I say my pet was?",
        "What was my name again?",
        "Thanks for the conversation!",
        "Can you summarize everything we talked about?"
    ]
    
    # Metrics collection
    store_times = []
    retrieve_times = []
    total_times = []
    message_tokens = []
    memory_tokens = []
    response_tokens = []
    total_tokens = []
    memory_counts = []
    
    # Run the conversation
    print(f"\n===== Conversation with {agent_name} =====\n")
    
    turn_metrics = []
    for i, message in enumerate(conversation):
        print(f"\n[Turn {i+1}] User: {message}")
        
        # Process the message and get response with metrics
        response, metrics = agent.process_user_message(message)
        turn_metrics.append(metrics)
        
        # Format and print the response
        wrapped_response = textwrap.fill(response, width=80)
        print(f"\n{agent_name}: {wrapped_response}\n")
        print(f"[Metrics] Store: {metrics['store_time']*1000:.1f}ms, " +
              f"Retrieve: {metrics['retrieve_time']*1000:.1f}ms, " +
              f"Memories: {metrics['num_memories']}, " +
              f"Tokens: {metrics['total_tokens']}")
              
        # Collect metrics
        store_times.append(metrics['store_time'])
        retrieve_times.append(metrics['retrieve_time'])
        total_times.append(metrics['total_time'])
        message_tokens.append(metrics['message_tokens'])
        memory_tokens.append(metrics['memory_tokens'])
        response_tokens.append(metrics['response_tokens'])
        total_tokens.append(metrics['total_tokens'])
        memory_counts.append(agent.get_memory_count())
    
    # Calculate aggregated metrics
    avg_store_ms = statistics.mean(store_times) * 1000
    avg_retrieve_ms = statistics.mean(retrieve_times) * 1000
    avg_total_ms = statistics.mean(total_times) * 1000
    avg_message_tokens = statistics.mean(message_tokens)
    avg_memory_tokens = statistics.mean(memory_tokens)
    avg_response_tokens = statistics.mean(response_tokens)
    avg_total_tokens = statistics.mean(total_tokens)
    
    # Print summary
    print(f"\n===== {agent_name} Performance Summary =====")
    print(f"Average store time: {avg_store_ms:.1f}ms")
    print(f"Average retrieve time: {avg_retrieve_ms:.1f}ms")
    print(f"Average total processing time: {avg_total_ms:.1f}ms")
    print(f"Average tokens per turn: {avg_total_tokens:.1f}")
    print(f"Final memory count: {memory_counts[-1]}")
    
    # Return metrics for comparison
    return {
        "agent": agent_name,
        "avg_store_ms": avg_store_ms,
        "avg_retrieve_ms": avg_retrieve_ms,
        "avg_total_ms": avg_total_ms,
        "avg_message_tokens": avg_message_tokens,
        "avg_memory_tokens": avg_memory_tokens,
        "avg_response_tokens": avg_response_tokens,
        "avg_total_tokens": avg_total_tokens,
        "final_memory_count": memory_counts[-1],
        "turn_metrics": turn_metrics
    }

def compare_conversation_systems():
    """Run and compare conversation benchmarks for both memory systems"""
    logger.info("Starting comparative conversation benchmark")
    
    # Run benchmarks for both systems
    neuroca_metrics = run_conversation_benchmark("neuroca")
    agno_metrics = run_conversation_benchmark("agno")
    
    # Print comparison table
    print("\n===== Memory System Comparison for Conversation =====")
    print(f"{'Metric':<25} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ("Store time (ms)", "avg_store_ms"),
        ("Retrieve time (ms)", "avg_retrieve_ms"),
        ("Total time (ms)", "avg_total_ms"),
        ("Tokens per turn", "avg_total_tokens"),
        ("Final memory count", "final_memory_count")
    ]
    
    for label, key in metrics_to_compare:
        neuroca_val = neuroca_metrics[key]
        agno_val = agno_metrics[key]
        
        # Calculate difference as percentage
        if agno_val != 0:
            diff_pct = ((neuroca_val - agno_val) / agno_val) * 100
            diff_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = "N/A"
            
        print(f"{label:<25} {neuroca_val:<15.1f} {agno_val:<15.1f} {diff_str:<15}")
    
    # Generate documentation
    generate_conversation_documentation(neuroca_metrics, agno_metrics)
    
    logger.info("Completed comparative conversation benchmark")

def generate_conversation_documentation(neuroca_metrics, agno_metrics):
    """Generate documentation about the conversation benchmark results"""
    doc_path = Path("conversation_benchmark.md")
    
    with open(doc_path, "w") as f:
        f.write("# Conversation Memory Benchmark\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document presents the results of a simulated conversation benchmark comparing\n")
        f.write("the Neuroca and Agno memory systems. The benchmark simulates a 10-turn conversation\n")
        f.write("where the agent needs to store and retrieve memories during the interaction.\n\n")
        
        f.write("## Methodology\n\n")
        f.write("The benchmark simulates a conversation with the following steps for each turn:\n\n")
        f.write("1. The user sends a message\n")
        f.write("2. The agent stores the message in memory\n")
        f.write("3. The agent retrieves relevant memories based on the message\n")
        f.write("4. The agent generates a response using the memories\n")
        f.write("5. The agent stores its own response in memory\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("```\n")
        f.write(f"{'Metric':<25} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
        f.write("-" * 70 + "\n")
        
        metrics_to_compare = [
            ("Store time (ms)", "avg_store_ms"),
            ("Retrieve time (ms)", "avg_retrieve_ms"),
            ("Total time (ms)", "avg_total_ms"),
            ("Tokens per turn", "avg_total_tokens"),
            ("Final memory count", "final_memory_count")
        ]
        
        for label, key in metrics_to_compare:
            neuroca_val = neuroca_metrics[key]
            agno_val = agno_metrics[key]
            
            if agno_val != 0:
                diff_pct = ((neuroca_val - agno_val) / agno_val) * 100
                diff_str = f"{diff_pct:+.1f}%"
            else:
                diff_str = "N/A"
                
            f.write(f"{label:<25} {neuroca_val:<15.1f} {agno_val:<15.1f} {diff_str:<15}\n")
            
        f.write("```\n\n")
        
        f.write("## Analysis\n\n")
        
        # Storage time comparison
        store_diff_pct = ((neuroca_metrics["avg_store_ms"] - agno_metrics["avg_store_ms"]) / 
                          max(agno_metrics["avg_store_ms"], 0.1)) * 100
                          
        if store_diff_pct > 20:
            f.write("### Memory Storage\n\n")
            f.write("Neuroca's storage operations are slower than Agno's. This is expected as Neuroca\n")
            f.write("computes vector embeddings during storage, which adds computational overhead but\n")
            f.write("enables more sophisticated semantic retrieval later.\n\n")
        else:
            f.write("### Memory Storage\n\n")
            f.write("Both systems show comparable storage performance in this limited benchmark.\n")
            f.write("With larger datasets or more complex content, we would expect to see more\n")
            f.write("significant differences in storage performance.\n\n")
            
        # Retrieval time comparison
        retrieve_diff_pct = ((neuroca_metrics["avg_retrieve_ms"] - agno_metrics["avg_retrieve_ms"]) / 
                            max(agno_metrics["avg_retrieve_ms"], 0.1)) * 100
                            
        retrieve_comparison = "faster" if retrieve_diff_pct < 0 else "slower"
        
        f.write("### Memory Retrieval\n\n")
        if abs(retrieve_diff_pct) < 15:
            f.write("Both systems demonstrate similar retrieval performance in this benchmark.\n")
            f.write("The retrieval times are close enough that other factors like network latency\n")
            f.write("would likely have a larger impact in real-world scenarios.\n\n")
        else:
            f.write(f"Neuroca's retrieval is {abs(retrieve_diff_pct):.1f}% {retrieve_comparison} than Agno's.\n")
            if retrieve_comparison == "faster":
                f.write("This demonstrates the advantage of having pre-computed vector embeddings\n")
                f.write("for similarity search, which pays off during retrieval operations.\n\n")
            else:
                f.write("This might be due to the overhead of vector similarity calculations\n")
                f.write("compared to Agno's simpler retrieval mechanism.\n\n")
        
        # Token usage
        token_diff_pct = ((neuroca_metrics["avg_total_tokens"] - agno_metrics["avg_total_tokens"]) / 
                         max(agno_metrics["avg_total_tokens"], 1)) * 100
                         
        token_comparison = "more" if token_diff_pct > 0 else "fewer"
        
        f.write("### Token Efficiency\n\n")
        if abs(token_diff_pct) < 10:
            f.write("Both memory systems use a similar number of tokens per conversation turn.\n")
            f.write("This suggests that neither system has a significant advantage in terms of\n")
            f.write("token efficiency for this particular conversation scenario.\n\n")
        else:
            f.write(f"Neuroca uses {abs(token_diff_pct):.1f}% {token_comparison} tokens than Agno per turn.\n")
            if token_comparison == "fewer":
                f.write("This could lead to cost savings in production deployments where\n")
                f.write("LLM API usage is charged based on token count.\n\n")
            else:
                f.write("This could potentially lead to higher costs in production deployments.\n")
                f.write("However, if the additional tokens provide better context, it might be\n")
                f.write("a worthwhile tradeoff for improved response quality.\n\n")
        
        f.write("## Limitations\n\n")
        f.write("- This benchmark uses simplified implementations of both memory systems\n")
        f.write("- The conversation is short and does not test long-term memory retention\n")
        f.write("- Response generation is mocked and does not evaluate response quality\n")
        f.write("- The benchmark does not test multi-user scenarios or memory isolation\n")
        f.write("- Vector similarity in Neuroca is simulated rather than using a proper embedding model\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This benchmark provides a basic comparison of the operational characteristics\n")
        f.write("of Neuroca and Agno memory systems in a conversational context. For production\n")
        f.write("deployment decisions, additional testing with real LLMs and actual user data\n")
        f.write("would be recommended to assess response quality and user experience impacts.\n")
        
    logger.info(f"Generated conversation benchmark documentation at {doc_path}")
    print(f"\nGenerated conversation benchmark documentation at {doc_path}")

if __name__ == "__main__":
    compare_conversation_systems()
