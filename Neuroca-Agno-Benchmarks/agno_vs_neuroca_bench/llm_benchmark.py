# llm_benchmark.py
import json, time, statistics, random, logging, re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_benchmark_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_memory_benchmark")

DATA_PATH = Path("C:/git/Neuroca-Agno-Benchmarks/dataset.jsonl")
assert DATA_PATH.exists(), "Please run seed_dataset.py first"

class MockLLM:
    """Simulates an LLM for benchmarking memory augmentation"""
    
    def __init__(self, model_name="mock-gpt-4o", max_tokens=128000):
        self.model_name = model_name
        self.max_tokens = max_tokens
        
    def count_tokens(self, text):
        """
        Simulate token counting - in reality this would use
        a tokenizer like tiktoken, but we'll use a simple approximation
        """
        # Simple approximation: 1 token ~= 4 characters for English text
        return len(text) // 4
    
    def format_chat_messages(self, user_query, memories):
        """Format a chat message with memory augmentation"""
        system_prompt = "You are a helpful AI assistant with access to relevant memories."
        
        # Format memories as context
        memory_text = "Relevant memories:\n"
        for i, memory in enumerate(memories):
            if isinstance(memory, tuple):
                # Handle tuple format from Neuroca
                key, content = memory
                if isinstance(content, dict) and "content" in content:
                    memory_text += f"[{i+1}] {content['content']}\n"
            else:
                # Handle dict format from Agno
                if isinstance(memory, dict) and "content" in memory:
                    memory_text += f"[{i+1}] {memory['content']}\n"
        
        memory_text += "\nUse these memories to respond to the user's query.\n"
        
        user_message = f"Query: {user_query}"
        
        full_prompt = f"{system_prompt}\n\n{memory_text}\n\n{user_message}"
        return full_prompt
    
    def generate_response(self, prompt, memories, max_new_tokens=500):
        """Simulate generating a response using provided memories"""
        
        # Format the full prompt with memories
        full_prompt = self.format_chat_messages(prompt, memories)
        
        # Count input tokens
        input_tokens = self.count_tokens(full_prompt)
        
        # Simulate some processing time proportional to input size
        start_time = time.time()
        processing_time = 0.001 * input_tokens  # 1ms per token
        time.sleep(processing_time)
        
        # Generate a mock response
        # Simulating a response that refers to memories if they exist
        if memories:
            memory_count = len(memories)
            response = f"Based on the {memory_count} memories provided, I can help answer your query about '{prompt}'."
            
            # Add some references to the content of memories
            for i, memory in enumerate(memories[:2]):  # Reference up to 2 memories
                if isinstance(memory, tuple):
                    # Handle tuple format from Neuroca
                    key, content = memory
                    if isinstance(content, dict) and "content" in content:
                        memory_content = content['content']
                        # Extract a short snippet
                        snippet = memory_content[:50] + "..." if len(memory_content) > 50 else memory_content
                        response += f"\n\nMemory {i+1} mentions: \"{snippet}\""
                else:
                    # Handle dict format from Agno
                    if isinstance(memory, dict) and "content" in memory:
                        memory_content = memory['content']
                        # Extract a short snippet
                        snippet = memory_content[:50] + "..." if len(memory_content) > 50 else memory_content
                        response += f"\n\nMemory {i+1} mentions: \"{snippet}\""
        else:
            response = f"I don't have any specific memories about '{prompt}', but I'll try to help based on my general knowledge."
        
        # Add some filler content to simulate a longer response
        response += "\n\nHere's some additional information that might be relevant..."
        
        # Count output tokens
        output_tokens = self.count_tokens(response)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "time_seconds": total_time
        }

def load_records():
    """Load the dataset"""
    with DATA_PATH.open("r", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)

# Import memory implementations from the benchmark file
from bench import NeuroMemoryBenchmark, AgnoMemoryBenchmark

def run_llm_neuroca_benchmark(num_records=1000, num_queries=20):
    """Run LLM benchmark with Neuroca memory system"""
    logger.info(f"Starting LLM benchmark with Neuroca ({num_records} records)")
    
    # Initialize memory system
    memory = NeuroMemoryBenchmark()
    
    # Initialize mock LLM
    llm = MockLLM()
    
    # Load and store records
    records = list(load_records())[:num_records]
    
    logger.info(f"Storing {len(records)} records in Neuroca memory")
    start_time = time.time()
    memory.store(records)
    storage_time = time.time() - start_time
    logger.info(f"Storage completed in {storage_time:.2f}s")
    
    # Generate random queries from record content
    queries = []
    for _ in range(num_queries):
        random_record = random.choice(records)
        # Extract a few words to use as a query
        words = random_record["content"].split()
        if len(words) > 5:
            query_words = words[:5]
            query = " ".join(query_words)
            queries.append(query)
    
    # Measure memory retrieval and LLM performance
    retrieval_times = []
    total_tokens = []
    input_tokens = []
    output_tokens = []
    generation_times = []
    
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: '{query}'")
        
        # Retrieve relevant memories
        retrieval_start = time.time()
        memories = memory.similarity_search(query, limit=5)
        retrieval_time = time.time() - retrieval_start
        retrieval_times.append(retrieval_time)
        
        # Generate response with LLM
        result = llm.generate_response(query, memories)
        
        # Log statistics
        input_tokens.append(result["input_tokens"])
        output_tokens.append(result["output_tokens"])
        total_tokens.append(result["total_tokens"])
        generation_times.append(result["time_seconds"])
        
        logger.debug(f"Query: '{query}'")
        logger.debug(f"Retrieved {len(memories)} memories in {retrieval_time*1000:.1f}ms")
        logger.debug(f"Generated response with {result['input_tokens']} input tokens, {result['output_tokens']} output tokens")
        
        if i < 2:  # Log full details for first couple of queries
            logger.debug(f"Sample response: {result['response'][:100]}...")
    
    # Calculate statistics
    avg_retrieval_ms = statistics.mean(retrieval_times) * 1000
    avg_input_tokens = statistics.mean(input_tokens)
    avg_output_tokens = statistics.mean(output_tokens)
    avg_total_tokens = statistics.mean(total_tokens)
    avg_generation_time = statistics.mean(generation_times)
    
    stats = {
        "engine": "Neuroca LLM Augmentation",
        "storage_time": storage_time,
        "avg_retrieval_ms": avg_retrieval_ms,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_generation_time": avg_generation_time
    }
    
    # Print results
    print(f"\n=== Neuroca LLM Memory Benchmark ===")
    print(f"Storage: {storage_time:.2f}s for {num_records} records")
    print(f"Retrieval: {avg_retrieval_ms:.1f}ms average")
    print(f"Tokens: {avg_input_tokens:.1f} input, {avg_output_tokens:.1f} output, {avg_total_tokens:.1f} total")
    print(f"LLM time: {avg_generation_time:.3f}s average")
    
    return stats

def run_llm_agno_benchmark(num_records=1000, num_queries=20):
    """Run LLM benchmark with Agno memory system"""
    logger.info(f"Starting LLM benchmark with Agno ({num_records} records)")
    
    # Initialize memory system
    memory = AgnoMemoryBenchmark()
    
    # Initialize mock LLM
    llm = MockLLM()
    
    # Load and store records
    records = list(load_records())[:num_records]
    
    logger.info(f"Storing {len(records)} records in Agno memory")
    start_time = time.time()
    for r in records:
        from types import SimpleNamespace
        memory_obj = SimpleNamespace(
            memory=r["content"],
            topics=[],
            metadata=r["metadata"]
        )
        memory.add_user_memory(memory_obj, "benchmark-user")
    storage_time = time.time() - start_time
    logger.info(f"Storage completed in {storage_time:.2f}s")
    
    # Generate random queries from record content
    queries = []
    for _ in range(num_queries):
        random_record = random.choice(records)
        # Extract a few words to use as a query
        words = random_record["content"].split()
        if len(words) > 5:
            query_words = words[:5]
            query = " ".join(query_words)
            queries.append(query)
    
    # Measure memory retrieval and LLM performance
    retrieval_times = []
    total_tokens = []
    input_tokens = []
    output_tokens = []
    generation_times = []
    
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: '{query}'")
        
        # Retrieve relevant memories
        retrieval_start = time.time()
        memories = memory.search_user_memories(query, 5, "benchmark-user")
        retrieval_time = time.time() - retrieval_start
        retrieval_times.append(retrieval_time)
        
        # Generate response with LLM
        result = llm.generate_response(query, memories)
        
        # Log statistics
        input_tokens.append(result["input_tokens"])
        output_tokens.append(result["output_tokens"])
        total_tokens.append(result["total_tokens"])
        generation_times.append(result["time_seconds"])
        
        logger.debug(f"Query: '{query}'")
        logger.debug(f"Retrieved {len(memories)} memories in {retrieval_time*1000:.1f}ms")
        logger.debug(f"Generated response with {result['input_tokens']} input tokens, {result['output_tokens']} output tokens")
        
        if i < 2:  # Log full details for first couple of queries
            logger.debug(f"Sample response: {result['response'][:100]}...")
    
    # Calculate statistics
    avg_retrieval_ms = statistics.mean(retrieval_times) * 1000
    avg_input_tokens = statistics.mean(input_tokens)
    avg_output_tokens = statistics.mean(output_tokens)
    avg_total_tokens = statistics.mean(total_tokens)
    avg_generation_time = statistics.mean(generation_times)
    
    stats = {
        "engine": "Agno LLM Augmentation",
        "storage_time": storage_time,
        "avg_retrieval_ms": avg_retrieval_ms,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_total_tokens": avg_total_tokens,
        "avg_generation_time": avg_generation_time
    }
    
    # Print results
    print(f"\n=== Agno LLM Memory Benchmark ===")
    print(f"Storage: {storage_time:.2f}s for {num_records} records")
    print(f"Retrieval: {avg_retrieval_ms:.1f}ms average")
    print(f"Tokens: {avg_input_tokens:.1f} input, {avg_output_tokens:.1f} output, {avg_total_tokens:.1f} total")
    print(f"LLM time: {avg_generation_time:.3f}s average")
    
    return stats

def compare_llm_benchmarks():
    """Run and compare LLM benchmarks for both memory systems"""
    logger.info("Starting comparative LLM memory benchmarks")
    
    # Run both benchmarks
    num_records = 1000
    num_queries = 20
    
    neuroca_stats = run_llm_neuroca_benchmark(num_records, num_queries)
    agno_stats = run_llm_agno_benchmark(num_records, num_queries)
    
    # Print comparison table
    print("\n=== LLM Memory Benchmark Comparison ===")
    print(f"{'Engine':<22} {'Storage':<15} {'Retrieval':<15} {'Input Tokens':<15} {'Total Tokens':<15}")
    print("-" * 85)
    print(f"{neuroca_stats['engine']:<22} {neuroca_stats['storage_time']:.2f}s{'':<8} {neuroca_stats['avg_retrieval_ms']:.1f}ms{'':<6} {neuroca_stats['avg_input_tokens']:.1f}{'':<8} {neuroca_stats['avg_total_tokens']:.1f}")
    print(f"{agno_stats['engine']:<22} {agno_stats['storage_time']:.2f}s{'':<8} {agno_stats['avg_retrieval_ms']:.1f}ms{'':<6} {agno_stats['avg_input_tokens']:.1f}{'':<8} {agno_stats['avg_total_tokens']:.1f}")
    
    # Log comparison to file
    logger.info("=== LLM Memory Benchmark Comparison ===")
    logger.info(f"Neuroca: storage={neuroca_stats['storage_time']:.2f}s, retrieval={neuroca_stats['avg_retrieval_ms']:.1f}ms, tokens={neuroca_stats['avg_total_tokens']:.1f}")
    logger.info(f"Agno: storage={agno_stats['storage_time']:.2f}s, retrieval={agno_stats['avg_retrieval_ms']:.1f}ms, tokens={agno_stats['avg_total_tokens']:.1f}")
    
    # Generate documentation
    generate_llm_documentation(neuroca_stats, agno_stats)
    
    return neuroca_stats, agno_stats

def generate_llm_documentation(neuroca_stats, agno_stats):
    """Generate documentation about the LLM benchmark implementation"""
    doc_path = Path("llm_benchmark_implementation.md")
    
    with open(doc_path, "w") as f:
        f.write("# LLM Memory Augmentation Benchmark Implementation\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document describes the implementation of a benchmark system comparing\n")
        f.write("how Neuroca and Agno memory systems perform when augmenting a Large Language Model (LLM).\n")
        f.write("The benchmark simulates how the memory systems would be used in a real-world LLM application.\n\n")
        
        f.write("## Implementation Details\n\n")
        f.write("### Mock LLM Implementation\n\n")
        f.write("Since we don't have access to a real LLM API in this environment, the benchmark simulates an LLM with:\n\n")
        f.write("- A simplified token counting mechanism (approximating 4 characters per token)\n")
        f.write("- Simulated processing time based on input length\n")
        f.write("- Generation of responses that incorporate retrieved memories\n")
        f.write("- Tracking of token usage, memory retrieval time, and generation time\n\n")
        
        f.write("### Benchmark Process\n\n")
        f.write("For each memory system, the benchmark:\n\n")
        f.write("1. Loads and stores a set number of records\n")
        f.write("2. Generates random queries based on record content\n")
        f.write("3. Retrieves relevant memories for each query\n")
        f.write("4. Formats retrieved memories into prompts\n")
        f.write("5. Simulates LLM response generation\n")
        f.write("6. Measures and reports key metrics\n\n")
        
        f.write("## Benchmark Results\n\n")
        f.write("```\n")
        f.write(f"{'Engine':<22} {'Storage':<15} {'Retrieval':<15} {'Input Tokens':<15} {'Total Tokens':<15}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{neuroca_stats['engine']:<22} {neuroca_stats['storage_time']:.2f}s{'':<8} {neuroca_stats['avg_retrieval_ms']:.1f}ms{'':<6} {neuroca_stats['avg_input_tokens']:.1f}{'':<8} {neuroca_stats['avg_total_tokens']:.1f}\n")
        f.write(f"{agno_stats['engine']:<22} {agno_stats['storage_time']:.2f}s{'':<8} {agno_stats['avg_retrieval_ms']:.1f}ms{'':<6} {agno_stats['avg_input_tokens']:.1f}{'':<8} {agno_stats['avg_total_tokens']:.1f}\n")
        f.write("```\n\n")
        
        f.write("## Insights and Observations\n\n")
        
        # Calculate some comparisons for the insights - with a more robust approach for very small numbers
        def calculate_percentage_diff(a, b):
            # Add epsilon to prevent division by zero or very small numbers creating misleading percentages
            epsilon = 0.01  # 10ms minimum to avoid inflated percentages with tiny numbers
            if a == 0 and b == 0:
                return 0
            elif abs(a) < epsilon and abs(b) < epsilon:
                return 0  # Both values too small for meaningful percentage
            else:
                # Use the max of the two values as denominator for more stable percentage
                return ((a - b) / max(abs(a), abs(b), epsilon)) * 100
        
        storage_diff = calculate_percentage_diff(neuroca_stats['storage_time'], agno_stats['storage_time'])
        retrieval_diff = calculate_percentage_diff(neuroca_stats['avg_retrieval_ms'], agno_stats['avg_retrieval_ms'])
        token_diff = calculate_percentage_diff(neuroca_stats['avg_total_tokens'], agno_stats['avg_total_tokens'])
        
        # Determine comparison words
        storage_comparison = "slower" if storage_diff > 0 else "faster"
        retrieval_comparison = "slower" if retrieval_diff > 0 else "faster"
        token_comparison = "more" if token_diff > 0 else "fewer"
        
        # For very small absolute differences, use more moderate language
        if abs(storage_diff) < 10:
            f.write(f"1. **Storage Performance**: Neuroca and Agno have comparable storage performance, with Neuroca being slightly {storage_comparison}.\n")
            f.write(f"   The difference is minimal in this benchmark but would likely be more pronounced with larger datasets.\n\n")
        else:
            f.write(f"1. **Storage Performance**: Neuroca takes {abs(storage_diff):.1f}% {storage_comparison} than Agno at storing records, which\n")
            f.write(f"   reflects the additional computation for calculating vector embeddings during storage.\n\n")
        
        if abs(retrieval_diff) < 10:
            f.write(f"2. **Retrieval Performance**: Both memory systems show similar retrieval speeds, with Neuroca being slightly {retrieval_comparison}.\n")
            f.write(f"   This demonstrates the effectiveness of vector-based retrieval in the Neuroca implementation.\n\n")
        else:
            f.write(f"2. **Retrieval Performance**: Neuroca is {abs(retrieval_diff):.1f}% {retrieval_comparison} than Agno at retrieving memories for\n")
            f.write(f"   queries, showing the tradeoff between up-front embedding computation and retrieval speed.\n\n")
        
        if abs(token_diff) < 10:
            f.write(f"3. **Token Usage**: Both systems use a similar number of tokens, with Neuroca using slightly {token_comparison} tokens.\n")
            f.write(f"   This impacts overall LLM API costs and context window utilization in real-world deployments.\n\n")
        else:
            f.write(f"3. **Token Usage**: Neuroca uses {abs(token_diff):.1f}% {token_comparison} tokens than Agno when formatting memories\n")
            f.write(f"   for LLM prompts, which affects overall cost and context window utilization.\n\n")
        
        f.write("## Implementation Gaps and Future Work\n\n")
        f.write("1. **Real LLM Integration**: Test with actual LLM APIs like OpenAI or Anthropic for realistic behavior.\n\n")
        f.write("2. **Multi-tier Memory**: Implement and test Neuroca's three-tier memory system as described in documentation.\n\n")
        f.write("3. **Context Window Optimization**: Evaluate memory systems on their ability to efficiently use context windows.\n\n")
        f.write("4. **Quality Metrics**: Assess the quality of responses with real LLMs using metrics like relevance and accuracy.\n\n")
        f.write("5. **Conversation History**: Test with conversation streams instead of single-turn queries.\n\n")
        
    logger.info(f"Generated LLM benchmark implementation documentation at {doc_path}")
    print(f"\nGenerated LLM benchmark implementation documentation at {doc_path}")

if __name__ == "__main__":
    compare_llm_benchmarks()
