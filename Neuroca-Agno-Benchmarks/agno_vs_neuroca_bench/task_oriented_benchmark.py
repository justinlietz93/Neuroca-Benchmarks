# task_oriented_benchmark.py
import json, time, statistics, random, logging, os
from pathlib import Path
import psutil
import pandas as pd
from types import SimpleNamespace
import textwrap
import re
import csv
import lorem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("task_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("task_benchmark")

# Import memory implementations from the benchmark file
from bench import NeuroMemoryBenchmark, AgnoMemoryBenchmark

class TaskOrientedBenchmark:
    """
    Benchmark that tests a memory system's performance in task-oriented scenarios
    using a QA dataset and simulated external API calls.
    """
    
    def __init__(self, name, memory_system, dataset_size=1000, query_count=100):
        self.name = name
        self.memory = memory_system
        self.dataset_size = dataset_size
        self.query_count = min(query_count, dataset_size)
        
    def generate_dataset(self, output_path="qa_dataset.csv"):
        """Generate a synthetic QA dataset for benchmarking"""
        logger.info(f"Generating synthetic QA dataset with {self.dataset_size} entries")
        
        # Define categories for more coherent data
        categories = [
            "Science", "History", "Geography", "Technology", 
            "Arts", "Sports", "Food", "Travel", "Health", "Business"
        ]
        
        # Generate dataset
        data = []
        for i in range(self.dataset_size):
            category = random.choice(categories)
            
            # Generate a context paragraph about the category
            context = lorem.paragraph()[:200]  # Truncate to avoid very long contexts
            
            # Create a "fact" based on the category
            facts = {
                "Science": f"The {random.choice(['atomic', 'quantum', 'molecular', 'cellular'])} {random.choice(['structure', 'theory', 'principle', 'mechanism'])} was discovered in {random.randint(1800, 2020)}.",
                "History": f"The {random.choice(['war', 'treaty', 'revolution', 'dynasty'])} of {random.choice(['Eastern', 'Western', 'Northern', 'Southern'])} {random.choice(['Europe', 'Asia', 'Africa', 'America'])} occurred in {random.randint(1400, 1950)}.",
                "Geography": f"The {random.choice(['mountain', 'river', 'lake', 'desert'])} of {random.choice(['Eastern', 'Western', 'Northern', 'Southern'])} {random.choice(['Europe', 'Asia', 'Africa', 'America'])} spans {random.randint(100, 5000)} kilometers.",
                "Technology": f"The {random.choice(['programming language', 'framework', 'protocol', 'algorithm'])} was developed by {random.choice(['Google', 'Microsoft', 'Apple', 'IBM'])} in {random.randint(1970, 2022)}.",
                "Arts": f"The {random.choice(['painting', 'sculpture', 'novel', 'symphony'])} was created by {random.choice(['famous', 'renowned', 'influential', 'visionary'])} artist in {random.randint(1500, 2010)}.",
                "Sports": f"The {random.choice(['championship', 'tournament', 'league', 'match'])} was won by the {random.choice(['team', 'player', 'athlete', 'competitor'])} from {random.choice(['Europe', 'Asia', 'Africa', 'America'])}.",
                "Food": f"The {random.choice(['dish', 'cuisine', 'recipe', 'ingredient'])} is popular in {random.choice(['Eastern', 'Western', 'Northern', 'Southern'])} {random.choice(['Europe', 'Asia', 'Africa', 'America'])}.",
                "Travel": f"The {random.choice(['city', 'destination', 'resort', 'landmark'])} attracts {random.randint(1, 10)} million visitors annually.",
                "Health": f"The {random.choice(['treatment', 'therapy', 'diet', 'exercise'])} has been shown to {random.choice(['improve', 'enhance', 'boost', 'increase'])} {random.choice(['health', 'wellbeing', 'fitness', 'recovery'])}.",
                "Business": f"The {random.choice(['company', 'corporation', 'startup', 'enterprise'])} reported {random.randint(1, 100)} billion in revenue in {random.randint(2010, 2023)}."
            }
            
            fact = facts[category]
            
            # Add the fact to the context
            full_context = context + " " + fact
            
            # Generate a question about the fact
            question_templates = {
                "Science": ["When was the {} {} discovered?", "What scientific principle was discovered in {}?"],
                "History": ["When did the {} of {} {} occur?", "Which historical event happened in {}?"],
                "Geography": ["How long is the {} of {} {}?", "What geographical feature spans {} kilometers?"],
                "Technology": ["Who developed the {} in {}?", "Which company created the {} in {}?"],
                "Arts": ["When was the {} created?", "Who created the {} in {}?"],
                "Sports": ["Who won the {}?", "Which {} won the {}?"],
                "Food": ["Where is the {} popular?", "Which cuisine is popular in {} {}?"],
                "Travel": ["How many visitors does the {} attract annually?", "Which destination attracts {} million visitors?"],
                "Health": ["What can {} {} {}?", "Which therapy has been shown to {} {}?"],
                "Business": ["How much revenue did the {} report in {}?", "Which company reported {} billion in revenue?"]
            }
            
            # Extract key elements from the fact for template filling
            fact_words = fact.split()
            template = random.choice(question_templates[category])
            
            # Simple template filling - not perfect but generates reasonable questions
            if "{}" in template:
                placeholders = []
                for _ in range(template.count("{}")):
                    placeholders.append(random.choice(fact_words))
                question = template.format(*placeholders)
            else:
                question = template
                
            # Extract answer from the fact - simplistic approach
            answer_words = [w for w in fact_words if w.isdigit() or len(w) > 6]
            answer = random.choice(answer_words) if answer_words else random.choice(fact_words)
            
            # Add entry to dataset
            data.append({
                "id": f"qa_{i}",
                "category": category,
                "question": question,
                "context": full_context,
                "answer": answer,
                "metadata": {
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                    "type": "factoid",
                    "source": "synthetic"
                }
            })
        
        # Save to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["id", "category", "question", "context", "answer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in data:
                # Simplified row with just the main fields
                writer.writerow({
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "context": item["context"],
                    "answer": item["answer"]
                })
        
        logger.info(f"Dataset saved to {output_path}")
        return data
    
    def load_dataset(self, path="qa_dataset.csv"):
        """Load the QA dataset from a CSV file"""
        if not Path(path).exists():
            logger.info(f"Dataset not found at {path}, generating new one")
            return self.generate_dataset(path)
        
        logger.info(f"Loading dataset from {path}")
        data = []
        with open(path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        
        logger.info(f"Loaded {len(data)} entries from dataset")
        return data
    
    def store_in_memory(self, data):
        """Store dataset entries in the memory system"""
        logger.info(f"Storing {len(data)} entries in {self.name}")
        
        start_time = time.time()
        
        if isinstance(self.memory, NeuroMemoryBenchmark):
            # For Neuroca memory implementation
            records = []
            for item in data:
                record = {
                    "id": item["id"],
                    "content": item["context"],
                    "metadata": {
                        "category": item["category"],
                        "question": item["question"],
                        "answer": item["answer"]
                    },
                    "importance": 0.7  # Moderate importance for all entries
                }
                records.append(record)
            
            self.memory.store(records)
        else:
            # For Agno memory implementation
            for item in data:
                memory_obj = SimpleNamespace(
                    memory=item["context"],
                    topics=[item["category"]],
                    metadata={
                        "question": item["question"],
                        "answer": item["answer"]
                    }
                )
                self.memory.add_user_memory(memory_obj, "benchmark-user")
                
        storage_time = time.time() - start_time
        logger.info(f"Storage completed in {storage_time:.2f}s")
        
        return storage_time
    
    def query_memory(self, question):
        """Query the memory system with a question"""
        if isinstance(self.memory, NeuroMemoryBenchmark):
            start_time = time.time()
            results = self.memory.similarity_search(question, limit=5)
            query_time = time.time() - start_time
            
            # Extract answers from results
            answers = []
            for record_id, item in results:
                # Handle Neuroca's multi-tiered metadata structure
                if isinstance(item, dict):
                    if "metadata" in item and isinstance(item["metadata"], dict) and "answer" in item["metadata"]:
                        # Direct metadata access
                        answers.append(item["metadata"]["answer"])
                    elif "metadata" in item and isinstance(item["metadata"], dict) and "question" in item["metadata"]:
                        # The answer might be in the context - try to extract it
                        context = item.get("content", "")
                        question_text = item["metadata"].get("question", "")
                        
                        # If we have both question and context, use a simple extraction heuristic
                        if context and question_text:
                            # Look for keywords from the question in the context
                            question_words = set(question_text.lower().split())
                            context_words = context.lower().split()
                            
                            # Find words near the question words in the context
                            for i, word in enumerate(context_words):
                                if word in question_words or any(qword in word for qword in question_words):
                                    # Extract a potential answer (words after the match)
                                    candidate = " ".join(context_words[i:i+5])
                                    answers.append(candidate)
                                    break
            
            # If we found no answers through metadata, try to extract directly from content
            if not answers:
                for record_id, item in results:
                    if isinstance(item, dict) and "content" in item:
                        # Simple extraction heuristic - return a segment of the content
                        content = item["content"]
                        # Take the most relevant 5-10 words
                        words = content.split()
                        if len(words) > 5:
                            potential_answer = " ".join(words[3:8])  # Skip first few words
                            answers.append(potential_answer)
            
            return answers, query_time, len(results)
        else:
            start_time = time.time()
            results = self.memory.search_user_memories(question, 5, "benchmark-user")
            query_time = time.time() - start_time
            
            # Extract answers from results
            answers = []
            for item in results:
                if isinstance(item, dict) and "metadata" in item and "answer" in item["metadata"]:
                    answers.append(item["metadata"]["answer"])
            
            return answers, query_time, len(results)
    
    def measure_memory_usage(self):
        """Measure current memory usage"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    
    def simulate_api_call(self, query):
        """Simulate an external API call"""
        # Simulate network latency and processing time
        time.sleep(0.05)
        
        # Generate a mock response based on the query
        words = query.split()
        relevant_word = words[min(len(words)-1, 2)]  # Pick a word from the query
        
        response = {
            "status": "success",
            "query": query,
            "results": [
                {"relevance": 0.95, "source": "api", "content": f"Information about {relevant_word}"},
                {"relevance": 0.82, "source": "api", "content": f"Additional data related to {relevant_word}"}
            ],
            "timestamp": time.time()
        }
        
        return response
    
    def run_benchmark(self):
        """Run the full benchmark suite"""
        logger.info(f"Starting task-oriented benchmark for {self.name}")
        print(f"\n===== Task-Oriented Benchmark: {self.name} =====\n")
        
        # Load or generate dataset
        dataset = self.load_dataset()
        if len(dataset) > self.dataset_size:
            dataset = dataset[:self.dataset_size]
        
        # Measure initial memory usage
        initial_memory = self.measure_memory_usage()
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Phase 1: Store data in memory
        print("\nPhase 1: Storing data in memory...")
        storage_time = self.store_in_memory(dataset)
        print(f"Storage completed in {storage_time:.2f}s")
        
        # Measure memory after storage
        post_storage_memory = self.measure_memory_usage()
        memory_increase = post_storage_memory - initial_memory
        print(f"Memory usage after storage: {post_storage_memory:.1f} MB (increase: {memory_increase:.1f} MB)")
        
        # Phase 2: Query memory
        print("\nPhase 2: Testing memory retrieval...")
        
        # Select random entries for querying
        query_samples = random.sample(dataset, min(self.query_count, len(dataset)))
        
        # Metrics
        retrieval_times = []
        retrieval_counts = []
        correct_answers = 0
        combined_latencies = []
        
        for i, item in enumerate(query_samples):
            question = item["question"]
            expected_answer = item["answer"]
            
            # Query memory system
            answers, query_time, result_count = self.query_memory(question)
            retrieval_times.append(query_time)
            retrieval_counts.append(result_count)
            
            # Check if expected answer is in results
            found = any(expected_answer.lower() in answer.lower() for answer in answers)
            
            # Debug output for first 5 queries
            if i < 5:
                print(f"\nDEBUG - Query {i+1}: '{question}'")
                print(f"Expected answer: '{expected_answer}'")
                print(f"Retrieved answers ({len(answers)}): {answers[:3]}")
                print(f"Match found: {found}")
            
            if found:
                correct_answers += 1
                
            # Simulate API call and combined response
            api_start = time.time()
            api_response = self.simulate_api_call(question)
            
            # Simulate combining memory and API results
            combined_response = {
                "memory_results": answers,
                "api_results": api_response["results"],
                "source_count": len(answers) + len(api_response["results"])
            }
            combined_time = time.time() - api_start + query_time
            combined_latencies.append(combined_time)
            
            # Log progress occasionally
            if i % 10 == 0 or i == len(query_samples) - 1:
                print(f"Processed {i+1}/{len(query_samples)} queries - Accuracy: {correct_answers/(i+1)*100:.1f}%")
        
        # Calculate metrics
        accuracy = correct_answers / len(query_samples) if query_samples else 0
        avg_retrieval_time = statistics.mean(retrieval_times) * 1000  # Convert to ms
        avg_combined_time = statistics.mean(combined_latencies) * 1000  # Convert to ms
        p95_retrieval = statistics.quantiles(retrieval_times, n=20)[-1] * 1000 if retrieval_times else 0
        
        # Final memory usage
        final_memory = self.measure_memory_usage()
        
        # Print results
        print(f"\n===== {self.name} Results =====")
        print(f"Accuracy: {accuracy*100:.1f}% ({correct_answers}/{len(query_samples)})")
        print(f"Average retrieval time: {avg_retrieval_time:.2f}ms")
        print(f"P95 retrieval time: {p95_retrieval:.2f}ms")
        print(f"Average combined time (memory + API): {avg_combined_time:.2f}ms")
        print(f"Final memory usage: {final_memory:.1f} MB (net change: {final_memory-initial_memory:.1f} MB)")
        
        # Return comprehensive metrics
        return {
            "system": self.name,
            "dataset_size": self.dataset_size,
            "query_count": self.query_count,
            "accuracy": accuracy,
            "correct_answers": correct_answers,
            "storage_time": storage_time,
            "avg_retrieval_time_ms": avg_retrieval_time,
            "p95_retrieval_time_ms": p95_retrieval,
            "avg_combined_time_ms": avg_combined_time,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": final_memory - initial_memory
        }

def compare_task_benchmarks(dataset_size=1000, query_count=100):
    """
    Run task-oriented benchmarks on both memory systems and compare results
    """
    logger.info(f"Starting comparative task-oriented benchmark with {dataset_size} records")
    
    # Initialize memory systems
    neuroca_memory = NeuroMemoryBenchmark()
    agno_memory = AgnoMemoryBenchmark()
    
    # Create benchmark instances
    neuroca_bench = TaskOrientedBenchmark("Neuroca", neuroca_memory, dataset_size, query_count)
    agno_bench = TaskOrientedBenchmark("Agno", agno_memory, dataset_size, query_count)
    
    # Run benchmarks
    neuroca_results = neuroca_bench.run_benchmark()
    agno_results = agno_bench.run_benchmark()
    
    # Compare results
    print("\n===== Task-Oriented Benchmark Comparison =====\n")
    print(f"{'Metric':<30} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}")
    print("-" * 75)
    
    # Accuracy
    neuroca_accuracy = neuroca_results["accuracy"] * 100
    agno_accuracy = agno_results["accuracy"] * 100
    acc_diff = neuroca_accuracy - agno_accuracy
    print(f"{'Accuracy (%)':<30} {neuroca_accuracy:<15.1f} {agno_accuracy:<15.1f} {acc_diff:+.1f}")
    
    # Storage time
    neuroca_store = neuroca_results["storage_time"]
    agno_store = agno_results["storage_time"]
    store_diff_pct = ((neuroca_store - agno_store) / max(agno_store, 0.001)) * 100
    print(f"{'Storage Time (s)':<30} {neuroca_store:<15.2f} {agno_store:<15.2f} {store_diff_pct:+.1f}%")
    
    # Retrieval time
    neuroca_retrieve = neuroca_results["avg_retrieval_time_ms"]
    agno_retrieve = agno_results["avg_retrieval_time_ms"]
    retrieve_diff_pct = ((neuroca_retrieve - agno_retrieve) / max(agno_retrieve, 0.001)) * 100
    print(f"{'Retrieval Time (ms)':<30} {neuroca_retrieve:<15.2f} {agno_retrieve:<15.2f} {retrieve_diff_pct:+.1f}%")
    
    # P95 retrieval time
    neuroca_p95 = neuroca_results["p95_retrieval_time_ms"]
    agno_p95 = agno_results["p95_retrieval_time_ms"]
    p95_diff_pct = ((neuroca_p95 - agno_p95) / max(agno_p95, 0.001)) * 100
    print(f"{'P95 Retrieval (ms)':<30} {neuroca_p95:<15.2f} {agno_p95:<15.2f} {p95_diff_pct:+.1f}%")
    
    # Combined time
    neuroca_combined = neuroca_results["avg_combined_time_ms"]
    agno_combined = agno_results["avg_combined_time_ms"]
    combined_diff_pct = ((neuroca_combined - agno_combined) / max(agno_combined, 0.001)) * 100
    print(f"{'Combined Time (ms)':<30} {neuroca_combined:<15.2f} {agno_combined:<15.2f} {combined_diff_pct:+.1f}%")
    
    # Memory usage
    neuroca_mem = neuroca_results["memory_increase_mb"]
    agno_mem = agno_results["memory_increase_mb"]
    mem_diff_pct = ((neuroca_mem - agno_mem) / max(agno_mem, 0.001)) * 100
    print(f"{'Memory Increase (MB)':<30} {neuroca_mem:<15.1f} {agno_mem:<15.1f} {mem_diff_pct:+.1f}%")
    
    # Generate detailed report
    generate_task_benchmark_report(neuroca_results, agno_results)
    
    return neuroca_results, agno_results

def generate_task_benchmark_report(neuroca_results, agno_results):
    """Generate a detailed report of the task-oriented benchmark results"""
    report_path = Path("task_benchmark_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Task-Oriented Benchmark Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This benchmark tests how well each memory system performs in a task-oriented scenario.\n")
        f.write("It simulates a real-world application where an agent needs to store information,\n")
        f.write("retrieve relevant information based on queries, and combine memory results with API data.\n\n")
        
        f.write("## Test Methodology\n\n")
        f.write("1. **Dataset Creation**\n")
        f.write(f"   * Generated a synthetic QA dataset with {neuroca_results['dataset_size']} entries\n")
        f.write("   * Each entry contains a question, context, and answer\n")
        f.write("   * Entries are categorized across diverse domains (Science, History, etc.)\n\n")
        
        f.write("2. **Memory Storage**\n")
        f.write("   * All dataset entries are stored in the memory system\n")
        f.write("   * Storage time and memory consumption are measured\n\n")
        
        f.write("3. **Memory Retrieval**\n")
        f.write(f"   * Randomly selects {neuroca_results['query_count']} questions from the dataset\n")
        f.write("   * Queries the memory system for each question\n")
        f.write("   * Simulates API calls to combine with memory results\n")
        f.write("   * Measures accuracy, retrieval time, and resource usage\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("```\n")
        f.write(f"{'Metric':<30} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
        f.write("-" * 75 + "\n")
        
        # Accuracy
        neuroca_accuracy = neuroca_results["accuracy"] * 100
        agno_accuracy = agno_results["accuracy"] * 100
        acc_diff = neuroca_accuracy - agno_accuracy
        f.write(f"{'Accuracy (%)':<30} {neuroca_accuracy:<15.1f} {agno_accuracy:<15.1f} {acc_diff:+.1f}\n")
        
        # Storage time
        neuroca_store = neuroca_results["storage_time"]
        agno_store = agno_results["storage_time"]
        store_diff_pct = ((neuroca_store - agno_store) / max(agno_store, 0.001)) * 100
        f.write(f"{'Storage Time (s)':<30} {neuroca_store:<15.2f} {agno_store:<15.2f} {store_diff_pct:+.1f}%\n")
        
        # Retrieval time
        neuroca_retrieve = neuroca_results["avg_retrieval_time_ms"]
        agno_retrieve = agno_results["avg_retrieval_time_ms"]
        retrieve_diff_pct = ((neuroca_retrieve - agno_retrieve) / max(agno_retrieve, 0.001)) * 100
        f.write(f"{'Retrieval Time (ms)':<30} {neuroca_retrieve:<15.2f} {agno_retrieve:<15.2f} {retrieve_diff_pct:+.1f}%\n")
        
        # P95 retrieval time
        neuroca_p95 = neuroca_results["p95_retrieval_time_ms"]
        agno_p95 = agno_results["p95_retrieval_time_ms"]
        p95_diff_pct = ((neuroca_p95 - agno_p95) / max(agno_p95, 0.001)) * 100
        f.write(f"{'P95 Retrieval (ms)':<30} {neuroca_p95:<15.2f} {agno_p95:<15.2f} {p95_diff_pct:+.1f}%\n")
        
        # Combined time
        neuroca_combined = neuroca_results["avg_combined_time_ms"]
        agno_combined = agno_results["avg_combined_time_ms"]
        combined_diff_pct = ((neuroca_combined - agno_combined) / max(agno_combined, 0.001)) * 100
        f.write(f"{'Combined Time (ms)':<30} {neuroca_combined:<15.2f} {agno_combined:<15.2f} {combined_diff_pct:+.1f}%\n")
        
        # Memory usage
        neuroca_mem = neuroca_results["memory_increase_mb"]
        agno_mem = agno_results["memory_increase_mb"]
        mem_diff_pct = ((neuroca_mem - agno_mem) / max(agno_mem, 0.001)) * 100
        f.write(f"{'Memory Increase (MB)':<30} {neuroca_mem:<15.1f} {agno_mem:<15.1f} {mem_diff_pct:+.1f}%\n")
        
        f.write("```\n\n")
        
        # Analysis section
        f.write("## Analysis\n\n")
        
        # Accuracy analysis
        f.write("### Task Accuracy\n\n")
        if abs(acc_diff) < 10:
            f.write("Both memory systems demonstrate similar accuracy in retrieving correct answers.\n")
            f.write("This suggests that both systems effectively index and retrieve information based on\n")
            f.write("semantic similarity, even with a moderately sized dataset.\n\n")
        else:
            better_system = "Neuroca" if acc_diff > 0 else "Agno"
            f.write(f"{better_system} achieves significantly better accuracy in answer retrieval.\n")
            if better_system == "Neuroca":
                f.write("This suggests that Neuroca's vector-based similarity search may provide\n")
                f.write("an advantage in mapping questions to relevant contexts that contain the answers.\n\n")
            else:
                f.write("This suggests that Agno's approach to memory storage and retrieval\n")
                f.write("may be more effective at maintaining the relationships between questions and answers.\n\n")
        
        # Storage performance
        f.write("### Storage Performance\n\n")
        if neuroca_store > agno_store * 1.2:  # 20% slower
            f.write("Neuroca's storage operations are slower than Agno's, which is expected due to\n")
            f.write("the computation of vector embeddings during storage. This upfront cost potentially\n")
            f.write("enables more sophisticated semantic retrieval during the query phase.\n\n")
        else:
            f.write("Both systems show comparable storage performance. While Neuroca computes\n")
            f.write("embeddings which theoretically should add overhead, the performance difference\n")
            f.write("is not significant with this dataset size.\n\n")
        
        # Retrieval performance
        f.write("### Retrieval Performance\n\n")
        if abs(retrieve_diff_pct) < 15:  # Within 15%
            f.write("Both systems demonstrate comparable retrieval speeds. This suggests that\n")
            f.write("for this dataset size, the retrieval algorithms of both systems are\n")
            f.write("adequately optimized.\n\n")
        else:
            faster_system = "Neuroca" if neuroca_retrieve < agno_retrieve else "Agno"
            f.write(f"{faster_system}'s retrieval operations are faster, which could be important\n")
            f.write(f"in scenarios requiring real-time memory access. ")
            if faster_system == "Neuroca":
                f.write("This demonstrates the\n")
                f.write("benefit of pre-computed vector embeddings for efficient similarity search.\n\n")
            else:
                f.write("This suggests that\n")
                f.write("the simpler retrieval approach may be more efficient for certain workloads.\n\n")
        
        # Memory usage
        f.write("### Memory Efficiency\n\n")
        if neuroca_mem > agno_mem * 1.5:  # 50% more memory
            f.write("Neuroca uses substantially more memory than Agno, likely due to the storage of\n")
            f.write("vector embeddings alongside the actual content. This is an important consideration\n")
            f.write("for deployment scenarios with memory constraints.\n\n")
        else:
            f.write("Both systems have comparable memory usage. Neuroca's additional storage of vector\n")
            f.write("embeddings does not significantly impact memory consumption at this dataset scale.\n\n")
        
        # Limitations section
        f.write("## Limitations\n\n")
        f.write("- The benchmark uses simulated memory systems rather than the actual implementations\n")
        f.write("- The synthetic dataset may not fully represent the complexity of real-world data\n")
        f.write("- The simulated API calls do not capture the full complexity of real API integrations\n")
        f.write("- The accuracy metric is simplistic, only checking for exact matches of the expected answer\n")
        f.write("- The test focuses on factoid QA, which is just one of many potential memory usage patterns\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("This benchmark demonstrates how both memory systems perform in a task-oriented scenario\n")
        f.write("that combines memory retrieval with external API calls. Such scenarios are common in\n")
        f.write("agent-based systems where memory acts as a knowledge repository that complements\n")
        f.write("external data sources.\n\n")
        
        if neuroca_accuracy > agno_accuracy and neuroca_retrieve < agno_retrieve:
            f.write("Neuroca shows advantages in both accuracy and retrieval speed, suggesting it may be\n")
            f.write("the better choice for applications where both correctness and performance are critical.\n")
        elif agno_accuracy > neuroca_accuracy and agno_retrieve < neuroca_retrieve:
            f.write("Agno shows advantages in both accuracy and retrieval speed, suggesting it may be\n")
            f.write("the better choice for applications where both correctness and performance are critical.\n")
        elif neuroca_mem < agno_mem:
            f.write("Neuroca uses less memory while maintaining similar performance metrics, which could\n")
            f.write("make it the preferred choice for memory-constrained environments.\n")
        elif agno_mem < neuroca_mem:
            f.write("Agno uses less memory while maintaining similar performance metrics, which could\n")
            f.write("make it the preferred choice for memory-constrained environments.\n")
        else:
            f.write("Both systems show comparable performance across key metrics, suggesting that either\n")
            f.write("could be effective depending on specific application requirements and constraints.\n")
    
    logger.info(f"Generated task benchmark report at {report_path}")
    print(f"\nGenerated task benchmark report at {report_path}")

if __name__ == "__main__":
    # Use a smaller dataset size by default to ensure it runs quickly
    # but still provides meaningful results
    compare_task_benchmarks(dataset_size=500, query_count=50)
