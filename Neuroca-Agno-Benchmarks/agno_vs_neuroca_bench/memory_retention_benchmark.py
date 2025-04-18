# memory_retention_benchmark.py
import json, time, statistics, random, logging, os
from pathlib import Path
from types import SimpleNamespace
import lorem  # For generating random text
import textwrap
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_retention_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("memory_retention_benchmark")

# Import memory implementations from the benchmark file
from bench import NeuroMemoryBenchmark, AgnoMemoryBenchmark

class MemoryRetentionTest:
    """
    Tests a memory system's ability to retain important facts
    amidst noise and recall them accurately when queried.
    """
    
    def __init__(self, name, memory_system, user_id="test-user"):
        self.name = name
        self.memory = memory_system
        self.user_id = user_id
        self.key_facts = []
        self.quiz_questions = []
        self.message_history = []
        
    def generate_key_facts(self, num_facts=5):
        """Generate key facts that will be tested later"""
        facts = [
            "My name is Alexander Thompson and I was born on June 12, 1985.",
            "I have a golden retriever named Charlie who is 7 years old.",
            "My favorite book is 'The Great Gatsby' by F. Scott Fitzgerald.",
            "I'm allergic to peanuts and shellfish, which can cause severe reactions.",
            "I graduated from Stanford University with a degree in Computer Science in 2007.",
            "My phone number is 555-123-4567 and my email is alex.thompson@example.com.",
            "I work as a Senior Software Engineer at TechCorp International.",
            "My wife's name is Sophia and we got married on August 23, 2015.",
            "I have a vacation home in Lake Tahoe that I visit during winter for skiing.",
            "I speak three languages fluently: English, Spanish, and Mandarin."
        ]
        
        # Select a subset of facts to use
        self.key_facts = random.sample(facts, min(num_facts, len(facts)))
        
        # Generate quiz questions for each fact
        self.quiz_questions = []
        for fact in self.key_facts:
            if "name is" in fact.lower():
                self.quiz_questions.append("What is my name?")
            elif "born on" in fact.lower():
                self.quiz_questions.append("When was I born?")
            elif "dog" in fact.lower() or "retriever" in fact.lower():
                self.quiz_questions.append("What is my dog's name and age?")
            elif "favorite book" in fact.lower():
                self.quiz_questions.append("What is my favorite book?")
            elif "allergic" in fact.lower():
                self.quiz_questions.append("What allergies do I have?")
            elif "graduated" in fact.lower():
                self.quiz_questions.append("Where did I graduate from and what was my degree?")
            elif "phone number" in fact.lower():
                self.quiz_questions.append("What is my phone number?")
            elif "work" in fact.lower():
                self.quiz_questions.append("Where do I work and what is my job title?")
            elif "wife" in fact.lower():
                self.quiz_questions.append("What is my wife's name and when did we get married?")
            elif "vacation home" in fact.lower():
                self.quiz_questions.append("Where is my vacation home?")
            elif "languages" in fact.lower():
                self.quiz_questions.append("What languages do I speak?")
                
        return self.key_facts, self.quiz_questions
    
    def generate_noise_text(self, min_sentences=3, max_sentences=7):
        """Generate random noise text to fill memory with irrelevant information"""
        # The lorem package doesn't accept number of sentences as a parameter
        # Let's use a different approach to generate random text
        sentences = []
        num_sentences = random.randint(min_sentences, max_sentences)
        for _ in range(num_sentences):
            sentences.append(lorem.sentence())
        return " ".join(sentences)
    
    def store_message(self, message, is_user=True, importance=0.5):
        """Store a message in the memory system"""
        if isinstance(self.memory, NeuroMemoryBenchmark):
            # For Neuroca memory implementation
            msg_id = f"msg_{len(self.message_history)}"
            memory_record = {
                "id": msg_id,
                "content": message,
                "metadata": {
                    "type": "user_message" if is_user else "assistant_message", 
                    "timestamp": time.time()
                },
                "importance": importance
            }
            start_time = time.time()
            self.memory.store([memory_record])
            store_time = time.time() - start_time
        else:
            # For Agno memory implementation
            memory_obj = SimpleNamespace(
                memory=message,
                topics=[],
                metadata={
                    "type": "user_message" if is_user else "assistant_message",
                    "timestamp": time.time()
                }
            )
            start_time = time.time()
            self.memory.add_user_memory(memory_obj, self.user_id)
            store_time = time.time() - start_time
            
        # Add to message history
        self.message_history.append({
            "role": "user" if is_user else "assistant",
            "content": message
        })
        
        return store_time
    
    def retrieve_memories(self, query):
        """Retrieve memories related to a query"""
        start_time = time.time()
        if isinstance(self.memory, NeuroMemoryBenchmark):
            memories = self.memory.similarity_search(query, limit=5)
        else:
            memories = self.memory.search_user_memories(query, 5, self.user_id)
        retrieve_time = time.time() - start_time
        
        return memories, retrieve_time
    
    def run_memory_flood_test(self, noise_messages=20):
        """
        Run a test that floods memory with a mix of important facts and noise,
        then tests recall of the important facts.
        """
        logger.info(f"Starting memory flood test with {self.name}")
        print(f"\n===== Memory Flood Test: {self.name} =====\n")
        
        # Generate key facts and corresponding quiz questions
        key_facts, quiz_questions = self.generate_key_facts(5)
        
        # Phase 1: Store key facts interspersed with noise
        print("Phase 1: Storing key facts and noise...\n")
        
        store_times = []
        for i in range(noise_messages):
            # Every 5th message is a key fact, others are noise
            if i % 5 == 0 and i // 5 < len(key_facts):
                fact_index = i // 5
                message = key_facts[fact_index]
                print(f"[KEY FACT] {message}")
                # Store key facts with higher importance
                store_time = self.store_message(message, is_user=True, importance=0.9)
                store_times.append(store_time)
            else:
                # Generate and store noise
                noise = self.generate_noise_text()
                print(f"[NOISE] {noise[:50]}...")
                store_time = self.store_message(noise, is_user=True, importance=0.3)
                store_times.append(store_time)
                
            # Simulate assistant response
            assistant_msg = f"I understand. Please continue."
            self.store_message(assistant_msg, is_user=False, importance=0.2)
            
        # Calculate average store time
        avg_store_time = statistics.mean(store_times) * 1000  # Convert to ms
        print(f"\nAverage storage time: {avg_store_time:.2f}ms")
        
        # Phase 2: Quiz on key facts
        print("\nPhase 2: Testing recall of key facts...\n")
        
        quiz_results = []
        for i, question in enumerate(quiz_questions):
            print(f"Question {i+1}: {question}")
            
            # Retrieve memories related to the question
            memories, retrieve_time = self.retrieve_memories(question)
            
            # Check if the corresponding fact was retrieved
            fact_found = False
            fact_position = -1
            corresponding_fact = key_facts[i]
            
            for j, memory in enumerate(memories):
                memory_content = ""
                if isinstance(memory, tuple):
                    # Handle tuple format from Neuroca
                    key, content = memory
                    if isinstance(content, dict) and "content" in content:
                        memory_content = content['content']
                else:
                    # Handle dict format from Agno
                    if isinstance(memory, dict) and "content" in memory:
                        memory_content = memory['content']
                
                # Check if this memory contains the key fact
                if self._is_same_fact(memory_content, corresponding_fact):
                    fact_found = True
                    fact_position = j
                    break
            
            # Record result
            result = {
                "question": question,
                "fact_found": fact_found,
                "position": fact_position + 1 if fact_position >= 0 else "Not found",
                "retrieve_time_ms": retrieve_time * 1000,
                "num_memories": len(memories)
            }
            quiz_results.append(result)
            
            # Print result
            status = "✅ Found" if fact_found else "❌ Not found"
            position_str = f" (position {fact_position+1})" if fact_position >= 0 else ""
            print(f"Result: {status}{position_str} in {retrieve_time*1000:.2f}ms")
            print(f"Returned {len(memories)} memories\n")
        
        # Calculate overall performance
        correct_answers = sum(1 for r in quiz_results if r["fact_found"])
        accuracy = correct_answers / len(quiz_results) if quiz_results else 0
        avg_retrieval_time = statistics.mean([r["retrieve_time_ms"] for r in quiz_results])
        
        print(f"===== {self.name} Results =====")
        print(f"Accuracy: {accuracy*100:.1f}% ({correct_answers}/{len(quiz_results)} facts recalled)")
        print(f"Average retrieval time: {avg_retrieval_time:.2f}ms")
        
        # Return metrics
        return {
            "system": self.name,
            "accuracy": accuracy,
            "correct_answers": correct_answers,
            "total_questions": len(quiz_results),
            "avg_store_time_ms": avg_store_time,
            "avg_retrieval_time_ms": avg_retrieval_time,
            "quiz_results": quiz_results
        }
    
    def _is_same_fact(self, memory_text, fact_text):
        """
        Check if the memory text contains the key fact.
        Uses fuzzy matching to account for slight variations.
        """
        # Simple case: exact match
        if fact_text in memory_text:
            return True
            
        # Split fact into key components for partial matching
        fact_words = set(re.findall(r'\b\w+\b', fact_text.lower()))
        memory_words = set(re.findall(r'\b\w+\b', memory_text.lower()))
        
        # Count matching words
        matching_words = fact_words.intersection(memory_words)
        
        # If we match enough key words, consider it a match
        # Require at least 70% of the fact words to be present
        match_threshold = 0.7
        match_ratio = len(matching_words) / len(fact_words) if fact_words else 0
        
        return match_ratio >= match_threshold

def compare_memory_retention():
    """
    Run the memory retention benchmark on both Neuroca and Agno memory systems
    and compare their performance.
    """
    logger.info("Starting memory retention comparison")
    
    # Initialize memory systems
    neuroca_memory = NeuroMemoryBenchmark()
    agno_memory = AgnoMemoryBenchmark()
    
    # Create test instances
    neuroca_test = MemoryRetentionTest("Neuroca Memory", neuroca_memory)
    agno_test = MemoryRetentionTest("Agno Memory", agno_memory)
    
    # Run tests
    neuroca_results = neuroca_test.run_memory_flood_test(noise_messages=30)
    agno_results = agno_test.run_memory_flood_test(noise_messages=30)
    
    # Compare results
    print("\n===== Memory Retention Test Results =====\n")
    print(f"{'Metric':<30} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}")
    print("-" * 75)
    
    # Accuracy
    neuroca_accuracy = neuroca_results["accuracy"] * 100
    agno_accuracy = agno_results["accuracy"] * 100
    acc_diff = neuroca_accuracy - agno_accuracy
    print(f"{'Recall Accuracy (%)':<30} {neuroca_accuracy:<15.1f} {agno_accuracy:<15.1f} {acc_diff:+.1f}")
    
    # Average storage time
    neuroca_store = neuroca_results["avg_store_time_ms"]
    agno_store = agno_results["avg_store_time_ms"]
    store_diff_pct = ((neuroca_store - agno_store) / max(agno_store, 0.001)) * 100
    print(f"{'Storage Time (ms)':<30} {neuroca_store:<15.2f} {agno_store:<15.2f} {store_diff_pct:+.1f}%")
    
    # Average retrieval time
    neuroca_retrieve = neuroca_results["avg_retrieval_time_ms"]
    agno_retrieve = agno_results["avg_retrieval_time_ms"]
    retrieve_diff_pct = ((neuroca_retrieve - agno_retrieve) / max(agno_retrieve, 0.001)) * 100
    print(f"{'Retrieval Time (ms)':<30} {neuroca_retrieve:<15.2f} {agno_retrieve:<15.2f} {retrieve_diff_pct:+.1f}%")
    
    # Generate detailed report
    generate_retention_report(neuroca_results, agno_results)
    
    return neuroca_results, agno_results

def generate_retention_report(neuroca_results, agno_results):
    """Generate a detailed report of the memory retention benchmark results"""
    report_path = Path("memory_retention_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Memory Retention Benchmark Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This benchmark tests how well each memory system retains important facts\n")
        f.write("when they are interspersed with noise, and how accurately those facts can be\n")
        f.write("retrieved when queried. This simulates a real-world scenario where important\n")
        f.write("information is often buried within longer conversations.\n\n")
        
        f.write("## Test Methodology\n\n")
        f.write("1. **Phase 1: Memory Flooding**\n")
        f.write("   * 5 key facts are inserted into the memory system\n")
        f.write("   * These facts are interspersed with random noise messages\n")
        f.write("   * The key facts are assigned higher importance scores than noise\n\n")
        
        f.write("2. **Phase 2: Recall Testing**\n")
        f.write("   * The system is queried with questions about the key facts\n")
        f.write("   * We check if the memory system successfully retrieves the relevant fact\n")
        f.write("   * We measure retrieval time and position of the fact in results\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("```\n")
        f.write(f"{'Metric':<30} {'Neuroca':<15} {'Agno':<15} {'Difference':<15}\n")
        f.write("-" * 75 + "\n")
        
        # Accuracy
        neuroca_accuracy = neuroca_results["accuracy"] * 100
        agno_accuracy = agno_results["accuracy"] * 100
        acc_diff = neuroca_accuracy - agno_accuracy
        f.write(f"{'Recall Accuracy (%)':<30} {neuroca_accuracy:<15.1f} {agno_accuracy:<15.1f} {acc_diff:+.1f}\n")
        
        # Average storage time
        neuroca_store = neuroca_results["avg_store_time_ms"]
        agno_store = agno_results["avg_store_time_ms"]
        store_diff_pct = ((neuroca_store - agno_store) / max(agno_store, 0.001)) * 100
        f.write(f"{'Storage Time (ms)':<30} {neuroca_store:<15.2f} {agno_store:<15.2f} {store_diff_pct:+.1f}%\n")
        
        # Average retrieval time
        neuroca_retrieve = neuroca_results["avg_retrieval_time_ms"]
        agno_retrieve = agno_results["avg_retrieval_time_ms"]
        retrieve_diff_pct = ((neuroca_retrieve - agno_retrieve) / max(agno_retrieve, 0.001)) * 100
        f.write(f"{'Retrieval Time (ms)':<30} {neuroca_retrieve:<15.2f} {agno_retrieve:<15.2f} {retrieve_diff_pct:+.1f}%\n")
        
        # Correct answers
        neuroca_correct = neuroca_results["correct_answers"]
        agno_correct = agno_results["correct_answers"]
        total_questions = neuroca_results["total_questions"]
        f.write(f"{'Correct Answers':<30} {neuroca_correct}/{total_questions}{' '*7} {agno_correct}/{total_questions}{' '*7} {neuroca_correct-agno_correct:+}\n")
        
        f.write("```\n\n")
        
        # Analysis section
        f.write("## Analysis\n\n")
        
        # Accuracy analysis
        f.write("### Fact Recall Accuracy\n\n")
        if abs(acc_diff) < 10:
            f.write("Both memory systems demonstrate similar accuracy in recalling key facts amongst noise.\n")
            f.write("This suggests that with proper importance scoring, both systems can effectively\n")
            f.write("prioritize critical information retention.\n\n")
        else:
            better_system = "Neuroca" if acc_diff > 0 else "Agno"
            f.write(f"{better_system} shows significantly better accuracy in retrieving key facts.\n")
            if better_system == "Neuroca":
                f.write("This suggests that Neuroca's vector-based similarity search may provide\n")
                f.write("an advantage in distinguishing important content from noise.\n\n")
            else:
                f.write("This suggests that Agno's approach to memory storage and retrieval\n")
                f.write("may be more effective at maintaining distinct representations of important facts.\n\n")
        
        # Storage time analysis
        f.write("### Storage Performance\n\n")
        if neuroca_store > agno_store * 1.2:  # 20% slower
            f.write("Neuroca's storage operations are slower than Agno's, which is expected due to\n")
            f.write("the computation of vector embeddings during storage. This represents a tradeoff,\n")
            f.write("as this additional computation potentially enables more sophisticated retrieval.\n\n")
        else:
            f.write("Both systems show comparable storage performance. While Neuroca computes\n")
            f.write("embeddings and Agno doesn't, the performance difference is not significant\n")
            f.write("in this benchmark with a relatively small dataset.\n\n")
        
        # Retrieval time analysis
        f.write("### Retrieval Performance\n\n")
        if abs(retrieve_diff_pct) < 15:  # Within 15%
            f.write("Both systems demonstrate comparable retrieval speeds. This suggests that\n")
            f.write("for the dataset size used in this benchmark, the retrieval algorithms of\n")
            f.write("both systems are adequately optimized.\n\n")
        else:
            faster_system = "Neuroca" if neuroca_retrieve < agno_retrieve else "Agno"
            f.write(f"{faster_system}'s retrieval operations are faster, which could provide\n")
            f.write(f"an advantage in scenarios requiring real-time memory access. ")
            if faster_system == "Neuroca":
                f.write("This demonstrates the\n")
                f.write("benefit of pre-computed vector embeddings for efficient similarity search.\n\n")
            else:
                f.write("This suggests that\n")
                f.write("the simpler retrieval approach may be more efficient for certain workloads.\n\n")
        
        # Detailed question results
        f.write("## Detailed Question Results\n\n")
        
        for i, (n_result, a_result) in enumerate(zip(
            neuroca_results["quiz_results"], 
            agno_results["quiz_results"]
        )):
            question = n_result["question"]
            f.write(f"### Question {i+1}: {question}\n\n")
            
            # Neuroca result
            n_status = "Found" if n_result["fact_found"] else "Not found"
            n_position = n_result["position"]
            n_time = n_result["retrieve_time_ms"]
            
            # Agno result
            a_status = "Found" if a_result["fact_found"] else "Not found"
            a_position = a_result["position"]
            a_time = a_result["retrieve_time_ms"]
            
            f.write("```\n")
            f.write(f"{'System':<10} {'Result':<10} {'Position':<10} {'Time (ms)':<10}\n")
            f.write("-" * 45 + "\n")
            f.write(f"{'Neuroca':<10} {n_status:<10} {str(n_position):<10} {n_time:.2f}\n")
            f.write(f"{'Agno':<10} {a_status:<10} {str(a_position):<10} {a_time:.2f}\n")
            f.write("```\n\n")
        
        # Limitations section
        f.write("## Limitations\n\n")
        f.write("- The benchmark uses simulated memory systems rather than the actual implementations\n")
        f.write("- The test uses a relatively small set of facts and noise messages\n")
        f.write("- The fact matching algorithm is simplistic and may not perfectly represent real usage\n")
        f.write("- No temporal decay is modeled (facts inserted early have the same weight as later ones)\n")
        f.write("- The test doesn't measure the quality of retrieved content, only whether facts were found\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("This benchmark demonstrates the ability of both memory systems to retain and retrieve\n")
        f.write("important information when it's mixed with less important content. In real-world\n")
        f.write("applications, this capability is essential for maintaining context in long conversations\n")
        f.write("and recalling critical details when needed.\n\n")
        
        if neuroca_accuracy > agno_accuracy:
            f.write("Neuroca's higher accuracy suggests its vector-based approach may provide advantages\n")
            f.write("for selective memory retrieval, particularly as the volume of stored memories increases.\n")
        elif agno_accuracy > neuroca_accuracy:
            f.write("Agno's higher accuracy suggests its approach to memory organization may be more\n")
            f.write("effective at maintaining the distinctiveness of important information.\n")
        else:
            f.write("Both systems show similar capabilities in this particular benchmark, suggesting that\n")
            f.write("either could be effective depending on specific application requirements and constraints.\n")
    
    logger.info(f"Generated memory retention report at {report_path}")
    print(f"\nGenerated memory retention report at {report_path}")

if __name__ == "__main__":
    compare_memory_retention()
