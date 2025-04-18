```mermaid
graph TD
    classDef core fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef integration fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef application fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef stability fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    subgraph "Core Performance Layer"
        Basic["Basic Memory Operations<br/>Benchmark<br/><small>Speed, Efficiency, Latency</small>"]
    end
    
    subgraph "Integration Layer"
        LLM["LLM Integration<br/>Benchmark<br/><small>Context Relevance, Token Usage</small>"]
        Conv["Conversation Context<br/>Benchmark<br/><small>Dialog Memory, Continuity</small>"]
    end
    
    subgraph "Application Layer"
        Task["Task-Oriented<br/>Benchmark<br/><small>QA Accuracy, Retrieval Precision</small>"]
        Retention["Memory Retention<br/>Benchmark<br/><small>Important Fact Recall, Noise Resistance</small>"]
    end
    
    subgraph "Stability Layer"
        Pressure["Memory Pressure &<br/>Maintenance Benchmark<br/><small>Self-Maintenance, Long-term Stability</small>"]
    end
    
    Basic --> LLM
    Basic --> Conv
    LLM --> Task
    Conv --> Task
    LLM --> Retention
    Conv --> Retention
    Task --> Pressure
    Retention --> Pressure
    
    class Basic core
    class LLM,Conv integration
    class Task,Retention application
    class Pressure stability
    
    Result{{"Comprehensive<br/>Evaluation"}}
    
    Pressure --> Result
```

The architecture diagram shows how our benchmarks form a layered approach to evaluating memory systems:

1. **Core Performance Layer**: Basic operations establish a performance baseline
2. **Integration Layer**: Tests how well systems integrate with LLMs and conversation contexts
3. **Application Layer**: Evaluates practical effectiveness in real-world scenarios
4. **Stability Layer**: Tests long-term viability under continuous load

Each benchmark builds on insights from lower layers, creating a complete picture of memory system capabilities from fundamental operations to sustained performance.
