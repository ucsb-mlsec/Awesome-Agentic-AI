# Others

## Environment simulation

In general, simulating environments with LLMs or other reasoning models may require a large amount of high-quality data with broad state and action coverage. This may be similar as GAN where we train a generator to generate data but the generator itself requires a lot of data to train.

- Agent Learning via Early Experience [[Arxiv'25/10](https://arxiv.org/abs/2510.08558)]
  - Run the given agent to correct trajectories and only mimic the state transitions with the collected trajectories
- Scaling Agent Learning via Experience Synthesis [[Arxiv'25/10](https://arxiv.org/abs/2510.08558)]
  - Collect trajectories from public leaderboard agents (e.g., WebArena leaderboard) and format data as $(s_t, a_t, s_{t+1}, r_{t+1})$
  - Generate explanations for state transitions: why action leads to the next state and reward
  - Train an experience model to output the next state and reward given the current state and action and explanations
  - Explanations act like critics that help improve the accuracy of the experience model; process reward can act as an additional supervision signal
- Simulating Environments with Reasoning Models for Agent Training [[Arxiv'25/11](https://arxiv.org/abs/2511.01824)]
  - Simia-SFT: Use LLM to generate environment feedback from seed trajectories
  - Simia-RL: Generate next state and reward; LLM-generated error messages may be more helpful than real environment
- LLMs as Scalable, General-Purpose Simulators For Evolving Digital Agent Training [[Arxiv'25/10](https://arxiv.org/abs/2510.14969)]
  - Simulator generates next state; guided rollout decides actions with reasoning; wrap infers overall user task
- Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents
  - Unified a broad collection of 13 existing agent training datasets using a standardized protocol
- CLI-Gym: Scalable CLI Task Generation via Agentic Environment Inversion [Arxiv'26/02]
  - Scalable generation of CLI-based tasks via environment inversion for agent training

## Memory management

The latest works on memory management are moving towards building specific sub-agents for memory management, which can better leverage the reasoning and planning capabilities of LLMs to decide what information to store, update, and retrieve.

- Titans + MIRAS: Helping AI have long-term memory [[Google Research Blog](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)]
  - Titans: Learning to Memorize at Test Time [[Arxiv'24](https://arxiv.org/pdf/2501.00663)]
  - It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization [[Arxiv'25](https://arxiv.org/pdf/2504.13173)]
- General Agentic Memory Via Deep Research [[Arxiv'25](https://arxiv.org/abs/2511.18423)]
  - Memorizer: When a new session arrives, it produces a concise memo as a snapshot of the session and creates pages to maintain complete trajectory information
  - Build a researcher agent: Retrieves and integrates useful information from the page-store through a plan-search-reflect loop
    - Plans what to search using available tools
    - Uses embedding model for vector search, BM25 for keyword-based search, and ID-based retriever for direct page exploration
    - Reflects on gathered information and iterates if needed
- Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models [[Arxiv'25](https://arxiv.org/abs/2510.04618)]
  - About curating prompts adaptive to a specific kind of tasks
  - General process: Several adaptive steps (run some tasks with guidelines -> llm generate guidelines) use guidelines to run other similar tasks
  - Previous methods tend to provide brief, generic prompts. After several adaptive steps, context would be summarized.
  - This method instead constructs the guidelines in a structured way called playbook with bullet points. And manage it using an agent, which selects what bullet entries to use, labels how many times a bullet entry is useful or misleading, and performs incremental edition.
- Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory [[Arxiv'25](https://arxiv.org/abs/2504.07952)]
- Scaling Agent Self-Evolving with Reasoning Memory [[Arxiv'25](https://arxiv.org/abs/2509.25140)]
- Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory [[Arxiv'25](https://arxiv.org/abs/2504.19413)]
  - Proposed two memory systems, mem0 and mem0g (graph-based)
  - Mem0:
    - Store and update:
      - Has a async summarization module that summarizes the conversation
      - Whenever there is a new message pair (a user message and a assistant reply), extract info by looking at 1) summarization 2) the previous N1 messages, and then retrieve N2 existing memories from the kb, and let model decide whether to add new memories or update existing ones
    - Retrieve: embedding-based
  - Mem0g:
    - Graph-based, memory are represented as nodes and edges, each node has its type, semantic embedding and meta-data, edges represents the relations between nodes
    - Store and update:
      - Llm-based, for each memory in natural-language, has a module to extract nodes and relations, and then find similar nodes, and choose to update existing nodes, update relations, or create new nodes
    - Retrieve:
      - Two mechanisms: embedding-based and filtered by similarity, Graph-based (find entities in query, find related nodes, then retrieve in and out edges)
- A-MEM: Agentic Memory for LLM Agents [[Arxiv'25](https://arxiv.org/abs/2502.12110)]
  - Llm extract info to store
  - Whenever a memory stored into the kb, ask llm whether it's related to each existing memory, create relations
- STMA: A Spatio-Temporal Memory Agent for Long-Horizon Embodied Task Planning [[Arxiv'25](https://arxiv.org/abs/2502.10177)]
  - Targeted at embodied tasks
  - Has a summarization for actions and observations, structured representation of finished actions, objectives, observations, provided in each query (temporal)
  - Has a graph-based knowledge graph for spatio modeling, can be updated to reflect the spatio changes, has a pre-induction of the relations, e.g., a is west of b, b is west of c -> a is west of c
  - When retrieve, find top n entities, for each entity, find top k neighbors
- HIAGENT: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model [[ACL'25](https://aclanthology.org/2025.acl-long.1575.pdf)]
  - First, let the LLM generate subgoals during task execution
  - For the current subgoal, retain all action-observation pairs
  - Once the subgoal is accomplished, compress its corresponding action-observation pairs into a summary, keeping only the parts relevant to the current subgoal
- Optimus-1: Hybrid multimodal memory empowered agents excel in long-horizon tasks [[NeurIPS'24](https://openreview.net/pdf?id=XXOMCwZ6by)]
  - Evaluated with minecraft
  - Has a Hierarchical Directed Knowledge Graph storing the task dependency and crafting relationship (in minecraft, e.g., Stick + coal -> torch)
  - Has a Abstracted Multimodal Experience Pool, storing failed and successful sub-objectives (like find the river, etc), including the planning, the trajectory (action-observation pairs), summary of the trajectory, env snapshot (e.g., minecraft map states), visual info (including 1. Raw video stream, 2. image buffer (sliding window) → dynamically filter similar frames and retain key frames)
- MemOS: A Memory OS for AI System [[Arxiv'25](https://arxiv.org/pdf/2507.03724)]
  - Offers a unified interface for Plaintext Memory, Activation Memory, and Parameter Memory. Not really related to our objective, which is to optimize Plaintext Memory management
- State and Memory is All You Need for Robust and Reliable AI Agents [[Arxiv'25](https://arxiv.org/abs/2507.00081)]
  - Trivial rag, human designed finite state automaton to maintain states
- RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval [[ICLR'24](https://openreview.net/forum?id=GN921JHCRw)]
  - Augment retrieval for long text documents, other methods only retrieve continuous small chunks
  - Divide document to small chunks by 100 bytes, cluster the chunks, layered summarization of each chunk, for a tree
  - Retrieval strategies: tree traversal (layer-by-layer) and collapsed tree (global comparison), they found collapsed tree is better
- Enhancing Reasoning with Collaboration and Memory [[Arxiv'25](https://arxiv.org/abs/2503.05944)]
  - Evaluation of some early explorations
  - Note that the memory bank in this paper stores exemplars with reasoning chain, like demo of tasks, the insights may not apply to works where the memory stores more fine-grained info within one task
  - Insights:
    - Varied-context agents perform similarly to self-consistency
      - varied-context agents: independently sample exemplars from the memory bank, the most common answer is chosen
      - Self-consistency: temperature-sampled agent with plurality voting
    - Random retrieval memory with its diversity of exemplars yields higher accuracy than similarity-based retrieval
    - Frozen memory performs comparably to incrementally-learned memory, while being more efficient to build
    - Analogical prompting proves more robust to changes in memory and few-shot design choices than standard chain-of-thought
    - Distributing exemplars to varied-context agents is more effective than giving them all to a single agent or multiple identical agents
    - Summarizer agent is most helpful when the reasoning agents are weaker and less so when they are already strong
    - Summarizer reviews the others' responses before determining the final answer
- MemoRAG: Boosting Long Context Processing with Global Memory-Enhanced Retrieval Augmentation [[WWW'25](https://dl.acm.org/doi/abs/10.1145/3696410.3714805?casa_token=fMTI1HaWSJEAAAAA:9Ik8jS9DYHnj0HYH0nRByTTc2Z0YGXkNy5qyCGDWVKKbrRPEPPpKP64mEzIT08ZFruSgsxzgluf3rE0)]
  - A trained-model to replace cosine similarity for embedding matching
- Cognitive Memory in Large Language Models [[Arxiv'25](https://arxiv.org/pdf/2504.02441)]
- MemGPT: Towards LLMs as Operating Systems [[Arxiv'23](https://par.nsf.gov/servlets/purl/10524107)]
- SRMT: Shared Memory for Multi-agent Lifelong Pathfinding [[Arxiv'25](https://arxiv.org/pdf/2501.13200)]
  - Not llm agent, but rl agents
- Zep: A Temporal Knowledge Graph Architecture for Agent Memory [[Arxiv'25](https://arxiv.org/abs/2501.13956)]
- Intrinsic Memory Agents: Heterogeneous Multi-Agent LLM Systems through Structured Contextual Memory [[Arxiv'25](https://arxiv.org/abs/2508.08997)]
- MIRIX: Multi-Agent Memory System for LLM-Based Agents [[Arxiv'25](https://arxiv.org/abs/2507.07957)]
  - Layered memory, hand-designed levels


## Session-aware Scheduling

Agent serving should not treat each LLM invocation as an independent request. The right abstraction is the agent session: a long-lived execution that spans multiple LLM calls, tool invocations, pauses, resumptions, and evolving intermediate state. The core issue is no longer just per-call TTFT or throughput, but whether the system can preserve execution continuity and optimize end-to-end completion for the whole session.

- Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live [[Arxiv'25/11](https://arxiv.org/abs/2511.02230)]
  - Identifies a central inefficiency in multi-turn agent serving: tool calls interrupt execution, but conventional engines often evict KV cache immediately after each turn
  - Introduces a KV cache time-to-live (TTL) mechanism: when a turn is likely to resume after a tool call, the system keeps its KV cache in GPU memory for a bounded time instead of evicting it eagerly
  - Combines tool-aware KV retention with program-level FCFS scheduling, allowing the same agent session to resume with lower reload cost and less queueing delay
  - Main insight: in agent workloads, preserving cross-turn continuity can matter more than optimizing each turn in isolation

- ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System [[Arxiv'26/02](https://arxiv.org/abs/2602.13692)]
  - Argues that existing stacks are fundamentally too request-centric: LLM calls and tool executions are scheduled separately, without a unified view of the agent workflow
  - Proposes LLM Programs as the scheduling unit, exposing heterogeneous resources such as KV cache, runtime state, disk memory, and tool-related assets to a single runtime abstraction
  - Builds a program-aware scheduler and a tool resource manager to improve KV hit rate, reduce memory imbalance, and asynchronously prepare tool environments
  - Main insight: the system should schedule agent programs, not disconnected inference calls

- Helium: Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective [[Arxiv'26/03](https://arxiv.org/abs/2603.16104)]
  - Recasts agent serving as a dataflow and query optimization problem rather than a sequence of unrelated model calls
  - Models an agent workflow as a query plan, with LLM invocations treated as first-class operators
  - Exploits reuse across overlapping prompts, intermediate outputs, KV states, and speculative branches through proactive caching and cache-aware scheduling
  - Main insight: the optimization target should move from call-level efficiency to workflow-level reuse

- Nalar: An agent serving framework [[Arxiv'26/01](https://arxiv.org/abs/2601.05109)]
  - Frames agent serving as a general workflow runtime for executions with dynamic control flow, heterogeneous components, long-lived state, and highly variable latency
  - Separates workflow specification from execution while preserving Python-style usability through generated stubs that convert agent and tool invocations into dependency-aware futures
  - Introduces a managed state layer to decouple logical state from physical placement, enabling reuse, migration, and retries
  - Uses a two-level control architecture with global policy computation and local event-driven enforcement
  - Main insight: agent serving is a runtime systems problem, not just an inference scheduling problem

- KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows [[Arxiv'25/07](https://arxiv.org/abs/2507.07400)]
  - Focuses on a key weakness of existing prefix-caching systems under agent workflows: standard LRU-style eviction is unaware of workflow structure, so it often evicts KV entries shortly before they are needed again
  - Introduces an Agent Step Graph abstraction to model execution dependencies across agents and estimate how soon each agent will be activated next
  - Uses this workflow signal to guide fine-grained KV eviction at the cache-node level, preserving entries that are more likely to be reused and handling shared prefixes in tree-structured caches more effectively
  - Adds fully overlapped KV prefetching, proactively loading tensors from CPU to GPU for agents likely to run in the next step, reducing cache-miss stalls during execution
  - Main insight: for agent workloads, KV-cache management should be driven by future workflow structure rather than past access recency

- CONCUR: Proactive Agent-Level Admission Control for Efficient Agentic Batch Inference [[Arxiv'26/01](https://arxiv.org/abs/2601.22705)]
  - Identifies a specific pathology in agentic batch inference called middle-phase thrashing: as long-lived agents accumulate context, KV-cache efficiency can collapse well before GPU memory is fully exhausted, leading to repeated eviction and recomputation
  - Argues that the right control knob is not per-request cache eviction, but agent-level admission control: regulate how many agents are concurrently active so that aggregate KV pressure stays below the point where throughput collapses
  - Adapts a congestion-control style loop, inspired by AIMD, to use runtime cache signals and dynamically modulate the number of admitted agents, while remaining a lightweight middleware layer compatible with existing agent frameworks and serving engines
  - Reports up to 4.09× throughput improvement on Qwen3-32B and 1.90× on DeepSeek-V3 under real agent workloads
  - Main insight: for long-horizon agent batches, the bottleneck is often not raw memory capacity but cache-pressure-induced concurrency collapse, so admission control can be more important than smarter eviction alone

- DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference [[Arxiv'26/02](https://arxiv.org/abs/2602.21548)]
  - Starts from the observation that multi-turn agentic inference is often I/O-bound rather than compute-bound: cache hit rates are very high, so the dominant cost becomes loading large KV states from storage rather than recomputing them
  - Identifies a structural imbalance in disaggregated serving: the storage NICs on prefill engines saturate, while NIC bandwidth on decode engines remains underused
  - Proposes dual-path KV loading: besides the standard storage→prefill path, it adds a storage→decode→prefill path, where decode engines load KV from storage and forward it to prefill engines via RDMA over the compute network
  - Adds a global scheduler that distributes traffic across the two paths and balances load across prefill and decode engines, effectively turning storage bandwidth into a pooled, schedulable resource
  - Reports up to 1.87× higher offline end-to-end throughput and about 1.96× average online serving throughput improvement without violating SLOs
  - Main insight: when agent workloads already have high KV reuse, the central systems problem shifts from compute scheduling to KV movement and storage-bandwidth orchestration

