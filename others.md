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

