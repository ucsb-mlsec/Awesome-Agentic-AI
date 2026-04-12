# Environment and Inference for Agents

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

## Inference

### LLM and Agent Scheduling

Agent serving should not treat each LLM invocation as an independent request. The right abstraction is the agent session: a long-lived execution that spans multiple LLM calls, tool invocations, pauses, resumptions, and evolving intermediate state. The core issue is no longer just per-call TTFT or throughput, but whether the system can preserve execution continuity and optimize end-to-end completion for the whole session.

- CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving [[FAST’26](https://www.usenix.org/system/files/fast26-liu-yang.pdf)]
  - Background:
    - PDC (Position-Dependent Caching): KV caches are tied to absolute token positions, so a cached segment can only be reused when it appears at the exact same position (e.g., standard prefix caching — only works for shared prefixes)
    - PIC (Position-Independent Caching): KV caches are stored without absolute position encoding, allowing a cached segment to be reused at any position by re-injecting positional information on-the-fly; enables reuse beyond strict prefix matching (few-shot, multi-doc QA, RAG)
    - RoPE (Rotary Position Embedding): injects position via rotation in query/key vectors
    - CoPE (Contextual Position Encoding): measures distance through a context-dependent soft gate — each intermediate token gets a gate value in $[0,1]$ computed from its key vector relative to the query, and the "position" of token $j$ w.r.t. query $i$ is the cumulative sum of gates between them. This produces fractional positions resolved via interpolation over learned position embeddings
      - Characteristics: positions are conditioned on content, so distances can be measured in variable units (tokens, words, sentences) and adapt per-head/per-layer
      - Relevance to KV cache reuse: because CoPE positions are derived from token content rather than absolute index, a cached segment's contextual position shifts more gracefully when it moves in the sequence, reducing the mismatch between cached and real positions during cross-position reuse
  - Identifies a Relative-Position-Dependent Caching (RPDC) pattern in agent prompts: reusable segments maintain consistent relative ordering despite absolute position shifts
  - Motivation: quantifies **PMKD (Positionally Misaligned KV Drift)** — the discrepancy in KV similarity between a cached fixed segment and the same segment under its new absolute position. RoPE has high positional sensitivity (large PMKD under shift), while CoPE has much lower PMKD because positions are indexed by content boundaries
  - Proposed method — CacheSlide with three core components:
    1. **CCPE (Chunked Contextual Position Encoding)**: Define reusable chuncks that appear at the same contextual positions: For each prompt $p$, run CoPE encoding $e = \text{CoPEencode}(p)$ and count the frequency in a histogram $H$. Pick the most frequent encoding $e^* = \arg\max_e H[e]$ and store it as the template for future reuse. Split the new prompt into chunks. If **reuse chunk**: Load from KV Cache or prefill. Its position encoding is taken from the **cached template** $c_i = e^*[i]$ (the pre-learned positional range). If **recompute chunk**: use standard CoPE on the current context: $c_i = pos[\text{chunk}_i]$
    2. **WCA (Weighted Correction Attention)**: Inside a reuse chunk, decide per-token whether the cached K is good enough or needs a refresh
        - Per layer, rank tokens by deviation $d_i = \|K^{\text{new}}_i - K^{\text{cache}}_i\|$ and pick top-k (~5-17%) as the fix list $S_\ell$
        - For $i \in S_\ell$: blend $K_i \leftarrow \alpha K^{\text{new}}_i + (1-\alpha) K^{\text{cache}}_i$; for $i \notin S_\ell$: use cache directly. Apply only every $\tau$ layers
    3. **SLIDE (KV cache manager)**: make WCA's I/O pattern SSD-friendly — its scattered per-token updates would otherwise cause small random writes that kill SSD latency
        - **Relocation**: write WCA's updated tokens to fresh extra pages instead of in-place, turning random writes into sequential ones
        - **Spill policy**: under memory pressure, evict *clean* pages (no corrected tokens) first — their cache is still valid and cheap to reload
        - **Decode overwrite**: reclaim the extra pages by overwriting them with new decode-step KVs, amortizing the scratch-page cost
  - 3.1-4.3x latency reduction and 3.5-5.8x throughput improvement over state-of-the-art baselines

- TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing [[Arxiv’26/04](https://arxiv.org/pdf/2604.03143v1)]
  - Background:
    - vLLM prefix caching: only matches from position zero, so once each agent's private history diverges the shared blocks at different offsets stop hitting
    - Per-request PIC (Position-Independent Caching): reusable at any offset, but RoPE rotation and important-position selection are repeated N times across agents that analyze the *same* shared blocks
    - Storage stays dense and duplicated per agent despite >90% pairwise KV similarity
  - Key problem & insight: multi-agent systems run in synchronized rounds with an All-Gather pattern — every agent's next prompt is `[shared outputs from last round] + [private history]`, so the same blocks reappear across all N agents at shifted positions. The reuse work (rotation, important-token picking) and the storage cost should both be paid *once per round*, not once per agent
  - Proposed method — TokenDance with four components:
    1. **Round-Aware Prompt Interface**: insert reserved separator tokens between logical blocks so the runtime can hash and match shared segments by content rather than absolute position, surviving the offset shifts that break prefix caching
    2. **Collective KV Cache Reuse**: group the N requests of one round and run a single shared RoPE rotation + important-position selection pass over each shared block, instead of N independent PIC passes — amortizing the analysis cost across the whole agent cohort
    3. **Diff-Aware Storage (Master–Mirror layout)**: keep one dense Master KV cache for the cohort and store every other agent as a block-sparse Mirror of differences against Master; since Mirrors are typically 10–20% of full size, memory scales with inter-agent divergence rather than agent count
    4. **Fused Diff Restore**: during the layerwise GPU transfer, apply Mirror corrections on top of Master inside SM memory using ping-pong buffers, so the dense per-agent cache is never materialized before attention
  - Results on GenerativeAgents and AgentSociety: up to 2.3× lower end-to-end latency vs vLLM prefix caching, up to 1.9× prefill speedup vs per-request PIC, 11–17× KV compression from Diff-Aware Storage (up to 17.5× per-agent reduction), and up to 2.7× more concurrent agents under the same SLO

- Helium: Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective [[Arxiv'26/03](https://arxiv.org/abs/2603.16104)]
  - Background:
    - vLLM / per-call inference engines: operate at single-call granularity, blind to cross-call workflow structure
    - Prefix caching (e.g., SGLang RadixAttention): passive and online-only, cannot anticipate reuse in future calls
    - LangGraph / AgentScope orchestrators: treat LLM calls as black-box UDFs, no visibility into KV/prompt internals
    - Parrot: workflow-aware but scheduling heuristics cause load imbalance on shared-prefix patterns
  - Key problem & insight: agentic workflows issue sequences of interdependent LLM calls with massive overlapping prompts and intermediate results, yet serving systems re-encode them call by call. If the workflow is compiled ahead of time into a query plan, the system can proactively pre-compute and pin shared KV state and schedule calls to maximize reuse — LLM invocations become first-class relational operators
  - Proposed method — Helium with six components:
    1. **Agentic Workflows DSL**: Python eDSL that builds a symbolic DAG of primitive operators via lazy dataflow; `compile()` binds placeholder inputs while preserving the symbolic structure for downstream optimization
    2. **Query Optimizer**: two-stage rewriter — initial *plan pruning* does dead-code elimination and common-subgraph elimination, then logical optimization probes the global prompt cache and rewrites hit operators into lightweight `CacheFetch` nodes
    3. **Templated Radix Tree (TRT)**: extends radix trees to model both static token sequences and dynamic operator outputs; leaves are LLM operators and directed edges encode inter-operator dependencies, giving a unified structure for prefix sharing and dataflow
    4. **Cache-Aware Scheduling Algorithm**: cost-based greedy scheduler over the TRT using a critical-path heuristic (prioritize subtrees with greatest dependency depth) and groups operators sharing static prefixes into nested sequences; accounts for token budget and precedence delays
    5. **Proactive Cache Management**: two-level cache — during scheduling, static prefixes are precomputed and their KV is pinned in GPU memory; a global prompt cache maps operator inputs to outputs across workflow invocations
    6. **Query Processor**: dispatches the cache-aware schedule to workers, which execute best-effort while preserving cache-friendly ordering
  - Results: up to $66.27\times$ avg speedup over vLLM and $1.56\times$ over KVFlow on primitive workflows; on the Trading composite workflow, $39.50\times$ vs vLLM, $2.51\times$ vs Parrot, $1.34\times$ vs KVFlow. Baselines: vLLM, OpWise, LangGraph, AgentScope, Parrot, KVFlow

- ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System [[Arxiv’26/02](https://arxiv.org/abs/2602.13692)]
  - Background:
    - vLLM (request-aware inference engine): evicts KV cache during tool execution to admit new requests, causing repeated eviction/reprefill and up to 7.14x latency inflation
    - Continuum (TTL-based KV cache pinning): unpredictable tool runtimes cause severe thrashing and strand KV memory under wrong TTL estimates
    - SGLang prefix-aware router: routes nearly all requests to one node, leaving others idle
  - Key problem & insight: current stacks loosely glue an LLM engine to a tool orchestrator with no end-to-end view of the agentic workflow, so KV cache and tool environments are managed per-request and mis-scheduled. ThunderAgent treats the entire agent trajectory as a first-class Agentic Program so a single scheduler can jointly reason about KV state, tool assets, and placement
  - Proposed method — ThunderAgent with three components:
    1. **Agentic Program abstraction**: unified tuple capturing program ID, context tokens, tool environments, backend placement, execution phase, and scheduling status, giving the runtime whole-workflow visibility across heterogeneous resources
    2. **Program-aware scheduler**: a Global Program-Aware Waiting Queue with a time-decay function $f(t)$, plus Periodic Thrashing Detection and Shortest-First Eviction to preserve KV hit rate and balance memory across nodes
    3. **Tool resource manager**: Asynchronous environment preparation overlaps tool env setup with inference, and Hook-based garbage collection reclaims tool assets tied to program lifecycle events
  - Results: 1.48-3.58x serving throughput over vLLM, 1.17-3.31x over Continuum, 1.79-3.92x RL rollout throughput over vLLM+SGLang Gateway, and up to 4.2x disk memory savings

- DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference [[Arxiv’26/02](https://arxiv.org/abs/2602.21548)]
  - Background:
    - Mooncake (distributed DRAM KV pool): unusable under memory pressure (e.g., RL rollout) and cost-ineffective at large working sets
    - CachedAttention / hierarchical KV stores: reduce retrieval volume but leave storage I/O imbalance between prefill and decode engines untouched
    - HCache / KVPR / TailorKV: GPU-assisted I/O, partial recompute, hybrid quantization — still bound to a single storage→prefill data path
  - Key problem & insight: in disaggregated agentic serving, storage NICs on Prefill Engines saturate while storage NICs on Decode Engines sit idle, making KV loading I/O- rather than compute-bound. KV-Cache loading does not have to be prefill-centric — the idle storage bandwidth on DEs plus the RDMA compute network can form a second loading path
  - Proposed method — DualPath with three components:
    1. **Dual-Path Loading Architecture**: adds a storage-to-decode path alongside the conventional storage-to-prefill path; KV-Cache lands in DE memory first, then streams to PEs over the compute-network RDMA fabric, aggregating both NIC pools
    2. **Traffic Manager (CNIC-centric)**: runs on the compute NIC to orchestrate H2D/D2H copies and cross-engine KV transfers; uses **Traffic Isolation** via virtual lanes (VLs) and **CNIC-Assisted KV-Cache Copy** (RDMA Write) so the auxiliary path does not interfere with normal decode traffic; supports Full Block and Layer Block KV layouts in PE/DE buffers
    3. **Request Scheduler**: global controller doing **Inter-Engine Scheduling** (splits load between PE- and DE-side loading phases), **Intra-Engine Scheduling** (compute-quota-based batch selection), and **KV-Cache Read Task Scheduling** to balance traffic across the two paths under SLOs
  - Results: up to 1.87x offline throughput (DS 660B), 1.78x on DS 27B; online serving ~1.96x average throughput improvement, with 2.25x (DS 660B) higher APS capacity without violating SLOs

- Nalar: An Agent Serving Framework [[Arxiv’26/01](https://arxiv.org/abs/2601.05109)]
  - Background:
    - Ayo: workflow-graph runtime, lacks managed state and dynamic control-flow support
    - CrewAI / AutoGen: agent orchestration libraries tangle spec with execution, no adaptive scheduling
  - Key problem & insight: agentic apps mix heterogeneous components, model-driven control flow, long-running state, and unpredictable latencies, so ad-hoc orchestration cannot exploit parallelism or recover safely. If workflow specification is separated from execution and state is decoupled from placement, a runtime can dynamically schedule, migrate, and retry without developer-visible orchestration
  - Proposed method — Nalar with three components:
    1. **Auto-generated stubs**: compile-time rewrite of agent/tool calls into dependency-aware futures. Stubs capture context and data dependencies, letting the runtime build a dynamic DAG and exploit parallelism without changing Python code
    2. **Managed state layer**: `managedList` / `managedDict` abstractions that decouple logical state from physical placement. Enables safe reuse, migration across workers, and consistent retry semantics under failures
    3. **Two-level control architecture**: a **Global Controller** computing system-wide policy over a logically central workflow view, paired with **Component-Level Controllers** for local event-driven enforcement, backed by a **Node Store** acting as metadata repository and telemetry/decision broker
  - Results: 34-74% P95-P99 tail-latency reduction and up to 2.9x end-to-end speedup over Ayo, CrewAI, and AutoGen across three workloads (Financial Analyst, Router-based, SWE); sustains 80 RPS where baselines fail

- CONCUR: Proactive Agent-Level Admission Control for Efficient Agentic Batch Inference [[Arxiv’26/01](https://arxiv.org/abs/2601.22705)]
  - Background:
    - SGLang (request-level batching + LRU eviction): reactive per-request eviction cannot prevent collapse once many agents are already resident
    - HiCache (CPU offloading): pushes KV traffic off-GPU but does not bound aggregate cache pressure from long-lived agents
  - Key problem & insight: long-horizon ReAct-style agents accumulate context and trigger *middle-phase thrashing*, where KV-cache efficiency collapses well before GPU memory is exhausted. Controlling how many agents are concurrently admitted, rather than evicting individual requests, is the right knob because thrashing is an aggregate property of the active agent set
  - Proposed method — CONCUR with three components:
    1. **Agent-Level Controller**: lightweight middleware sitting between the agent execution layer and the LLM serving engine; exposes *admit*, *pause*, *resume* primitives that gate whole agents (preserving their state) instead of evicting per-request KV blocks
    2. **Cache-Aware Admission Control Algorithm**: AIMD-style feedback loop driven by runtime KV-cache utilization $U_t$ and hit-rate $H_t$; linearly explores concurrency when underutilized, multiplicatively reduces on thrashing signals, and stabilizes near saturation
    3. **Serving-Engine-Agnostic Integration**: implemented as a control layer over existing engines (e.g., SGLang) without modifying kernels or scheduler internals, keeping compatibility with request-level batching and hierarchical caches
  - Results: up to 4.09x throughput on Qwen3-32B and 1.90x on DeepSeek-V3 over SGLang, SGLang+Request Control, and SGLang+HiCache baselines under real agent workloads

- MEPIC: Memory Efficient Position Independent Caching for LLM Serving [[Arxiv’25/12](https://arxiv.org/pdf/2512.16822)]
  - Background:
    - CacheBlend: per-request 15% token recomputation makes chunk KV non-shareable across requests
    - EPIC: 16 fixed tokens recomputed per chunk, still stores position-encoded KV that fragments HBM
  - Key problem & insight: PIC methods store per-request KV variants with unaligned layouts, wasting HBM; shifting recomputation to block granularity and stripping RoPE makes all-but-first blocks byte-identical and page-shareable across requests
  - Proposed method — MEPIC with four components:
    1. **Segmentation and Canonicalization**: partitions each request into immutable chunk segments and request-specific prompt segments with block-aligned padding so chunk KV falls on page boundaries
    2. **Chunk-Aware KV Residency Management**: tracks HBM residency via a Chunk Matcher and does reference-count-based LRU eviction so shared chunk blocks survive across requests
    3. **Selective KV Recomputation (block-level)**: recomputes only the first KV block of each chunk per request; the remaining blocks are reused verbatim, then committed deterministically into paged KV slots
    4. **Fused RoPE Attention**: stores K in position-encoding-free form and applies rotary offsets on-the-fly inside the attention kernel, so one physical copy serves any absolute position
  - Results: up to 2x HBM reduction vs CacheBlend/EPIC at matched latency; 5.74x HBM reduction vs CacheBlend under varying QPS; 11.48% lower latency than CacheBlend and 9.1% lower than EPIC

- Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live [[Arxiv’25/11](https://arxiv.org/abs/2511.02230)]
  - Background:
    - vLLM/SGLang/Dynamo: evict KV cache immediately at turn end, forcing recomputation when tool calls return
    - Autellix/InferCept: priority schedulers unaware of tool-call pause durations, mis-pinning cache under variable tool latency
  - Key problem & insight: agent turns pause on tool calls of unpredictable duration; naive pinning wastes GPU memory while naive eviction wastes compute — retention should be bounded by a per-request TTL derived from empirical tool-latency distributions weighed against reload cost
  - Proposed method — Continuum with five components:
    1. **Tool Call Handler**: parses tool invocations and records per-tool execution latency to build empirical CDFs of pause durations
    2. **Utility Model** (Cost + Benefit Estimation): quantifies KV reload cost (recompute vs CPU offload) against queueing-delay benefit of keeping cache resident
    3. **TTL Value Calculator**: picks per-request TTL from the tool-latency CDF that maximizes the utility model; cache auto-evicts on expiry for robustness to tail tool latencies
    4. **Priority Scheduler**: TTL-aware multi-key ranking with program-level FCFS, keeping turns of the same agent program together to preserve cache locality
    5. **Offline Profiler**: measures prefill cost and CPU-GPU bandwidth to parameterize the cost model
  - Results: 1.12-3.66x delay reduction and 1.10-3.22x throughput gain on multi-turn workloads; up to 8.18x latency/throughput on real SWE-agent traces; ~2x average response-time reduction over vanilla vLLM

- KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows [[Arxiv’25/07](https://arxiv.org/abs/2507.07400)]
  - Background:
    - LRU eviction (e.g., SGLang RadixAttention): workflow-agnostic, discards KV entries shortly before their reuse by the next agent
    - Hierarchical radix cache with CPU offload: cache misses stall execution on synchronous CPU-to-GPU reloads
  - Key problem & insight: multi-agent workflows have known execution structure, so future KV reuse is predictable from the agent schedule rather than recent access history
  - Proposed method — KVFlow with four components:
    1. **Agent Step Graph**: abstracts the workflow as a DAG where each node is an agent invocation, capturing branching and synchronization independent of control flow
    2. **Steps-to-Execution Value**: per-node scalar estimating how soon an agent will run; propagated via step aggregation functions (`max` for sync dependencies, `min` for conditional branches)
    3. **Workflow-Aware Eviction Policy**: fine-grained eviction at the KV cache-tree node level; eviction priority = steps-to-execution, and a shared parent inherits the `min` priority of its children so shared prefixes are retained while only distant-agent branches are dropped
    4. **Overlapped KV Prefetching**: background threads proactively load KV tensors of next-step agents from CPU to GPU; a status-aware scheduler tracks node states (in-GPU, CPU backup, loading, offloading) and reorders ready requests to fully hide transfer latency
  - Results: vs SGLang hierarchical radix cache, up to 1.83x speedup on single workflows with large prompts and up to 2.19x on concurrent workflows

- FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving [[Arxiv'26/02](https://arxiv.org/abs/2602.16603)]
  - Background:
    - Chunked prefill (Sarathi/vLLM): small chunks hurt throughput, large chunks worsen HoL blocking
    - Layer-level preemption (DistServe-style): couples execution granularity to scheduling frequency, adding control overhead even when no preemption is needed
  - Key problem & insight: long prefills monopolize GPUs and violate TTFT SLOs; decoupling preemption granularity from scheduling frequency allows fine-grained interruption while keeping scheduling cheap
  - Proposed method — FlowPrefill with two components:
    1. **Operator-Level Preemption**: interrupt a running prefill at operator (kernel) boundaries inside a layer rather than at chunk/layer boundaries, enabling near-instant yield without shrinking the batched kernel and losing throughput
    2. **Event-Driven Scheduling**: scheduler wakes only on request arrival/completion events, using a **Slack-aware EDF (S-EDF)** policy plus an **SLO-Aware Batching** algorithm to admit/reorder requests by remaining TTFT slack instead of polling every layer
  - Results: 4.7x higher sustained request rate than DistServe at equal SLO, 4.7-5.6x goodput gain, 1.5-3.1x tighter SLO support vs chunked prefill, 3.5-4.2x lower preemption blocking time than layer-level preemption


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

