# Agentic model and techniques

Below, we summarize the latest agentic models, as well as some notable and recent reasoning techniques.  

## Table of Contents
- [Agentic model and techniques](#agentic-model-and-techniques)
  - [Table of Contents](#table-of-contents)
  - [Newest models](#newest-models)
  - [Agentic RL](#agentic-rl)
    - [Agent training framework](#agent-training-framework)
    - [Compute step-wise rewards](#compute-step-wise-rewards)
    - [Different applications](#different-applications)
    - [Misc](#misc)
      - [Environment simulation](#environment-simulation)
      - [Efficiency, Stability and others](#efficiency-stability-and-others)
      - [Control Experiments](#control-experiments)
  - [Memory management](#memory-management)
  - [Agentic modeling (linear attentions)](#agentic-modeling-linear-attentions)

## Newest models

-  ***GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*** [[Arxiv'25/8](https://arxiv.org/pdf/2508.06471)]
    - Model: MoE with 335B and 32B active parameters
    - Mid-training: repo-level code training, synthetic reasoning data, Long-context & agent training
    - Post-training:
      - SFT: 
        - Expert SFT: Empower the model with basic chat, reasoning, and tool-use capabilities; enable hybrid reasoning (short & long) 
        - Unified SFT: Distill the capabilities of different expert models into one hybrid reasoning model
        - Rejected sampling: answer correctness, prevent hallucination, toll-calling (invoke proper protocols, reach expected terminal states)
        - Agentic SFT data collection: 
          - Agentic Framework and Tool Collection: MCP (standard format)
          - Task Synthesis: Single step and multi-step
          - Trajectory Generation: teacher model
          - Quality filtering: correctness 
      - RL (GRPO without KL term)
        - Reasoning RL: techniques to improve training efficiency, sample diversity, data quality
      - Agentic RL 
        - Web and coding agent (Github issue resolving with sandbox execution)
        - Only optimize model generated tokens
        - ORM with process format penalty (stop traj with wrong format)
        - Self distillation and encourage interaction
      - General RL 
        - Holistic RL: Rule-based, human, and model-based feedback on generate tasks
        - Instruction following 
        - Function calling: step-wise (part of general RL) and multi-turn RL (distill from specialized models)
        - Pathology RL (Final step)
  - RL infra 

- DeepSeek-V3.1 [[2025/8](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)]
  - Hybrid thinking with different templates (Mixed thinking with non-thinking when answering) 
  - Higher thinking efficiency and smarter tool calling **[only support in non-thinking mode]**

- ***Kimi K2: Open Agentic Intelligence*** [[Arxiv'25/7](https://arxiv.org/pdf/2507.20534)]
  - MoE model with 1 trillion total params and 32B active params
  - Pre-training: 
    - 15.5 trillion tokens + data efficiency techniques
    - MuonClip optimizer: Muon + weight decay + QK clip 
    - Post-training:
      -  SFT
         -  **Agentic data synthesis**: Tool spec generation, agent and task generation, trajectory generation; simulated env with real execution env
      - RL 
        - Verifiable rewards gym; Self-critique rubric reward 
        - GRPO with KL diff + PTX loss for preventing forget critical data
        - RL infra: efficient engine switching; system startup; agentic rollout

- Qwen3 Technical Report [[Arxiv'25/5](https://arxiv.org/abs/2505.09388)]
  - Model: Both dense and MoE model, flagship model -- Qwen3-235B-A22B
    - Grouped Query Attention, SwiGLU, Rotary Positional Embeddings, and RMSNorm with pre-normalization; Remove QKV-bias and introduce QK-Norm
  - Pre-training: 36 trillion tokens (length 32,768 tokens), with synthetic data generated from other Qwen models
    - General stage, reasoning stage, long context stage
    - Scaling low for optimal hyper-parameters 
  - Post-training: 
    - Long CoT cold-start SFT
      - Data: math, code, logical reasoning, and general STEM problems with ground truth
      - Filter: Query filtering (filter out non-verifiable queries) and response filtering (filter out simple questions)
      - Use QwQ for reasoning data generation and filter out: incorrect answers; repetition; hallucinations; Inconsistencies between the thinking and summary contents; inappropriate language mixing or stylistic shifts; being overly similar to potential validation set items
    - Reasoning RL with GRPO
      - Data: Challenging but learnable data pairs (3,995)
      - Use a large batch size and a high number of rollouts; off-policy training (reuse logged data for training); Control model entropy
    - Hybrid thinking with SFT: 
      - Combine data with and without reasoning paths into a unified dataset to combine thinking with non-thinking mode 
      - Provide final answers with partial thinking with a manually stop thinking token  
    - General RL: Instruction following; format following; preference alignment; **agent ability (tasks with tool invoke)**; ability to understand specific context
      - Outcome rewards: Rule-based reward; model-based reward with/without reference answers
  - Strong-to-weak distillation for small models, outperforms RL
    - Off-policy Distillation: Combine the outputs of teacher models generated with both /think and /no think modes for response distillation
    - On-policy Distillation: Student model generates on-policy sequences, query teacher models to get the targets, and is then fine-tuned by aligning its logits with those of a teacher model to minimize the KL divergence


## Agentic RL

### Agent training framework
- Compare of different frameworks [[anatomy-of-rl-frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks/)]
- Slime [[Github](https://github.com/THUDM/slime)]
  - Asynchronous rollout: Training and inference are fully separated using sglang
  - Customized rollout: Supports task-specific handling such as filtering incomplete/invalid trajectories
  - **Support partial rollout for efficiency:**
    - Issuer: response lengths vary greatly between samples; traditional approach waits for the longest rollout, causing GPU idle time
    - Solution
      - Unfinished partial results are stored in a buffer and continued in the next round
      - Completed rollouts are immediately used for loss computation and optimization
      - Rollouts can interact across batches
  - Importance sampling for data reuse: Save logits at generation time to correct for policy drift ($\mathbb{E}_{a \sim \pi_{\text{old}}}[f(a)] = \mathbb{E}_{a \sim \pi_{\text{old}}}[\frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)} \cdot f(a)]$)
  - Note: Inference and training may have different precision, which can cause problems

### Compute step-wise rewards
- SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution [[Arxiv'25/05](https://arxiv.org/abs/2505.20732)]
  - Train an LLM as a progress estimator to assign a contribution score for each step
  - The sum of the contribution scores is the final reward (either 0 or 1) 
  - Model assigns a reward to each step, training objective is to make the sum of the contribution scores close to the final reward (MSE loss) [No constraints on individual step scores]

- Group-in-Group Policy Optimization for LLM Agent Training [[Arxiv'25/05](https://arxiv.org/abs/2505.10978)]
  - GRPO with outcome reward  
  - Step-wise reward: Group similar states and use GRPO to compute the states in the group

- Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/abs/2509.20616)]
  - Train LLM agents for multi-turn task planning by decomposing long-horizon tasks into single-turn reasoning problems, where the policy (Qwen2.5-1.5B) learns to predict the correct next action at each state using GRPO with dense rewards from expert trajectories (Llama3-70B) 
  - Theory proves that improving single-step optimality leads to higher overall multi-turn success rate

- Online Process Reward Leanring for Agentic Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/abs/2509.19199v2)]
  - Similar to PRIME, the differences are 1. DPO vs CE loss for training PRM; 2. GRPO vs RLOO for calculating the reward

- **RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents [[Arxiv'25/07](https://arxiv.org/abs/2507.22844)]**
  - Create different rules to calculate the reward for each step

- Process reward models for llm agents: Practical framework and directions [[Arxiv'25/02](https://arxiv.org/abs/2502.10325)]
  - Two frameworks:
    - SFT via rollouts (similar to Group-in-Group, group the same states)
    - Inverse RL (collecting expert trajectories to train PRM)

- **From novice to expert: Llm agent policy optimization via step-wise reinforcement learning [[Arxiv'24/11](https://arxiv.org/abs/2411.03817)]**
  - Collect expert trajectories, fix n steps of the trajectory, and train the model to generate the next step using RL

- Reinforcement Learning for Long-Horizon Interactive LLM Agents [[Arxiv'25/02](https://arxiv.org/abs/2502.01600)]
  - PPO with RLOO to estimate the reward, examined three different variants: token-level, step-level, and trajectory-level. Token-level is the most effective.

- RAGEN: Understanding self-evolution in LLM agents via multi-turn reinforcement learning [[Arxiv'25/04](https://arxiv.org/abs/2504.20073)]
  - Extend PPO and GRPO to multi-turn reasoning

### Different applications

- UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning [[Arxiv'2/9](https://arxiv.org/pdf/2509.02544v1)]
  
- ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents [[Arxiv'25/08](https://arxiv.org/abs/2508.14040)]
  - Implement API calls (via LLM) for agents to interact with the environment
  - Step reward is the same as the final reward
  - Incorporate SFT in the later phase of RL training to mitigate entropy collapse

- MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment [[Arxiv'25/07](https://arxiv.org/abs/2507.05720)]
  - Task synthesis and filtering out low-quality tasks
  - Step reward is the same as the final reward
  - Define a set of rules to scale the reward of each trajectory

- Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning [[ICLR 25](https://arxiv.org/abs/2411.02337)]
  - web navigation tasks
  - long-horizon interaction yet provide sparse and delayed rewards. Making policy improvement challenging and costly, and often result in training collapse

### Misc

#### Environment simulation 

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

#### Efficiency, Stability and others

- Stabilizing Reinforcement Learning with LLMs: Formulation and Practices [[Qwen Team, Arxiv'25/12](https://arxiv.org/abs/2512.01374)]
  - Three issues: (1) training–inference discrepancy (FP8 vs. BF16); (2) policy staleness (due to asynchronous rollout); (3) training–inference discrepancy can cause inconsistent routed experts for MoE models
  - Existing solution:
    - (1) & (2): importance sampling
    - (3): routing replay (fix the expert used by inference when do training)
  - Given the assumption: the IS for each token is small: $1+\epsilon \ll 1$, the token-level optimization objective can be viewed as the first-order approximation of the sequence-level optimization objective.
  - Proposed MiniRL: GRPO + Clip (mask the gradient for tokens which are higher than a threshold when reward > 0 and lower than a threshold when reward < 0)
  - Evalute on Qwen3-30B-A3B with synchronous training:
    - For on-policy training (the global batch size equals the mini-batch size):
      - MiniRL is the best, adding length normalization leads to suboptimal performance
    - For off-policy training:
      - Routing Replay and clipping become essential for stable training

- Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models [[Arxiv'25/12](https://arxiv.org/abs/2512.13607)]
  - Cascade RL process begins with applying general-domain Reinforcement Learning from Human Feedback (RLHF) to the SFT models, followed by domain-wise Reinforcement Learning with Verifiable Rewards (RLVR).

- SimpleTIR: End-to-End Reinforcement Learning  for Multi-Turn Tool-Integrated Reasoning [[Arxiv'25/09](https://arxiv.org/abs/2509.02479)]
  - problem: LLM generated response after multiple turns (with tool observation) is OOD compared to the pre-training and SFT data, each token has low probability, which leads to extremely high importance sampling ratio ($\frac{\pi_{\text{new}}(y)}{\pi_{\text{old}}(y)}$) for trajs with negative reward.
  - solution: clipping the importance ratio is appealing but the threshold is hard to set. The authors filter out the trajs with void turns (no tool invocation or final answer).
  - scenario: Zero RL with math problems

- GenEnv: Difficulty-Aligned Co-Evolution Between  LLM Agents and Environment Simulators [[Arxiv'25/12](https://arxiv.org/abs/2512.19682)]
  - Env LLM generates tasks, evaluation metrics, and potentially ground truth, the goal is to generate tasks that are challenging for the agent (avg acc = 0.5)
  - Agent LLM rollouts to generate trajectories, and then use RL (GRPO)to train the agent to improve the performance
  - Env LLM use Reward-Weighted Regression (weighted SFT loss) to update its parameters
  - Env LLM is like a task generator, it does not generate new env or simulate the env.

- AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/pdf/2509.08755)]
  - Provide env and training framework
  - Propose ScalingInter-RL: progressively add interaction rounds - their experiments show that beginning with a large number of interaction turns often leads the model into redundant reasoning and unproductive actions.

- RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training [[Arxiv'25/09](https://arxiv.org/abs/2509.21009)]
  - A new framework for efficient RL training.

- Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents
  - Expected gradient norm is monotonically coupled with policy entropy
  - A_mod(i, t) = A^{i} * g(H_{t}) + f(H_{t+1})
    - g(H_{t}): For a confident step, g(H_{t}) > 1, which amplify its gradient; Conversely, for an uncertain step, g(H_{t}) < 1, which attenuates its gradient.
    - f(H_{t+1}): encourages the agent to select actions that lead to a more predictable and less ambiguous future state
  - Task: Webshop and ALFWorld (i.e., agent benchmark with sparse reward)
  - LLM model: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct

- When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch [[Notion'25/10](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)]
  - Mismatch between training and inference: cause collapse of RL training: $\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \textcolor{red}{\pi _{\theta}^{\mathrm{vllm}}}\left( \cdot |x \right)}\left[ R\left( x,y \right) \nabla _{\theta}\log \textcolor{blue}{\pi _{\theta}^{\mathrm{fsdp}}}\left( y|x \right) \right]$
  - Long context amplifies this issue as the difference is accumulated
  - The **agent** receives a tool response, which is often structured text (e.g., context enveloped by `<python_output>` and `</python_output>` tags) that is OOD compared to its pre-training and SFT data.Faced with this unfamiliar OOD context, the **agent's policy** becomes more uncertain, making it more likely to sample low-probability tokens in its subsequent turns (which is also observed in [**SimpleTIR**](https://github.com/ltzheng/SimpleTIR)).
  - Hardware differences also contribute to the instability
  - Solution: sequence-level importance sampling ($\frac{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(y|x)}{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}(y|x)}$)

- Effective Harnesses for Long-Running Agents [[Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)]
  - Problems: Agent tries to one-shot complex tasks, runs out of context mid-implementation; later agent sees progress and declares done prematurely
  - Solution: Initializer agent sets up environment and progress file; coding agent makes incremental progress with structured updates
  - Open questions: Single general-purpose agent vs multi-agent architecture; generalization to other domains (scientific research, financial modeling)

#### Control Experiments
- RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs? [[Arxiv'25/09](https://arxiv.org/abs/2509.21016)]
  - Using synthetic coding promblems to verify how RL can unlock new algorithms in LLMs

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
  - Previous methods tend to provide brief, generic prompts. After several adaptive step, context would be summarized.
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
  - Targeted of embodied tasks
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
  - Augment retrieval for long text document, other methods only retrieve continuous small chunks
  - Divide document to small trunks by 100 bytes, cluster the chunks, layered summarization of each chunk, for a tree
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

## Agentic modeling (linear attentions)

- Kimi Linear [[Arxiv'25/10](https://arxiv.org/abs/2510.26692)]
  - Propose Kimi Delta Attention 
  - Kimi Linear can be a drop-in replacement for full attention architectures with superior performance and efficiency, including tasks with longer input and output lengths.

