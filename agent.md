# Agentic models and techniques

## Table of Contents
- [Agentic models and techniques](#agentic-models-and-techniques)
  - [QWEN models](#qwen-models)
  - [DeepSeek](#deepseek)
  - [GLM & Kimi](#glm--kimi)
  - [NVIDIA](#nvidia-nemotron)
  - [Agentic RL](#agentic-rl)
    - [Agent training framework](#agent-training-framework)
    - [Overall recipes](#overall-recipes)
    - [Process rewards](#process-rewards)
    - [Objective functions](#objective-functions-mainly-about-weighting-different-rollouts)
    - [Sampling strategies](#sampling-strategies)
    - [Stability and others](#stability-and-others)
  - [Agentic modeling](#agentic-modeling)

## QWEN models

- Qwen3.6-Plus: Towards Real World Agents [[26/04](https://qwen.ai/blog?id=qwen3.6)]
  - Next-generation hybrid architecture with always-on chain-of-thought reasoning (no thinking mode toggle), 1M context window (up from 262K in 3.5), and up to 65K output tokens
  - Improved agentic reliability in multi-step workflows with native function calling; reduced overthinking on simple tasks
  - ~2-3x inference speed over Claude Opus 4.6 (community-reported) via efficiency-focused architecture with lower inference energy consumption

- Qwen3.5 Towards Native Multimodal Agents [[26/02](https://qwen.ai/blog?id=qwen3.5)]
  - Qwen3.5-397B-A17B, a vision-language model with 1M context window
  - Scaling of virtual RL tasks and environments compared to Qwen3 series
    - More than 15K environments
  - Pre-training: More tokens, more efficient MoEs, early text-vision token fusion
  - Infrastructure: 
    - MM training via a heterogeneous infrastructure that decouples parallelism strategies across vision and language components (exploiting sparse activations for cross-component computation overlap)
    - A native FP8 pipeline applies low precision to activations, MoE routing, and GEMM operations—with runtime monitoring preserving BF16 in sensitive layers
    - asynchronous RL framework: improve hardware utilization, dynamic load balancing, and fine-grained fault recovery
      - FP8 end-to-end training, rollout router replay, speculative decoding, and multi-turn rollout locking

- Qwen3 Technical Report [[Arxiv'25/5](https://arxiv.org/abs/2505.09388)]
  - Model: Both dense and MoE model, flagship model -- Qwen3-235B-A22B
    - Grouped Query Attention, SwiGLU, Rotary Positional Embeddings, and RMSNorm with pre-normalization; Remove QKV-bias and introduce QK-Norm
  - Pre-training: 36 trillion tokens (length 32,768 tokens), with synthetic data generated from other Qwen models
    - General stage, reasoning stage, long context stage
    - Scaling law for optimal hyper-parameters 
  - Post-training: 
    - Long CoT cold-start SFT
      - Data: math, code, logical reasoning, and general STEM problems
      - **Filter**: Query filtering (filter out non-verifiable queries) and response filtering (filter out simple questions) 
        - incorrect answers; repetition; hallucinations; Inconsistencies between the thinking and summary contents; inappropriate language mixing or stylistic shifts; being overly similar to potential validation set items
    - Reasoning RL with GRPO
      - Data: Challenging but learnable data pairs (3,995)
      - Use a large batch size and a high number of rollouts; off-policy training (reuse logged data for training); Control model entropy
    - Hybrid thinking with SFT 
      - Combine data with and without reasoning paths into a unified dataset to combine thinking with non-thinking mode 
      - Provide final answers with partial thinking with a manually stop thinking token  
    - General RL: Instruction following; format following; preference alignment; agent ability (tasks with tool invoke); ability to understand specific context
      - Outcome rewards: Rule-based reward; model-based reward with/without reference answers
  - **Strong-to-weak distillation** for small models, outperforms RL
    - Off-policy Distillation: Combine the outputs of teacher models generated with both /think and /no think modes for response distillation
    - On-policy Distillation: Student model generates on-policy sequences, query teacher models to get the targets, and is then fine-tuned by aligning its logits with those of a teacher model to minimize the KL divergence
  
## DeepSeek 
  - DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models [[2025/12](https://arxiv.org/pdf/2512.02556)]
    - DeepSeek Sparse Attention: lightning indexer and fine-grained token selection
      - lightning indexer to compute the weights for preceding tokens and only the selected tokens will be used for computing attentions.
      - MQA (expand the dim of queries) with sparse attention (adding indexing)
    - Scalable RL training framework
      - [K3 estimator](http://joschu.net/blog/kl-approx.html) with importance score for unbiased KL estimation
      - Merge different tasks requiring RL into one stage
      - Off-policy RL with a threshold for controlling whether the rollouts are useful based on policy divergence
    - Synthesize agentic tasks 

  - DeepSeek-V3.1 [[2025/8](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)]
    - Hybrid thinking with different templates (Mixed thinking with non-thinking when answering) 
    - Higher thinking efficiency and smarter tool calling **[only support in non-thinking mode]**

## GLM & Kimi
- GLM-5 [[26/02](https://arxiv.org/pdf/2602.15763)]
  - Performs a little bit worse than Kimi-2.5 on coding tasks
  - Scalability for model and pre-training: Increased from 355B (32B activated) to 744B (40B activated) with pre-training data upgraded from 23T to 28.5T
    - Vanilla MLA cannot match GQA; propose Muon split that splits K/Q/V matrices into smaller matrices for different heads and applies matrix orthogonalization to these independent matrices
    - Use multi-token prediction and DSA
  - SFT: three different thinking modes: interleaved thinking, preserved thinking, turn-level thinking 
  - Asynchronous RL: SLIME-based RL with async rollout and off-policy learning
    - Remove off-policy trajs from the rollout if the distribution discrepancy is large
    - On-policy distillation
      - Sequentially optimizing for distinct objectives can lead to the cumulative degradation of previously acquired capabilities
      - Use the log policy diff as the per-token reward

-  GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models [[Arxiv'25/8](https://arxiv.org/pdf/2508.06471)]
    - Model: MoE with 335B and 32B active parameters
    - Mid-training: repo-level code training, synthetic reasoning data, Long-context & agent training
    - Post-training:
      - SFT: 
        - Expert SFT: Empower the model with basic chat, reasoning, and tool-use capabilities; enable hybrid reasoning (short & long) 
        - Unified SFT: Distill the capabilities of different expert models into one hybrid reasoning model
        - Rejection sampling: answer correctness, prevent hallucination, tool-calling (invoke proper protocols, reach expected terminal states)
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
        - Holistic RL: Rule-based, human, and model-based feedback on generated tasks
        - Instruction following 
        - Function calling: step-wise (part of general RL) and multi-turn RL (distill from specialized models)
        - Pathology RL (Final step)

- Kimi 2.5: Visual Agentic intelligence [[26/2]](https://arxiv.org/pdf/2602.02276)
  - Similar performance as QWEN-3.5
  - Model is common; token-efficient MuonClip optimizer with QK-clip
  - Post training: 
    - RL loss: strict clip without timing advantage, use k2 estimator
    - Rule-based reward, budget-control reward, intermediate reward with a critic model
      - helpfulness, response readiness, contextual relevance, appropriate level of detail, aesthetic quality of generated artifacts, and strict instruction following
    - Token efficient RL: when task acc is high, encourage the model to generate more concise answers
  - Agentic training:
    - Infra: Rollout manager with async and partial rollout
    - Agent swarm: dynamically creates specialized frozen subagents and decomposes complex tasks into parallelizable subtasks 
    - a trainable orchestrator and frozen subagents; make training more stable and avoid credit assignment ambiguity
    - Parallel agent RL: performance reward, parallel reward (avoid single agent), sub-agent finish reward (avoid too many sub-agents)
    - Critical steps as resource constraints: constrain training based critical steps (main agent steps + max step of sub-agents) -> incentive parallel strategies that minimize the total steps

- Kimi K2: Open Agentic Intelligence [[Arxiv'25/7](https://arxiv.org/pdf/2507.20534)]
  - MoE model with 1 trillion total params and 32B active params
  - Pre-training: 
    - 15.5 trillion tokens + data efficiency techniques
    - MuonClip optimizer: Muon + weight decay + QK clip 
    - Post-training:
      -  SFT
         -  Agentic data synthesis: Tool spec generation, agent and task generation, trajectory generation; simulated env with real execution env
      - RL 
        - Verifiable rewards gym; Self-critique rubric reward 
        - GRPO with KL diff + PTX loss for preventing forgetting critical data
        - RL infra: efficient engine switching; system startup; agentic rollout

## NVIDIA-Nemotron

- Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation [[Arxiv'26/03](https://arxiv.org/abs/2603.19220)]
  - Post training pipeline built on top of Nemotron-3 (30B MoE model with 3B activated parameters)
  - SFT: 
    - Math, Code, Science, Agent (conversational tool use, SWE and terminal agents)
    - Use DeepSeek (Math) and GPT-OSS (Code) as the teacher model;
  - Cascade RL
    - Determine the order: Mitigating Inter-Domain Interference (IF and RLHF are conflict); Scaling via Multi-Domain Integration (integrate domains are not conflict with the overall performance); Stabilization through On-policy Distillation
    - RL sampling and objective:
      - strict on-policy learning, with a REINFORCE objective (No trusted region):
      - $\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\{o_i\}_{i=1}^G\sim\pi_\theta(\cdot|q)}\left[\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}\hat{A}_{i,t}\right]$, where $\hat{A}_{i,t}=\frac{r_i-\text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)}$ for all $t$
    - Instruction-following RL: Filter out all correct/wrong samples; constrain the output length
    - Multi-domain RL & on-policy distillation: Large batch size & Distill from the strongest intermediate teacher models
      - $\mathcal{L}_{\text{MOPD}} = -\mathbb{E}_{x\sim\mathcal{D}, y\sim\pi^{\text{inf}}(\cdot|x)} \left[\frac{1}{|\mathcal{V}(y)|} \sum_{t\in\mathcal{V}(y)} w_t \, \text{sg}[a_t^{\text{MOPD}}] \log \pi^{\text{train}}(y_t|s_t)\right]$
      - $w_t$: truncated importance weight correcting for inference-training policy mismatch; $\text{sg}[\cdot]$: stop-gradient; $a_t^{\text{MOPD}}$: token-level advantage from domain-specific teacher models
    - RLHF
    - Long-context RL: Use LLM as a judge for reward
    - Code and SWE RL: Long output token, binary reward, agentic RL as a much larger batch size and context length (use SWE-Gym + OpenHands); Do filtering 

- NVIDIA Nemotron 3: Efficient and Open Intelligence [[Arxiv'25/12](https://arxiv.org/abs/2512.20856)]
  - Family of Nano, Super, and Ultra models using hybrid Mamba-Transformer MoE architecture with up to 1M context length
  - Model: Hybrid Mamba-Transformer MoE: interleave MoE with Mamba-2 layers, which donot need linearly increasing KV Cache; LatentMoE (add a down proj in the input and up proj in the output)
  - Multi-token prediction: predict multiple future tokens simultaneously during training, providing richer training signals and encouraging the model to plan ahead; ~2.4% average benchmark improvement; enables built-in speculative decoding
  - NVFP4 training: 4-bit floating-point training (pre-training with FP4 rather than BF16)


## Agentic RL

### Agent training framework
- Comparison of different frameworks [[anatomy-of-rl-frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks/)]
  - Rollout Architecture: Engine (Rollout happens in process) vs Server (Rollout runs as a separate service)
  - Weight Synchronization Strategy & worker organization
  - Parallelism model & Data flow pattern

- AREAL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning [[25/11](https://arxiv.org/pdf/2505.24298)]
  - Fully async with staleness-aware RL, better handle off-policy
  - SLIME is better at modularization and supports partial rollout

- Slime [[Github](https://github.com/THUDM/slime)]
  - Asynchronous rollout: Training and inference are fully separated using sglang
  - Customized rollout: Supports task-specific handling such as filtering incomplete/invalid trajectories
  - **Support partial rollout for efficiency:**
    - Issue: response lengths vary greatly between samples; traditional approach waits for the longest rollout, causing GPU idle time
    - Solution
      - Unfinished partial results are stored in a buffer and continued in the next round
      - Completed rollouts are immediately used for loss computation and optimization
      - Rollouts can interact across batches
  - Importance sampling for data reuse: Save logits at generation time to correct for policy drift ($\mathbb{E}_{a \sim \pi_{\text{old}}}[f(a)] = \mathbb{E}_{a \sim \pi_{\text{old}}}[\frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)} \cdot f(a)]$)
  - Note: Inference and training may have different precision, which can cause problems

### Overall recipes

- OpenClaw-RL: Train Any Agent Simply by Talking [[Arxiv'26/03](https://arxiv.org/abs/2603.10165)]
  - Unified RL framework built on SLIME: policy serving, rollout collection, PRM judging, and policy training are fully async
  - Leverage all evn. changes for training (next state signals): User replies, tool outputs, terminal responses, GUI state changes, error message
    - Evaluation signals are fed to PRM
    -  Directive signal: Hindsight-Guided On-Policy Distillation (OPD) extracts textual hints from the next state, constructs an enhanced teacher context, and distills token-level directional supervision back into the student, 

- Agentic Critical Training [[Arxiv'26/03](https://arxiv.org/abs/2603.08706)]
  - Collect expert data as reference and train the model to select better actions between reference and the model's own actions (enable the model to do critique)
  - Continue training the model with normal RL 

- Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning [[Arxiv'26/02](https://arxiv.org/abs/2602.01058)]
  - Claims to address distribution mismatch between SFT samples (generated by teacher models) and RL samples (generated by the policy)
  - Token-level importance sampling, assigning large weights to SFT samples whose distribution more closely matches the current model
  - Train the model to not change too much from the current policy (based on the importance weights)

- Experiential Reinforcement Learning [[Arxiv'26/02](https://www.arxiv.org/abs/2602.13949)]
  - Teach the model to do self-critique (similar to self-distillation but different methods)
  - Ask the policy to do two attempts in one response round, where the second response leverages the model's reflection of the first one   
  - Use RL to update the policy based on both attempts 
  - Use the second response for distillation (ask the model to output the second response without the reflection) 

- RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System [[Arxiv'26/02](https://www.arxiv.org/pdf/2602.02488)]
  - Train a parametric and discrete intermediate reward model
  - Intermediate reward: Reward model rates the quality of each step m times, the final reward is the average of the m scores plus the final reward. The step-wise advantage is calculated by standardizing rewards across trajectories at the same step index.
  - The reward model is jointly optimized with the policy using a consistency feedback signal: $R_{S_{\tau_{i},j}} = R_{\tau_{i}} \cdot S_{\tau_{i,j}}$. 
    - USE RL loss to train or not? it needs to compute advantage...
  - The framework dynamically adjusts task difficulty when policy accuracy falls outside predefined thresholds ($\alpha_{low}=0.2, \alpha_{high}=0.8$). A language model modifies tasks to be harder or easier based on "critic feedback"—summarized error patterns provided by the reward model

- ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents [[Arxiv'25/08](https://arxiv.org/abs/2508.14040)]
  - Implement API calls (via LLM) for agents to interact with the environment
  - Step reward is the same as the final reward
  - Alternate between RL and SFT phases rather than doing SFT→RL sequentially. When entropy drops too low during RL, they inject SFT rounds to restore diversity in the policy's output distribution

- RAGEN: Understanding self-evolution in LLM agents via multi-turn reinforcement learning [[Arxiv'25/04](https://arxiv.org/abs/2504.20073)]
  - Extend PPO and GRPO to multi-turn reasoning, the equations are the same
  - Assume the reward for each turn is available

### Process rewards

- RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models [[Arxiv'26/03](https://arxiv.org/abs/2603.00724)]
  - Static, domain-specific reward models generalize poorly to OOD scenarios during RL iterations
  - Use LLM agents to dynamically select/synthesize reward functions per query: retrieve reward models from the Internet and generate programmatic verifiers via code generation
  - Self-evolving reward system that adapts to shifting data distributions during training

#### Parametric reward model

- Agentic Reinforcement Learning with Implicit Step Rewards [[Arxiv’25/09](https://arxiv.org/abs/2509.19199)]
  - Train a PRM through DPO using the DPO loss and use it to train the policy together
  - Similar to PRIME and shows marginal improvements
  - PRIME uses MSE loss and this one uses preference loss but uses the policy model loss to train PRM
  - DPO vs CE loss for training PRM; 2. GRPO vs RLOO for calculating the reward

- SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution [[Arxiv'25/05](https://arxiv.org/abs/2505.20732)]
  - Train an LLM as a progress estimator to assign a contribution score for each step
  - The sum of the contribution scores is the final reward (either 0 or 1)
  - Model assigns a reward to each step, training objective is to make the sum of the contribution scores close to the final reward (MSE loss) [No constraints on individual step scores]
    - May suffer from the non-identical issue

- Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/abs/2509.20616)]
    - Train LLM agents for multi-turn task planning by decomposing long-horizon tasks into single-turn reasoning problems, where the model (Qwen2.5-1.5B) learns to predict the correct next action at each state using GRPO with dense rewards from expert trajectories (Llama3-70B). Theory proves that improving single-step optimality leads to higher overall multi-turn success rate.

- Process reward models for llm agents: Practical framework and directions [[Arxiv'25/02](https://arxiv.org/abs/2502.10325)]
  - Uses either classification loss or IRL loss to update the PRM

#### Non-parametric reward model

- Reinforcement Learning via Self-Distillation [[Arxiv'26/01](https://arxiv.org/pdf/2601.20802)]
  - $\mathcal{L}_{\text{SDPO}}(\theta) := \sum_{t} \text{KL}(\pi_{\theta}(\cdot \mid x, y_{<t}) \| \text{stopgrad}(\pi_{\theta}(\cdot \mid x, f, y_{<t})))$
  - $y$ is generated by giving only $x$ to model $\pi_{\theta}$, $f$ is the feedback from the environment.
  - Use $\text{log} \frac{\pi(y|x,f,y)}{\pi(y|x,y)}$ as the intermediate reward, $\pi(y|x,f,y)$ is the self policy as the teacher model.
  - All the experiments are conducted on single-turn tasks. When comparing this method with GRPO, the models are trained 1 hr and 5 hrs (a strange setting).
  - This method can work depending on whether the policy itself knows where it is wrong given the feedback.
  - Similar paper: Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models [[Arxiv'26/01](https://arxiv.org/abs/2601.18734)]

- CriticSearch: Fine-Grained Credit Assignment for Search Agents via a Retrospective Critic [[Arxiv'25/11](https://arxiv.org/pdf/2511.12159)]
  - Use a teacher model to produce step-wise reward for RL training

- RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents [[Arxiv'25/07](https://arxiv.org/abs/2507.22844)]
  - Create three meta rewards: planning, exploration, and reflection (traj has the pattern of <reflection> + corrective actions)

- Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design [[Arxiv'25/05](https://arxiv.org/abs/2505.11821)]
  - Design multi-turn GRPO and PPO, the intermediate rewards are obtained using verifiable rewards / LLM-as-Judge

- Group-in-Group Policy Optimization for LLM Agent Training [[Arxiv'25/05](https://arxiv.org/abs/2505.10978)]
  - GRPO with outcome reward  
  - Step-wise reward: Group similar states, where the reward of each state is its discounted return, and use GRPO to compute the advantages for actions in each group
  - May suffer if the states are diverse

- Reinforcement Learning for Long-Horizon Interactive LLM Agents [[Arxiv'25/02](https://arxiv.org/abs/2502.01600)]
  - PPO with RLOO to estimate the reward, examined three different variants: token-level, step-level, and trajectory-level. Token-level is the most effective.

- From novice to expert: Llm agent policy optimization via step-wise reinforcement learning [[Arxiv'24/11](https://arxiv.org/abs/2411.03817)]
  - Collect expert trajectories, fix n steps of the trajectory, and train the model to generate the next step using RL
  - Still the idea of distillation; turn distillation into a RL problem (not exactly PRM; similar to using expert as PRM)

### Objective functions & training scheduling 

- Maximum Likelihood Reinforcement Learning [[Arxiv'26'02](https://arxiv.org/abs/2602.02710)]
  - Use pass k rate as sample weight for training -> mimic the maximum likelihood loss
  - Compute gradient estimation as a REINFORCE-style with score function
  - Use unbiased estimation to reduce variance
  - Final gradient is similar to GRPO with pass rate as the sample weight 

- Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents [[Arxiv'25/09](https://arxiv.org/abs/2509.09265)]
  - Expected gradient norm is monotonically coupled with policy entropy
  - $A_{mod}(i, t) = A^{i} * g(H_{t}) + f(H_{t+1})$
    - g(H_{t}): For a confident step, g(H_{t}) > 1, which amplifies its gradient; conversely, for an uncertain step, g(H_{t}) < 1, which attenuates its gradient.
    - f(H_{t+1}): encourages the agent to select actions that lead to a more predictable and less ambiguous future state
  - Task: Webshop and ALFWorld (i.e., agent benchmark with sparse reward)
  - LLM model: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct

- Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning (GR³) [[Arxiv'26/03](https://arxiv.org/abs/2603.10535)]
  - Problem: length inflation — model learns verbosity to maximize rewards. 
    - Additive length penalty $R' = R - \lambda \cdot L$ creates optimization shortcuts (model inflates $R$ to offset $\lambda \cdot L$)
  - GR³: multiplicative rescaling $\hat{R}^{(i)} = R^{(i)} \cdot \frac{1}{1 + \alpha \cdot \ell^{(i)} / \bar{\ell}}$ where $\ell^{(i)}$ is response length and $\bar{\ell}$ is group mean length.
    - Encourage high reward and short responses
    - This works within a group so only consider the relative length

- Differentiable Evolutionary Reinforcement Learning [[Arxiv'25/12](https://arxiv.org/abs/2512.13399)]
  - Automatically search the best combination of reward functions (e.g., format reward, length reward) via a meta reward function and train this reward function together with the policy

- Beyond Precision: Training-Inference Mismatch is an Optimization Problem and Simple LR Scheduling Fixes It [[Arxiv'26/02](https://arxiv.org/abs/2602.01826)]
  - Training-inference mismatch (e.g., vLLM vs FSDP) amplifies gradient noise, and both escalate together as training progresses. Since LR directly controls update magnitude, shrinking LR suppresses the mismatch by reducing the update size.
  - Response length serves as an early-warning signal for impending instability (longer responses → more gradient noise). Propose a dynamic LR scheduler that decays LR when response length grows, proactively preventing divergence.


### Sampling strategies

- RetroAgent: From Solving to Evolving via Retrospective Dual Intrinsic Feedback [[Arxiv'26/03](https://arxiv.org/abs/2603.08561)]
  - Encourage self-reflection
  - Introduces a PRM to encourage self-evolving, where the reward is designed as the improvement of the current reward over the reward of previous trajectories
  - Introduces a memory which can be used as prior knowledge for future generation
  - Train a reflection policy to generate reflection

- Meta-Reinforcement Learning with Self-Reflection for Agentic Search (MR-Search) [[Arxiv'26/03](https://arxiv.org/abs/2603.11327)]
  - Instead of generating each traj independently, generate later trajs based on the reflection of existing ones 
  - Train the policy with these reflection trajs using RLOO

- Complementary Reinforcement Learning [[Arxiv'26/03](https://arxiv.org/abs/2603.17621)]
  - Two co-evolving components: (1) policy actor optimized via sparse outcome rewards, (2) experience extractor that distills useful experiences from past episodes
  - The experience extractor is trained based on whether its distilled experiences actually help the actor succeed — get reward based on the outcome reward of the traj it guides
  - Two models are trained async and interact via an experience manager
  - Improves sample efficiency by leveraging cross-episode experience; 10% improvement in single-task, scales to multi-task

- RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization [[Arxiv'26/03](https://arxiv.org/abs/2603.03078)]
  - Hybrid-policy Agentic Rollout: retrieve off-policy step-level traces to expand reasoning scope during rollout
  - Retrieval-aware Policy Optimization: 
    - Add a retrieval reward to guide the model to retrieve helpful trajs

- In-Context Reinforcement Learning for Tool Use in Large Language Models (ICRL) [[Arxiv'26/03](https://arxiv.org/abs/2603.08068)]
  - RL-only framework that eliminates SFT by using few-shot in-context examples during RL rollouts to teach tool invocation
  - Procedure: (1) prepend $k$ few-shot tool-use demos to the rollout prompt → (2) model generates its own tool-augmented trajectory and receives outcome reward → (3) update policy via RL → (4) curriculum: gradually reduce $k$ across stages (e.g., $5 \to 3 \to 1 \to 0$) until the model invokes tools zero-shot
  - In-context examples bootstrap exploration (so the model doesn't start from random tool calls), then get removed as the model internalizes the format

- Spark: Strategic Policy-Aware Exploration via Dynamic Branching for Long-Horizon Agentic Learning [[Arxiv'26/01](https://arxiv.org/pdf/2601.20209)]
  - Ask the policy to 'identify' the current state (in system prompt): either in explore state or normal thinking state (differentiated by the tag `[EXPLORING]` and `[THINKING]`). If identified as explore state, then the system will force beam search. The experiments show it is more effective than GRPO.

- R3L: Reflect-then-Retry Reinforcement Learning with Language-Guided Exploration, Pivotal Credit, and Positive Amplification [[Arxiv'26/01](https://arxiv.org/pdf/2601.03715)]
  - Algorithm (Use LLM to find important steps)
    - step 1: sample n/2 trajectories.
    - step 2: For each trajectory, use the current model to perform structural analysis: for example, error analysis, then provide improvement suggestions, and determine which step is the pivot step that caused the error
    - step 3: Retain the trajectory before the pivot step, given the error analysis and improvement suggestions, ask the LLM to continue generating the trajectory after the pivot
    - step 4: Combine the steps before the pivot step and the trajectory generated by step 3 after the pivot, forming a new set of n/2 trajectories.
  - Shared prefix does not calculate gradients
  - Scaling advantages to address difficult problems are mostly negative reward issues
  - Two additional SFT objectives were designed to enhance the model’s diagnostic and retry capabilities (useful for steps 2 and 3).

- Meta-RL Induces Exploration in Language Agents [[Arxiv'25/12](https://arxiv.org/abs/2512.16848)]
  - Rollout and training with cross-traj signals
  - After each traj finishes, prompt the agent to generate the textual reflection on this traj, providing specific feedback and plan to guide the next traj.
  - The return for each action is the sum of the discounted return of future actions plus the discounted return of future trajectories at step 0 (The step reward is provided by the environment or the task)

- Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning [[Arxiv'25/05](https://arxiv.org/abs/2505.24726)]
  - If a traj fails, the model is prompted to generate a self-reflection and continue generating (question + failed traj + reflection + ...)
  - If it succeeds, use GRPO to reward only the tokens generated in the self-reflection

- Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards [[Arxiv'25/06](https://arxiv.org/abs/2506.11425)]
  - If a traj fails, the model is prompted to generate a self-reflection and continue generating (question + reflection + ...)
  - Use DPO for training

- S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models [[Arxiv'25/05](https://arxiv.org/abs/2505.07686)]
  - One reasoning path with several early-exit branches. For example, early-exit at 50% of the current reasoning path. Using this to create a group (like GRPO) and calculate the advantage.
  - Aims to reduce the reasoning length and improve efficiency.

- Effective Harnesses for Long-Running Agents [[Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)]
  - Problems: Agent tries to one-shot complex tasks, runs out of context mid-implementation; later agent sees progress and declares done prematurely
  - Solution: Initializer agent sets up environment and progress file; coding agent makes incremental progress with structured updates
  - Open questions: Single general-purpose agent vs multi-agent architecture; generalization to other domains (scientific research, financial modeling)

- First return, then explore [[Nature'21/09](https://www.nature.com/articles/s41586-020-03157-9)]
  - Visual game exploration
  - Phase 1: Return and explore until solved -- record states based on similarity, select promising states (less explored or leading to more new states), replace states with better solutions (fewer steps)
  - Phase 2: Use SFT for robustification

- Exploration by Random Network Distillation [[Arxiv'18/10](https://arxiv.org/abs/1810.12894)]
  - Encourages the agent to explore novel, unseen states
  - Two networks: a randomly initialized fixed target network and a predictor network trained to predict the target's output
  - Prediction error between the two networks is used as the intrinsic exploration bonus (higher error = state is novel = higher reward)
  - Based on visual games; using PPO with a Dual Value Head to separately estimate stationary extrinsic and non-stationary intrinsic rewards


### Stability and others

- Meta-Harness: End-to-End Optimization of Model Harnesses [[Arxiv'26/03](https://arxiv.org/abs/2603.28052)]
  - Harness: Code  determines what to store, retrieve, and show to the model
  - Propose an agentic harness management system that propose new harnesses for new tasks based on existing harnesses stored in the file system  

- Harness Design for Long-Running Application Development [[Anthropic](https://www.anthropic.com/engineering/harness-design-long-running-apps)]
  - Problems: context anxiety (model wraps up prematurely near perceived context limits); self-evaluation bias (agent praises its own mediocre work)
  - Solutions:
    - Frontend design: generator-evaluator loop (GAN-inspired); four gradable criteria (design quality, originality, craft, functionality); evaluator uses Playwright MCP to interact with live pages before scoring
    - Full-stack coding: three-agent architecture (planner → generator → evaluator) with sprint-based decomposition and file-based inter-agent communication. Before each sprint, the generator and evaluator negotiated a sprint contract: agreeing on what "done" looked like for that chunk of work before any code was written. 
  - Takeaway: separating creation from evaluation is more tractable than self-evaluation; convert subjective judgments into concrete gradable criteria

- Effective harnesses for long-running agents [[Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)]
  - For long-running agents, merely rely on compaction leads to problems
    - model running out of context in the middle of its implementation, leaving the next session to start with a feature half-implemented and undocumented
    - After some features had already been built, a later agent instance would see that progress had been made, and declare the job done
  - Solutions:
    -  The very first agent session asks the model to write a json of feature requirements expanding on the user’s initial prompt (each feature with description, steps, and status), and a claude-progress.txt file that keeps a log of what agents have done.
    -  Every subsequent session is prompted to firts read the progress.txt and git history, then work on one feature, commits the change (enabling easy reverts), then leaves updates in progress.txt. 

- The Optimal Token Baseline: Variance Reduction for Long-Horizon LLM-RL [[Arxiv'26/03](https://arxiv.org/abs/2602.07078)]
  - Standard REINFORCE-style leads to high variance for long-horizon tasks as noise accumulates and the reward is sparse
    - Existing baselines do not consider sequence heterogeneity and differences in sequence energy 
  - Propose a per-token optimal baseline: weighted average of the return, where the weight depends on the squared norm of the score function at each token (each token has its own baseline rather than a shared baseline for the sequence)

- Stabilizing Reinforcement Learning with LLMs: Formulation and Practices [[Qwen Team, Arxiv'25/12](https://arxiv.org/abs/2512.01374)]
  - Three issues: (1) training–inference discrepancy (FP8 vs. BF16); (2) policy staleness (due to asynchronous rollout); (3) training–inference discrepancy can cause inconsistent routed experts for MoE models
  - Existing solution:
    - (1) & (2): importance sampling
    - (3): routing replay (fix the expert used by inference when do training)
  - Given the assumption: the IS for each token is small: $1+\epsilon \ll 1$, the token-level optimization objective can be viewed as the first-order approximation of the sequence-level optimization objective.
  - Proposed MiniRL: GRPO + Clip (mask the gradient for tokens which are higher than a threshold when reward > 0 and lower than a threshold when reward < 0)
  - Evaluated on Qwen3-30B-A3B with synchronous training:
    - For on-policy training (the global batch size equals the mini-batch size):
      - MiniRL is the best, adding length normalization leads to suboptimal performance
    - For off-policy training:
      - Routing Replay and clipping become essential for stable training

- SimpleTIR: End-to-End Reinforcement Learning  for Multi-Turn Tool-Integrated Reasoning [[Arxiv'25/09](https://arxiv.org/abs/2509.02479)]
  - Problem: LLM-generated responses after multiple turns (with tool observation) are OOD compared to the pre-training and SFT data; each token has low probability, which leads to extremely high importance sampling ratio ($\frac{\pi_{\text{new}}(y)}{\pi_{\text{old}}(y)}$) for trajs with negative reward.
  - Solution: clipping the importance ratio is appealing but the threshold is hard to set. The authors filter out trajs with void turns (no tool invocation or final answer).
  - Scenario: Zero RL with math problems

- When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch [[Notion'25/10](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)]
  - Mismatch between training and inference causes collapse of RL training: $\mathbb{E} _{x\sim \mathcal{D}}\mathbb{E} _{y\sim \textcolor{red}{\pi _{\theta}^{\mathrm{vllm}}}\left( \cdot |x \right)}\left[ R\left( x,y \right) \nabla _{\theta}\log \textcolor{blue}{\pi _{\theta}^{\mathrm{fsdp}}}\left( y|x \right) \right]$
  - Long context amplifies this issue as the difference is accumulated
  - The **agent** receives a tool response, which is often structured text (e.g., context enveloped by `<python_output>` and `</python_output>` tags) that is OOD compared to its pre-training and SFT data.Faced with this unfamiliar OOD context, the **agent's policy** becomes more uncertain, making it more likely to sample low-probability tokens in its subsequent turns (which is also observed in [**SimpleTIR**](https://github.com/ltzheng/SimpleTIR)).
  - Hardware differences also contribute to the instability
  - Solution: sequence-level importance sampling ($\frac{\textcolor{blue}{\pi_{\theta}^{\mathrm{fsdp}}}(y|x)}{\textcolor{red}{\pi_{\theta}^{\mathrm{vllm}}}(y|x)}$)

- UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/pdf/2509.02544v1)]

- MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment [[Arxiv'25/07](https://arxiv.org/abs/2507.05720)]
  - Task synthesis and filtering out low-quality tasks
  - Step reward is the same as the final reward
  - Define a set of rules to scale the reward of each trajectory

- Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning [[ICLR 25](https://arxiv.org/abs/2411.02337)]
  - Web navigation tasks
  - Long-horizon interaction yet provides sparse and delayed rewards, making policy improvement challenging and costly, and often resulting in training collapse

- Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models [[Arxiv'25/12](https://arxiv.org/abs/2512.13607)]
  - Cascade RL process begins with applying general-domain Reinforcement Learning from Human Feedback (RLHF) to the SFT models, followed by domain-wise Reinforcement Learning with Verifiable Rewards (RLVR).

- GenEnv: Difficulty-Aligned Co-Evolution Between  LLM Agents and Environment Simulators [[Arxiv'25/12](https://arxiv.org/abs/2512.19682)]
  - Env LLM generates tasks, evaluation metrics, and potentially ground truth, the goal is to generate tasks that are challenging for the agent (avg acc = 0.5)
  - Agent LLM rollouts to generate trajectories, and then use RL (GRPO)to train the agent to improve the performance
  - Env LLM uses Reward-Weighted Regression (weighted SFT loss) to update its parameters
  - Env LLM is like a task generator; it does not generate new environments or simulate the environment.

- AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/pdf/2509.08755)]
  - Provide env and training framework
  - Propose ScalingInter-RL: progressively add interaction rounds - their experiments show that beginning with a large number of interaction turns often leads the model into redundant reasoning and unproductive actions.

- RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs? [[Arxiv'25/09](https://arxiv.org/abs/2509.21016)]
  - Using synthetic coding problems to verify how RL can unlock new algorithms in LLMs

- TL-GRPO: Turn-Level RL for Reasoning-Guided Iterative Optimization [[Arxiv'26/01](https://arxiv.org/pdf/2601.16480)]
  - Designed for iterative optimization - play the same game multiple times and take the best reward (like multi-arm bandits)
  - First generate only one traj, then generate more trajs based on the generated one and use GRPO to calculate the advantage of each step.
    - Turn a bandit problem into a multi-step problem by considering the dependency across turns and applying GRPO across turns. 


## New model architectures

- Attention Residuals [[Arxiv'26/03](https://arxiv.org/abs/2603.15031)]
  - Problem: standard residual connections ($\mathbf{h}_l = \mathbf{h}_{l-1} + F(\mathbf{h}_{l-1})$) accumulate all layer outputs with fixed unit weights, causing uncontrolled hidden-state growth and progressive dilution of each layer's contribution
  - **AttnRes**: replace fixed accumulation with softmax attention over all preceding layer outputs
    - $\mathbf{h}_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot \mathbf{v}_i$, where $\alpha_{i \to l} = \frac{\exp(\mathbf{w}_l^\top \text{RMSNorm}(\mathbf{k}_i))}{\sum_{j=0}^{l-1} \exp(\mathbf{w}_l^\top \text{RMSNorm}(\mathbf{k}_j))}$
    - $\mathbf{w}_l \in \mathbb{R}^d$: learned pseudo-query per layer; $\mathbf{k}_i$: keys from prior layer outputs; $\mathbf{v}_i$: prior layer representations
    - Each layer selectively aggregates earlier representations with learned, input-dependent weights
  - **Block AttnRes**: partition layers into $N$ blocks; apply attention only over block-level representations instead of all layers, reducing memory from $O(Ld)$ to $O(Nd)$

- mHC: Manifold-Constrained Hyper-Connections [[Arxiv'25/12](https://arxiv.org/abs/2512.24880)]
  - Hyper connections extends standard Residual Connections by widening the residual stream and using dynamic mixing matrices instead of simple addition.
    - $y = W_{skip} \cdot x + W_{res} \cdot F(x)$ x here is the input, the shape is (n, d) where in residual stream is d. Hyper connections widen the residual stream.
  - Issue: HC breaks the Identity Mapping property, causing training instability and high memory access overhead.
  - Manifold-Constrained Hyper-Connections (mHC) projects the connection matrices onto Birkhoff polytope (the sum of each row and each column is 1).
  - Also provides the optimization for training mHC, including kernel fusion and memory efficient training.

- Kimi Linear [[ArxXiv'25.10](https://arxiv.org/abs/2510.26692)]
  - Propose Kimi Delta Attention
  - Kimi Linear can be a drop-in replacement for full attention architectures with superior performance and efficiency, including tasks with longer input and output lengths.

- Step-3 is Large yet Affordable: Model-System Co-Design for Cost-Effective Decoding [[Arxiv'25/07](https://arxiv.org/abs/2507.19427)]
  - System and model co-design
  - Key findings:
    - Neither total nor activated parameter count is a good indicator for decoding cost — architecture design matters more
    - Attention design dominates decoding cost, especially at longer contexts (attention cost grows with context while FFN cost stays constant)
    - Hardware-attention alignment is critical: MFA achieves arithmetic intensity of 128, well-matched to diverse hardware; DeepSeek-V3's MLA (intensity 512) bottlenecks on lower-tier accelerators
    - Over-sparse MoE models suffer efficiency losses despite fewer activated parameters — sparsity needs hardware-aware design (Hardware aware sparsity threshold)
  - **Multi-Matrix Factorization Attention (MFA)** [[Arxiv'24/12](https://arxiv.org/abs/2412.19255)]: low-rank factorization in QK circuit; share projection matrices across heads so only low-rank compressed KV needs caching. 
  - **Attention-FFN Disaggregation (AFD)**: decouple attention and FFN layers into specialized subsystems; attention is memory-bound (KV cache access) while FFN is compute-bound (matrix multiply)
    - Similar to NV models

