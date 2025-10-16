# Agentic model and techniques

Below, we summarize the latest agentic models, as well as some notable and recent reasoning techniques.  

## Table of Contents
- [Agentic model and techniques](#agentic-model-and-techniques)
  - [Table of Contents](#table-of-contents)
  - [Newest models](#newest-models)
  - [Code reasoning](#code-reasoning)
  - [SFT-based reasoning](#sft-based-reasoning)
  - [RL-based reasoning](#rl-based-reasoning)
    - [Online RL (train LLMs with ORM or PRM)](#online-rl-train-llms-with-orm-or-prm)
    - [Offline RL](#offline-rl)
    - [Training with PRM](#training-with-prm)
  - [Agentic RL](#agentic-rl)
  - [Memory management](#memory-management)
  - [Agentic modeling (linear attentions)](#agentic-modeling-linear-attentions)

## Newest models

- UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning [[Arxiv'2/9](https://arxiv.org/pdf/2509.02544v1)]
  
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

- SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution [[Arxiv'25/05](https://arxiv.org/abs/2505.20732)]
  - Train a progress estimator to assign a contribution score for each step. The sum of the contribution scores is the final reward (either 0 or 1). The training goal is to make the sum of the contribution scores close to the final reward (MSE loss).

- Group-in-Group Policy Optimization for LLM Agent Training [[Arxiv'25/05](https://arxiv.org/abs/2505.10978)]
  - A two-level grouping structure: preserving episode-level grouping for holistic performance comparison (GRPO), while dynamically constructing an additional set of step-level groups by retroactively aggregating actions encountering the same environment states.

- Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/abs/2509.20616)]
  - Train LLM agents for multi-turn task planning by decomposing long-horizon tasks into single-turn reasoning problems, where the model (Qwen2.5-1.5B) learns to predict the correct next action at each state using GRPO with dense rewards from expert trajectories (Llama3-70B). Theory proves that improving single-step optimality leads to higher overall multi-turn success rate.

- RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training [[Arxiv'25/09](https://arxiv.org/abs/2509.21009)]
  - A new framework for efficient RL training.

- Online Process Reward Leanring for Agentic Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/abs/2509.19199v2)]
  - Similar to PRIME, the differences are 1. DPO vs CE loss for training PRM; 2. GRPO vs RLOO for calculating the reward.

- RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents [[Arxiv'25/07](https://arxiv.org/abs/2507.22844)]
  - Create different rules to calculate the reward for each step.

- COMPUTERRL: SCALING END-TO-END ONLINE REINFORCEMENT LEARNING FOR COMPUTER USE AGENTS [[Arxiv'25/08](https://arxiv.org/abs/2508.14040)]
  - Implement API calls (via LLM) for agents to interact with the environment
  - Step reward is defined as the final reward
  - Incorporate SFT in the later phase of RL training to mitigate entropy collapse

- MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment [[Arxiv'25/07](https://arxiv.org/abs/2507.05720)]
  - Task synthesis and filtering out low-quality tasks
  - Step reward is defined as the final reward
  - Define a set of rules to scale the reward of each trajectory

- From novice to expert: Llm agent policy optimization via step-wise reinforcement learning [[Arxiv'24/11](https://arxiv.org/abs/2411.03817)]
  - Collect expert trajectories, fix n steps of the trajectory, and train the model to generate the next step using RL (similar to )

- Process reward models for llm agents: Practical framework and directions [[Arxiv'25/02](https://arxiv.org/abs/2502.10325)]
  - Two frameworks:
    - SFT via rollouts (similar to Group-in-Group, group the same states)
    - Inverse RL (collecting expert trajectories to train PRM)

- Reinforcement Learning for Long-Horizon Interactive LLM Agents [[Arxiv'25/02](https://arxiv.org/abs/2502.01600)]
  - PPO with RLOO to estimate the reward, examined three different variants: token-level, step-level, and trajectory-level. Token-level is the most effective.

- RAGEN: Understanding self-evolution in LLM agents via multi-turn reinforcement learning [[Arxiv'25/04](https://arxiv.org/abs/2504.20073)]
  - Extend PPO and GRPO to multi-turn reasoning

- COMPUTERRL: SCALING END-TO-END ONLINE REINFORCEMENT LEARNING FOR COMPUTER USE AGENTS

Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents
Expected gradient norm is monotonically coupled with policy entropy
A_mod(i, t) = A^{i} * g(H_{t}) + f(H_{t+1})
g(H_{t}): For a confident step, g(H_{t}) > 1, which amplify its gradient; Conversely, for an uncertain step, g(H_{t}) < 1, which attenuates its gradient.
f(H_{t+1}): encourages the agent to select actions that lead to a more predictable and less ambiguous future state
Task: Webshop and ALFWorld (i.e., agent benchmark with sparse reward)
LLM model: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct

## Memory management

## Agentic modeling (linear attentions)


