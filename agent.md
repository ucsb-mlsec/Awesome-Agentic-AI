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
  - Hybrid thinking with different template; higher thinking efficiency; smarter tool calling **[only support in non-thinking mode]**

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
    - Thinking mode fusion with SFT: 
      - Combine data with and without reasoning paths into a unified dataset to combine thinking with non-thinking mode 
      - Provide final answers with partial thinking with a manually stop thinking token  
    - General RL: Instruction following; format following; preference alignment; **agent ability (tasks with tool invoke)**; ability to understand specific context
      - Outcome rewards: Rule-based reward; model-based reward with/without reference answers
  - Strong-to-weak distillation for small models, outperforms RL
    - Off-policy Distillation: Combine the outputs of teacher models generated with both /think and /no think modes for response distillation
    - On-policy Distillation: Student model generates on-policy sequences, query teacher models to get the targets, and is then fine-tuned by aligning its logits with those of a teacher model to minimize the KL divergence

## Code reasoning 

- Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR [[Arxiv'25/07](https://arxiv.org/abs/2507.15778)]
  - Archer: new data construction technique
  - Weaker KL regularization and higher clipping thresholds to reasoning tokens to encourage exploration, while using stronger constraints on knowledge tokens to maintain factual knowledge

- Multi-Turn Code Generation Through Single-Step Rewards [[ICML'25/02](https://arxiv.org/pdf/2502.20380)]
  - Iterative code generation and refinement with a generator and a verifier

- Integrate code interpret as part of reasoning rollout

  - CoRT: Code-integrated Reasoning within Thinking [[Arxiv'25/06](https://arxiv.org/pdf/2506.09820)]
    - Strong-to-weak distillation
    - Code integrated RL

  - R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning [[Arxiv'25/05](https://arxiv.org/pdf/2505.21668)]
    - Train the model to automatically call code interpreter during reasoning
    - SFT + RL with interpreter response (which is masked during training)

  - Towards Effective Code-Integrated Reasoning [[Arxiv'25/05](https://arxiv.org/pdf/2505.24480)]
    - Similar as R1-Code-Interpreter: reasoning + interpreter
    - Force to do execution in every step + RL

  - ReVeal: Self-Evolving Code Agents via Iterative Generation-Verification [[Arxiv'25/06](https://arxiv.org/pdf/2506.11442)]
    - Propose turn-aware PPO, which calculates return based on turns; the rest is similar as other works that involve interpreter

## SFT-based reasoning
- OpenThoughts: Data Recipes for Reasoning Models [[Arxiv'25/02](https://arxiv.org/abs/2506.04178)]
- s1: Simple test-time scaling [[Arxiv'25/03](https://arxiv.org/abs/2501.19393)]
  - Low sample size with diverse difficulty levels and topics
  - Budget forcing during test time: end thinking by appending end-of-thinking token; or extend thinking by appending “Wait” to reasoning trace
- LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters! [[Arxiv'25/02](https://arxiv.org/pdf/2502.07374)]
  - Small models learn reasoning structures; do not filter out wrong answers

## RL-based reasoning

### Online RL (train LLMs with ORM or PRM)

- Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model [[Arxiv'25/07](https://arxiv.org/pdf/2503.24290)]
  - PPO over GRPO; No KL; Simplified reward function design; scale up training data
  - Requires fewer training steps than deepseek distilled Qwen-32B

- **Ring-lite: Scalable Reasoning via C3PO-Stabilized Reinforcement Learning for LLMs [[Arxiv'25/06](https://arxiv.org/pdf/2506.14731)]**
  - MoE model with 16B active params (**smallest open-source MoE model**), outperforming some 8B and 14B models
  - Distillation and RL integration
    - Select distillation checkpoints based on entropy loss for RL is more efficient than validation performance
  - Constrained Contextual Computation Policy Optimization (**C3PO**) to stabilize training
    - Issues with GRPO: within-step length bias; across-step gradient variance
    - Fixed token budget per optimization step at token level
  - Two-stage training paradigm for multi-domain data integration
    - Long CoT SFT -> Math RL -> Code RL -> General SFT 

- **AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning** [[Arxiv'25/06](https://arxiv.org/pdf/2505.16400)] 
  - RL for small models (7B and 14B), outperform SFT  
  - Algorithm
    - GRPO with rule-based reward and token-level loss
    - On-policy training: Stable RL training and helps prevent entropy collapse
    - **No KL term**, the loss becomes the REINFORCE objective
    - Rule-based reward with execution results as reward for coding
  - **Length extension and curriculum learning**
  - SFT -> Math only training -> Code only training

- Reinforcement Pre-Training [[Arxiv'25/06](https://arxiv.org/pdf/2506.08007)]

- **Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning** [[Arxiv'25/08](https://arxiv.org/pdf/2508.08221)]
  - Normalization:
    - Group-level normalization is stable under different datasets (easy and hard math dataset) and different model sizes (Qwen 4B & 8B, base and aligned)
    - Batch-level normalization is unstable, which is due to the standard deviation of the batch will swiftly decrease over training
    - Calculating the **mean** at the local (group) level and the **standard deviation** at the global (batch) level enables more robust reward shaping.
  - Clip value:
    - For models with stronger fundamental reasoning abilities, increasing the clip higher parameter is more likely to facilitate exploration of better solution paths.
  - Loss aggregation:
    - Compared to sequence-level calculation (GRPO) token-level loss (DAPO) proves to be more effective on Base models, while showing limited improvement on Instruct models.
  - Overlong filtering:
    - Overlong filtering (DAPO) shows limited effectiveness on long-tail reasoning tasks; however, it can enhance the accuracy and clarity of responses in medium and short-length reasoning tasks. Still better than truncate (GRPO)

- Earlier methods can be found [here](https://docs.google.com/document/d/1w_0oVWrUQxq6rU2KmY4JrbbVYbq0odLTAf4ta7ZiIdo/edit?usp=sharing)

### Offline RL

- Representative works can be found [here](https://docs.google.com/document/d/1w_0oVWrUQxq6rU2KmY4JrbbVYbq0odLTAf4ta7ZiIdo/edit?usp=sharing)

### Training with PRM

- Learn PRM from expert trajs
  - ***BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL for Reasoning*** [[Arxiv'25/06](https://arxiv.org/pdf/2506.17211)]
    - For hard problem; SFT cannot learn well; RL cannot get correct answer in early steps
    - Pre-fill partial expert demonstrations and let small models to continue generating based on the partial expert demonstration
      - Dynamically adjust the length based on the accuracy of small model rollout
    - SFT + RL (GRPO)
  - UFT: Unifying Supervised and Reinforcement Fine-Tuning [[Arxiv'25/05](https://arxiv.org/abs/2505.16984)]

- Use model internal signal as PRM
  - Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning [[Arxiv'25/08](https://arxiv.org/pdf/2509.00125)]
  - ***Deep Think with Confidence*** [[Arxiv'25/08](https://arxiv.org/abs/2508.15260)]
    - Design different conference computing methods 
      - token confidence is similar as entropy
      - Group confidence based on sliding windows and positions
      - Weighted majority voting based on confidence during inference 
  - Spurious rewards: rethinking training signals in RLVR [[Arxiv'25/06](https://arxiv.org/abs/2506.10947?)]
  - Learning to Reason without External Reward [[Arxiv'25/06](https://arxiv.org/abs/2505.19590)]

- ***Learn a PRM through iterative training reward and policy***
  - RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning [[Arxiv'25/05](https://arxiv.org/pdf/2505.15034)]
    - RL for small model Qwen2.5-7B, outperform prime.
    - Algorithm
      - Train Generator model and Verifier model together, Verifier model will generate step-wise (\n\n) reward (+1/-1) for Generator.
      - For Generator model, the reward would be a combine of step-wise reward and final outcome reward
      - For the Verifier model, the reward is a final reward, consisting of an outcome reward plus a format reward
      - Use the same dataset as prime, first supervised training, then RL fine-tuning.

  - ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs [[Arixv'25/06](https://arxiv.org/abs/2506.18896)]
    - Trajectory response data: the thinking trajectory is $s=(s_{1}, ..., s_{t})$, and the answer trajectory is $a=(a_{1}, ..., a_{t})$.
    - The goal is to train an RPM model to assign a value to each $s_t$, denoted as $R(s_t \mid x, s_{<t}, a)$.
    - Supervised training of the RPM model, include two loss term. One is for step-wise reward, using labels derived from a combination of LLM-Judge, the alignment score between $s_t$ and $a_t$, and the coherence score; Another is for outcome reward, using labels derived ground-truth.
    - The learned step reward could be used to train online RL model, specificially, the final reward is a combination of outcome reward and mean of learned step reward.
    - LLM model: Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct.

  <!-- - SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data [[Arxiv'25/05](https://arxiv.org/pdf/2505.20347)]
    - Setting: start with a relatively small Q-A dataset and finetune the RL model on it
    - Self-instruction: prompt the current LLM to generate additional questions
    - Self-rewarding: derive rewards using majority voting on the final answer
    - Training: fine-tune the LLM with the generated questions and majority-voted answers
    - LLM model: Qwen-2.5-7B, Llama-3.2-3B -->

  - SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning [[Arxiv'25/04](https://arxiv.org/abs/2504.19162)]
    - This paper focus on error detection of wrong reasoning steps
    - Algorithm
      - Build an adversarial game between sneak generator and critic
      - The sneak generator try to convert the last correct step of a partial reasoning trajectory into wrong step.
      - The critic attempts to detect errors in the snake's output; after training, only the critic is used

  - S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning [[ACL'25/02](https://arxiv.org/abs/2502.12853)]
    - Train the RL agent to perform self-correction and self-verification in sequence: $[s_{1}, v_{1}, s_{2}, v_{2}, ...]$
    - First perform supervised training to learn the pattern, then fine-tune using RL
    - For RL training, consider both ORM (i.e., only final correction $s_M$) and PRM training (PRM here is the answer correctness of the intermediate steps); ORM have better results
    - LLM model: Qwen2.5-7B-instruct, Lllama-3.1-8B.

- Learn PRM from ORM
  - Process Reinforcement through Implicit Rewards [[Arxiv'25/02](https://arxiv.org/abs/2502.01456)]
  - Learn PRM from ORM and train the model with RLOO

- Learn PRM from annotated data

- Learn PRM from MCTS rollout
  - ***StepWiser: Stepwise Generative Judges for Wiser Reasoning*** [[Arxiv'25/08](https://arxiv.org/abs/2508.19229)]
    - PRM with better step segment
      - Split reasoning steps by "\n\n" is problemic
      - Use a strong model to divide the reasoning generated by the target model -> SFT data -> train the target model
        - Target model can better generate chunks
    - MCTS for computing reward for each step
      - New reward design to capture more signal
    - RL training with GRPO for assigning reward for each step
  - Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations [[Arxiv'23/12](https://arxiv.org/abs/2312.08935)]

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
  - Extend PPO and GRPO to multi-turn reasoning`

## Memory management

## Agentic modeling (linear attentions)
