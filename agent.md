# Agentic model and techniques

Below, we summarize the latest agentic models, as well as some notable and recent reasoning techniques.  

## Table of Contents
- [Newest models](#newest-models)
- [Code reasoning](#code-reasoning)
- [SFT-based reasoning](#sft-based-reasoning)
- [RL-based reasoning](#rl-based-reasoning)
  - [Online RL (train LLMs with ORM or PRM)](#online-rl-train-llms-with-orm-or-prm)
  - [Offline RL](#offline-rl)
  - [Training with PRM](#training-with-prm)
- [Agentic modeling (linear attentions)](#agentic-modeling-linear-attentions)
- [Memory management](#memory-management)

## Newest models

- ***GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models*** [[Arxiv'25/8](https://arxiv.org/pdf/2508.06471)]
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

- Earlier methods can be found [here](https://docs.google.com/document/d/1w_0oVWrUQxq6rU2KmY4JrbbVYbq0odLTAf4ta7ZiIdo/edit?usp=sharing)
- Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (https://arxiv.org/pdf/2508.08221)
### Offline RL

- Representative works can be found [here](https://docs.google.com/document/d/1w_0oVWrUQxq6rU2KmY4JrbbVYbq0odLTAf4ta7ZiIdo/edit?usp=sharing)

### Training with PRM

- Learn PRM from ORM
 - Process Reinforcement through Implicit Rewards [[Arxiv'25/02](https://arxiv.org/abs/2502.01456)]
  - Learn PRM from ORM and train the model with RLOO 

- Learn PRM from expert trajs
  - ***BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL for Reasoning*** [[Arxiv'25/06](https://arxiv.org/pdf/2506.17211)]

- Use model internal signal as PRM
  - ***Deep Think with Confidence*** [[Arxiv'25/08](https://arxiv.org/abs/2508.15260)]
  - Spurious rewards: rethinking training signals in RLVR [[Arxiv'25/06](https://arxiv.org/abs/2506.10947?)]
  - Learning to Reason without External Reward [[Arxiv'25/06](https://arxiv.org/abs/2505.19590)]

- ***Self-play***
  - RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning [[Arxiv'25/05](https://arxiv.org/pdf/2505.15034)]
    - RL for small model Q-Wen7B, outperfom prime.
    - Algorithm
      - Train Generator model and Verifier model together, Verifier model will generate step-wise reward (+1/-1) for Generator.
      - For Generator model, the reward would be a combine of step-wise reward and final outcome reward.
      - For the Verifier model, the reward would be a final reward, consisting of an outcome reward plus a format reward.

  - ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs
    - Trajectory response data: the thinking trajectory is $s=(s_{1}, ..., s_{t})$, and the answer trajectory is $a=(a_{1}, ..., a_{t})$.
    - The goal is to train an RPM model to assign a value to each $s_t$, denoted as $R(s_t \mid x, s_{<t}, a)$. 
    - Supervised training of the RPM model, include two loss term. One is for step-wise reward, using labels derived from a combination of LLM-Judge, the   alignment score between $s_t$ and $a_t$, and the coherence score; Another is for outcome reward, using labels derived ground-truth. 
    - The learned step reward could be used to train online RL model, specificially, the final reward is a combination of outcome reward and mean of learned step reward. 

  - SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning
    - This paper focus on error detection of wrong reasoning steps.
    - Algorithm
      - build an adversarial game between sneak generator and critic
      - The sneak generator try to convert the last correct step of a partial reasoning trajectory into wrong step.
      - The critic attempts to detect errors in the snake's output; after training, only the critic is used

  - S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning
    - Try to train RL agent to do self-correct and self-verifier: [s1, v1, s2, v2, ...,]
    - First perform supervised training to learn the pattern, then fine-tune using RL. 
    - For RL training, consider both ORL (i.e., only final correct v_M) and PRL training; ORL have better results. 

  - SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data
    - Setting: start from relative small Q-A dataset and try to finetune RL with this dataset.
    - Self-instruction: prompt the current LLM to generate additional questions.
    - Self-rewarding: derive rewards using majority voting on the final answer.
    - Training: fine-tune the LLM with the generated questions and majority-voted answers.

- Learn PRM from annotated data

- Learn PRM from MCTS rollout
  - ***StepWiser: Stepwise Generative Judges for Wiser Reasoning*** [[Arxiv'25/08](https://arxiv.org/abs/2508.19229)]
  
  - Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations [[Arxiv'23/12](https://arxiv.org/abs/2312.08935)]


## Memory management 


## Agentic modeling (linear attentions)

