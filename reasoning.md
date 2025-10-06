# General LLM reasoning models and techniques

Below, we summarize some recent works on general LLM reasoning (without involving agents). 
Note that this direct has been extensively studied, with many techniques on improving RL-based post-training methods and tricks regarding the training process.
Here, we mainly focus on the techniques that provide process reward signals, as well as specific training methods for code reasoning.
On top of the basic training procedure, existing works have proposed many tricks; some of them are conflict with each other.
Here, we also try to summarize the key training takeaways for both SFT and RL-based reasoning training from our experiences.

## Table of Contents
- [Code reasoning](#code-reasoning)
- [SFT-based reasoning](#sft-based-reasoning)
- [RL-based reasoning](#rl-based-reasoning)
  - [Online RL (train LLMs with ORM or PRM)](#online-rl-train-llms-with-orm-or-prm)
  - [Offline RL](#offline-rl)
  - [Training with PRM](#training-with-prm)


## Code reasoning 

- CoDA: Coding LM via Diffusion Adaptive [[Arxiv'25/10](https://github.com/SalesforceAIResearch/CoDA/blob/main/technical_report.pdf)]
  - Diffusion model based code generation with 1.7B params (better conduct infilling)
  - All SFT with different masking strategies for pre-, mid-, and post- training
    - General data for pre-training; code centric data for mid- and post- training
  - Outperform 7B Qwen model on HumanEval/MBPP

- Multi-Turn Code Generation Through Single-Step Rewards [[ICML'25/07](https://arxiv.org/pdf/2502.20380)]
  - Iterative code generation with a generator (trained with SFT and data is selected based on the verifier and testing case passing rewards) and a verifier (trained with cross-entropy loss)

- Integrate code interpreter as part of reasoning rollouts

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

**This [post](https://henrygwb.github.io/posts/sft.htm) lists our key takeaways on SFT-based reasoning, with a focus on SWE and security tasks.**

- OpenThoughts: Data Recipes for Reasoning Models [[Arxiv'25/06](https://arxiv.org/abs/2506.04178)] 
    - Sampling multiple answers per question from a teacher model 
    - Models with better performance are not necessarily better teachers (QwQ-32B is a stronger teacher than DeepSeek-R1)
    - Verification and answer filtering methods are not important
    - Select questions from a small number (top 1 or 2) of high-quality sources leads to better downstream performance compared to optimizing for diversity
    - LLM-based difficulty and length filtering is better than embedding or fastText-based filtering

- s1: Simple test-time scaling [[Arxiv'25/03](https://arxiv.org/abs/2501.19393)] 
  - Low sample size with diverse difficulty levels and topics
  - Budget forcing during test time: end thinking by appending end-of-thinking token; or extend thinking by appending “Wait” to reasoning trace
- LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters! [[Arxiv'25/02](https://arxiv.org/pdf/2502.07374)] 
  - Small models learn reasoning structures; do not filter out wrong answers
 
## RL-based reasoning

- **Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning** [[Arxiv'25/08](https://arxiv.org/pdf/2508.08221)]

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

### Offline RL

- Representative works can be found [here](https://docs.google.com/document/d/1w_0oVWrUQxq6rU2KmY4JrbbVYbq0odLTAf4ta7ZiIdo/edit?usp=sharing)

### Training with PRM

- Learn PRM from expert trajs
  - ***BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL for Reasoning*** [[Arxiv'25/06](https://arxiv.org/pdf/2506.17211)]
    - For hard problem; SFT cannot learn well; RL cannot get correct answer in early steps
    - Pre-fill partial expert demonstrations and let small models to continue generating based on the partial expert demonstration
      - Dynamically adjust the length based on the accuracy of small model rollout
    - SFT + RL (GRPO)

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

- Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR [[Arxiv'25/07](https://arxiv.org/abs/2507.15778)]
  - Archer: new data construction technique
  - Weaker KL regularization and higher clipping thresholds to reasoning tokens to encourage exploration, while using stronger constraints on knowledge tokens to maintain factual knowledge