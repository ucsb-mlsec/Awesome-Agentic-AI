# General LLM reasoning models and techniques

Below, we summarize some recent works on general LLM reasoning (without involving agents). 
Note that this direction has been extensively studied, with many techniques on improving RL-based post-training methods and tricks regarding the training process.
Here, we mainly focus on the techniques that provide process reward signals, as well as specific training methods for code reasoning.
On top of the basic training procedure, existing works have proposed many tricks; some of them conflict with each other.
Here, we also try to summarize the key training takeaways for both SFT and RL-based reasoning training from our experiences.

## Table of Contents
- [General LLM reasoning models and techniques](#general-llm-reasoning-models-and-techniques)
  - [Table of Contents](#table-of-contents)
  - [Code reasoning](#code-reasoning)
  - [SFT-based reasoning](#sft-based-reasoning)
  - [RL-based reasoning](#rl-based-reasoning)
    - [Online RL](#online-rl)
    - [Offline RL](#offline-rl)
    - [Training with process reward](#training-with-process-reward)
      - [Explicitly train a process reward model](#explicitly-train-a-process-reward-model)
      - [Non-parametric process reward](#non-parametric-process-reward)
    - [Misc](#misc)
      - [Combat entropy collapse](#combat-entropy-collapse)


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

- DeepSeek-Math-V2 [[Github](https://github.com/deepseek-ai/DeepSeek-Math-V2)]
  - Training:
    - Train an initial verifier with GRPO using format reward and score reward (expert-labeled proof scores: 1/0.5/0)
    - Meta-verifier: Train a meta-verifier to evaluate verification quality; integrate meta-verification score into verifier reward
    - Proof generator initialized from trained verifier; iteratively train generator and verifier
  - Inference (IMO gold medal):
    - Generate 64 proofs, each with 64 verifications
    - Iterative refinement (up to 16 rounds): select top-64 proofs by avg score, pair each with 8 analyses (prioritize low-score ones), generate 512 new proofs
    - Stop when a proof passes all 64 verifications

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

- The Art of Scaling Reinforcement Learning Compute for LLMs [[Arxiv'25/10](https://arxiv.org/pdf/2510.13786)]
  - Propose a scaling law to predict the performance from RL compute budget: log(R) = a + b * sigmoid(log(C)), where R is the performance, C is the RL compute budget.
  - Scale training compute for different methods, they encounter different ceilings on their achievable performance
  - Methods that appear superior at small compute budgets can be worse when
extrapolated to large-compute regime

- RL-Scaling: Scaling Reinforcement Learning for LLMs with Rich Rewards [[Arxiv'25/10](https://arxiv.org/pdf/2510.14021)]
  - Propose a scaling law to predict the performance from RL compute budget: log(R) = a + b * sigmoid(log(C)), where R is the performance, C is the RL compute budget.
  - Scale training compute for different methods, they encounter different ceilings on their achievable performance
  - Methods that appear superior at small compute budgets can be worse when
extrapolated to large-compute regime
  - Common interventions thought to improve peak performance (e.g., loss aggregation, data curriculum, length penalty, advantage normalization) mainly adjust compute efficiency, while not changing the performance ceiling considerably.

- Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? [[Arxiv'25/04](https://arxiv.org/abs/2504.13837)]

### Online RL

Online RL methods are mainly used for post-training with verifiable outcome reward. Some recent works also explore using RL for [pre-training](https://arxiv.org/pdf/2506.08007) or mid-training (a new training stage between pre-training and post-training, mainly aim to improve agentic capabilities).


- QUESTA: EXPANDING REASONING CAPACITY IN LLMS VIA QUESTION AUGMENTATION [[Arxiv'25/09](https://arxiv.org/pdf/2507.13266)]
  - RL with easy prompts/questions hurts pass @k and reasoning ability; RL with hard prompts/questions leads to slow learning
  - Introduce partial solutions during training to reduce problem difficulty 
  - Dataset: OpenR1-Math-220K dataset -> filter to 26K hardest items, partial solution generated via DEEPSEEK-R1
  - LLM model: Nemotron-1.5B, DeepScaleR-1.5B

  - BREAD: Branched Rollouts from Expert Anchors Bridge SFT & RL for Reasoning [[Arxiv'25/06](https://arxiv.org/pdf/2506.17211)]
    - For hard problem; SFT cannot learn well; RL cannot get correct answer in early steps
    - Pre-fill partial expert demonstrations and let small models to continue generating based on the partial expert demonstration
      - Dynamically adjust the length based on the accuracy of small model rollout
    - SFT + GRPO


- Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model [[Arxiv'25/07](https://arxiv.org/pdf/2503.24290)]
  - Shows that PPO with GAE and rule-based reward without KL 
    - GAE hyper-parameters are important: the discount factor $\lambda$ controls the effective sequence length
    - Scale up data quantity and diversity (Do not only use the common training sets)
  - Requires fewer training steps than deepseek distilled Qwen-32B

- Ring-lite: Scalable Reasoning via C3PO-Stabilized Reinforcement Learning for LLMs [[Arxiv'25/06](https://arxiv.org/pdf/2506.14731)]
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
  - **Length extension and curriculum learning**:
    - Math only (8K-24K) -> Code only (24K-32K) -> Math only (32K)
    - From easy to hard problems based on LLM judge
    - Math only RL improve coding performance; not the other way around

- ICPO: Intrinsic Confidence-Driven Group Relative Preference Optimization for Efficient Reinforcement Learning [[Arxiv'25/11](https://arxiv.org/abs/2511.21005)]
  - add confidence to the reward, the intuition is: wrong response with high confidence and correct response with low confidence should have more weight

- Earlier methods can be found [here](https://docs.google.com/document/d/1w_0oVWrUQxq6rU2KmY4JrbbVYbq0odLTAf4ta7ZiIdo/edit?usp=sharing)
  - PPO, GRPO, DAPO
  - Value estimation: REINFORCE, GAE, RLOO

### Offline RL

Offline RL methods mainly refers to the methods that do not require rollout during training. DPO is the most representative method that learns from pairwise data. Follow up works generalize DPO to cases where pairwise data is not available. 

- Representative works can be found [here](https://docs.google.com/document/d/1w_0oVWrUQxq6rU2KmY4JrbbVYbq0odLTAf4ta7ZiIdo/edit?usp=sharing)

### Training with process reward

#### Explicitly train a process reward model

- Learn a PRM through iteratively training reward and policy
  - ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs [[Arixv'25/06](https://arxiv.org/abs/2506.18896)]
    - Trajectory response data: the thinking trajectory is $s=(s_{1}, ..., s_{t})$, and the answer trajectory is $a=(a_{1}, ..., a_{t})$.
    - The goal is to train an PRM to assign a value to each $s_t$, denoted as $R(s_t \mid x, s_{<t}, a)$. 
    - Supervised training of the RPM, include two loss term. One is for step-wise reward, using labels derived from a combination of LLM-Judge, the alignment score between $s_t$ and $a_t$, and the coherence score; Another is for outcome reward, using labels derived ground-truth. 
    - The learned step reward could be used to train online RL model, specificially, the final reward is a combination of outcome reward and mean of learned step reward. 
    - LLM model: Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct.

  - RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning [[Arxiv'25/05](https://arxiv.org/pdf/2505.15034)]
    - RL for small model Qwen2.5-7B, outperform prime.
    - Algorithm
      - Train Generator model and Verifier model together, Verifier model will generate step-wise (\n\n) reward (+1/-1) for Generator.
      - For Generator model, the reward would be a combine of step-wise reward and final outcome reward
      - For the Verifier model, the reward is a final reward, consisting of an outcome reward plus a format reward
      - Use the same dataset as prime, first supervised training, then RL fine-tuning. 

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

  - **Process Reinforcement through Implicit Rewards** [[Arxiv'25/02](https://arxiv.org/abs/2502.01456)]
    - Learn PRM from ORM and train the model with RLOO 

- Learn PRM from MCTS rollout
  - ***StepWiser: Stepwise Generative Judges for Wiser Reasoning*** [[Arxiv'25/08](https://arxiv.org/abs/2508.19229)]
    - PRM with better step segment
      - Split reasoning steps by "\n\n" is problemic 
      - Use a strong model to divide the reasoning generated by the target model -> SFT data -> train the target model
        - Target model can better generate chunks
    - MCTS for computing reward for each step
      - New reward design to capture more signal 
    - Use a reasoning model (RL training with GRPO) for assigning reward for each step
  - Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations [[Arxiv'23/12](https://arxiv.org/abs/2312.08935)]
  - DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding [[Arxiv'25/12](https://arxiv.org/pdf/2512.15000)]
    - Define the PRM for coding as individual functions
    - Obtain the PRM through bi-level optimization and meta-learning: 
      - Update the PRM using the MCTS rollout data and labels $\theta(Y) = \text{argmin} \ \mathcal{L}(f_{\theta}(X), Y)$ 
      - Update the intermediate steps' labels with ground truth labels 
      $Y = \text{argmin} \ \mathcal{L}(f_{\theta}(X_{meta}), Y_{meta})$


#### Non-parametric process reward

This line of methods explore using generation entropy or confidence as the process reward signal or factual knowledge constraint for token-level reward. There are also some works on exploring using entropy for hallucination detection. 

- How Far Can Unsupervised RLVR Scale LLM Training? [[Arxiv'26/03](https://arxiv.org/abs/2603.08660)]
  - Method: Classify unsupervised RLVR methods into two categories: intrinsic (deriving rewards from model's own signals, e.g., consistency, confidence) and external (grounding verification in computational asymmetries). Establish a unified framework showing intrinsic methods work by sharpening the model's initial distribution.
  - Finding 1: Intrinsic reward methods succeed when initial confidence aligns with correctness, but fail catastrophically when misaligned (confidence-correctness misalignment).
  - Finding 2: Intrinsic rewards consistently exhibit a rise-then-fall pattern; collapse timing is determined by the model's prior knowledge rather than engineering choices.
  - Finding 3: Intrinsic rewards remain useful for test-time training on small datasets. Propose "Model Collapse Step" as a metric for measuring model priors and predicting RL trainability.
  - Finding 4: External reward methods grounded in computational asymmetries show preliminary evidence of escaping the confidence-correctness ceiling, offering more scalable alternatives
    - confidence-correctness ceiling: intrinsic methods can only refine existing knowledge, not discover new knowledge

- Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning [[Arxiv'25/09](https://arxiv.org/pdf/2509.03646)]
  - Classify token into high-level planning tokens (i.e., i notice that, let's look at) and low-level execution tokens
  - Force the model to learn high-level strategic planning tokens
    - for high-level tokens, A(i, t) -> A(i, t) + \alpha * |A(i, t)|; for low-level tokens, keep the advantage unchanged
  - LLM model: Qwen2.5-7B, Qwen3-4B, LLama-3.1-8

- Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning [[Arxiv'25/08](https://arxiv.org/pdf/2509.00125)]
  - Encourage exploration for difficult tasks and exploitation for easy tasks
  - Control the degree of exploration by penalizing or rewarding high certainty

- Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR [[Arxiv'25/07](https://arxiv.org/abs/2507.15778)]
  - Define low-entropy knowledge-based tokens and high-entropy reasoning-based tokens
  - Apply weaker KL regularization and higher clipping thresholds to reasoning tokens to encourage exploration
  - Using stronger constraints on knowledge tokens to maintain factual knowledge

- Deep Think with Confidence [[Arxiv'25/08](https://arxiv.org/abs/2508.15260)]
  - Design different conference computing methods 
    - token confidence is similar as entropy
    - Group confidence based on sliding windows and positions
    - Weighted majority voting based on confidence during inference 

- Spurious rewards: rethinking training signals in RLVR [[Arxiv'25/06](https://arxiv.org/abs/2506.10947?)]

- Learning to Reason without External Reward [[Arxiv'25/06](https://arxiv.org/abs/2505.19590)]
  - Using model internal confidence as reward, encourage the model on its confident outputs
  - Follow up works propose variants on reward format, but still confidence/entropy related
  - Mainly help with early stage training and will encounter entropy collapse

### Misc

#### Combat entropy collapse

As an extension of the entropy and confidence based PRM, some works find that the post-training process is likely to encounter entropy collapse issue, where the model becomes overly confident and stops exploring better reasoning paths. Some works propose techniques to mitigate this issue. 

- Entropy-Preserving Reinforcement Learning [[Arxiv'26/03](https://arxiv.org/abs/2603.11682)]
  - Policy gradient algorithms naturally reduce entropy during training, limiting exploration diversity
  - REPO: $A_{\text{REPO}}(s, a) = A(s, a) - \beta_s \cdot L(s, a)$ where $L(s,a) = \log \pi(a|s) - \mathbb{E}[\log \pi(a'|s)]$ is mean-centered log-prob. Subtracting $\beta_s \cdot L$ reduces advantage of high-prob actions and boosts rare actions, counteracting natural entropy decay
  - ADAPO: asymmetric clipping $[\epsilon_{\text{low}}, \epsilon_{\text{high}}]$ with $\epsilon_{\text{low}} < \epsilon_{\text{high}}$. Tighter clip on entropy-reducing updates (model doubling down on confident tokens), looser clip on entropy-increasing updates (model exploring rare tokens)

- The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models [[Arxiv'25/06](https://arxiv.org/abs/2505.22617)]
  - The change in policy entropy is driven by the covariance between action probability and the change in logits
    - A high-probability action with high advantage would reduce policy entropy, while a rare action with high advantage would increase policy entropy
      - High covariance means the action is already good but the advantage is still high
  - Clip-Cov and KL-Cov, which clip and apply KL penalty to tokens with high covariances respectively 
    - Encourage actions with high advantage but low logits

- ICPO: Intrinsic Confidence-Driven Group Relative Preference Optimization for Efficient Reinforcement Learning [[Arxiv'25/12](https://arxiv.org/abs/2511.21005)]
  - Improvements over confidence-based reward to resolve the entropy collapse issue
  - Calculates a preference advantage score for each response by comparing the relative generation probabilities of multiple responses under the same input prompt; encourages responses with high reward but low preference (similar as the work above)

- M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization [[Arxiv'25/12](https://arxiv.org/pdf/2512.13070)]
  - Leverage a slowly evolving momentum model to provide a stable training target $\pi_{\theta_k} \leftarrow m\pi_{\theta_k} + (1 - m)\pi_{\theta_q}$ 
  - Filter out low-entropy trajectories, preserving essential policy diversity

#### Others

- Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning [[NeurIPS'25](https://openreview.net/pdf?id=3Sxby0hH1q)]
  - Prevent the reward hacking introduce in PRM during testing-phase scaling: the canonical summation-form credit assignment (cumulative gamma-decayed future rewards) easily induces LLMs to hack steps with high rewards
  - Propose minform credit assignment: The return of the steps before the worst step is the same as the worst step, and the returns of the steps after the worst step are all zero
  - Show that sum form has a larger error bound of Q function than the min form ($\frac{\epsilon}{1-\lambda}$) vs $\epsilon$

- RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs? [[Arxiv'25/10](https://arxiv.org/pdf/2509.21016)]
  - Propose a new dataset to evaluate LLMs on hard coding tasks that pretrained models always fail as well as OOD test sets
  - The following techs helps with grokking
    - Staged warm-up with dense rewards (use per-test pass rate as dense reward)
    - Experience replay (Retain and reinsert the previously successful traces)
    - Curriculum training 
    - Verification-in-the-loop (Include the failure feedback in the generation process; similar to add an explanation)

- Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs [[Arxiv'25/10](https://arxiv.org/pdf/2506.14245)]
  - Propose an LLM as a judege for CoT internal step correctness
  - Show that RL with verifiable outcome reward implicitly incentivizes correct reasoning

 - Reinforcement Learning for Reasoning in Large Language Models with One Training Example [[NeurIPS'25](https://arxiv.org/pdf/2504.20571)]
   - RLVR with one training example for math reasoning
   - Had some useful observations: cross-category generalization, increased frequency of self-reflection, and post-saturation generalization (sustained test performance improvement even after the training accuracy has saturated)
   - 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon

- S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models [[NeurIPS'25](https://arxiv.org/pdf/2505.07686)]
  - Train the model to prevent overthinking and exit early
  - Unlike parallel group,  S-GRPO only samples one reasoning path and serially selects multiple temporal positions from the path to exit thinking and directly generate answers (serial group)
    - Propose a decay reward strategy to (penalize long but correct answers) encourage early exit

- Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay [[NeurIPS'25](https://openreview.net/pdf?id=uwUkETPIJN)]
  - Difficulty-targeted online data selection: select moderate difficulty questions based on similiarity with reference questions and their ground difficulty
    - Reference questions will be rollouted during training to evaluate their in-training difficulty
  - Rollout replay: reuse recent rollouts

- Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts [[NeurIPS'25](https://arxiv.org/pdf/2506.02177)]
  - Filter out uninformative prompts during training based on the previous rollouts

