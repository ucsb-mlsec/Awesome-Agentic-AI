# SWE Agents and Agentic Post-training

In the following, we summarize the recent works in SWE agents and agentic training. Previously, we also summarize the works on pre-training and fine-tuning datasets, early-stage coding models, which can be found in this [[document](https://docs.google.com/document/d/1UuoWGaVmk_W7tSGeVm0Eb4-Hi4HU1BJ9YLcPQ-fKrEE/edit?usp=sharing)].
This [repo](https://github.com/codefuse-ai/Awesome-Code-LLM) has a more comprehensive summary of more applications and earlier papers. 

[Back to home](README.md)

## Table of Contents
- [SWE Agents and Agentic Post-training](#swe-agents-and-agentic-post-training)
  - [Table of Contents](#table-of-contents)
  - [General coding agents](#general-coding-agents)
  - [Issue resolving agents](#issue-resolving-agents)
  - [Debugging and test generation agents](#debugging-and-test-generation-agents)
  - [DevOps agents](#devops-agents)
  - [Agentic training](#agentic-training)
    - [Issue resolving](#issue-resolving)
    - [Test generation](#test-generation)
    - [Scaling up RL environment](#scaling-up-rl-environment)
      - [Scaling up data](#scaling-up-data)
      - [Automaticly build environment dependency](#automaticly-build-environment-dependency)
    - [Bug Trace for code agent](#bug-trace-for-code-agent)
    - [How model understand code](#how-model-understand-code)

## General coding agents

Below, we list some widely used coding agents that are mostly commercial products. These agents are mostly used for code development. They mainly use search, retrieval, file edit, and bash tools. They can also be connected to other software applications through MCP.

- ClaudeCode
- Cursor
- Github Copilot
- OpenHands: An Open Platform for AI Software Developers as Generalist Agents [[ICLR'24](https://arxiv.org/abs/2407.16741)]

## Issue resolving agents

Issue resolving is a typical SWE task on the development side. Below, we list and discuss some most representative works and top ranking works on SWE-bench. A more complete list can be found in this [document](https://docs.google.com/document/d/1UuoWGaVmk_W7tSGeVm0Eb4-Hi4HU1BJ9YLcPQ-fKrEE/edit?usp=sharing) as well as this [repo](https://github.com/codefuse-ai/Awesome-Code-LLM).

- SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering [[NeurIPS'24](https://arxiv.org/abs/2405.15793)]
  - Proposes an agent-computer interface between the LM agent and computer to facilitate tool use  
  - Offer a set of tools for file-related tasks with fewer options and shorter documents than bash commands (e.g., LM-friendly commands of viewing, searching through, and editing files, find_file(), scroll_up(), scroll_down(), open(), edit())
  - A simplified version: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)

- AutoCodeRover: Autonomous Program Improvement [[ISSTA'24](https://arxiv.org/abs/2404.05427)]
  - Early agents on issue resolving
  - Context retrieval agent: search for related functions in the AST; Use SBFL for root cause analysis based on path differences

- Agentless: Demystifying LLM-based Software Engineering Agents [[ASE'25](https://arxiv.org/abs/2407.01489)]
  - Have three sub-agents for localization, generation and validation, 
  - Localization is divided into a three-step procedure

- Moatless [[link](https://github.com/aorwall/moatless-tools)]
  - Agentic loop that functions as a finite state machine, transitioning between states. Each state can have its own prompts and response handling
    - Search&Identify: find and identify relevant code 
    - PlanToCode: task decomposition, breaking large code changes into smaller changes 
    - ClarifyChange: If the proposed changes affect too large a portion of the code, ask llm to reduce the amount of modifications.
    - EditCode: search/replace code block: generate replace block 
    Modularize and isolate the code that needs to be patched  

- PathPilot: A Cost-Efficient Software Engineering Agent with Early Attempts on Formal Verification [[ICML'25](https://arxiv.org/abs/2502.02747)]
  - Optimization over Agentless to improve resolve rate and cost efficiency
  - Early attempts on formal verification after generation

- UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench

## Debugging and test generation agents

- CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System [[Arxiv'25](https://arxiv.org/abs/2502.01821)]

- Agentic Bug Reproduction for Effective Automated Program Repair at Google [[Arxiv'25](https://arxiv.org/abs/2502.01821)]

- AEGIS: An Agent-based Framework for General Bug Reproduction from Issue Descriptions [[Arxiv'25](https://arxiv.org/abs/2411.18015)]

- TestForge: Feedback-Driven, Agentic Test Suite Generation [[Arxiv'25](https://arxiv.org/abs/2503.14713)]

## DevOps agents

- Enabling Autonomic Microservice Management through Self-Learning Agents [[Arxiv'25](https://arxiv.org/abs/2501.19056)]

- An LLM-based Agent for Reliable Docker Environment Configuration [[Arxiv'25](https://www.arxiv.org/pdf/2502.13681v2)]

## Agentic training 

Here, we list recent papers on training coding LLMs under agent scaffolds, including general coding models that integrate agentic data and specialized models for individual applications. Works about pure reasoning without agentic scaffolds are in this [page](reasoning.md).

### Issue resolving

- ***SWE-Swiss: A Multi-Task Fine-Tuning and RL Recipe for High-Performance Issue Resolution***

- ***DeepSWE: Training a Fully Open-sourced, State-of-the-Art Coding Agent by Scaling RL***

- ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models

- Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks

- SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution

- Thinking Longer, Not Larger: Enhancing Software Engineering Agents via Scaling Test-Time Compute

- SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution

- SWE-Gym: Training Software Engineering Agents and Verifiers with SWE-Gym

- Training Long-Context, Multi-Turn Software Engineering Agents with Reinforcement Learning

- Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards

- Co-PatcheR: Collaborative Software Patching with Component(s)-specific Small Reasoning Models

- Skywork-SWE: Unveiling Data Scaling Laws for Software Engineering in LLMs
  - New data creation technique

- Training Software Engineering Agents and Verifiers with SWE-Gym

### Test generation

- CURE: Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning
- CoCoEvo: Co-Evolution of Programs and Test Cases to Enhance Code Generation
- CodeContests+: High-Quality Test Case Generation for Competitive Programming
- Learning to Generate Unit Tests for Automated Debugging
- ASTER: Natural and Multi-language Unit Test Generation with LLMs
- Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation
- LLM-based Unit Test Generation via Property Retrieval
- Automatic Unit Test Data Generation and Actor-Critic Reinforcement Learning for Code Synthesis

Benchmarks:
- TestGenEval: A Real World Unit Test Generation and Test Completion Benchmark [[Arxiv'25](https://arxiv.org/abs/2410.00752)]
- On the Evaluation of Large Language Models in Unit Test Generation [[Arxiv'25](https://arxiv.org/abs/2406.18181)]
- SWT-Bench: Testing and Validating Real-World Bug-Fixes with Code Agents [[Arxiv'25](https://arxiv.org/abs/2406.12952)]

### Scaling up RL environment
#### Scaling up data
- Bottleneck: lack of large-scale, high-quality RL training data
  - Existing datasets: small (only a few thousand tasks).
  - Costly to build, heavily manual.
  - Poor scalability.

- SWE-smith: Scaling Data for Software Engineering Agents [[Arxiv'25](https://arxiv.org/abs/2504.21798)]
  - Given any Python repository:
    - Build executable environment.
    - Inject bugs into code.
    - Run tests, keep only those that fail.
    - Use an LLM to generate GitHub issue-style bug reports.

- SWE-Flow: Synthesizing Software Engineering Data in a Test-Driven Manner [[Arxiv'25](https://arxiv.org/abs/2506.09003)]

#### Automaticly build environment dependency
- EnvBench: A Benchmark for Automated Environment Setup [[Arxiv'25](https://arxiv.org/abs/2503.14443)]
  - 329 Python repos
  - 665 JVM repos (Java + Kotlin)
  - Chosen to include real challenges, not trivial setups.


- SWE-bench Goes Live! [[Arxiv'25](https://arxiv.org/html/2505.23419v1)]
  - REPOLAUNCH Automated Framework:
    - Data Collection: Extract issue-PR pairs from active GitHub repos
    - Env Setup: Parse README & CI configs, select base Docker image 
    - Dependency Installation & Test Execution
    - Validation: Task only valid if test goes from FAIL → PASS
    - Packaging: Release reproducible Docker environments for each task


### Bug Trace for code agent
- AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems? [[Arxiv'25](https://arxiv.org/abs/2509.03312)]
  - when the system fails → which agent and which step is responsible?
    - Counterfactual replay: replace actions with oracle guidance to find the decisive error step.
    - Fault injection: perturb successful trajectories to synthesize failures.
  - Qwen3-8B + multi-granular RL.


### How model understand code
- What Makes Large Language Models Reason in (Multi-Turn) Code Generation? [[Arxiv'25](https://arxiv.org/abs/2410.08105)]
- Beyond Syntax: How Do LLMs Understand Code? [[IEEE'25](https://ieeexplore.ieee.org/document/11023969)]
- How Does LLM Reasoning Work for Code? A Survey and a Call to Action [[Arxiv'25](https://arxiv.org/abs/2506.13932)]

