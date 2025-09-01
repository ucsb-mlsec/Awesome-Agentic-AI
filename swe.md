# SWE Agents and Agentic Post-training

In the following, we summarize the recent works in SWE agents and agentic training. Previously, we also summarize the works on pre-training and fine-tuning datasets, early-stage coding models, which can be found in this [[document](https://docs.google.com/document/d/1UuoWGaVmk_W7tSGeVm0Eb4-Hi4HU1BJ9YLcPQ-fKrEE/edit?usp=sharing)].
This [repo](https://github.com/codefuse-ai/Awesome-Code-LLM) has a more comprehensive summary of more applications and earlier papers. 

[Back to home](README.md)

## Table of Contents
- [General Coding Agents](#general-coding-agents)
- [Issue resolving Agents](#issue-resolving-agents)
- [Debugging and test generation Agents](#debugging-and-test-generation-agents)
- [DevOps Agents](#devops-agents)
- [Agentic training](#agentic-training)
  - [Issue resolving](#issue-resolving)
  - [Test generation](#test-generation)

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

### Test generation

- CURE: Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning
