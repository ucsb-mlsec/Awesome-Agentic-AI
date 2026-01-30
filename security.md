# Security and SWE Agents

This repository contains recent papers and resources related to agentic AI in software engineering and security. 
We categorize the papers based on the problems they tackle. 
Under each category, we have techniques and benchmarks. Under each paper, we list the summary of key techniques and our key takeaways.  

[Back to home](README.md)


## Table of Contents
  - [General coding agents](#general-coding-agents)
  - [Agentic Training](#agentic-training)
  - [Survey](#survey)
  - [End-to-end frameworks](#end-to-end-frameworks)
  - [Vulnerability detection](#vulnerability-detection)
    - [🛠️ Techniques](#️-techniques)
    - [📋 Benchmarks](#-benchmarks)
  - [Vulnerability triage](#vulnerability-triage)
    - [🛠️ Techniques](#️-techniques-1)
    - [📋 Benchmarks](#-benchmarks-1)
  - [Vulnerability patching](#vulnerability-patching)
    - [🛠️ Techniques](#️-techniques-2)
    - [📋 Benchmarks](#-benchmarks-2)
  - [Others](#others)
    - [🛠️ Techniques](#️-techniques-3)
    - [📋 Benchmarks](#-benchmarks-3)
  - [SWE tasks](#swe-tasks)
    - [Issue resolving](#️issue-resolving)
    - [Test generation](#test-generation)

## General coding agents

Below, we list some widely used coding agents that are mostly commercial products. These agents are mostly used for code development. They mainly use search, retrieval, file edit, and bash tools. They can also be connected to other software applications through MCP.

- ClaudeCode
- Cursor
- Github Copilot
- OpenHands: An Open Platform for AI Software Developers as Generalist Agents [[ICLR'24](https://arxiv.org/abs/2407.16741)]


## Agentic Training

- Cyber-Zero: Training Cybersecurity Agents without Runtime [[Arxiv'25/08](https://www.arxiv.org/pdf/2508.00910)]
  - Use EnlGMA as the agent scaffold and target CTF benchmarks
  - Use LLM to simulate trajectories and SFT for training 

## Survey

- Frontier AI's Impact on the Cybersecurity Landscape [[Arxiv'25/04](https://arxiv.org/abs/2504.05408)]
  - Summarize recent works on agentic AI in security (including analysis on AIxCC CRSs)
  - Summarize recent benchmarks on AI for security
  - Provide concrete recommendations for future research
- Beyond Syntax: How Do LLMs Understand Code? [[IEEE'25](https://ieeexplore.ieee.org/document/11023969)]
- How Does LLM Reasoning Work for Code? A Survey and a Call to Action [[Arxiv'25](https://arxiv.org/abs/2506.13932)]

## End-to-end frameworks
- AIxCC frameworks
  - TeamAtlenta: [[Final code](https://github.com/Team-Atlanta/aixcc-afc-atlantis)] [[Semifinal code](https://github.com/Team-Atlanta/aixcc-asc-atlantis)]
    - Include multiple prompting strategy to increase the diversity of LLM-based fuzzing seed generation  
  - Shellphish: [[Final code](https://github.com/shellphish/artiphishell/releases/tag/Finals)] [[Semifinal code](https://github.com/shellphish/artiphishell/releases/tag/Semi-Finals)]
  - Theori: [[Code](https://theori-io.github.io/aixcc-public/index.html)]
  - Trail of Bits: [[Final code](https://github.com/trailofbits/afc-buttercup)] [[Semifinal code](https://github.com/trailofbits/asc-buttercup)]
  - All You Need IS A Fuzzing Brain: [[Final code](https://github.com/o2lab/afc-crs-all-you-need-is-a-fuzzing-brain)] [[Semifinal code](https://github.com/o2lab/asc-crs-all-you-need-is-a-fuzzing-brain)]
    - Simple pipeline, fuzzer (libfuzzer and llm-based poc gen) + llm-based patching
      - Multiple prompts and multiple models, cwe-specific prompts.
    - Poc Gen: Find all reachable functions from entrypoint (static analysis) & LLM-based ranking for reachable funcs (all reachable function concatenated as a huge prompt)
     - Patching
       - LLM-based patching location identification, prompt includes commit diff + crash log + history from poc gen (optional) + coverage(optional) + RAG
  - 42 b3yond 6ug: [[Final code](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs)] [[Semifinal code](https://github.com/42-b3yond-6ug/42-b3yond-6ug-asc)]
  - Lacrosse: [[Final code](https://github.com/siftech/afc-crs-lacrosse)] [[Semifinal code](https://github.com/siftech/asc-crs-lacrosse)]

    


## Vulnerability detection

### 🛠️ Techniques 

- **Agent system**
  - EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities [[ICML'25/07](https://arxiv.org/abs/2409.16165)]
    - Agent with interactive tools and debuggers for CTF challenges
    - Agent does not use the tool feedback efficiently; The prompts are restricted to CTF
  - From Naptime to Big Sleep: Using Large Language Models To Catch Vulnerabilities In Real-World Code (Google Blog 2024)
    - Target on finding variants of previously found and patched vulnerabilities
    - Scan one commit at a time, collected recent commits to the SQLite repository, manually removing trivial and documentation-only changes and provide the agent with both the commit message and a diff for the change (and previous fixed vulns), let the agent review whether there is unfixed variants
    - Toolset: debugger_run (just run the project, without a debugger) and code_browser_source (search in the codebase) are used
- **LLM-based detection with additional information**
  - LLMxCPG: Context-Aware Vulnerability Detection Through Code Property Graph-Guided Large Language Models [ [USENIX Security'25/07](https://arxiv.org/abs/2507.16585)]
    - Use Code property graph together with LLMs for detection
    - [todo: is this paper useful?]
  - Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG [[Arxiv'25/06](https://arxiv.org/abs/2406.11147)]
    - RAG-based vulnerability detection with LLMs 
- **LLM-facilitated static analysis**
  - High-level ideas:
    - Given sources and sinks (potential vulns) and decide if there is a valid path between them
    - Taint analysis types of vulns: command injection, path traversals, SQL injection, null pointer dereference 
    - They leverage LLM’s capabilities in reasoning about program states and also conditions -> filter out wrong paths
  - Automated Static Vulnerability Detection via a Holistic Neuro-symbolic Approach  [[arxiv'25/04](https://arxiv.org/abs/2504.16057)]
    - Use LLM to write CodeQL/Joern Queris, built a system to facilitate this, by 1) extract dsl doc, shrink it by only keeping basic features 2) instrument the query runtime and provide syntax and semenatic feedbacks
    - LLM should be able to write these queries by itself in the near future.
  - RepoAudit: An Autonomous LLM-Agent for Repository-Level Code Auditing [[ICML'25/07](https://arxiv.org/abs/2501.18160)]
    - For each kind of vulnerability, 
      - Use tree-sitter parser to find all possible sources (e.g, for npd, every place that ptr=null, return null or an attribute is set to null) and sinks (for npd, every place a ptr or field is about to deref)
      - For each source, find a path that can trigger the vulnerable states 
        - Use LLM to extract data-flow facts: starting from the source function and reason when the source condition will be satisfied
          - E.g., Following path 1 inside the func, a certain variable at line 3 will be null and the return value will be null,
        - If the data-flow facts propagate across function boundaries, retrieve relevant functions from the call graph, and extract data-flow facts for these functions
        - If a data-flow fact touches a sink, then consider that there is a potential path from the source to the sink
        - Use another LLM to collect the constraints along the whole propagation path, and see whether there are contradictory constraints; if not, report a vulnerability
      - LLM infers the data flow facts and analyzes the constraints
  - IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities [[ICLR'25/04](https://arxiv.org/abs/2405.17238)]
    - CodeQL-based taint analysis given by source/sinks: find a valid path in the data flow graph and then report 
    - LLM-based optimization codeql-based taint analysis: Use LLM to retrieve more source/sinks (Use static analysis and heuristics to retrieve candidates and let LLM check); Extract (1) all external APIs and (2) internal APIs that are public and may be invoked by a downstream library; After running CodeQL with the source/sinks, they get a path that triggers alerts. They filter false-positive paths using LLM
    - No tool calls; Target on Java
- **LLM-facilitated fuzzer**
  - Papers that use LLM to generate seeds or grammars 

### 📋 Benchmarks

- SecCodePLT: A Unified Platform for Evaluating the Security of Code GenAI [[ICML Workshop of MAS'25/07](https://icml.cc/virtual/2025/49367)]
- A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries (BigVul) [[IEEE/ACM MSR'20/05](https://dl.acm.org/doi/10.1145/3379597.3387501)]
    - 3,754 CVE data spanning 91 different vulnerability types extracted from 348 Github projects; Each CVE data contains the commit before and after fix, and also some metadata (CWE-ID, CVE-ID, program language, file changed, CVSS-related features)
- Vulnerability Detection with Code Language Models: How Far Are We? (Primevul) [[ICSE'25/04](https://arxiv.org/abs/2403.18624)]
    - Contains 6,968 vulnerable and 228,800 benign functions, covering 140 CWEs
    - Data collection and processing method:
        - Merge all security-related commits and functions changed by them from BigVul, CrossVul, CVEfixes, and DiverseVul
        - Normalization (by removing \n, \t, and \r) and deduplication. 
        - More accurate data labeling: label a function as vulnerable if
            - It’s the only changed function in a fixing commit
            - It’s explicitly mentioned in the CVE description
        - Temporal Splits: oldest 80% as the training set, 10% most recent as the testing set, and other samples as the validation set
- LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks (SecLLMHolmes) [[IEEE S&P'24/05](https://arxiv.org/abs/2312.12575)]
    - 228 code scenarios across 8 vulns for C and python with augmentation
    - Prompt templates: the set contains combinations of strategies including zero-shot/few shot, task-oriented (assign a task in the prompt), role-oriented (assign a role), step-by-step, definition-based (provide the definition of a vuln while asking the model to detect that vuln)
    - Metrics: Accuracy, cosine similarity, Gpt-4, Rouge score

## Vulnerability triage

### 🛠️ Techniques 

- **PoC generation**

  - Agentic Concolic Execution (S&P'26)
    - Use llm to perform basic-block level instrumentation
    - Has a corpus that contains interesting inputs, initially a random input, in each iteration
      - Pop one input from the corpus, use an LLM agent for collecting constraints
      - Execute the input, collect the trace, remove the unexecuted blocks, and ask an agent (with code retrieval and z3) to summarize the constraints of the path
      - LLM selects a branch to flip, and generate a new set of constraints
      - Use LLM agent to solve the constraint (with code retrieval and z3) and generate an input.
      - If the generated input yields new coverage, add the new input to the corpus

  - Claude-Sonnet-4-5-System-Card
    - Improvement on cybergym over openhands can be attributed to more flexible token constraints (a 200k token context window vs 2048 token output in openhands) and auto-summarization
    - Has editing tool and bash tool, in a kali virtual machine, with pwntools(for pwn challenges)、Metasploit(N-day vulns)、Ghidra(rev) and tshark(for cybench web challenge)
    - asynchronous management of multiple terminal sessions
    - Refinement is useful

  - FaultLine: Automated Proof-of-Vulnerability Generation Using LLM Agents [[arxiv'25/07](https://arxiv.org/abs/2507.15241)]
      - It is a good agent with slicing and fixed workflow

  - FalseCrashReducer: Mitigating False Positive Crashes in OSS-Fuzz-Gen Using Agentic AI (arxiv 25)
    - LLM-based fuzz driver generator of oss-fuzz is not good enough that the fuzz driver generates infeasible inputs to a target function for fuzz.
    - Has one agent to extract constraints of inputs to the target function, with grep and function search tool (given func name return func implementation), the constraint is used to generate fuzz driver
    - Has another agent to validate whether the crash is triggerable from program entry point

  - LLM Agents can Autonomously Exploit One-day Vulnerabilities [[arxiv'24/04](https://arxiv.org/abs/2404.08144)]
      - Tools: Web browsing elements (retrieving HTML, clicking on elements, etc.), A terminal, Web search results, File creation and editing, A code interpreter
      - All web vulns, xss, rce, csrf
      - Tested on 15 hand-picked CVEs
          - When providing vulnerability description and documentation of the target, 87% success rate
          - When removing vulnerability description, 7% success rate
  - Teams of LLM Agents can Exploit Zero-Day Vulnerabilities [[arxiv'25/03](https://arxiv.org/abs/2406.01637)]
      - Has 3 kinds of agents, a hierarchical planner which explores the environment and generates a plan, a team manager which determines which expert agent to use for a specific page, and some expert agents, each for one vuln type (SQLi, xss, csrf, …)
      - Tools in common for expert agents
          - Playwright (a browser testing framework to access the websites), the terminal, and file management tools
          - Each expert agent has specific tools for the specific vuln
              - E.g., SQLi agent has access to sqlmap
              - Zap agent has access to zap (a scanner for xss, csrf, ..)
  - On the Feasibility of Using LLMs to Autonomously Execute Multi-host Network Attacks [[arxiv'25/05](https://arxiv.org/abs/2501.16466)]

  - General agents
      - OpenHands: An Open Platform for AI Software Developers as Generalist Agents [[ICLR'25/04](https://arxiv.org/abs/2407.16741)]
      - Evaluating Large Language Models Trained on Code (Codex of openai) [[arxiv'21/07](https://arxiv.org/abs/2107.03374)]
  - CTF agents
      - Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models [[ICLR'25/04](https://arxiv.org/abs/2408.08926)]
      - NYU CTF Bench: A Scalable Open-Source Benchmark Dataset for Evaluating LLMs in Offensive Security [[NeurIPS'24/12](https://arxiv.org/abs/2406.05590)]

- **Root cause analysis**

  - Enhancing Fault Localization Through Ordered Code Analysis with LLM Agents and Self-Reflection [[arxiv'25/03](https://www.arxiv.org/abs/2409.13642v1)]
    - Target on Java projects, fault localization on the method level.
    - Has 3 agents:
        - Context Extraction Agent: First run GZoltar and use SBFL to rank all covered methods by the crashing testcase. Then group all methods into subgroups, each subgroup within the input context length of the LLM. Then provide the summarized failed test and stack trace, and use LLM to find the most likely root cause method of each group. No tool call for this step.
        - Debugger Agent: Construct a call graph, provide LLM with the concatenated most likely root cause method of each group, give LLM tools of get_callers, get_callees, and get_method body, let LLM rank all provided methods.
        - Reviewer Agent: let LLM critique, provide get_callgraph, get_callers, get_callees, and get_method body.
    - Lacking tools and only using simple tools
  - **Agentic Bug Reproduction for Effective Automated Program Repair at Google [[Arxiv'25](https://arxiv.org/abs/2502.01821)]**
    - Agent Scaffold
      - React style
      - Two LLMs, one for planning and general tool calling, one for code-editing.
      - Input: Bug report + Identified buggy file content
      - Tools: cat, code_search, bazel test, finish, edit
        - The edit tool will call another code-editing LLM, with bug report, file content, and change description as prompt. The code-editing LLM will generate a patch which will be auto applied.
      - How the models are fine-tuned are not mentioned. Only mentioned fine-tuned on Google’s internal code.
      - Used a google internal benchmark, GITS-Eval with 178 bugs.
  - **AEGIS: An Agent-based Framework for General Bug Reproduction from Issue Descriptions [[Arxiv'25](https://arxiv.org/abs/2411.18015)]**
    - Agent scaffold:
      - Search agent
        - trivial agent with general retrieval tools, e.g., search_class, review_file, ls, grep, etc.
      - Bug reproduction agent
        - Concise Context Construction Module: Input: bug report + localization result by Searcher agent (containing irrelevant contents), output: structured issue information (issue description + existing reproduction method, current res, expected res), relevant code snippets + explanation of why code snippets are relevant
        - Defined a Finite State Machine-based workflow, including 7 states, e.g., Create, Execute, Verify. etc, enforce rules that one state can only transfer to some specific states. Restrictions are on system prompt and tools.
  - **Locagent: Graph-guided LLM Agents for Code Localization [[Arxiv'25/03](https://arxiv.org/abs/2503.09089)]**
    - Build a code graph G(V,E,A,R) using python ast to support graph-based retrieval, where
      - V = {vᵢ}, nodes
      - E ⊆ V × V, edges,
      - For nodes, τ (v) ∶ V → A maps to types A = {directory, file, class, function}
      - For edges, ϕ(e) ∶ E → R maps to relationships R = {contain, import, invoke, inherit}
    - Tools:
      - SearchEntity: input is a keyword, output is an entity id and code snippet
        - Support exact match (filepath+func name) and three levels of fuzzy match (all func with same name, BM25, code content inverted index) 
      - TraverseGraph: performs BFS uder rules specified by LLM. Input is a starting entity and some rules provided by LLM, e.g., number of hops, only consider what type of nodes, only consider what type of edges. output is a subgraph.
      - RetrieveEntity: input is an entity id, output is all attributes of an entity, including filepath, linenum, code, other metadata.
      - Constructed a LOC-BENCH, including github issues after 2024.10, including bug report, feature requests and security and performance.


### 📋 Benchmarks
- CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale [[Arxiv'25/08](https://arxiv.org/abs/2506.02548)]
- BountyBench: Dollar Impact of AI Agent Attackers and Defenders on Real-World Cybersecurity Systems [[arxiv'25/07](https://bountybench.github.io/)]

## Vulnerability patching

### 🛠️ Techniques 
- A Case Study of LLM for Automated Vulnerability Repair: Assessing Impact of Reasoning and Patch Validation Feedback [[ACM AIware'24/07](https://dl.acm.org/doi/10.1145/3664646.3664770)]
    - Human-based planning
    - Procedure
        - Provide LLM with a code snippet and exact vulnerable lines, prompt llm to generate a patch
        - Run compile, poc, and functionality test
        - If any of them fail, retry with the compiler/sanitizer/unit test output as feedback
- RepairAgent: An Autonomous, LLM-Based Agent for Program Repair [[arxiv'24/10](https://arxiv.org/abs/2403.17134)]
    - LLM-based planning, but it is a state machine
    - Tools
        - Localization: search_code_base, find_similar_api_calls,read_range, get_classes_and_methods, extract_method, Run_fault_localization (use GZoltar for SBFL)
        - Generation: Generate_method_body, write_fix
        - Execution tools: Run_tests/Extract_tests
- APPATCH: Automated Adaptive Prompting Large Language Models for Real-World Software Vulnerability Patching (USENIX 2025) [[USENIX Security'25/08](https://www.usenix.org/conference/usenixsecurity25/presentation/nong)]
    - A fixed pipeline: Summarize the vulnerability semantics and do root cause analysis (LLM can request context as needed), match existing knowledge base, generate a patch, and validate the patch with an LLM judge
        - Construct a static knowledge base
    - Tools: SPG (control flow graph + data dependency graph)
- PATCHAGENT: A Practical Program Repair Agent Mimicking Human Expertise [[USENIX Security'25/08](https://www.usenix.org/conference/usenixsecurity25/presentation/yu-zheng)]
    - Human-based planning with specific prompts and a tool combo. 
        - Report Purification: reformat the sanitizer report to a more structured version
        - Chain Compression: used when retrieving contexts, has two mechanisms
            - When llm view a piece of code, sample symbols that are near the lines in the sanitizer report, and get their definition
            - When getting a definition of a symbol, recursively retrieve info.
                - E.g., 1) the definition of “info” is in line 100-> 2) get the code snippet around line 100 -> 3)in line 100, “info” is defined as a return value of a function, so get the definition of the function
            - Autocorelation
                - Fix line numbers by heuristics
            - Counterexample Feedback
                - Sample failed patches and instructing the LLM not to generate similar patches again

- AutoSafeCoder: A Multi-Agent Framework for Securing LLM Code Generation through Static Analysis and Fuzz Testing [[Arxiv'24](https://arxiv.org/abs/2409.10737)]

### 📋 Benchmarks

- CVE-Bench: Benchmarking LLM-based Software Engineering Agent’s Ability to Repair Real-World CVE Vulnerabilities [[NAACL'25/04](https://aclanthology.org/2025.naacl-long.212/)]
- SEC-bench: Automated Benchmarking of LLM Agents on Real-World Software Security Tasks [[arxiv'25/06](https://arxiv.org/abs/2506.11791)]
- Introducing AutoPatchBench: A Benchmark for AI-Powered Security Fixes (Meta AI) [[blog'25/04](https://engineering.fb.com/2025/04/29/ai-research/autopatchbench-benchmark-ai-powered-security-fixes/)]

## Others

### 🛠️ Techniques 

- **Penn test**

  - PentestGPT: Evaluating and Harnessing Large Language Models for Automated Penetration Testing [[USENIX Security'24/08](https://www.usenix.org/conference/usenixsecurity24/presentation/deng)]
  - On the Feasibility of Using LLMs to Autonomously Execute Multi-host Network Attacks [[arxiv'25/05](https://arxiv.org/abs/2501.16466)]
  - From Sands to Mansions: Towards Automated Cyberattack Emulation with Classical Planning and Large Language Models [[arxiv'25/04](https://arxiv.org/abs/2407.16928)]
  - VulnBot: Autonomous Penetration Testing for A Multi-Agent Collaborative Framework [[arxiv'25/01](https://arxiv.org/abs/2501.13411)]
  - AutoAttacker: A Large Language Model Guided System to Implement Automatic Cyber-attacks [[arxiv'24/03](https://arxiv.org/abs/2403.01038)]

- **CTF**

  - D-CIPHER: Dynamic Collaborative Intelligent Multi-Agent System with Planner and Heterogeneous Executors for Offensive Security [[arxiv'25/05](https://arxiv.org/abs/2502.10931)]
  - EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities [[ICML'25/07](https://arxiv.org/abs/2409.16165)]
  - Measuring and Augmenting Large Language Models for Solving Capture-the-Flag Challenges [[ACM CCS'25/10](https://arxiv.org/abs/2506.17644)]
      - CTF QA benchmark
      - Propose an agent with RAG and terminal tools

- **Binary analysis/reverse engineering**

  - FidelityGPT: Correcting Decompilation Distortions with Retrieval-Augmented Generation[[[NDSS'26](https://arxiv.org/abs/2510.19615)]
  - FoC: Figure out the Cryptographic Functions in Stripped Binaries with LLMs [[ACM Transactions on Software Engineering and Methodology'25/04](https://dl.acm.org/doi/10.1145/3731449)]
  - Large Language Models for Code Analysis: Do LLMs Really Do Their Job? [[USENIX Security'24/08](https://arxiv.org/abs/2310.12357)]

### 📋 Benchmarks

- SECURE: Benchmarking Large Language Models for Cybersecurity [[IEEE ACSAC'24/12](https://www.computer.org/csdl/proceedings-article/acsac/2024/208800a015/25bv7zxqzyo)]
- CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence [[NeurIPS'24/12](https://arxiv.org/abs/2406.07599)]

## SWE tasks

### Benchmarks

- TerminalBench [[link](https://www.tbench.ai/)]
  - Tier 1: Infrastructure & Core Systems: 
    - Software Build & CompilationGoal: Build from source, fix compilation errors, and handle complex dependencies.
      - Skills: make, gcc, cmake, cython, rustc, version compatibility debugging.
      - Typical Tasks: build-linux-kernel-qemu, build-cython-ext, compile-compcert, magsac-install.
    - System Administration & DevOpsGoal: Configure services, manage environments, and debug system-level issues.
      - Skills: nginx, ssh, git server, docker, qemu, conda/npm dependency management, cron.
      - Typical Tasks: configure-git-webserver, home-server-https, conda-env-conflict-resolution, broken-networking.
    - Security, Reverse Engineering & ForensicsGoal: Exploit vulnerabilities, patch bugs, reverse engineer binaries/protocols, and recover deleted or corrupted data.
      - Skills: SQL Injection, RCE, password cracking, git history analysis, binary analysis, gdb, file system forensics.
      - Typical Tasks: sql-injection-attack, security-celery-redis-rce, git-leak-recovery, db-wal-recovery.
  - Tier 2: Data & Algorithm Applications
    - Data Processing & ETLGoal: Clean, transform, and aggregate data from various sources and formats.
      - Skills: pandas, duckdb, jq, awk/sed/grep, SQL, file format conversion (CSV, JSON, Parquet).
      - Typical Tasks: multi-source-data-merger, pandas-sql-query, log-summary-date-ranges, jq-data-processing.
    - Machine Learning & MLOpsGoal: Implement, train, evaluate, debug, and deploy machine learning models.
      - Skills: pytorch, huggingface transformers/peft, scikit-learn, LoRA, model parallelism, MLOps tools (mlflow, mteb).
      - Typical Tasks: hf-train-lora-adapter, torch-pipeline-parallelism, classifier-debug, sam-cell-seg.
    - Algorithms & Logic PuzzlesGoal: Solve self-contained puzzles that require algorithmic or logical reasoning.
      - Skills: Search algorithms (BFS), constraint satisfaction, mathematical modeling, image recognition (for Sudoku, chess).
      - Typical Tasks: huarong-dao-solver, assign-seats, solve-sudoku, code-from-image.
  - Tier 3: Specialized Domains & Advanced Development
    - Software Development, Porting & Bug FixingGoal: Develop new features, port code between languages/frameworks, and fix bugs in real-world open-source libraries.
      - Skills: Multi-language programming (Python, C++, Rust), web frameworks (Flask), API design, code refactoring (MATLAB->Python, C->Rust).
      - Typical Tasks: swe-bench-* (series), port-compressor (C to Safe Rust), cobol-modernization, solana-data (API dev).
    - Scientific & Domain-Specific ComputingGoal: Solve problems in specific scientific fields like biology, chemistry, physics, or statistics.
      - Skills: R, Stan (statistics), rdkit (chemistry), mujoco (physics), bioinformatics algorithms.
      - Typical Tasks: dna-assembly, rstan-to-pystan, fmri-encoding-r, rare-mineral-allocation.
    - Interactive Environments & GamesGoal: Complete tasks by interacting in multiple steps with a running process, API, or game.
      - Skills: Process I/O redirection, socket programming, curl (REST API), game strategy.
      - Typical Tasks: blind-maze-explorer-5x5, interactive-maze-game, find-restaurant, play-zork.
    - Distributed & Parallel ComputingGoal: Leverage multi-core or multi-node capabilities to accelerate computation.
      - Skills: OpenMP, MPI, multiprocessing, Spark, Hadoop.
      - Typical Tasks: parallel-particle-simulator, torch-tensor-parallelism, predicate-pushdown-bench.
    - Formal Verification & GraphicsGoal: Prove theorems using formal methods, or perform 3D rendering and image processing.
      - Skills: Coq, Lean, SAT/SMT solvers, pyrender, osmesa, path tracing algorithms.
      - Typical Tasks: lean4-proof, weighted-max-sat-solver, path-tracing, unprivileged-headless-pyrender.
- SWE-Gym: Training Software Engineering Agents and Verifiers with SWE-Gym
- What Makes Large Language Models Reason in (Multi-Turn) Code Generation? [[Arxiv'25](https://arxiv.org/abs/2410.08105)]
  - Not a benchmark but studies the impact of different prompting strategies on multi-turn code generation tasks & fine-tuning with the good prompting strategies

### DevOps agents

- Enabling Autonomic Microservice Management through Self-Learning Agents [[Arxiv'25](https://arxiv.org/abs/2501.19056)]

- An LLM-based Agent for Reliable Docker Environment Configuration [[Arxiv'25](https://www.arxiv.org/pdf/2502.13681v2)]


### Issue resolving

Issue resolving is a typical SWE task on the development side. Below, we list and discuss some most representative works and top ranking works on SWE-bench. A more complete list can be found in this [document](https://docs.google.com/document/d/1UuoWGaVmk_W7tSGeVm0Eb4-Hi4HU1BJ9YLcPQ-fKrEE/edit?usp=sharing) as well as this [repo](https://github.com/codefuse-ai/Awesome-Code-LLM).

- **SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent [[Arxiv'25/11](https://arxiv.org/abs/2511.16108)]**
  - Training recipe for SA-SWE-32B
  - Motivation: Error localization is a key bottleneck; agents over-rely on viewing files instead of leveraging search utilities; multi-turn RL suffers from repetitive/unproductive behaviors
  - Bootstrap training with better tools: AST-based search tool supporting fuzzy matching and structural pattern search
  - RL: Fully on-policy setup; mask out trajectories exceeding max context length or step limit during gradient updates
  - Use leave-one-out advantage estimation; remove standard deviation and length normalization in advantage computation
  - Disable KL and entropy loss in RL training
  - Add hints: structured cues to help agents recover from failed actions (suggestions on tool failure, notifications about remaining budget/context, corrections for invalid function calls)

- **SWE-Swiss: A Multi-Task Fine-Tuning and RL Recipe for High-Performance Issue Resolution[[Blog](https://www.notion.so/SWE-Swiss-A-Multi-Task-Fine-Tuning-and-RL-Recipe-for-High-Performance-Issue-Resolution-21e174dedd4880ea829ed4c861c44f88#245174dedd488067a9e7eea04315dad5)]**
  - Agent framework
    - Localize, input: issue description + repo's file structure, output: predicted relevant files
    - Repair, input: predicted files + files retrieved via text-embedding-3-small, output: a candidate patch
    - Validation, input: patches + existing regression test, process: first filter out patches that do not pass existing regression test, then generate multiple unit tests and select one genearted test by majority voting, then use the final test to filter out patches and do majority voting for patches.
  - Recipe: first SFT for localization, repair, and unit test gen, then GRPO for repair
    - SFT
      - Rejection sampling for each task
      - teacher model: DeepSeek-R1-0528, generates reasoning data, base model: Qwen2.5-32B-Instruct, jointly trained on three tasks, 36.0% pass@1 on SWE-bench Verified
        - Localization:
          - data: SWE-bench and SWE-Gym-Raw training set
          - input: issue description + repo's file structure, output: files that need to be changed
          - only keep recall=1.0 and predicted file num <=5, 5.3K data
        - Repair:
          - data: SWE-bench and SWE-smith
          - input: issue description + ground-truth files +  distractor files (incorrect files predicted by an LLM to make the task more challenging), output: patch
          - only keep patches that can pass all tests, 3.9K data
        - Unit test gen
          - data: SWE-bench and SWE-smith
          - input: issue description, output: unit tests
          - only keeps unit tests that fail before patching and pass after patching, 1K data
    - GRPO
      - 45.0 pass@1 on SWE-bench Verified
      - Input:  issue descriptions + relevant files, output: patch
      - GRPO + no KL loss, clip higher, dynamic sampling, and token-level policy gradient loss from DAPO
      - Reward: outcome reward, 1 if patch passed all unit tests, -1 if not
      - Two stages: stage 1 uses all tasks, includes 200 steps; stage 2 removes tasks that the model has already achieved over 90% accuracy, 90 steps

- **Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks [[ICLR'25/06](https://arxiv.org/pdf/2505.16901)]**
  - Human-based planning+RAG, Qwen2.5-72B, 43% on SWE-bench Lite
  - Build the repo as a graph (doesn't handle indirect calls):
    - Nodes: repo, package, file, textfile, class, function, attribute; Edges: contains, calls, extends, imports, implements(only for Java, a class implements an interface)
  - Agent scaffold:
    - Rewriter: extract code entities and keywords, rewrite the issue description to queries. E.g., file name containing ‘separable.py’
    - Retriever: find the relevant nodes from the code graph, and then expand the context by including one-hop neighbors and upstream nodes, form a subgraph
    - Reranker: Rank the files in subgraph by file name and file skeleton, output top-5 files
    - Patcher: Fine-tuned model based on Qwen2.5-72B, input: issue description + subgraph + top-5 file content, output: patch
      - Representation of the subgraph:
        - Each node is represented as one token. First the code in node are splited by 512-token chunks, then an encoder CodeT5+ produce embedding vectors, these embedding vectors are projected to one token using a 2 layer MLP.
        - The attention mask is constructed from the adjacency matrix to reflect the graph structure. E.g., if there is an edge from node U to node G, then attention_mask[U][G]=1
  - Recipe:
    - Train CodeT5+(lora), 2 layer MLP, and Qwen2.5-72B(lora) at same time
    - Two phases
      - Subgraph Reconstruction: input: randomly sampled subgraphs from repo, output: code
      - Issue resolving: input: subgraph+issue description+top-5 file content, output: patch
        - include noise:  10% include an irrelevant file, another 10% omit at least one ground-truth file
  - Ablation:
    -   a naive graph-based baseline which flattens code snippets based on topological structure achieved 5.33% while the original one achieved 37.67%.

- Co-PatcheR: Collaborative Software Patching with Component(s)-specific Small Reasoning Models
  - SFT and 46% on SWE-bench-Verified
  - Some other earlier works 
    - SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution
    - Thinking Longer, Not Larger: Enhancing Software Engineering Agents via Scaling Test-Time Compute
    - SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
    - Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards

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


### Test generation

- CURE: Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning
- CoCoEvo: Co-Evolution of Programs and Test Cases to Enhance Code Generation
- CodeContests+: High-Quality Test Case Generation for Competitive Programming
- Learning to Generate Unit Tests for Automated Debugging
- ASTER: Natural and Multi-language Unit Test Generation with LLMs
- Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation
- TestForge: Feedback-Driven, Agentic Test Suite Generation [[Arxiv'25](https://arxiv.org/abs/2503.14713)]
  - A test generation agent with feedback from coverage tool and a test oracle



