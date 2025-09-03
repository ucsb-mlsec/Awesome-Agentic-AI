# Security Agents

This repository contains recent papers and resources related to agentic AI in software security. 
We categorize the papers based on the problems they tackle, including vulnerability detection, triage, patching, others, and end-to-end frameworks. 
Under each category, we have techniques and benchmarks. Under each paper, we list the summary of key techniques and our key takeaways.  

[Back to home](README.md)


## Table of Contents

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

## Agentic Training

- Cyber-Zero: TRAINING CYBERSECURITY AGENTS WITHOUT RUNTIME ([Arxiv'25/08](https://www.arxiv.org/pdf/2508.00910)]


## Survey

- Frontier AI's Impact on the Cybersecurity Landscape [[Arxiv'25/04](https://arxiv.org/abs/2504.05408)]
  - Summarize recent works on agentic AI in security (including analysis on AIxCC CRSs)
  - Summarize recent benchmarks on AI for security
  - Provide concrete recommendations for future research

## End-to-end frameworks
- AIxCC frameworks [TODO hongwei add theori, shellphish?] 
  - TeamAtlenta: [[Code](https://github.com/Team-Atlanta/aixcc-afc-atlantis)]
    - Include multiple prompting strategy to increase the diversity of LLM-based fuzzing seed generation  

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
- **Claim to be an agent**
    - Autonomously Uncovering and Fixing a Hidden Vulnerability in SQLite3 with an LLM-Based System (team Atlanta in AIXCC) [[blog'24/08](https://team-atlanta.github.io/blog/post-asc-sqlite/)]
        - Not much information
        - “Incorporate traditional program analysis techniques (both dynamic and static) to assist LLMs in decision-making”
        - “distilled our collective experience and common practices in manual auditing and reverse engineering into structured prompts, significantly enhancing the system’s capabilities”
    - From Naptime to Big Sleep: Using Large Language Models To Catch Vulnerabilities In Real-World Code (Google) [[blog'24/11](https://googleprojectzero.blogspot.com/2024/10/from-naptime-to-big-sleep.html)]
        - Target on finding variants of previously found and patched vulnerabilities
        - Scan one commit at a time
            - Collected a number of recent commits to the SQLite repository, manually removing trivial and documentation-only changes
            - Provide the agent with both the commit message and a diff for the change (and previous fixed vulns), let the agent review whether there is unfixed variants
        - Did not provide the tool set, only know that debugger_run (just run the project, without a debugger) and code_browser_source (search in the codebase) are provided

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
    - 228 code scenarios across 8 vulns for C and python with **augmentation**
        - Trivial augmentation (84): mutate vuln codes from hand-crafted code scenarios by randomly renaming funcs/params, adding unreachable code, inserting whitespace or \n, and adding comments
        - Non-trivial augmentation (66): mutate vuln and benign codes from hand-crafted code scenarios by changing func/param names to vulnerability-related keywords or `non_vulnerable`, adding a potentially dangerous library function (e.g., ‘strcpy’ or ‘strcat’) but use it in a safe way
    - Prompt templates: the set contains combinations of strategies including zero-shot/few shot, task-oriented (assign a task in the prompt), role-oriented (assign a role), step-by-step, definition-based (provide the definition of a vuln while asking the model to detect that vuln)
    - Metrics: Accuracy, cosine similarity, Gpt-4, Rouge score

## Vulnerability triage

### 🛠️ Techniques 

- **PoC generation**

  - LLM with simple tools (non-PL tools)
      - LLM Agents can Autonomously Exploit One-day Vulnerabilities [[arxiv'24/04](https://arxiv.org/abs/2404.08144)]
          - Tools:
              - Web browsing elements (retrieving HTML, clicking on elements, etc.)
              - A terminal 
              - Web search results 
              - File creation and editing 
              - A code interpreter.
          - Did not disclose the prompt
          - All web vulns, xss, rce, csrf
          - Tested on 15 hand-picked CVEs.
              - When providing vulnerability description and documentation of the target, 87% success rate
              - When removing vulnerability description, 7% success rate
      - Teams of LLM Agents can Exploit Zero-Day Vulnerabilities [[arxiv'25/03](https://arxiv.org/abs/2406.01637)]
          - Target on web vulns
          - Has 3 kinds of agents, a hierarchical planner which explores the environment and generates a plan, a team manager which determines which expert agent to use for a specific page, and some expert agents, each for one vuln type (SQLi, xss, csrf, …)
          - Tools in common for expert agents
              - Playwright (a browser testing framework to access the websites), the terminal, and file management tools
              - Each expert agent has specific tools for the specific vuln
                  - E.g., SQLi agent has access to sqlmap
                  - Zap agent has access to zap (a scanner for xss, csrf, ..)
  - FaultLine: Automated Proof-of-Vulnerability Generation Using LLM Agents [[arxiv'25/07](https://arxiv.org/abs/2507.15241)]
      - It is a good agent with slicing and fixed workflow
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
  - We have a debugger in our AIxCC

### 📋 Benchmarks

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

  - FoC: Figure out the Cryptographic Functions in Stripped Binaries with LLMs [[ACM Transactions on Software Engineering and Methodology'25/04](https://dl.acm.org/doi/10.1145/3731449)]
  - Large Language Models for Code Analysis: Do LLMs Really Do Their Job? [[USENIX Security'24/08](https://arxiv.org/abs/2310.12357)]

### 📋 Benchmarks

- SECURE: Benchmarking Large Language Models for Cybersecurity [[IEEE ACSAC'24/12](https://www.computer.org/csdl/proceedings-article/acsac/2024/208800a015/25bv7zxqzyo)]
- CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence [[NeurIPS'24/12](https://arxiv.org/abs/2406.07599)]
