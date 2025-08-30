# Awesome-Agentic-AI-for-Security

This repository contains recent papers and resources related to agentic AI in software security. 
We categorize the papers based on the problems they tackle, including vulnerability detection, triage, patching, others, and end-to-end frameworks. 
Under each category, we have techniques and benchmarks. Under each paper, we list the summary of key techniques and our key takeaways.  


## Table of Contents

- [Awesome-Agentic-AI-for-Security](#awesome-agentic-ai-for-security)
  - [Table of Contents](#table-of-contents)
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
  - [Citation](#citation)
  - [How to Contribute](#how-to-contribute)

[TODO: hongwei; you can add more icon to this]

## Survey

- Frontier AI's Impact on the Cybersecurity Landscape [[Arxiv'25](https://arxiv.org/abs/2504.05408)]
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
  - EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities [[Arxiv'24'](https://arxiv.org/abs/2409.16165)]
    - Agent with interactive tools and debuggers for CTF challenges
    - Agent does not use the tool feedback efficiently; The prompts are restricted to CTF
  - From Naptime to Big Sleep: Using Large Language Models To Catch Vulnerabilities In Real-World Code (Google Blog 2024)
    - Target on finding variants of previously found and patched vulnerabilities
    - Scan one commit at a time, collected recent commits to the SQLite repository, manually removing trivial and documentation-only changes and provide the agent with both the commit message and a diff for the change (and previous fixed vulns), let the agent review whether there is unfixed variants
    - Toolset: debugger_run (just run the project, without a debugger) and code_browser_source (search in the codebase) are used
- **LLM-based detection with additional information**
  - LLMxCPG: Context-Aware Vulnerability Detection Through Code Property Graph-Guided Large Language Models [ [USENIX'25](https://arxiv.org/abs/2507.16585)]
    - Use Code property graph together with LLMs for detection
    - [todo: is this paper useful?]
  - Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG [[Arxiv'24](https://arxiv.org/abs/2406.11147)]
    - RAG-based vulnerability detection with LLMs 
- **LLM-facilitated static analysis**
  - High-level ideas:
    - Given sources and sinks (potential vulns) and decide if there is a valid path between them
    - Taint analysis types of vulns: command injection, path traversals, SQL injection, null pointer dereference 
    - They leverage LLM’s capabilities in reasoning about program states and also conditions -> filter out wrong paths 
  - RepoAudit: An Autonomous LLM-Agent for Repository-Level Code Auditing [[ICML'25](https://arxiv.org/abs/2501.18160)]
    - For each kind of vulnerability, 
      - Use tree-sitter parser to find all possible sources (e.g, for npd, every place that ptr=null, return null or an attribute is set to null) and sinks (for npd, every place a ptr or field is about to deref)
      - For each source, find a path that can trigger the vulnerable states 
        - Use LLM to extract data-flow facts: starting from the source function and reason when the source condition will be satisfied
          - E.g., Following path 1 inside the func, a certain variable at line 3 will be null and the return value will be null,
        - If the data-flow facts propagate across function boundaries, retrieve relevant functions from the call graph, and extract data-flow facts for these functions
        - If a data-flow fact touches a sink, then consider that there is a potential path from the source to the sink
        - Use another LLM to collect the constraints along the whole propagation path, and see whether there are contradictory constraints; if not, report a vulnerability
      - LLM infers the data flow facts and analyzes the constraints
  - IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities [[ICLR'25](https://arxiv.org/abs/2405.17238)]
    - CodeQL-based taint analysis given by source/sinks: find a valid path in the data flow graph and then report 
    - LLM-based optimization codeql-based taint analysis: Use LLM to retrieve more source/sinks (Use static analysis and heuristics to retrieve candidates and let LLM check); Extract (1) all external APIs and (2) internal APIs that are public and may be invoked by a downstream library; After running CodeQL with the source/sinks, they get a path that triggers alerts. They filter false-positive paths using LLM
    - No tool calls; Target on Java
- **LLM-facilitated fuzzer**
  - Papers that use LLM to generate seeds or grammars 


[TODO: hongwei; fill in the rest]

### 📋 Benchmarks


## Vulnerability triage

### 🛠️ Techniques 

- PoC generation

- Root cause analysis

### 📋 Benchmarks

## Vulnerability patching

### 🛠️ Techniques 



### 📋 Benchmarks


## Others

### 🛠️ Techniques 

- Penn test

- CTF


- Binary analysis/reverse engineering 

### 📋 Benchmarks


## Citation

If you find this repo useful, please cite:

```bibtex
@misc{surfi2025agent,
   title={Recent Progresses on Agentic AI for Security},
   author={Hongwei Li, Zhun Wang, Wenbo Guo},
   year={2025},
   url={https://github.com/ucsb-mlsec/Awesome-Agentic-AI-for-Security}
}
```

## How to Contribute

For questions or collaboration, please contact the maintainers via GitHub Issues.

