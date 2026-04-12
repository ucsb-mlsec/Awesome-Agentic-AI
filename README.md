# Awesome-Agentic-AI

In this repo, we categorize and summarize recent works on agentic AI models, as well as agents for security and software engineering.  
Under each category, we list the summary of key techniques and our key takeaways.  

## [Agentic models and techniques](agent.md)

Here, we discuss general agentic models and training techniques. 

## [Security/SWE agents and agentic training](security.md)

This part focuses on security and SWE agents. We categorize the agents based on the problems they tackle, such as vulnerability detection, triage, patching, etc. 
Under each category, we include the corresponding benchmarks and techniques.

## [Environment, Inference, and Memory](env_inference.md)

This part covers three key infrastructure topics for agentic AI. **Environment simulation** discusses how to build scalable simulated environments (using LLMs or learned models) for agent training, reducing reliance on expensive real-world interactions. **Inference** focuses on efficient LLM serving for agentic workloads, including session-aware scheduling, KV cache reuse/management, and disaggregated serving optimizations that treat multi-turn agent trajectories as first-class execution units. **Memory management** surveys approaches for equipping agents with long-term memory, from structured knowledge graphs to agentic memory sub-systems that autonomously decide what to store, update, and retrieve.

## [General LLM reasoning](reasoning.md)

Finally, we provide an incomplete summary of recent works on general LLM reasoning (without involving agents). This direction has been extensively studied, with many techniques on improving RL-based post-training methods and tricks regarding the training process. Here, we mainly focus on the techniques that provide process reward signals, as well as specific training methods for code reasoning. 

## How to contribute

For questions or collaboration, please contact the maintainers via GitHub Issues.
