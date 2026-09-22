# Formal Verification and AI

## Table of Contents

- [Development trajectory](#development-trajectory)
- [Code](#code)
  - [Smart-contract application map](#smart-contract-application-map)
  - [Benchmark](#code-benchmark)
  - [Training](#code-training)
  - [Agent](#code-agent)
- [Math](#math)
  - [Benchmark](#math-benchmark)
  - [Training](#math-training)
  - [Agent](#math-agent)
- [Hardware](#hardware)
  - [Benchmark](#hardware-benchmark)
  - [Training](#hardware-training)
  - [Agent](#hardware-agent)

## Development trajectory

The common workflow is **propose an artifact → check it with a formal tool → use feedback to revise the artifact or train the proposer**. What changes across the three domains is the artifact, the checker, and the source of missing information.

| Domain | Main artifact | Development of the approach | Remaining bottleneck |
| --- | --- | --- | --- |
| Code | Contracts, invariants, implementations, proof scripts, protocol models | Learned hints → LLM generation and repair → verifier-supervised training → module and repository verification | Faithful specifications, cross-function composition, trusted dependencies |
| Math | Formal statements, tactics, complete proofs, auxiliary lemmas | Tactic search and retrieval → informal-to-formal decomposition → RL and self-play → agents operating over proof libraries | Long proof planning, useful lemma discovery, statement fidelity |
| Hardware | SVA properties, counterexample explanations, solver heuristics | Learned SAT/PDR guidance alongside NL-to-SVA → design-aware assertions → causal debugging and solver evolution | Temporal semantics, signal grounding, useful property coverage, solver cost |

Three distinctions matter when reading the list:

- **Proof validity versus specification fidelity**: a checked proof establishes the encoded statement; it does not establish that an LLM encoded the intended requirement correctly. Verus-SpecGym, FormalAlign, and FVEval target different versions of this gap.
- **Training versus inference**: SAFE and PSV update model weights; AutoVerus and AlphaVerus improve generation through orchestration, search, and examples. Both can use verifier feedback, but their learning mechanisms differ.
- **Local success versus system completion**: completing a supplied lemma or function proof is different from designing a module abstraction, discovering all required contracts, or verifying a production library. DafnyCOMP, VeriStruct, and CryptoProver progressively expose these additional responsibilities.

## Code

**Research trajectory.** The early thread learns individual proof decisions: invariant candidates, identifier representations, or which proof state to explore next (DistAI, Cider, Passport, QEDCartographer). The LLM thread broadens the generated artifact to complete specifications and proofs, then closes the loop with verifier errors, counterexamples, and symbolic repair (Lemur, AutoSpec, AutoVerus, ExVerus). A parallel training thread turns accepted artifacts into new supervision and eventually generates its own tasks (SAFE, Re:Form, SpecRL, PSV, Formal Disco). Systems work then moves beyond isolated obligations to shared module invariants, repository context, and internal specifications for production code (Rango, VeruSAGE, VeriStruct, CryptoProver).

| Stage | Question being addressed | Representative work |
| --- | --- | --- |
| Learned guidance, 2021–2023 onward | Can learning reduce the search for invariants and proof steps? | DistAI, Cider, Passport, iRank |
| Generation with checking, 2023–2025 onward | Can an LLM propose useful artifacts while a verifier controls acceptance? | Clover, Lemur, AutoSpec, AutoVerus, AlphaVerus |
| Verifier-driven training, 2024–2026 | Can successful proofs, failed attempts, and generated tasks supply scalable supervision? | SAFE, DafnySynth, SpecRL, PSV, Formal Disco |
| Composition and systems, 2025–2026 | Can agents manage dependencies, abstractions, internal contracts, and long proof workflows? | Rango, DafnyCOMP, VeriStruct, AutoRocq, CryptoProver |
| Specification fidelity, across stages | Did the agent prove the intended behavior, or merely a weak/mistranslated statement? | nl2postcond, VeriEquivBench, OSVBench, Verus-SpecGym, ARTEMIS |

Security applications add a related path: **learn or extract a protocol model → generate security properties → check for violations → validate the resulting attack or bug**. MPInspector and Hermes emphasize model extraction; PropertyGPT emphasizes properties; VeriExploit emphasizes executable reproduction after a counterexample is already available.

<a id="smart-contract-application-map"></a>

### Smart-contract application map

The early emphasis on learned invariants and property generation now extends to checking recovered code, synthesizing implementations with proofs, and validating model-generated violation witnesses. These are complementary applications across the contract lifecycle.

| Application | Concrete task | Representative work | Verification boundary |
| --- | --- | --- | --- |
| Invariant inference | Learn persistent arithmetic or business-logic constraints | Cider, SmartInv — Training | Candidate invariants need independent checking |
| Specification and compliance | Derive contract properties or executable ERC rules | PropertyGPT, SymGPT — Agent | Guarantees depend on the chosen properties and rule translation |
| Decompilation | Recover readable semantics without changing analyzed behavior | SmartHalo — Agent | Equivalence to decompiler output does not certify the original decompiler |
| Contract synthesis and repair | Combine code generation, proof attempts, and adversarial feedback | LeVer — Agent | Proofs concern a translated model; tests cover observed executions |
| Violation witnesses | Turn a counterexample into a reproduction, or search from a supplied specification | VeriExploit, Neuroforger — Agent | A checked witness establishes a violation; unsuccessful search does not establish safety |
| Verification capability evaluation | Assess reasoning about contract-specific properties | LLMs as verification oracles for Solidity — Benchmark | Prediction accuracy is not proof soundness |

The research progression is **infer what to check → check existing contracts → constrain code recovery and synthesis → require independently checkable evidence for claimed violations**. The detailed Agent entries are grouped under [Smart contracts](#smart-contract-agents); the learned methods remain under Training.

<a id="code-benchmark"></a>

### Benchmark

#### Programs, proofs, and cross-language comparisons

- FVEL: Interactive Formal Verification Environment with Large Language Models via Theorem Proving [[NeurIPS'24 — Datasets and Benchmarks; cited by 2](https://proceedings.neurips.cc/paper_files/paper/2024/file/62c6d7893b13a13c659cb815852dd00d-Paper-Datasets_and_Benchmarks_Track.pdf)]
  - Background: Code2Inv-style tasks focus on isolated invariants; Isabelle developments contain larger proofs and dependencies that are difficult to expose to learning systems.
  - Key problem & insight: A prover needs both an interactive verification environment and training examples that preserve the surrounding theory context.
  - Proposed method — FVEL with two components:
    1. **FVEL**: Translate verification tasks into Isabelle and let an LLM propose proof steps against the live prover state.
    2. **FVELER**: Extract dependency-aware theories, lemmas, and proof trajectories for supervised fine-tuning and evaluation.
  - Results: 758 theories, 29,304 lemmas, and 201,498 proof steps; fine-tuning increases solved SV-COMP cases from 69 to 81 for Llama3-8B and from 75 to 84 for Mistral-7B.

- miniCodeProps: a Minimal Benchmark for Proving Code Properties [[arXiv'24; cited by 2](https://arxiv.org/abs/2406.11915)]
  - Background: Mathematical theorem-proving benchmarks do not establish whether a model can prove properties of executable programs.
  - Key problem & insight: Hold the implementation and specification fixed to isolate proof-generation ability.
  - Proposed method — miniCodeProps with two evaluation choices:
    1. **Program-property pairs**: Provide Lean programs over lists, natural numbers, and binary trees, together with unproved correctness statements.
    2. **Difficulty tiers**: Separate straightforward properties from problems requiring substantial induction or auxiliary reasoning; accept only Lean-checked proofs.
  - Results: 201 specifications; evaluated neural provers solve some easy tasks but almost none of the medium and hard tasks.

- DafnyBench: A Benchmark for Formal Software Verification [[TMLR'25; cited by 5](https://openreview.net/forum?id=yBgTVWccIx)] [[arXiv'24](https://arxiv.org/abs/2406.08467)]
  - Background: Dafny checks functional correctness automatically only after users supply suitable assertions, invariants, and other proof hints.
  - Key problem & insight: Removing hints from already verified programs creates a reproducible task with an executable correctness oracle.
  - Proposed method — DafnyBench with two components:
    1. **Hint-completion tasks**: Keep program logic and target specifications while asking a model to restore missing verification annotations.
    2. **Feedback-based evaluation**: Run Dafny after generation and optionally return error messages for iterative repair.
  - Results: More than 750 programs and approximately 53,000 lines of code; the original study's best model/prompting configuration verifies 68% of tasks. This is a historical baseline, not a current leaderboard ceiling.

- Proving the Coding Interview: A Benchmark for Formally Verified Code Generation [[LLM4Code@ICSE'25; cited by 6](https://github.com/quinn-dougherty/fvapps)] [[arXiv'25](https://arxiv.org/abs/2502.05714)]
  - Background: APPS checks programming-contest solutions with tests; mathematical proof benchmarks usually provide a theorem without requiring an executable implementation.
  - Key problem & insight: Evaluate implementation and proof construction together on ordinary programming problems.
  - Proposed method — FVAPPS with two components:
    1. **Lean 4 task construction**: Convert APPS-style programming puzzles and their correctness requirements into implementations and unproved formal statements.
    2. **Joint completion**: Require a solver to fill the program and proof holes and pass Lean's checker.
  - Results: 4,715 samples, including 1,083 curated samples; on 406 theorems from 100 sampled problems, Sonnet proves 30% and Gemini 18% in the reported setup.

- A benchmark for vericoding: formally verified program synthesis [[arXiv'25; cited by 0](https://arxiv.org/abs/2509.22908)] [[Dafny@POPL'26](https://popl26.sigplan.org/details/dafny-2026-papers/13/A-benchmark-for-vericoding-formally-verified-program-synthesis)]
  - Background: Dafny, Verus, and Lean benchmarks largely measure different task collections, and many assess proof completion rather than implementation synthesis.
  - Key problem & insight: Make generation from a fixed formal specification an explicit task: vericoding.
  - Proposed method — multilingual vericoding benchmark with two components:
    1. **Specification-only inputs**: Remove implementations and ask models to produce code plus the proof material required by each verifier.
    2. **Tool-based evaluation**: Check generated artifacts in Dafny, Verus/Rust, or Lean; separately examine unseen tasks and optional natural-language descriptions.
  - Results: 12,504 specifications, including 6,174 new unseen problems; reported success rates are 82% in Dafny, 44% in Verus/Rust, and 27% in Lean. The language subsets are not matched problems, so these rates are not a controlled language comparison.

- AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms [[arXiv'26; cited by 0](https://arxiv.org/abs/2602.09464)]
  - Background: Different problem sets make Dafny-versus-Verus-versus-Lean success rates difficult to interpret.
  - Key problem & insight: Align functional contracts across languages so differences reflect the verification workflow rather than task selection.
  - Proposed method — AlgoVeri with two components:
    1. **Aligned algorithms**: Specify the same 77 classical algorithms in Dafny, Verus, and Lean.
    2. **Iterative-repair evaluation**: Measure verifier success and analyze how additional repair attempts change failure modes in each language.
  - Results: Gemini-3 Flash reaches 40.3%, 24.7%, and 7.8% respectively; this matched setup exposes the additional memory-model and proof-construction burdens in Verus and Lean.

- Neural Theorem Proving for Verification Conditions: A Real-World Benchmark [[ICLR'26; cited by 0](https://proceedings.iclr.cc/paper_files/paper/2026/hash/41efc6e1f29cf7c6bf7c6d9909850761-Abstract-Conference.html)]
  - Background: Industrial verifiers leave verification conditions (VCs) that SMT solvers cannot discharge; competition mathematics is an imperfect proxy for these obligations.
  - Key problem & insight: Reuse existing verification-condition generators to obtain software-derived proof tasks with controlled formal translations.
  - Proposed method — NTP4VC with two components:
    1. **VC extraction**: Use Why3/Frama-C pipelines and verified projects, including Linux and Contiki-OS examples, to extract obligations.
    2. **Multilingual proof tasks**: Translate semantically equivalent obligations into Isabelle, Lean, and Rocq; construct harder instances through controlled transformations.
  - Results: Provides a benchmark across three proof-assistant languages; general LLMs and specialized theorem provers still leave substantial VC coverage gaps. It measures VC discharge rather than specification synthesis.

- Learning Loop Invariants for Program Verification [[NeurIPS'18; cited by 54](https://papers.nips.cc/paper_files/paper/2018/hash/65b1e92c585fd4c2159d5f33b5030ff2-Abstract.html)]
  - Background: Search-based invariant generators cannot learn from earlier attempts; decision-tree and template methods depend on hand-designed predicates that restrict the invariant language.
  - Key problem & insight: The theorem prover supplies only a sparse binary reward after a complete invariant is proposed. Construct the invariant incrementally over a learned program-graph representation and turn counterexamples into intermediate learning signals.
  - Proposed method — Code2Inv with three components:
    1. **Structured external memory**: Encode the program graph with a graph neural network so the policy can query variables, expressions, and control structure.
    2. **Multi-step autoregressive model**: Generate the invariant as a sequence of syntax- and semantics-constrained decisions, using attention to shift focus across the program.
    3. **Reinforcement learning**: Combine stepwise shaping with Z3 acceptance and counterexamples, avoiding supervised ground-truth invariants.
  - Results: Solves 106/133 programs versus 100 for ICE-DT, 77 for LoopInvGen, and 74 for C2I, while requiring orders of magnitude fewer Z3 queries for comparable solved counts.

- LoopInvGen: A Loop Invariant Generator based on Precondition Inference [[arXiv'17; cited by 7](https://arxiv.org/abs/1707.02029)]
  - Background: Data-driven invariant learners usually compose a fixed set of atomic predicates; choosing that set in advance limits expressiveness and can dominate search cost.
  - Key problem & insight: Reduce invariant inference to repeated precondition-inference queries, and synthesize new predicates only when counterexamples show that the current feature set is insufficient.
  - Proposed method — LoopInvGen with three components:
    1. **PROCESS**: Simplify the SyGuS-INV problem with static analysis, remove unused variables, and reject immediately inconsistent inputs.
    2. **RECORD**: Use a solver as an execution engine to sample reachable loop-head and exit states, restarting from uncovered counterexamples when needed.
    3. **INFER**: Strengthen the postcondition into an inductive invariant through PIE, whose **BFL** learns Boolean combinations and whose **SYNTH** component grows arithmetic features on demand.
  - Results: The competition paper reports substantially faster convergence than fixed-feature tools on the SyGuS-Comp 2018 INV track, but does not provide an aggregate solved-count table; the implementation reduces its initial state sample from 6,400 to 512 and uses batched counterexamples to reduce CEGIS rounds.

- Diffy: Inductive Reasoning of Array Programs Using Difference Invariants [[CAV'21; cited by 2](https://arxiv.org/abs/2105.14748)]
  - Background: Quantified loop invariants for parameterized array programs are difficult to infer, especially with sequential or nested loops; earlier induction tools fail when control flow depends on the array-size parameter.
  - Key problem & insight: Relations between executions at sizes $N$ and $N-1$ are often simpler than a direct invariant for either program. Use these **difference invariants** to automate the inductive step.
  - Proposed method — Diffy with three components:
    1. **Program transformation**: Construct aligned versions of the input program at adjacent parameter values and peel loop iterations to expose their behavioral difference.
    2. **Difference-invariant inference**: Infer relations between corresponding scalar and array states at key joint control-flow points.
    3. **Inductive verification**: Prove base cases concretely, then use the difference invariants and Z3 to establish the parameterized Hoare triple for all $N$.
  - Results: Verifies 151/157 safe programs versus 110 for Vajra, 110 for VeriAbs, and 20 for VIAP; it also detects 142/146 unsafe variants versus 115, 125, and 120 respectively.

- Laurel: Generating Dafny Assertions Using Large Language Models [[OOPSLA'25; cited by 3](https://arxiv.org/abs/2405.16792)]
  - Background: Whole-proof benchmarks emphasize short standalone programs, whereas large Dafny developments more often fail because an existing proof needs a local assertion involving project-specific definitions.
  - Key problem & insight: LLMs frequently choose the wrong insertion point and lack the local proof idioms needed for the assertion. Derive the location from verifier diagnostics and retrieve assertions from structurally similar proofs in the same codebase.
  - Proposed method — Laurel with three components:
    1. **DafnyGym**: Remove indispensable assertions from Dafny Libraries, Cedar, and DafnyVMC, retaining tasks where deletion breaks verification.
    2. **Assertion placeholder localization**: Analyze the Dafny error and control-flow graph to place a single hole at the location most likely to unblock the proof.
    3. **Proof-similarity retrieval**: Rank surrounding lemmas with hierarchical edit distance, supply six assertion examples, and accept a generation only when Dafny verifies the repaired lemma.
  - Results: The headline result is 82/145 assertions generated successfully (56.6%), versus 34.4% for placeholders alone; the evaluation section refers to 143 tasks, so the paper contains a 143-versus-145 denominator discrepancy.

- VERINA: Benchmarking Verifiable Code Generation [[ICLR'26; cited by 2](https://arxiv.org/abs/2505.23135)]
  - Background: Existing verified-code benchmarks usually isolate code, specification, or proof generation and therefore cannot identify which stage limits an end-to-end system.
  - Key problem & insight: Package all three artifacts with independent or composable evaluators, and test specification meaning separately from whether a proof happens to compile.
  - Proposed method — VERINA with three components:
    1. **Curated Lean tasks**: Provide 189 problems with detailed descriptions, reference code, pre/postconditions, optional proofs, and positive and negative tests.
    2. **Modular task graph**: Evaluate **CodeGen**, **SpecGen**, and **ProofGen** independently or compose them into code-and-specification and end-to-end workflows.
    3. **Multi-stage specification evaluator**: Check soundness and completeness through Lean proofs when possible, then concrete and property-based tests when proof search is inconclusive.
  - Results: The best general model reaches 72.6% CodeGen, 52.3% sound-and-complete SpecGen, and 4.9% ProofGen at pass@1; the best specialized prover reaches 11.2%, and verifier-guided refinement raises proof success to about 20% at 64 attempts.

- RAG-Verus: Repository-Level Program Verification with LLMs using Retrieval Augmented Generation [[arXiv'25; cited by 1](https://arxiv.org/abs/2502.05344)]
  - Background: Function-level Verus tasks fit all relevant context in one file and are close to saturation, while real projects require repository-specific lemmas, types, and cross-module dependencies.
  - Key problem & insight: Repository proof completion is primarily a context-selection problem. Index both code structure and semantic summaries, then retrieve examples and premises separately before generation.
  - Proposed method — RAG-Verus with three stages:
    1. **Mining code properties**: Extract signatures, code modes, calls, types, and module relationships into searchable code and metadata indices.
    2. **Context retrieval**: Retrieve code- or informalization-similar demonstrations and candidate dependencies using FAISS, learned projections, or compiler-derived graphs.
    3. **Proof generation and refinement**: Add the retrieved context to an AutoVerus-style generator and feed failed Verus diagnostics back for repair.
  - Results: On VerusBench, code retrieval reaches 60.4% success versus 18.0% for direct generation under a five-sample budget. On the 383-task RepoVBench, retrieval plus refinement reaches 19.6% versus 15.4%, a 27% relative increase; the Simple subset improves from 13.5% to 44.2%.

- Towards Repository-Level Program Verification with Large Language Models [[LMPL'25; cited by 1](https://arxiv.org/abs/2509.25197)]
  - Background: The initial RepoVBench covered one Verus repository, leaving unclear whether retrieval helps across projects with different proof conventions and dependency structures.
  - Key problem & insight: Expand evaluation across heterogeneous verified systems and represent each task with repository-wide metadata, while preserving the fixed implementation and specification during proof completion.
  - Proposed method — RVBench and RagVerus with three components:
    1. **Task identification**: Parse four Verus projects, remove proof annotations from nontrivial functions, and retain 755 tasks across 337 modules and 3,464 functions.
    2. **Metadata extraction**: Index signatures, calls, types, variables, code modes, and structural relations to expose project-specific premises.
    3. **Hybrid retrieval and refinement**: Retrieve global proof examples plus local semantic dependencies, generate annotations, and iteratively repair them from Verus feedback while checking that executable code remains intact.
  - Results: The paper reports roughly 3x higher pass rates on VerusBench and a 27% relative gain on RVBench. On individual repositories, retrieval plus refinement raises success from 16.3% to 20.9% on IronKV, 6.15% to 13.4% on Vest-core, and 20.0% to 40.0% on Anvil.

- VerifyThisBench: Generating Code, Specifications, and Proofs All at Once [[arXiv'25; cited by 1](https://arxiv.org/abs/2505.19271)]
  - Background: Most formal-verification benchmarks supply either the implementation, the specification, or the proof scaffold, so high local completion rates do not measure natural-language-to-verified-program capability.
  - Key problem & insight: Start from competition descriptions and require the complete artifact, then provide controlled relaxed variants to identify whether code, specification, or invariant generation is the bottleneck.
  - Proposed method — VerifyThisBench with three components:
    1. **End-to-end benchmark**: Port 41 VerifyThis challenges into 154 tasks where the model generates specifications, implementations, and proofs for seven verification tools.
    2. **VerifyThisBenchXS**: Derive 580 completion tasks by removing code, specifications, or loop invariants from human solutions.
    3. **Unified feedback environment**: Compile and verify each artifact, return tool errors for four repair rounds, and keep tool-specific execution reproducible.
  - Results: On the end-to-end benchmark, o3-mini improves from 3.62% zero-shot to 9.37% after refinement. On VerifyThisBenchXS, the best refined result is 17.24%; averaged across models, refinement reaches 9.73% for code, 11.27% for specifications, and 9.81% for loop invariants.

- VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean [[arXiv'26; cited by 0](https://arxiv.org/abs/2602.18307)]
  - Background: Lean prover benchmarks largely use Mathlib mathematics or standalone verified functions, whereas software-verification proofs depend on custom datatypes, semantics, invariants, and lemmas distributed across project files.
  - Key problem & insight: Preserve the original repository dependency structure and compare an oracle-like relevant context with the complete noisy repository to separate proof reasoning from context retrieval.
  - Proposed method — VeriSoftBench with three components:
    1. **Repository-grounded extraction**: Select 500 nontrivial, hole-free proof obligations from 23 Lean formal-methods repositories at fixed commits, retaining their local definitions and cross-file dependencies.
    2. **Two context regimes**: Evaluate with **Curated Context**, derived from ground-truth dependencies, and **Full Repo Context**, where the prover must find useful declarations and analogous proofs itself.
    3. **Dependency analysis**: Measure direct and transitive project/library dependencies and correlate their count and nesting depth with proof success.
  - Results: Gemini-3-Pro solves 41.0% with curated context and 34.8% with full-repository context; Claude Opus 4.5 reaches 31.2% and 23.2%, while Goedel-Prover-v2 solves 0% in the repository-scale setting. Transitive dependency count correlates negatively with success ($r=-0.359$, $p<0.001$).

- RoMA-V: Benchmarking LLM-Based Theorem Proving Capabilities in Program Verification [[KSEM'26 proceedings; cited by 0](https://doi.org/10.1007/978-981-92-2852-2_42)]
  - Background: Theorem-proving benchmarks emphasize formal mathematics and usually assign one coarse difficulty label, obscuring repository context and program-verification-specific proof structure.
  - Key problem & insight: Characterize each theorem along multiple structural attributes and aggregate them into a difficulty measure that exposes model strengths hidden by a single success rate.
  - Proposed method — RoMA-V with three components:
    1. **Repository corpus**: Preserve theorem contexts from 30 real Rocq projects and package 1,312 program-verification and supporting-foundation proof tasks.
    2. **Multi-attribute annotations**: Record theorem and repository properties so results can be sliced by proof and context characteristics rather than only by aggregate pass rate.
    3. **TDM (Tri-dimensional Difficulty Metric)**: Combine theorem-level and repository-level dimensions to analyze hierarchical difficulty and model specialization.
  - Results: The study evaluates eight representative LLMs and reports hierarchical performance patterns and dimension-specific strengths. The publicly accessible Springer abstract does not disclose per-model success numbers, and the full chapter is not open access as of September 22, 2026.

- CLEVER: A Curated Benchmark for Formally Verified Code Generation [[NeurIPS'25 — Datasets and Benchmarks; cited by 0](https://arxiv.org/abs/2505.13938)]
  - Background: Earlier verified-code datasets can contain incomplete, implementation-leaking, or automatically generated specifications, allowing trivial code or a self-consistent but unintended code-spec pair to pass.
  - Key problem & insight: Hide a human-written, non-computable reference specification and require separate machine-checked certificates for semantic equivalence and implementation correctness.
  - Proposed method — CLEVER with two certification stages:
    1. **Specification certification**: Generate a Lean predicate from the natural-language task and prove it equivalent to the held-out ground-truth specification.
    2. **Implementation certification**: Generate a Lean implementation and prove it satisfies the ground-truth specification, preventing errors in the generated specification from weakening the code check.
  - Results: Across 161 HumanEval-derived tasks, every evaluated few-shot or agentic configuration solves at most 1 task end to end (0.621%); specification-equivalence proof rates are at most 1.863%, while the best implementation-certification rate is 8.696%.

- VeriContest: A Competitive-Programming Benchmark for Verifiable Code Generation [[arXiv'26; cited by 0](https://arxiv.org/abs/2605.08553)]
  - Background: Existing verified-code benchmarks are smaller and dominated by interview-style functions; they do not pair harder contest algorithms with expert Verus specifications, proofs, and specification-completeness tests.
  - Key problem & insight: Build a large corpus from judge-accepted Rust and use proof checking plus positive/negative tests so specification, code, and proof generation can be evaluated independently and jointly.
  - Proposed method — VeriContest with three construction phases:
    1. **Manual seeds**: Experts author 91 sound-and-complete Verus specifications, judge-accepted implementations, and checked proofs.
    2. **Human-reviewed expansion**: A coding agent expands the set to 946 LeetCode and Codeforces problems using reusable proof guidance, with online-judge, expert, and Verus validation.
    3. **Test generation and Post2Exe**: Generate valid inputs and mutation-based incorrect outputs, then execute translatable postconditions to expose incompleteness; this process finds 60 contracts needing revision.
  - Results: GPT-5.5 reaches 48.31% SpecGen, 92.18% NL-to-code, 13.95% ProofGen, and only 5.29% end to end. For Qwen 3.6, verifier-guided repair raises proof success from 7.08% at one attempt to 11.84% at twenty.

- NL2VBench: A Benchmark for End-to-End Natural-Language to Verifiable Rust Generation [[ASE'26 — Tools and Data Sets; cited by 0](https://conf.researchr.org/details/ase-2026/ase-2026-tools-and-data-sets/27/NL2VBench-A-Benchmark-for-End-to-End-Natural-Language-to-Verifiable-Rust-Generation)]
  - Background: Verus benchmarks generally begin with formal code or specifications, so they do not measure whether a model can recover a complete verifiable program from an ordinary natural-language request.
  - Key problem & insight: Reverse-construct natural-language descriptions from verified artifacts and compare different generation orders, revealing whether code-first or specification-first decomposition helps end-to-end verification.
  - Proposed method — NL2VBench with three pipelines and one semantic check:
    1. **P1 Direct**: Generate the specification, executable Rust, and proof together from natural language.
    2. **P2 Code-first**: Generate Rust first, freeze it, and then synthesize the specification and proof.
    3. **P3 Spec-first**: Generate the specification first, then synthesize code and proof against it.
    4. **SVC**: Expert-review verifier-passing artifacts for semantic validity, separating proof acceptance from intent alignment.
  - Results: The 762-task artifact reports GPT-5.2 few-shot pass@1/pass@5 of 29.92%/50.00% for P1, 10.37%/15.35% for P2, and 20.21%/40.55% for P3; zero-shot success is near zero across pipelines.

- Vero: Can AI Agents Build Formally Verified Software Repositories? [[arXiv'26; cited by 0](https://arxiv.org/abs/2608.13522)]
  - Background: Repository benchmarks fix implementations and ask only for local proof completion, leaving unmeasured the coupling between implementation choices, shared lemmas, and proofs across a multi-module codebase.
  - Key problem & insight: Freeze APIs and specifications but let agents jointly choose implementations and proofs; separately accept machine-checked negative evidence when the benchmark's own reference code or specifications are defective.
  - Proposed method — Vero with three components:
    1. **Multi-language curation pipeline**: Translate selected Python, Dafny, Verus, and Coq repositories into 43 reviewed Lean projects containing 743 scored APIs and 2,705 specifications.
    2. **Two task modes**: In **proof-only**, prove every fixed specification for reference code; in **code-and-proof**, implement every API and prove every specification for the agent's implementations.
    3. **Formal audit and anti-cheating grader**: Accept proofs of incorrect reference code, unsatisfiable specifications, or inconsistent specification sets, while rebuilding clean submissions and rejecting untrusted axioms or semantic bypasses.
  - Results: GPT-5.5 (xhigh) fully solves 27/43 repositories in code-and-proof and 25/43 in proof-only; ten repositories resist every evaluated configuration. Although it closes 87.3% of individual specifications in code-and-proof, repository completion requires reusable helper libraries: 80/82 full solves share a helper across at least two specifications.

- VeCoGen: Automating Generation of Formally Verified C Code with Large Language Models [[FormaliSE'25; citations N/A (method-paper set)](https://arxiv.org/abs/2411.19275)]
  - Background: LLM-generated C can pass tests while violating functional or runtime-safety requirements; earlier verified synthesis systems either need manual feedback or are restricted to narrow program grammars.
  - Key problem & insight: Use natural-language and ACSL specifications together, then feed compiler and Frama-C failures back into the same candidate rather than repeatedly sampling from scratch.
  - Proposed method — VeCoGen with two stages:
    1. **Initial candidate generation**: Sample several C implementations from the natural-language description, ACSL contract, function signature, and tests; compile them and check Frama-C WP and RTE obligations.
    2. **Iterative code improvement**: Rank failed candidates with tests, select the best attempt, and regenerate it using compiler/verifier diagnostics until all obligations pass or the budget is exhausted.
  - Results: Solves 9/15 VECOSET tasks in the initial generation and 13/15 after repair. Combining natural-language and formal specifications retains the 13 solved tasks while reducing total runtime to 3,722 seconds versus 5,447 for natural language alone and 10,538 for ACSL alone.

- SynVer: Generating High-Assurance Programs Using Large Language Models and Formal Verification [[arXiv'24; citations N/A (method-paper set)](https://arxiv.org/abs/2410.14835)]
  - Background: Deductive synthesizers provide machine-checked correctness but specialize in narrow domains, while unconstrained LLMs cover more programs without producing proofs for rich heap and API specifications.
  - Key problem & insight: Bias LLM-generated C toward shapes that VST automation can handle, then combine a deterministic tactic with LLM-selected proof steps for the remaining obligations.
  - Proposed method — SynVer with three components:
    1. **Biased program generation**: Generate compilable C from a VST separation-logic specification, description, examples, and allowed helper functions, replacing loops with recursion to simplify proof structure.
    2. **SEPAUTO**: Apply custom VST/Rocq tactics for control flow, calls, heap side conditions, arithmetic, and common data-structure lemmas.
    3. **GenProof**: Ask a prover LLM for a tactic when SEPAUTO stalls, reject non-progressing steps, and backtrack over failed proof paths.
  - Results: Fully verifies 34/48 generated programs (70%), with 10 solved by SEPAUTO alone. On the common 42-task comparison, GenProof solves 30 tasks versus 6 for Tactician and 3 for Rango; a hand-engineered stronger tactic solves 36 after about 100 person-hours of development.

- The FormAI Dataset: Generative AI in Software Security through the Lens of Formal Verification [[PROMISE'23; citations N/A (source dataset)](https://arxiv.org/abs/2307.02192)]
  - Background: Vulnerability datasets typically contain human code or one injected flaw per sample, providing limited evidence about the error distribution of unconstrained LLM-generated C.
  - Key problem & insight: Generate a large diverse corpus by varying task and style prompts, then use bounded model checking to attach counterexample-backed vulnerability labels at scale.
  - Proposed method — FormAI construction with three components:
    1. **Dynamic zero-shot prompts**: Combine 200 task types with 100 coding styles and temperature sampling to generate independent GPT-3.5 C programs.
    2. **Compilation filter**: Retain 112,000 compilable programs and record external-library requirements rather than restricting the corpus to toy functions.
    3. **ESBMC labeling**: Check overflow, bounds, pointer, divide-by-zero, memory, and related properties with a 30-second bounded run; store vulnerability type, line, function, CWE mapping, and counterexample.
  - Results: ESBMC analyzes 106,139 files and finds counterexample-backed vulnerabilities in 57,389 unique programs (over 54%), totaling 197,800 findings. A successful bounded run establishes only that no counterexample was found within the configured bound, not full program safety.

- CASP: An Evaluation Dataset for Formal Verification of C Code [[AISoLA'25; cited by 0](https://doi.org/10.1007/978-3-032-07132-3_5)]
  - Background: ACSL examples are scattered across small repositories and complete files, making it difficult to benchmark generation between a standalone C implementation and its formal contract.
  - Key problem & insight: Mine large code corpora, repair whole files until they verify, and only then extract self-contained function-contract pairs that remain independently verifiable.
  - Proposed method — CASP with three construction stages:
    1. **Source filtering**: Search deduplicated Stack v1/v2 C files for ACSL-specific syntax and retain 1,180 minimally complete, dependency-light files.
    2. **Verification and repair**: Run Frama-C 30 WP/RTE with Z3, Alt-Ergo, and CVC4; send failed goals to Gemini 2.0 Flash for up to seven minimal-repair iterations, yielding 469 verified files.
    3. **Pair extraction**: Combine regex and LLM extraction, include needed local declarations, reverify every pair, deduplicate, remove one-line implementations, and manually inspect all survivors.
  - Results: Releases 506 verified C/ACSL pairs, selected from 513 verified extracted candidates. The paper establishes dataset scale/diversity rather than a model leaderboard; downstream AutoACSL evaluates on a 413-program CASP partition.

- EqBench: A Dataset of Equivalent and Non-equivalent Program Pairs [[MSR'21; cited by 11](https://doi.org/10.1109/MSR52588.2021.00084)]
  - Background: Equivalence checkers are commonly evaluated on tiny straight-line programs that omit loops, nonlinear arithmetic, floating point, strings, arrays, and object manipulation.
  - Key problem & insight: Unify prior benchmarks and inject controlled semantics-preserving or behavior-changing edits, while publishing matched C and Java pairs plus change metadata and counterexamples.
  - Proposed method — EqBench with three components:
    1. **Source aggregation**: Merge RÊVE, ModDiff/CLEVER, ARDiff, and complex symbolic-execution functions from 18 benchmark families.
    2. **EQ/NEQ transformation**: Create equivalent versions through refactoring, renaming, and dead/unreachable code; create non-equivalent versions through AST insertion, deletion, and update mutations.
    3. **Validation and metadata**: Independently review every EQ pair, use checkers where possible, provide distinguishing tests for NEQ pairs, and translate each pair between C and Java.
  - Results: Contains 147 equivalent and 125 non-equivalent pairs in both languages. Of 272 programs, 58.8% contain loops and 57.3% contain nonlinear arithmetic; versions change 14.4% of statements on average.

- VeriBench: An End-to-End Formal Verification Benchmark for AI Coding Agents [[AI for Math@ICML'25; citations N/A (not indexed)](https://openreview.net/forum?id=rWkGFmnSNl)] [[2026 project update](https://brando90.github.io/veribench-blog/blog/veribench-launch/)]
  - Background: Python-to-Lean systems can produce files that compile or pass examples while leaving `sorry` placeholders, stating vacuous theorems, or omitting essential behavioral obligations.
  - Key problem & insight: Evaluate implementation, tests, specifications, theorem statements, and proof closure conjunctively, with theorem coverage calibrated to human judgments rather than inferred from compilation.
  - Proposed method — VeriBench with three components:
    1. **Python-to-Lean tasks**: Pair source programs with hand-curated Lean artifacts following a fixed nine-part schema across educational, algorithmic, standard-library, and security-focused code.
    2. **Smooth Conjunctive Score for Code Verification (SCSC)**: Combine file compilation, no-`sorry` theorem rate, nontrivial formal content, and coverage of gold theorem obligations so one easy dimension cannot hide another failure.
    3. **Human-calibrated semantic coverage**: Train and calibrate an LLM judge against five human raters for candidate-to-gold theorem coverage.
  - Results: The May 8, 2026 project update reports 884 tasks, differing from the fork's earlier 614+282=896 accounting. Codex/GPT-5.4 scores 1.000 compilation, 0.237 no-`sorry`, 0.102 theorem coverage, and 0.289 agent skill; the coverage judge reaches about 0.70 held-out Pearson correlation with humans. The original OpenReview PDF remains challenge-gated, so these numbers are from the official current project release.

#### Specifications and semantic faithfulness

- From Informal to Formal – Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs [[ACL'25; citations N/A (method-paper set)](https://aclanthology.org/2025.acl-long.1310/)]
  - Background: End-to-end formal-reasoning scores mix specification writing, mathematical reasoning, and proof construction.
  - Key problem & insight: Split the path from informal requirements to checked proofs so individual capabilities can be measured and trained.
  - Proposed method — task decomposition with two components:
    1. **Instruction dataset**: Distill GPT-4o into approximately 18,000 instruction-response pairs across six tasks and Coq, Lean 4, Dafny, ACSL, and TLA+.
    2. **Task-specific evaluation and fine-tuning**: Compare open models on individual stages instead of relying on a single end-to-end score.
  - Results: Evaluates ten open models; formal-data fine-tuning produces improvements of up to approximately 3x in the reported tasks, with additional transfer to reasoning and coding evaluations.

- Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification [[ICLR'26](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c04d37be05ba74419d2d5705972a9d64-Abstract-Conference.html)]
  - Background: A model can verify individual functions while producing contracts too weak or inconsistent for their callers.
  - Key problem & insight: Composition needs specifications that transport the right facts across function boundaries.
  - Proposed method — DafnyCOMP with two components:
    1. **Compositional programs**: Construct verified programs containing two to five interacting functions, using chain and non-chain acyclic call graphs.
    2. **Specification regeneration**: Ask models to reconstruct contracts and proof annotations, then verify the complete composed program.
  - Results: 400 programs: 300 chain and 100 DAG instances; the strongest evaluated model reaches only 2% verification at Pass@8 on the chain split despite much higher performance on single-function tasks.

- VeriEquivBench: An Equivalence Score for Ground-Truth-Free Evaluation of Formally Verifiable Code [[ICLR'26](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ebfa4297cd6419f64efe86f657ba49d0-Abstract-Conference.html)] [[arXiv'25](https://arxiv.org/abs/2510.06296)]
  - Background: Generated code can satisfy an underspecified contract; matching an expert reference specification is expensive and can inherit reference errors.
  - Key problem & insight: Evaluate semantic agreement rather than treating any compilable, verified code-specification pair as correct.
  - Proposed method — VeriEquivBench with two components:
    1. **Algorithmic tasks**: Construct substantially more complex Dafny generation tasks than textbook examples.
    2. **Equivalence score**: Use formal equivalence checks to assess generated code and specifications without requiring an expert-written reference specification for every task.
  - Results: 2,389 problems; the reported Claude-4-Sonnet setup solves none under Pass@4 despite achieving 75.81% on CloverBench, exposing the gap between small examples and complex algorithmic specifications.

- Can Large Language Models Model Programs Formally? [[arXiv'26](https://arxiv.org/abs/2604.01851)]
  - Background: Most LLM formal-reasoning benchmarks start with a formal theorem; model checking first needs a faithful transition-system model of the program.
  - Key problem & insight: Evaluate program-to-model translation as a separate capability, including whether the result is usable by a model checker.
  - Proposed method — Model-Bench with two components:
    1. **Program collection**: Select Python tasks from HumanEval, MBPP, and LiveCodeBench.
    2. **Modeling pipeline**: Generate verification-ready formal models and check them with the associated model-checking workflow.
  - Results: 400 programs; the evaluation identifies substantial modeling failures, establishing that code generation proficiency alone does not establish faithful formal abstraction.

- Verus-SpecGym: An Agentic Environment for Evaluating Specification Autoformalization [[arXiv'26; cited by 0](https://arxiv.org/abs/2605.26457)]
  - Background: A proof establishes a formal specification, but the specification may omit input assumptions or accept incorrect outputs.
  - Key problem & insight: Make generated specifications executable so their meaning can be tested independently of the implementation proof.
  - Proposed method — Verus-SpecGym with two components:
    1. **Verus-SpecBench**: Provide 581 Codeforces-derived specification-writing tasks and an agent environment with Verus, shell, and filesystem tools.
    2. **Executable specifications**: Extend Verus's `exec_spec` support and evaluate against official tests and adversarial Codeforces hacks.
  - Results: The strongest evaluated model reaches 77.8%; other frontier models reach 51.1–57.8%. An LLM judge misses 26% of failures detected by this evaluator. Test-based specification validation is empirical, not a proof of complete intent alignment.

- Evaluating LLM-driven User-Intent Formalization for Verification-Aware Languages [[FMCAD'24; cited by 1](https://arxiv.org/abs/2406.09757)]
  - Background: Dafny can prove an implementation against a supplied contract, but cannot decide whether that contract captures the user's natural-language intent; runtime-only mutation metrics also fail on quantified or ghost-state specifications.
  - Key problem & insight: Treat tests as concrete intent examples and use symbolic execution to ask both whether the specification accepts valid outputs and whether it rejects deliberately corrupted outputs.
  - Proposed method — symbolic specification testing with three components:
    1. **Correctness metric**: Substitute every valid input-output test pair into the candidate postcondition and require Dafny to prove the resulting Hoare triple.
    2. **Completeness metric**: Mutate each expected output up to five ways and measure the fraction rejected by the specification, analogous to mutation-testing kill rate.
    3. **Dafny harness generation**: Compile MBPP-DFY signatures, tests, and candidate contracts into verifier programs that support quantifiers and ghost predicates.
  - Results: The prototype analyzes 64 of 153 available specifications; its scores mostly agree with human weak/strong labels, while exposing at least three supposedly strong contracts as incomplete and three test/specification transcription errors.

- Beyond Postconditions: Can Large Language Models infer Formal Contracts for Automatic Software Verification? [[arXiv'25; cited by 0](https://arxiv.org/abs/2510.12702)]
  - Background: NL-to-postcondition systems often omit input assumptions; a verifier then produces counterexamples on invalid inputs, creating false alarms even when the implementation is correct.
  - Key problem & insight: Infer preconditions and postconditions together as a functional contract, and evaluate them over the complete symbolic input space rather than only against tests.
  - Proposed method — NL2Contract with three components:
    1. **Exception-based contract encoding**: Express inferred preconditions as invalid-input exceptions and postconditions as executable assertions around the target function.
    2. **Verification metrics**: Use CrossHair to measure all-input soundness and mutant-killing completeness, alongside test-based acceptance and bug discrimination.
    3. **Real-bug evaluation**: Apply generated contracts to Python-by-Contract implementations with CrossHair and Pynguin, checking whether reported violations trigger genuine behavioral differences.
  - Results: On HumanEval+, NL2Contract raises sound@1 from 11.5–13.4% for postconditions alone to 66.7–81.1%, and verifier-guided contracts kill up to 73.7% of mutants. On real bugs, CrossHair detects 14/19 bugs that it can expose, versus 3–5 with postconditions alone.

- SpecGen: Automated Generation of Formal Program Specifications via Large Language Models [[ICSE'25; citations N/A (method-paper set)](https://arxiv.org/abs/2401.08807)]
  - Background: Houdini and Daikon enumerate human-designed specification templates, which scale cheaply but miss relational postconditions and invariants needed to characterize nontrivial Java behavior.
  - Key problem & insight: LLM outputs are often semantically close even when OpenJML rejects them. Repair the specification language itself with targeted mutations instead of discarding the candidate.
  - Proposed method — SpecGen with two phases:
    1. **Conversation-driven generation**: Prompt with JML examples, insert generated preconditions, postconditions, and invariants, and return OpenJML failures for iterative correction.
    2. **Mutation-based generation**: Apply four operator families to near-miss formulas and use weighted heuristic selection to verify the most promising variants first.
  - Results: Generates verifiable specifications for 279/385 Java programs versus 247 for AutoSpec and 98 for Houdini. Expert ratings average 4.54/5 for semantic quality, close to 4.83 for ground-truth specifications.

- SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification [[ICLR'26](https://proceedings.iclr.cc/paper_files/paper/2026/hash/70e692da44c19710386648694e2b899b-Abstract-Conference.html)]
  - Background: Text-to-SQL evaluation often accepts two queries because they produce the same output on one database, even when their semantics differ.
  - Key problem & insight: Search for a distinguishing database instead of trusting a fixed test database.
  - Proposed method — SpotIt with two components:
    1. **Bounded equivalence verification**: Encode generated and reference SQL queries and ask the verifier for a database that makes their outputs disagree.
    2. **SQL support extensions**: Extend the verifier's supported constructs to cover a larger part of practical Text-to-SQL benchmarks.
  - Results: Re-evaluates ten Text-to-SQL methods on BIRD and finds differences missed by execution-based evaluation. Absence of a distinguishing database establishes only the checked bounded result.

- Can LLMs Reason Like Automated Theorem Provers for Rust Verification? VCoT-Bench: Evaluating via Verification Chain of Thought [[ICML'26](https://icml.cc/virtual/2026/poster/63236)]
  - Background: A binary Verus success score hides which deductive steps an LLM can actually reconstruct.
  - Key problem & insight: Expose the solver's reasoning as explicit intermediate verification obligations.
  - Proposed method — VCoT-Lift and VCoT-Bench with two components:
    1. **VCoT-Lift**: Lift low-level solver reasoning into human-readable Verification Chain-of-Thought steps.
    2. **Completion tasks**: Remove proof content at different rates, locations, and proof types to diagnose reasoning gaps.
  - Results: 1,988 completion tasks and ten evaluated models; performance is fragile across the three diagnostic dimensions, rather than uniformly explained by whole-proof pass/fail.

- How Powerful are LLMs in Generating Formal Program Specifications? [[ICML'26](https://icml.cc/virtual/2026/poster/66406)]
  - Background: Full implementation verification and specification-equivalence proving can fail because a proof is hard, even when the proposed specification is meaningful.
  - Key problem & insight: Instantiate specifications on trusted examples to separate semantic quality from general proof difficulty.
  - Proposed method — Coins with two components:
    1. **Rocq specifications**: Curate human-written reference specifications for HumanEval tasks and collect generated alternatives.
    2. **Concrete proof obligations**: Instantiate the specifications with trusted test cases and use Rocq to check the resulting obligations.
  - Results: The HumanEval study shows that verification complexity can obscure differences in specification quality; Coins provides a more discriminative evaluation, while failed proofs remain inconclusive evidence.

#### Smart-contract property reasoning

- Accessible Smart Contracts Verification: Synthesizing Formal Models with Tamed LLMs [[ICST'25; citations N/A (method-paper set)](https://arxiv.org/abs/2501.12972)]
  - Background: Model-based smart-contract audits require manually translating Rust/CosmWasm state transitions into Quint or TLA+-style models before checking properties and generating tests.
  - Key problem & insight: Mechanically preserve the contract's interface and control skeleton, and limit the LLM to filling typed semantic stubs that can be repaired against executable examples.
  - Proposed method — model synthesis with three phases:
    1. **Mechanical transpilation**: Parse CosmWasm Rust and emit a Quint model skeleton containing declarations, message handlers, and typed holes.
    2. **LLM stub completion**: Generate one Quint action at a time from the Rust fragment, user description, examples, and a CTF-01 demonstration set.
    3. **Testing and repair**: Iteratively fix static, runtime, and semantic failures using Quint diagnostics and I/O traces, then use the model for checking and model-based testing.
  - Results: Generates models for 23 functions across five CosmWasm CTF contracts in five independent runs. Most unseen-test pass rates are 90–100%; the hardest CTF-09 functions succeed in 4/5 runs and reach 73–80% on held-out examples.

- LLMs as verification oracles for Solidity [[FC'26](https://www.ifca.ai/fc26/program.html)] [[arXiv'25](https://arxiv.org/abs/2509.19153)]
  - Background: SolCMC and Certora require formal encodings and have different limits on expressible contract properties; ordinary vulnerability benchmarks mostly evaluate fixed bug classes.
  - Key problem & insight: Measure whether reasoning models can assess supplied business-logic properties, including transaction ordering and liveness, rather than merely recognize vulnerability patterns.
  - Proposed method — verification-oracle evaluation with three components:
    1. **Controlled dataset**: Pair five contract families and their mutations with properties and manually established ground truth.
    2. **Quantitative analysis**: Compare GPT-5 and GPT-4 predictions; evaluate SolCMC and Certora on the subsets expressible in each tool.
    3. **Qualitative analysis**: Inspect explanations and proposed counterexamples for coherent interpretation and valid reasoning.
  - Results: On 667 tasks, GPT-5 reaches 92% accuracy and 92% F1 versus GPT-4's 63% and 64%; GPT-5 still makes 31 false-positive and 21 false-negative predictions. These are untrusted model judgments, not machine-checked proofs.

#### Systems and protocol verification

- LTLBench: Towards Benchmarks for Evaluating Temporal Reasoning in Large Language Models [[arXiv'24; cited by 0](https://arxiv.org/abs/2407.05434)]
  - Background: Temporal-reasoning datasets rely on informal annotations and often cannot distinguish linguistic shortcuts from correct reasoning over explicit event transitions.
  - Key problem & insight: Generate the world model and temporal formula together, so every true/false answer has an executable LTL/transition-system oracle and difficulty can be varied by events and operators.
  - Proposed method — LTLBench with four construction stages:
    1. **Random directed graphs**: Sample events, transitions, and an initial state to define the temporal world.
    2. **LTL formula generation**: Compose formulas with controlled operator count and evaluate them on the graph.
    3. **Natural-language conversion**: Render the transition context, constraints, and hypothesis through templates.
    4. **NuSMV validation**: Model-check each generated item and retain its formal truth label.
  - Results: On 2,000 tasks, GPT-5-Mini with few-shot chain of thought reaches 93.95%; averaged across 12 models, few-shot CoT reaches 76.33% versus 65.83% for direct prompting. Increasing events from 2 to 9 lowers accuracy by 17.89 points on average.

- TLA+-Bench: An Execution-Grounded Benchmark and Dataset for Natural-Language to TLA+ Specification Generation [[arXiv'26; cited by 0](https://arxiv.org/abs/2607.23425)]
  - Background: Earlier NL-to-TLA+ resources stop at parsing or model-based grading, which can accept valid-looking modules that do not bind to the intended constants/properties or that pass vacuously.
  - Key problem & insight: Ship the hidden runnable configuration with each gold task and expose interface, behavior, and property-vacuity choices as separate evaluation bounds rather than one opaque accuracy number.
  - Proposed method — TLA+-Bench with three components:
    1. **Gold and silver corpus**: Curate 403 TLC-runnable and 897 parse-only specifications from 13 public repositories, each with multiple generated descriptions and difficulty metadata.
    2. **Execution-grounded grader**: Require SANY parsing, bind the candidate to the hidden reference `.cfg`, and run TLC over the complete reachable state space for its bounded constants.
    3. **Correctness envelope**: Compare default and interface-aware prompts, then screen passes for multi-state behavior and mutation-surviving properties to quantify name mismatch and vacuity.
  - Results: The strongest model is 16% correct by default and 26% when interface names are supplied, while open models reach at most 1%. Depending on interface and pass-quality rules, the measured rate on the same outputs ranges from 18.7% to 1.7%; only 5/300 outputs survive the strict property-mutation test.

- CryptoFormalEval: Integrating LLMs and Formal Verification for Automated Cryptographic Protocol Vulnerability Detection [[arXiv'24; cited by 0](https://arxiv.org/abs/2411.13627)]
  - Background: Tamarin can discover protocol attacks, but experts must translate informal protocols and security goals into its modeling language.
  - Key problem & insight: Test whether an agent can complete the modeling-and-attack-discovery workflow on previously unseen protocols, rather than recall known attacks.
  - Proposed method — CryptoFormalEval with two components:
    1. **Protocol tasks**: Pair newly designed, flawed protocols with target security properties.
    2. **Tamarin interaction and validation**: Let agents revise models using tool feedback and validate the submitted attack artifacts.
  - Results: The dataset contains 15 protocols; the reported five-task comparison exposes syntax, semantic-modeling, and instruction-following failures even when a model informally understands an attack.

- Constructing formal models of cryptographic protocols from Alice&Bob style specifications via LLM [[Scientific Reports'25; cited by 10](https://doi.org/10.1038/s41598-025-93373-y)]
  - Background: Protocol analysts must repeatedly translate concise Alice&Bob descriptions into tool-specific ProVerif declarations; direct LLM generation lacks enough syntax and type control for verifier-ready models.
  - Key problem & insight: Decompose formal declaration synthesis into terminology-aware generation, verifier checking, and error-driven modification, while leaving protocol process/security queries fixed for controlled evaluation.
  - Proposed method — P2FGPT with three components:
    1. **Generator**: Extract cryptographic primitives and translate them with demonstrations, abstract definitions, and an abstract chain-of-thought modeled on manual formalization.
    2. **Checker**: Insert generated declarations into a complete ProVerif scaffold, use the tool return code for syntax validity, and compare content with manually authored references.
    3. **Modifier**: Feed grammar errors back through a model-specific repair prompt and regenerate missing or malformed declarations.
  - Results: The dataset derives 106 sub-protocols from four official ProVerif examples (53/25/14/14). For GLM4.0, modification raises syntax correctness from 65% to 82% and improves content F1 from 0.58 to 0.62; Qwen2.5-7B achieves the strongest content metrics, while repair prompts transfer poorly to Llama3-8B.

- Can Large Language Models Verify System Software? A Case Study Using FSCQ as a Benchmark [[HotOS'25](https://users.cs.duke.edu/~mlentz/papers/llmverif_hotos2025.pdf)]
  - Background: Success on small verified programs does not establish proof-generation ability for a file system with project-specific abstractions.
  - Key problem & insight: Evaluate directly inside FSCQ, preserving the context that its existing proofs depend on.
  - Proposed method — FSCQ study with two components:
    1. **Context construction**: Present target Rocq/Coq theorems with relevant surrounding definitions and proof context.
    2. **Best-first search**: Query an off-the-shelf LLM for proof steps and expand candidates that the prover accepts.
  - Results: 38% proof coverage on sampled FSCQ theorems; over 57% on the simpler subset with human proofs shorter than 64 tokens. These are theorem-level results, not autonomous verification of the entire file system.

- OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification [[AAAI'26](https://ojs.aaai.org/index.php/AAAI/article/view/40437)]
  - Background: Kernel specifications describe state transitions and intended behavior, which cannot safely be copied from potentially buggy implementations.
  - Key problem & insight: Test whether models recover intended kernel semantics while respecting a supplied verification model.
  - Proposed method — OSVBench with two components:
    1. **Hyperkernel tasks**: Supply functional descriptions, a state-transition programming model, and implementation context, including injected bugs.
    2. **Specification checking**: Use generated specifications in the kernel verification pipeline and categorize syntax, semantic, and bug-type failures.
  - Results: 245 tasks with approximately 20k–30k-token contexts and 12 evaluated models; the released five-shot comparison reports 55.10% Pass@1 for Doubao-1.5-pro versus 38.78% for GPT-4o.

- CrypFormBench: Benchmarking Formal Analysis Capability of Large Language Models for Cryptographic Schemes [[FSE'26](https://doi.org/10.1145/3808184)] [[arXiv'26](https://arxiv.org/abs/2606.25561)]
  - Background: Tamarin/Scyther and CryptoVerif/EasyCrypt cover different security models and require specialized input languages.
  - Key problem & insight: Measure tool-specific formalization skills separately instead of equating informal cryptographic reasoning with executable verification artifacts.
  - Proposed method — CrypFormBench with two components:
    1. **Capability matrix**: Cover interpretation, generation, completion, transformation, and correction across seven verifier languages.
    2. **Cross-tool evaluation**: Test symbolic and computational security tasks and examine few-shot prompting, repeated sampling, and lightweight fine-tuning.
  - Results: 700 instances spanning 677 schemes and 160 security properties; among nine evaluated LLMs, Claude-3.5 obtains the highest aggregate score, 48.7/100.

- Selene: Pioneering Automated Proof in Software Verification [[ACL'24; cited by 4](https://aclanthology.org/2024.acl-long.98/)]
  - Background: Small isolated theorems omit the dependencies and proof styles encountered in an industrial verification project.
  - Key problem & insight: Benchmark proof generation inside seL4 while reusing prechecked dependencies to keep interactive evaluation affordable.
  - Proposed method — Selene with two components:
    1. **Project-level benchmark**: Extract Isabelle proof tasks from seL4 and group them by proof style and complexity.
    2. **Lightweight verification environment**: Retain verified dependencies and check generated replacements within the project context; study retrieval and feedback augmentations.
  - Results: Collects 5,464 lemmas, including 5,419 procedural proofs; GPT-4 achieves 51.8% ACC@5 on the easiest procedural tier, with substantially harder results for more complex tiers.

<a id="code-training"></a>

### Training

#### Learned invariants and proof-search guidance

- DistAI: Data-Driven Automated Invariant Learning for Distributed Protocols [[OSDI'21](https://www.usenix.org/conference/osdi21/presentation/yao)]
  - Background: Distributed-protocol proofs require inductive invariants that remain valid for arbitrary system sizes; hand-written invariants are expensive.
  - Key problem & insight: Learn candidate formulas from small executions, then establish their validity symbolically rather than trusting the sampled traces.
  - Proposed method — DistAI with two components:
    1. **Data-driven enumeration**: Simulate several protocol sizes and enumerate strong, small formulas consistent with the observed states.
    2. **Monotonic weakening**: Ask an SMT solver whether the candidate conjunction is inductive; weaken it when necessary rather than restarting the search.
  - Results: Automatically verifies 13 distributed protocols and reports over two orders of magnitude speedup in some comparisons. This is symbolic/data-driven invariant learning, not LLM fine-tuning.

- Diversity-Driven Automated Formal Verification [[ICSE'22](https://people.cs.umass.edu/~brun/pubs/pubs/First22icse.pdf)]
  - Background: ASTactic and TacTok guide proof search with learned models, but a single trained model repeatedly misses similar regions of the proof space.
  - Key problem & insight: Training diversity produces complementary proof-search behavior even when the search mechanism is unchanged.
  - Proposed method — Diva with two components:
    1. **Diverse proof models**: Vary training choices to obtain models with different successful proof sets.
    2. **Independent proof searches**: Run model-guided searches and combine their verified successes; models do not exchange partial proofs.
  - Results: The reported Diva configuration proves 21.7% of theorems, versus 12.9% for TacTok and 12.3% for ASTactic—68% and 77% more theorems respectively.

- Learning Contract Invariants Using Reinforcement Learning [[ASE'22](https://www.cs.utexas.edu/~isil/cider.pdf)]
  - Background: Solidity arithmetic-safety checks need contract invariants; enumerative inference can be slow and leave unnecessary runtime checks.
  - Key problem & insight: Reward an invariant for the arithmetic obligations it discharges, not only for being inductive.
  - Proposed method — Cider with two components:
    1. **Policy-gradient learning**: Model invariant generation as an MDP and train a neural policy from the verifier's own successful and failed attempts.
    2. **Verifier-guided inference**: Generate likely invariants at deployment, independently validate them, and use them in refinement-type-based arithmetic verification.
  - Results: Discharges 97.1% of arithmetic checks versus 87.2% for SolidChc and 79.5% for Houdini; average verification is 1.8x faster than SolidChc in the reported evaluation.

- Passport: Improving Automated Formal Verification Using Identifiers [[TOPLAS'23](https://doi.org/10.1145/3593374)] [[arXiv'22](https://arxiv.org/abs/2204.10370)]
  - Background: ASTactic, Tac, and Tok underuse identifiers by discarding names or collapsing unfamiliar identifiers into an unknown token.
  - Key problem & insight: Names and namespaces encode information about the role and likely use of proof objects.
  - Proposed method — Passport with three encodings:
    1. **Category vocabulary indexing**: Distinguish identifier categories instead of treating all names alike.
    2. **Subword sequence modeling**: Recover useful structure from rare or unseen names.
    3. **Path elaboration**: Incorporate qualified paths to provide context beyond a local identifier.
  - Results: Proves 29% more theorems than the best base tool in head-to-head comparisons; combining the three enhanced tools proves 38% more than combining their unmodified versions.

- Can Large Language Models Reason about Program Invariants? [[ICML'23](https://proceedings.mlr.press/v202/pei23a.html)]
  - Background: Daikon-style dynamic invariant inference needs execution traces, which may be unavailable before running a program.
  - Key problem & insight: A code model can predict likely abstract program properties directly from source, using earlier predictions as reasoning context.
  - Proposed method — invariant prediction with two components:
    1. **Code-model fine-tuning**: Train on source-to-invariant examples to perform static prediction.
    2. **Scratchpad prediction**: Generate invariants sequentially through the program, exposing intermediate predictions to subsequent steps.
  - Results: Predicted invariant quality is comparable to dynamic analysis supplied with five execution traces. These are inferred candidate properties, not automatically a sound proof of program correctness.

- Loop Invariant Inference through SMT Solving Enhanced Reinforcement Learning [[ISSTA'23](https://doi.org/10.1145/3597926.3598047)] [[Appendix](https://zenodo.org/records/7970436/files/Loop%20Invariant%20Inference%20through%20SMT%20Solving%20Enhanced%20Reinforcement%20Learning-Appendix.pdf)]
  - Background: Code2Inv-style RL faces sparse success signals, while direct SMT search becomes expensive over nonlinear invariant templates.
  - Key problem & insight: Learn how to prune the hypothesis space while allowing the SMT solver to perform the symbolic search it handles well.
  - Proposed method — LIPuS with two components:
    1. **RL pruner**: Restrict a general nonlinear invariant search space before SMT solving.
    2. **Two-dimensional reward**: Incorporate solver feedback to learn useful pruning and the solver's capability boundary.
  - Results: The paper compares against Code2Inv, ICE-DT, GSpacer, SymInfer, ImplCheck, and Eldarica; its supplementary CHC experiment solves 89/209 filtered instances versus Eldarica's 81/209 under a five-minute limit.

- Ranking LLM-Generated Loop Invariants for Program Verification [[EMNLP'23](https://www.microsoft.com/en-us/research/publication/ranking-llm-generated-loop-invariants-for-program-verification/)] [[arXiv'23](https://arxiv.org/abs/2310.09342)]
  - Background: GPT-generated invariant lists contain useful candidates, but checking them in generation order wastes verifier calls.
  - Key problem & insight: Learn to rank candidates for the current program before paying for symbolic verification.
  - Proposed method — iRank with two components:
    1. **Contrastive ranking**: Pull valid program-invariant pairs together and separate incorrect candidates in the learned ranking space.
    2. **Rank-then-check**: Deduplicate and reorder generated candidates, leaving final inductiveness checking to the verifier.
  - Results: In the GPT-3.5 deduplicated-candidate experiment, iRank-ada reduces the mean rank of the first valid invariant from 65.24 to 18.79; lower rank reduces verification effort, though embedding overhead also matters.

- SmartInv: Multimodal Learning for Smart Contract Invariant Inference [[IEEE S&P'24](https://www.cs.columbia.edu/~junfeng/papers/smartinv/)]
  - Background: Pattern-based smart-contract analyzers miss business-logic bugs when the intended transaction behavior is not explicit in the code.
  - Key problem & insight: Infer properties from both code and natural-language transaction context, then check where the implementation violates them.
  - Proposed method — SmartInv with two components:
    1. **Tier of Thought (ToT)**: Fine-tune and prompt a foundation model to reason across source code and contextual descriptions before generating invariants.
    2. **Invariant checking**: Validate generated properties against the contract and use violations to localize suspicious behavior.
  - Results: Reports 119 previously unknown bugs; of eight sampled reports sent to developers, six were fixed and five confirmed as high severity. Bug counts are not equivalent to a completeness guarantee for the inferred properties.

- QEDCartographer: Automating Formal Verification Using Reward-Free Reinforcement Learning [[ICSE'25 — Extended version](https://www.alexsanchezstern.com/papers/qed-cartographer-icse2025-extended.pdf)]
  - Background: Supervised next-tactic models provide weak guidance about how close an unfinished proof is to completion.
  - Key problem & insight: Learn progress estimates that account for branching proof obligations rather than rely only on sparse complete-proof rewards.
  - Proposed method — QEDCartographer with two components:
    1. **Supervised tactic prediction**: Generate candidate Coq proof steps from proof-corpus training.
    2. **Reward-free reinforcement learning**: Learn a proof-progress estimator from the branching proof search and use it to prioritize search states.
  - Results: Proves 21.4% of CoqGym test theorems versus 19.8% for Proverbot9001 and 19.2% for Diva; the extended paper reports 26% shorter proofs and 27% lower time on jointly solved theorems.

#### Verifier-supervised synthesis and self-improvement

- Automated Proof Generation for Rust Code via Self-Evolution [[ICLR'25](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b2e20d7402c9985eae4ba924c65370a8-Abstract-Conference.html)]
  - Background: Open models have little exposure to Verus proofs, and human-written Rust proof corpora are too small for ordinary large-scale fine-tuning.
  - Key problem & insight: A verifier labels both successful proofs and failed attempts, supporting generation training and debugging training together.
  - Proposed method — SAFE with two components:
    1. **Self-evolving synthesis**: Generate proofs, retain verifier-accepted examples, fine-tune, and repeat to expand the training corpus.
    2. **Self-debugging**: Train on incorrect proofs plus verifier feedback so the model learns to repair its own failures.
  - Results: Achieves 52.52% proof-generation accuracy on the authors' expert-built benchmark versus 14.39% for GPT-4o; this comparison concerns that benchmark and configuration.

- Towards Neural Synthesis for SMT-Assisted Proof-Oriented Programming [[ICSE'25](https://www.microsoft.com/en-us/research/publication/towards-neural-synthesis-for-smt-assisted-proof-oriented-programming/)] [[arXiv'24](https://arxiv.org/abs/2405.01787)]
  - Background: F* mixes programs and proofs and delegates many obligations to SMT, but still requires experts to construct typed definitions and select useful premises.
  - Key problem & insight: Treat each top-level definition as a type-directed synthesis problem with a reproducible F* checker.
  - Proposed method — F* synthesis with two components:
    1. **FStarDataSet**: Extract specifications, definitions, context, and checker support from production-related F* projects.
    2. **Fine-tuning and premise retrieval**: Train smaller code models and augment prompts using type-based retrieval of relevant definitions.
  - Results: The extended corpus contains approximately 940k lines and 54k definitions; on its cross-project evaluation, fine-tuned StarCoder reaches 58.13% verify@10 versus 41.63% for GPT-3.5.

- Re:Form -- Reducing Human Annotations in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny [[arXiv'25](https://arxiv.org/abs/2507.16331)]
  - Background: RL for verified programming is limited by scarce annotated demonstrations and the difficulty of producing initially valid formal-language programs.
  - Key problem & insight: Automatically construct Dafny training tasks, bootstrap syntax and proof competence with SFT, then refine using verifier feedback.
  - Proposed method — Re:Form with two stages:
    1. **Data curation and SFT**: Build formal-program examples and teach models to generate Dafny implementations and annotations.
    2. **Regularized RL**: Use formal verification feedback while retaining regularization to improve generalization beyond the supervised corpus.
  - Results: On the paper's 300-task out-of-distribution DafnyComp subset, the 14B RL model reaches 14.0% Pass@1 versus 8.3% for its SFT counterpart and 2.7% for the Claude data-generator baseline; the study also demonstrates initial verifiable-code competence with a 0.5B model.

- SpecRL: Reinforcement Learning with Test-Based Completeness Rewards for Formal Specification Synthesis [[arXiv'26](https://arxiv.org/abs/2604.05820)]
  - Background: A verifier can accept `ensures true`; rewarding verification success alone encourages weak specifications that say little about the implementation.
  - Key problem & insight: Add negative input-output examples that distinguish useful specifications from vacuous ones.
  - Proposed method — SpecRL with two components:
    1. **Spectests**: Construct implementation-impossible input-output pairs that an underspecified contract may still allow.
    2. **Completeness reward**: For verifier-accepted candidates, reward the fraction of spectests rejected by the generated specification.
  - Results: On out-of-distribution DafnyComp-Spec, the 7B model improves verification success by 49.96% and empirical completeness by 26.46% relative to SFT. Spectests improve measured completeness; they do not establish logical completeness.

- Formal Disco: Scalable Open-Ended Generation of Formally Verified Programs [[arXiv'26](https://arxiv.org/abs/2607.04631)]
  - Background: Self-training is constrained by a small seed corpus and can keep regenerating similar easy programs.
  - Key problem & insight: Separate the creation, repair, and extension of verified programs, then train for both success and diversity.
  - Proposed method — Formal Disco with three worker roles:
    1. **Initiators**: Use repository READMEs and documentation to propose new verification tasks and programs.
    2. **Fixers**: Repair candidates using compiler and verifier diagnostics.
    3. **Extenders**: Expand already verified programs; collect trajectories for distillation and iterative SFT with an entropy-maximization objective.
  - Results: Produces datasets for Dafny, Verus, and Frama-C; the final Qwen generation's largest verified examples exceed the largest Claude seed examples by 22–62% in lines of code across those languages, alongside downstream verification evaluations.

- Propose, Solve, Verify: Self-Play Through Formal Verification [[ICML'26](https://icml.cc/virtual/2026/poster/63571)]
  - Background: Expert iteration on a fixed problem set eventually runs out of new solvable examples, while test-only self-play can reinforce incorrect solutions.
  - Key problem & insight: Couple a difficulty-aware problem proposer to a solver, using formal verification as the acceptance signal.
  - Proposed method — Propose, Solve, Verify (PSV) with two learned roles:
    1. **Proposer**: Generate new formal programming tasks calibrated to the current solver's ability.
    2. **Solver**: Attempt the tasks, retain verified solutions, and improve through expert iteration before the next proposal round.
  - Results: PSV-Verus improves Pass@1 by up to 9.6x over the paper's inference-only and expert-iteration baselines across three benchmarks; gains depend on both verification and difficulty-aware proposal.

#### Formal constraints for learned security classifiers

- Learning Security Classifiers with Verified Global Robustness Properties [[CCS'21](https://people.eecs.berkeley.edu/~daw/papers/verif-ccs21.pdf)]
  - Background: Local robustness certificates cover neighborhoods of selected inputs and can leave security classifiers exposed under distribution shift.
  - Key problem & insight: Train classifiers to satisfy universally quantified security properties, with counterexamples identifying violations during learning.
  - Proposed method — booster-fixer training with two components:
    1. **Booster**: Increase predictive capacity using an ensemble of logic rules.
    2. **Fixer**: Apply counterexample-guided inductive synthesis and formal property checking to repair violations of global robustness requirements.
  - Results: Evaluates three security datasets; a Twitter-spam classifier satisfies five global properties with a reported 5.4% decrease in true-positive rate. This is the reverse direction—formal verification constraining an AI model—rather than AI generating software proofs.

<a id="code-agent"></a>

### Agent

#### Verified synthesis and annotation repair

- Clover: Closed-Loop Verifiable Code Generation [[SAIV'24; cited by 36](https://theory.stanford.edu/~barrett/pubs/SSP%2B24-abstract.html)] [[arXiv'23](https://arxiv.org/abs/2310.17807)]
  - Background: Generating code and annotations together can produce a self-consistent pair that still disagrees with the user's description.
  - Key problem & insight: Check agreement among three artifacts—implementation, formal annotations, and docstring—instead of trusting a single generation.
  - Proposed method — Clover with two components:
    1. **Consistency checks**: Use an LLM to reconstruct or compare artifacts across code, annotations, and natural-language descriptions.
    2. **Formal verification**: Check implementation-specification consistency with Dafny and reject candidates that fail the combined checks.
  - Results: Accepts up to 87% of correct CloverBench instances with no false positives on the tested adversarial incorrect instances; also identifies six incorrect programs in MBPP-DFY-50. The finite evaluation does not prove universal docstring fidelity.

- Towards AI-Assisted Synthesis of Verified Dafny Methods [[FSE'24; cited by 34](https://2024.esec-fse.org/details/fse-2024-research-papers/75/Towards-AI-Assisted-Synthesis-of-Verified-Dafny-Methods)]
  - Background: Directly prompting GPT-4 or PaLM-2 to write Dafny often produces code that cannot be verified.
  - Key problem & insight: Expose verified examples and an explicit problem-solving decomposition rather than provide only a natural-language request or signature.
  - Proposed method — Dafny synthesis study with two components:
    1. **Prompt variants**: Compare Contextless, Signature, and retrieval-augmented Chain-of-Thought prompts.
    2. **Verification and human evaluation**: Check generated implementations in Dafny and inspect whether their specifications capture the intended task.
  - Results: On 178 MBPP problems, GPT-4 reaches 58% with retrieval-augmented CoT versus 19% with Contextless and 10% with Signature prompts; releases 153 verified solutions, including 50 manually written ones.

- Enchanting Program Specification Synthesis by Large Language Models Using Static Analysis and Program Verification [[CAV'24](https://doi.org/10.1007/978-3-031-65630-9_16)]
  - Background: Existing invariant synthesizers often specialize in numeric loops and struggle with arrays, pointers, nested loops, and function calls.
  - Key problem & insight: Use program structure and failed verification obligations to control what an LLM should specify next.
  - Proposed method — AutoSpec with three components:
    1. **Static analysis**: Locate program structures and specification needs before generation.
    2. **Program decomposition**: Present manageable verification-relevant pieces to the LLM for candidate annotation synthesis.
    3. **Incremental validation**: Check each round's candidates and feed unresolved obligations into the next round to prevent error accumulation.
  - Results: Verifies 79% of evaluated programs through automatic specification synthesis, a reported 1.592x improvement over prior work; also applies the approach to an X509-parser project.

- Guiding Enumerative Program Synthesis with Large Language Models [[CAV'24](https://doi.org/10.1007/978-3-031-65630-9_15)]
  - Background: Enumerative SyGuS solvers respect exact logical specifications but can explore a large grammar; a standalone LLM often fails those exact constraints.
  - Key problem & insight: Use the LLM to shape the enumerator's search distribution instead of replacing symbolic synthesis.
  - Proposed method — LLM-guided enumeration with three components:
    1. **Standalone LLM synthesis**: Try to generate a complete program directly from the formal specification.
    2. **pCFG-synth**: Use LLM-derived grammar weights to guide probabilistic enumeration, with both enumeration-based and A* variants.
    3. **iLLM-synth**: Alternatively interleave LLM queries with search so the model can react to enumerator progress.
  - Results: On 609 SyGuS tasks, standalone LLM plus A*-pCFG-synth solves 80.1%, versus 49.8% for the LLM and 68.1% for cvc5; the interleaved A*-iLLM-synth variant reaches 67.0%. The best result comes from the combined offline-guidance configuration.

- VerMCTS: Synthesizing Multi-Step Programs using a Verifier, a Large Language Model, and Tree Search [[MATH-AI@NeurIPS'24](https://namin.seas.harvard.edu/pubs/vermcts.pdf)] [[arXiv'24](https://arxiv.org/abs/2402.08147)]
  - Background: Sampling complete programs wastes tokens on branches whose partial code is already inconsistent with the target specification.
  - Key problem & insight: Intermediate verification feedback can prune search before the whole program is generated.
  - Proposed method — VerMCTS with two components:
    1. **LLM expansion**: Generate candidate next fragments within a Monte Carlo tree search over Dafny or Coq programs.
    2. **Verifier-guided search values**: Check partial programs and use the feedback to constrain branch values and allocate further generation.
  - Results: On the paper's multi-step suite, average pass@5000 improves by more than 30 percentage points over repeated base-model sampling; the metric fixes sampled-token budget rather than attempt count.

- dafny-annotator: AI-Assisted Verification of Dafny Programs [[arXiv'24; citations N/A (method-paper set)](https://arxiv.org/abs/2411.15143)]
  - Background: Small open models rarely produce the annotations required to make existing Dafny code verify, and human training examples are scarce.
  - Key problem & insight: Combine annotation search with synthetic verified-program generation to improve the model that guides the search.
  - Proposed method — dafny-annotator with two components:
    1. **Annotation search**: Add candidate logical hints and retain progress according to Dafny feedback.
    2. **DafnySynth**: Generate, implement, and extend new program ideas while using Dafny to filter the resulting training examples.
  - Results: Llama-3.1-8B's success on the evaluated DafnyBench subset increases from 15.7% to 50.6% after training on DafnySynth plus DafnyBench.

- AutoVerus: Automated Proof Generation for Rust Code [[OOPSLA'25; cited by 10](https://doi.org/10.1145/3763174)] [[arXiv'24](https://arxiv.org/abs/2409.13082)]
  - Background: Verus has Rust-specific proof syntax and recurring proof failures that generic code-generation prompts do not handle well.
  - Key problem & insight: Match the orchestration to how verification experts construct, strengthen, and debug proofs.
  - Proposed method — AutoVerus with three phases:
    1. **Preliminary proof generation**: Produce an initial proof for the given Rust code and specification.
    2. **Proof refinement**: Apply general verification tips to strengthen proof structure and missing annotations.
    3. **Proof debugging**: Route Verus errors into targeted repair prompts until the proof checks or the budget expires.
  - Results: Verifies more than 90% of a 150-task benchmark; over half of the tasks finish in under 30 seconds or three LLM calls.

- AlphaVerus: Bootstrapping Formally Verified Code Generation through Self-Improving Translation and Treefinement [[ICML'25](https://arxiv.org/abs/2412.06176)]
  - Background: Verus lacks the verified examples available in higher-resource verification languages.
  - Key problem & insight: Translate verified source programs, repair them with the target verifier, and recycle successful translations as examples.
  - Proposed method — AlphaVerus with three phases:
    1. **Exploration**: Generate candidate translations from Dafny into Verus.
    2. **Treefinement**: Search over repairs using verifier feedback rather than commit to a single linear repair trajectory.
    3. **Filtering**: Reject misaligned specifications and programs that exploit the verification objective; reuse accepted artifacts as in-context examples.
  - Results: Llama-3.1-70B reaches 33% on Verified-HumanEval in the reported setup without weight fine-tuning. “Self-improving” here refers to the translation/example pipeline, not gradient updates.

- Verifying LLM-Generated Code in the Context of Software Verification with Ada/SPARK [[arXiv'25](https://arxiv.org/abs/2502.07728)]
  - Background: SPARK can verify Ada programs, but developers still need to supply contracts and auxiliary annotations.
  - Key problem & insight: Evaluate annotation restoration inside an existing industrial verification language rather than require a new proof formalism.
  - Proposed method — Marmaragan with two components:
    1. **Annotation removal**: Curate SPARK programs and selectively remove annotations to expose specific verification tasks.
    2. **LLM-assisted completion**: Ask GPT-4o to regenerate the missing annotations and check the resulting programs through SPARK's verification workflow.
  - Results: Correct annotations are generated for 50.7% of benchmark cases in the reported GPT-4o configuration.

- Towards Formal Verification of LLM-Generated Code from Natural Language Prompts [[arXiv'25](https://arxiv.org/abs/2507.13290)]
  - Background: A generated Ansible program can look plausible while violating the intended system-administration task.
  - Key problem & insight: Introduce a readable formal query that the user can confirm, then verify generated code against that explicit contract.
  - Proposed method — Astrogator with three components:
    1. **Formal Query Language**: Represent user intent in a formally defined, natural-language-like form.
    2. **Knowledge Base**: Supply system-specific implementation dependencies without forcing users to encode every detail.
    3. **Symbolic verification**: Use an Ansible calculus, symbolic interpreter, and unification algorithm to check generated programs.
  - Results: On 21 code-generation tasks, verifies correct code in 83% of cases and identifies incorrect code in 92%; user confirmation of the query is part of the intent-alignment boundary.

- DafnyPro: LLM-Assisted Automated Verification for Dafny Programs [[arXiv'26](https://arxiv.org/abs/2601.05385)]
  - Background: An annotation-repair loop can silently change executable code, accumulate unnecessary invariants, or repeat an unhelpful proof strategy.
  - Key problem & insight: Enforce the repair boundary and provide reusable proof guidance alongside error feedback.
  - Proposed method — DafnyPro with three components:
    1. **Diff-checker**: Reject modifications to the base program logic.
    2. **Pruner**: Remove unnecessary invariants that obstruct or complicate verification.
    3. **Hint augmentation**: Retrieve predefined, problem-independent strategies to guide subsequent proof attempts.
  - Results: Claude Sonnet 3.5 reaches 86% on DafnyBench, 16 percentage points above its base setup; distilled 7B and 14B models reach 68% and 70% respectively.

#### Invariant synthesis and solver-guided search

- Lemur: Integrating Large Language Models in Automated Program Verification [[ICLR'24](https://arxiv.org/abs/2310.04870)]
  - Background: SMT-based program verifiers can stall when they lack useful intermediate facts; unconstrained LLM reasoning provides no soundness guarantee.
  - Key problem & insight: Let the LLM propose intermediate properties while a symbolic verifier controls which deductions can establish the goal.
  - Proposed method — Lemur with two components:
    1. **Verification calculus**: Define sound transition rules for introducing, checking, and using intermediate program properties.
    2. **LLM-guided search**: Request candidate properties and revise them in response to verification outcomes, keeping accepted conclusions under symbolic control.
  - Results: With GPT-4, solves 25/47 selected hard SV-COMP tasks under a 15-minute limit versus one each for ESBMC and UAutomizer. The subset was selected for difficulty for those symbolic baselines, so the result is not an overall SV-COMP ranking.

- LLM Meets Bounded Model Checking: Neuro-symbolic Loop Invariant Inference [[ASE'24](https://doi.org/10.1145/3691620.3695014)]
  - Background: An LLM may fail to generate the complete invariant even when its responses contain all the needed predicates.
  - Key problem & insight: Recover useful predicates across attempts and combine them symbolically.
  - Proposed method — LaM4Inv's query-filter-reassemble loop:
    1. **Query**: Generate several candidate invariants with an LLM without task-specific training.
    2. **Filter and reassemble**: Use bounded model checking to filter predicates, rebuild candidate invariants, and check them with SMT; return failures to the next prompt.
  - Results: Solves 309/316 invariant-generation tasks versus at most 219 for the evaluated baselines. BMC filters candidates; the final invariant obligations supply the proof check.

- LLM-Generated Invariants for Bounded Model Checking Without Loop Unrolling [[ASE'24](https://pure.manchester.ac.uk/ws/portalfiles/portal/344502006/ESBMC_and_Vampire.pdf)]
  - Background: Bounded model checking can repeatedly unroll a loop without reaching the depth needed to establish a safety property.
  - Key problem & insight: Replace loops by sound abstractions based on independently proved invariants.
  - Proposed method — ESBMC ibmc with two components:
    1. **LLM invariant proposals**: Generate candidate facts about loop behavior.
    2. **Vampire-backed transformation**: Prove the required invariant conditions with Vampire, transform the loop into a loop-free control-flow fragment, and verify it with ESBMC.
  - Results: The reported best-configuration comparison verifies 101 programs, versus 46 for ESBMC k-induction, 73 for VeriAbs, and 115 for SeaHorn; ibmc also solves two cases not solved by the other compared configurations.

- Clause2Inv: A Generate-Combine-Check Framework for Loop Invariant Inference [[ISSTA'25](https://conf.researchr.org/details/issta-2025/issta-2025-papers/44/Clause2Inv-A-Generate-Combine-Check-Framework-for-Loop-Invariant-Inference)]
  - Background: Repeated guess-and-check can miss a correct invariant because the required clauses occur in different failed guesses.
  - Key problem & insight: Separate discovering useful clauses from choosing their logical combination.
  - Proposed method — Clause2Inv with two components:
    1. **LLM-based clause generator**: Accumulate clauses across proposed invariants.
    2. **Counterexample-driven clause combinator**: Use verification counterexamples to choose combinations and submit reconstructed invariants for checking.
  - Results: Solves 312/316 linear and 44/50 nonlinear tasks, at least 93 and 16 more than the evaluated baselines respectively; the combinator can also wrap existing generators.

- Quokka: Accelerating Program Verification with LLMs via Invariant Synthesis [[arXiv'25](https://arxiv.org/abs/2509.21629)]
  - Background: Earlier LLM invariant pipelines often combine many generated predicates through substantial symbolic post-processing.
  - Key problem & insight: A useful invariant must both hold and help establish the target assertion; validate these obligations directly.
  - Proposed method — Quokka with two components:
    1. **Invariant synthesis and checking**: Ask an LLM for candidate invariants and retain candidates only through sound verification checks.
    2. **Training and sampling**: Study supervised fine-tuning and Best-of-N generation on SV-COMP-derived tasks.
  - Results: Releases 866 evaluation and 3,589 training instances and evaluates nine LLMs; both fine-tuning and additional samples improve verification. Earlier versions used the title “InvBench.”

#### Specification generation, model construction, and analysis

- Can Large Language Models Transform Natural Language Intent into Formal Method Postconditions? [[FSE'24; cited by 1](https://www.microsoft.com/en-us/research/publication/formalizing-natural-language-intent-into-program-specifications-via-large-language-models/)] [[arXiv'23](https://arxiv.org/abs/2310.01831)]
  - Background: Code comments describe intended behavior, but existing implementations may violate that intent; checking only the implementation cannot recover the missing contract.
  - Key problem & insight: Translate natural-language intent into executable postconditions and assess both correctness and bug-discriminating power.
  - Proposed method — nl2postcond with two components:
    1. **Postcondition generation**: Prompt an LLM to convert documentation into program assertions.
    2. **Correctness and discriminative-power metrics**: Evaluate whether assertions accept intended behavior and reject faulty implementations.
  - Results: Generated postconditions detect 64 historical Defects4J bugs. This is an important specification-generation precursor; its assertion-based evaluation is not an end-to-end formal proof of the programs.

- PAT-Agent: Autoformalization for Model Checking [[ASE'25](https://conf.researchr.org/details/ase-2025/ase-2025-papers/200/PAT-Agent-Autoformalization-for-Model-Checking)] [[arXiv'25](https://arxiv.org/abs/2509.23675)]
  - Background: Directly generating a model-checker input from prose often produces syntactically valid but semantically incorrect models.
  - Key problem & insight: Separate the modeling plan from code generation, then use model-checking counterexamples to drive repair.
  - Proposed method — PAT-Agent with three components:
    1. **Planning LLM**: Extract entities, state, events, and required behaviors into a structured modeling plan.
    2. **Code-Generation LLM**: Translate the plan into CSP# for the Process Analysis Toolkit (PAT).
    3. **Verification-Repair Loop**: Check user-specified properties and use counterexamples to revise the model; expose the workflow through an interactive interface.
  - Results: Reports 100% verification success across 40 tasks in its evaluation. Passing the encoded properties remains distinct from independently establishing that all natural-language requirements were modeled faithfully.

- Automating Requirements Formalization: Using LLMs and Low-Complexity Distinguishing Traces for Semantic Validation [[ICSE'26](https://cs.stanford.edu/~trippel/pubs/mendoza_ICSE26.pdf)]
  - Background: An LLM-generated temporal-logic formula can be syntactically correct but costly for a user to validate semantically.
  - Key problem & insight: Ask users about simple distinguishing behaviors instead of requiring them to inspect full formulas.
  - Proposed method — ARTEMIS with three components:
    1. **Structured natural language**: Translate requirements into FRETish, whose mapping to temporal logic is unambiguous.
    2. **Sub-specification generation**: Construct low-complexity traces corresponding to candidate specification fragments.
    3. **Distinguishing traces**: Present behaviors that separate candidate meanings and use the responses to refine the specification.
  - Results: Reports 1.57x higher translation accuracy and up to 10.38x lower validation effort than the compared baselines on real-world requirements; human semantic validation remains part of the workflow.

- Event-B Agent: Towards LLM Agent for Formal Model Synthesis and Repair [[FSE'26](https://hongshuw.github.io/files/Event_B_Agent%28FSE26%29.pdf)] [[arXiv'26](https://arxiv.org/abs/2605.17475)]
  - Background: Model synthesis without proof-guided revision can produce Event-B models that neither satisfy requirements nor discharge their proof obligations.
  - Key problem & insight: Evolve the formal model and its proofs together through successive refinement steps.
  - Proposed method — Event-B Agent with three stages:
    1. **Refinement Strategy Planning**: Distribute requirements across abstraction and refinement levels.
    2. **Model Synthesis**: Generate structured Event-B artifacts and repair well-formedness errors inside Rodin.
    3. **Model & Proof Repair**: Use undischarged obligations, model checking, and repair operations to revise both the model and proof artifacts.
  - Results: Discharges 97.86% of proof obligations versus 95.56% for adapted PAT-Agent and 90.07% for Cursor; refinement-specific discharge is 92.56%, so complete refinement correctness is not established for every case.

- ADARULE: LLM-Driven Natural Language to LTL Conversion via Pattern-Adaptive Rule Induction [[ICSE'26](https://conf.researchr.org/details/icse-2026/icse-2026-research-track/293/ADARULE-LLM-Driven-Natural-Language-to-LTL-Conversion-via-Pattern-Adaptive-Rule-Indu)]
  - Background: NL-to-LTL translators rely on hand-written patterns that transfer poorly across domains and linguistic conventions.
  - Key problem & insight: Translation errors reveal which pattern-specific rules are missing from the prompt.
  - Proposed method — ADARULE with two components:
    1. **Translator**: Generate LTL using general rules plus the currently learned rule set.
    2. **Learner**: Compare failed translations with reference formulas, induce corrective rules, and add them to subsequent translator prompts.
  - Results: Evaluated on four datasets and three foundation models; reports at least 21.1% average improvement over the compared baselines. The adaptation uses reference formulas and changes prompt rules rather than model weights.

- ABSINT-AI: Agentic Heap Abstractions for Abstract Interpretation [[ICML'26](https://openreview.net/forum?id=ozu9ZRETYE)]
  - Background: TAJS/WALA-style analyses use fixed heap abstractions that can merge unrelated JavaScript objects and produce false positives.
  - Key problem & insight: Let an LLM choose abstractions through a restricted interface while the abstract interpreter retains responsibility for sound state transfer.
  - Proposed method — ABSINT-AI with two components:
    1. **Adaptive heap abstractions**: Use names and access patterns to choose different abstractions for different objects.
    2. **Sound abstract interpretation**: Restrict agent decisions to permitted abstraction choices; do not let the agent directly invent or edit abstract states.
  - Results: Reports up to 34% fewer false positives than fixed-abstraction analyses while retaining formal guarantees; agentic interaction improves over non-agentic LM predictions by 25% in the paper's comparison.

- Towards Verified Code Reasoning by LLMs [[arXiv'25](https://arxiv.org/abs/2509.26546)]
  - Background: Code-reasoning agents may confidently explain a bug or claim program equivalence without evidence that their intermediate claims hold.
  - Key problem & insight: Translate an explanation into checkable claims and validate the reasoning steps with program-analysis tools.
  - Proposed method — verified code reasoning with two components:
    1. **Formal claim extraction**: Recover a formal representation of the agent's explanation or equivalence argument.
    2. **Tool-based validation**: Check the extracted steps with formal verification and program analysis before accepting the answer.
  - Results: Validates reasoning for 13/20 uninitialized-variable cases and catches 6/8 incorrect judgments in a separate 20-query equivalence study; the evidence is a small feasibility evaluation.

#### Repository and systems proofs

- Baldur: Whole-Proof Generation and Repair with Large Language Models [[ESEC/FSE'23](https://2023.esec-fse.org/details/fse-2023-research-papers/2/Baldur-Whole-Proof-Generation-and-Repair-with-Large-Language-Models)]
  - Background: Thor and earlier learned provers typically expand one tactic at a time, requiring costly proof-space search.
  - Key problem & insight: Generate an entire proof first, then condition a separate repair model on the failed attempt and checker message.
  - Proposed method — Baldur with two components:
    1. **Whole-proof generation**: Fine-tune a transformer on Isabelle/HOL proofs and generate complete proof scripts.
    2. **Proof repair**: Fine-tune on failed proofs and their errors to produce corrected scripts.
  - Results: Evaluated on 6,336 Isabelle/HOL theorems; Baldur finds proofs for an additional 8.7% of theorems beyond Thor, and their combined coverage reaches 65.7%.

- Proof Automation with Large Language Models [[ASE'24](https://conf.researchr.org/details/ase-2024/ase-2024-research/121/Proof-Automation-with-Large-Language-Models)] [[arXiv'24](https://arxiv.org/abs/2409.14274)]
  - Background: LLMs often identify the right proof structure but fail on local tactic arguments, names, or other formal details.
  - Key problem & insight: Preserve a useful high-level proof and repair low-level failures with symbolic methods.
  - Proposed method — PALM with two components:
    1. **Generate**: Ask an LLM for an initial complete Coq proof rather than enumerate every tactic from scratch.
    2. **Repair**: Diagnose rejected proof steps and apply targeted symbolic repairs, checking the result with Coq.
  - Results: The formative study analyzes 520 GPT-3.5 errors; on over 10k theorems, PALM proves 76.6–180.4% more than the compared methods and solves 1,270 theorems missed by existing approaches.

- Rango: Adaptive Retrieval-Augmented Proving for Automated Software Verification [[ICSE'25; cited by 7](https://www.cs.cornell.edu/~lerner/papers/rango-final.pdf)] [[arXiv'24](https://arxiv.org/abs/2412.14063)]
  - Background: Static retrieval misses the changing relevance of lemmas as a proof progresses and often ignores reusable proof examples from the current project.
  - Key problem & insight: Retrieve both premises and analogous proofs anew at each proof state.
  - Proposed method — Rango with two components:
    1. **Adaptive retrieval**: Select relevant definitions, lemmas, and proof examples from the available project context.
    2. **Fine-tuned proof search**: Condition the next-step model on these examples, check candidates in Coq, and repeat retrieval after state changes.
  - Results: Releases CoqStoq with 2,226 projects and 196,929 theorems; proves 32.0% on the curated evaluation, 29% more theorems than Tactician.

- VeruSAGE: A Study of Agent-Based Verification for Rust Systems [[arXiv'25; citations N/A (method-paper set)](https://arxiv.org/abs/2512.18436)]
  - Background: Performance on textbook Verus tasks does not show whether agents can work with systems-specific libraries, dependencies, and proof conventions.
  - Key problem & insight: Evaluate agent scaffolds and model choices on proofs extracted from actual verified Rust systems.
  - Proposed method — VeruSAGE with two components:
    1. **VeruSAGE-Bench**: Extract proof-completion tasks while retaining their systems context.
    2. **Agent-system study**: Compare models and tool configurations, including access to contextual information and verification feedback.
  - Results: 849 tasks from eight systems; the best evaluated configuration completes over 80%, and over 90% on a separate set of previously unfinished proof tasks. These are proof-task results under supplied verification context.

- Cobblestone: A Divide-and-Conquer Approach for Automating Formal Verification [[ICSE'26](https://doi.org/10.1145/3744916.3773178)] [[arXiv'24](https://arxiv.org/abs/2410.19940)]
  - Background: Whole-proof generation discards useful progress when a script contains a few unproved or incorrect subparts.
  - Key problem & insight: Treat a partially successful proof as a decomposition into verified pieces and remaining obligations.
  - Proposed method — Cobblestone with three components:
    1. **Whole-proof sampling**: Obtain a candidate proof and first exploit existing CoqHammer automation.
    2. **Fail-safe localization**: Identify the subparts that check and isolate the unresolved obligations.
    3. **Recursive completion**: Reinvoke the solver on those obligations and assemble a final proof checked by Coq.
  - Results: Evaluated on four benchmarks; average reported run cost is $1.25 and 14.7 minutes. An additional oracle-assisted setting reaches up to 58% success and must be distinguished from fully automatic performance.

- ProofCoop: Collaborative Automated Formal Verification [[ICSE'26](https://people.cs.umass.edu/brun/pubs/pubs/Kaufman26icse.pdf)]
  - Background: Diva runs diverse models independently, so one model cannot use another's partial progress during search.
  - Key problem & insight: Make model diversity useful inside a shared proof search, not only when combining final success sets.
  - Proposed method — ProofCoop with two groups of collaboration mechanisms:
    1. **Prediction collaboration**: Combine next-tactic proposals using union, certainty bidding, voting, or a learned stacking model.
    2. **Proof collaboration**: Share proved lemmas during search and let models complete one another's partial proofs.
  - Results: With the same five component models, proves 3,937 CoqGym test theorems (33.0%) versus Diva's 3,287 (27.5%), a 19.8% relative increase.

- Neuro-Symbolic Proof Generation for Scaling Systems Software Verification [[OSDI'26](https://www.usenix.org/conference/osdi26/presentation/he-baoding)]
  - Background: Large systems proofs require extensive project knowledge, and neural next-step generation wastes search on rejected or unproductive states.
  - Key problem & insight: Couple learned proposals with the proof assistant's symbolic repair, pruning, and subgoal-solving capabilities.
  - Proposed method — neuro-symbolic proof generation with three components:
    1. **Proof-state training**: Fine-tune LLMs on state–step pairs from verification developments.
    2. **Best-first search**: Expand promising checked proof states rather than repeatedly generate whole scripts.
    3. **Isabelle integration**: Expose fine-grained state through a REPL and invoke symbolic tools to repair tactics, rank/filter states, and discharge stalled subgoals.
  - Results: Proves up to 77.6% of theorems on the evaluated FVEL seL4 benchmark; this measures completion of supplied obligations within an existing development.

- Agentic Verification of Software Systems [[FSE'26](https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/110/Agentic-Verification-of-Software-Systems)] [[arXiv'25](https://arxiv.org/abs/2511.17330)]
  - Background: Proof models trained on existing libraries can struggle with the long, structured obligations produced by software verification.
  - Key problem & insight: Let an agent inspect the evolving Rocq proof tree and retrieve context when repeated tactic failures indicate missing information.
  - Proposed method — AutoRocq with three components:
    1. **Context-Aware Tactic Generation**: Use the program and proof state to propose the next step.
    2. **Proof Tree-Aware Interpretation**: Track branching obligations and coordinate their completion.
    3. **Context-Assisted Feedback Handling**: Repair local failures, search for missing context, and reuse successful proof history during the run.
  - Results: On 60 Linux-kernel verification lemmas, proves 12/60 versus Copra's 7/60; adding CoqHammer raises AutoRocq to 18/60. These counts do not mean the whole kernel was verified.

- VeriStruct: AI-assisted Automated Verification of Data-Structure Modules in Verus [[TACAS'26; citations N/A (method-paper set)](https://www.microsoft.com/en-us/research/publication/veristruct/)] [[arXiv'25](https://arxiv.org/abs/2510.25015)]
  - Background: Proving isolated functions is insufficient for data structures whose methods share an abstraction, representation invariant, and cross-method contracts.
  - Key problem & insight: Plan the specification and proof artifacts at module scope before repairing individual functions.
  - Proposed method — VeriStruct with three components:
    1. **Planner**: Coordinate abstractions, type invariants, method specifications, and proof-code generation.
    2. **Syntax guidance**: Provide Verus-specific annotation and semantic guidance in prompts.
    3. **Repair stage**: Correct annotation and verification errors while maintaining the module's shared proof context.
  - Results: Verifies ten of eleven Rust data-structure modules and 128/129 functions (99.2%) in the reported evaluation.

- ExVerus: Verus Proof Repair via Counterexample Reasoning [[ICML'26; citations N/A (method-paper set)](https://icml.cc/virtual/2026/poster/65247)] [[arXiv'26](https://arxiv.org/abs/2603.25810)]
  - Background: Textual verifier errors often omit the concrete behavior that explains why a proposed invariant fails.
  - Key problem & insight: Use validated source-level counterexamples to help the LLM generalize a failure into a stronger invariant.
  - Proposed method — ExVerus with two components:
    1. **Counterexample generation and validation**: Recover concrete behavior associated with a failed proof and check that the example is meaningful.
    2. **Counterexample-guided repair**: Ask the LLM to explain and block the failure through inductive proof annotations, then rerun Verus.
  - Results: Reports 38% more solved tasks on average than AutoVerus, with approximately $0.04 average cost per task and 4.25x lower cost. This is a within-paper comparison; the paper also documents that Verus-version differences affect reproduced AutoVerus scores.

- IsabeLLM: Automated Theorem Proving Applied to Formally Verifying Consensus [[arXiv'26](https://arxiv.org/abs/2606.18098)]
  - Background: General Isabelle prompting provides insufficient local context for consensus-protocol proofs.
  - Key problem & insight: Supply retrieved library material, traced errors, and counterexamples around the current proof attempt.
  - Proposed method — IsabeLLM-RAG with two components:
    1. **Context retrieval and feedback**: Retrieve relevant facts and augment prompts with error traces and counterexample information.
    2. **Isabelle/Sledgehammer loop**: Generate and repair proofs against the updated toolchain.
  - Results: Evaluates 16 non-trivial Bitcoin Proof-of-Work lemmas with repeated attempts; reports 94.4% success for its Chimera configuration. This is a targeted consensus-proof study, not verification of a complete cryptocurrency implementation.

<a id="smart-contract-agents"></a>

#### Smart contracts

- Augmenting Smart Contract Decompiler Output Through Fine-Grained Dependency Analysis and LLM-Facilitated Semantic Recovery [[TSE'25](https://doi.org/10.1109/TSE.2025.3623325)] [[arXiv'25](https://arxiv.org/abs/2501.08670)]
  - Background: Gigahorse-style decompilers lose function boundaries, variable types, and contract attributes; unconstrained LLM edits can change program behavior.
  - Key problem & insight: Recover readable semantics using dependency information, while independently rejecting behavior-changing edits.
  - Proposed method — SmartHalo with three components:
    1. **Dependency Graph Construction**: Represent type, state, and control-flow dependencies to select relevant code context.
    2. **LLM-driven Semantic Enrichment**: Guide recovery with dependency-derived context, reasoning steps, and candidate types or attributes.
    3. **Correctness Verification**: Apply a **Program-behavior Equivalence Check** using symbolic summaries and Z3, plus a **Rule-based Type Violation Check**; return violations for revision.
  - Results: The revised arXiv version reports GPT-4o-mini precision of 91.32% for boundaries, 90.40% for types, and 80.66% for attributes. Downstream reentrancy-analysis precision improves from 72.16% with SliSE to 80.41% with SliSE+SmartHalo. Equivalence is checked against the initial decompiler output, not directly against original bytecode.

- Towards Trustworthy Smart Contract Synthesis: A Multi-Agent Framework with Lean-Based Verification [[ACL'26](https://aclanthology.org/2026.acl-long.1836/)]
  - Background: Direct and FSM-guided Solidity generation do not establish that generated implementations satisfy security properties.
  - Key problem & insight: Combine proof feedback with adversarial traces, using discovered attacks to expand the properties being checked.
  - Proposed method — LeVer with three components:
    1. **Coder**: Generate Solidity from requirements and repair it using combined feedback.
    2. **Neuro-Symbolic Verifier**: Select properties, translate Solidity into Lean state transitions, and generate checked proofs through PropertySelector, LeanFormalizer, and proof search.
    3. **Adversarial Attacker**: Search sandbox executions for violations; the **Attack-to-Property** mechanism converts traces into additional verification targets.
  - Results: With Gemini-3-Pro, sandbox attack success falls from 29.0% for direct generation to 2.4%, while verified-property rate rises from 51.4% to 74.1%. The proofs establish properties of the translated Lean model; the paper explicitly does not prove Solidity-to-Lean semantic preservation.

- Neuroforger: certified violation witnesses for smart contracts verification via LLMs [[arXiv'26/05](https://arxiv.org/abs/2605.31389)]
  - Background: LLM verification judgments lack checkable evidence; natural-language properties can also be ambiguous.
  - Key problem & insight: Express a violation as a partially specified executable test, then require a valid instantiation rather than trust the model's explanation.
  - Proposed method — Neuroforger with three components:
    1. **GATE**: Extend Solidity specifications with abstract contracts, transactions, and variables representing unknown parts of a violation witness.
    2. **Concretization**: Use GPT-5 to fill these abstract entities, revising candidates from checking feedback.
    3. **Type checking and concrete execution**: Check that substitutions respect the specification and run the witness with Forge. The prototype requires manual validation of type checking.
  - Results: Finds witnesses for 48 of 53 violating tasks and reports none for 57 nonviolating tasks; headline metrics exclude one impractically long witness case. Failure to find a witness returns an inconclusive `true?`, not a proof of safety.

- SciviK: A Versatile Framework for Specifying and Verifying Smart Contracts [[arXiv'21](https://arxiv.org/abs/2103.02209)]
  - Background: Smart-contract analyzers often cover fixed vulnerability patterns, while full functional proofs require substantial manual specification and proof work.
  - Key problem & insight: Combine automated invariant inference with a verification framework that can also express contract-specific properties.
  - Proposed method — SciviK with three components:
    1. **Annotations**: Express vulnerability checks, neural loop-invariant inference requests, and richer contract properties.
    2. **EVM model**: Preserve low-level execution semantics in the intermediate representation used for verification.
    3. **SMT and Coq backends**: Discharge routine obligations automatically and retain interactive proofs for harder properties.
  - Results: Across 12 benchmark contracts and one DeFi contract, 151 of 158 specified properties verify automatically within two seconds; five need moderate modifications and two require manual Coq proofs.

- PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation [[NDSS'25; cited by 55](https://www.ndss-symposium.org/ndss-paper/propertygpt-llm-driven-formal-verification-of-smart-contracts-through-retrieval-augmented-property-generation/)]
  - Background: Smart-contract provers need contract-specific properties that expert auditors normally write manually.
  - Key problem & insight: Retrieve human-written properties from related contracts, adapt them with an LLM, and filter them through compiler and verification feedback.
  - Proposed method — PropertyGPT with three components:
    1. **Property retrieval**: Embed existing properties and retrieve relevant exemplars for the target code.
    2. **Generation and repair**: Adapt properties, use compilation/static-analysis errors to repair them, and rank candidate relevance.
    3. **Formal checking**: Validate candidate properties and investigate violations in the target contract.
  - Results: Recovers 80% of reference properties at 64% precision in the reported human assessment; detects 9/13 CVEs and 17/24 historical attack incidents.

- VeriExploit: Automatic Bug Reproduction in Smart Contracts via LLMs and Formal Methods [[ASE'25](https://pure.manchester.ac.uk/ws/portalfiles/portal/1632624289/ASE2025.pdf)]
  - Background: A smart-contract verifier may report a counterexample without producing an attacker contract or an executable transaction sequence.
  - Key problem & insight: Treat the formal counterexample as a construction guide for an executable bug reproduction.
  - Proposed method — VeriExploit with two components:
    1. **Reproduction synthesis**: Give the LLM the vulnerable contract and counterexample to generate an attacker/reproduction contract and interaction steps.
    2. **Validation and refinement**: Check whether the generated artifact re-triggers the target bug, then repair failed attempts using formal and execution feedback.
  - Results: Achieves 85.60% reproduction success on the authors' benchmark. The task starts from a supplied vulnerable contract and counterexample rather than discovering every bug from scratch.

- SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models [[OOPSLA'26; cited by 0](https://2026.splashcon.org/details/oopsla-2026/18/SymGPT-Auditing-Smart-Contracts-via-Combining-Symbolic-Execution-with-Large-Language)] [[arXiv'25](https://arxiv.org/abs/2502.07644)]
  - Background: ERC requirements are written in natural language, while manual audits and fixed analyzers cover only part of their behavioral constraints.
  - Key problem & insight: Translate rules into a structured language from which executable violation constraints can be synthesized.
  - Proposed method — SymGPT with three components:
    1. **Rule translation**: Use an LLM to map ERC statements into a domain-specific language.
    2. **Constraint synthesis**: Turn translated rules into symbolic conditions describing non-compliance.
    3. **Symbolic execution**: Search contract behavior for paths satisfying those violation conditions.
  - Results: Studies 132 rules from three ERC standards; reports 5,783 violations in 4,000 contracts, including 1,375 with identified financial-theft attack paths, and compares with six automated tools plus an auditing service.

#### Security protocols and agent policies

- MPInspector: A Systematic and Automatic Approach for Evaluating the Security of IoT Messaging Protocols [[USENIX Security'21](https://www.usenix.org/conference/usenixsecurity21/presentation/wang-qinying)]
  - Background: IoT messaging implementations differ from their nominal standards, making manually modeled protocol security checks hard to scale.
  - Key problem & insight: Learn the implementation's interaction model and use that model as input to formal security analysis.
  - Proposed method — MPInspector with three stages:
    1. **Model learning**: Extract parameter semantics and interaction logic to infer the implementation's state machine.
    2. **Property generation**: Instantiate security properties from meta-properties and the learned model.
    3. **Formal analysis**: Check properties and interpret violations as attack opportunities.
  - Results: On MQTT, CoAP, and AMQP implementations across nine IoT platforms, identifies 252 property violations and eleven attack types. This is active model learning, preceding LLM-based protocol agents.

- Automated Attack Synthesis by Extracting Finite State Machines from Protocol Specification Documents [[IEEE S&P'22](https://arxiv.org/abs/2202.09470)]
  - Background: Attacker synthesis requires finite-state protocol models, but RFCs usually describe behavior in English prose.
  - Key problem & insight: Learn a protocol-independent intermediate representation before mapping text into protocol-specific state machines.
  - Proposed method — RFCNLP with three components:
    1. **Technical-language representations**: Learn embeddings adapted to protocol documents.
    2. **Zero-shot information extraction**: Map protocol text into an intermediate information language.
    3. **FSM construction**: Apply rules to build a formal state machine and supply it to attacker synthesis.
  - Results: Evaluates extraction on six protocols—BGPv4, DCCP, LTP, PPTP, SCTP, and TCP—and demonstrates downstream attack synthesis for TCP and DCCP.

- Hermes: Unlocking Security Analysis of Cellular Network Protocols by Synthesizing Finite State Machines from Natural Language Specifications [[USENIX Security'24](https://www.usenix.org/system/files/usenixsecurity24-al-ishtiaq.pdf)]
  - Background: Cellular protocol model checking traditionally depends on expensive, error-prone manual translation of large 3GPP specifications.
  - Key problem & insight: Recover transition structure with domain-aware parsing and translate it into logical state-machine semantics.
  - Proposed method — Hermes with three components:
    1. **NEUTREX**: Use a neural constituency parser to extract states, conditions, and actions from transition-relevant text.
    2. **Domain-specific language**: Convert extracted components into logical formulas using dependency-parse information.
    3. **FSM compilation**: Assemble transitions and run formal security analysis on the resulting models.
  - Results: Reports 81–87% extraction accuracy on 4G NAS, 5G NAS, and 5G RRC; finds three new vulnerabilities, reproduces 19 prior attacks, and identifies seven deviations in commercial 4G basebands.

- LLM-Aided Automatic Modeling for Security Protocol Verification [[ICSE'25](https://ieeexplore.ieee.org/document/11029741/)]
  - Background: Direct LLM generation of Tamarin/ProVerif-style models is unreliable because natural-language descriptions omit or ambiguously express protocol semantics.
  - Key problem & insight: Limit the LLM to semantic parsing and use sound transformations between explicit intermediate models.
  - Proposed method — staged protocol modeling with three components:
    1. **Semantic parsing**: Extract protocol actions and data dependencies from the natural-language description.
    2. **Disambiguation**: Use lightweight human interaction where the text does not determine a unique meaning.
    3. **Model transformation**: Apply transformations justified against a formal execution model to produce the final symbolic protocol model.
  - Results: Produces correct moderate-scale models for 10/18 real-world protocols. The transformation guarantees do not remove the need to validate the initial interpretation and manual disambiguation.

- VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation [[arXiv'25](https://arxiv.org/abs/2510.05156)]
  - Background: Prompt-level safety instructions do not enforce constraints on every action an autonomous agent may propose.
  - Key problem & insight: Verify a behavioral policy offline and enforce it through a runtime monitor rather than trust the agent to remember the policy.
  - Proposed method — VeriGuard with two stages:
    1. **Offline policy construction**: Clarify intent, synthesize a policy, and iteratively test and formally check it against explicit safety requirements.
    2. **Online monitoring**: Validate proposed agent actions against the preverified policy before execution.
  - Results: Reports attack success falling from 53.5% without defenses to 0% for the full system on the evaluated attack suite. This empirical zero is limited to that suite; formal guarantees depend on the modeled policy and enforcement boundary.

- Towards Verifiably Safe Tool Use for LLM Agents [[ICSE-NIER'26](https://doi.org/10.1145/3786582.3786839)] [[arXiv'26](https://arxiv.org/abs/2601.08012)]
  - Background: Information-flow controls and temporal restrictions for tool-using agents often require substantial manual policy annotation.
  - Key problem & insight: Derive tool-use constraints from explicit hazards and attach structured security labels to the tool interface.
  - Proposed method — capability-enhanced MCP with two components:
    1. **System-Theoretic Process Analysis (STPA)**: Identify unsafe workflows and derive requirements on data flows and tool sequences.
    2. **Labeled tool capabilities**: Express confidentiality, integrity/trust, and permitted actions; use Alloy to analyze the resulting workflow constraints.
  - Results: A preliminary Alloy model exhibits unsafe flows without policies and excludes them within the analyzed bounds once policies are enforced; the paper is a NIER feasibility/vision contribution, not a large-scale agent benchmark.

- A Formal Lens on Android Permissions System: Modeling, Verification, and Exploitation Using LLMs and Model Checking [[TOPS'26](https://doi.org/10.1145/3799897)]
  - Background: Earlier Android permission models omit newer behaviors such as one-time permissions and their interactions with app lifecycle states.
  - Key problem & insight: Use documentation and source jointly to construct an updated formal permission model, then investigate checker-produced violations in real implementations.
  - Proposed method — Android permission analysis with two components:
    1. **LLM-assisted modeling**: Use chain-of-thought assistance to extract behavior and security properties into a formal state-transition model.
    2. **TLC verification and validation**: Check the properties, analyze counterexamples, and validate the resulting permission flaw with a concrete implementation test.
  - Results: Reproduces a known Android 6 issue and identifies Permanent Permission Access in Android 11; the reported implementation test also works on Android 15. This is a case study, not a complete proof of Android's permission system.

- An AI Approach to Verified Production Cryptographic Libraries [[arXiv'26](https://arxiv.org/abs/2608.00965)]
  - Background: Most proof-generation tasks supply internal contracts and lemmas; production cryptographic libraries also require discovering those intermediate interfaces.
  - Key problem & insight: Plan internal specifications and proofs while mechanically preventing changes that weaken the trusted problem statement.
  - Proposed method — CryptoProver with three components:
    1. **Specification and proof synthesis**: Start from high-level API contracts and generate internal Verus specifications and proofs without editing executable code.
    2. **Mechanical gates**: Reject specification weakening, new axioms, and cross-module breakage.
    3. **Isolation**: Block retrieval of reference proofs, including through repository history, while exposing a fixed trusted specification/fact library.
  - Results: Constructs an independent curve25519-dalek proof in 11.4 hours at $466.99 recorded API cost, and verifies RustCrypto chacha20 against an RFC 8439 specification; trusted API contracts and arithmetic facts remain supplied inputs.

- Towards Tackling Application Logic Flaws through Autonomous Formal-Logic Modeling and Automated Reasoning [[arXiv'26; citations N/A (method-paper set)](https://arxiv.org/abs/2609.10537)]
  - Background: Application-logic vulnerabilities depend on vendor-specific roles, permissions, and protocol semantics that generic vulnerability patterns miss.
  - Key problem & insight: Translate those semantics into a common logical modeling language, then let a model checker explore the resulting state space.
  - Proposed method — LL-Verifier with two components:
    1. **Autonomous logical modeling**: Extract principals, attributes, rules, and security goals from natural-language protocol descriptions into a Maude-based language.
    2. **Logic-level model checking**: Compile the models into logical state machines and search for security-property violations.
  - Results: The application study covers 27 IoT access-control protocols; the modeling evaluation reports 96.8% rule coverage but only 64.3% property coverage, with 65.5% of analyzed flaws checkable without intervention. Property extraction remains a key bottleneck.

## Math

**Research trajectory.** Early benchmarks and environments make theorem proving measurable and expose proof states and library dependencies (miniF2F, LeanDojo). DSP uses an informal proof as a plan; LEGO-Prover turns intermediate lemmas into reusable library entries. DeepSeek-Prover and AlphaProof then use checked proofs as training feedback, while STP learns a conjecture curriculum instead of relying on a fixed problem set. Numina-Lean-Agent represents a complementary path: use a general coding agent to manage Lean, retrieval, files, and proof repair without building a bespoke trained prover for every task.

**Why these papers.** The selection keeps the benchmark foundation, a harder undergraduate benchmark, statement-alignment evaluation, verifier-reward training, recursive decomposition, self-play, test-time RL, reusable lemmas, and a general agent interface. AlphaGeometry2 is retained as a contrasting specialized approach: strong domain-specific deduction plus learned construction proposals. It should not be treated as the same verification stack as a Lean prover.

The main shift is from **predicting the next tactic** to **managing a proof-development process**. A proof assistant provides a strong acceptance signal, but the agent must still choose productive subgoals, find or invent useful lemmas, and preserve the meaning of the original statement.

<a id="math-benchmark"></a>

### Benchmark

- MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics [[ICLR'22; cited by 25](https://arxiv.org/abs/2109.00110)]
  - Background: Proof-assistant-specific corpora make it difficult to compare provers on the same mathematical problems.
  - Key problem & insight: Formalize a shared set of competition problems across proof systems and require machine-checked solutions.
  - Proposed method — miniF2F with two components:
    1. **Shared mathematical statements**: Draw problems from AMC, AIME, IMO, and high-school/undergraduate material.
    2. **Cross-system formalizations**: Provide statements in Metamath, Lean, and partially Isabelle and HOL Light, with common validation/test organization.
  - Results: 488 statements; establishes a widely reused comparison point for formal competition mathematics. Compare reported success only after checking the language version, split, and sampling budget.

- LeanDojo: Theorem Proving with Retrieval-Augmented Language Models [[NeurIPS'23 — Datasets and Benchmarks; cited by 40](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4441469427094f8873d0fecb0c4e1cee-Abstract-Datasets_and_Benchmarks.html)]
  - Background: Neural provers need reproducible interaction with Lean and access to relevant lemmas from a large mathematical library.
  - Key problem & insight: Treat premise selection as a learned retrieval problem with supervision extracted from actual proof dependencies.
  - Proposed method — LeanDojo and ReProver with three components:
    1. **LeanDojo**: Extract proof states and premise-use annotations and expose a programmatic proof environment.
    2. **Premise retriever**: Retrieve accessible lemmas using hard negatives and dependency-aware training data.
    3. **ReProver**: Generate tactics conditioned on retrieved premises and search through Lean-checked proof states.
  - Results: Releases 98,734 theorems and proofs, including a split testing unseen premises; ReProver requires approximately one GPU-week of training and improves over the paper's non-retrieval and GPT-4 baselines.

- PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition [[NeurIPS'24 — Datasets and Benchmarks; cited by 4](https://proceedings.neurips.cc/paper_files/paper/2024/file/1582eaf9e0cf349e1e5a6ee453100aa1-Paper-Datasets_and_Benchmarks_Track.pdf)]
  - Background: Improving miniF2F scores may reflect progress on relatively short competition proofs rather than broader undergraduate mathematics.
  - Key problem & insight: Use difficult Putnam problems with manually constructed formalizations across several proof assistants.
  - Proposed method — PutnamBench with two components:
    1. **Human formalization**: Encode statements from the William Lowell Putnam Mathematical Competition in Lean 4 and Isabelle, with a substantial Coq subset.
    2. **Cross-prover evaluation**: Require checked proofs and compare neural and symbolic systems on the same source problems.
  - Results: The original release contains 1,692 formalizations of 640 problems; its evaluated systems solve only a handful. Later papers use expanded versions, so denominators such as 640, 644, and 658 must not be silently mixed.

- FormalAlign: Automated Alignment Evaluation for Autoformalization [[ICLR'25; cited by 1](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fceedf8c9c0ff51f41b9fe0294ef0070-Abstract-Conference.html)]
  - Background: Lean accepts a proof of the statement it receives, even if autoformalization changed or weakened the original mathematical claim.
  - Key problem & insight: Assess informal–formal semantic alignment separately from proof validity.
  - Proposed method — FormalAlign with two training objectives:
    1. **Autoformalization**: Learn to generate the formal statement from its informal counterpart.
    2. **Representational alignment**: Jointly train representations to distinguish aligned pairs from mismatched statements and use the learned score for selection.
  - Results: Alignment-selection accuracy reaches 99.21% versus GPT-4's 88.91% on FormL4-Basic, and 66.39% versus 64.34% on miniF2F-Valid. This is a learned evaluator, not a sound proof of translation equivalence.

- Learning to Prove Theorems via Interacting with Proof Assistants [[ICML'19; cited by 24](https://arxiv.org/abs/1905.09381)]
  - Background: Earlier learned interactive provers train on a few thousand theorems and choose tactics from a fixed vocabulary, preventing project-disjoint evaluation and tactics containing new local terms.
  - Key problem & insight: Record large-scale Coq interactions and generate tactics as structured programs whose arguments are selected from the current goal and premises.
  - Proposed method — CoqGym and ASTactic with three components:
    1. **CoqGym**: Extract 70,856 human proofs from 123 buildable Coq projects, split by project, and expose a stepwise proof-assistant environment.
    2. **Synthetic shorter proofs**: Turn intermediate proof goals into additional theorem/proof pairs to increase training coverage.
    3. **ASTactic**: Encode goals and premises as trees, generate tactic abstract-syntax trees under a grammar, and run depth-first proof search against Coq.
  - Results: ASTactic proves 12.2% of 13,137 test theorems versus under 4.9% for Coq's built-in automation; combining ASTactic with a hammer reaches 30.0%, 5.2 points above the hammer alone. The fork's CoqGym row incorrectly linked the unrelated GPT-f paper (`arXiv:2009.03393`).

- ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics [[arXiv'23; cited by 15](https://arxiv.org/abs/2302.12433)]
  - Background: miniF2F emphasizes school competitions, while Mathlib declarations rarely pair an undergraduate textbook statement with its informal proof and a held-out Lean theorem.
  - Key problem & insight: Curate parallel informal and formal mathematics, then improve statement translation by retrieving domain-relevant examples or synthesizing training pairs from monolingual formal corpora.
  - Proposed method — ProofNet with three components:
    1. **Undergraduate corpus**: Provide 371 Lean 3 statements paired with natural-language statements and proofs across analysis, algebra, linear algebra, and topology.
    2. **Prompt retrieval**: Produce an initial formalization, retrieve four nearby Mathlib declarations in embedding space, and rebuild the few-shot prompt.
    3. **Distilled backtranslation**: Use a larger teacher to informalize Mathlib theorems and fine-tune a smaller proofGPT model on the synthetic pairs.
  - Results: Codex few-shot formalization reaches 13.4% accuracy and 23.7% typecheck rate; retrieval raises these to 16.1% and 45.2%. Codex informalizes 62.3% correctly, while the paper leaves formal-proof-generation baselines for future work.

- FIMO: A Challenge Formal Dataset for Automated Theorem Proving [[arXiv'23; cited by 10](https://arxiv.org/abs/2309.04295)]
  - Background: Existing formal benchmarks contain few genuine IMO-level problems and underrepresent the long mathematical arguments needed for shortlisted algebra and number-theory tasks.
  - Key problem & insight: Use proof-assistant and human feedback during autoformalization, then keep the official informal solutions so models can be tested on proof generation with or without a human-written plan.
  - Proposed method — FIMO with three stages:
    1. **OCR and problem normalization**: Extract IMO Shortlist problems/solutions and rewrite answer-style questions as propositions.
    2. **Reflective autoformalization**: Translate to Lean, repair syntax from Lean errors, and repair semantics from human feedback.
    3. **Informal-guided proving**: Prompt GPT-4 with the formal theorem and either the official proof or a newly generated informal proof, accepting only Lean-checked outputs.
  - Results: Formalizes 149/245 selected problems (60.8%) versus 80/245 without reflective feedback. GPT-4 solves 0/149 FIMO theorems at Pass@8, despite reaching 20.9% on miniF2F-test without the official human proof.

- Lean Workbook: A large-scale Lean problem set formalized from natural language math problems [[NeurIPS'24 — Datasets and Benchmarks; cited by 3](https://arxiv.org/abs/2406.03847)]
  - Background: Human formalization cannot supply enough contest-style Lean statements for expert iteration, and automatically translated corpora contain silent semantic errors even when they compile.
  - Key problem & insight: Alternate formalization and backtranslation, filter with compilation and NLI, and feed human-corrected failure patterns into the next active-learning round.
  - Proposed method — Lean Workbook with three components:
    1. **Bidirectional translator**: Fine-tune on formal/informal pairs and translate selected questions from a 1.09M-question math-forum crawl.
    2. **Three-stage filtering**: Require Lean compilation, natural-language backtranslation, and an NLI equivalence judgment.
    3. **Human diagnostic loop**: Correct representative type, notation, quantifier, extremum, and semantic mistakes over six rounds and add them to training.
  - Results: Produces 57,231 accepted pairs with 93.5% sampled human accuracy and 21 newly formalized IMO questions. InternLM-Math-Plus solves 4,898 tasks at Pass@1024 (8.6%); adding the corpus raises miniF2F-test accuracy from 37.3% to 46.7%.

- TRIGO: Benchmarking Formal Mathematical Proof Reduction for Generative Language Models [[EMNLP'23; cited by 2](https://arxiv.org/abs/2310.10180)]
  - Background: Standard ATP benchmarks emphasize symbolic premise application and contain little controlled reasoning over trigonometric identities, equivalent angles, and numerical term manipulation.
  - Key problem & insight: Combine manually annotated real reductions with a rule-based generator whose proof length and numeric distribution can be varied independently.
  - Proposed method — TRIGO with three components:
    1. **Real and web problems**: Collect 427 textbook/exam reductions plus 453 held-out web problems and annotate intermediate transformations.
    2. **Lean formalization**: Convert each reduction step into Lean tactics and replay it through Lean-Gym.
    3. **Controlled generator**: Sample from 85 transformation rules to create three synthetic difficulty/distribution sets for in- and out-of-distribution evaluation.
  - Results: GPT-2-Large with PACT pretraining reaches 32.55% on TRIGO-real and 25.60% on TRIGO-web, but only 18.99% on the hardest synthetic tier. Synthetic fine-tuning raises generated-set scores to 84.29%/60.09%/25.29% while reducing real-data performance, exposing the distribution gap.

- MLFMF: Data Sets for Machine Learning for Mathematical Formalization [[NeurIPS'23 — Datasets and Benchmarks; cited by 2](https://arxiv.org/abs/2310.16005)]
  - Background: Proof-learning corpora expose language-specific text or tactic traces, making cross-assistant experiments and graph-based declaration recommendation difficult.
  - Key problem & insight: Normalize library declarations into computational syntax graphs and a reference multigraph, turning missing-lemma recommendation into standard link prediction.
  - Proposed method — MLFMF with three components:
    1. **Language-specific extraction**: Parse Lean Mathlib4, Agda stdlib, Agda-unimath, and TypeTopology declarations with their types, bodies, and references.
    2. **Common graph representation**: Convert approximately 250,000 entries into DAGs linked by typed dependency edges.
    3. **Reference-recommendation benchmark**: Hide proof-body references and compare popularity, bag-of-words, TF-IDF, fastText, analogy, and node2vec recommenders.
  - Results: Node2vec classification accuracy is 0.95–0.98 across the four libraries and ranks a true reference within 3.24–5.81 positions on the Agda corpora, but its mean minimum rank is 195 on Mathlib4, leaving large-library retrieval unresolved.

- miniCTX: Neural Theorem Proving with (Long-)Contexts [[ICLR'25 Oral; cited by 0](https://arxiv.org/abs/2408.03350)]
  - Background: Competition theorems are self-contained and Mathlib splits risk contamination, whereas real proof development requires learning definitions and lemmas that were introduced recently in an unseen project.
  - Key problem & insight: Evaluate theorem, context, and project generalization separately, and train the prover to condition on the preceding Lean file rather than only the current goal state.
  - Proposed method — miniCTX with three components:
    1. **Temporal project benchmark**: Select 762 theorems from six Lean projects with commit metadata, in-file context, imports, proof dependencies, and project-disjoint/recency controls.
    2. **File-tuning**: Fine-tune tactic prediction on `(proof state, preceding file, tactic)` triples so new local notation, definitions, and proof patterns are visible at inference time.
    3. **Cross-file premise selection**: Retrieve project-local declarations from imported modules and append them to the in-file context.
  - Results: File-tuning reaches 35.94% on miniCTX versus 19.53% for state-only tuning; GPT-4o improves from 11.72% to 27.08% when given file context. The same tuning changes miniF2F only from 32.79% to 33.61%, showing why context-free benchmarks miss this capability.

- Herald: A Natural Language Annotated Lean 4 Dataset [[ICLR'25; cited by 0](https://arxiv.org/abs/2410.10878)]
  - Background: Large Lean corpora lack aligned natural-language statements and proofs, while direct LLM informalization ignores declaration dependencies and propagates mathematical errors.
  - Key problem & insight: Translate dependencies before dependents and expose compiler-derived structure at each step, then augment both tactic-level statements and informal variants.
  - Proposed method — Herald with three components:
    1. **Hierarchy-aware informalization**: Use Lean-Jixia to extract declarations, dependencies, proof structure, and term explanations; translate Mathlib4 in dependency order with retrieval context.
    2. **Dual augmentation**: Break tactic proofs into checked intermediate statements and generate additional informal counterparts, expanding the formal corpus to about 580K valid statements and 44K NL-formal theorem pairs.
    3. **Herald Translator**: Fine-tune bidirectional statement translation and filter generated Lean with compilation, backtranslation, and NLI checks.
  - Results: Achieves 96.7% Pass@128 on miniF2F-test and 23.5% on a 200-item graduate-textbook set versus 73.0% and 7.5% for InternLM2-Math-Plus-7B. It translates a Stacks Project section with minor human corrections, but the paired prover closes only one short theorem.

- FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models [[arXiv'25; cited by 0](https://arxiv.org/abs/2505.02735)]
  - Background: miniF2F and ProofNet are relatively small and concentrated in algebra/number theory, while recent provers approach saturation on their easier portions.
  - Key problem & insight: Scale statement curation with multiple independent semantic filters before expert review, and retain broad domain labels so aggregate proof rates cannot hide subject-specific failures.
  - Proposed method — FormalMATH with three filtering layers:
    1. **Ensemble autoformalization**: Translate problems from seven olympiad and advanced-math sources with multiple specialized LLMs and best-of-$N$ sampling.
    2. **Automated validation**: Require Lean compilation, multi-LLM semantic agreement, and prover-based attempts to refute the statement's negation.
    3. **Expert verification**: Have IMO-level annotators review the surviving statements and assign domain/difficulty metadata.
  - Results: Produces 5,560 Lean 4 statements, 22.8x miniF2F's test size. Kimina-Prover leads at 16.46% under Pass@32; performance is strongest in algebra and applied mathematics and weakest in calculus, and adding human natural-language solutions can reduce proof success.

- CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics [[ICML'26 AI4Math Workshop; cited by 0](https://arxiv.org/abs/2505.03171)]
  - Background: Lean benchmarks contain little combinatorics, whose counting constructions and fill-in-the-blank answers often require definitions absent from Mathlib and cannot be graded as a fixed theorem proof alone.
  - Key problem & insight: Pair proof tasks with an evaluator that lets a model propose an answer and then formally prove either exact agreement or mathematical equivalence to the reference.
  - Proposed method — CombiBench and FINE-EVAL with three components:
    1. **Combinatorics corpus**: Manually formalize 100 problems from HackMath, a combinatorics textbook, IMO, and other olympiads across more than ten topics.
    2. **Whole-proof evaluation**: Replace proof holes and require Lean-accepted completions for ordinary theorem tasks.
    3. **FINE-EVAL**: For fill-in-the-blank tasks, first check the proposed value and proof; when it differs from the reference, require a second Lean proof of equivalence.
  - Results: At Pass@16, the strongest system, Kimina-Prover Preview, solves only 7/100 tasks with or without the informal solution; all evaluated 7B specialized provers solve 0, and the best general reasoning model solves at most 4.

- FATE: A Formal Benchmark Series for Frontier Algebra of Multiple Difficulty Levels [[ICLR'26; cited by 0](https://openreview.net/forum?id=3bD19r4jqh)] [[arXiv'25](https://arxiv.org/abs/2511.02872)]
  - Background: Competition and introductory-textbook benchmarks do not represent the abstraction, nested definitions, and library gaps encountered in graduate and research-level algebra.
  - Key problem & insight: Build a progressive same-domain series and inspect natural-language reasoning separately from Lean formalization, revealing whether failure is mathematical or representational.
  - Proposed method — FATE with three tiers and a two-stage evaluator:
    1. **FATE-M/H/X**: Curate 150 undergraduate, 100 graduate, and 100 post-qualifying-exam abstract/commutative algebra problems.
    2. **Component packaging**: Provide any definitions, instances, and lemmas needed beyond Mathlib while preserving the target theorem as the proof obligation.
    3. **Natural/formal analysis**: Experts score the intermediate natural-language proof, then Lean checks up to 64 formal proof attempts and errors are classified by Mathlib hallucination, Lean proficiency, reasoning, and alignment.
  - Results: The best model reaches only 3% Pass@64 on FATE-H and 0% on FATE-X, despite intermediate natural-language accuracy as high as 73% and 21% respectively. This isolates formalization, not only mathematical planning, as the dominant bottleneck.

- FaithformBench: Benchmarking Faithfulness of Mathematical Chain-of-Thought Autoformalisation [[arXiv'26; cited by 0](https://arxiv.org/abs/2608.10916)]
  - Background: Autoformalization evaluations usually assume correct inputs and reward provable outputs, so a system can silently repair an invalid reasoning step and appear accurate in a chain-of-thought verifier.
  - Key problem & insight: Test both validity preservation and invalidity preservation by introducing controlled errors and asking whether the formal output remains provable or refutable.
  - Proposed method — FaithformBench with three components:
    1. **Dependency-aware steps**: Convert 1,179 error-free ProcessBench chains into DAGs and extract 12,784 reasoning steps across GSM8K, MATH, OlympiadBench, and Omni-MATH.
    2. **Invalid perturbations**: Create a minimally changed invalid counterpart for every step and validate perturbation quality with LLM and human review.
    3. **Proof-based faithfulness metrics**: Autoformalize original and perturbed steps, try to prove each statement and its negation, and report error induction, silent correction, AF failure, and a guaranteed unfaithfulness lower bound.
  - Results: All four specialized autoformalizers show substantial silent correction; Goedel has the lowest overall unfaithfulness bound (about 0.14–0.21 across datasets) but the highest silent-correction tendency. General-purpose frontier models are less sycophantic overall, showing that higher valid-input accuracy does not imply higher faithfulness.

- SHADOWBENCH: Toward Reliable Automatic Evaluation of Semantic Alignment in Autoformalization [[EMNLP'26 accepted; cited by 0](https://arxiv.org/abs/2608.29270)]
  - Background: Compilation, lexical overlap, LLM judges, and even equivalence tactics can accept a type-correct Lean theorem that omits or changes part of the intended natural-language claim.
  - Key problem & insight: Decompose the intended theorem into independently checkable semantic shadows and require both directions of implication, so alternative but equivalent formalizations can pass without rewarding weakened statements.
  - Proposed method — SHADOWBENCH and SA-PASS with three components:
    1. **Expert benchmark**: Curate 178 postgraduate-to-research problems in eight areas with reference Lean statements and revised shadow declarations.
    2. **Forward and backward checkers**: Prove each shadow from the candidate and prove the reference from the conjunction of shadows; hide checkers during generation.
    3. **SA-PASS / SA-PASSsoft**: Require all checks for the strict score or average forward coverage with the backward indicator for diagnostic scoring.
  - Results: The best agent compiles on 61.8% but reaches only 18.3% SA-PASSsoft and 11.2% SA-PASS. Against expert labels, compile precision is 0.178, whereas SA-PASS achieves 1.000 precision, 0.930 recall, and 0.988 agreement.

- NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry [[arXiv'26; cited by 0](https://arxiv.org/abs/2608.28481)]
  - Background: AlphaGeometry can solve olympiad geometry only after a human encodes every point, construction, constraint, and goal in its specialized DSL.
  - Key problem & insight: Grade translation by executing the generated DSL rather than by text similarity, and separate syntax failures from missing or incorrect geometric semantics.
  - Proposed method — NL2AGBench with three components:
    1. **Executable geometry set**: Select 48 diverse, runtime-feasible tasks from 231 JGEX problems previously formalized for AlphaGeometry.
    2. **Direct translation evaluation**: Generate a two-line DSL problem, parse it, and run AlphaGeometry to validate the construction and target.
    3. **Mitigation studies**: Compare zero/few-shot prompting, supervised fine-tuning, diagram hints, and targeted error-specific corrections.
  - Results: GPT-5.4 reaches 75.0% zero-shot and 79.2% few-shot; Gemini 3.1 rises from 62.5% to 83.3%. Qwen3-235B improves from 12.5% to 45.8%, while smaller open models remain at 0%; targeted human hints add 14.29–32.14 points for frontier models.

- MathAdv: What Theorem Provers Know, Reason, Formalize, and Generalize [[arXiv'26; cited by 0](https://arxiv.org/abs/2608.25449)]
  - Background: A single Lean proof rate cannot reveal whether failure comes from missing mathematical knowledge, informal reasoning, formalization skill, or sensitivity to a familiar problem presentation.
  - Key problem & insight: Attach capability-isolating tasks and mathematically equivalent rewrites to the same advanced problems, enabling component-wise diagnosis.
  - Proposed method — MathAdv with four task views:
    1. **Formal proof**: Provide 298 expert-reviewed Lean targets from 321 problems across 13 undergraduate and graduate domains.
    2. **Knowledge and informal reasoning**: Pair problems with multiple-choice concept/strategy questions and direct fill-in-the-blank answers.
    3. **Robustness transformations**: Have experts rewrite problems while preserving their mathematical content and reasoning requirements.
    4. **Verifier-in-the-loop curation**: Combine LLM drafting, Lean syntax repair, independent semantic checks, and final domain-expert review.
  - Results: Goedel-Prover-V2 leads Lean proving at 21.88%, while GPT-5.4 reaches 82.88% on multiple choice but 10.62% on proofs; all models score 0% on topology. Lean feedback raises GPT-5.4 from 9.06% to 13.75%, yet equivalent reformulations still cause asymmetric failures.

- AxQM: A Textbook-Scale Benchmark for Formal Proof Synthesis in a Library of Finite-Dimensional Quantum Mechanics [[arXiv'26; cited by 0](https://arxiv.org/abs/2609.05157)]
  - Background: Physics formalization benchmarks sample isolated claims, while a textbook develops interdependent definitions and idealizations whose hidden assumptions can affect many later results.
  - Key problem & insight: Formalize the rigorous finite-dimensional content of one foundational textbook into a shared Lean library, and release proof holes while keeping complete kernel-checked solutions private.
  - Proposed method — AxQM with three components:
    1. **Textbook census**: Extract 687 Nielsen-and-Chuang items, formalize 505 mathematically rigorous ones, and expose 1,019 proof tasks over 479 benchmark items.
    2. **Quantum-mechanics library**: Supply 3,519 public declarations covering states, channels, circuits, algorithms, error correction, and entropy on a pinned Mathlib fork.
    3. **Private solutions and deterministic grading**: Maintain 10,560 declarations in the complete library, label proof-length bands from dependency counts, and reject `sorry`, new axioms, or admitted dependencies.
  - Results: The release contains 1,019 guaranteed-solvable tasks, including 257 large or very-large proofs and reference developments exceeding 450 additional declarations. The paper does not report a model baseline; its evaluated result is the benchmark/library construction and kernel-checked solution coverage.

- StochBench: A Domain-Specific Benchmark for Stochastic Processes in Lean [[arXiv'26; cited by 0](https://arxiv.org/abs/2609.09264)]
  - Background: Competition and broad textbook benchmarks underrepresent the recurring abstractions of graduate stochastic processes and can hide whether a prover works consistently inside one applied domain.
  - Key problem & insight: Mix direct Mathlib statements with explicitly abstracted hypotheses when infrastructure is missing, while recording the representation boundary so proof success is not confused with full formalization of the source theory.
  - Proposed method — StochBench with three components:
    1. **Domain-focused curation**: Select or author 450 problems across Markov chains, martingales, renewal, random walks, Brownian motion, Poisson/continuous-time processes, and weak convergence.
    2. **Shared representations**: Define reusable chain laws, transition powers, stopping-time adapters, drift conditions, and convergence interfaces.
    3. **Scope-aware tasks**: Label 114 direct targets using Mathlib/shared objects and 336 abstracted targets whose required source properties are supplied as hypotheses.
  - Results: A tool-using Opus 4.8 agent produces 157/450 clean Lean proofs (34.9%) under a 15-minute limit. The authors identify proof-search/library failures and some missing-assumption defects, so source faithfulness remains subject to further review.

- OEIS Open: How many conjectures can language models turn into theorems? [[arXiv'26; cited by 0](https://arxiv.org/abs/2608.11941)]
  - Background: Open-problem benchmarks usually restrict tasks to objects with bespoke computational verifiers, excluding general proof questions and making false or unsatisfiable tasks one-sided.
  - Key problem & insight: State each conjecture in Lean and allow either a proof or a disproof, so the kernel can certify both true and false OEIS claims with a uniform checker.
  - Proposed method — OEIS Open with three components:
    1. **Open conjecture corpus**: Package 492 OEIS conjectures over integer sequences and a random 100-item Lite subset, retaining proposer/date and literature-attention metadata.
    2. **Adversarially isolated verification**: Separate agent, clean compiler, and SafeVerify scorer containers; require the original theorem type and a standard-axiom allowlist.
    3. **Budgeted generic agent**: Give each model Lean, shell/editor tools, and optional offline arXiv access or a deeper agent loop, with fixed dollar and wall-time limits.
  - Results: The headline SafeVerify result is 147/492 (30%) for Claude Opus 4.8 at $50 per conjecture, while independent Comparator replay yields 144/492 (29%) after accounting for five defective targets and two resource-limit recoveries. The best Lite result is 44% at $200; literature retrieval and a more elaborate agent loop do not improve it.

- FormalTCS: Benchmarking End-to-End Frontier Formal Theoretical Computer Science Research of Large Language Models [[arXiv'26; cited by 0](https://arxiv.org/abs/2608.20153)]
  - Background: TCS benchmarks use textbook or isolated theorems and do not preserve paper-specific definitions, assumptions, dependencies, or the research stages between a core claim and a formal proof.
  - Key problem & insight: Annotate every intermediate natural/formal artifact from recent accepted papers and score each stage independently, preventing early errors from masking later capabilities.
  - Proposed method — FormalTCS with four evaluated stages:
    1. **Theorem elicitation (CC2NT)**: Expand a paper's concise core claim into a self-contained natural-language theorem.
    2. **Autoformalization (NT2FT)**: Translate that theorem and its paper-specific concepts into an equivalent Lean statement, checked bidirectionally with BEq+.
    3. **Proof elicitation (T2NP)**: Produce a concise natural-language proof strategy faithful to the source paper.
    4. **Theorem proving (FT2FP)**: Generate a Lean proof under the preserved definitions/dependencies and an axiom-restricted grader.
  - Results: Across 143 claims from STOC, FOCS, SODA, and COLT 2025–2026, Claude Opus 5 reaches 66.9/11.5/68.7/28.6 on the four stages, making autoformalization the bottleneck. An automated research loop proposes 64 claims, but only 6 survive expert novelty/value review and proof verification.

- ProofJudge: Tool-Grounded LLM Evaluation of Formal Proof Quality in Mathlib [[arXiv'26; cited by 0](https://arxiv.org/abs/2608.20432)]
  - Background: Lean's kernel certifies correctness, not whether a proof reuses the library, matches appropriate automation, exposes reusable structure, or follows Mathlib review conventions.
  - Key problem & insight: Ground an LLM judge in the exact repository commit and test whether it prefers the revision human reviewers accepted over the valid-but-rejected initial version.
  - Proposed method — ProofJudge with three components:
    1. **Five-part rubric**: Score library leverage, automation fit, structural clarity, statement quality, and Mathlib conventions on 1–10 scales.
    2. **Tool-grounded agent**: Give the judge up to 20 shell/library queries so it can inspect declarations, signatures, and local conventions before scoring.
    3. **PR preference dataset**: Pair initial and final versions of 218 declarations from distinct Mathlib pull requests, score them independently, and measure agreement with reviewer acceptance.
  - Results: Six judges achieve 63.5–80.8% reviewer alignment versus 50% chance; Claude Sonnet 5 leads at 80.8%, while two open-weight judges reach 69.3–70.2% at roughly one-tenth the cost. Repeated judgments remain noisy, so single-run comparisons are not reliable.

- A Lean Dataset for International Math Olympiad: Small Steps towards Writing Math Proofs for Hard Problems [[TMLR'25; cited by 0](https://arxiv.org/abs/2411.18872)]
  - Background: miniF2F contained IMO statements without public complete proofs, so a failed model gave little evidence about which intermediate facts or proof-composition skills were missing.
  - Key problem & insight: Write full proofs and decompose them into approachable but nontrivial lemmas, separating local execution from long-horizon proof planning.
  - Proposed method — IMO Steps with three components:
    1. **Complete IMO formalizations**: Add original Lean 4 proofs for 14 missing miniF2F IMO problems plus three problems from IMO 2022–2023.
    2. **Lemma decomposition**: Break 13 full proofs into 1,329 labeled obligations with reference proofs and length/topic metadata.
    3. **Proof and feedback evaluation**: Compare specialized provers and give o3-mini up to ten rounds of Lean REPL diagnostics, followed by human error categorization.
  - Results: DeepSeek-Prover-V1.5-RL solves 39.3% of the lemmas, Goedel-Prover 37.9%, and o3-mini 23.8%. For o3-mini, feedback raises success from 6.0% zero-shot to 23.8% after ten rounds, while its natural-language steps are correct 75.5% of the time.

- RePro: Proof-Verified Benchmark Rewriting for Reliable Evaluation of LLM Mathematical Problem Solving [[EMNLP'26 accepted; cited by 0](https://arxiv.org/abs/2609.00062)]
  - Background: Paraphrase and numerical-rewrite benchmarks reduce memorization cues but can silently create ill-defined questions, infeasible constraints, or wrong reference answers.
  - Key problem & insight: Treat rewriting as verified dataset construction: screen semantic feasibility, formalize the retained problem, and accept its answer only after a Lean-checked proof and query-aligned answer extraction.
  - Proposed method — RePro with four stages:
    1. **LLM rewriting**: Generate structurally changed GSM8K/MATH problems and answers without solving the original task in the rewriter prompt.
    2. **Validity screening**: Reject ambiguous, contradictory, infeasible, or real-world-inconsistent candidates with conservative LLM checks.
    3. **Executable formalization**: Convert the rewritten problem into Lean and repair syntax/semantic mismatches through feedback.
    4. **Proof-level verification**: Use neural Lean provers for candidate proofs, kernel-check them, and extract only the answer quantity justified by the verified proof.
  - Results: Human audit reports 100% well-definedness, feasibility, and answer correctness on retained RePro instances, while prior rewrite methods still contain invalid cases. Pass@3 retains 88.3% of sampled GSM8K candidates but only 33.7% of MATH Level 5, and several small models lose 10–21 accuracy points after verified rewriting.

- Solving Inequality Proofs with Large Language Models [[NeurIPS'25 Spotlight; citations N/A (not indexed)](https://arxiv.org/abs/2506.07927)]
  - Background: Fully formal inequality datasets are small or synthetic, while final-answer grading rewards models that guess a tight bound despite invalid algebra or missing proof steps.
  - Key problem & insight: Reformulate informal inequality proving into automatically checked bound estimation and relation prediction, then judge the reasoning chain with specialized error detectors.
  - Proposed method — IneqMath with three components:
    1. **Expert corpus**: Provide 200 contamination-resistant test problems, 100 public development problems, and 1,252 training problems with up to four solutions and 83 theorem annotations.
    2. **Verifiable subtasks**: Ask for an extremal constant or the relation between two expressions, enabling deterministic final-answer checks without requiring proof-assistant syntax.
    3. **Modular proof judges**: Combine an answer-equivalence judge with detectors for toy-case reliance, logical gaps, unjustified numerical approximations, and calculation errors.
  - Results: Across 29 models, overall proof accuracy drops by as much as 65.5 points relative to final-answer accuracy; o1 reaches 62.5% answer accuracy but only 8.0% overall. Golden theorem hints add up to 11 points and one critic-guided revision raises Gemini 2.5 Pro from 43% to 48%.

<a id="math-training"></a>

### Training

- DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search [[ICLR'25](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b3b55c366d641c07180c40e4f978f311-Abstract-Conference.html)]
  - Background: Supervised proof completion underuses Lean's feedback, and sparse complete-proof rewards make search inefficient.
  - Key problem & insight: Use the prover both as a training reward source and as an observable state space for exploration.
  - Proposed method — DeepSeek-Prover-V1.5 with two components:
    1. **Reinforcement Learning from Proof Assistant Feedback**: Update the proof-completion policy using Lean verification outcomes.
    2. **RMaxTS**: Organize Monte Carlo search around intermediate tactic states and give intrinsic exploration rewards for discovering new states.
  - Results: The 7B RL model with RMaxTS reaches 63.5% on miniF2F-test at the paper's largest mixed-prompt budget of 32 x 6,400 samples; the SFT counterpart reaches 60.2% at that budget.

- STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving [[ICML'25](https://proceedings.mlr.press/v267/dong25h.html)]
  - Background: Expert iteration on fixed statements plateaus when the prover cannot solve enough remaining problems to obtain new training data.
  - Key problem & insight: Learn to propose problems near the current prover's frontier of difficulty.
  - Proposed method — Self-play Theorem Prover (STP) with two roles:
    1. **Conjecturer**: Train on generated conjectures that are barely provable by the current model, gradually shifting the curriculum.
    2. **Prover**: Attempt conjectures and improve through expert iteration on formally verified proofs; feed successes back to the conjecturer.
  - Results: On LeanWorkbook, proves 28.5% of statements versus 13.1% for prior expert iteration; reaches 65.0% miniF2F-test and 23.9% ProofNet-test at pass@3200. Its self-play uses verified data generation and fine-tuning, not merely inference-time debate.

- DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition [[arXiv'25; cited by 2](https://arxiv.org/abs/2504.21801)]
  - Background: Whole-proof RL receives little useful signal on problems whose complete proofs are initially beyond the model.
  - Key problem & insight: Recursively solve simpler subgoals and assemble them into training examples that connect informal plans to formal proofs.
  - Proposed method — DeepSeek-Prover-V2 with two stages:
    1. **Recursive cold-start synthesis**: Use DeepSeek-V3 to decompose problems, solve subgoals, and combine checked subproofs with step-by-step reasoning.
    2. **Formal-reasoning RL**: Train the resulting prover to generate complete Lean 4 proofs using verification feedback.
  - Results: The 671B model reaches 88.9% on miniF2F-test and solves 47/658 PutnamBench problems in the reported setup; it also introduces the 325-problem ProverBench.

- Olympiad-level formal mathematical reasoning with reinforcement learning [[Nature'25](https://doi.org/10.1038/s41586-025-09833-y)]
  - Background: Human proof corpora are limited, and standard inference-time search cannot adapt model parameters to an exceptionally hard new problem.
  - Key problem & insight: Train through large-scale interaction with Lean and continue learning on related problem variants at inference time.
  - Proposed method — AlphaProof with three components:
    1. **Autoformalized curriculum**: Convert large collections of informal problems into formal training statements.
    2. **AlphaZero-inspired RL**: Learn proof-search behavior from machine-checked success over millions of formal problems.
    3. **Test-time RL**: Generate and learn from many related variants of the target problem to obtain problem-specific adaptations.
  - Results: Solves three of five non-geometry IMO 2024 problems; combined with AlphaGeometry 2, the system achieves silver-medal-equivalent performance using multi-day computation. This was not an ordinary timed, fully automatic natural-language competition entry.

- Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2 [[JMLR'25](https://www.jmlr.org/papers/v26/25-1654.html)] [[arXiv'25](https://arxiv.org/abs/2502.03544)]
  - Background: General-purpose theorem proving is difficult, but Euclidean geometry admits strong domain-specific symbolic deduction and large-scale synthetic data.
  - Key problem & insight: Let a neural model suggest auxiliary constructions while a specialized symbolic engine derives and checks geometric consequences.
  - Proposed method — AlphaGeometry2 with three components:
    1. **Expanded geometry language**: Cover moving objects, non-constructive statements, and linear relations among angles, ratios, and distances.
    2. **Neural construction proposals**: Use a Gemini-based model and synthetic geometry data to propose steps beyond the symbolic engine's direct reach.
    3. **Symbolic deduction and shared search**: Check consequences with the geometry engine and share discoveries across search trees.
  - Results: Solves 84% of all IMO geometry problems from 2000–2024 versus AlphaGeometry's 54%; language coverage increases from 66% to 88%. The checker and language are geometry-specific, not a general Lean pipeline.

<a id="math-agent"></a>

### Agent

- Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs [[ICLR'23](https://arxiv.org/abs/2210.12283)]
  - Background: An automated prover often fails on a large theorem even when a useful informal proof already describes its intermediate steps.
  - Key problem & insight: Translate the informal argument into a formal skeleton and delegate the smaller gaps to symbolic automation.
  - Proposed method — Draft, Sketch, and Prove (DSP) with three stages:
    1. **Draft**: Obtain an informal proof from a human or language model.
    2. **Sketch**: Translate its structure into an Isabelle proof with explicit intermediate claims and unfilled subproofs.
    3. **Prove**: Use automated provers to discharge the gaps and accept only a completed checked proof.
  - Results: Improves performance from 20.9% to 39.3% on the reported competition-mathematics collection, showing that decomposition can make existing symbolic automation more effective.

- LEGO-Prover: Neural Theorem Proving with Growing Libraries [[ICLR'24](https://proceedings.iclr.cc/paper_files/paper/2024/hash/85dca46374dc0f27b4bb5f265b3d17f0-Abstract-Conference.html)]
  - Background: DSP-style systems can still fail when a sketch needs an intermediate lemma absent from the fixed library.
  - Key problem & insight: Convert newly verified intermediate results into reusable skills and continue expanding the library during solving.
  - Proposed method — LEGO-Prover with two components:
    1. **Modular proving and retrieval**: Decompose a theorem and retrieve verified skills relevant to each subgoal.
    2. **Skill evolution**: Generate and verify additional lemmas, including variants of existing skills, and store accepted ones for future problems.
  - Results: Raises miniF2F-test success from 45.5% to 50.0% and validation success from 48.0% to 57.0%; adds more than 20,000 verified skills during the study.

- Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics [[ICML'26](https://icml.cc/virtual/2026/poster/66755)] [[arXiv'26](https://arxiv.org/abs/2601.14027)]
  - Background: Specialized proving pipelines often hard-code orchestration around a particular trained prover and benchmark.
  - Key problem & insight: Treat formal mathematics as a tool-using coding task that a general agent can manage through a structured Lean interface.
  - Proposed method — Numina-Lean-Agent with two components:
    1. **General coding agent**: Use Claude Code to plan proofs, edit files, retrieve library facts, and coordinate auxiliary reasoning.
    2. **Numina-Lean-MCP**: Expose Lean interactions and proof feedback as tools, enabling the agent to revise its plan and proof artifacts autonomously.
  - Results: With Claude Opus 4.5, solves all twelve Putnam 2025 problems in the reported setup; additionally assists mathematicians in formalizing the Brascamp–Lieb theorem. The latter is explicitly a human–AI collaboration.

## Hardware

**Research trajectory.** Two threads develop in parallel. One learns decisions inside symbolic verification: NeuroPDR guides clause generalization, and NeuroBack predicts SAT phases while leaving the solver in control. The other automates verification engineering: FVEval and AssertionBench make assertion quality measurable, AssertLLM extracts properties from complete design specifications, and CodeV-SVA trains specialized assertion models. FVDebug then uses counterexamples to explain and repair failures; IC3-Evolve uses agents offline to improve the model checker itself.

**Why these papers.** The selection covers two benchmark designs, specialized SVA training, learned PDR/SAT guidance, specification-to-assertion generation, counterexample debugging, and solver-code evolution. Generic RTL generation evaluated only through simulation is outside this representative subset.

The workflow is **design intent → signals and temporal properties → formal checking → counterexample diagnosis**, with learned solver guidance reducing the cost of the checking step. An SVA can compile and hold while expressing a weak or vacuous property, so functional equivalence, triggering behavior, and useful coverage matter alongside pass rate.

<a id="hardware-benchmark"></a>

### Benchmark

- AssertionBench: A Benchmark to Evaluate Large-Language Models for Assertion Generation [[Findings of NAACL'25; cited by 10](https://aclanthology.org/2025.findings-naacl.449/)]
  - Background: Many LLM assertion-generation demonstrations evaluate only a few hand-selected circuits and lack verified reference collateral.
  - Key problem & insight: Compare models on a shared collection of realistic RTL designs with formally checked assertions.
  - Proposed method — AssertionBench with two components:
    1. **Design and assertion corpus**: Curate OpenCores Verilog designs and reference assertions obtained with GoldMine and HARM.
    2. **Assertion evaluation**: Generate properties from RTL and distinguish syntactic validity from functional correctness using formal checking.
  - Results: 100 curated hardware designs; the study finds substantial room for improvement even for GPT-4o. Correctness of individual assertions is measured separately from completeness of design coverage.

- FVEval: Understanding Language Model Capabilities in Formal Verification of Digital Hardware [[DATE'25; cited by 10](https://doi.org/10.23919/DATE64628.2025.10992720)] [[arXiv'24](https://arxiv.org/abs/2410.23299)]
  - Background: Compiling an SVA does not establish that it expresses the intended temporal property or provides useful design coverage.
  - Key problem & insight: Separate natural-language translation from RTL-based assertion discovery and evaluate functional semantics rather than string matching alone.
  - Proposed method — FVEval with three tasks:
    1. **NL2SVA-Human**: Translate expert-written natural-language properties into assertions.
    2. **NL2SVA-Machine**: Evaluate scalable synthetic translation cases covering varied assertion structures.
    3. **Design2SVA**: Infer assertions directly from RTL and evaluate their relationship to reference design properties.
  - Results: The evaluated models can exceed 80% syntax correctness with examples while remaining substantially weaker on functional correctness; the benchmark exposes both strict and relaxed functional metrics.

- OpenLLM-RTL: Open Dataset and Benchmark for LLM-Aided Design RTL Generation [[ICCAD'24; cited by 27](https://doi.org/10.1145/3676536.3697118)] [[arXiv'25](https://arxiv.org/abs/2503.15112)]
  - Background: Public RTL resources are too small for controlled training studies and lack a shared formal benchmark for evaluating assertions generated from complete specifications.
  - Key problem & insight: Release aligned generation, verification, and training resources in one framework, and formally filter synthetic instruction-code pairs for functional consistency rather than syntax alone.
  - Proposed method — OpenLLM-RTL with three resources:
    1. **RTLLM 2.0**: Expand the natural-language-to-RTL benchmark to 50 hand-authored designs with tests and golden implementations.
    2. **AssertEval**: Provide 18 documented open-source designs, golden RTL, signal definitions, and one-click JasperGold scripts; score SVA syntax, FPV outcome, and cone-of-influence coverage.
    3. **RTLCoder-Data**: Generate 80K instruction/RTL pairs and a 7K subset whose functionality is checked through generated assertions and formal verification.
  - Results: Training on 80K samples raises Eval-Machine Pass@1 from 53.7% at 5K samples to 64.7%, 4.7 points above GPT-4. The verified 7K subset outperforms the raw 27K set across all benchmarks and the raw 50K set on 6/8 metrics while using under 20% of its training time.

- WaveformQA: Benchmarking LLM Temporal Reasoning on Digital Waveforms [[IEEE ICLAD'26 abridged; cited by 0](https://arxiv.org/abs/2607.20638)]
  - Background: Hardware debugging depends on long temporal traces, but code-oriented benchmarks do not isolate lookup, ordering, counting, correlation, and state-machine reasoning directly over waveforms.
  - Key problem & insight: Generate questions from simulated traces with exact answers and compare representations, revealing that token efficiency and reasoning clarity trade off differently for VCD and structured event data.
  - Proposed method — WaveformQA with three components:
    1. **Trace corpus**: Simulate five RISC-V cores into 13 traces spanning 338–4,185 signals and increasing transition windows.
    2. **Question generator**: Produce 360 questions in eight temporal/multi-signal categories over a 3-by-5 signal/transition complexity grid.
    3. **Dual representation evaluation**: Encode each trace as raw VCD or event-time JSON and report aggregate accuracy, in-context accuracy, context overflow, and error type.
  - Results: JSON improves in-context accuracy by 37–53 points over VCD despite using 15–30% more tokens. Claude Sonnet 4.6 reaches 64.7% aggregate/80.9% in-context accuracy and Gemini 2.5 Pro 57.8%/78.8%; accuracy falls 8–12 points as transitions grow from 5K to 30K.

- AssertLLM2: A Comprehensive LLM Benchmark for Assertion Generation from Design Specifications [[ICCAD'26 accepted; cited by 0](https://arxiv.org/abs/2605.27472)]
  - Background: Earlier SVA benchmarks give models golden RTL, use small modules or weak specification snippets, and reward assertions that compile or prove even when they are vacuous, shallow, or ineffective at finding bugs.
  - Key problem & insight: Separate pre-implementation bug prevention from post-implementation bug hunting, keep golden RTL grader-only, and score meaningful coverage and injected-fault detection alongside FPV.
  - Proposed method — AssertLLM2 with three components:
    1. **Realistic corpus**: Curate 83 system-level designs in 13 categories with structured textual specifications, raw PDFs, dependency-complete golden RTL, and mutation-derived faulty variants.
    2. **Two generation settings**: Generate from specification alone for **bug prevention**, or from specification plus a five-bug RTL for **bug hunting**.
    3. **Multi-axis grader**: Measure syntax, proven/counterexample/undetermined FPV status, COI, proof and formal coverage, and single-bug mutation kill ratio.
  - Results: GPT-5.2 has the highest syntax rate (95.0%), while Claude Sonnet 4.5 has the strongest average proof/formal coverage (23.30%/22.38%) and union bug-kill ratio (16.31%). Union generation improves coverage and bug killing, but even leading models remain below 30% proof coverage and 19% bug kill.

- HierSVA: A Data Synthesis Pipeline, Dataset, and Benchmark for LLM-Driven Hierarchical Hardware Formal Verification [[arXiv'26; cited by 0](https://arxiv.org/abs/2606.13706)]
  - Background: Existing SVA datasets flatten module context and evaluate syntax or proof pass/fail without vacuity, mutation, specification faithfulness, or hierarchical coverage.
  - Key problem & insight: Synthesize reference assertions bottom-up with assume-guarantee contracts, then evaluate generated assertion sets along independent formal-evidence axes.
  - Proposed method — HierSVA with three reusable components:
    1. **HierSVA-SP**: Preprocess hierarchical RTL, expand parameter combinations, and iterate LLM generation against VC Formal FPV, formal-core, and fault-analysis feedback.
    2. **HierSVA-DS**: Package 342 BaseJump STL modules at hierarchy depths 0–9 plus 28 expert-reviewed module/bug pairs with natural-language specifications.
    3. **HierSVA-B**: Score compilation, proven and non-vacuous assertions, specification faithfulness, mutation coverage, formal-core coverage, and static-versus-agentic behavior.
  - Results: Across twelve LLMs, module compile rate is 67.1%; 82.1% of assertions in evaluable runs prove non-vacuously, yet they detect only 70.2% of eligible faults and cover 36.2% of the formal core. On the deep subset, buggy-RTL recall is 0.87 but precision only 0.60.

- Robustness of LLM-Generated SystemVerilog Assertions to Semantics-Preserving RTL Transformations [[arXiv'26; cited by 0](https://arxiv.org/abs/2609.05658)]
  - Background: A model can score well on one RTL spelling while changing its assertion when operand order, identifiers, or redundant parentheses change without altering behavior.
  - Key problem & insight: Use metamorphic testing to separate aggregate point accuracy from conditional stability on the behaviors a model originally handled correctly.
  - Proposed method — controlled RTL-to-SVA robustness evaluation with three transformations:
    1. **Operand reordering**: Reorder commutative condition operands while preserving the path predicate.
    2. **Identifier renaming**: Deterministically rename signals without changing connectivity or behavior.
    3. **Parentheses insertion**: Add redundant grouping syntax and score each assignment behavior by exhaustive Boolean equivalence.
  - Results: On 40 programs/295 behaviors, Qwen2.5-Coder-7B loses 21.8% of baseline-correct behaviors under parentheses, while DeepSeek-Coder-V2-Lite loses 27.0%. DeepSeek's renamed-input accuracy rises by 9.8 points even though 19.5% of its previously correct behaviors regress, showing that net accuracy can hide instability.

- CircuitProver: Agentic Lean 4 Theorem Proving with Reusable Circuit Proof Library for Hardware Verification [[arXiv'26; citations N/A (method-paper set)](https://arxiv.org/abs/2607.27259)]
  - Background: Model checking verifies hardware instances independently and hides proof knowledge inside solver heuristics; interactive theorem proving yields reusable artifacts but requires extensive manual modeling and proof engineering.
  - Key problem & insight: Translate parameterized circuits into executable Lean semantics and accumulate successful proof plans and lemmas so related designs do not restart proof search from zero.
  - Proposed method — CircuitProver with three phases:
    1. **Hardware autoformalization**: Translate supported Chisel ASTs into parameterized Lean state/step/output models and formalize the natural-language specification.
    2. **Agentic proving**: Build a hardware-aware proof scaffold, query Lean for diagnostics, and iteratively generate or repair a kernel-checked proof.
    3. **Proof accumulation**: Distill successful trajectories into natural-language guidance and a Lean theorem library for future arithmetic, state-transition, invariant, and composition tasks.
  - Results: Solves all 63 tasks versus 58/63 for the matched vanilla agent and 31/63 for Chicala/Stainless. Reuse cuts proof rounds by 50.0%, verification time by 23.2%, and proof length by 16.3%; Opus 4.8 is the only tested backbone to reach 63/63.

- Rtl2Lean: Automated RTL-to-Lean Translation with Hierarchical Theorem Generation and Lemma Reuse [[arXiv'26; citations N/A (method-paper set)](https://arxiv.org/abs/2607.16855)]
  - Background: Proving SystemVerilog in an interactive theorem prover normally requires manual cycle semantics, state modeling, and repeated auxiliary lemmas for combinational and sequential behavior.
  - Key problem & insight: Compile RTL into a pure Lean transition system and generate proofs bottom-up, admitting LLM-proposed lemmas only after kernel checking so they become reusable library facts.
  - Proposed method — Rtl2Lean with three stages:
    1. **RTL-to-Lean semantic compilation**: Parse typed SystemVerilog and lower blocking/nonblocking assignments, bit-vector operations, and processes into `State`, `Inputs`, `Outputs`, and `step` definitions.
    2. **Four-layer theorem framework**: Generate combinational, sequential-block, step-level, and reachability/invariant theorems from local semantics to multi-cycle properties.
    3. **Autonomous lemma generation**: When automation stalls, prompt an LLM with the goal, definitions, and Lean errors; check, repair, and store accepted intermediate lemmas.
  - Results: Across six designs, generates and checks 403/403 theorems. Of 358 foundational lemmas, 287 are reusable automatically, an 80.2% reuse ratio, with all six designs achieving 100% proof success.

- AssertionForge: Enhancing Formal Verification Assertion Generation with Structured Representation of Specifications and RTL [[ICLAD'25; citations N/A (method-paper set)](https://arxiv.org/abs/2503.19174)]
  - Background: Specification-only generation misses internal RTL interactions, while raw specification-plus-RTL prompts overwhelm the model and fail to connect design intent with implementation paths.
  - Key problem & insight: Build one hardware-specific knowledge graph spanning both artifacts and retrieve multi-resolution paths around each target signal.
  - Proposed method — AssertionForge with three components:
    1. **Unified knowledge graph**: Extract specification entities/relations with an LLM, parse RTL structure and dataflow, and align names into one graph.
    2. **Multi-resolution context synthesis**: Combine a global design summary, signal-specific retrieval, and **GRW-AS** guided random walks through verification-relevant graph paths.
    3. **Context pruning and SVA generation**: Use an LLM to remove redundant context, produce verification plans and assertions, and check syntax/proof/COI with JasperGold.
  - Results: On five designs, AssertionForge yields 208 proven ETHMAC assertions versus 15 for AssertLLM and raises its COI metrics from roughly 45–50% to 98.84–100%; it also reaches near-complete COI coverage on APB, OpenMSP430, and SOCKIT. These are structural coverage results over proven properties, not requirement completeness.

- SANGAM: SystemVerilog Assertion Generation via Monte Carlo Tree Self-Refine [[ICLAD'25; citations N/A (method-paper set)](https://arxiv.org/abs/2506.13983)]
  - Background: Single-pass spec-to-SVA pipelines underuse waveforms and have little opportunity to explore alternative assertions or repair syntax and semantic omissions.
  - Key problem & insight: Treat each signal's assertion set as a search state and use critic plus formal-tool feedback inside Monte Carlo Tree Self-Refine.
  - Proposed method — SANGAM with three stages:
    1. **Specification processing**: Use Signal Mapper, Spec Analyzer, and Waveform Analyzer agents to build a signal-wise information bank from text, diagrams, and timing waveforms.
    2. **MCTSr assertion reasoning**: Select assertion nodes with UCT, expand them through critic/JasperGold feedback and RAG, rescore, and backpropagate quality over four rollouts.
    3. **Assertion combination**: Merge reasoning paths, correct syntax, and deduplicate the final signal-level SVA set.
  - Results: Produces 152 correct I2C assertions versus 50 for AssertLLM (+204%) and 46 RV-Timer assertions versus 11 for ChIRAAG (+254%). Both designs exceed 90% branch/toggle coverage and reach 74–84% property coverage in the reported JasperGold setup.

- AssertGen: Enhancement of LLM-aided Assertion Generation through Cross-Layer Signal Bridging [[ATS'25; citations N/A (method-paper set)](https://arxiv.org/abs/2509.23674)]
  - Background: Top-module assertions miss faults rooted in submodules because specification behaviors are scattered across text chunks and semantic levels and are not linked to concrete hierarchical RTL paths.
  - Key problem & insight: Extract cross-signal verification objectives first, then construct a cross-layer signal chain that maps each objective to the relevant code slices before SVA generation.
  - Proposed method — AssertGen with four steps:
    1. **Contextual entity extraction**: Iteratively retrieve descriptions of signals, registers, and related entities from specification chunks.
    2. **Verification-objective generation**: Use chain-of-thought prompting to combine fragmented behaviors into cross-signal properties.
    3. **Cross-layer bridging**: Map objectives through module hierarchy and signal chains to the matching Verilog segments.
    4. **Structured SVA generation**: Prompt over aligned objectives and RTL context, then check FPV, COI, proof-core coverage, and mutation detection.
  - Results: On SHA3, mutation bug detection reaches 15.35% versus AssertLLM's 11.36%; on PicoRV32 it reaches 23.20%, about twice the baseline, and on ECG 8.12% versus 1.34%/0.81%. The five-design study also reports higher FPV and structural/proof coverage than AssertLLM and Spec2Assertion.

- From Indiscriminate to Targeted: Functionally Critical Signal-Driven Assertion Generation using LLMs for Efficient RTL Verification [[arXiv'26; citations N/A (method-paper set)](https://arxiv.org/abs/2604.08932)]
  - Background: Existing methods generate many assertions across arbitrary signals, while industrial verification seeks high coverage and fault exposure with minimal properties and context cost.
  - Key problem & insight: Rank signals by their structural and behavioral influence, then give the model only the RTL slice needed to state a tightly constrained property about each selected signal.
  - Proposed method — AgileAssert with three components:
    1. **Semantic graph construction**: Build data/control dependencies and classify signal roles across module hierarchy.
    2. **Hybrid criticality ranking**: Combine PageRank, observability, output boost, mux-branch influence, and redundancy filtering to select top-$K$ signals.
    3. **Structure-aware slicing**: Extract signal-centered RTL cones and generate targeted SVAs incrementally until coverage/error-detection goals are met.
  - Results: Uses 66.68% fewer assertions on average while improving normalized block-level branch/statement/toggle/COI coverage by 8.99/8.27/12.61/18.78 points and CPU-level coverage by 19.27/17.65/11.37/23.76. It cuts prompt tokens by 64% and, when beating baselines on mutation detection, uses 72.74% fewer assertions.

- Assertain: Automated Security Assertion Generation Using Large Language Models [[MDTS'26; citations N/A (method-paper set)](https://arxiv.org/abs/2604.01583)]
  - Background: Functional SVA generators lack explicit threat models and standardized vulnerability mapping, causing security properties to be generic, hallucinated, or unrelated to the architecture's real attack surface.
  - Key problem & insight: Intersect RTL-structure and user-threat CWE mappings before generation, then inject the full design and refine every property against actual identifiers and temporal rules.
  - Proposed method — Assertain with three phases:
    1. **Knowledge mapping**: Classify the design, map both design category and threat strings to CWE sets, and retain their intersection as target weaknesses.
    2. **Context-aware generation**: Prompt per CWE with vulnerability knowledge and full RTL to produce a scenario, natural-language property, and SVA triplet over multiple sampled runs.
    3. **Self-reflection refinement**: Remove undefined signals, enforce reset/timing/non-vacuity rules, and emit a clean property module with both security and functional checks.
  - Results: Across 11 designs, Assertain improves correct-assertion count, unique CWE coverage, and detected architectural flaws over direct GPT-5 by 61.22%, 59.49%, and 67.92%. It finds 15 flaws versus 8 on MIPS and 15 versus 6 on MMU, while maintaining 88–100% functional correctness across designs.

<a id="hardware-training"></a>

### Training

- QiMeng-CodeV-SVA: Training Specialized LLMs for Hardware Assertion Generation via RTL-Grounded Bidirectional Data Synthesis [[DAC'26](https://github.com/wyt2000/CodeV-SVA)] [[arXiv'26](https://arxiv.org/abs/2603.14239)]
  - Background: General code models have limited high-quality SVA training data, while arbitrary synthetic NL–SVA pairs can drift semantically.
  - Key problem & insight: Ground assertion synthesis in actual RTL and use translation in both directions to filter data.
  - Proposed method — CodeV-SVA with two components:
    1. **RTL-grounded synthesis**: Generate assertions using large-scale open-source hardware code as the design context.
    2. **Bidirectional data selection and training**: Translate between natural-language properties and SVA, select consistent pairs, and train specialized assertion models.
  - Results: CodeV-SVA-14B reaches 75.8% Func.@1 on NL2SVA-Human and 84.0% on NL2SVA-Machine, matching or exceeding the compared GPT-5 and DeepSeek-R1 configurations. Bidirectional consistency is a data filter, not a proof that every NL–SVA pair is equivalent.

- NeuroPDR: Integrating Neural Networks in the PDR Algorithm for Hardware Model Checking [[MLCAD'23](https://guangyuhu.me/publication/mlcad23-neuropdr/)]
  - Background: PDR/IC3 runtime depends heavily on inductive generalization: finding compact blocking clauses that support progress across frames.
  - Key problem & insight: Learn generalization guidance from circuit structure while retaining the model checker's symbolic validity checks.
  - Proposed method — NeuroPDR with two components:
    1. **Message-passing GNN**: Encode circuit information and predict guidance for inductive-clause generalization.
    2. **PDR integration**: Apply the learned guidance during generalization, leaving final inductiveness and safety decisions to the solver.
  - Results: Reduces convergence time by approximately 26.4% on average in the reported experiments and transfers benefits to a different benchmark set. It optimizes verification search rather than generating RTL or specifications.

- NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks [[ICLR'24](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html)]
  - Background: Earlier GNN-enhanced SAT solvers can spend more time on repeated neural inference than they save in symbolic search.
  - Key problem & insight: A single prediction of useful variable phases can guide a complete CDCL run without repeated GPU calls.
  - Proposed method — NeuroBack with two components:
    1. **DataBack and phase prediction**: Train a GNN on 120,286 samples to predict variable values associated with satisfying assignments/backbone structure.
    2. **Kissat integration**: Query the model before solving, initialize phase guidance, and let the conventional CDCL solver finish the search on CPU.
  - Results: Enables Kissat to solve up to 5.2% more SATCOMP-2022 and 7.4% more SATCOMP-2023 problems. This is general SAT infrastructure relevant to verification, not a hardware-only benchmark.

<a id="hardware-agent"></a>

### Agent

- AssertLLM: Generating Hardware Verification Assertions from Design Specifications via Multi-LLMs [[ASP-DAC'25; cited by 51](https://zhiyuanyan.netlify.app/publication/aspdac25/)] [[arXiv'24](https://arxiv.org/abs/2411.14436)]
  - Background: Prior NL-to-SVA methods often assume an engineer has already extracted a clean sentence and identified its corresponding RTL signals.
  - Key problem & insight: Process complete specifications, including waveform diagrams, before attempting final assertion generation.
  - Proposed method — AssertLLM with two components:
    1. **Structured specification extraction**: Convert unstructured text and waveforms into template-based descriptions of signals and required behavior.
    2. **Assertion generation**: Use the structured descriptions and a customized LLM to generate SystemVerilog assertions tied to the design.
  - Results: Reports 88% syntactic-and-functional correctness for generated assertions and 97% cone-of-influence coverage. Coverage is a structural measure, not a claim that every architectural requirement has been verified.

- FVDebug: An LLM-Driven Debugging Assistant for Automated Root Cause Analysis of Formal Verification Failures [[arXiv'25](https://arxiv.org/abs/2510.15906)]
  - Background: A failed hardware property yields a multi-cycle counterexample, but engineers must still correlate waveforms, RTL, and design intent to identify the cause.
  - Key problem & insight: Convert the trace into causal structure before asking the LLM to explain and repair the failure.
  - Proposed method — FVDebug with four components:
    1. **Causal Graph Synthesis**: Build and consolidate a directed graph from the failure trace.
    2. **Graph Scanner**: Analyze batches of nodes with for-and-against prompting to identify plausible causes.
    3. **Insight Rover**: Explore candidate causal explanations and rank root-cause hypotheses.
    4. **Fix Generator**: Propose RTL repairs using multiple strategies and validate/rank the candidates.
  - Results: On 38 SVA-Eval-Human debugging cases, reaches 71.1% Pass@1 and 86.8% Pass@5 versus 60.5% and 65.8% for the direct-LLM baseline; also studies two production-scale counterexamples.

- IC3-Evolve: Proof-/Witness-Gated Offline LLM-Driven Heuristic Evolution for IC3 Hardware Model Checking [[IJCAI'26](https://www.ijcai.org/proceedings/2026/31)]
  - Background: IC3 performance depends on interacting heuristics, and automatically optimizing solver code can accidentally introduce unsound speedups.
  - Key problem & insight: Evolve small heuristic patches offline and require independently checkable evidence for every solved instance before accepting a patch.
  - Proposed method — IC3-Evolve with three components:
    1. **Slot-restricted patches**: Limit edits to defined heuristic regions rather than allow unrestricted changes to the solver.
    2. **Proof-/witness-gated evaluation**: Independently check SAFE certificates and replay UNSAFE traces; reject candidates that fail either requirement.
    3. **Compass&Jump**: Coordinate programmer and evaluator agents to select edit scopes and retain benchmark-improving candidates.
  - Results: In the reported clause-propagation evolution example, PAR2 drops from 1,050.61s to 943.07s and timeouts from 25 to 21; the deployed checker performs no LLM inference. Certificate gating validates evaluated runs, not a universal proof of the modified solver's implementation.
