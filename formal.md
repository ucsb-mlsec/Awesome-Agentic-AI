# Formal Verification and AI

## Table of Contents

- [Code](#code)
  - [Benchmark](#code-benchmark)
  - [Training](#code-training)
  - [Agent](#code-agent)
- [Math](#math)
  - [Benchmark](#math-benchmark)
  - [Training](#math-training)
  - [Agent](#math-agent)

## Code

<a id="code-benchmark"></a>

### Benchmark

- **VERINA** — **VERINA: Benchmarking Verifiable Code Generation** [[ICLR'26](https://arxiv.org/abs/2505.23135)] — function scope; 189 programming tasks in lean, with separate SpecGen, CodeGen, ProofGen, and combined settings.

  **Tasks**: Spec gen; impl gen; proof gen; combined settings. **Level**: Function level.

  **Best reported (one attempt per task)**: CodeGen — o3, 72.6% test-correct; SpecGen — o3, 52.3% sound and complete; ProofGen — Goedel-Prover-V2-32B, 11.2% Lean-checked. For combined CodeGen + SpecGen + ProofGen, o3 and o4-mini tie at 3.2% end-to-end success.

  - each task has a problem description, code implementation, specifications (pre-condition and post-condition), a proof (optional), and comprehensive test cases (input-output pairs, including both positive and negative)
  - specgen: give the model description and lean function signature, ask model to generate the speficication. When verifying the result, a model will try to prove the preconditions are equivalent and postconditions are equivalent given the precondition, if the model cannot prove, then use test cases.
    - Good precondition should reject illegal inputs and accept legal inputs, good postconditions should reject (legal input+wrong output) and accept (legal input+correct output).
  - CodeGen: Give the model a description and Lean function signature, optionally with the reference specification, generate the implementation and check its outputs against test cases. 
  - ProofGen: Give the model the description, function signature, implementation, and specification; generate a correctness proof and check it with Lean.
  - Combined:
      - CodeGen＋ProofGen: Give the LLM description＋Lean function signature＋reference specification; generate code and its correctness proof.
      - SpecGen＋ProofGen: Give the LLM description＋Lean function signature＋implementation; generate specification and a proof that the implementation satisfies it.
      - CodeGen＋SpecGen＋ProofGen: Give the LLM description＋Lean function signature; generate code and specification, then provide the reference specification to generate the code’s correctness proof.
    
- **TLA+-Bench** — **TLA+-Bench: An Execution-Grounded Benchmark and Dataset for Natural-Language to TLA+ Specification Generation** [[arXiv'26](https://arxiv.org/abs/2607.23425)] — focus on model checking and temporal logic; 403 TLC-runnable gold specifications and 897 parse-only silver specifications.

  **Tasks**: Model gen (states/transitions); spec gen (properties). **Level**: System-model level.

  **Best reported**: Claude Opus 4.5 — 16% TLC-correct on the 100-task system-model evaluation with only the description; 26% when the configuration's constant and property names are also supplied.

  - **LLM input**: A natural-language system description; a configuration including invariant names, specification names etc.
  - **LLM output**: A TLA+ specification of states, transitions, and properties.
  - **Verification**: Parse with SANY, bind to the reference configuration, and run TLC over the configured finite state space. Some heuristics to prevent the model from generating trivial properties.

  
- **DafnyBench** — **DafnyBench: A Benchmark for Formal Software Verification** [[TMLR'25](https://openreview.net/forum?id=yBgTVWccIx)] [[arXiv'24](https://arxiv.org/abs/2406.08467)] — 1,326 Dafny programs.

  **Tasks**: Proof gen (invariants and annotations). **Level**: Function / standalone-program level.

  **Best reported in the original paper**: Claude 3 Opus — 67.8% ± 1.7% verified on the 782-program evaluation set with up to 10 attempts.

  - **LLM input**: A Dafny implementation and its target specifications, with selected verification annotations (invariants, intermediate assertions) removed; verifier feedbacks.
  - **LLM output**: Missing verification annotations
  - **Verification**: Run Dafny on the completed program and require the target obligations to pass without changing the implementation or target specifications.

- **VeriSoftBench** — **VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean** [[arXiv'26](https://arxiv.org/abs/2602.18307)] — 500 proof obligations from 23 Lean repositories, each repo is a abstract verification project(e.g., an algorithm), doesn't necessarily correspond to a software repo.

  **Tasks**: Proof gen. **Level**: Repo context.

  **Best reported on the 500-task main set**: Gemini-3-Pro — 41.0% verified with curated dependencies and 34.8% with full-repository context (Pass@8, up to three repair rounds). On the separate, easier 100-task Aristotle-compatible subset, Aristotle reaches 69%; these scores are not directly comparable.
  remove the prof of one theorem, ask model to generate the prof

  - **Curated context**
    - **LLM input**: Target theorem, and author-selected reference-proof dependencies
    - **LLM output**: Lean proof
    - **Verification**: Check the theorem in its project environment.

  - **Full repository context**
    - **LLM input**: Target theorem, and whole repo with truncation
    - **LLM output**: Lean proof
    - **Verification**: Check the theorem in its project environment.

<a id="benchmark-vero"></a>

- **Vero** (agentic setting) — **Vero: Can AI Agents Build Formally Verified Software Repositories?** [[arXiv'26](https://arxiv.org/abs/2608.13522)] — 43 multi-module Lean repositories, 743 scored APIs, and 2,705 specifications.

  **Tasks**: Impl gen + proof gen; proof-only; specification audit. **Level**: Repo level (real world software repos translated to lean4).

  **Best reported**: Codex with GPT-5.5 (xhigh) — 27/43 fully solved repositories in code-and-proof and 25/43 in proof-only, with a 90-minute budget per run.

  **Verification target**: The benchmark's curated Lean rewrites of source repositories, not the original Python, Dafny, Verus, or Coq source repositories.


  - **Code-and-proof**
    - **LLM input**: Repository with some empty functions/APIs, and specifications
    - **LLM output**: API implementations and proofs across modules
    - **Verification**: Require every specification for a full repository solve.

  - **Proof-only**
    - **LLM input**: The same repository plus reference implementations
    - **LLM output**: Proofs for those implementations
    - **Verification**:Require every specification for a full repository solve.

  Agent can also submit a proof that a specification is wrong

  - **Impl: reference-code violation** Proof that the reference implementation fails at least one specification

  - **Spec: unsatisfiable requirement** Proof that no implementation satisfies it

  - **Spec: conflicting requirements** Proof of joint inconsistency plus individual satisfiability


- **VeriExploit** — Impl counterexample-to-reproduction; **VeriExploit: Automatic Bug Reproduction in Smart Contracts via LLMs and Formal Methods** [[ASE'25](https://pure.manchester.ac.uk/ws/portalfiles/portal/1632624289/ASE2025.pdf)] — a method evaluation suite for executable smart-contract bug reproduction.

  **Tasks**: Exploit reproduction from a supplied counterexample. **Level**: Contract level.

  **Reported system (no model comparison)**: VeriExploit uses GPT-4o and reaches 85.60% successful reproductions with the ESBMC backend, or 64.60% with SolCMC, on 100 contracts over five trials.

  - **Background**: SolCMC can find a property violation under abstract external-call behavior and report a counterexample without providing the external contract code that realizes that behavior. Given this counterexample, the LLM generates a concrete external contract; the verifier then checks whether its interactions with the vulnerable contract trigger the same violation.
  - **LLM input**: A vulnerable contract and an already available formal counterexample (generated by SolCMC; specifications are rule-based, like reentrancy checks, no divide by 0 etc).
  - **LLM output**: A reproduction contract—an external attacker/exploit contract that implements the behavior needed to reproduce the reported violation.
  - **Verification**: analyze the vulnerable and generated contracts together using SolCMC. A successful check produces a concrete interaction trace that triggers the same violation.


- **VeriBench** (agentic setting) — **VeriBench: An End-to-End Formal Verification Benchmark for AI Coding Agents in Lean 4** [[Preprint'26](https://openreview.net/pdf?id=vnXrEM5nNO)] [[Project](https://ehersch.github.io/veribench-blog/)] — Python-to-Lean formalization, starting from existing Python implementations.

  **Tasks**: Impl translation (Python to Lean); spec gen; proof gen; test translation. **Level**: Function / standalone-program level.

  **Best reported**: Codex (GPT-5.4) — 0.289 agent-skill composite score (combining compilation, proof completion, and specification coverage); this is not a 28.9% solve rate.

  - **LLM input**: A Python source file containing a reference implementation, a docstring describing its intended behavior, and tests.
  - **LLM output**: A Lean 4 implementation, translated tests, formal specifications and proof attempts for those specs.
  - **Verification**:
    1. **Implementation and tests**: Typecheck the generated Lean file and check its translated tests against the generated Lean implementation.
    2. **Proofs**: Check the generated theorem proofs with Lean and measure how many are completed without `sorry` placeholders.
    3. **Specification coverage**: Compare the generated theorem statements with human-curated reference specifications using an LLM judge.

- **DafnyCOMP** — **Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification** [[ICLR'26](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c04d37be05ba74419d2d5705972a9d64-Abstract-Conference.html)] — multiple interacting functions and their data dependencies.

  **Tasks**: Spec gen; proof gen (supporting annotations). **Level**: Multi-function level.

  **Best reported**: Gemini 2.5 Pro — 2.00% verified Pass@8 on the 300-task chain split (independent attempts). With three verifier-feedback turns, DeepSeek-R1 and o4-mini tie at 9.67% verified on that split.

  - **LLM input**: A Dafny program with its executable logic retained and contracts/supporting annotations to reconstruct across function boundaries.
  - **LLM output**: Preconditions, postconditions, and proof annotations strong enough for callers and callees to compose.
  - **Verification**: Verify the complete composed program with Dafny; separately successful local proofs are insufficient when caller obligations fail. Acceptance establishes correctness relative to the generated contracts, not their faithfulness to an unstated intent.

- **OSVBench** — **OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification** [[AAAI'26](https://ojs.aaai.org/index.php/AAAI/article/view/40437)] — operating-system state and syscall behavior.

  **Tasks**: Spec gen (syscall state transitions). **Level**: Function/syscall level with kernel context.

  **Best reported**: Doubao-1.5-pro — 55.10% Pass@1 across 245 tasks with a five-shot prompt.

  - **LLM input**: A syscall description, the permitted state-transition programming model, verification assumptions, and kernel implementation context that may contain injected bugs.
  - **LLM output**: An executable state-machine specification for the syscall.
  - **Verification**: Run the Hyperkernel verifier and compare the generated specification's verdicts with reference-specification verdicts across kernel variants. A specification that merely agrees with buggy code is insufficient.

- VerusBench (agentic setting) — *AutoVerus: Automated Proof Generation for Rust Code* [[OOPSLA'25](https://doi.org/10.1145/3763174)] [[arXiv'24](https://arxiv.org/abs/2409.13082)] — the original benchmark contains 150 Rust/Verus proof tasks; evaluation subsets vary across papers.

  **Tasks**: Proof gen (invariants and annotations). **Level**: Function / standalone-program level.

  - **LLM input**: Rust/Verus code and fixed target contracts with proof annotations to complete; repair attempts can include Verus errors.
  - **LLM output**: Invariants, assertions, ghost code, and supporting proof annotations.
  - **Verification**: Run Verus and require the target obligations to pass while preserving executable code and target contracts.

- RVBench — *Towards Repository-Level Program Verification with Large Language Models* [[LMPL'25](https://arxiv.org/abs/2509.25197)] — tasks drawn from four Verus projects.

  **Tasks**: Proof gen. **Level**: Repo context; local verification obligations.

  - **LLM input**: Fixed Verus code and contracts, a proof hole, and project context.
  - **LLM output**: Missing proof annotations using the project's definitions and lemmas.
  - **Verification**: Run Verus in the project environment. Completing the target obligation does not establish completion of the entire repository.

- Selene — *Selene: Pioneering Automated Proof in Software Verification* [[ACL'24](https://aclanthology.org/2024.acl-long.98/)] — proof tasks drawn from the seL4 verification development.

  **Tasks**: Proof gen. **Level**: Repo context; individual theorem obligations.

  - **LLM input**: An Isabelle lemma, fixed definitions and specifications, and relevant seL4 project context.
  - **LLM output**: Isabelle proof commands completing the supplied lemma.
  - **Verification**: Check the completed lemma in Isabelle within the verification development. A successful lemma proof does not mean the entire kernel has been verified by the model.

- Verus-SpecGym (benchmark: Verus-SpecBench) (agentic setting) — *Verus-SpecGym: An Agentic Environment for Evaluating Specification Autoformalization* [[arXiv'26](https://arxiv.org/abs/2605.26457)] — function scope; 581 Codeforces-derived specification tasks.

  **Tasks**: Spec gen. **Level**: Function level.

  - **LLM input**: A problem statement and Verus specification scaffold, with access to the verifier, shell, and filesystem.
  - **LLM output**: Input assumptions and required output behavior encoded as a Verus specification.
  - **Verification**: Execute specifications through Verus `exec_spec` and compare their acceptance of input/output cases with official tests and adversarial Codeforces hacks. This tests both omitted requirements and overrestrictive specifications; it is not a universal intent-equivalence proof.

- miniCodeProps — *miniCodeProps: a Minimal Benchmark for Proving Code Properties* [[arXiv'24](https://arxiv.org/abs/2406.11915)] — small, self-contained Lean programs and properties.

  **Tasks**: Proof gen. **Level**: Function/theorem level.

  - **LLM input**: A fixed program, its definitions, and a formal statement about its behavior.
  - **LLM output**: Lean tactics or a complete proof of the supplied property, with code and property unchanged.
  - **Verification**: Check the completed theorem in Lean; induction or auxiliary lemmas may be necessary even for short programs.

- FVAPPS — *Proving the Coding Interview: A Benchmark for Formally Verified Code Generation* [[LLM4Code@ICSE'25](https://github.com/quinn-dougherty/fvapps)] [[arXiv'25](https://arxiv.org/abs/2502.05714)] — coding-problem scope; 4,715 samples, including 1,083 curated samples.

  **Tasks**: Impl gen; proof gen. **Level**: Function / standalone-program level.

  - **LLM input**: A Lean 4 coding task with implementation/proof holes and supplied correctness requirements.
  - **LLM output**: The missing implementation and proofs that it meets those requirements.
  - **Verification**: Check the completed artifacts in Lean without unfinished proofs or added untrusted assumptions. The guarantee concerns the supplied statements; dataset size does not imply that every specification is equally faithful to the original problem.

- Vericoding benchmark — *A benchmark for vericoding: formally verified program synthesis* [[arXiv'25](https://arxiv.org/abs/2509.22908)] [[Dafny@POPL'26](https://popl26.sigplan.org/details/dafny-2026-papers/13/A-benchmark-for-vericoding-formally-verified-program-synthesis)] — 12,504 formal specifications across Dafny, Verus/Rust, and Lean; aggregates multiple sources, including FVAPPS and VERINA.

  **Tasks**: Impl gen; proof gen. **Level**: Function / standalone-program level.

  - **LLM input**: A formal functional specification with the implementation removed; a separate setting additionally supplies a natural-language description.
  - **LLM output**: An implementation plus the annotations or proof scripts required by the target language.
  - **Verification**: Run the corresponding Dafny, Verus, or Lean checker against the fixed specification. The language subsets have different source distributions.

- AlgoVeri — *AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms* [[arXiv'26](https://arxiv.org/abs/2602.09464)] — 77 classical algorithms aligned across Dafny, Verus, and Lean.

  **Tasks**: Impl gen; proof gen. **Level**: Function / standalone-program level.

  - **LLM input**: A formal specification for an algorithm in the target language, with matching functional contracts across the language versions.
  - **LLM output**: The algorithm implementation and the verification annotations or explicit proof scripts needed to establish its correctness.
  - **Verification**: Check the generated implementation and proof with the target language's verifier against the supplied contract. The aligned tasks support comparisons across verification languages.

- CLEVER — *CLEVER: A Curated Benchmark for Formally Verified Code Generation* [[NeurIPS'25 — Datasets and Benchmarks](https://arxiv.org/abs/2505.13938)] — 161 HumanEval-derived function tasks.

  **Tasks**: Spec gen; impl gen; proof gen (spec equivalence and implementation correctness). **Level**: Function level.


  The evaluation has four stages; the reference specification is hidden during specification generation and supplied for certification.

  - **Specification generation**
    - **LLM input**: Natural-language task and Lean scaffold/signatures
    - **LLM output**: Formal specification
    - **Verification**: Check compilation; semantic certification follows below.

  - **Specification certification**
    - **LLM input**: Generated and reference specifications, equivalence theorem
    - **LLM output**: Equivalence proof
    - **Verification**: Lean checks equivalence.

  - **Implementation generation**
    - **LLM input**: Natural-language task, function signature, generated specification
    - **LLM output**: Lean implementation
    - **Verification**: Check compilation; correctness certification follows below.

  - **Implementation certification**
    - **LLM input**: Generated implementation, reference specification, correctness theorem
    - **LLM output**: Correctness proof
    - **Verification**: Lean checks implementation correctness against the reference specification.

  A full solve requires both certifications; compiling artifacts alone is insufficient.

- VerifyThisBench — *VerifyThisBench: Generating Code, Specifications, and Proofs All at Once* [[arXiv'25](https://arxiv.org/abs/2505.19271)] — 41 verification-competition challenges represented as 154 tasks across seven tools, plus 580 completion tasks in VerifyThisBenchXS.

  **Tasks**: Spec gen; impl gen; proof gen (including loop invariants). **Level**: Function / multi-function / module level, depending on the challenge.


  - **VerifyThisBench: full task**
    - **LLM input**: Informal challenge and target verification language/tool
    - **LLM output**: Implementation or model, specifications, and proof annotations/scripts required by the challenge
    - **Verification**: Compile and verify with the designated tool; diagnostics can drive repair.

  - **XS Code-Gen: 226 tasks**
    - **LLM input**: Function specifications; implementation and proof annotations removed
    - **LLM output**: Implementation and supporting proof annotations
    - **Verification**: Verify the completed program against the supplied specifications.

  - **XS Specification-Gen: 233 tasks**
    - **LLM input**: Implementation and proof annotations; function specifications removed
    - **LLM output**: Function specifications
    - **Verification**: Verify the completed artifact with the restored specifications.

  - **XS Loop-Gen: 121 tasks**
    - **LLM input**: Specifications and implementation; loop invariants removed
    - **LLM output**: Loop invariants
    - **Verification**: Check invariant obligations and overall program verification.

  Verifier acceptance concerns the encoded requirements; it does not independently certify their faithfulness to the informal challenge.

<a id="code-training"></a>

### Training

- **[SmartInv: Multimodal Learning for Smart Contract Invariant Inference](https://www.cs.columbia.edu/~junfeng/papers/smartinv-sp24.pdf)** [IEEE S&P'24]
  - sft, lora, base model LLaMA-7B
  - manually annotates 572 contracts with their transaction context, critical code locations, relevant invariants (for known vulns).
  - use question templates to turn the annotation into staged questions, eventually providing invariants. e.g.,
    - “what is this transaction and where should we check?”
    - “what invariant belongs there?”
    - “which invariants are most likely to reveal a bug?”

- **[Automated Proof Generation for Rust Code via Self-Evolution](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b2e20d7402c9985eae4ba924c65370a8-Abstract-Conference.html)** [ICLR'25]
  - SFT of DeepSeekCoder-33B-Instruct for function-level Verus/Rust proof generation.
  - Initially, to build tasks, GPT-4o adapts small MBPP/CodeNet programs to Verus-compatible Rust and generates `requires`/`ensures`.
  - Train two models
    - Specification-generation model (to build proof tasks): input = Verus-compatible Rust implementation + docstring; target = `requires`/`ensures`. GPT-4o supplies the initial examples; a fine-tuned DeepSeekCoder generates later candidates, which are filtered against input-output tests.
    - Proof+impl geenration model:
      - Proof-generation SFT: input = fixed Rust implementation + `requires`/`ensures`; target = the full function with proof annotations such as loop invariants and assertions. Keep a target only if Verus verifies the function against its specification.
      - Repair SFT: input = an earlier failed proof attempt + its Verus error; target = a later, Verus-verified version of the full function.
  - **Self-evolution**: GPT-4o supplies initial verified proofs; the fine-tuned model generates new proof candidates for the existing function/specification pairs, Verus selects successful proofs, and the model is fine-tuned again. 

- **[Towards Neural Synthesis for SMT-Assisted Proof-Oriented Programming](https://www.microsoft.com/en-us/research/publication/towards-neural-synthesis-for-smt-assisted-proof-oriented-programming/)** [ICSE'25]
  - SFT of Phi-2, Orca-2, and StarCoder on top-level definitions extracted from eight real F* projects.
  - Task construction: remove one definition's body, keeping its declared type, which may specify a program's behavior or state a lemma. The original body becomes the SFT target.
  - LLM input: the type/signature, preceding file context, retrieved similar examples, and relevant definitions from the project. LLM output: one F* body, either an implementation or a proof.
  - Verification: insert the body at its original location and run F*'s type checker with SMT support. `verify@k` counts a task solved if at least one of k generated bodies passes. Each task completes one definition in repository context, not the whole repository.

- **[Re:Form — Reducing Human Annotations in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny](https://arxiv.org/abs/2507.16331)** [arXiv'25]
  - SFT followed by GRPO on Qwen2.5-based models for Dafny specification generation.
  - Data: collect public Dafny programs and use Claude 3.5 Sonnet to translate Python programs and add specifications; Dafny errors guide up to ten repair rounds. Verifier-accepted programs supply SFT examples.
  - Model input: a Dafny implementation with specification annotations removed. Model output: the full program with `requires`, `ensures`, loop invariants, and other supporting annotations.
  - RL reward: syntax validity, Dafny verification, and specification strength relative to the Claude-generated reference. Dafny checks `P_ref ⇒ P_gen` for preconditions and `P_ref ∧ Q_gen ⇒ Q_ref` for postconditions; this discourages weak specifications such as `ensures true`.
    
- **SpecRL: Reinforcement Learning with Test-Based Completeness Rewards for Formal Specification Synthesis** [[arXiv'26](https://arxiv.org/abs/2604.05820)]
  - Re:Form rewards a specification if Dafny proves it at least as strong as a reference; SpecRL instead rewards how many wrong outputs it rejects. For each method, an LLM proposes five inputs; the implementation supplies the real outputs, and the LLM proposes three wrong outputs per input. The reward measures only these finite tests.

- **Formal Disco: Scalable Open-Ended Generation of Formally Verified Programs** [[arXiv'26](https://arxiv.org/abs/2607.04631)]
  - **Goal / corpus**: Generate complete Dafny, Verus, and Frama-C programs containing implementations, specifications, and proof annotations. 
  - **Agentic data generation**: Three LLM workers. Initiator takes a random GitHub README and up to two language-documentation (Dafny, Verus, Frama-C) snippets and outputs a new program. Fixer takes a failed program and compiler/verifier errors and outputs a repair diff. Extender takes a verified program and outputs a diff adding a method or lemma. Compile and verify after each change; successful programs enter the corpus.
  - Two SFTs, sft on task generation and also sft on annotation generation.
    - **Worker SFT**: Record each worker's prompt, response, and verification result. Seed successful examples with Claude 4.5 Sonnet/Opus, then LoRA/SFT Qwen2.5-Coder-32B-Instruct on successful calls. Let Qwen generate the next round.
    - **Downstream SFT tasks**: From a verified program, remove assertions and loop invariants: input = implementation plus specifications; output = a diff restoring the annotations. Or remove a lemma body: input = program plus lemma statement; output = its proof. Separately fine-tune Qwen2.5-Coder-32B on these pairs and check generated completions with the verifier.

- **Propose, Solve, Verify: Self-Play Through Formal Verification** [[ICML'26](https://icml.cc/virtual/2026/poster/63571)]
  - **Goal / seed tasks**: Train a model to implement Verus function specifications. In the headline test-time-training setting, each benchmark's existing specifications form the initial question pool; reference implementations are not training answers. A question specifies a function interface and optional `requires`/`ensures`, but leaves the implementation blank.
  - **Solver input / output**: Given one specification and a worked prompt example, Qwen2.5-Coder-3B-Instruct generates a Rust/Verus implementation with any needed proof annotations. Sample ten solutions per question and check each against the fixed specification with Verus.
  - **Solver training (RFT/SFT)**: For each question with a verified solution, retain at most one `(input = specification, target = verified implementation plus proof annotations)` pair. Fine-tune the solver on these pairs; questions with no verified solution add no training example.
  - **Proposer input / output**: Its prompt contains twelve existing specifications labeled Easy/Medium/Hard/Impossible by the current solver's pass rate, plus a requested difficulty. The LLM reasons about the examples and writes a *new* Verus function signature and specification, without an implementation; no algorithm or application scenario is prescribed. Parse, deduplicate, and check that proposals compile as specifications before adding them to the next round. This does not establish that a proposal is nontrivial or solvable.
  - **Self-play / agentic?** The solver's verified pass rates refresh the proposer's examples; new questions supply the solver's next training opportunities. Only solver weights change; the proposer adapts through its prompt. Individual calls generate complete candidates rather than exploring tools.
  - **Evaluation**: On MBPP-Verified in the headline test-time-training setting, PSV-Verus reaches 36.78% pass@1.

<a id="code-agent"></a>

### Agent

Representative methods are grouped by their main technical contribution. Closely related approaches are consolidated; selection considers verification evidence and evaluation scope, not success rates across incomparable benchmarks.

#### Verified synthesis and proof repair

- **ExVerus: Verus Proof Repair via Counterexample Reasoning** [[ICML'26](https://icml.cc/virtual/2026/poster/65247)] [[arXiv'26](https://arxiv.org/abs/2603.25810)]
  - Background: Textual verifier errors often omit the concrete behavior that explains why a proposed invariant fails.
  - Key problem & insight: Use validated source-level counterexamples to help the LLM generalize a failure into a stronger invariant.
  - Proposed method — ExVerus with two components:
    1. **Counterexample generation and validation**: Recover concrete behavior associated with a failed proof and check that the example is meaningful.
    2. **Counterexample-guided repair**: Ask the LLM to explain and block the failure through inductive proof annotations, then rerun Verus.
  - Results: Reports 38% more solved tasks on average than AutoVerus, with approximately $0.04 average cost per task and 4.25x lower cost. This is a within-paper comparison; the paper also documents that Verus-version differences affect reproduced AutoVerus scores.

- AlphaVerus: Bootstrapping Formally Verified Code Generation through Self-Improving Translation and Treefinement [[ICML'25](https://arxiv.org/abs/2412.06176)]
  - Background: Verus lacks the verified examples available in higher-resource verification languages.
  - Key problem & insight: Translate verified source programs, repair them with the target verifier, and recycle successful translations as examples.
  - Proposed method — AlphaVerus with three phases:
    1. **Exploration**: Generate candidate translations from Dafny into Verus.
    2. **Treefinement**: Search over repairs using verifier feedback rather than commit to a single linear repair trajectory.
    3. **Filtering**: Reject misaligned specifications and programs that exploit the verification objective; reuse accepted artifacts as in-context examples.
  - Results: Llama-3.1-70B reaches 33% on Verified-HumanEval in the reported setup without weight fine-tuning. “Self-improving” here refers to the translation/example pipeline, not gradient updates.

- Guiding Enumerative Program Synthesis with Large Language Models [[CAV'24](https://doi.org/10.1007/978-3-031-65630-9_15)]
  - Background: Enumerative SyGuS solvers respect exact logical specifications but can explore a large grammar; a standalone LLM often fails those exact constraints.
  - Key problem & insight: Use the LLM to shape the enumerator's search distribution instead of replacing symbolic synthesis.
  - Proposed method — LLM-guided enumeration with three components:
    1. **Standalone LLM synthesis**: Try to generate a complete program directly from the formal specification.
    2. **pCFG-synth**: Use LLM-derived grammar weights to guide probabilistic enumeration, with both enumeration-based and A* variants.
    3. **iLLM-synth**: Alternatively interleave LLM queries with search so the model can react to enumerator progress.
  - Results: On 609 SyGuS tasks, standalone LLM plus A*-pCFG-synth solves 80.1%, versus 49.8% for the LLM and 68.1% for cvc5; the interleaved A*-iLLM-synth variant reaches 67.0%. The best result comes from the combined offline-guidance configuration.

#### Invariant synthesis

- Clause2Inv: A Generate-Combine-Check Framework for Loop Invariant Inference [[ISSTA'25](https://conf.researchr.org/details/issta-2025/issta-2025-papers/44/Clause2Inv-A-Generate-Combine-Check-Framework-for-Loop-Invariant-Inference)]
  - Background: Repeated guess-and-check can miss a correct invariant because the required clauses occur in different failed guesses.
  - Key problem & insight: Separate discovering useful clauses from choosing their logical combination.
  - Proposed method — Clause2Inv with two components:
    1. **LLM-based clause generator**: Accumulate clauses across proposed invariants.
    2. **Counterexample-driven clause combinator**: Use verification counterexamples to choose combinations and submit reconstructed invariants for checking.
  - Results: Solves 312/316 linear and 44/50 nonlinear tasks, at least 93 and 16 more than the evaluated baselines respectively; the combinator can also wrap existing generators.

#### Specification validation and formal modeling

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

- ABSINT-AI: Agentic Heap Abstractions for Abstract Interpretation [[ICML'26](https://openreview.net/forum?id=ozu9ZRETYE)]
  - Background: TAJS/WALA-style analyses use fixed heap abstractions that can merge unrelated JavaScript objects and produce false positives.
  - Key problem & insight: Let an LLM choose abstractions through a restricted interface while the abstract interpreter retains responsibility for sound state transfer.
  - Proposed method — ABSINT-AI with two components:
    1. **Adaptive heap abstractions**: Use names and access patterns to choose different abstractions for different objects.
    2. **Sound abstract interpretation**: Restrict agent decisions to permitted abstraction choices; do not let the agent directly invent or edit abstract states.
  - Results: Reports up to 34% fewer false positives than fixed-abstraction analyses while retaining formal guarantees; agentic interaction improves over non-agentic LM predictions by 25% in the paper's comparison.

#### Repository and systems proofs

- **Rango: Adaptive Retrieval-Augmented Proving for Automated Software Verification** [[ICSE'25](https://www.cs.cornell.edu/~lerner/papers/rango-final.pdf)] [[arXiv'24](https://arxiv.org/abs/2412.14063)]
  - Background: Static retrieval misses the changing relevance of lemmas as a proof progresses and often ignores reusable proof examples from the current project.
  - Key problem & insight: Retrieve both premises and analogous proofs anew at each proof state.
  - Proposed method — Rango with two components:
    1. **Adaptive retrieval**: Select relevant definitions, lemmas, and proof examples from the available project context.
    2. **Fine-tuned proof search**: Condition the next-step model on these examples, check candidates in Coq, and repeat retrieval after state changes.
  - Results: Releases CoqStoq with 2,226 projects and 196,929 theorems; proves 32.0% on the curated evaluation, 29% more theorems than Tactician.

- **An AI Approach to Verified Production Cryptographic Libraries** [[arXiv'26](https://arxiv.org/abs/2608.00965)]
  - Verification target: Existing Rust production-library source, kept executable-code-identical while Verus specifications and proofs are added. The reported chacha20 result covers the portable backend, not SIMD backends.
  - Background: Most proof-generation tasks supply internal contracts and lemmas; production cryptographic libraries also require discovering those intermediate interfaces.
  - Key problem & insight: Plan internal specifications and proofs while mechanically preventing changes that weaken the trusted problem statement.
  - Proposed method — CryptoProver with three components:
    1. **Specification and proof synthesis**: Start from high-level API contracts and generate internal Verus specifications and proofs without editing executable code.
    2. **Mechanical gates**: Reject specification weakening, new axioms, and cross-module breakage.
    3. **Isolation**: Block retrieval of reference proofs, including through repository history, while exposing a fixed trusted specification/fact library.
  - Results: Constructs an independent curve25519-dalek proof in 11.4 hours at $466.99 recorded API cost, and verifies RustCrypto chacha20 against an RFC 8439 specification; trusted API contracts and arithmetic facts remain supplied inputs.

<a id="smart-contract-agents"></a>

#### Smart contracts

- **PropertyGPT: LLM-driven Formal Verification of Smart Contracts through Retrieval-Augmented Property Generation** [[NDSS'25](https://www.ndss-symposium.org/ndss-paper/propertygpt-llm-driven-formal-verification-of-smart-contracts-through-retrieval-augmented-property-generation/)]
  - Background: Smart-contract provers need contract-specific properties that expert auditors normally write manually.
  - Key problem & insight: Retrieve human-written properties from related contracts, adapt them with an LLM, and filter them through compiler and verification feedback.
  - Proposed method — PropertyGPT with three components:
    1. **Property retrieval**: Embed existing properties and retrieve relevant exemplars for the target code.
    2. **Generation and repair**: Adapt properties, use compilation/static-analysis errors to repair them, and rank candidate relevance.
    3. **Formal checking**: Validate candidate properties and investigate violations in the target contract.
  - Results: Recovers 80% of reference properties at 64% precision in the reported human assessment; detects 9/13 CVEs and 17/24 historical attack incidents.

- Augmenting Smart Contract Decompiler Output Through Fine-Grained Dependency Analysis and LLM-Facilitated Semantic Recovery [[TSE'25](https://doi.org/10.1109/TSE.2025.3623325)] [[arXiv'25](https://arxiv.org/abs/2501.08670)]
  - Background: Gigahorse-style decompilers lose function boundaries, variable types, and contract attributes; unconstrained LLM edits can change program behavior.
  - Key problem & insight: Recover readable semantics using dependency information, while independently rejecting behavior-changing edits.
  - Proposed method — SmartHalo with three components:
    1. **Dependency Graph Construction**: Represent type, state, and control-flow dependencies to select relevant code context.
    2. **LLM-driven Semantic Enrichment**: Guide recovery with dependency-derived context, reasoning steps, and candidate types or attributes.
    3. **Correctness Verification**: Apply a **Program-behavior Equivalence Check** using symbolic summaries and Z3, plus a **Rule-based Type Violation Check**; return violations for revision.
  - Results: The revised arXiv version reports GPT-4o-mini precision of 91.32% for boundaries, 90.40% for types, and 80.66% for attributes. Downstream reentrancy-analysis precision improves from 72.16% with SliSE to 80.41% with SliSE+SmartHalo. Equivalence is checked against the initial decompiler output, not directly against original bytecode.

- **VeriExploit: Automatic Bug Reproduction in Smart Contracts via LLMs and Formal Methods** [[ASE'25](https://pure.manchester.ac.uk/ws/portalfiles/portal/1632624289/ASE2025.pdf)]
  - Background: SolCMC can report a property violation under abstract external-call behavior without providing the external contract code that realizes that behavior. The task is to implement the required external behavior and check whether it triggers the same violation in the vulnerable contract.
  - Key problem & insight: Treat the formal counterexample as a construction guide for an executable bug reproduction.
  - Proposed method — VeriExploit with two components:
    1. **Reproduction synthesis**: Give the LLM the vulnerable contract and counterexample to generate a reproduction contract: an external attacker/exploit contract that implements the required behavior.
    2. **Validation and refinement**: Check compilation and use bounded cross-contract verification (BCCV) to validate interactions between the generated and vulnerable contracts, producing a concrete trace of the same violation or feedback for repair.
  - Results: Achieves 85.60% reproduction success on the authors' benchmark. The task starts from a supplied vulnerable contract and counterexample rather than discovering every bug from scratch.


#### Verified policies for LLM agents

- VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation [[arXiv'25](https://arxiv.org/abs/2510.05156)]
  - Background: Prompt-level safety instructions do not enforce constraints on every action an autonomous agent may propose.
  - Key problem & insight: Verify a behavioral policy offline and enforce it through a runtime monitor rather than trust the agent to remember the policy.
  - Proposed method — VeriGuard with two stages:
    1. **Offline policy construction**: Clarify intent, synthesize a policy, and iteratively test and formally check it against explicit safety requirements.
    2. **Online monitoring**: Validate proposed agent actions against the preverified policy before execution.
  - Results: Reports attack success falling from 53.5% without defenses to 0% for the full system on the evaluated attack suite. This empirical zero is limited to that suite; formal guarantees depend on the modeled policy and enforcement boundary.

## Math

<a id="math-benchmark"></a>

### Benchmark

PutnamBench covers formalized undergraduate competition problems. Subsequent use includes Goedel-Prover-V2.

- **PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition** [[NeurIPS'24 — Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/file/1582eaf9e0cf349e1e5a6ee453100aa1-Paper-Datasets_and_Benchmarks_Track.pdf)]

  **Tasks**: Proof gen. **Level**: Individual mathematical theorem.

  **Best reported in the original paper**: DSP using GPT-4o — 4/640 theorems proved in Isabelle (pass@10). In Lean 4, GPT-4o and COPRA using GPT-4o each prove 1/640; the language-specific settings should be compared separately.

  - **LLM input**: A formalized Putnam competition problem with the required definitions in Lean 4, Isabelle, or Coq.
  - **LLM output**: A proof of the supplied formal statement.
  - **Verification**: Check the proof in the corresponding assistant without changing the theorem or introducing unproved assumptions; use the task set of the stated benchmark version.

<a id="math-training"></a>

### Training

Only papers that train or fine-tune LLM parameters are included in this section.

- **DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** [[ICLR'25](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b3b55c366d641c07180c40e4f978f311-Abstract-Conference.html)]
  - **LLM training**: Supervised fine-tuning followed by reinforcement learning from Lean proof-assistant feedback; RMaxTS is an additional inference-time search component.
  - Background: Supervised proof completion underuses Lean's feedback, and sparse complete-proof rewards make search inefficient.
  - Key problem & insight: Use the prover both as a training reward source and as an observable state space for exploration.
  - Proposed method — DeepSeek-Prover-V1.5 with two components:
    1. **Reinforcement Learning from Proof Assistant Feedback**: Update the proof-completion policy using Lean verification outcomes.
    2. **RMaxTS**: Organize Monte Carlo search around intermediate tactic states and give intrinsic exploration rewards for discovering new states.
  - Results: The 7B RL model with RMaxTS reaches 63.5% on miniF2F-test at the paper's largest mixed-prompt budget of 32 x 6,400 samples; the SFT counterpart reaches 60.2% at that budget.

- STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving [[ICML'25](https://proceedings.mlr.press/v267/dong25h.html)]
  - **LLM training**: Iteratively fine-tune the conjecturer and prover on selected conjectures and formally verified proofs.
  - Background: Expert iteration on fixed statements plateaus when the prover cannot solve enough remaining problems to obtain new training data.
  - Key problem & insight: Learn to propose problems near the current prover's frontier of difficulty.
  - Proposed method — Self-play Theorem Prover (STP) with two roles:
    1. **Conjecturer**: Train on generated conjectures that are barely provable by the current model, gradually shifting the curriculum.
    2. **Prover**: Attempt conjectures and improve through expert iteration on formally verified proofs; feed successes back to the conjecturer.
  - Results: On LeanWorkbook, proves 28.5% of statements versus 13.1% for prior expert iteration; reaches 65.0% miniF2F-test and 23.9% ProofNet-test at pass@3200. Its self-play uses verified data generation and fine-tuning, not merely inference-time debate.

- DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition [[arXiv'25](https://arxiv.org/abs/2504.21801)]
  - **LLM training**: Cold-start supervised training on synthesized reasoning and proofs, followed by reinforcement learning for Lean proof generation.
  - Background: Whole-proof RL receives little useful signal on problems whose complete proofs are initially beyond the model.
  - Key problem & insight: Recursively solve simpler subgoals and assemble them into training examples that connect informal plans to formal proofs.
  - Proposed method — DeepSeek-Prover-V2 with two stages:
    1. **Recursive cold-start synthesis**: Use DeepSeek-V3 to decompose problems, solve subgoals, and combine checked subproofs with step-by-step reasoning.
    2. **Formal-reasoning RL**: Train the resulting prover to generate complete Lean 4 proofs using verification feedback.
  - Results: The 671B model reaches 88.9% on miniF2F-test and solves 47/658 PutnamBench problems in the reported setup; it also introduces the 325-problem ProverBench.

- **Olympiad-level formal mathematical reasoning with reinforcement learning** [[Nature'25](https://doi.org/10.1038/s41586-025-09833-y)]
  - **LLM training**: Train the language-model-based prover through supervised learning and AlphaZero-style reinforcement learning, including test-time adaptation; separately fine-tune Gemini for statement autoformalization.
  - Background: Human proof corpora are limited, and standard inference-time search cannot adapt model parameters to an exceptionally hard new problem.
  - Key problem & insight: Train through large-scale interaction with Lean and continue learning on related problem variants at inference time.
  - Proposed method — AlphaProof with three components:
    1. **Autoformalized curriculum**: Convert large collections of informal problems into formal training statements.
    2. **AlphaZero-inspired RL**: Learn proof-search behavior from machine-checked success over millions of formal problems.
    3. **Test-time RL**: Generate and learn from many related variants of the target problem to obtain problem-specific adaptations.
  - Results: Solves three of five non-geometry IMO 2024 problems; combined with AlphaGeometry 2, the system achieves silver-medal-equivalent performance using multi-day computation. This was not an ordinary timed, fully automatic natural-language competition entry.

- Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2 [[JMLR'25](https://www.jmlr.org/papers/v26/25-1654.html)] [[arXiv'25](https://arxiv.org/abs/2502.03544)]
  - **LLM training**: Train Gemini-based language models on synthetic geometry proofs, including fine-tuning a pretrained math-specialized Gemini model.
  - Background: General-purpose theorem proving is difficult, but Euclidean geometry admits strong domain-specific symbolic deduction and large-scale synthetic data.
  - Key problem & insight: Let a neural model suggest auxiliary constructions while a specialized symbolic engine derives and checks geometric consequences.
  - Proposed method — AlphaGeometry2 with three components:
    1. **Expanded geometry language**: Cover moving objects, non-constructive statements, and linear relations among angles, ratios, and distances.
    2. **Neural construction proposals**: Use a Gemini-based model and synthetic geometry data to propose steps beyond the symbolic engine's direct reach.
    3. **Symbolic deduction and shared search**: Check consequences with the geometry engine and share discoveries across search trees.
  - Results: Solves 84% of all IMO geometry problems from 2000–2024 versus AlphaGeometry's 54%; language coverage increases from 66% to 88%. The checker and language are geometry-specific, not a general Lean pipeline.

<a id="math-agent"></a>

### Agent

- **Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics** [[ICML'26](https://icml.cc/virtual/2026/poster/66755)] [[arXiv'26](https://arxiv.org/abs/2601.14027)]
  - Background: Specialized proving pipelines often hard-code orchestration around a particular trained prover and benchmark.
  - Key problem & insight: Treat formal mathematics as a tool-using coding task that a general agent can manage through a structured Lean interface.
  - Proposed method — Numina-Lean-Agent with two components:
    1. **General coding agent**: Use Claude Code to plan proofs, edit files, retrieve library facts, and coordinate auxiliary reasoning.
    2. **Numina-Lean-MCP**: Expose Lean interactions and proof feedback as tools, enabling the agent to revise its plan and proof artifacts autonomously.
  - Results: With Claude Opus 4.5, solves all twelve Putnam 2025 problems in the reported setup; additionally assists mathematicians in formalizing the Brascamp–Lieb theorem. The latter is explicitly a human–AI collaboration.
