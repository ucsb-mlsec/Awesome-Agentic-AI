# Formal Verification and AI

## Table of Contents

- [Code](#code)
  - [Benchmark](#code-benchmark): [Spec](#code-benchmark-spec) / [Proof](#code-benchmark-proof) / [Impl](#code-benchmark-impl) / [E2E](#code-benchmark-e2e) / [Counterexample](#code-benchmark-counterexample)
  - [Training](#code-training)
  - [Agent](#code-agent)
- [Math](#math)
  - [Benchmark](#math-benchmark)
  - [Training](#math-training)
  - [Agent](#math-agent)

## Code

<a id="code-benchmark"></a>

### Benchmark

<a id="code-benchmark-spec"></a>

#### Spec — generate what should hold

- **VERINA** — **VERINA: Benchmarking Verifiable Code Generation** [[ICLR'26](https://arxiv.org/abs/2505.23135)] — function scope; 189 programming tasks in lean, with separate SpecGen, CodeGen, ProofGen, and combined settings.
  - each task has a problem description, code implementation, specifications (pre-condition and post-condition), a proof (optional), and comprehensive test cases (input-output pairs, including both positive and negative)
  - specgen: give the model description and lean function signature, ask model to generate the speficication. When verifying the result, a model will try to prove the preconditions are equivalent and postconditions are equivalent given the precondition, if the model cannot prove, then use test cases.
    - Good precondition should reject illegal inputs and accept legal inputs, good postconditions should reject (legal input+wrong output) and accept (legal input+correct output).
  - CodeGen: Give the model a description and Lean function signature, optionally with the reference specification, generate the implementation and check its outputs against test cases. 
  - ProofGen: Give the model the description, function signature, implementation, and specification; generate a correctness proof and check it with Lean.
  - Combined:
      - CodeGen＋ProofGen: Give the LLM description＋Lean function signature＋reference specification; generate code and its correctness proof.
      - SpecGen＋ProofGen: Give the LLM description＋Lean function signature＋implementation; generate specification and a proof that the implementation satisfies it.
      - CodeGen＋SpecGen＋ProofGen: Give the LLM description＋Lean function signature; generate code and specification, then provide the reference specification to generate the code’s correctness proof.
    
- TLA+-Bench — **TLA+-Bench: An Execution-Grounded Benchmark and Dataset for Natural-Language to TLA+ Specification Generation** [[arXiv'26](https://arxiv.org/abs/2607.23425)] — focus on model checking and temporal logic; 403 TLC-runnable gold specifications and 897 parse-only silver specifications.
  - **LLM input**: A natural-language system description; a configuration including invariant names, specification names etc.
  - **LLM output**: A TLA+ specification of states, transitions, and properties.
  - **Verification**: Parse with SANY, bind to the reference configuration, and run TLC over the configured finite state space. Some heuristics to prevent the model from generating trivial properties.

  
- Verus-SpecBench / Verus-SpecGym — *Verus-SpecGym: An Agentic Environment for Evaluating Specification Autoformalization* [[arXiv'26](https://arxiv.org/abs/2605.26457)] — function scope; 581 Codeforces-derived specification tasks.
  - **LLM input**: A problem statement and Verus specification scaffold, with access to the verifier, shell, and filesystem.
  - **LLM output**: Input assumptions and required output behavior encoded as a Verus specification.
  - **Verification**: Execute specifications through Verus `exec_spec` and compare their acceptance of input/output cases with official tests and adversarial Codeforces hacks. This tests both omitted requirements and overrestrictive specifications; it is not a universal intent-equivalence proof.

- DafnyCOMP — *Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification* [[ICLR'26](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c04d37be05ba74419d2d5705972a9d64-Abstract-Conference.html)] — multiple interacting functions and their data dependencies.
  - **LLM input**: A Dafny program with its executable logic retained and contracts/supporting annotations to reconstruct across function boundaries.
  - **LLM output**: Preconditions, postconditions, and proof annotations strong enough for callers and callees to compose.
  - **Verification**: Verify the complete composed program with Dafny; separately successful local proofs are insufficient when caller obligations fail. Acceptance establishes correctness relative to the generated contracts, not their faithfulness to an unstated intent.

- OSVBench — *OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification* [[AAAI'26](https://ojs.aaai.org/index.php/AAAI/article/view/40437)] — operating-system state and syscall behavior.
  - **LLM input**: A syscall description, the permitted state-transition programming model, verification assumptions, and kernel implementation context that may contain injected bugs.
  - **LLM output**: An executable state-machine specification for the syscall.
  - **Verification**: Run the Hyperkernel verifier and compare the generated specification's verdicts with reference-specification verdicts across kernel variants. A specification that merely agrees with buggy code is insufficient.


<a id="code-benchmark-proof"></a>

#### Proof — keep the implementation and target property fixed

- **DafnyBench** / **VerusBench** — annotation completion in Dafny / Rust-Verus; standalone-program scope.
  Papers: **DafnyBench: A Benchmark for Formal Software Verification** [[TMLR'25](https://openreview.net/forum?id=yBgTVWccIx)] [[arXiv'24](https://arxiv.org/abs/2406.08467)]; **AutoVerus: Automated Proof Generation for Rust Code** [[OOPSLA'25](https://doi.org/10.1145/3763174)] [[arXiv'24](https://arxiv.org/abs/2409.13082)]. DafnyBench contains 1,326 programs; the original VerusBench contains 150 proof tasks, with evaluation subsets varying across papers.
  - **LLM input**: An implementation and its target contracts with selected proof annotations removed; repair attempts may also receive verifier diagnostics.
  - **LLM output**: Missing invariants, assertions, ghost code, and supporting proof annotations.
  - **Verification**: Run Dafny or Verus and require the target obligations to pass while preserving the executable code and target contracts. The two benchmarks share a task type; their programs and solver behavior are not interchangeable.

- miniCodeProps — *miniCodeProps: a Minimal Benchmark for Proving Code Properties* [[arXiv'24](https://arxiv.org/abs/2406.11915)] — small, self-contained Lean programs and properties.
  - **LLM input**: A fixed program, its definitions, and a formal statement about its behavior.
  - **LLM output**: Lean tactics or a complete proof of the supplied property, with code and property unchanged.
  - **Verification**: Check the completed theorem in Lean; induction or auxiliary lemmas may be necessary even for short programs.

- RVBench / VeriSoftBench / Selene — proof completion with repository and systems context, grouped across Verus / Lean / Isabelle.
  Papers: *Towards Repository-Level Program Verification with Large Language Models* [[LMPL'25](https://arxiv.org/abs/2509.25197)]; *VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean* [[arXiv'26](https://arxiv.org/abs/2602.18307)]; *Selene: Pioneering Automated Proof in Software Verification* [[ACL'24](https://aclanthology.org/2024.acl-long.98/)]. RVBench spans four Verus projects; VeriSoftBench contains 500 obligations from 23 Lean repositories; Selene draws lemmas from the seL4 verification development.
  - **LLM input**: A proof hole or target lemma, fixed definitions and specifications, and project context. VeriSoftBench separately supplies curated dependencies or the full repository.
  - **LLM output**: Verus annotations, Lean proofs, or Isabelle proof commands that complete the target obligation using project definitions and lemmas.
  - **Verification**: Check the replacement with the corresponding verifier in its project environment. A solved repository-dependent lemma counts as a local success, not completion of the entire repository; context selection and project diversity also differ across these datasets.

<a id="code-benchmark-impl"></a>

#### Impl — generate code under a supplied specification

- FVAPPS — *Proving the Coding Interview: A Benchmark for Formally Verified Code Generation* [[LLM4Code@ICSE'25](https://github.com/quinn-dougherty/fvapps)] [[arXiv'25](https://arxiv.org/abs/2502.05714)] — coding-problem scope; 4,715 samples, including 1,083 curated samples.
  - **LLM input**: A Lean 4 coding task with implementation/proof holes and supplied correctness requirements.
  - **LLM output**: The missing implementation and proofs that it meets those requirements.
  - **Verification**: Check the completed artifacts in Lean without unfinished proofs or added untrusted assumptions. The guarantee concerns the supplied statements; dataset size does not imply that every specification is equally faithful to the original problem.

- Vericoding benchmark / AlgoVeri — fixed-specification synthesis across Dafny, Verus, and Lean.
  Papers: *A benchmark for vericoding: formally verified program synthesis* [[arXiv'25](https://arxiv.org/abs/2509.22908)] [[Dafny@POPL'26](https://popl26.sigplan.org/details/dafny-2026-papers/13/A-benchmark-for-vericoding-formally-verified-program-synthesis)]; *AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms* [[arXiv'26](https://arxiv.org/abs/2602.09464)]. The former aggregates multiple task sources, including FVAPPS and VERINA; the latter aligns classical algorithms across languages.
  - **LLM input**: A formal functional specification with the implementation removed, optionally accompanied by a natural-language description.
  - **LLM output**: An implementation plus the annotations or proof scripts required by the target language.
  - **Verification**: Run the relevant checker against the fixed specification. AlgoVeri supports comparisons on aligned algorithm tasks; aggregate results from the broader vericoding collection involve different source distributions.

- **Vero** — **Vero: Can AI Agents Build Formally Verified Software Repositories?** [[arXiv'26](https://arxiv.org/abs/2608.13522)] — 43 multi-module Lean repositories, 743 scored APIs, and 2,705 specifications.
  - **LLM input**: A repository scaffold with fixed APIs, definitions, and formal specifications; proof-only mode additionally supplies implementations.
  - **LLM output**: Implementations and proofs across modules, or proofs alone in proof-only mode.
  - **Verification**: Rebuild under the benchmark's fixed interface and axiom restrictions; distinguish individual specification success from completion of every obligation in a repository. This is repository-scale code-and-proof synthesis from fixed specifications, not natural-language-to-specification generation.


<a id="code-benchmark-e2e"></a>

#### E2E — generate specifications, implementations, and proofs

The **VERINA** combined settings are the reused starting point here; see [its input/output and separate evaluators](#code-benchmark-spec). The following retain different evaluation boundaries.

- **CLEVER** — **CLEVER: A Curated Benchmark for Formally Verified Code Generation** [[NeurIPS'25 — Datasets and Benchmarks](https://arxiv.org/abs/2505.13938)] — 161 HumanEval-derived function tasks.
  - **LLM input**: A natural-language programming task and Lean scaffold; the evaluator retains a human-written reference specification.
  - **LLM output**: A formal specification, an implementation, and certification proofs for specification equivalence and implementation correctness.
  - **Verification**: Check specification equivalence against the held-out reference and implementation correctness against the reference requirement in Lean. Code satisfying a weak, self-generated specification alone does not pass; the reference still defines the intended meaning.

- VerifyThisBench — *VerifyThisBench: Generating Code, Specifications, and Proofs All at Once* [[arXiv'25](https://arxiv.org/abs/2505.19271)] — 41 verification-competition challenges represented as 154 tasks across seven tools, plus 580 completion tasks in VerifyThisBenchXS.
  - **LLM input**: An informal challenge description and target verification language; XS variants supply partial artifacts with code, specifications, or invariants removed.
  - **LLM output**: Specifications, implementations, and proof annotations/scripts, or the missing artifact in a completion task.
  - **Verification**: Compile and verify with the designated tool, feeding diagnostics back for repair. Verifier acceptance checks the encoded requirement; it does not by itself certify a faithful translation of the informal challenge.

<a id="code-benchmark-counterexample"></a>

#### Counterexample — produce checkable negative evidence

These distinguish the artifact being challenged (Spec or Impl) and whether the LLM discovers a violation or receives a counterexample as input. Where no broadly reused standalone benchmark is established here, representative evaluation suites remain explicitly labeled.

- Vero formal audit — Spec / Impl; an audit mode of the [repository benchmark above](#code-benchmark-impl), not an additional dataset [[arXiv'26](https://arxiv.org/abs/2608.13522)].
  - **LLM input**: A fixed specification or specification family and the reference implementation.
  - **LLM output**: A Lean proof that a specification is unsatisfiable, specifications conflict, or the reference implementation violates a specification.
  - **Verification**: Check the negative certificate in Lean under the audit restrictions. A code/specification mismatch needs adjudication to identify which artifact is wrong; an unsuccessful audit establishes neither consistency nor correctness.

- Neuroforger — Impl violation discovery; *Neuroforger: certified violation witnesses for smart contracts verification via LLMs* [[arXiv'26](https://arxiv.org/abs/2605.31389)] — a method evaluation suite of 110 smart-contract verification tasks.
  - **LLM input**: A Solidity contract and a GATE specification describing the shape of an admissible violation witness.
  - **LLM output**: Concrete contracts, transaction sequences, and values instantiating that witness.
  - **Verification**: Check the instantiation against the specification and execute it with Forge; the prototype's type validation includes manual checking. Successful execution certifies the particular violation; failed search is inconclusive.

- **VeriExploit** — Impl counterexample-to-reproduction; **VeriExploit: Automatic Bug Reproduction in Smart Contracts via LLMs and Formal Methods** [[ASE'25](https://pure.manchester.ac.uk/ws/portalfiles/portal/1632624289/ASE2025.pdf)] — a method evaluation suite for executable smart-contract bug reproduction.
  - **LLM input**: A vulnerable contract and an already available formal counterexample.
  - **LLM output**: An attacker/reproduction contract and concrete interaction steps that realize the counterexample.
  - **Verification**: Compile and run the reproduction and check that it triggers the target violation, using formal/execution feedback for repair. This evaluates realization of a known counterexample, not independent vulnerability discovery.

- CryptoFormalEval — protocol attack discovery; *CryptoFormalEval: Integrating LLMs and Formal Verification for Automated Cryptographic Protocol Vulnerability Detection* [[arXiv'24](https://arxiv.org/abs/2411.13627)] — protocol-level models and message traces.
  - **LLM input**: An informal cryptographic protocol description, target security properties, and access to Tamarin feedback.
  - **LLM output**: A formal protocol model, encoded properties, and an attack explanation supported by tool analysis; the formal tool supplies the attack trace.
  - **Verification**: Run Tamarin and validate the attack against the intended protocol. A violation of a mistranslated model is insufficient evidence of a flaw in the original protocol.

<a id="code-training"></a>

### Training


#### LLM-based invariant inference

- **SmartInv: Multimodal Learning for Smart Contract Invariant Inference** [[IEEE S&P'24](https://www.cs.columbia.edu/~junfeng/papers/smartinv/)]
  - Background: Pattern-based smart-contract analyzers miss business-logic bugs when the intended transaction behavior is not explicit in the code.
  - Key problem & insight: Infer properties from both code and natural-language transaction context, then check where the implementation violates them.
  - Proposed method — SmartInv with two components:
    1. **Tier of Thought (ToT)**: Fine-tune and prompt a foundation model to reason across source code and contextual descriptions before generating invariants.
    2. **Invariant checking**: Validate generated properties against the contract and use violations to localize suspicious behavior.
  - Results: Reports 119 previously unknown bugs; of eight sampled reports sent to developers, six were fixed and five confirmed as high severity. Bug counts are not equivalent to a completeness guarantee for the inferred properties.

#### Verifier-supervised synthesis and self-improvement

- **Automated Proof Generation for Rust Code via Self-Evolution** [[ICLR'25](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b2e20d7402c9985eae4ba924c65370a8-Abstract-Conference.html)]
  - Background: Open models have little exposure to Verus proofs, and human-written Rust proof corpora are too small for ordinary large-scale fine-tuning.
  - Key problem & insight: A verifier labels both successful proofs and failed attempts, supporting generation training and debugging training together.
  - Proposed method — SAFE with two components:
    1. **Self-evolving synthesis**: Generate proofs, retain verifier-accepted examples, fine-tune, and repeat to expand the training corpus.
    2. **Self-debugging**: Train on incorrect proofs plus verifier feedback so the model learns to repair its own failures.
  - Results: Achieves 52.52% proof-generation accuracy on the authors' expert-built benchmark versus 14.39% for GPT-4o; this comparison concerns that benchmark and configuration.

- **Towards Neural Synthesis for SMT-Assisted Proof-Oriented Programming** [[ICSE'25](https://www.microsoft.com/en-us/research/publication/towards-neural-synthesis-for-smt-assisted-proof-oriented-programming/)] [[arXiv'24](https://arxiv.org/abs/2405.01787)]
  - Background: F* mixes programs and proofs and delegates many obligations to SMT, but still requires experts to construct typed definitions and select useful premises.
  - Key problem & insight: Treat each top-level definition as a type-directed synthesis problem with a reproducible F* checker.
  - Proposed method — F* synthesis with two components:
    1. **FStarDataSet**: Extract specifications, definitions, context, and checker support from production-related F* projects.
    2. **Fine-tuning and premise retrieval**: Train smaller code models and augment prompts using type-based retrieval of relevant definitions.
  - Results: The extended corpus contains approximately 940k lines and 54k definitions; on its cross-project evaluation, fine-tuned StarCoder reaches 58.13% verify@10 versus 41.63% for GPT-3.5.

- **Re:Form -- Reducing Human Annotations in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny** [[arXiv'25](https://arxiv.org/abs/2507.16331)]
  - Background: RL for verified programming is limited by scarce annotated demonstrations and the difficulty of producing initially valid formal-language programs.
  - Key problem & insight: Automatically construct Dafny training tasks, bootstrap syntax and proof competence with SFT, then refine using verifier feedback.
  - Proposed method — Re:Form with two stages:
    1. **Data curation and SFT**: Build formal-program examples and teach models to generate Dafny implementations and annotations.
    2. **Regularized RL**: Use formal verification feedback while retaining regularization to improve generalization beyond the supervised corpus.
  - Results: On the paper's 300-task out-of-distribution DafnyComp subset, the 14B RL model reaches 14.0% Pass@1 versus 8.3% for its SFT counterpart and 2.7% for the Claude data-generator baseline; the study also demonstrates initial verifiable-code competence with a 0.5B model.

- **SpecRL: Reinforcement Learning with Test-Based Completeness Rewards for Formal Specification Synthesis** [[arXiv'26](https://arxiv.org/abs/2604.05820)]
  - Background: A verifier can accept `ensures true`; rewarding verification success alone encourages weak specifications that say little about the implementation.
  - Key problem & insight: Add negative input-output examples that distinguish useful specifications from vacuous ones.
  - Proposed method — SpecRL with two components:
    1. **Spectests**: Construct implementation-impossible input-output pairs that an underspecified contract may still allow.
    2. **Completeness reward**: For verifier-accepted candidates, reward the fraction of spectests rejected by the generated specification.
  - Results: On out-of-distribution DafnyComp-Spec, the 7B model improves verification success by 49.96% and empirical completeness by 26.46% relative to SFT. Spectests improve measured completeness; they do not establish logical completeness.

- **Formal Disco: Scalable Open-Ended Generation of Formally Verified Programs** [[arXiv'26](https://arxiv.org/abs/2607.04631)]
  - Background: Self-training is constrained by a small seed corpus and can keep regenerating similar easy programs.
  - Key problem & insight: Separate the creation, repair, and extension of verified programs, then train for both success and diversity.
  - Proposed method — Formal Disco with three worker roles:
    1. **Initiators**: Use repository READMEs and documentation to propose new verification tasks and programs.
    2. **Fixers**: Repair candidates using compiler and verifier diagnostics.
    3. **Extenders**: Expand already verified programs; collect trajectories for distillation and iterative SFT with an entropy-maximization objective.
  - Results: Produces datasets for Dafny, Verus, and Frama-C; the final Qwen generation's largest verified examples exceed the largest Claude seed examples by 22–62% in lines of code across those languages, alongside downstream verification evaluations.

- **Propose, Solve, Verify: Self-Play Through Formal Verification** [[ICML'26](https://icml.cc/virtual/2026/poster/63571)]
  - Background: Expert iteration on a fixed problem set eventually runs out of new solvable examples, while test-only self-play can reinforce incorrect solutions.
  - Key problem & insight: Couple a difficulty-aware problem proposer to a solver, using formal verification as the acceptance signal.
  - Proposed method — Propose, Solve, Verify (PSV) with two learned roles:
    1. **Proposer**: Generate new formal programming tasks calibrated to the current solver's ability.
    2. **Solver**: Attempt the tasks, retain verified solutions, and improve through expert iteration before the next proposal round.
  - Results: PSV-Verus improves Pass@1 by up to 9.6x over the paper's inference-only and expert-iteration baselines across three benchmarks; gains depend on both verification and difficulty-aware proposal.

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
  - Background: A smart-contract verifier may report a counterexample without producing an attacker contract or an executable transaction sequence.
  - Key problem & insight: Treat the formal counterexample as a construction guide for an executable bug reproduction.
  - Proposed method — VeriExploit with two components:
    1. **Reproduction synthesis**: Give the LLM the vulnerable contract and counterexample to generate an attacker/reproduction contract and interaction steps.
    2. **Validation and refinement**: Check whether the generated artifact re-triggers the target bug, then repair failed attempts using formal and execution feedback.
  - Results: Achieves 85.60% reproduction success on the authors' benchmark. The task starts from a supplied vulnerable contract and counterexample rather than discovering every bug from scratch.

- Neuroforger: certified violation witnesses for smart contracts verification via LLMs [[arXiv'26/05](https://arxiv.org/abs/2605.31389)]
  - Background: LLM verification judgments lack checkable evidence; natural-language properties can also be ambiguous.
  - Key problem & insight: Express a violation as a partially specified executable test, then require a valid instantiation rather than trust the model's explanation.
  - Proposed method — Neuroforger with three components:
    1. **GATE**: Extend Solidity specifications with abstract contracts, transactions, and variables representing unknown parts of a violation witness.
    2. **Concretization**: Use GPT-5 to fill these abstract entities, revising candidates from checking feedback.
    3. **Type checking and concrete execution**: Check that substitutions respect the specification and run the witness with Forge. The prototype requires manual validation of type checking.
  - Results: Finds witnesses for 48 of 53 violating tasks and reports none for 57 nonviolating tasks; headline metrics exclude one impractically long witness case. Failure to find a witness returns an inconclusive `true?`, not a proof of safety.

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

PutnamBench covers formalized undergraduate competition problems. Subsequent use includes [Goedel-Prover-V2](https://arxiv.org/abs/2508.03613).

- PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition [[NeurIPS'24 — Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2024/file/1582eaf9e0cf349e1e5a6ee453100aa1-Paper-Datasets_and_Benchmarks_Track.pdf)]
  - **LLM input**: A formalized Putnam competition problem with the required definitions in Lean 4, Isabelle, or Coq.
  - **LLM output**: A proof of the supplied formal statement.
  - **Verification**: Check the proof in the corresponding assistant without changing the theorem or introducing unproved assumptions; use the task set of the stated benchmark version.

<a id="math-training"></a>

### Training

- **DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** [[ICLR'25](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b3b55c366d641c07180c40e4f978f311-Abstract-Conference.html)]
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

- DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition [[arXiv'25](https://arxiv.org/abs/2504.21801)]
  - Background: Whole-proof RL receives little useful signal on problems whose complete proofs are initially beyond the model.
  - Key problem & insight: Recursively solve simpler subgoals and assemble them into training examples that connect informal plans to formal proofs.
  - Proposed method — DeepSeek-Prover-V2 with two stages:
    1. **Recursive cold-start synthesis**: Use DeepSeek-V3 to decompose problems, solve subgoals, and combine checked subproofs with step-by-step reasoning.
    2. **Formal-reasoning RL**: Train the resulting prover to generate complete Lean 4 proofs using verification feedback.
  - Results: The 671B model reaches 88.9% on miniF2F-test and solves 47/658 PutnamBench problems in the reported setup; it also introduces the 325-problem ProverBench.

- **Olympiad-level formal mathematical reasoning with reinforcement learning** [[Nature'25](https://doi.org/10.1038/s41586-025-09833-y)]
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

- **Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics** [[ICML'26](https://icml.cc/virtual/2026/poster/66755)] [[arXiv'26](https://arxiv.org/abs/2601.14027)]
  - Background: Specialized proving pipelines often hard-code orchestration around a particular trained prover and benchmark.
  - Key problem & insight: Treat formal mathematics as a tool-using coding task that a general agent can manage through a structured Lean interface.
  - Proposed method — Numina-Lean-Agent with two components:
    1. **General coding agent**: Use Claude Code to plan proofs, edit files, retrieve library facts, and coordinate auxiliary reasoning.
    2. **Numina-Lean-MCP**: Expose Lean interactions and proof feedback as tools, enabling the agent to revise its plan and proof artifacts autonomously.
  - Results: With Claude Opus 4.5, solves all twelve Putnam 2025 problems in the reported setup; additionally assists mathematicians in formalizing the Brascamp–Lieb theorem. The latter is explicitly a human–AI collaboration.
