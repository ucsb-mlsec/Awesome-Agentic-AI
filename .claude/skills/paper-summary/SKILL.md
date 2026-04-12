---
name: paper-summary
description: Write structured research paper summaries in a background → problem → method → results format. Use when the user asks to summarize a paper, add a paper entry to a reading list/awesome list, or explain a research method. Triggers on phrases like "summarize this paper", "add this paper to <file>", "write up <paper name>", or when given an arxiv/conference URL with instructions to integrate it into notes.
---

# Paper Summary Skill

Write compact, technically substantive paper entries that a reader can skim in ~30 seconds and still walk away with (a) what existed before, (b) what was broken, (c) how the method fixes it, and (d) whether it worked.

## When to use

- User asks to summarize a paper, add it to a markdown reading list, or explain a research technique
- User provides a paper URL (arXiv, USENIX, OpenReview, etc.) with instructions like "add this to X"
- User asks you to "explain the key ideas of X" for note-taking purposes

## Format

Produce four blocks in this order. Each block has a fixed role — do not reorder or merge them. Use a markdown bullet hierarchy (paper title at top level, subsections as nested bullets).

```
- <Paper Title> [[Venue'YY/MM](url)]
  - Background: <existing techniques and their limitations>
  - Key problem & insight: <what's broken and the core insight that fixes it>
  - Proposed method — <name> with N components:
    1. **<Component 1 name>**: <one-line idea>. <high-level mechanism in 1-3 lines>
    2. **<Component 2 name>**: ...
    3. **<Component 3 name>**: ...
  - Results: <headline numbers, vs. which baselines>
```

## Block rules

### 1. Background / existing techniques

- Name the prior art explicitly (give acronyms if the field uses them: PDC/PIC, GRPO, RoPE, etc.)
- State each one's limitation in **one clause**, not a paragraph
- If there are two competing approaches, contrast them (e.g., "X is fast but rigid; Y is flexible but expensive")
- Skip generic motivation ("LLMs are important..."). Start from the technical baseline

### 2. Key problem & insight

- One sentence identifying the specific failure mode the paper targets
- One sentence stating the core insight (the "aha") that makes the method possible
- This is the block where a reader decides whether to keep reading — make it concrete

### 3. Proposed method

- Lead with the method name and a count of its components ("CacheSlide with three core components")
- For each component:
  - **Bold the component name** (use the paper's actual name, not a paraphrase — grep the paper if needed)
  - Start with a one-line high-level idea (what it does, why it exists)
  - Follow with the mechanism in 1-3 lines: the key variables, the decision rule, or the equation
  - When the mechanism has discrete steps (e.g., pretraining/inference phases, top-k selection + blending), use a sub-bullet list — but keep each sub-bullet ≤1 line
- Do NOT describe every algorithm line. Pick the parts that distinguish the method from the baseline
- If the paper has equations that are central (e.g., a loss function, a novel advantage formulation, a position encoding rule), include them with $\LaTeX$ — but only the ones a reader needs to understand the idea

### 4. Results

- Report the headline numbers with units (latency X×, accuracy +N points, throughput Y×)
- Name the baselines the paper beats (e.g., "vs. vLLM prefix caching", "vs. DAPO", "over GRPO")
- Skip ablations unless the ablation itself is the main contribution

## Style rules

- **Explain the insight.** Don't directly copy text from the paper or combining multiple sentences, understand and explain the insight of the paper.
- **Be concise, not terse.** A reader should be able to understand the mechanism, not just memorize its name. But cut any sentence that only restates what a well-named component already implies.
- **Use the paper's actual terminology.** If the paper calls it "WCA (Weighted Correction Attention)", do not paraphrase it as "selective token correction." Open the paper and check.
- **Avoid marketing language.** No "novel", "cutting-edge", "state-of-the-art" unless quoting a benchmark result. No "seamlessly", "elegantly", "robustly".
- **Prefer verbs over nouns.** "Ranks tokens by deviation and blends the top-k" > "performs a ranking-based blending mechanism".
- **Equations inline, not blocks.** Use `$...$` for short formulas. Only use `$$...$$` if the formula is longer than a line and central to the method.
- **No emojis.** Don't add them unless the surrounding file already uses them.

## Length guidance

- Background + Key problem: 2-4 bullets total
- Proposed method: 1 bullet per component, each with an idea line + ≤3 mechanism sub-bullets
- Results: 1 bullet
- Total target: 8-15 lines for a typical systems/ML paper. Go longer only if the method has >3 components or equations are essential.

## Process

1. **Fetch the paper.** Use WebFetch on the arXiv HTML or the PDF. If WebFetch returns binary PDF metadata, try WebFetch on the HTML mirror (`arxiv.org/html/<id>`) or download via curl and use Read with `pages:`
2. **Find the real component names.** Skim the method section and the algorithm pseudocode. Do not invent names.
3. **Identify the baseline being beaten.** This tells you what goes in the Background block.
4. **Draft all four blocks.** Then re-read and cut anything that's restating the obvious.
5. **Check against the file's existing style.** If the target markdown file uses a different bullet convention, match it.

## Anti-patterns

- **Listing every contribution in the abstract.** Papers oversell. Pick the 2-3 that actually matter.
- **Copy-pasting the abstract's framing.** Abstracts are written for acceptance; summaries are written for recall. Re-frame.
- **"The authors propose..."** Drop it. Start with the technique name.
- **Deep-diving one component while ignoring the others.** Balance coverage across the components — readers need the whole shape of the method.
- **Claiming results without numbers.** "Outperforms baselines" is useless. "3.1-4.3× lower latency vs. vLLM" is useful.

## Example (reference)

```
- CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving [[FAST'26](https://www.usenix.org/system/files/fast26-liu-yang.pdf)]
  - Background:
    - PDC (Position-Dependent Caching): KV tied to absolute positions, reuse only on exact prefix matches
    - PIC (Position-Independent Caching): strips position encoding, reusable anywhere but loses attention fidelity
    - RoPE: high positional sensitivity — any shift invalidates cached keys
    - CoPE: content-gated position encoding, less sensitive to shifts
  - Key problem & insight: agent prompts have reusable segments that maintain consistent *relative* ordering despite absolute shifts (RPDC pattern). PDC/PIC don't exploit this; CoPE can, if positions are locked to a learned template rather than recomputed from live context
  - Proposed method — CacheSlide with three components:
    1. **CCPE (Chunked Contextual Position Encoding)**: pretrain a template $e^*$ of the most frequent CoPE encoding per task; at inference, reuse chunks get positions from $e^*[i]$ (pinned), recompute chunks get live CoPE
    2. **WCA (Weighted Correction Attention)**: token-level gate on top of CCPE — rank tokens in a reuse chunk by $d_i = \|K^{\text{new}}_i - K^{\text{cache}}_i\|$, top-k (~5-17%) get blended $K_i \leftarrow \alpha K^{\text{new}}_i + (1-\alpha) K^{\text{cache}}_i$, rest use cache as-is, applied every $\tau$ layers
    3. **SLIDE (KV cache manager)**: make WCA's I/O pattern SSD-friendly — relocate updated tokens to fresh pages (sequential writes), spill clean pages first, reclaim scratch pages during decode
  - Results: 3.1-4.3× latency reduction, 3.5-5.8× throughput improvement over state-of-the-art baselines
```

This example shows all four blocks at the target length — use it as a template, not a script.
