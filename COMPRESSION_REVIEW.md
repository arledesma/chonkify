# chonkify Compression Quality Review

## Executive Summary

chonkify achieved a **3.4x compression ratio** (70.6% token reduction) on a complex, rule-dense instructional document. The extractive algorithm correctly identified and preserved high-information-density passages — core workflow structure, critical safety constraints, and mandatory behavioral directives all survived selection.

However, the review exposed a **category limitation**: documents where nearly every sentence functions as an independent behavioral rule tolerate far less extractive compression than narrative or analytical documents. The algorithm optimized for information density and diversity as designed, but had no awareness of **functional dependency chains** — sentences that are individually lower-density but operationally critical because other preserved sentences depend on them.

**Key findings:**

- **Structural preservation was strong.** The compressed output maintained coherent workflow ordering and retained the highest-priority directives.
- **Tool usage policy was disproportionately lost.** The algorithm treated behavioral rules around tool use as redundant relative to higher-level instructions. In modern agentic architectures where tool schemas are provided separately via the API's `tools` parameter, the prompt content governing tools is *usage policy* — not definitions. The agent retains full knowledge of what tools exist and their parameters; what it loses is the *when*, *why*, and *in what order* guidance.
- **The compression ceiling depends on prompt architecture.** When tool schemas live outside the prompt, more of the prompt's content is policy rather than definition, and the safe compression range shifts upward. For prompts that embed tool definitions inline, the ceiling is lower.

**Recommendation:** For agentic/instructional prompts, a lower target token budget (closer to 40-50% of original) or a domain-aware preservation mode would produce safer results. For prompts where tool schemas are provided via the API, up to 60% reduction may be achievable. For narrative, analytical, or multi-document summarization workloads — chonkify's primary design target — the 70%+ compression range remains well-suited.

## Review Methodology

This review was conducted by distributing the original and compressed documents to **6 independent AI agents** across 3 model tiers (Opus, Sonnet, Haiku — 2 agents each). Each pair was assigned a complementary analysis task:

- **Executive Summary + SVOR Analysis** (Strengths, Vulnerabilities, Opportunities, Risks)
- **GAP Analysis + Recommendations** (specific functional gaps with severity ratings)

Results were synthesized by cross-referencing findings across all 6 agents and noting agreement levels. A finding required majority agreement (4/6+) to be included in the consolidated analysis.

## Document Characteristics

The test document was a **system prompt for an autonomous AI agent** — a category characterized by:

- **High rule density** — nearly every sentence is a behavioral instruction, not a narrative passage
- **Implicit dependency chains** — later instructions assume earlier definitions are present
- **Safety-critical specificity** — validation rules, tool prerequisites, and workflow ordering are not optional elaboration but functional contracts
- **Low natural redundancy** — unlike reports, papers, or meeting notes, instructional prompts rarely restate the same idea in different ways

These characteristics are adversarial to extractive compression, which assumes some passages carry more information than others. In a rule-dense prompt, the information distribution is nearly flat.

## Tool Definitions vs. Tool Usage Policy

In modern agentic architectures, tool information is split across two channels:

1. **Tool schemas** (via the API `tools` parameter) — define *what* tools exist, their parameters, and types. These are **never in the prompt** and are **unaffected by compression**.
2. **Tool usage policy** (in the prompt) — define *when*, *why*, and *in what order* to call tools, plus behavioral guardrails. These **are** subject to compression.

When evaluating compression quality on agentic prompts, this distinction is critical. Losing a tool's schema means the agent cannot call it at all. Losing a tool's usage policy means the agent can still call it, but may do so at the wrong time, in the wrong order, or without checking domain-specific preconditions. The severity is meaningfully different.

The test document's tool schemas were provided via the API, so the compressed prompt's tool-related losses are policy losses — not capability losses.

## What the Algorithm Did Well

1. **Workflow skeleton preserved.** The multi-step process (retrieve context → analyze/classify → build action plan → submit response) remained intact and in order.
2. **Mandatory directives retained.** The single most critical behavioral constraint — a mandatory final submission step — survived with emphasis.
3. **Validation rules partially kept.** Key guard rails for input validation and precondition checking were preserved, preventing the most common targeting errors.
4. **Decision threshold logic intact.** The framework for when to proceed with actions based on scoring thresholds was retained.
5. **Output formatting templates partially preserved.** Structured output sections survived in fragment form, providing at least a partial scaffold for response formatting.

## What the Algorithm Lost

### Critical (workflow-breaking)

- **Workflow ordering.** The explicit "call this first, then this" sequencing was flattened into fragments. An autonomous agent following the compressed prompt would likely execute steps in the wrong order or skip them.

### High (degrades quality or safety)

- **Tool-call ordering rules.** The prompt specified which tools to call before others. This policy was lost. The agent still knows the tools exist (from the schema) but not the intended call sequence.
- **Tool-call discipline rules.** Cost and context-window optimization directives were entirely absent, risking runaway token consumption.
- **Output template.** A multi-section structured format was reduced to fragments, degrading the quality and auditability of the agent's output.

### Medium (recoverable or partially inferrable)

- **Precondition rules.** Rules governing which input types are eligible for which actions were partially lost. With tool schemas available, the model can often infer basic type requirements, but domain-specific preconditions are not derivable from schemas alone.
- **Platform-specific prohibitions.** Rules preventing certain actions on certain platforms were dropped.
- **Action sequencing logic.** The rationale for ordering response steps was absent, leaving the agent to guess.
- **Scoring methodology.** Modifier examples and the framework for calculating decision scores were stripped.

## Guidance for Practitioners

### Choosing a target token budget

The relationship between compression ratio and information loss is **not linear** — it depends on document type:

| Document Type | Safe Compression Range | Rationale |
| - | - | - |
| Multi-document narrative (reports, papers, transcripts) | 60-80% reduction | High natural redundancy; extractive selection excels |
| Single long-form document (README, spec, article) | 40-60% reduction | Moderate redundancy; section structure helps selection |
| Agentic prompts (tool schemas provided via API) | 30-50% reduction | Policy-only content; definitions live outside the prompt |
| Agentic prompts (tool definitions inline) | 20-40% reduction | Near-zero redundancy; most sentences are load-bearing |
| API contracts, schemas, enums | 0-10% reduction | Every token is structural; compression is destructive |

### When extractive compression works best

- Documents with **repetitive themes** across sections (the algorithm's diversity penalty deduplicates effectively)
- Multi-document bundles where **overlap between sources** is high
- Content where the **consumer tolerates lossy summarization** (e.g., RAG context injection, background reading)

### When to use caution

- **Agentic system prompts** — where omitted rules cause wrong autonomous actions, not just incomplete understanding
- **Compliance/legal documents** — where every clause has independent legal weight
- **Configuration-like documents** — where the format itself carries meaning (enums, schemas, parameter lists)

### Mitigation strategies for rule-dense documents

1. **Reduce the compression target.** Use `--target-tokens` closer to 50-60% of the original rather than 30%.
2. **Split before compressing.** Separate instructional scaffolding (compressible) from behavioral rules (not compressible) and only compress the former.
3. **Post-compression validation.** Diff the output against a checklist of must-have elements before deploying.
4. **Two-stage prompting.** Use the compressed version as a skeleton and inject critical rules at runtime from a separate source.
