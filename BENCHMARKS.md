# chonkify Benchmark Context vs LLMLingua and LLMLingua2

This file summarizes the most recent multidocument comparison from the internal document-compression benchmark suite. It is intended to be self-contained inside the minimal `chonkify` handoff folder.

## Suite Scope

- Documents: `5`
- Budgets: `1500`, `1000`
- Comparison metric for the percentage deltas below: mean `composite_info_recovery`
- Best internal baseline on this suite: `cpc_mmr/key_sentence_mmr`

## Composite Recovery Comparison

| Budget | Internal best | LLMLingua | Internal delta vs LLMLingua | LLMLingua2 | Internal delta vs LLMLingua2 |
|---|---:|---:|---:|---:|---:|
| `1500` | `0.4302` | `0.2713` | `+58.61%` | `0.1559` | `+175.99%` |
| `1000` | `0.3312` | `0.1804` | `+83.54%` | `0.1211` | `+173.49%` |
| mean across both budgets | `0.3807` | `0.2258` | `+68.57%` | `0.1385` | `+174.90%` |

## Win Pattern

- Budget `1500`: the internal `cpc_mmr/key_sentence_mmr` method wins `4/5` documents.
- Budget `1000`: the internal `cpc_mmr/key_sentence_mmr` method wins `3/5` documents.
- Across the `10` document-budget cells in this suite, internal methods win `9/10`.

## Important Caveat

The current benchmark recovery scorer used for this comparison still relies on proxy metrics for sentence, heading, and numeric-fact recovery. These numbers are therefore useful as relative comparison evidence, but they are not a fully ground-truth semantic measure.

## Positioning

`chonkify` now packages the actual internal winner path label from that comparison: `cpc_mmr/key_sentence_mmr`. The packaged runtime uses the same CPC/MMR selection logic and benchmark-style key-sentence preparation chain, while still keeping the benchmark scoring harness itself outside the product package.
