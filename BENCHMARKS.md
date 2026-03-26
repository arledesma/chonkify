# chonkify Benchmark Snapshot vs LLMLingua and LLMLingua2

This handoff packages the current non-PDF release evidence for `chonkify`.

## Suite A: General `txt/md` Compression (`20` cases)

| Method | fact_recall_mean | exact_success_rate | budget_ok_rate | mean_budget_overrun_tokens | weighted token savings |
|---|---:|---:|---:|---:|---:|
| `chonkify` | `0.8833` | `0.6500` | `1.0000` | `0.00` | `15.02%` |
| `LLMLingua` | `1.0000` | `1.0000` | `0.0000` | `26.80` | `-77.40%` |
| `LLMLingua2` | `0.8667` | `0.6500` | `0.3500` | `4.45` | `0.00%` |

Interpretation: `LLMLingua` v1 can keep more raw facts on the very smallest texts only by violating the requested token budget and, on aggregate, expanding the input. `chonkify` is the budget-valid line on this corridor.

## Suite B: Fact-Heavy Quant Research + Reasoning Traces (`22` cases)

| Method | fact_recall_mean | exact_success_rate | budget_ok_rate | mean_budget_overrun_tokens | weighted token savings |
|---|---:|---:|---:|---:|---:|
| `chonkify` | `0.5606` | `0.2727` | `1.0000` | `0.00` | `78.40%` |
| `LLMLingua` | `0.1061` | `0.0000` | `0.2727` | `30.86` | `70.41%` |
| `LLMLingua2` | `0.1212` | `0.0000` | `0.1364` | `54.55` | `66.10%` |

Interpretation: on fact-heavy corpora the current `chonkify` release is both smaller and higher quality than both `LLMLingua` variants.

## Combined Token Savings

| Method | source tokens | compressed tokens | weighted token savings |
|---|---:|---:|---:|
| `chonkify` | `12802` | `3175` | `75.20%` |
| `LLMLingua` | `12802` | `4743` | `62.95%` |
| `LLMLingua2` | `12802` | `4767` | `62.76%` |

## Positioning

These two non-PDF benchmark corridors are the headline public evidence for this handoff and reflect the current `0.3.0` release line under hard budget constraints.
