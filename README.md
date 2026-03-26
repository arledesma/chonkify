# chonkify

**Extractive document compression that actually preserves what matters.**

chonkify compresses long documents into tight, information-dense context for RAG pipelines, agent memory, and any workflow where token budget matters as much as factual recovery. This release focuses on strong factual recovery under hard token budgets across general `txt`/`md` and fact-heavy document workloads.

Today, the clearest validated fit is content-dense non-PDF text: quantitative research digests, structured engineering notes, and reasoning traces where downstream models need exact facts more than fluent paraphrase. It remains a general-purpose document compressor, but this is the workload family where the current release is strongest.

By [Thomas "Thom" Heinrich](mailto:th@thomheinrich.de) · [chonkyDB.com](https://chonkydb.com)

---

## Why chonkify

Most compression tools optimize for token reduction. chonkify optimizes for **information recovery** — the compressed output retains the facts, structure, and reasoning that downstream models actually need.

On the current release corridors against Microsoft's LLMLingua family:

| Suite | chonkify | LLMLingua | LLMLingua2 |
|---|---:|---:|---:|
| general `txt/md` (`20` cases), `fact_recall_mean` | **0.8833** | 1.0000 | 0.8667 |
| general `txt/md`, `budget_ok_rate` | **1.0000** | 0.0000 | 0.3500 |
| fact-heavy quant/reasoning (`22` cases), `fact_recall_mean` | **0.5606** | 0.1061 | 0.1212 |
| fact-heavy quant/reasoning, `budget_ok_rate` | **1.0000** | 0.2727 | 0.1364 |

Across both suites combined, `chonkify` currently saves **75.20%** of source tokens, versus **62.95%** for `LLMLingua` and **62.76%** for `LLMLingua2`. Full methodology and caveats are in [BENCHMARKS.md](BENCHMARKS.md).

## How It Works

chonkify builds source-faithful document units, scores them through a strict `768`-dimensional embedding interface, and returns a compact output that respects your token budget. Performance-sensitive implementation ships as compiled extension modules.

## Install

This refreshed handoff includes the current native `cp311` wheel matrix for the supported desktop/server targets:

```bash
# Linux x86_64
pip install ./chonkify-0.3.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl

# Windows amd64
py -3.11 -m pip install .\chonkify-0.3.0-cp311-cp311-win_amd64.whl

# macOS arm64
python3.11 -m pip install ./chonkify-0.3.0-cp311-cp311-macosx_11_0_arm64.whl

# macOS x86_64
python3.11 -m pip install ./chonkify-0.3.0-cp311-cp311-macosx_10_9_x86_64.whl
```

These four wheels were produced by the native GitHub Actions matrix run `23559149680`, and the Linux manylinux artifact was revalidated afterwards with a fresh-venv `ci/wheel_smoke.py` install smoke.

For local CPU/GPU embeddings (no API calls), also install:

```bash
pip install sentence-transformers
```

Or use the optional extra: `pip install chonkify[local]`

## Quick Start

### CLI

```bash
chonkify compress ./paper.pdf \
  --target-tokens 1200 \
  --output ./paper_compressed.txt \
  --metadata-out ./paper_meta.json
```

Multiple documents in one pass:

```bash
chonkify compress ./brief.md ./appendix.pdf \
  --target-tokens 1400 \
  --output ./bundle.txt
```

Pipe from stdin:

```bash
cat ./notes.txt | chonkify compress - --target-tokens 900 --output -
```

### Python API

```python
from chonkify import compress_documents

# With additional control over embedding providers:
from chonkify import (
    LocalEmbeddingConfig,
    LocalSentenceTransformerEmbeddingProvider,
    OpenAIEmbeddingConfig,
    OpenAIEmbeddingProvider,
    compress_documents,
)
```

Minimal example:

```python
from chonkify import compress_documents

result = compress_documents(
    ["Quarterly revenue rose 18%. Operating margin expanded to 27%. Guidance remains unchanged."],
    target_tokens=24,
)

print(result.compressed_text)
print(result.compressed_tokens)
```

## Embedding Backends

### Azure OpenAI (default)

```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="<secret>"
export AZURE_OPENAI_API_VERSION="2024-10-21"
export CHONKIFY_AZURE_EMBEDDING_DEPLOYMENT="<deployment-name>"
```

### OpenAI

```bash
export OPENAI_API_KEY="<secret>"
export CHONKIFY_OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
```

```bash
chonkify compress ./paper.pdf --embedding-backend openai --target-tokens 1200
```

### OpenAI-Compatible Endpoints

For providers like Together, Fireworks, or self-hosted APIs:

```bash
export OPENAI_API_KEY="<key>"
export CHONKIFY_OPENAI_BASE_URL="https://<provider>/v1"
export CHONKIFY_OPENAI_EMBEDDING_MODEL="<model-id>"
```

```bash
chonkify compress ./paper.pdf --embedding-backend openai-compatible --target-tokens 1200
```

If your endpoint rejects the `dimensions` parameter, add `--openai-omit-dimensions-parameter`. chonkify still validates 768-dimensional output.

### Local (SentenceTransformers)

Fully offline after first model download. Default model: `sentence-transformers/all-mpnet-base-v2`.

```bash
chonkify compress ./paper.pdf \
  --embedding-backend local \
  --local-device cuda \
  --target-tokens 1200
```

Device options: `cpu`, `cuda`, `cuda:0`, `mps`.

Validated with `sentence-transformers 5.1.0` and `torch 2.8.0+cu128` on NVIDIA RTX 3090. Cold-cache run: ~13s. Warm-cache run: ~6s. Model footprint: ~419 MB. With `HF_HUB_OFFLINE=1`, the local backend runs fully air-gapped once cached.

## Output Metadata

The optional `--metadata-out` JSON includes:

- Original and compressed token counts
- Compression factor and token reduction percentage
- Selected source blocks with source IDs and ranks
- Embedding provider used

If you pass `--query`, it is preserved in metadata for provenance tracking.

## License

chonkify is proprietary software. The current release is licensed for **evaluation, testing, and review only** — not for production use. See [LICENSE.md](LICENSE.md) for full terms.

For commercial licensing, production access, or integration partnerships:
**th@chonkydb.com**

## Benchmark Details

See [BENCHMARKS.md](BENCHMARKS.md) for the current release benchmark methodology and numbers.
