# chonkify

**Extractive document compression that actually preserves what matters.**

chonkify compresses long documents into tight, information-dense context — built for RAG pipelines, agent memory, and anywhere you need to fit more signal into fewer tokens. It uses a proprietary algorithm that consistently outperforms existing compression methods.

By [Thomas "Thom" Heinrich](mailto:th@thomheinrich.de) · [chonkyDB.com](https://chonkydb.com)

![chonkify-logo](chonkify-logo.png)

---

## Why chonkify

Most compression tools optimize for token reduction. chonkify optimizes for **information recovery** — the compressed output retains the facts, structure, and reasoning that downstream models actually need.

In head-to-head multidocument benchmarks against Microsoft's LLMLingua family:

| Budget | chonkify | LLMLingua | LLMLingua2 |
|---|---:|---:|---:|
| 1500 tokens | **0.4302** | 0.2713 | 0.1559 |
| 1000 tokens | **0.3312** | 0.1804 | 0.1211 |

That's **+69% composite information recovery** vs LLMLingua and **+175%** vs LLMLingua2 on average across both budgets, winning 9 out of 10 document-budget cells in the test suite. Full methodology in [BENCHMARKS.md](BENCHMARKS.md).

## How It Works

chonkify embeds document content, scores passages by information density and diversity, and extracts the highest-value subset under your token budget. The selection core ships as compiled extension modules — the benchmarks speak for themselves.

## Install

chonkify ships as a compiled, platform-specific Python 3.11 wheel.

```bash
# Linux x86_64
pip install ./chonkify-0.2.2-cp311-cp311-manylinux*_x86_64.whl

# macOS Apple Silicon
pip install ./chonkify-0.2.2-cp311-cp311-macosx*_arm64.whl

# macOS Intel
pip install ./chonkify-0.2.2-cp311-cp311-macosx*_x86_64.whl

# Windows
pip install .\chonkify-0.2.2-cp311-cp311-win_amd64.whl
```

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
- Embedding provider and selection strategy used

If you pass `--query`, it is preserved in metadata for provenance tracking.

## License

chonkify is proprietary software. The current release is licensed for **evaluation, testing, and review only** — not for production use. See [LICENSE.md](LICENSE.md) for full terms.

For commercial licensing, production access, or integration partnerships:
**th@chonkydb.com**

## Benchmark Details

See [BENCHMARKS.md] for the full multidocument comparison methodology and per-document results.
