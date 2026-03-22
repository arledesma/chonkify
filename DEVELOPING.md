# Developing chonkify

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and virtual environments.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- For GPU embeddings: an NVIDIA GPU with the appropriate CUDA toolkit installed

## Environment Setup

### 1. Choose your embedding backend

chonkify supports cloud-based and local embedding backends. Pick the sync command that matches your setup:

```bash
# Cloud only (Azure OpenAI / OpenAI) — no torch, no GPU
uv sync

# Local CPU embeddings (installs sentence-transformers + CPU torch)
uv sync --extra local

# Local GPU embeddings — pick the CUDA version that matches your driver
uv sync --extra local-cu124   # CUDA 12.4
uv sync --extra local-cu128   # CUDA 12.8
uv sync --extra local-cu130   # CUDA 13.0
```

The `local-cu*` extras are mutually exclusive with each other and with `local`. uv enforces this via a conflicts table in `pyproject.toml` and routes torch to the correct PyTorch wheel index for each CUDA version.

To check which CUDA version your driver supports:

```bash
nvidia-smi
```

The "CUDA Version" in the top-right corner is your maximum supported version. Pick the highest `local-cu*` extra that does not exceed it.

### 2. Configure credentials

The `.env` file is loaded by `demo.py` via `python-dotenv` (a dev dependency). The CLI reads environment variables directly.

**For local GPU development** (no API keys needed):

```bash
cp .env.local .env
```

This sets `CHONKIFY_DEMO_BACKEND=local` and `CHONKIFY_DEMO_LOCAL_DEVICE=cuda` — ready to go with any `local-cu*` extra.

**For cloud backends**, copy the full example and fill in your values:

```bash
cp .env.example .env
```

See `.env.example` for all available variables. The backends require:

- **Azure OpenAI** — `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `CHONKIFY_AZURE_EMBEDDING_DEPLOYMENT` (must output 768 dimensions)
- **OpenAI** — `OPENAI_API_KEY`, optionally `CHONKIFY_OPENAI_EMBEDDING_MODEL` (defaults to `text-embedding-3-large`)
- **OpenAI-compatible** — `OPENAI_API_KEY`, `CHONKIFY_OPENAI_BASE_URL`, `CHONKIFY_OPENAI_EMBEDDING_MODEL`

### 3. Verify the setup

```bash
# Run the demo with built-in sample documents
uv run python demo.py --target-tokens 120

# Run via the CLI entry point
uv run chonkify compress README.md --target-tokens 500 --embedding-backend local --local-device cuda
```

## Project Structure

```
pyproject.toml              # Package metadata, dependencies, uv config
src/chonkify/
  __init__.py               # Public API surface
  __main__.py               # python -m chonkify
  backends.py               # OpenAI / local embedding providers
  cli.py                    # CLI (chonkify compress ...)
  config.py                 # Azure embedding config + env resolution
  engine.py                 # CPC/MMR compression engine
  io.py                     # Document loading and output
  telemetry.py              # Structured logging
  types.py                  # Data contracts (Document, CompressionRequest, etc.)
demo.py                     # Standalone demo script
```

## How the extras work

The `pyproject.toml` defines four mutually exclusive optional dependency groups for local embeddings:

| Extra | What it installs | Torch source |
|-|-|-|
| `local` | sentence-transformers + CPU torch | PyPI |
| `local-cu124` | sentence-transformers + CUDA 12.4 torch | pytorch.org/whl/cu124 |
| `local-cu128` | sentence-transformers + CUDA 12.8 torch | pytorch.org/whl/cu128 |
| `local-cu130` | sentence-transformers + CUDA 13.0 torch | pytorch.org/whl/cu130 |

The `[tool.uv.sources]` section maps `torch` to the correct PyTorch wheel index based on which extra is active. The `[tool.uv]` conflicts table prevents installing more than one variant simultaneously.

Without any extra, torch is not installed and only cloud embedding backends (Azure OpenAI, OpenAI, OpenAI-compatible) are available.

## Running tests

```bash
uv run python -m pytest
```

## Building a wheel

```bash
uv build
```

This produces a pure-Python wheel in `dist/` that works on any platform.
