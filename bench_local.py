"""Benchmark local embedding backends across CUDA variants."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sentence_transformers
import torch

from chonkify import (
    CompressionRequest,
    Document,
    build_local_embedding_provider,
    compress_documents,
)


DOCS = [
    Document(
        source_id="launch-brief",
        source_name="launch_brief.md",
        text=(
            "# Launch Brief\n\n"
            "The product launch remains on schedule for early April if the team finishes the "
            "customer migration runbook and the support escalation matrix this week.\n\n"
            "The highest operational risk is the manual rollback path for enterprise tenants. "
            "The current runbook depends on two engineers and a sequence of shell commands that "
            "has not been rehearsed end to end.\n\n"
            "Cost control improved after the team reduced duplicate embedding requests, but the "
            "support budget still assumes a conservative incident rate during the first two weeks "
            "of rollout.\n\n"
            "The legal review is complete, security approved the current release branch, and the "
            "remaining blocker is a final customer-facing FAQ that explains expected downtime and "
            "escalation windows."
        ),
    ),
    Document(
        source_id="field-feedback",
        source_name="field_feedback.md",
        text=(
            "# Field Feedback\n\n"
            "Pilot customers care most about a predictable rollout schedule, fast rollback if data "
            "quality drops, and clear ownership for post-launch support.\n\n"
            "Three accounts explicitly asked for proof that usage-based costs will stay within the "
            "budget envelope agreed during procurement, especially once their larger document sets "
            "move onto the new workflow.\n\n"
            "The account team reported that training quality matters less than concise operational "
            "documentation. Customers prefer one launch checklist, one escalation contact chain, "
            "and one short explanation of the major risks.\n\n"
            "No customer asked for extra features before launch. The repeated theme was readiness, "
            "rollback confidence, and cost visibility."
        ),
    ),
]

REQUEST = CompressionRequest(target_tokens=120)
RUNS = 5


@dataclass(frozen=True, slots=True)
class RunResult:
    run: int
    elapsed_s: float
    original_tokens: int
    compressed_tokens: int
    compression_factor: float
    token_reduction_pct: float
    units_considered: int
    units_selected: int
    truncated: bool


@dataclass(slots=True)
class BenchmarkResult:
    python: str
    torch: str
    sentence_transformers: str
    cuda_available: bool
    gpu: str
    gpu_memory_gb: float
    device: str
    runs: list[RunResult] = field(default_factory=list)


def run_benchmark(device: str) -> BenchmarkResult:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    gpu_mem = (getattr(props, "total_memory", 0) or getattr(props, "total_mem", 0)) / (1024**3) if props else 0

    benchmark = BenchmarkResult(
        python=platform.python_version(),
        torch=torch.__version__,
        sentence_transformers=sentence_transformers.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu=gpu_name,
        gpu_memory_gb=round(gpu_mem, 1),
        device=device,
    )

    for i in range(RUNS):
        provider = build_local_embedding_provider(device=device)
        t0 = time.perf_counter()
        result = compress_documents(DOCS, REQUEST, provider)
        elapsed = time.perf_counter() - t0

        benchmark.runs.append(
            RunResult(
                run=i + 1,
                elapsed_s=round(elapsed, 4),
                original_tokens=result.original_tokens,
                compressed_tokens=result.compressed_tokens,
                compression_factor=round(result.compression_factor, 4),
                token_reduction_pct=round(result.token_reduction_pct, 2),
                units_considered=result.units_considered,
                units_selected=len(result.selected_units),
                truncated=result.final_output_truncated,
            )
        )
        print(f"  Run {i + 1}: {elapsed:.4f}s  ({result.compressed_tokens} tokens)", file=sys.stderr)

    return benchmark


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking with torch {torch.__version__} on {device}", file=sys.stderr)
    result = run_benchmark(device)

    epoch = int(time.time())
    out_dir = Path("benchmarks") / str(epoch)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    print(f"Results written to {out_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
