"""Standard public-API demo for the installed chonkify wheel.

Run this file after installing the bundled wheel in the same folder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Sequence

from chonkify import (
    DEFAULT_LOCAL_MODEL_NAME,
    CompressionRequest,
    Document,
    build_azure_embedding_provider_from_env,
    build_local_embedding_provider,
    build_openai_compatible_embedding_provider_from_env,
    build_openai_embedding_provider_from_env,
    compress_documents,
)

DEFAULT_BACKEND = "azure"
DEFAULT_QUERY = "launch readiness, cost controls, and operational risks"


def _default_documents() -> list[Document]:
    """Return sample documents that force a real compression workflow."""

    return [
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


def _read_documents(paths: Sequence[str]) -> list[Document]:
    """Load user-supplied UTF-8 text or markdown files into public Document objects."""

    documents: list[Document] = []
    for index, raw_path in enumerate(paths, start=1):
        path = Path(raw_path).expanduser().resolve()
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                source_id=f"input-{index:03d}",
                source_name=path.name,
                text=text,
                origin_path=path,
                metadata={"demo_input": True},
            )
        )
    return documents


def _build_provider(args: argparse.Namespace):
    """Resolve the requested embedding backend through chonkify's public builders."""

    if args.backend == "azure":
        return build_azure_embedding_provider_from_env(
            endpoint=args.azure_endpoint,
            api_key=args.azure_api_key,
            api_version=args.azure_api_version,
            deployment=args.azure_embedding_deployment,
            model_label=args.azure_embedding_model,
        )
    if args.backend == "openai":
        return build_openai_embedding_provider_from_env(
            api_key=args.openai_api_key,
            model=args.openai_embedding_model,
            base_url=args.openai_base_url,
            send_dimensions_parameter=not args.openai_omit_dimensions_parameter,
        )
    if args.backend == "openai-compatible":
        return build_openai_compatible_embedding_provider_from_env(
            api_key=args.openai_api_key,
            model=args.openai_embedding_model,
            base_url=args.openai_base_url,
            send_dimensions_parameter=not args.openai_omit_dimensions_parameter,
        )
    if args.backend == "local":
        return build_local_embedding_provider(
            model_name=args.local_model_name,
            device=args.local_device,
            batch_size=args.local_batch_size,
            cache_folder=args.local_cache_folder,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def _summary_payload(result) -> dict[str, object]:
    """Return the compact KPI view printed by the demo."""

    return {
        "strategy": result.strategy,
        "provider_name": result.provider_name,
        "documents": [document.describe() for document in result.documents],
        "original_tokens": result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "compression_factor": round(result.compression_factor, 4),
        "token_reduction_pct": round(result.token_reduction_pct, 2),
        "units_considered": result.units_considered,
        "units_selected": len(result.selected_units),
        "selected_units_preview": [
            {
                "rank": unit.selection_rank,
                "source_name": unit.source_name,
                "token_count": unit.token_count,
                "objective_score": round(unit.objective_score, 6),
            }
            for unit in result.selected_units[:5]
        ],
        "final_output_truncated": result.final_output_truncated,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the demo CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run a standard chonkify workflow through the public Python API.",
        epilog=(
            "If no input paths are provided, demo.py uses built-in sample documents.\n"
            "Azure backend reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,\n"
            "AZURE_OPENAI_API_VERSION, and CHONKIFY_AZURE_EMBEDDING_DEPLOYMENT.\n"
            "OpenAI backend reads OPENAI_API_KEY and optionally CHONKIFY_OPENAI_EMBEDDING_MODEL.\n"
            "OpenAI-compatible also needs CHONKIFY_OPENAI_BASE_URL.\n"
            "Local backend uses sentence-transformers and requires a 768-dim model."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Optional UTF-8 text or markdown files for the demo. "
            "If omitted, built-in sample documents are used."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("azure", "openai", "openai-compatible", "local"),
        default=os.environ.get("CHONKIFY_DEMO_BACKEND", DEFAULT_BACKEND),
        help="Embedding backend to drive the demo workflow.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=120,
        help="Maximum tokens in the compressed output.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Optional user note carried in request metadata.",
    )
    parser.add_argument(
        "--max-unit-tokens",
        type=int,
        default=126,
        help="Maximum semantic span size passed into CompressionRequest.",
    )
    parser.add_argument(
        "--lambda-relevance",
        type=float,
        default=0.75,
        help="CPC/MMR relevance weight in [0, 1].",
    )
    parser.add_argument(
        "--encoding-name",
        default="o200k_base",
        help="tiktoken encoding used for budgeting.",
    )
    parser.add_argument(
        "--local-model-name",
        default=os.environ.get("CHONKIFY_DEMO_LOCAL_MODEL_NAME", DEFAULT_LOCAL_MODEL_NAME),
        help="SentenceTransformer model for the local backend.",
    )
    parser.add_argument(
        "--local-device",
        default=os.environ.get("CHONKIFY_DEMO_LOCAL_DEVICE", "cpu"),
        help="Local embedding device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=32,
        help="Batch size for the local embedding backend.",
    )
    parser.add_argument(
        "--local-cache-folder",
        default=os.environ.get("CHONKIFY_DEMO_LOCAL_CACHE_FOLDER"),
        help="Optional local cache folder for sentence-transformers weights.",
    )
    parser.add_argument(
        "--openai-omit-dimensions-parameter",
        action="store_true",
        help="Do not send the dimensions parameter to OpenAI-compatible APIs that reject it.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Optional override for AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--azure-api-key",
        default=None,
        help="Optional override for AZURE_OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--azure-api-version",
        default=None,
        help="Optional override for AZURE_OPENAI_API_VERSION.",
    )
    parser.add_argument(
        "--azure-embedding-deployment",
        default=None,
        help="Optional override for CHONKIFY_AZURE_EMBEDDING_DEPLOYMENT.",
    )
    parser.add_argument(
        "--azure-embedding-model",
        default=None,
        help="Optional human-readable Azure embedding model label.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional override for OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional override for CHONKIFY_OPENAI_BASE_URL or OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--openai-embedding-model",
        default=None,
        help="Optional override for CHONKIFY_OPENAI_EMBEDDING_MODEL.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standard chonkify workflow and print both text and KPIs."""

    if importlib.util.find_spec("dotenv") is not None:
        from dotenv import load_dotenv
        load_dotenv()

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    documents = _read_documents(args.inputs) if args.inputs else _default_documents()
    request = CompressionRequest(
        target_tokens=args.target_tokens,
        query=args.query,
        max_unit_tokens=args.max_unit_tokens,
        lambda_relevance=args.lambda_relevance,
        encoding_name=args.encoding_name,
    )
    provider = _build_provider(args)
    result = compress_documents(documents, request, provider)

    print("=== Compressed Text ===")
    print(result.text)
    print()
    print("=== Summary ===")
    print(json.dumps(_summary_payload(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
