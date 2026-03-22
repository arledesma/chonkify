"""Command-line interface for chonkify."""

from __future__ import annotations

import argparse
from typing import Sequence

from chonkify.backends import (
    DEFAULT_LOCAL_MODEL_NAME,
    build_azure_embedding_provider_from_env,
    build_local_embedding_provider,
    build_openai_compatible_embedding_provider_from_env,
    build_openai_embedding_provider_from_env,
)
from chonkify.engine import compress_documents
from chonkify.io import load_documents, write_json_output, write_text_output
from chonkify.telemetry import configure_logging, log_event
from chonkify.types import CompressionRequest, EmbeddingProvider


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="chonkify",
        description="Semantic document compaction with Azure, OpenAI, openai-compatible, or local embeddings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress = subparsers.add_parser(
        "compress",
        help="Compress one or more documents under a token budget.",
    )
    compress.add_argument(
        "inputs",
        nargs="+",
        help="Input paths or '-' for stdin.",
    )
    compress.add_argument(
        "--target-tokens",
        type=int,
        required=True,
        help="Maximum tokens in the compressed output.",
    )
    compress.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional user note stored in metadata; the packaged CPC/MMR winner path itself is queryless.",
    )
    compress.add_argument(
        "--input-format",
        choices=("auto", "text", "markdown", "pdf"),
        default="auto",
        help="Force the input format instead of using suffix-based auto-detection.",
    )
    compress.add_argument(
        "--output",
        type=str,
        default="-",
        help="Destination text file or '-' for stdout.",
    )
    compress.add_argument(
        "--metadata-out",
        type=str,
        default=None,
        help="Optional path for a JSON metadata sidecar.",
    )
    compress.add_argument(
        "--encoding-name",
        type=str,
        default="o200k_base",
        help="tiktoken encoding name used for budgeting.",
    )
    compress.add_argument(
        "--max-unit-tokens",
        type=int,
        default=126,
        help="Maximum semantic span tokens; default matches the CPC/MMR benchmark winner corridor.",
    )
    compress.add_argument(
        "--lambda-relevance",
        type=float,
        default=0.75,
        help="CPC/MMR relevance weight in [0,1]; default matches the benchmark winner.",
    )
    compress.add_argument(
        "--embedding-backend",
        choices=("azure", "openai", "openai-compatible", "local"),
        default="azure",
        help="Embedding runtime to drive the protected CPC/MMR winner core.",
    )
    compress.add_argument(
        "--azure-endpoint",
        type=str,
        default=None,
        help="Override AZURE_OPENAI_ENDPOINT.",
    )
    compress.add_argument(
        "--azure-api-key",
        type=str,
        default=None,
        help="Override AZURE_OPENAI_API_KEY.",
    )
    compress.add_argument(
        "--azure-api-version",
        type=str,
        default=None,
        help="Override AZURE_OPENAI_API_VERSION.",
    )
    compress.add_argument(
        "--azure-embedding-deployment",
        type=str,
        default=None,
        help="Override CHONKIFY_AZURE_EMBEDDING_DEPLOYMENT.",
    )
    compress.add_argument(
        "--azure-embedding-model",
        type=str,
        default=None,
        help="Optional human-readable embedding model label for metadata.",
    )
    compress.add_argument(
        "--azure-max-batch-size",
        type=int,
        default=32,
        help="Maximum embedding batch size sent per Azure request.",
    )
    compress.add_argument(
        "--azure-timeout-seconds",
        type=float,
        default=30.0,
        help="Request timeout passed to the Azure OpenAI client.",
    )
    compress.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Override OPENAI_API_KEY for OpenAI or openai-compatible backends.",
    )
    compress.add_argument(
        "--openai-base-url",
        type=str,
        default=None,
        help="Optional override for OPENAI_BASE_URL / CHONKIFY_OPENAI_BASE_URL.",
    )
    compress.add_argument(
        "--openai-embedding-model",
        type=str,
        default=None,
        help="Override CHONKIFY_OPENAI_EMBEDDING_MODEL.",
    )
    compress.add_argument(
        "--openai-max-batch-size",
        type=int,
        default=32,
        help="Maximum embedding batch size for OpenAI or openai-compatible requests.",
    )
    compress.add_argument(
        "--openai-timeout-seconds",
        type=float,
        default=30.0,
        help="Request timeout for OpenAI or openai-compatible clients.",
    )
    compress.add_argument(
        "--openai-omit-dimensions-parameter",
        action="store_true",
        help="Do not send the dimensions parameter for OpenAI-compatible servers that reject it.",
    )
    compress.add_argument(
        "--local-model-name",
        type=str,
        default=DEFAULT_LOCAL_MODEL_NAME,
        help="SentenceTransformer model used for local embeddings; it must emit 768 dimensions.",
    )
    compress.add_argument(
        "--local-device",
        type=str,
        default="cpu",
        help="Local device for SentenceTransformer embeddings, for example cpu, cuda, cuda:0, or mps.",
    )
    compress.add_argument(
        "--local-batch-size",
        type=int,
        default=32,
        help="Batch size for local SentenceTransformer embedding calls.",
    )
    compress.add_argument(
        "--local-cache-folder",
        type=str,
        default=None,
        help="Optional cache folder for local SentenceTransformer weights.",
    )
    compress.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    compress.add_argument(
        "--json-logs",
        action="store_true",
        help="Render stderr logs as JSON.",
    )
    return parser


def build_embedding_provider_from_args(args: argparse.Namespace) -> EmbeddingProvider:
    """Construct the runtime embedding provider from CLI arguments."""

    if args.embedding_backend == "azure":
        return build_azure_embedding_provider_from_env(
            endpoint=args.azure_endpoint,
            api_key=args.azure_api_key,
            api_version=args.azure_api_version,
            deployment=args.azure_embedding_deployment,
            model_label=args.azure_embedding_model,
            max_batch_size=args.azure_max_batch_size,
            timeout_seconds=args.azure_timeout_seconds,
        )
    if args.embedding_backend == "openai":
        return build_openai_embedding_provider_from_env(
            api_key=args.openai_api_key,
            model=args.openai_embedding_model,
            base_url=args.openai_base_url,
            max_batch_size=args.openai_max_batch_size,
            timeout_seconds=args.openai_timeout_seconds,
            send_dimensions_parameter=not args.openai_omit_dimensions_parameter,
        )
    if args.embedding_backend == "openai-compatible":
        return build_openai_compatible_embedding_provider_from_env(
            api_key=args.openai_api_key,
            model=args.openai_embedding_model,
            base_url=args.openai_base_url,
            max_batch_size=args.openai_max_batch_size,
            timeout_seconds=args.openai_timeout_seconds,
            send_dimensions_parameter=not args.openai_omit_dimensions_parameter,
        )
    if args.embedding_backend == "local":
        return build_local_embedding_provider(
            model_name=args.local_model_name,
            device=args.local_device,
            batch_size=args.local_batch_size,
            cache_folder=args.local_cache_folder,
        )
    raise ValueError(f"Unsupported embedding backend: {args.embedding_backend}")


def _handle_compress(args: argparse.Namespace) -> int:
    logger = configure_logging(verbose=args.verbose, json_logs=args.json_logs)
    provider = build_embedding_provider_from_args(args)
    documents = load_documents(args.inputs, input_format=args.input_format)
    request = CompressionRequest(
        target_tokens=args.target_tokens,
        query=args.query,
        max_unit_tokens=args.max_unit_tokens,
        lambda_relevance=args.lambda_relevance,
        encoding_name=args.encoding_name,
    )

    log_event(
        logger,
        "compression_started",
        input_count=len(documents),
        target_tokens=request.target_tokens,
        provider=provider.name,
    )
    result = compress_documents(documents=documents, request=request, provider=provider)
    write_text_output(args.output, result.text)
    if args.metadata_out:
        write_json_output(args.metadata_out, result.to_dict())
    log_event(
        logger,
        "compression_completed",
        strategy=result.strategy,
        original_tokens=result.original_tokens,
        compressed_tokens=result.compressed_tokens,
        compression_factor=result.compression_factor,
        units_selected=len(result.selected_units),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Programmatic CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "compress":
            return _handle_compress(args)
    except Exception as exc:
        logger = configure_logging(verbose=getattr(args, "verbose", False), json_logs=getattr(args, "json_logs", False))
        log_event(logger, "compression_failed", error=str(exc))
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


def run() -> None:
    """Console script entrypoint."""

    raise SystemExit(main())
