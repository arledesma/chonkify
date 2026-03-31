"""Local structured-document runtime for chonkify compaction."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib import error, request

import numpy as np
import tiktoken

from chonkify.types import Document
from chonkify.local_semantic_runtime import (
    LocalSemanticRuntimeConfig,
    load_local_sentence_transformer_model,
    resolve_local_semantic_profile,
    semantic_model_status_snapshot,
    semantic_similarity_matrix,
)

DEFAULT_DOCUMENT_MODEL = "llama3.2:1b"
DEFAULT_DOCUMENT_FALLBACK_MODEL = "qwen2.5:7b-instruct"
DEFAULT_DOCUMENT_SENTENCE_SEGMENTER_MODEL = "sat-12l-sm"
DEFAULT_DOCUMENT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_DOCUMENT_TIMEOUT_SECONDS = 180.0
DEFAULT_DOCUMENT_MAX_INPUT_TOKENS = 2_048
DEFAULT_DOCUMENT_NUM_PREDICT = 4_096
DEFAULT_DOCUMENT_FALLBACK_NUM_PREDICT = 512
DEFAULT_DOCUMENT_GROUNDING_RERANK_PROFILE = "cross_encoder_minilm_l6_local"
DEFAULT_DOCUMENT_GROUNDING_SPAN_TOKENS = 126
DEFAULT_DOCUMENT_SELECTION_UNIT_TOKENS = 96
_LONG_PDF_SELECTION_UNIT_TOKENS = 96
DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT = 160
DEFAULT_DOCUMENT_GLOBAL_SELECTION_NUM_PREDICT = 1_024
DEFAULT_DOCUMENT_GLOBAL_MODE_SAMPLE_LIMIT = 24
DEFAULT_DOCUMENT_SALIENCE_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
_DOCUMENT_SEMANTIC_SALIENCE_LAMBDA = 0.78
_DOCUMENT_CONTRASTIVE_RERANK_LAMBDA = 0.72
_LONG_PDF_ADAPTIVE_MAX_INPUT_TOKENS = 512
_LONG_PDF_ADAPTIVE_NUM_PREDICT = 1_024
_LONG_PDF_TRIGGER_TOKENS = 4_096
_MIN_RETRY_SPLIT_TOKENS = 256
_SELECTION_MODE_CONTENT_POOL = "content_pool"
_SELECTION_MODE_MASK_LOW_SIGNAL = "mask_low_signal"
_SELECTION_MODE_EXTRACTIVE_LONG_PDF = "semantic_extractive_long_pdf"
_SELECTION_MODE_HIGH_RECALL_SALIENT_ONLY = "high_recall_salients_only"
_DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL = "chunk_local_salients"
_DOCUMENT_SALIENCE_MODE_SECTION_KEEP_ALL = "section_keep_all"
_SENTENCE_SEGMENTER_MIN_LENGTH = 24
_LONG_PDF_EXTRACTIVE_UNIT_TOKENS = 48
_LONG_PDF_LEAD_SEED_LIMIT = 12
_LONG_PDF_DOCUMENT_SALIENT_LIMIT = 32
_WINNER_PREPARATION_RECOVERY_BENCHMARK_MODULE = "chonkify_document_ai_recovery_benchmark"


class DocumentStructurerError(RuntimeError):
    """Base error type for the local document-structuring runtime."""


class DocumentStructurerTransportError(DocumentStructurerError):
    """Raised when the local Ollama transport fails."""


class DocumentStructurerResponseError(DocumentStructurerError):
    """Raised when the model response violates the structuring contract."""


class DocumentStructurer(Protocol):
    """Contract for local document-structure providers."""

    name: str

    def structure_documents(
        self,
        documents: Sequence[Document],
        *,
        encoder: tiktoken.Encoding,
    ) -> list["StructuredDocument"]:
        """Return one structured representation per input document."""


@dataclass(frozen=True, slots=True)
class StructuredSection:
    """A model-derived section with faithful extractive sentences."""

    heading: str | None
    sentences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuredDocument:
    """A model-derived structure used by the CPC/MMR winner path."""

    source_document: Document
    rendered_text: str
    sections: tuple[StructuredSection, ...]
    salient_sentences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuredChunkTrace:
    """Trace payload for one model-structured input chunk."""

    chunk_index: int
    chunk_count: int
    model_name: str
    input_characters: int
    input_tokens: int
    fragment_title: str | None
    section_count: int
    sentence_count: int
    salient_count: int
    section_headings: tuple[str | None, ...]
    salient_sentences: tuple[str, ...]
    grounding_candidate_span_count: int
    grounding_sentence_requests: int
    grounding_unique_sentences: int
    grounding_rerank_calls: int
    grounding_cache_hits: int
    grounding_salient_reuse_count: int
    selection_mode: str
    candidate_unit_count: int
    kept_unit_count: int
    dropped_unit_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly trace payload."""

        return {
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "model_name": self.model_name,
            "input_characters": self.input_characters,
            "input_tokens": self.input_tokens,
            "fragment_title": self.fragment_title,
            "section_count": self.section_count,
            "sentence_count": self.sentence_count,
            "salient_count": self.salient_count,
            "section_headings": list(self.section_headings),
            "salient_sentences": list(self.salient_sentences),
            "grounding_candidate_span_count": self.grounding_candidate_span_count,
            "grounding_sentence_requests": self.grounding_sentence_requests,
            "grounding_unique_sentences": self.grounding_unique_sentences,
            "grounding_rerank_calls": self.grounding_rerank_calls,
            "grounding_cache_hits": self.grounding_cache_hits,
            "grounding_salient_reuse_count": self.grounding_salient_reuse_count,
            "selection_mode": self.selection_mode,
            "candidate_unit_count": self.candidate_unit_count,
            "kept_unit_count": self.kept_unit_count,
            "dropped_unit_count": self.dropped_unit_count,
        }


@dataclass(frozen=True, slots=True)
class StructuredDocumentTrace:
    """Trace payload for the structured document passed to the winner selector."""

    source_id: str
    source_name: str
    input_characters: int
    input_tokens: int
    rendered_characters: int
    rendered_tokens: int
    section_count: int
    sentence_count: int
    salient_count: int
    global_salient_candidate_count: int
    global_salient_selected_count: int
    global_salient_model_name: str | None
    sections: tuple[StructuredSection, ...]
    salient_sentences: tuple[str, ...]
    chunk_traces: tuple[StructuredChunkTrace, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly trace payload."""

        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "input_characters": self.input_characters,
            "input_tokens": self.input_tokens,
            "rendered_characters": self.rendered_characters,
            "rendered_tokens": self.rendered_tokens,
            "section_count": self.section_count,
            "sentence_count": self.sentence_count,
            "salient_count": self.salient_count,
            "global_salient_candidate_count": self.global_salient_candidate_count,
            "global_salient_selected_count": self.global_salient_selected_count,
            "global_salient_model_name": self.global_salient_model_name,
            "sections": [
                {
                    "heading": section.heading,
                    "sentences": list(section.sentences),
                }
                for section in self.sections
            ],
            "salient_sentences": list(self.salient_sentences),
            "chunk_traces": [trace.to_dict() for trace in self.chunk_traces],
        }


@dataclass(frozen=True, slots=True)
class LocalDocumentStructurerConfig:
    """Runtime configuration for the local Ollama document structurer."""

    model_name: str = DEFAULT_DOCUMENT_MODEL
    fallback_model_name: str | None = DEFAULT_DOCUMENT_FALLBACK_MODEL
    ollama_base_url: str = DEFAULT_DOCUMENT_BASE_URL
    timeout_seconds: float = DEFAULT_DOCUMENT_TIMEOUT_SECONDS
    max_input_tokens: int = DEFAULT_DOCUMENT_MAX_INPUT_TOKENS
    num_predict: int = DEFAULT_DOCUMENT_NUM_PREDICT
    fallback_num_predict: int = DEFAULT_DOCUMENT_FALLBACK_NUM_PREDICT
    grounding_rerank_profile_id: str = DEFAULT_DOCUMENT_GROUNDING_RERANK_PROFILE
    grounding_span_tokens: int = DEFAULT_DOCUMENT_GROUNDING_SPAN_TOKENS
    salience_embedding_model_name: str = DEFAULT_DOCUMENT_SALIENCE_EMBED_MODEL
    selection_mode: str | None = None


@dataclass(frozen=True, slots=True)
class _StructuredFragment:
    """A single structured chunk emitted by the local model."""

    title: str | None
    sections: tuple[StructuredSection, ...]
    salient_sentences: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CandidateUnit:
    """One exact source-grounded evidence unit exposed to the local model."""

    unit_id: str
    text: str


@dataclass(frozen=True, slots=True)
class _StructuredChunkAttempt:
    """Structured output plus the local model profile that produced it."""

    chunk_text: str
    model_name: str
    fragment: _StructuredFragment
    grounding_candidate_span_count: int
    grounding_sentence_requests: int
    grounding_unique_sentences: int
    grounding_rerank_calls: int
    grounding_cache_hits: int
    grounding_salient_reuse_count: int
    selection_mode: str
    candidate_unit_count: int
    kept_unit_count: int
    dropped_unit_count: int


@dataclass(frozen=True, slots=True)
class _GroundedFragmentResult:
    """Grounded fragment plus per-chunk grounding telemetry."""

    fragment: _StructuredFragment
    candidate_span_count: int
    sentence_requests: int
    unique_sentences: int
    rerank_calls: int
    cache_hits: int
    salient_reuse_count: int


@dataclass(frozen=True, slots=True)
class _GroundingBatchRuntime:
    """Resolved local cross-encoder resources for chunk-grounding batches."""

    model_name: str
    model: Any


@dataclass(frozen=True, slots=True)
class _SemanticSalienceRuntime:
    """Resolved local bi-encoder resources for document salience selection."""

    model_name: str
    model: Any


@dataclass(frozen=True, slots=True)
class _DocumentGlobalSelectionResult:
    """Document-level evidence-pool selection metadata."""

    salient_sentences: tuple[str, ...]
    candidate_count: int
    selected_count: int
    model_name: str | None


@dataclass(frozen=True, slots=True)
class _DocumentSalienceModeDecision:
    """Model-selected document salience regime for long PDFs."""

    salience_mode: str
    model_name: str | None


def _resolve_value(
    explicit: str | None,
    env: Mapping[str, str],
    *keys: str,
) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    for key in keys:
        value = str(env.get(key, "")).strip()
        if value:
            return value
    return ""


def document_structurer_config_from_env(
    *,
    model_name: str | None = None,
    fallback_model_name: str | None = None,
    ollama_base_url: str | None = None,
    timeout_seconds: float | None = None,
    max_input_tokens: int | None = None,
    num_predict: int | None = None,
    fallback_num_predict: int | None = None,
    grounding_rerank_profile_id: str | None = None,
    grounding_span_tokens: int | None = None,
    salience_embedding_model_name: str | None = None,
    selection_mode: str | None = None,
    env: Mapping[str, str] | None = None,
) -> LocalDocumentStructurerConfig:
    """Resolve local document-structurer config from env plus overrides."""

    values = env if env is not None else os.environ
    resolved_model_name = _resolve_value(
        model_name,
        values,
        "CHONKIFY_DOCUMENT_MODEL",
    ) or DEFAULT_DOCUMENT_MODEL
    resolved_fallback_model_name = _resolve_value(
        fallback_model_name,
        values,
        "CHONKIFY_DOCUMENT_FALLBACK_MODEL",
    ) or DEFAULT_DOCUMENT_FALLBACK_MODEL
    resolved_ollama_base_url = _resolve_value(
        ollama_base_url,
        values,
        "CHONKIFY_DOCUMENT_OLLAMA_BASE_URL",
        "OLLAMA_HOST",
    ) or DEFAULT_DOCUMENT_BASE_URL
    resolved_timeout_seconds = float(
        timeout_seconds
        if timeout_seconds is not None
        else values.get("CHONKIFY_DOCUMENT_TIMEOUT_SECONDS", DEFAULT_DOCUMENT_TIMEOUT_SECONDS)
    )
    resolved_max_input_tokens = int(
        max_input_tokens
        if max_input_tokens is not None
        else values.get("CHONKIFY_DOCUMENT_MAX_INPUT_TOKENS", DEFAULT_DOCUMENT_MAX_INPUT_TOKENS)
    )
    resolved_num_predict = int(
        num_predict
        if num_predict is not None
        else values.get("CHONKIFY_DOCUMENT_NUM_PREDICT", DEFAULT_DOCUMENT_NUM_PREDICT)
    )
    resolved_fallback_num_predict = int(
        fallback_num_predict
        if fallback_num_predict is not None
        else values.get("CHONKIFY_DOCUMENT_FALLBACK_NUM_PREDICT", DEFAULT_DOCUMENT_FALLBACK_NUM_PREDICT)
    )
    resolved_grounding_rerank_profile_id = _resolve_value(
        grounding_rerank_profile_id,
        values,
        "CHONKIFY_DOCUMENT_GROUNDING_RERANK_PROFILE",
    ) or DEFAULT_DOCUMENT_GROUNDING_RERANK_PROFILE
    resolved_grounding_span_tokens = int(
        grounding_span_tokens
        if grounding_span_tokens is not None
        else values.get("CHONKIFY_DOCUMENT_GROUNDING_SPAN_TOKENS", DEFAULT_DOCUMENT_GROUNDING_SPAN_TOKENS)
    )
    resolved_salience_embedding_model_name = _resolve_value(
        salience_embedding_model_name,
        values,
        "CHONKIFY_DOCUMENT_SALIENCE_EMBED_MODEL",
    ) or DEFAULT_DOCUMENT_SALIENCE_EMBED_MODEL
    resolved_selection_mode = _resolve_value(
        selection_mode,
        values,
        "CHONKIFY_DOCUMENT_SELECTION_MODE",
    ) or ""
    if not resolved_model_name:
        raise ValueError("document model_name must be non-empty.")
    if not resolved_ollama_base_url:
        raise ValueError("document ollama_base_url must be non-empty.")
    if resolved_timeout_seconds <= 0:
        raise ValueError("document timeout_seconds must be positive.")
    if resolved_max_input_tokens <= 0:
        raise ValueError("document max_input_tokens must be positive.")
    if resolved_num_predict <= 0:
        raise ValueError("document num_predict must be positive.")
    if resolved_fallback_num_predict <= 0:
        raise ValueError("document fallback_num_predict must be positive.")
    if not resolved_grounding_rerank_profile_id:
        raise ValueError("document grounding_rerank_profile_id must be non-empty.")
    if resolved_grounding_span_tokens <= 0:
        raise ValueError("document grounding_span_tokens must be positive.")
    if not resolved_salience_embedding_model_name:
        raise ValueError("document salience_embedding_model_name must be non-empty.")
    if resolved_selection_mode and resolved_selection_mode not in {
        _SELECTION_MODE_CONTENT_POOL,
        _SELECTION_MODE_MASK_LOW_SIGNAL,
    }:
        raise ValueError(
            "document selection_mode must be one of "
            f"{_SELECTION_MODE_CONTENT_POOL!r} or {_SELECTION_MODE_MASK_LOW_SIGNAL!r}."
        )
    return LocalDocumentStructurerConfig(
        model_name=resolved_model_name,
        fallback_model_name=resolved_fallback_model_name or None,
        ollama_base_url=resolved_ollama_base_url.rstrip("/"),
        timeout_seconds=resolved_timeout_seconds,
        max_input_tokens=resolved_max_input_tokens,
        num_predict=resolved_num_predict,
        fallback_num_predict=resolved_fallback_num_predict,
        grounding_rerank_profile_id=resolved_grounding_rerank_profile_id,
        grounding_span_tokens=resolved_grounding_span_tokens,
        salience_embedding_model_name=resolved_salience_embedding_model_name,
        selection_mode=resolved_selection_mode or None,
    )


def _token_count(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text, disallowed_special=()))


def _normalize_line(raw_line: str) -> str:
    return " ".join(str(raw_line or "").replace("\t", " ").split())


def _dedupe_keep_order(values: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return tuple(ordered)


def _normalize_validation_text(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _compact_validation_text(text: str) -> str:
    return "".join(_normalize_validation_text(text).split())


def _alnum_validation_text(text: str) -> str:
    return "".join(char for char in _normalize_validation_text(text) if char.isalnum())


def _chunk_text(
    text: str,
    *,
    encoder: tiktoken.Encoding,
    max_input_tokens: int,
) -> list[str]:
    def _split_oversize_text(raw_text: str) -> list[str]:
        token_ids = encoder.encode(raw_text, disallowed_special=())
        return [
            encoder.decode(token_ids[start : start + max_input_tokens]).strip()
            for start in range(0, len(token_ids), max_input_tokens)
            if encoder.decode(token_ids[start : start + max_input_tokens]).strip()
        ]

    normalized_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    if not normalized_text.strip():
        return []
    if _token_count(normalized_text, encoder) <= max_input_tokens:
        return [normalized_text.strip()]

    chunks: list[str] = []
    current_lines: list[str] = []
    current_tokens = 0
    blank_pending = False
    for raw_line in normalized_text.split("\n"):
        line = _normalize_line(raw_line)
        if not line:
            blank_pending = True
            continue
        line_tokens = _token_count(line, encoder)
        if line_tokens > max_input_tokens:
            chunk = "\n".join(current_lines).strip()
            if chunk:
                chunks.append(chunk)
            chunks.extend(_split_oversize_text(line))
            current_lines = []
            current_tokens = 0
            blank_pending = False
            continue
        prefix = "\n\n" if blank_pending and current_lines else ("\n" if current_lines else "")
        candidate_piece = f"{prefix}{line}"
        piece_tokens = _token_count(candidate_piece, encoder)
        if current_lines and current_tokens + piece_tokens > max_input_tokens:
            chunk = "\n".join(current_lines).strip()
            if chunk:
                chunks.append(chunk)
            current_lines = [line]
            current_tokens = _token_count(line, encoder)
            blank_pending = False
            continue
        if blank_pending and current_lines:
            current_lines.append("")
        current_lines.append(line)
        current_tokens += piece_tokens
        blank_pending = False

    chunk = "\n".join(current_lines).strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _schema_limits(num_predict: int) -> tuple[int, int, int]:
    budget = max(256, int(num_predict))
    section_limit = 1
    sentence_limit = max(8, min(48, budget // 48))
    salient_limit = max(4, min(16, budget // 80))
    return section_limit, sentence_limit, salient_limit


def _selection_schema(
    *,
    unit_ids: Sequence[str],
    num_predict: int,
    selection_mode: str,
) -> dict[str, Any]:
    _section_limit, sentence_limit, salient_limit = _schema_limits(num_predict)
    if selection_mode == _SELECTION_MODE_HIGH_RECALL_SALIENT_ONLY:
        return {
            "type": "object",
            "properties": {
                "document_title": {"type": ["string", "null"]},
                "salient_unit_ids": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": min(int(salient_limit), max(1, len(unit_ids))),
                    "items": {
                        "type": "string",
                        "enum": list(unit_ids),
                    },
                },
            },
            "required": ["document_title", "salient_unit_ids"],
            "additionalProperties": False,
        }
    if selection_mode == _SELECTION_MODE_MASK_LOW_SIGNAL:
        return {
            "type": "object",
            "properties": {
                "document_title": {"type": ["string", "null"]},
                "drop_unit_ids": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": max(1, len(unit_ids)),
                    "items": {
                        "type": "string",
                        "enum": list(unit_ids),
                    },
                },
                "salient_unit_ids": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": min(int(salient_limit), max(1, len(unit_ids))),
                    "items": {
                        "type": "string",
                        "enum": list(unit_ids),
                    },
                },
            },
            "required": ["document_title", "drop_unit_ids", "salient_unit_ids"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "document_title": {"type": ["string", "null"]},
            "content_unit_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": min(int(sentence_limit), max(1, len(unit_ids))),
                "items": {
                    "type": "string",
                    "enum": list(unit_ids),
                },
            },
            "salient_unit_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": min(int(salient_limit), max(1, len(unit_ids))),
                "items": {
                    "type": "string",
                    "enum": list(unit_ids),
                },
            },
        },
        "required": ["document_title", "content_unit_ids", "salient_unit_ids"],
        "additionalProperties": False,
    }


def _global_selection_schema(*, unit_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "document_title": {"type": ["string", "null"]},
            "content_unit_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": min(int(DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT), max(1, len(unit_ids))),
                "items": {
                    "type": "string",
                    "enum": list(unit_ids),
                },
            },
        },
        "required": ["document_title", "content_unit_ids"],
        "additionalProperties": False,
    }


def _document_salience_mode_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "document_title": {"type": ["string", "null"]},
            "salience_mode": {
                "type": "string",
                "enum": [
                    _DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL,
                    _DOCUMENT_SALIENCE_MODE_SECTION_KEEP_ALL,
                ],
            },
        },
        "required": ["document_title", "salience_mode"],
        "additionalProperties": False,
    }


def _runtime_profile_for_document(
    document: Document,
    *,
    source_text: str,
    encoder: tiktoken.Encoding,
    config: LocalDocumentStructurerConfig,
) -> tuple[int, int]:
    source_format = str(document.metadata.get("format", "")).strip().lower()
    document_tokens = _token_count(source_text, encoder)
    if source_format == "pdf" and document_tokens > _LONG_PDF_TRIGGER_TOKENS:
        return (
            min(int(config.max_input_tokens), _LONG_PDF_ADAPTIVE_MAX_INPUT_TOKENS),
            min(int(config.num_predict), _LONG_PDF_ADAPTIVE_NUM_PREDICT),
    )
    return int(config.max_input_tokens), int(config.num_predict)


def _prefer_fallback_model_for_long_pdf(
    *,
    document: Document,
    primary_num_predict: int,
    fallback_model_name: str | None,
) -> bool:
    source_format = str(document.metadata.get("format", "")).strip().lower()
    return (
        source_format == "pdf"
        and int(primary_num_predict) <= _LONG_PDF_ADAPTIVE_NUM_PREDICT
        and bool(str(fallback_model_name or "").strip())
    )


def _use_high_recall_long_pdf_mode(
    *,
    document: Document,
    primary_num_predict: int,
) -> bool:
    """Return whether the current document should stay recall-first end-to-end."""

    source_format = str(document.metadata.get("format", "")).strip().lower()
    return source_format == "pdf" and int(primary_num_predict) <= _LONG_PDF_ADAPTIVE_NUM_PREDICT


def _system_prompt() -> str:
    return _system_prompt_for_mode(_SELECTION_MODE_CONTENT_POOL)


def _system_prompt_for_mode(selection_mode: str) -> str:
    if selection_mode == _SELECTION_MODE_HIGH_RECALL_SALIENT_ONLY:
        return (
            "You select exact evidence units from one chunk of extracted document text for downstream compression.\n"
            "Rules:\n"
            "- Candidate units are already exact source-grounded text spans in source order.\n"
            "- This is a high-recall candidate-selection step for a downstream selector; prefer keeping too many salient units over too few.\n"
            "- Do not invent, rewrite, paraphrase, or merge evidence text.\n"
            "- Select unit IDs that carry concrete facts, entities, numbers, dates, claims, findings, definitions, or important transitions.\n"
            "- Work chunk-locally; do not infer missing context from other chunks.\n"
            "- document_title may be null; if present, keep it short and supported by the chunk.\n"
            "- Return only JSON matching the provided schema.\n"
            "- salient_unit_ids must reference the most important exact source units from the provided list.\n"
        )
    if selection_mode == _SELECTION_MODE_MASK_LOW_SIGNAL:
        return (
            "You select exact evidence units from one chunk of extracted document text for downstream compression.\n"
            "Rules:\n"
            "- Candidate units are already exact source-grounded text spans in source order.\n"
            "- This is a high-recall long-document pass: keep substantive evidence by default.\n"
            "- drop_unit_ids should contain only clearly non-substantive units such as author blocks, affiliations, copyright/license notices, acknowledgements, table-of-contents entries, bibliography-only lines, and other page furniture.\n"
            "- If unsure whether a unit is substantive, keep it.\n"
            "- salient_unit_ids is the smaller highest-priority subset from the remaining substantive units.\n"
            "- Do not invent, rewrite, paraphrase, or merge evidence text.\n"
            "- Work chunk-locally; do not infer missing context from other chunks.\n"
            "- document_title may be null; if present, keep it short and supported by the chunk.\n"
            "- Return only JSON matching the provided schema.\n"
            "- drop_unit_ids and salient_unit_ids must reference exact source unit IDs from the provided list.\n"
        )
    return (
        "You select exact evidence units from one chunk of extracted document text for downstream compression.\n"
        "Rules:\n"
        "- Candidate units are already exact source-grounded text spans in source order.\n"
        "- content_unit_ids is the high-recall content pool for downstream compression; include exact units that carry substance.\n"
        "- salient_unit_ids is the smaller highest-priority subset and should usually be a subset of content_unit_ids.\n"
        "- Prefer keeping too many substantive content units over too few; content_unit_ids may be empty only when the chunk is pure front matter, references, or other clearly non-substantive boilerplate.\n"
        "- Do not invent, rewrite, paraphrase, or merge evidence text.\n"
        "- Select unit IDs that carry concrete facts, entities, numbers, dates, claims, findings, definitions, or important transitions.\n"
        "- If the chunk contains abstract, introduction, methods, results, conclusion, or other body prose, salient_unit_ids must prefer those evidence-bearing body units over front matter.\n"
        "- Never mark a bare section label such as 'Abstract', 'Introduction', or 'Conclusion' as salient when the same chunk contains the surrounding body sentences.\n"
        "- Do not mark title-only lines, author lists, affiliations, email addresses, equal-contribution footnotes, venue metadata, or page furniture as salient when nearby body prose is available.\n"
        "- Do not spend salient_unit_ids on bibliography/reference entries or citation-only lines when the chunk contains substantive prose.\n"
        "- Standalone figure or table captions are salient only when they contain unique numeric findings or claims that are not already stated in nearby prose.\n"
        "- Work chunk-locally; do not infer missing context from other chunks.\n"
        "- Exclude author blocks, affiliations, acknowledgements, copyright/license text, tables of contents, and bibliography/reference-only units unless they are clearly substantive.\n"
        "- document_title may be null; if present, keep it short and supported by the chunk.\n"
        "- Return only JSON matching the provided schema.\n"
        "- content_unit_ids and salient_unit_ids must reference exact source unit IDs from the provided list.\n"
    )


def _document_global_selection_prompt() -> str:
    return (
        "You select a high-recall exact evidence pool for one full document from a candidate list.\n"
        "Rules:\n"
        "- Candidate units are already exact source-grounded text spans; never rewrite or paraphrase them.\n"
        "- This is the document-level evidence pool for downstream compression, not a tiny highlight list.\n"
        "- Any selected unit can survive into the final compressed output, so do not keep low-signal units just because they look related.\n"
        "- content_unit_ids should include substantive claims, definitions, results, formulas, numeric findings, conclusions, and major setup details.\n"
        "- Prefer keeping too many substantive units over too few; use the available budget to keep broad document coverage.\n"
        "- Prefer abstract and body evidence over title pages, author metadata, venue strings, equal-contribution notes, tables of contents, appendices, and bibliography entries.\n"
        "- A bare heading or label is not substantive evidence by itself; keep the surrounding claim-bearing or result-bearing units instead.\n"
        "- Figure or table captions should be kept only when they contain unique findings, formulas, or metrics that are not already covered by nearby prose.\n"
        "- Do not keep bibliography/reference entries, citation-only lines, author/contact blocks, or table-of-contents rows when substantive body evidence is available elsewhere in the candidate list.\n"
        "- Do not keep standalone parameter tables or row fragments unless they carry a uniquely important result that downstream selection would otherwise miss.\n"
        "- Ignore front matter, author/contact blocks, acknowledgements, tables of contents, and reference/bibliography entries unless they are central to the document's substance.\n"
        "- Keep document-wide coverage; do not focus only on one local region.\n"
        "- Return only JSON matching the provided schema.\n"
    )


def _lead_page_seed_prompt() -> str:
    return (
        "You select exact evidence units from the first pages of a scientific paper for a downstream compression seed.\n"
        "Rules:\n"
        "- Candidate units are exact source-grounded lines or sentences from the title page, abstract, and early introduction.\n"
        "- content_unit_ids should keep claim-bearing abstract and introduction body prose that explains the problem, method, results, or conclusions.\n"
        "- salient_unit_ids is the smaller highest-priority subset for the document seed and should focus on the strongest claim-bearing body units.\n"
        "- Exclude title-only lines, author names, affiliations, email addresses, copyright or permission notices, equal-contribution notes, venue strings, arXiv metadata, page numbers, and bare labels such as 'Abstract' or 'Introduction' when nearby body prose is available.\n"
        "- Do not invent, paraphrase, or merge text. Return exact source unit IDs only.\n"
        "- Return only JSON matching the provided schema.\n"
    )


def _document_salience_mode_prompt() -> str:
    return (
        "You choose the safer document-level salience regime for a long extracted document.\n"
        "Rules:\n"
        f"- `{_DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL}` means: keep the broad section pool unchanged, but use only the chunk-local salient sentences as the document salience seed.\n"
        f"- `{_DOCUMENT_SALIENCE_MODE_SECTION_KEEP_ALL}` means: keep the broad section pool unchanged, and use all section sentences as the document salience seed.\n"
        f"- Choose `{_DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL}` when chunk-local salients already give strong document-wide coverage and the full section pool contains too much front matter, OCR noise, bibliography, or low-signal boilerplate.\n"
        f"- Choose `{_DOCUMENT_SALIENCE_MODE_SECTION_KEEP_ALL}` when chunk-local salients are too sparse, too local, or likely to miss important later sections, definitions, formulas, setup details, or result-bearing body prose.\n"
        "- Prefer exact evidence coverage over cosmetic brevity.\n"
        "- Never assume access to content that is not shown in the prompt.\n"
        "- Return only JSON matching the provided schema.\n"
    )


def _sample_candidate_texts_for_prompt(
    texts: Sequence[str],
    *,
    limit: int = DEFAULT_DOCUMENT_GLOBAL_MODE_SAMPLE_LIMIT,
) -> tuple[str, ...]:
    ordered = _dedupe_keep_order(texts)
    if len(ordered) <= int(limit):
        return ordered
    if int(limit) <= 1:
        return (ordered[0],)
    last_index = len(ordered) - 1
    sampled_indices = {
        min(last_index, round((last_index * step) / float(int(limit) - 1)))
        for step in range(int(limit))
    }
    return tuple(ordered[index] for index in sorted(sampled_indices))


def _render_structured_text(title: str | None, sections: Sequence[StructuredSection]) -> str:
    blocks: list[str] = []
    if title:
        blocks.append(title)
    for section in sections:
        lines: list[str] = []
        if section.heading:
            lines.append(section.heading)
        lines.extend(section.sentences)
        block = "\n".join(line for line in lines if line.strip()).strip()
        if block:
            blocks.append(block)
    return "\n\n".join(blocks).strip()


def build_structured_document_trace(
    structured_document: StructuredDocument,
    *,
    encoder: tiktoken.Encoding,
    input_text: str | None = None,
    chunk_traces: Sequence[StructuredChunkTrace] = (),
    global_salient_candidate_count: int = 0,
    global_salient_selected_count: int = 0,
    global_salient_model_name: str | None = None,
) -> StructuredDocumentTrace:
    """Build a JSON-friendly trace for a structured document."""

    source_text = str(input_text) if input_text is not None else structured_document.source_document.text
    sentence_count = sum(len(section.sentences) for section in structured_document.sections)
    return StructuredDocumentTrace(
        source_id=structured_document.source_document.source_id,
        source_name=structured_document.source_document.source_name,
        input_characters=len(source_text),
        input_tokens=_token_count(source_text, encoder),
        rendered_characters=len(structured_document.rendered_text),
        rendered_tokens=_token_count(structured_document.rendered_text, encoder),
        section_count=len(structured_document.sections),
        sentence_count=sentence_count,
        salient_count=len(structured_document.salient_sentences),
        global_salient_candidate_count=int(global_salient_candidate_count),
        global_salient_selected_count=int(global_salient_selected_count),
        global_salient_model_name=(
            str(global_salient_model_name).strip()
            if global_salient_model_name is not None and str(global_salient_model_name).strip()
            else None
        ),
        sections=tuple(structured_document.sections),
        salient_sentences=tuple(structured_document.salient_sentences),
        chunk_traces=tuple(chunk_traces),
    )


def _candidate_unit_budget(
    *,
    chunk_text: str,
    encoder: tiktoken.Encoding,
    max_unit_tokens: int = DEFAULT_DOCUMENT_SELECTION_UNIT_TOKENS,
) -> int:
    chunk_tokens = max(1, _token_count(chunk_text, encoder))
    resolved_max_unit_tokens = max(16, int(max_unit_tokens))
    lower_bound = min(48, resolved_max_unit_tokens)
    return max(lower_bound, min(resolved_max_unit_tokens, chunk_tokens // 8 or lower_bound))


@lru_cache(maxsize=1)
def _sentence_segmenter() -> Any:
    """Load the local sentence segmenter used for noisy PDF text."""

    try:
        from wtpsplit import SaT  # noqa: PLC0415
    except ImportError as exc:
        raise DocumentStructurerResponseError(
            "Document structurer requires wtpsplit for sentence segmentation on noisy document text."
        ) from exc

    return SaT(DEFAULT_DOCUMENT_SENTENCE_SEGMENTER_MODEL)


@lru_cache(maxsize=1)
def _markdown_parser() -> Any:
    """Load the markdown parser used for exact non-PDF markdown unitization."""

    try:
        from markdown_it import MarkdownIt  # noqa: PLC0415
    except ImportError as exc:
        raise DocumentStructurerResponseError(
            "Document structurer requires markdown-it-py for parser-first markdown unitization."
        ) from exc

    parser = MarkdownIt("commonmark")
    try:
        parser.enable("table")
    except Exception:
        # Table support is optional in some markdown-it builds; uncovered rows fall back to exact line units below.
        pass
    return parser


def _segment_block_sentences(block: str) -> tuple[str, ...]:
    """Segment one paragraph-like block into locally predicted sentences."""

    segmenter = _sentence_segmenter()
    segments = segmenter.split(
        block,
        min_length=_SENTENCE_SEGMENTER_MIN_LENGTH,
        strip_whitespace=True,
        split_on_input_newlines=False,
    )
    return tuple(str(sentence).strip() for sentence in segments if str(sentence).strip())


def _append_exact_candidate_text(
    sink: list[str],
    *,
    block_text: str,
    encoder: tiktoken.Encoding,
    unit_token_budget: int,
) -> None:
    """Append an exact source-grounded candidate block, splitting only when token limits require it."""

    normalized_block = str(block_text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n").strip()
    if not normalized_block:
        return
    if _token_count(normalized_block, encoder) > unit_token_budget:
        sink.extend(
            _chunk_text(
                normalized_block,
                encoder=encoder,
                max_input_tokens=unit_token_budget,
            )
        )
        return
    sink.append(normalized_block)


def _markdown_candidate_texts(
    chunk_text: str,
    *,
    encoder: tiktoken.Encoding,
    max_unit_tokens: int = DEFAULT_DOCUMENT_SELECTION_UNIT_TOKENS,
) -> tuple[str, ...]:
    """Derive parser-backed exact markdown units while preserving uncovered rows as atomic line units."""

    unit_token_budget = max(16, int(max_unit_tokens))
    normalized_text = str(chunk_text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n").strip()
    if not normalized_text:
        return ()

    lines = normalized_text.split("\n")
    parser = _markdown_parser()
    tokens = parser.parse(normalized_text)
    candidate_texts: list[str] = []
    covered_nonempty_lines: set[int] = set()
    table_ranges: list[tuple[int, int]] = []
    structural_types = {
        "heading_open",
        "paragraph_open",
        "list_item_open",
        "fence",
        "code_block",
        "tr_open",
    }

    def _append_line_range(start_line: int, end_line: int) -> None:
        start = max(0, int(start_line))
        end = min(len(lines), int(end_line))
        if end <= start:
            return
        nonempty_indices = [index for index in range(start, end) if str(lines[index]).strip()]
        if not nonempty_indices:
            return
        if all(index in covered_nonempty_lines for index in nonempty_indices):
            return
        block_text = "\n".join(lines[start:end]).strip()
        if not block_text:
            return
        _append_exact_candidate_text(
            candidate_texts,
            block_text=block_text,
            encoder=encoder,
            unit_token_budget=unit_token_budget,
        )
        covered_nonempty_lines.update(nonempty_indices)

    for token in tokens:
        token_type = str(getattr(token, "type", "") or "").strip()
        token_map = getattr(token, "map", None)
        if token_type == "table_open" and isinstance(token_map, (list, tuple)) and len(token_map) == 2:
            table_ranges.append((int(token_map[0]), int(token_map[1])))
        if token_type not in structural_types or not isinstance(token_map, (list, tuple)) or len(token_map) != 2:
            continue
        _append_line_range(int(token_map[0]), int(token_map[1]))

    for start_line, end_line in table_ranges:
        for line_index in range(max(0, start_line), min(len(lines), end_line)):
            if str(lines[line_index]).strip():
                covered_nonempty_lines.add(line_index)

    for line_index, line in enumerate(lines):
        normalized_line = str(line or "").strip()
        if not normalized_line or line_index in covered_nonempty_lines:
            continue
        _append_exact_candidate_text(
            candidate_texts,
            block_text=normalized_line,
            encoder=encoder,
            unit_token_budget=unit_token_budget,
        )

    return _dedupe_keep_order(candidate_texts)


def _sentence_candidate_texts(
    chunk_text: str,
    *,
    encoder: tiktoken.Encoding,
    max_unit_tokens: int = DEFAULT_DOCUMENT_SELECTION_UNIT_TOKENS,
    preserve_input_lines: bool = False,
    parser_first_markdown: bool = False,
) -> tuple[str, ...]:
    unit_token_budget = (
        max(16, int(max_unit_tokens))
        if preserve_input_lines
        else _candidate_unit_budget(
            chunk_text=chunk_text,
            encoder=encoder,
            max_unit_tokens=max_unit_tokens,
        )
    )
    normalized_text = str(chunk_text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n").strip()
    if not normalized_text:
        return ()
    if preserve_input_lines and parser_first_markdown:
        parser_units = _markdown_candidate_texts(
            normalized_text,
            encoder=encoder,
            max_unit_tokens=max_unit_tokens,
        )
        if parser_units:
            return parser_units
    candidate_texts: list[str] = []
    for raw_block in normalized_text.split("\n\n"):
        raw_lines = [line.strip() for line in str(raw_block or "").split("\n") if line.strip()]
        if preserve_input_lines:
            block_candidates = raw_lines
        else:
            merged_block = " ".join(raw_lines).strip()
            block_candidates = [merged_block] if merged_block else []
        for block in block_candidates:
            if not block:
                continue
            if preserve_input_lines:
                sentence_candidates = [block]
            else:
                sentence_candidates = list(_segment_block_sentences(block))
                if not sentence_candidates:
                    sentence_candidates = [block]
            for sentence in sentence_candidates:
                if _token_count(sentence, encoder) > unit_token_budget:
                    candidate_texts.extend(
                        _chunk_text(
                            sentence,
                            encoder=encoder,
                            max_input_tokens=unit_token_budget,
                        )
                    )
                else:
                    candidate_texts.append(sentence)
    return _dedupe_keep_order(candidate_texts)


def _candidate_units_from_chunk(
    chunk_text: str,
    *,
    encoder: tiktoken.Encoding,
    max_unit_tokens: int = DEFAULT_DOCUMENT_SELECTION_UNIT_TOKENS,
    preserve_input_lines: bool = False,
    use_sentence_segmentation: bool = True,
    parser_first_markdown: bool = False,
) -> tuple[_CandidateUnit, ...]:
    raw_units: tuple[str, ...] = ()
    if use_sentence_segmentation:
        raw_units = _sentence_candidate_texts(
            chunk_text,
            encoder=encoder,
            max_unit_tokens=max_unit_tokens,
            preserve_input_lines=preserve_input_lines,
            parser_first_markdown=parser_first_markdown,
        )
    if not raw_units:
        raw_units = _dedupe_keep_order(
            _chunk_text(
                chunk_text,
                encoder=encoder,
                max_input_tokens=_candidate_unit_budget(
                    chunk_text=chunk_text,
                    encoder=encoder,
                    max_unit_tokens=max_unit_tokens,
                ),
            )
        )
    if not raw_units:
        raise DocumentStructurerResponseError("Document structurer derived no candidate units from the input chunk.")
    return tuple(
        _CandidateUnit(unit_id=f"U{index + 1:02d}", text=text)
        for index, text in enumerate(raw_units)
    )


def _format_candidate_units_for_prompt(candidate_units: Sequence[_CandidateUnit]) -> str:
    return "\n".join(f"{unit.unit_id}: {unit.text}" for unit in candidate_units)


def _coerce_fragment(
    payload: dict[str, Any],
    *,
    candidate_units: Sequence[_CandidateUnit],
    selection_mode: str,
) -> _StructuredFragment:
    title_value = payload.get("document_title")
    title = str(title_value).strip() if isinstance(title_value, str) and title_value.strip() else None
    candidate_by_id = {unit.unit_id: unit.text for unit in candidate_units}
    ordered_candidate_texts = tuple(unit.text for unit in candidate_units)
    if not ordered_candidate_texts:
        raise DocumentStructurerResponseError("Document structurer has no candidate units to materialize.")

    if selection_mode == _SELECTION_MODE_HIGH_RECALL_SALIENT_ONLY:
        content_sentences = ordered_candidate_texts
    elif selection_mode == _SELECTION_MODE_MASK_LOW_SIGNAL:
        raw_drop = payload.get("drop_unit_ids")
        if not isinstance(raw_drop, list):
            raise DocumentStructurerResponseError("Document structurer returned a non-array drop_unit_ids payload.")
        dropped_ids = {
            str(unit_id).strip()
            for unit_id in raw_drop
            if str(unit_id).strip()
        }
        content_sentences = tuple(
            unit.text
            for unit in candidate_units
            if unit.unit_id not in dropped_ids
        )
    else:
        raw_content = payload.get("content_unit_ids")
        if not isinstance(raw_content, list):
            raise DocumentStructurerResponseError("Document structurer returned a non-array content_unit_ids payload.")
        content_sentences = _dedupe_keep_order(
            candidate_by_id.get(str(unit_id).strip(), "")
            for unit_id in raw_content
        )
    content_sentence_keys = {_grounding_cache_key(sentence) for sentence in content_sentences}

    raw_salient = payload.get("salient_unit_ids")
    if not isinstance(raw_salient, list):
        raise DocumentStructurerResponseError("Document structurer returned a non-array salient_unit_ids payload.")
    salient_sentences = tuple(
        sentence
        for sentence in _dedupe_keep_order(
            candidate_by_id.get(str(unit_id).strip(), "")
            for unit_id in raw_salient
        )
        if selection_mode == _SELECTION_MODE_HIGH_RECALL_SALIENT_ONLY
        or _grounding_cache_key(sentence) in content_sentence_keys
    )
    if selection_mode == _SELECTION_MODE_HIGH_RECALL_SALIENT_ONLY and not salient_sentences:
        salient_sentences = ordered_candidate_texts
    return _StructuredFragment(
        title=title,
        sections=(
            (
                StructuredSection(
                    heading=None,
                    sentences=content_sentences,
                ),
            )
            if content_sentences
            else ()
        ),
        salient_sentences=salient_sentences,
    )


def _validate_fragment_against_input(
    fragment: _StructuredFragment,
    *,
    chunk_text: str,
) -> None:
    if not _normalize_validation_text(chunk_text):
        raise DocumentStructurerResponseError("Document structurer cannot validate an empty input chunk.")
    for section in fragment.sections:
        for sentence in section.sentences:
            if not _sentence_is_anchored_in_input(sentence, chunk_text=chunk_text):
                raise DocumentStructurerResponseError(
                    "Document structurer returned a sentence that is not anchored in the input chunk: "
                    f"{sentence[:200]!r}"
                )


def _sentence_is_anchored_in_input(sentence: str, *, chunk_text: str) -> bool:
    normalized_input = _normalize_validation_text(chunk_text)
    compact_input = _compact_validation_text(chunk_text)
    alnum_input = _alnum_validation_text(chunk_text)
    normalized_sentence = _normalize_validation_text(sentence)
    compact_sentence = _compact_validation_text(sentence)
    alnum_sentence = _alnum_validation_text(sentence)
    if not normalized_sentence:
        raise DocumentStructurerResponseError("Document structurer returned an empty normalized sentence.")
    return (
        normalized_sentence in normalized_input
        or compact_sentence in compact_input
        or alnum_sentence in alnum_input
    )


def _grounding_cache_key(sentence: str) -> str:
    normalized_sentence = _normalize_validation_text(sentence)
    if normalized_sentence:
        return normalized_sentence
    compact_sentence = _compact_validation_text(sentence)
    if compact_sentence:
        return compact_sentence
    return _alnum_validation_text(sentence)


def _reuse_grounded_section_sentence(
    sentence: str,
    *,
    grounded_section_sentences: Sequence[str],
    section_grounding_by_key: Mapping[str, str],
) -> str | None:
    cache_key = _grounding_cache_key(sentence)
    if cache_key:
        reused = section_grounding_by_key.get(cache_key)
        if reused:
            return reused
    for grounded_sentence in grounded_section_sentences:
        if _sentence_is_anchored_in_input(sentence, chunk_text=grounded_sentence):
            return grounded_sentence
        if _sentence_is_anchored_in_input(grounded_sentence, chunk_text=sentence):
            return grounded_sentence
    return None


@lru_cache(maxsize=1)
def _grounding_runtime_config() -> LocalSemanticRuntimeConfig:
    return LocalSemanticRuntimeConfig.from_env()


@lru_cache(maxsize=8)
def _grounding_batch_runtime(profile_id: str) -> _GroundingBatchRuntime:
    config = _grounding_runtime_config()
    profile = resolve_local_semantic_profile(profile_id)
    model_obj = load_local_sentence_transformer_model(
        str(profile.model),
        device=str(config.semantic_pair_device),
        local_files_only=bool(config.semantic_pair_local_files_only),
        model_kind=str(profile.model_kind),
    )
    if model_obj is None:
        snapshot = semantic_model_status_snapshot(str(profile.model))
        raise DocumentStructurerResponseError(
            "Document grounding local cross-encoder is unavailable "
            f"for profile={profile_id!r} detail={json.dumps(snapshot, sort_keys=True)}"
        )
    return _GroundingBatchRuntime(model_name=str(profile.model), model=model_obj)


@lru_cache(maxsize=4)
def _semantic_salience_runtime(model_name: str) -> _SemanticSalienceRuntime:
    config = _grounding_runtime_config()
    model_obj = load_local_sentence_transformer_model(
        str(model_name or ""),
        device=str(config.semantic_pair_device),
        local_files_only=bool(config.semantic_pair_local_files_only),
        model_kind="bi",
    )
    if model_obj is None or not callable(getattr(model_obj, "encode", None)):
        raise DocumentStructurerResponseError(
            f"Document semantic salience bi-encoder is unavailable for model={model_name!r}."
        )
    return _SemanticSalienceRuntime(model_name=str(model_name), model=model_obj)


def _semantic_sentence_matrix(
    candidate_sentences: Sequence[str],
    *,
    model_name: str,
) -> np.ndarray:
    ordered_candidates = [str(text or "").strip() for text in candidate_sentences if str(text or "").strip()]
    if not ordered_candidates:
        raise DocumentStructurerResponseError("Document semantic salience derived no candidate sentences.")
    runtime = _semantic_salience_runtime(model_name)
    config = _grounding_runtime_config()
    vectors = runtime.model.encode(
        ordered_candidates,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=str(config.semantic_pair_device),
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or int(matrix.shape[0]) != len(ordered_candidates):
        raise DocumentStructurerResponseError(
            "Document semantic salience returned an invalid embedding matrix."
        )
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms[row_norms == 0.0] = 1.0
    return matrix / row_norms


def _select_semantic_salient_sentences(
    candidate_sentences: Sequence[str],
    *,
    model_name: str,
    limit: int = DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT,
) -> tuple[str, ...]:
    ordered_candidates = _dedupe_keep_order(candidate_sentences)
    if not ordered_candidates:
        return ()
    resolved_limit = max(1, min(int(limit), len(ordered_candidates)))
    if len(ordered_candidates) <= resolved_limit:
        return ordered_candidates

    matrix = _semantic_sentence_matrix(
        ordered_candidates,
        model_name=model_name,
    )
    centroid = np.mean(matrix, axis=0, keepdims=True)
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid_norm[centroid_norm == 0.0] = 1.0
    centroid = centroid / centroid_norm
    relevance = (matrix @ centroid.T)[:, 0]

    selected_indices = [int(np.argmax(relevance))]
    selected_set = {selected_indices[0]}
    max_redundancy = np.zeros(len(ordered_candidates), dtype=np.float32)
    while len(selected_indices) < resolved_limit:
        similarities = matrix @ matrix[selected_indices[-1]]
        max_redundancy = np.maximum(max_redundancy, similarities)
        scores = (
            float(_DOCUMENT_SEMANTIC_SALIENCE_LAMBDA) * relevance
            - float(1.0 - _DOCUMENT_SEMANTIC_SALIENCE_LAMBDA) * max_redundancy
        )
        scores[list(selected_set)] = -1e9
        next_index = int(np.argmax(scores))
        if next_index in selected_set or float(scores[next_index]) <= -1e8:
            break
        selected_indices.append(next_index)
        selected_set.add(next_index)
    selected_indices.sort()
    return tuple(ordered_candidates[index] for index in selected_indices[:resolved_limit])


def _select_contrastive_salient_sentences(
    candidate_sentences: Sequence[str],
    *,
    rerank_profile_id: str,
    semantic_model_name: str,
    limit: int = DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT,
) -> tuple[str, ...]:
    """Select salient units with a local contrastive reranker plus semantic diversity."""

    ordered_candidates = _dedupe_keep_order(candidate_sentences)
    if not ordered_candidates:
        return ()
    resolved_limit = max(1, min(int(limit), len(ordered_candidates)))
    if len(ordered_candidates) <= resolved_limit:
        return ordered_candidates

    runtime = _grounding_batch_runtime(rerank_profile_id)
    score_matrix = semantic_similarity_matrix(
        queries=(
            "Core substantive document content: abstract, methods, findings, definitions, numeric results, and conclusions.",
            "Non-substantive document noise: author metadata, affiliations, acknowledgements, references, bibliography, venue strings, and page furniture.",
        ),
        docs=tuple(ordered_candidates),
        model_name=runtime.model_name,
        model=runtime.model,
        batch_size=64,
    )
    if score_matrix is None or len(score_matrix) != 2 or any(len(row) != len(ordered_candidates) for row in score_matrix):
        raise DocumentStructurerResponseError("Document contrastive rerank selection returned an invalid score matrix.")

    matrix = _semantic_sentence_matrix(
        ordered_candidates,
        model_name=semantic_model_name,
    )
    centroid = np.mean(matrix, axis=0, keepdims=True)
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid_norm[centroid_norm == 0.0] = 1.0
    centroid = centroid / centroid_norm
    centroid_rel = (matrix @ centroid.T)[:, 0]

    positive = np.asarray(score_matrix[0], dtype=np.float32)
    negative = np.asarray(score_matrix[1], dtype=np.float32)
    relevance = positive - negative + 0.15 * centroid_rel

    selected_indices = [int(np.argmax(relevance))]
    selected_set = {selected_indices[0]}
    max_redundancy = np.zeros(len(ordered_candidates), dtype=np.float32)
    while len(selected_indices) < resolved_limit:
        similarities = matrix @ matrix[selected_indices[-1]]
        max_redundancy = np.maximum(max_redundancy, similarities)
        scores = (
            float(_DOCUMENT_CONTRASTIVE_RERANK_LAMBDA) * relevance
            - float(1.0 - _DOCUMENT_CONTRASTIVE_RERANK_LAMBDA) * max_redundancy
        )
        scores[list(selected_set)] = -1e9
        next_index = int(np.argmax(scores))
        if next_index in selected_set or float(scores[next_index]) <= -1e8:
            break
        selected_indices.append(next_index)
        selected_set.add(next_index)
    selected_indices.sort()
    return tuple(ordered_candidates[index] for index in selected_indices[:resolved_limit])


def _long_pdf_requires_extractive_fast_path(
    *,
    document: Document,
    source_text: str,
    encoder: tiktoken.Encoding,
    primary_num_predict: int,
) -> bool:
    """Return whether to bypass chunk-local structuring for this document."""

    del document, source_text, encoder, primary_num_predict
    # The handoff quality corridor is materially stronger when long PDFs stay on the
    # chunk-local, recall-first path. Keep the extractive fast path disabled until it
    # beats the canonical handoff benchmark again.
    return False


def _ground_fragment_sentences_batched(
    sentences: Sequence[str],
    *,
    chunk_text: str,
    candidate_spans: Sequence[str],
    source_name: str,
    chunk_index: int,
    rerank_profile_id: str,
) -> dict[str, str]:
    del chunk_text
    normalized_sentences = [
        str(sentence or "").strip()
        for sentence in list(sentences or [])
        if str(sentence or "").strip()
    ]
    if not normalized_sentences:
        return {}
    if not candidate_spans:
        raise DocumentStructurerResponseError(
            f"Document grounding derived no candidate spans for {source_name} chunk={chunk_index}."
        )
    runtime = _grounding_batch_runtime(rerank_profile_id)
    score_matrix = semantic_similarity_matrix(
        queries=tuple(normalized_sentences),
        docs=tuple(candidate_spans),
        model_name=runtime.model_name,
        model=runtime.model,
        batch_size=64,
    )
    if score_matrix is None or len(score_matrix) != len(normalized_sentences):
        raise DocumentStructurerResponseError(
            f"Document grounding scoring failed for {source_name} chunk={chunk_index}."
        )
    grounded_sentences: dict[str, str] = {}
    for sentence, row in zip(normalized_sentences, score_matrix):
        if not row:
            raise DocumentStructurerResponseError(
                f"Document grounding returned no span match for {source_name} chunk={chunk_index}."
            )
        best_index = max(range(len(row)), key=row.__getitem__)
        grounded = str(candidate_spans[best_index]).strip()
        if not grounded:
            raise DocumentStructurerResponseError(
                f"Document grounding selected an empty span for {source_name} chunk={chunk_index}."
            )
        grounded_sentences[sentence] = grounded
    return grounded_sentences


def _ground_fragment_against_input(
    fragment: _StructuredFragment,
    *,
    chunk_text: str,
    encoder: tiktoken.Encoding,
    source_name: str,
    chunk_index: int,
    rerank_profile_id: str,
    grounding_span_tokens: int,
) -> _GroundedFragmentResult:
    candidate_spans = _dedupe_keep_order(
        _sentence_candidate_texts(
            chunk_text,
            encoder=encoder,
            max_unit_tokens=max(DEFAULT_DOCUMENT_SELECTION_UNIT_TOKENS, int(grounding_span_tokens)),
            preserve_input_lines=False,
        )
    )
    if not candidate_spans:
        candidate_spans = _dedupe_keep_order(
            _chunk_text(
                chunk_text,
                encoder=encoder,
                max_input_tokens=max(8, int(grounding_span_tokens)),
            )
        )
    if not candidate_spans:
        raise DocumentStructurerResponseError(
            f"Document grounding derived no candidate spans for {source_name} chunk={chunk_index}."
        )

    sentence_requests = 0
    unique_sentences = 0
    rerank_calls = 0
    cache_hits = 0
    salient_reuse_count = 0
    sentence_grounding_cache: dict[str, str] = {}
    section_grounding_by_key: dict[str, str] = {}
    pending_grounding: dict[str, str] = {}

    def _register_grounding_request(sentence: str) -> None:
        nonlocal sentence_requests, unique_sentences, cache_hits
        normalized_sentence = str(sentence or "").strip()
        if not normalized_sentence:
            return
        sentence_requests += 1
        cache_key = _grounding_cache_key(normalized_sentence)
        if cache_key and cache_key in sentence_grounding_cache:
            cache_hits += 1
            return
        if cache_key and cache_key in pending_grounding:
            cache_hits += 1
            return
        unique_sentences += 1
        if _sentence_is_anchored_in_input(normalized_sentence, chunk_text=chunk_text):
            if cache_key:
                sentence_grounding_cache[cache_key] = normalized_sentence
            return
        if cache_key:
            pending_grounding[cache_key] = normalized_sentence

    def _flush_pending_grounding() -> None:
        nonlocal rerank_calls
        if not pending_grounding:
            return
        grounded_batch = _ground_fragment_sentences_batched(
            tuple(pending_grounding.values()),
            chunk_text=chunk_text,
            candidate_spans=candidate_spans,
            source_name=source_name,
            chunk_index=chunk_index,
            rerank_profile_id=rerank_profile_id,
        )
        rerank_calls += 1
        for cache_key, sentence in tuple(pending_grounding.items()):
            grounded = str(grounded_batch.get(sentence) or "").strip()
            if grounded:
                sentence_grounding_cache[cache_key] = grounded
        pending_grounding.clear()

    grounded_sections: list[StructuredSection] = []
    seen_global_sentences: set[str] = set()
    for section in fragment.sections:
        for sentence in section.sentences:
            _register_grounding_request(sentence)
    _flush_pending_grounding()

    for section in fragment.sections:
        grounded_sentences: list[str] = []
        for sentence in section.sentences:
            normalized_sentence = str(sentence or "").strip()
            if not normalized_sentence:
                continue
            cache_key = _grounding_cache_key(normalized_sentence)
            grounded = sentence_grounding_cache.get(cache_key, "")
            if not grounded:
                continue
            key = grounded.casefold()
            if key in seen_global_sentences:
                continue
            seen_global_sentences.add(key)
            grounded_sentences.append(grounded)
            if cache_key:
                section_grounding_by_key[cache_key] = grounded
        if grounded_sentences:
            grounded_sections.append(
                StructuredSection(
                    heading=section.heading,
                    sentences=tuple(grounded_sentences),
                )
            )

    grounded_section_sentences = _dedupe_keep_order(
        sentence
        for section in grounded_sections
        for sentence in section.sentences
    )

    grounded_salient_raw: list[str] = []
    if grounded_section_sentences:
        target_salient_count = len(fragment.salient_sentences)
        grounded_salient_seen: set[str] = set()
        for sentence in fragment.salient_sentences:
            grounded = _reuse_grounded_section_sentence(
                sentence,
                grounded_section_sentences=grounded_section_sentences,
                section_grounding_by_key=section_grounding_by_key,
            )
            if not grounded:
                continue
            key = grounded.casefold()
            if key in grounded_salient_seen:
                continue
            grounded_salient_seen.add(key)
            grounded_salient_raw.append(grounded)
            salient_reuse_count += 1
        for grounded in grounded_section_sentences:
            if len(grounded_salient_raw) >= target_salient_count:
                break
            key = grounded.casefold()
            if key in grounded_salient_seen:
                continue
            grounded_salient_seen.add(key)
            grounded_salient_raw.append(grounded)
    else:
        for sentence in fragment.salient_sentences:
            _register_grounding_request(sentence)
        _flush_pending_grounding()
        grounded_salient_raw = []
        for sentence in fragment.salient_sentences:
            normalized_sentence = str(sentence or "").strip()
            if not normalized_sentence:
                continue
            cache_key = _grounding_cache_key(normalized_sentence)
            grounded = sentence_grounding_cache.get(cache_key, "")
            if grounded:
                grounded_salient_raw.append(grounded)

    grounded_salient = _dedupe_keep_order(grounded_salient_raw)

    if not grounded_sections:
        if not grounded_salient:
            return _GroundedFragmentResult(
                fragment=_StructuredFragment(
                    title=fragment.title,
                    sections=(),
                    salient_sentences=(),
                ),
                candidate_span_count=len(candidate_spans),
                sentence_requests=sentence_requests,
                unique_sentences=unique_sentences,
                rerank_calls=rerank_calls,
                cache_hits=cache_hits,
                salient_reuse_count=salient_reuse_count,
            )
        grounded_sections = [
            StructuredSection(
                heading=next((section.heading for section in fragment.sections if section.heading), None),
                sentences=tuple(grounded_salient),
            )
        ]
    elif fragment.salient_sentences and not grounded_salient:
        grounded_salient = _dedupe_keep_order(
            sentence
            for section in grounded_sections
            for sentence in section.sentences
        )

    return _GroundedFragmentResult(
        fragment=_StructuredFragment(
            title=fragment.title,
            sections=tuple(grounded_sections),
            salient_sentences=grounded_salient,
        ),
        candidate_span_count=len(candidate_spans),
        sentence_requests=sentence_requests,
        unique_sentences=unique_sentences,
        rerank_calls=rerank_calls,
        cache_hits=cache_hits,
        salient_reuse_count=salient_reuse_count,
    )


class LocalOllamaDocumentStructurer:
    """Local schema-bound document structurer backed by Ollama."""

    def __init__(self, config: LocalDocumentStructurerConfig) -> None:
        self._config = config
        self.name = f"local-ollama-document:{config.model_name}"

    def _request_structured_payload(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        num_predict: int,
    ) -> dict[str, Any]:
        payload = {
            "model": str(model_name).strip(),
            "stream": False,
            "format": dict(schema),
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "options": {
                "temperature": 0,
                "num_predict": int(num_predict),
            },
        }
        req = request.Request(
            f"{self._config.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DocumentStructurerTransportError(
                f"Document structurer HTTP {exc.code} from Ollama at {self._config.ollama_base_url}: {body}"
            ) from exc
        except error.URLError as exc:
            raise DocumentStructurerTransportError(
                "Document structurer could not reach the local Ollama runtime at "
                f"{self._config.ollama_base_url}. Ensure Ollama is running and the model "
                f"{model_name} is available."
            ) from exc
        except TimeoutError as exc:
            raise DocumentStructurerTransportError(
                f"Document structurer timed out after {self._config.timeout_seconds:.1f}s "
                f"while calling {model_name}."
            ) from exc

        content = (
            raw_response.get("message", {}).get("content")
            if isinstance(raw_response, dict)
            else None
        )
        if not isinstance(content, str) or not content.strip():
            raise DocumentStructurerResponseError("Document structurer returned an empty response body.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise DocumentStructurerResponseError(f"Document structurer returned invalid JSON: {content}") from exc
        if not isinstance(parsed, dict):
            raise DocumentStructurerResponseError("Document structurer returned a non-object JSON payload.")
        return parsed

    def _selection_mode_for_document(
        self,
        *,
        document: Document,
        primary_num_predict: int,
    ) -> str:
        if self._config.selection_mode:
            return str(self._config.selection_mode)
        source_format = str(document.metadata.get("format", "")).strip().lower()
        if source_format == "pdf" and int(primary_num_predict) <= _LONG_PDF_ADAPTIVE_NUM_PREDICT:
            # Product evidence from the handoff corridor is clear here: long-PDF quality
            # is materially stronger when the chunk-local pass keeps a broad content pool,
            # and the document-level salience regime decides how aggressively to focus it.
            return _SELECTION_MODE_CONTENT_POOL
        return _SELECTION_MODE_CONTENT_POOL

    def _select_document_salience_mode(
        self,
        *,
        document: Document,
        raw_salient_sentences: Sequence[str],
        section_sentences: Sequence[str],
        primary_num_predict: int,
    ) -> _DocumentSalienceModeDecision:
        raw_salients = _dedupe_keep_order(raw_salient_sentences)
        section_pool = _dedupe_keep_order(section_sentences)
        if not section_pool:
            return _DocumentSalienceModeDecision(
                salience_mode=_DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL,
                model_name=None,
            )
        if not raw_salients:
            return _DocumentSalienceModeDecision(
                salience_mode=_DOCUMENT_SALIENCE_MODE_SECTION_KEEP_ALL,
                model_name=None,
            )
        if not _use_high_recall_long_pdf_mode(
            document=document,
            primary_num_predict=primary_num_predict,
        ):
            return _DocumentSalienceModeDecision(
                salience_mode=_DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL,
                model_name=None,
            )
        if len(section_pool) <= int(DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT):
            return _DocumentSalienceModeDecision(
                salience_mode=_DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL,
                model_name=_DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL,
            )

        raw_preview = tuple(
            _CandidateUnit(unit_id=f"R{index + 1:03d}", text=text)
            for index, text in enumerate(_sample_candidate_texts_for_prompt(raw_salients))
        )
        section_preview = tuple(
            _CandidateUnit(unit_id=f"S{index + 1:03d}", text=text)
            for index, text in enumerate(_sample_candidate_texts_for_prompt(section_pool))
        )
        fallback_model_name = str(self._config.fallback_model_name or "").strip()
        attempt_order: list[tuple[str, int]] = []
        if fallback_model_name:
            attempt_order.append(
                (
                    fallback_model_name,
                    max(int(self._config.fallback_num_predict), int(DEFAULT_DOCUMENT_GLOBAL_SELECTION_NUM_PREDICT)),
                )
            )
        primary_model_name = str(self._config.model_name).strip()
        if primary_model_name and primary_model_name != fallback_model_name:
            attempt_order.append(
                (
                    primary_model_name,
                    max(int(primary_num_predict), int(DEFAULT_DOCUMENT_GLOBAL_SELECTION_NUM_PREDICT)),
                )
            )
        for attempt_model_name, attempt_num_predict in attempt_order:
            try:
                parsed = self._request_structured_payload(
                    model_name=attempt_model_name,
                    system_prompt=_document_salience_mode_prompt(),
                    user_prompt=(
                        f"source_name: {document.source_name}\n"
                        f"source_format: {str(document.metadata.get('format', 'text')).strip().lower() or 'text'}\n"
                        f"chunk_local_salient_count: {len(raw_salients)}\n"
                        f"section_sentence_count: {len(section_pool)}\n\n"
                        "Chunk-local salient sample (exact source text):\n"
                        f"{_format_candidate_units_for_prompt(raw_preview) if raw_preview else '(none)'}\n\n"
                        "Full section-pool sample (exact source text):\n"
                        f"{_format_candidate_units_for_prompt(section_preview) if section_preview else '(none)'}"
                    ),
                    schema=_document_salience_mode_schema(),
                    num_predict=attempt_num_predict,
                )
                resolved_mode = str(parsed.get("salience_mode", "")).strip()
                if resolved_mode not in {
                    _DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL,
                    _DOCUMENT_SALIENCE_MODE_SECTION_KEEP_ALL,
                }:
                    raise DocumentStructurerResponseError(
                        "Document structurer returned an invalid document salience mode."
                    )
                return _DocumentSalienceModeDecision(
                    salience_mode=resolved_mode,
                    model_name=attempt_model_name,
                )
            except DocumentStructurerTransportError:
                continue
            except DocumentStructurerResponseError:
                continue

        return _DocumentSalienceModeDecision(
            salience_mode=_DOCUMENT_SALIENCE_MODE_SECTION_KEEP_ALL,
            model_name=None,
        )

    def _select_semantic_document_salients(
        self,
        *,
        section_sentences: Sequence[str],
    ) -> _DocumentGlobalSelectionResult:
        ordered_candidates = _dedupe_keep_order(section_sentences)
        if not ordered_candidates:
            return _DocumentGlobalSelectionResult(
                salient_sentences=(),
                candidate_count=0,
                selected_count=0,
                model_name=None,
            )
        if len(ordered_candidates) <= int(DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT):
            return _DocumentGlobalSelectionResult(
                salient_sentences=ordered_candidates,
                candidate_count=len(ordered_candidates),
                selected_count=len(ordered_candidates),
                model_name=None,
            )
        selected = _select_semantic_salient_sentences(
            ordered_candidates,
            model_name=str(self._config.salience_embedding_model_name),
            limit=DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT,
        )
        return _DocumentGlobalSelectionResult(
            salient_sentences=selected,
            candidate_count=len(ordered_candidates),
            selected_count=len(selected),
            model_name=f"semantic_key_pool:{self._config.salience_embedding_model_name}",
        )

    def _select_document_global_salients(
        self,
        *,
        document: Document,
        candidate_sentences: Sequence[str],
        primary_num_predict: int,
    ) -> _DocumentGlobalSelectionResult:
        ordered_candidates = _dedupe_keep_order(candidate_sentences)
        if not ordered_candidates:
            return _DocumentGlobalSelectionResult(
                salient_sentences=(),
                candidate_count=0,
                selected_count=0,
                model_name=None,
            )
        if len(ordered_candidates) <= int(DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT):
            return _DocumentGlobalSelectionResult(
                salient_sentences=ordered_candidates,
                candidate_count=len(ordered_candidates),
                selected_count=len(ordered_candidates),
                model_name=(
                    "chunk_local_salients"
                    if _use_high_recall_long_pdf_mode(
                        document=document,
                        primary_num_predict=primary_num_predict,
                    )
                    else None
                ),
            )
        fallback_model_name = str(self._config.fallback_model_name or "").strip()
        if not _prefer_fallback_model_for_long_pdf(
            document=document,
            primary_num_predict=primary_num_predict,
            fallback_model_name=fallback_model_name,
        ):
            return _DocumentGlobalSelectionResult(
                salient_sentences=ordered_candidates,
                candidate_count=len(ordered_candidates),
                selected_count=len(ordered_candidates),
                model_name=None,
            )

        attempt_order: list[tuple[str, int]] = []
        if fallback_model_name:
            attempt_order.append(
                (
                    fallback_model_name,
                    max(int(self._config.fallback_num_predict), int(DEFAULT_DOCUMENT_GLOBAL_SELECTION_NUM_PREDICT)),
                )
            )
        primary_model_name = str(self._config.model_name).strip()
        if primary_model_name and primary_model_name != fallback_model_name:
            attempt_order.append(
                (
                    primary_model_name,
                    max(int(primary_num_predict), int(DEFAULT_DOCUMENT_GLOBAL_SELECTION_NUM_PREDICT)),
                )
            )
        primary_error: DocumentStructurerResponseError | None = None
        for attempt_model_name, attempt_num_predict in attempt_order:
            try:
                selected = self._select_document_global_salients_hierarchical(
                    document=document,
                    ordered_candidates=ordered_candidates,
                    model_name=attempt_model_name,
                    num_predict=attempt_num_predict,
                )
                if not selected:
                    raise DocumentStructurerResponseError(
                        "Document structurer selected no usable global evidence units."
                    )
                return _DocumentGlobalSelectionResult(
                    salient_sentences=selected,
                    candidate_count=len(ordered_candidates),
                    selected_count=len(selected),
                    model_name=attempt_model_name,
                )
            except DocumentStructurerTransportError:
                continue
            except DocumentStructurerResponseError as exc:
                primary_error = exc
        raise primary_error or DocumentStructurerResponseError(
            "Document structurer failed to select global salient evidence units."
        )

    def _select_document_global_salients_window(
        self,
        *,
        document: Document,
        candidate_units: Sequence[_CandidateUnit],
        model_name: str,
        num_predict: int,
    ) -> tuple[str, ...]:
        parsed = self._request_structured_payload(
            model_name=model_name,
            system_prompt=_document_global_selection_prompt(),
            user_prompt=(
                f"source_name: {document.source_name}\n"
                f"source_format: {str(document.metadata.get('format', 'text')).strip().lower() or 'text'}\n"
                f"candidate_unit_count: {len(candidate_units)}\n\n"
                "Candidate units (exact source text, preserve IDs):\n"
                f"{_format_candidate_units_for_prompt(candidate_units)}"
            ),
            schema=_global_selection_schema(unit_ids=tuple(unit.unit_id for unit in candidate_units)),
            num_predict=num_predict,
        )
        raw_content = parsed.get("content_unit_ids")
        if not isinstance(raw_content, list):
            raise DocumentStructurerResponseError(
                "Document structurer returned a non-array global content_unit_ids payload."
            )
        selected_ids = {
            str(unit_id).strip()
            for unit_id in raw_content
            if str(unit_id).strip()
        }
        selected = tuple(
            unit.text
            for unit in candidate_units
            if unit.unit_id in selected_ids
        )
        return selected

    def _select_document_global_salients_hierarchical(
        self,
        *,
        document: Document,
        ordered_candidates: Sequence[str],
        model_name: str,
        num_predict: int,
    ) -> tuple[str, ...]:
        candidate_units = tuple(
            _CandidateUnit(unit_id=f"G{index + 1:03d}", text=text)
            for index, text in enumerate(ordered_candidates)
        )
        if len(candidate_units) <= int(DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT):
            return self._select_document_global_salients_window(
                document=document,
                candidate_units=candidate_units,
                model_name=model_name,
                num_predict=num_predict,
            )

        window_size = int(DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT)
        staged_selected: list[str] = []
        for start in range(0, len(candidate_units), window_size):
            window_units = candidate_units[start : start + window_size]
            staged_selected.extend(
                self._select_document_global_salients_window(
                    document=document,
                    candidate_units=window_units,
                    model_name=model_name,
                    num_predict=num_predict,
                )
            )
        deduped_selected = _dedupe_keep_order(staged_selected)
        if not deduped_selected:
            return ()

        reduced_units = tuple(
            _CandidateUnit(unit_id=f"G{index + 1:03d}", text=text)
            for index, text in enumerate(deduped_selected)
        )
        final_selected = self._select_document_global_salients_window(
            document=document,
            candidate_units=reduced_units,
            model_name=model_name,
            num_predict=num_predict,
        )
        return final_selected or deduped_selected

    def _build_long_pdf_lead_seed(
        self,
        *,
        document: Document,
        encoder: tiktoken.Encoding,
    ) -> _DocumentGlobalSelectionResult | None:
        """Select a high-quality lead-page seed for long scientific PDFs."""

        origin_path = str(document.origin_path or "").strip()
        if not origin_path:
            return None
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError:
            return None
        path = Path(origin_path)
        if not path.exists():
            return None
        try:
            reader = PdfReader(str(path))
            lead_text = "\n\n".join(
                (reader.pages[index].extract_text() or "").strip()
                for index in range(min(2, len(reader.pages)))
            ).strip()
        except Exception:
            return None
        if not lead_text:
            return None
        seed_sentences = _sentence_candidate_texts(
            lead_text,
            encoder=encoder,
            max_unit_tokens=_LONG_PDF_EXTRACTIVE_UNIT_TOKENS,
            preserve_input_lines=True,
        )
        if not seed_sentences:
            return None
        candidate_units = tuple(
            _CandidateUnit(unit_id=f"L{index + 1:03d}", text=text)
            for index, text in enumerate(seed_sentences)
        )
        fallback_model_name = str(self._config.fallback_model_name or "").strip()
        attempt_order: list[str] = []
        if fallback_model_name:
            attempt_order.append(fallback_model_name)
        primary_model_name = str(self._config.model_name).strip()
        if primary_model_name and primary_model_name not in attempt_order:
            attempt_order.append(primary_model_name)
        for attempt_model_name in attempt_order:
            try:
                parsed = self._request_structured_payload(
                    model_name=attempt_model_name,
                    system_prompt=_lead_page_seed_prompt(),
                    user_prompt=(
                        f"source_name: {document.source_name}\n"
                        "source_format: pdf\n"
                        f"candidate_unit_count: {len(candidate_units)}\n\n"
                        "Lead-page candidate units (exact source text, preserve IDs):\n"
                        f"{_format_candidate_units_for_prompt(candidate_units)}"
                    ),
                    schema=_selection_schema(
                        unit_ids=tuple(unit.unit_id for unit in candidate_units),
                        num_predict=max(int(DEFAULT_DOCUMENT_GLOBAL_SELECTION_NUM_PREDICT), int(self._config.fallback_num_predict)),
                        selection_mode=_SELECTION_MODE_CONTENT_POOL,
                    ),
                    num_predict=max(int(DEFAULT_DOCUMENT_GLOBAL_SELECTION_NUM_PREDICT), int(self._config.fallback_num_predict)),
                )
                fragment = _coerce_fragment(
                    parsed,
                    candidate_units=candidate_units,
                    selection_mode=_SELECTION_MODE_CONTENT_POOL,
                )
                model_selected_seed = fragment.salient_sentences or tuple(
                    sentence
                    for section in fragment.sections
                    for sentence in section.sentences
                )
                selected_seed = tuple(model_selected_seed[: int(_LONG_PDF_LEAD_SEED_LIMIT)])
                if len(selected_seed) < int(_LONG_PDF_LEAD_SEED_LIMIT):
                    try:
                        contrastive_seed = _select_contrastive_salient_sentences(
                            seed_sentences,
                            rerank_profile_id=self._config.grounding_rerank_profile_id,
                            semantic_model_name=self._config.salience_embedding_model_name,
                            limit=min(int(_LONG_PDF_LEAD_SEED_LIMIT), len(seed_sentences)),
                        )
                    except (DocumentStructurerTransportError, DocumentStructurerResponseError):
                        contrastive_seed = ()
                    supplemented_seed = _dedupe_keep_order(
                        tuple(selected_seed) + tuple(contrastive_seed)
                    )[: int(_LONG_PDF_LEAD_SEED_LIMIT)]
                else:
                    contrastive_seed = ()
                    supplemented_seed = selected_seed
                if selected_seed:
                    model_name = f"lead_page_model_seed:{attempt_model_name}"
                    if len(supplemented_seed) > len(selected_seed):
                        model_name = (
                            "lead_page_model_plus_contrastive_seed:"
                            f"{attempt_model_name}+{self._config.grounding_rerank_profile_id}"
                        )
                    return _DocumentGlobalSelectionResult(
                        salient_sentences=supplemented_seed,
                        candidate_count=len(seed_sentences),
                        selected_count=len(supplemented_seed),
                        model_name=model_name,
                    )
            except (DocumentStructurerTransportError, DocumentStructurerResponseError):
                continue
        try:
            resolved_seed = _select_contrastive_salient_sentences(
                seed_sentences,
                rerank_profile_id=self._config.grounding_rerank_profile_id,
                semantic_model_name=self._config.salience_embedding_model_name,
                limit=min(int(_LONG_PDF_LEAD_SEED_LIMIT), len(seed_sentences)),
            )
        except (DocumentStructurerTransportError, DocumentStructurerResponseError):
            return None
        if not resolved_seed:
            return None
        return _DocumentGlobalSelectionResult(
            salient_sentences=resolved_seed,
            candidate_count=len(seed_sentences),
            selected_count=len(resolved_seed),
            model_name=(
                "lead_page_extract_contrastive_seed:"
                f"{self._config.grounding_rerank_profile_id}"
            ),
        )

    def _resolve_long_pdf_source_text(
        self,
        *,
        document: Document,
        source_text: str,
    ) -> str:
        """Prefer direct local PDF extraction over degraded pre-extracted layout text."""

        source_format = str(document.metadata.get("format", "")).strip().lower()
        origin_path = str(document.origin_path or "").strip()
        if source_format != "pdf" or not origin_path:
            return source_text
        try:
            from pypdf import PdfReader  # noqa: PLC0415
        except ImportError:
            return source_text

        path = Path(origin_path)
        if not path.exists():
            return source_text
        try:
            reader = PdfReader(str(path))
            extracted = "\n\n".join(
                (page.extract_text() or "").strip()
                for page in reader.pages
            ).strip()
        except Exception:
            return source_text
        return extracted or source_text

    def _structure_long_pdf_extractive(
        self,
        *,
        document: Document,
        source_text: str,
        encoder: tiktoken.Encoding,
        primary_num_predict: int,
        capture_trace: bool,
    ) -> tuple[StructuredDocument, StructuredDocumentTrace | None]:
        """Build a high-recall long-PDF structure without chunk-wise generation."""

        resolved_source_text = self._resolve_long_pdf_source_text(
            document=document,
            source_text=source_text,
        )
        sentence_units = _sentence_candidate_texts(
            resolved_source_text,
            encoder=encoder,
            max_unit_tokens=_LONG_PDF_EXTRACTIVE_UNIT_TOKENS,
            preserve_input_lines=True,
        )
        if not sentence_units:
            raise RuntimeError(
                f"Document structurer derived no extractive sentence units for {document.source_name}."
            )
        lead_seed = self._build_long_pdf_lead_seed(
            document=document,
            encoder=encoder,
        )
        sections = (
            StructuredSection(
                heading=None,
                sentences=tuple(sentence_units),
            ),
        )
        try:
            salient_sentences = _select_contrastive_salient_sentences(
                sentence_units,
                rerank_profile_id=self._config.grounding_rerank_profile_id,
                semantic_model_name=self._config.salience_embedding_model_name,
                limit=min(int(_LONG_PDF_DOCUMENT_SALIENT_LIMIT), len(sentence_units)),
            )
            global_selection = _DocumentGlobalSelectionResult(
                salient_sentences=salient_sentences,
                candidate_count=len(sentence_units),
                selected_count=len(salient_sentences),
                model_name=f"contrastive_document_salience_pool:{self._config.grounding_rerank_profile_id}",
            )
        except DocumentStructurerResponseError:
            if lead_seed is not None:
                global_selection = lead_seed
            else:
                global_selection = self._select_semantic_document_salients(
                    section_sentences=sentence_units,
                )
        rendered_text = _render_structured_text(None, sections)
        if not rendered_text:
            raise RuntimeError(
                f"Document structurer produced no rendered text for {document.source_name}."
            )
        structured_document = StructuredDocument(
            source_document=Document(
                source_id=document.source_id,
                source_name=document.source_name,
                text=rendered_text,
                origin_path=document.origin_path,
                metadata=dict(document.metadata),
            ),
            rendered_text=rendered_text,
            sections=sections,
            salient_sentences=global_selection.salient_sentences,
        )
        if not capture_trace:
            return structured_document, None
        chunk_trace = StructuredChunkTrace(
            chunk_index=0,
            chunk_count=1,
            model_name=f"extractive:{DEFAULT_DOCUMENT_SENTENCE_SEGMENTER_MODEL}",
            input_characters=len(resolved_source_text),
            input_tokens=_token_count(resolved_source_text, encoder),
            fragment_title=None,
            section_count=len(sections),
            sentence_count=len(sentence_units),
            salient_count=len(global_selection.salient_sentences),
            section_headings=(None,),
            salient_sentences=tuple(global_selection.salient_sentences),
            grounding_candidate_span_count=0,
            grounding_sentence_requests=0,
            grounding_unique_sentences=0,
            grounding_rerank_calls=0,
            grounding_cache_hits=0,
            grounding_salient_reuse_count=0,
            selection_mode=_SELECTION_MODE_EXTRACTIVE_LONG_PDF,
            candidate_unit_count=len(sentence_units),
            kept_unit_count=len(sentence_units),
            dropped_unit_count=0,
        )
        return structured_document, build_structured_document_trace(
            structured_document,
            encoder=encoder,
            input_text=resolved_source_text,
            chunk_traces=(chunk_trace,),
            global_salient_candidate_count=global_selection.candidate_count,
            global_salient_selected_count=global_selection.selected_count,
            global_salient_model_name=global_selection.model_name,
        )

    def _call_ollama(
        self,
        *,
        document: Document,
        chunk_text: str,
        encoder: tiktoken.Encoding,
        chunk_index: int,
        chunk_count: int,
        model_name: str | None = None,
        num_predict: int | None = None,
        selection_mode: str = _SELECTION_MODE_CONTENT_POOL,
    ) -> tuple[_StructuredFragment, int, str]:
        resolved_model_name = str(model_name or self._config.model_name).strip()
        resolved_num_predict = int(num_predict if num_predict is not None else self._config.num_predict)
        source_format = str(document.metadata.get("format", "text")).strip().lower() or "text"
        effective_selection_mode = selection_mode
        if (
            selection_mode == _SELECTION_MODE_CONTENT_POOL
            and _use_high_recall_long_pdf_mode(
                document=document,
                primary_num_predict=resolved_num_predict,
            )
        ):
            effective_selection_mode = _SELECTION_MODE_HIGH_RECALL_SALIENT_ONLY
        candidate_unit_max_tokens = (
            _LONG_PDF_SELECTION_UNIT_TOKENS
            if _use_high_recall_long_pdf_mode(
                document=document,
                primary_num_predict=resolved_num_predict,
            )
            else DEFAULT_DOCUMENT_SELECTION_UNIT_TOKENS
        )
        candidate_units = _candidate_units_from_chunk(
            chunk_text,
            encoder=encoder,
            max_unit_tokens=candidate_unit_max_tokens,
            preserve_input_lines=source_format == "markdown",
            parser_first_markdown=source_format == "markdown",
            use_sentence_segmentation=effective_selection_mode != _SELECTION_MODE_HIGH_RECALL_SALIENT_ONLY,
        )
        user_prompt = (
            f"source_name: {document.source_name}\n"
            f"source_format: {source_format}\n"
            f"chunk_index: {chunk_index + 1}\n"
            f"chunk_count: {chunk_count}\n"
            f"candidate_unit_count: {len(candidate_units)}\n\n"
            "Candidate units (exact source text, preserve IDs):\n"
            f"{_format_candidate_units_for_prompt(candidate_units)}"
        )
        parsed = self._request_structured_payload(
            model_name=resolved_model_name,
            system_prompt=_system_prompt_for_mode(effective_selection_mode),
            user_prompt=user_prompt,
            schema=_selection_schema(
                unit_ids=tuple(unit.unit_id for unit in candidate_units),
                num_predict=resolved_num_predict,
                selection_mode=effective_selection_mode,
            ),
            num_predict=resolved_num_predict,
        )
        fragment = _coerce_fragment(
            parsed,
            candidate_units=candidate_units,
            selection_mode=effective_selection_mode,
        )
        return (fragment, len(candidate_units), effective_selection_mode)

    def _structure_chunk_with_retries(
        self,
        *,
        document: Document,
        chunk_text: str,
        encoder: tiktoken.Encoding,
        chunk_index: int,
        chunk_count: int,
        primary_num_predict: int,
        selection_mode: str,
    ) -> list[_StructuredChunkAttempt]:
        def _attempt(model_name: str, num_predict: int) -> _StructuredChunkAttempt:
            fragment, candidate_unit_count, effective_selection_mode = self._call_ollama(
                document=document,
                chunk_text=chunk_text,
                encoder=encoder,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                model_name=model_name,
                num_predict=num_predict,
                selection_mode=selection_mode,
            )
            grounded_result = _ground_fragment_against_input(
                fragment,
                chunk_text=chunk_text,
                encoder=encoder,
                source_name=document.source_name,
                chunk_index=chunk_index,
                rerank_profile_id=self._config.grounding_rerank_profile_id,
                grounding_span_tokens=self._config.grounding_span_tokens,
            )
            grounded_fragment = grounded_result.fragment
            _validate_fragment_against_input(grounded_fragment, chunk_text=chunk_text)
            return _StructuredChunkAttempt(
                chunk_text=chunk_text,
                model_name=model_name,
                fragment=grounded_fragment,
                grounding_candidate_span_count=grounded_result.candidate_span_count,
                grounding_sentence_requests=grounded_result.sentence_requests,
                grounding_unique_sentences=grounded_result.unique_sentences,
                grounding_rerank_calls=grounded_result.rerank_calls,
                grounding_cache_hits=grounded_result.cache_hits,
                grounding_salient_reuse_count=grounded_result.salient_reuse_count,
                selection_mode=effective_selection_mode,
                candidate_unit_count=int(candidate_unit_count),
                kept_unit_count=sum(len(section.sentences) for section in grounded_fragment.sections),
                dropped_unit_count=max(
                    0,
                    int(candidate_unit_count) - sum(len(section.sentences) for section in grounded_fragment.sections),
                ),
            )

        fallback_model_name = str(self._config.fallback_model_name or "").strip()
        attempt_order: list[tuple[str, int]] = []
        if _prefer_fallback_model_for_long_pdf(
            document=document,
            primary_num_predict=primary_num_predict,
            fallback_model_name=fallback_model_name,
        ) and fallback_model_name != self._config.model_name:
            attempt_order.append((fallback_model_name, self._config.fallback_num_predict))
            attempt_order.append((self._config.model_name, primary_num_predict))
        else:
            attempt_order.append((self._config.model_name, primary_num_predict))
            if fallback_model_name and fallback_model_name != self._config.model_name:
                attempt_order.append((fallback_model_name, self._config.fallback_num_predict))

        primary_error: DocumentStructurerResponseError | None = None
        for attempt_model_name, attempt_num_predict in attempt_order:
            try:
                return [
                    _attempt(
                        attempt_model_name,
                        attempt_num_predict,
                    )
                ]
            except DocumentStructurerTransportError:
                continue
            except DocumentStructurerResponseError as exc:
                primary_error = exc

        try:
            raise primary_error or DocumentStructurerResponseError(
                "Document structurer failed before the retry path could inspect the chunk."
            )
        except DocumentStructurerResponseError:
            chunk_tokens = _token_count(chunk_text, encoder)
            if chunk_tokens <= _MIN_RETRY_SPLIT_TOKENS:
                raise
            retry_budget = max(_MIN_RETRY_SPLIT_TOKENS, chunk_tokens // 2)
            subchunks = _chunk_text(
                chunk_text,
                encoder=encoder,
                max_input_tokens=retry_budget,
            )
            normalized_chunk = chunk_text.strip()
            if len(subchunks) <= 1 or tuple(subchunks) == (normalized_chunk,):
                raise
            results: list[_StructuredChunkAttempt] = []
            for subchunk_index, subchunk_text in enumerate(subchunks):
                results.extend(
                    self._structure_chunk_with_retries(
                        document=document,
                        chunk_text=subchunk_text,
                        encoder=encoder,
                        chunk_index=subchunk_index,
                        chunk_count=len(subchunks),
                        primary_num_predict=primary_num_predict,
                        selection_mode=selection_mode,
                    )
                )
            return results

    def _structure_documents_internal(
        self,
        documents: Sequence[Document],
        *,
        encoder: tiktoken.Encoding,
        capture_trace: bool,
    ) -> tuple[list[StructuredDocument], list[StructuredDocumentTrace]]:
        structured_documents: list[StructuredDocument] = []
        structured_traces: list[StructuredDocumentTrace] = []
        for document in documents:
            source_text = str(document.text or "").strip()
            if not source_text:
                continue
            chunk_token_budget, primary_num_predict = _runtime_profile_for_document(
                document,
                source_text=source_text,
                encoder=encoder,
                config=self._config,
            )
            if _long_pdf_requires_extractive_fast_path(
                document=document,
                source_text=source_text,
                encoder=encoder,
                primary_num_predict=primary_num_predict,
            ):
                structured_document, structured_trace = self._structure_long_pdf_extractive(
                    document=document,
                    source_text=source_text,
                    encoder=encoder,
                    primary_num_predict=primary_num_predict,
                    capture_trace=capture_trace,
                )
                structured_documents.append(structured_document)
                if capture_trace and structured_trace is not None:
                    structured_traces.append(structured_trace)
                continue
            chunks = _chunk_text(
                source_text,
                encoder=encoder,
                max_input_tokens=chunk_token_budget,
            )
            if not chunks:
                continue
            chunk_results: list[_StructuredChunkAttempt] = []
            chunk_traces: list[StructuredChunkTrace] = []
            selection_mode = self._selection_mode_for_document(
                document=document,
                primary_num_predict=primary_num_predict,
            )
            for chunk_index, chunk_text in enumerate(chunks):
                chunk_results.extend(
                    self._structure_chunk_with_retries(
                        document=document,
                        chunk_text=chunk_text,
                        encoder=encoder,
                        chunk_index=chunk_index,
                        chunk_count=len(chunks),
                        primary_num_predict=primary_num_predict,
                        selection_mode=selection_mode,
                    )
                )
            fragments = [attempt.fragment for attempt in chunk_results]
            final_chunk_count = len(chunk_results)
            if capture_trace:
                for final_chunk_index, attempt in enumerate(chunk_results):
                    fragment = attempt.fragment
                    chunk_traces.append(
                        StructuredChunkTrace(
                            chunk_index=final_chunk_index,
                            chunk_count=final_chunk_count,
                            model_name=attempt.model_name,
                            input_characters=len(attempt.chunk_text),
                            input_tokens=_token_count(attempt.chunk_text, encoder),
                            fragment_title=fragment.title,
                            section_count=len(fragment.sections),
                            sentence_count=sum(len(section.sentences) for section in fragment.sections),
                            salient_count=len(fragment.salient_sentences),
                            section_headings=tuple(section.heading for section in fragment.sections),
                            salient_sentences=tuple(fragment.salient_sentences),
                            grounding_candidate_span_count=attempt.grounding_candidate_span_count,
                            grounding_sentence_requests=attempt.grounding_sentence_requests,
                            grounding_unique_sentences=attempt.grounding_unique_sentences,
                            grounding_rerank_calls=attempt.grounding_rerank_calls,
                            grounding_cache_hits=attempt.grounding_cache_hits,
                            grounding_salient_reuse_count=attempt.grounding_salient_reuse_count,
                            selection_mode=attempt.selection_mode,
                            candidate_unit_count=attempt.candidate_unit_count,
                            kept_unit_count=attempt.kept_unit_count,
                            dropped_unit_count=attempt.dropped_unit_count,
                        )
                    )
            title = next((fragment.title for fragment in fragments if fragment.title), None)
            sections = tuple(
                section
                for fragment in fragments
                for section in fragment.sections
            )
            if not sections:
                raise RuntimeError(f"Document structurer produced no sections for {document.source_name}.")
            raw_salient_sentences = _dedupe_keep_order(
                sentence
                for fragment in fragments
                for sentence in fragment.salient_sentences
            )
            section_sentences = _dedupe_keep_order(
                sentence
                for section in sections
                for sentence in section.sentences
            )
            if _use_high_recall_long_pdf_mode(
                document=document,
                primary_num_predict=primary_num_predict,
            ):
                global_selection = _DocumentGlobalSelectionResult(
                    salient_sentences=raw_salient_sentences or section_sentences,
                    candidate_count=len(raw_salient_sentences or section_sentences),
                    selected_count=len(raw_salient_sentences or section_sentences),
                    model_name=None,
                )
            else:
                salience_mode = self._select_document_salience_mode(
                    document=document,
                    raw_salient_sentences=raw_salient_sentences,
                    section_sentences=section_sentences,
                    primary_num_predict=primary_num_predict,
                )
                if salience_mode.salience_mode == _DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL and raw_salient_sentences:
                    global_selection = _DocumentGlobalSelectionResult(
                        salient_sentences=raw_salient_sentences,
                        candidate_count=len(raw_salient_sentences),
                        selected_count=len(raw_salient_sentences),
                        model_name=(
                            salience_mode.model_name
                            if salience_mode.model_name is not None
                            else _DOCUMENT_SALIENCE_MODE_CHUNK_LOCAL
                        ),
                    )
                else:
                    global_selection = _DocumentGlobalSelectionResult(
                        salient_sentences=section_sentences,
                        candidate_count=len(section_sentences),
                        selected_count=len(section_sentences),
                        model_name=(
                            salience_mode.model_name
                            if salience_mode.model_name is not None
                            else _DOCUMENT_SALIENCE_MODE_SECTION_KEEP_ALL
                        ),
                    )
            salient_sentences = global_selection.salient_sentences
            rendered_text = _render_structured_text(title, sections)
            if not rendered_text:
                raise RuntimeError(f"Document structurer produced no rendered text for {document.source_name}.")
            structured_documents.append(
                StructuredDocument(
                    source_document=Document(
                        source_id=document.source_id,
                        source_name=document.source_name,
                        text=rendered_text,
                        origin_path=document.origin_path,
                        metadata=dict(document.metadata),
                    ),
                    rendered_text=rendered_text,
                    sections=sections,
                    salient_sentences=salient_sentences,
                )
            )
            if capture_trace:
                structured_traces.append(
                    build_structured_document_trace(
                        structured_documents[-1],
                        encoder=encoder,
                        input_text=source_text,
                        chunk_traces=chunk_traces,
                        global_salient_candidate_count=global_selection.candidate_count,
                        global_salient_selected_count=global_selection.selected_count,
                        global_salient_model_name=global_selection.model_name,
                    )
                )
        return structured_documents, structured_traces

    def structure_documents(
        self,
        documents: Sequence[Document],
        *,
        encoder: tiktoken.Encoding,
    ) -> list[StructuredDocument]:
        structured_documents, _ = self._structure_documents_internal(
            documents,
            encoder=encoder,
            capture_trace=False,
        )
        return structured_documents

    def structure_documents_with_trace(
        self,
        documents: Sequence[Document],
        *,
        encoder: tiktoken.Encoding,
    ) -> tuple[list[StructuredDocument], list[StructuredDocumentTrace]]:
        """Return structured documents plus a detailed trace payload."""

        return self._structure_documents_internal(
            documents,
            encoder=encoder,
            capture_trace=True,
        )


def build_local_document_structurer_from_env(
    *,
    model_name: str | None = None,
    fallback_model_name: str | None = None,
    ollama_base_url: str | None = None,
    timeout_seconds: float | None = None,
    max_input_tokens: int | None = None,
    num_predict: int | None = None,
    fallback_num_predict: int | None = None,
    grounding_rerank_profile_id: str | None = None,
    grounding_span_tokens: int | None = None,
    salience_embedding_model_name: str | None = None,
    selection_mode: str | None = None,
    env: Mapping[str, str] | None = None,
) -> LocalOllamaDocumentStructurer:
    """Build the local Ollama document structurer from env plus overrides."""

    config = document_structurer_config_from_env(
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        ollama_base_url=ollama_base_url,
        timeout_seconds=timeout_seconds,
        max_input_tokens=max_input_tokens,
        num_predict=num_predict,
        fallback_num_predict=fallback_num_predict,
        grounding_rerank_profile_id=grounding_rerank_profile_id,
        grounding_span_tokens=grounding_span_tokens,
        salience_embedding_model_name=salience_embedding_model_name,
        selection_mode=selection_mode,
        env=env,
    )
    return LocalOllamaDocumentStructurer(config)


@lru_cache(maxsize=1)
def _load_winner_preparation_recovery_benchmark() -> Any:
    """Load the canonical handoff recovery benchmark helpers for the proven winner preparation."""

    module = sys.modules.get(_WINNER_PREPARATION_RECOVERY_BENCHMARK_MODULE)
    if module is not None:
        return module
    benchmark_path = (
        Path(__file__).resolve().parents[6]
        / "benchmarks"
        / "document_compression"
        / "benchmark_document_compression_recovery.py"
    )
    spec = importlib.util.spec_from_file_location(
        _WINNER_PREPARATION_RECOVERY_BENCHMARK_MODULE,
        benchmark_path,
    )
    if spec is None or spec.loader is None:
        raise DocumentStructurerResponseError(
            f"Winner preparation could not load recovery benchmark helpers from {benchmark_path}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_WINNER_PREPARATION_RECOVERY_BENCHMARK_MODULE] = module
    spec.loader.exec_module(module)
    return module


class WinnerPreparationDocumentStructurer:
    """Exact handoff-winner preparation path for long PDF documents."""

    def __init__(self) -> None:
        self.name = "winner-preparation-document"

    def _build_sections(
        self,
        source_text: str,
        *,
        encoder: tiktoken.Encoding,
    ) -> tuple[StructuredSection, ...]:
        recovery_benchmark = _load_winner_preparation_recovery_benchmark()
        raw_sections = recovery_benchmark._sectionize_text(  # noqa: SLF001
            source_text,
            token_counter=lambda value: _token_count(str(value or ""), encoder),
            max_span_tokens=int(DEFAULT_DOCUMENT_GROUNDING_SPAN_TOKENS),
        )
        sections = [
            StructuredSection(
                heading=None if str(heading).strip() == "Preamble" else str(heading).strip() or None,
                sentences=tuple(str(sentence).strip() for sentence in sentences if str(sentence).strip()),
            )
            for heading, sentences in raw_sections
        ]
        filtered = tuple(section for section in sections if section.sentences)
        if filtered:
            return filtered
        fallback_sentences = tuple(
            recovery_benchmark._split_sentences(  # noqa: SLF001
                source_text,
                token_counter=lambda value: _token_count(str(value or ""), encoder),
                max_span_tokens=int(DEFAULT_DOCUMENT_GROUNDING_SPAN_TOKENS),
            )
        )
        if not fallback_sentences:
            raise DocumentStructurerResponseError("Winner preparation derived no candidate sentences.")
        return (StructuredSection(heading=None, sentences=fallback_sentences),)

    def _build_salient_sentences(
        self,
        source_text: str,
        *,
        sections: Sequence[StructuredSection],
        encoder: tiktoken.Encoding,
    ) -> tuple[str, ...]:
        recovery_benchmark = _load_winner_preparation_recovery_benchmark()
        raw_salient = tuple(
            recovery_benchmark._build_key_sentences(  # noqa: SLF001
                source_text,
                limit=int(DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT),
                token_counter=lambda value: _token_count(str(value or ""), encoder),
                max_span_tokens=int(DEFAULT_DOCUMENT_GROUNDING_SPAN_TOKENS),
            )
        )
        if raw_salient:
            return raw_salient
        return tuple(sentence for section in sections for sentence in section.sentences)[: int(DEFAULT_DOCUMENT_GLOBAL_SALIENT_LIMIT)]

    def structure_documents(
        self,
        documents: Sequence[Document],
        *,
        encoder: tiktoken.Encoding,
    ) -> list[StructuredDocument]:
        structured_documents, _ = self.structure_documents_with_trace(documents, encoder=encoder)
        return structured_documents

    def structure_documents_with_trace(
        self,
        documents: Sequence[Document],
        *,
        encoder: tiktoken.Encoding,
    ) -> tuple[list[StructuredDocument], list[StructuredDocumentTrace]]:
        structured_documents: list[StructuredDocument] = []
        traces: list[StructuredDocumentTrace] = []
        for document in documents:
            source_text = str(document.text or "").strip()
            if not source_text:
                continue
            sections = self._build_sections(source_text, encoder=encoder)
            salient_sentences = self._build_salient_sentences(
                source_text,
                sections=sections,
                encoder=encoder,
            )
            structured_document = StructuredDocument(
                source_document=Document(
                    source_id=document.source_id,
                    source_name=document.source_name,
                    text=source_text,
                    origin_path=document.origin_path,
                    metadata=dict(document.metadata),
                ),
                rendered_text=source_text,
                sections=sections,
                salient_sentences=salient_sentences,
            )
            structured_documents.append(structured_document)
            traces.append(
                build_structured_document_trace(
                    structured_document,
                    encoder=encoder,
                    input_text=source_text,
                )
            )
        return structured_documents, traces


class AdaptiveDocumentStructurer:
    """Use the proven winner preparation for PDFs and the local model elsewhere."""

    def __init__(
        self,
        *,
        local_structurer: DocumentStructurer,
        long_pdf_structurer: DocumentStructurer,
        long_pdf_trigger_tokens: int = _LONG_PDF_TRIGGER_TOKENS,
    ) -> None:
        self._local_structurer = local_structurer
        self._long_pdf_structurer = long_pdf_structurer
        self._long_pdf_trigger_tokens = int(long_pdf_trigger_tokens)
        self.name = (
            f"adaptive-document-structurer:{getattr(local_structurer, 'name', type(local_structurer).__name__)}"
            "+winner-pdf"
        )

    def _use_long_pdf_winner_path(
        self,
        document: Document,
        *,
        encoder: tiktoken.Encoding,
    ) -> bool:
        del encoder
        source_format = str(document.metadata.get("format", "")).strip().lower()
        source_text = str(document.text or "").strip()
        return source_format == "pdf" and bool(source_text)

    def structure_documents(
        self,
        documents: Sequence[Document],
        *,
        encoder: tiktoken.Encoding,
    ) -> list[StructuredDocument]:
        structured_documents, _ = self.structure_documents_with_trace(documents, encoder=encoder)
        return structured_documents

    def structure_documents_with_trace(
        self,
        documents: Sequence[Document],
        *,
        encoder: tiktoken.Encoding,
    ) -> tuple[list[StructuredDocument], list[StructuredDocumentTrace]]:
        structured_documents: list[StructuredDocument] = []
        structured_traces: list[StructuredDocumentTrace] = []
        for document in documents:
            active_structurer = (
                self._long_pdf_structurer
                if self._use_long_pdf_winner_path(document, encoder=encoder)
                else self._local_structurer
            )
            trace_method = getattr(active_structurer, "structure_documents_with_trace", None)
            if callable(trace_method):
                partial_documents, partial_traces = trace_method([document], encoder=encoder)
            else:
                partial_documents = active_structurer.structure_documents([document], encoder=encoder)
                partial_traces = [
                    build_structured_document_trace(
                        structured_document,
                        encoder=encoder,
                        input_text=document.text,
                    )
                    for structured_document in partial_documents
                ]
            structured_documents.extend(partial_documents)
            structured_traces.extend(partial_traces)
        return structured_documents, structured_traces


def build_default_document_structurer_from_env(
    *,
    model_name: str | None = None,
    fallback_model_name: str | None = None,
    ollama_base_url: str | None = None,
    timeout_seconds: float | None = None,
    max_input_tokens: int | None = None,
    num_predict: int | None = None,
    fallback_num_predict: int | None = None,
    grounding_rerank_profile_id: str | None = None,
    grounding_span_tokens: int | None = None,
    salience_embedding_model_name: str | None = None,
    selection_mode: str | None = None,
    env: Mapping[str, str] | None = None,
) -> AdaptiveDocumentStructurer:
    """Build the production default structurer: winner-preparation for PDFs, local AI elsewhere."""

    local_structurer = build_local_document_structurer_from_env(
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        ollama_base_url=ollama_base_url,
        timeout_seconds=timeout_seconds,
        max_input_tokens=max_input_tokens,
        num_predict=num_predict,
        fallback_num_predict=fallback_num_predict,
        grounding_rerank_profile_id=grounding_rerank_profile_id,
        grounding_span_tokens=grounding_span_tokens,
        salience_embedding_model_name=salience_embedding_model_name,
        selection_mode=selection_mode,
        env=env,
    )
    return AdaptiveDocumentStructurer(
        local_structurer=local_structurer,
        long_pdf_structurer=WinnerPreparationDocumentStructurer(),
    )
