"""Core data contracts for chonkify."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Embedding provider contract used by the semantic selector."""

    name: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text, preserving order."""


@dataclass(frozen=True, slots=True)
class Document:
    """A source document presented to the compaction engine."""

    source_id: str
    source_name: str
    text: str
    origin_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Return compact metadata without echoing the full text payload."""

        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "origin_path": str(self.origin_path) if self.origin_path is not None else None,
            "metadata": dict(self.metadata),
            "character_count": len(self.text),
        }


@dataclass(frozen=True, slots=True)
class CompressionRequest:
    """User-facing request contract for a compression run."""

    target_tokens: int
    query: str | None = None
    max_unit_tokens: int = 126
    lambda_relevance: float = 0.75
    encoding_name: str = "o200k_base"


@dataclass(frozen=True, slots=True)
class CompressionUnit:
    """A structurally derived selection unit inside a document."""

    unit_id: str
    source_id: str
    source_name: str
    text: str
    token_count: int
    document_index: int
    unit_index: int
    block_index: int
    chunk_index: int


@dataclass(frozen=True, slots=True)
class SelectedUnit:
    """A selected output unit with trace metadata."""

    unit_id: str
    source_id: str
    source_name: str
    text: str
    token_count: int
    document_index: int
    unit_index: int
    selection_rank: int
    relevance_score: float
    redundancy_score: float
    objective_score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly trace payload."""

        return {
            "unit_id": self.unit_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "text": self.text,
            "token_count": self.token_count,
            "document_index": self.document_index,
            "unit_index": self.unit_index,
            "selection_rank": self.selection_rank,
            "relevance_score": self.relevance_score,
            "redundancy_score": self.redundancy_score,
            "objective_score": self.objective_score,
        }


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """Structured result of a compaction run."""

    strategy: str
    provider_name: str
    request: CompressionRequest
    documents: list[Document]
    selected_units: list[SelectedUnit]
    text: str
    original_tokens: int
    compressed_tokens: int
    units_considered: int
    final_output_truncated: bool = False

    @property
    def compression_factor(self) -> float:
        """Return original/compressed token ratio."""

        if self.compressed_tokens <= 0:
            return 0.0
        return float(self.original_tokens) / float(self.compressed_tokens)

    @property
    def token_reduction_pct(self) -> float:
        """Return fractional token reduction in percent."""

        if self.original_tokens <= 0:
            return 0.0
        return 100.0 * (1.0 - (float(self.compressed_tokens) / float(self.original_tokens)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result payload."""

        return {
            "strategy": self.strategy,
            "provider_name": self.provider_name,
            "request": {
                "target_tokens": self.request.target_tokens,
                "query": self.request.query,
                "max_unit_tokens": self.request.max_unit_tokens,
                "lambda_relevance": self.request.lambda_relevance,
                "encoding_name": self.request.encoding_name,
            },
            "documents": [document.describe() for document in self.documents],
            "selected_units": [unit.to_dict() for unit in self.selected_units],
            "text": self.text,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_factor": self.compression_factor,
            "token_reduction_pct": self.token_reduction_pct,
            "units_considered": self.units_considered,
            "units_selected": len(self.selected_units),
            "final_output_truncated": self.final_output_truncated,
        }
