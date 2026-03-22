"""Exact CPC/MMR winner-path compaction engine for chonkify."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import tiktoken

from chonkify.config import AzureEmbeddingConfig
from chonkify.types import (
    CompressionRequest,
    CompressionResult,
    CompressionUnit,
    Document,
    EmbeddingProvider,
    SelectedUnit,
)

EMBEDDING_DIMENSIONS = 768


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _token_count(text: str, encoder: tiktoken.Encoding) -> int:
    """Count tokens in *text* using the given tiktoken encoder."""
    return len(encoder.encode(text))


def _trim_to_token_budget(text: str, encoder: tiktoken.Encoding, budget: int) -> tuple[str, bool]:
    """Trim *text* to at most *budget* tokens. Returns (trimmed_text, was_truncated)."""
    tokens = encoder.encode(text)
    if len(tokens) <= budget:
        return text, False
    return encoder.decode(tokens[:budget]), True


# ---------------------------------------------------------------------------
# Text normalization / cleaning
# ---------------------------------------------------------------------------

_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def _normalize_keep_lines(text: str) -> str:
    """Collapse runs of blank lines but preserve meaningful single newlines."""
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _split_digit_run_if_axis_ticks(text: str) -> str:
    """Split smashed chart-axis digit runs when the structure is obvious."""

    def is_monotone_small_step(nums: list[int]) -> bool:
        if len(nums) < 3:
            return False
        return all(
            0 < nums[i + 1] - nums[i] <= nums[1] - nums[0] + 1
            for i in range(len(nums) - 1)
        )

    def _try_split(match: re.Match[str]) -> str:
        run = match.group(0)
        # Try splitting into 1-digit, 2-digit, 3-digit, 4-digit chunks
        for width in (1, 2, 3, 4):
            if len(run) % width != 0:
                continue
            chunks = [run[i:i + width] for i in range(0, len(run), width)]
            nums = [int(c) for c in chunks]
            if is_monotone_small_step(nums):
                return " ".join(chunks)
        return run

    return re.sub(r"\d{4,}", _try_split, text)


_PROTECTED_TOKEN_RE = re.compile(
    r"(?:"
    r"\d+\.\d+"            # decimal numbers
    r"|[A-Z]{2,}"          # acronyms
    r"|https?://\S+"       # URLs
    r"|\S+@\S+"            # emails
    r"|v\d+\.\d+"          # version strings
    r"|\d{1,2}[-/]\d{1,2}" # dates
    r")"
)


def _desquash_line_token_guarded(line: str, encoder: tiktoken.Encoding, max_tokens: int) -> str:
    """Re-insert spaces in lines that are smashed together, respecting protected tokens."""

    def is_protected_token(word: str) -> bool:
        return bool(_PROTECTED_TOKEN_RE.fullmatch(word))

    def digit_run_repl(match: re.Match[str]) -> str:
        return _split_digit_run_if_axis_ticks(match.group(0))

    line = re.sub(r"\d{6,}", digit_run_repl, line)

    if _token_count(line, encoder) <= max_tokens:
        return line

    # Try splitting camelCase / PascalCase smashes
    parts = re.split(r"(?<=[a-z])(?=[A-Z])", line)
    result = " ".join(parts)
    if _token_count(result, encoder) <= max_tokens:
        return result

    return line


_SHORT_LINE_HEADING_RE = re.compile(r"^(?:table|figure|fig\.|appendix|section|chapter)\s", re.IGNORECASE)
_SHORT_LINE_NUMERIC_RE = re.compile(r"^\d+[\.\)]\s")


def _clean_pdf_text(text: str) -> str:
    """Clean extracted PDF text: fix squashed lines, collapse whitespace runs."""

    def is_high_signal_short_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if len(stripped) < 4:
            return False
        if _SHORT_LINE_HEADING_RE.match(stripped):
            return True
        if _SHORT_LINE_NUMERIC_RE.match(stripped):
            return True
        return len(stripped) >= 20

    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if is_high_signal_short_line(stripped):
            kept.append(stripped)
        elif len(stripped) >= 20:
            kept.append(stripped)

    return _normalize_keep_lines("\n".join(kept))


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+\.)+\s+\S")
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s]{4,}$")


def _is_heading_line(line: str) -> bool:
    """Return True if *line* looks like a section heading."""
    stripped = line.strip()
    if not stripped:
        return False
    if _MD_HEADING_RE.match(stripped):
        return True
    if _NUMBERED_HEADING_RE.match(stripped):
        return True
    if _ALLCAPS_HEADING_RE.match(stripped):
        return True
    # Short ALL-CAPS line after blank
    if len(stripped) < 60 and stripped == stripped.upper() and any(c.isalpha() for c in stripped):
        return True
    return False


def _should_use_numbered_headings(text: str) -> bool:
    """Detect whether the document uses numbered heading style (1. / 1.1. / etc.)."""
    lines = text.split("\n")
    numbered_count = sum(1 for line in lines if _NUMBERED_HEADING_RE.match(line.strip()))
    return numbered_count >= 2


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])"       # lookbehind for sentence-ending punctuation
    r'(?:\s*["\')\]])*'  # optional trailing quotes/parens
    r"\s+"               # whitespace separator
    r"(?=[A-Z\d\"\'(\[])" # lookahead for new sentence start
)


def _split_words_to_token_windows(
    words: list[str],
    encoder: tiktoken.Encoding,
    max_span_tokens: int,
) -> list[str]:
    """Split a word list into spans that each fit within *max_span_tokens*."""
    if not words:
        return []
    spans: list[str] = []
    current_words: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = _token_count(word, encoder)
        if current_words and current_tokens + 1 + word_tokens > max_span_tokens:
            spans.append(" ".join(current_words))
            current_words = []
            current_tokens = 0
        current_words.append(word)
        current_tokens += word_tokens + (1 if len(current_words) > 1 else 0)
    if current_words:
        spans.append(" ".join(current_words))
    return spans


def _split_sentences(
    text: str,
    encoder: tiktoken.Encoding,
    max_span_tokens: int = 126,
) -> list[str]:
    """Split text into sentence-level spans respecting the token budget."""
    if not text.strip():
        return []

    raw_sentences = _SENTENCE_END_RE.split(text.strip())

    def split_overlong(sentence: str) -> list[str]:
        if _token_count(sentence, encoder) <= max_span_tokens:
            return [sentence]
        words = sentence.split()
        return _split_words_to_token_windows(words, encoder, max_span_tokens)

    result: list[str] = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        result.extend(split_overlong(sentence))
    return result


# ---------------------------------------------------------------------------
# Sectionizing
# ---------------------------------------------------------------------------

def _sectionize_text(
    text: str,
    encoder: tiktoken.Encoding,
    max_span_tokens: int = 126,
) -> list[tuple[str, list[str]]]:
    """Split text into (heading, sentences) sections."""
    use_numbered = _should_use_numbered_headings(text)

    def is_heading(line: str) -> bool:
        stripped = line.strip()
        if _MD_HEADING_RE.match(stripped):
            return True
        if use_numbered and _NUMBERED_HEADING_RE.match(stripped):
            return True
        if _ALLCAPS_HEADING_RE.match(stripped):
            return True
        return False

    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sentences = _split_sentences(body, encoder, max_span_tokens)
            if sentences:
                sections.append((current_heading, sentences))
        current_lines = []

    def split_overlong(sentence: str) -> list[str]:
        if _token_count(sentence, encoder) <= max_span_tokens:
            return [sentence]
        words = sentence.split()
        return _split_words_to_token_windows(words, encoder, max_span_tokens)

    for line in lines:
        if is_heading(line.strip()):
            flush()
            current_heading = line.strip()
        else:
            current_lines.append(line)

    flush()

    if not sections and text.strip():
        sentences = _split_sentences(text, encoder, max_span_tokens)
        if sentences:
            sections.append(("", sentences))

    return sections


# ---------------------------------------------------------------------------
# Key sentences
# ---------------------------------------------------------------------------

def _build_key_sentences(
    sections: list[tuple[str, list[str]]],
    encoder: tiktoken.Encoding,
) -> list[str]:
    """Extract key sentences: first sentence of each section, longest sentences overall."""
    key: list[str] = []

    for heading, sentences in sections:
        if sentences:
            key.append(sentences[0])

    all_sentences = [s for _, sents in sections for s in sents]
    if all_sentences:
        by_length = sorted(all_sentences, key=lambda s: _token_count(s, encoder), reverse=True)
        for sentence in by_length[:max(3, len(sections))]:
            if sentence not in key:
                key.append(sentence)

    return key


# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------

def _normalize_vector(vec: list[float]) -> list[float]:
    """Return the L2-normalized copy of *vec*."""
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude == 0.0:
        return vec
    return [v / magnitude for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (assumed to be normalized or raw)."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Document text preparation
# ---------------------------------------------------------------------------

def _prepare_document_text(document: Document, encoder: tiktoken.Encoding, max_unit_tokens: int) -> str:
    """Clean and normalize document text for the compression pipeline."""
    text = document.text

    # PDF-specific cleaning
    fmt = document.metadata.get("format", "")
    if fmt == "pdf" or (document.source_name and document.source_name.lower().endswith(".pdf")):
        text = _clean_pdf_text(text)

    text = _normalize_keep_lines(text)
    text = _split_digit_run_if_axis_ticks(text)

    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        cleaned = _desquash_line_token_guarded(line, encoder, max_unit_tokens)
        cleaned_lines.append(cleaned)

    return _normalize_keep_lines("\n".join(cleaned_lines))


# ---------------------------------------------------------------------------
# Azure OpenAI embedding provider
# ---------------------------------------------------------------------------

class AzureOpenAIEmbeddingProvider:
    """Embedding provider backed by the Azure OpenAI service."""

    name: str

    def __init__(self, config: AzureEmbeddingConfig) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for Azure OpenAI embeddings."
            ) from exc

        self._config = config
        self._client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            timeout=float(config.timeout_seconds),
        )
        self.name = f"azure:{config.model_label or config.deployment}"

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._config.max_batch_size):
            batch = list(texts[start : start + self._config.max_batch_size])
            response = self._client.embeddings.create(
                model=self._config.deployment,
                input=batch,
                dimensions=self._config.dimensions,
            )
            ordered = sorted(response.data, key=lambda item: int(item.index))
            for item in ordered:
                vec = item.embedding
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                if len(vec) != self._config.dimensions:
                    raise ValueError(
                        f"Azure embedding dimension mismatch: "
                        f"expected {self._config.dimensions}, got {len(vec)}"
                    )
                vectors.append([float(v) for v in vec])

        if len(vectors) != len(texts):
            raise ValueError(
                f"Azure returned {len(vectors)} vectors for {len(texts)} texts."
            )
        return vectors


# ---------------------------------------------------------------------------
# CPC candidate unit construction
# ---------------------------------------------------------------------------

def _build_cpc_candidate_units(
    documents: list[Document],
    encoder: tiktoken.Encoding,
    max_unit_tokens: int,
) -> list[CompressionUnit]:
    """Build candidate units from documents using sectionized sentence splitting."""
    units: list[CompressionUnit] = []
    global_unit_index = 0

    for doc_index, document in enumerate(documents):
        prepared_text = _prepare_document_text(document, encoder, max_unit_tokens)
        sections = _sectionize_text(prepared_text, encoder, max_unit_tokens)

        for block_index, (heading, sentences) in enumerate(sections):
            for chunk_index, sentence in enumerate(sentences):
                token_count = _token_count(sentence, encoder)
                if token_count == 0:
                    continue
                unit = CompressionUnit(
                    unit_id=f"{document.source_id}:b{block_index}:c{chunk_index}",
                    source_id=document.source_id,
                    source_name=document.source_name,
                    text=sentence,
                    token_count=token_count,
                    document_index=doc_index,
                    unit_index=global_unit_index,
                    block_index=block_index,
                    chunk_index=chunk_index,
                )
                units.append(unit)
                global_unit_index += 1

    return units


# ---------------------------------------------------------------------------
# CPC/MMR selection
# ---------------------------------------------------------------------------

def _select_cpc_mmr_units(
    units: list[CompressionUnit],
    unit_vectors: list[list[float]],
    key_vectors: list[list[float]],
    token_budget: int,
    lambda_relevance: float,
    encoder: tiktoken.Encoding,
) -> list[SelectedUnit]:
    """Select units using the CPC/MMR algorithm under a strict token budget."""
    if not units or not unit_vectors:
        return []

    # Normalize all vectors
    normed_units = [_normalize_vector(v) for v in unit_vectors]
    normed_keys = [_normalize_vector(v) for v in key_vectors] if key_vectors else []

    # Compute relevance scores: max cosine similarity to any key sentence
    relevance_scores: list[float] = []
    for unit_vec in normed_units:
        if normed_keys:
            max_sim = max(_cosine(unit_vec, kv) for kv in normed_keys)
        else:
            max_sim = 0.0
        relevance_scores.append(max_sim)

    selected: list[int] = []
    selection_order: list[int] = []
    used_tokens = 0
    newline_cost = _token_count("\n\n", encoder)
    remaining = set(range(len(units)))

    while remaining:
        best_index = -1
        best_score = -float("inf")

        for idx in remaining:
            unit = units[idx]
            cost = unit.token_count + (newline_cost if selected else 0)
            if used_tokens + cost > token_budget:
                continue

            cand_relevance = relevance_scores[idx]

            # Compute max redundancy with already selected units
            max_redundancy = 0.0
            cand_norm = normed_units[idx]
            for sel_idx in selected:
                sim = _cosine(cand_norm, normed_units[sel_idx])
                if sim > max_redundancy:
                    max_redundancy = sim

            # CPC/MMR objective: balance relevance vs redundancy
            objective = lambda_relevance * cand_relevance - (1.0 - lambda_relevance) * max_redundancy

            if objective > best_score:
                best_score = objective
                best_index = idx

        if best_index < 0:
            break

        unit = units[best_index]
        cost = unit.token_count + (newline_cost if selected else 0)
        used_tokens += cost
        selected.append(best_index)
        selection_order.append(best_index)
        remaining.discard(best_index)

    # Build SelectedUnit results in document order for output, but record selection rank
    result: list[SelectedUnit] = []
    for rank, idx in enumerate(selection_order):
        unit = units[idx]
        result.append(
            SelectedUnit(
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                source_name=unit.source_name,
                text=unit.text,
                token_count=unit.token_count,
                document_index=unit.document_index,
                unit_index=unit.unit_index,
                selection_rank=rank + 1,
                relevance_score=relevance_scores[idx],
                redundancy_score=0.0,
                objective_score=best_score if rank == len(selection_order) - 1 else 0.0,
            )
        )

    # Re-score redundancy and objective for each selected unit
    for i, sel_unit_idx in enumerate(selection_order):
        cand_norm = normed_units[sel_unit_idx]
        max_red = 0.0
        for j, other_idx in enumerate(selection_order):
            if i == j:
                continue
            sim = _cosine(cand_norm, normed_units[other_idx])
            if sim > max_red:
                max_red = sim
        obj = lambda_relevance * relevance_scores[sel_unit_idx] - (1.0 - lambda_relevance) * max_red
        old = result[i]
        result[i] = SelectedUnit(
            unit_id=old.unit_id,
            source_id=old.source_id,
            source_name=old.source_name,
            text=old.text,
            token_count=old.token_count,
            document_index=old.document_index,
            unit_index=old.unit_index,
            selection_rank=old.selection_rank,
            relevance_score=old.relevance_score,
            redundancy_score=max_red,
            objective_score=obj,
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_documents(
    documents: Sequence[Document],
    request: CompressionRequest,
    provider: EmbeddingProvider,
) -> CompressionResult:
    """Compress one or more documents using the benchmark-winning CPC/MMR path."""
    encoder = tiktoken.get_encoding(request.encoding_name)
    doc_list = list(documents)

    # Count original tokens
    original_tokens = sum(_token_count(doc.text, encoder) for doc in doc_list)

    # Build candidate units
    units = _build_cpc_candidate_units(doc_list, encoder, request.max_unit_tokens)
    if not units:
        return CompressionResult(
            strategy="cpc_mmr/key_sentence_mmr",
            provider_name=provider.name,
            request=request,
            documents=doc_list,
            selected_units=[],
            text="",
            original_tokens=original_tokens,
            compressed_tokens=0,
            units_considered=0,
            final_output_truncated=False,
        )

    # Build key sentences for relevance scoring
    all_sections: list[tuple[str, list[str]]] = []
    for doc in doc_list:
        prepared = _prepare_document_text(doc, encoder, request.max_unit_tokens)
        sections = _sectionize_text(prepared, encoder, request.max_unit_tokens)
        all_sections.extend(sections)
    key_sentences = _build_key_sentences(all_sections, encoder)

    # Embed all candidate units and key sentences
    unit_texts = [u.text for u in units]
    all_texts = unit_texts + key_sentences
    all_vectors = provider.embed_texts(all_texts)

    unit_vectors = all_vectors[: len(units)]
    key_vectors = all_vectors[len(units) :]

    # Run CPC/MMR selection
    selected_units = _select_cpc_mmr_units(
        units=units,
        unit_vectors=unit_vectors,
        key_vectors=key_vectors,
        token_budget=request.target_tokens,
        lambda_relevance=request.lambda_relevance,
        encoder=encoder,
    )

    # Sort selected units by document order for coherent output
    ordered = sorted(selected_units, key=lambda u: (u.document_index, u.unit_index))

    # Assemble output text
    output_text = "\n\n".join(u.text for u in ordered)

    # Final trim if needed
    output_text, was_truncated = _trim_to_token_budget(output_text, encoder, request.target_tokens)
    compressed_tokens = _token_count(output_text, encoder)

    return CompressionResult(
        strategy="cpc_mmr/key_sentence_mmr",
        provider_name=provider.name,
        request=request,
        documents=doc_list,
        selected_units=ordered,
        text=output_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        units_considered=len(units),
        final_output_truncated=was_truncated,
    )
