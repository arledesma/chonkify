"""Document ingestion and output helpers for chonkify."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from chonkify.types import Document


def _resolve_format(path: str, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    if path == "-":
        return "text"
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return "text"


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf_file(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to read PDF inputs.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip(), {"page_count": len(reader.pages)}


def load_documents(
    inputs: Sequence[str],
    *,
    input_format: str = "auto",
) -> list[Document]:
    """Load source documents from files or stdin."""

    documents: list[Document] = []
    for index, raw_input in enumerate(inputs):
        resolved_format = _resolve_format(raw_input, input_format)
        source_id = f"doc-{index:03d}"

        if raw_input == "-":
            text = sys.stdin.read()
            documents.append(
                Document(
                    source_id=source_id,
                    source_name="stdin",
                    text=text,
                    origin_path=None,
                    metadata={"format": resolved_format},
                )
            )
            continue

        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        metadata: dict[str, Any] = {"format": resolved_format}
        if resolved_format == "pdf":
            text, extra_metadata = _read_pdf_file(path)
            metadata.update(extra_metadata)
        else:
            text = _read_text_file(path)

        documents.append(
            Document(
                source_id=source_id,
                source_name=path.name,
                text=text,
                origin_path=path,
                metadata=metadata,
            )
        )
    return documents


def write_text_output(destination: str, text: str) -> None:
    """Write text output to stdout or a file."""

    if destination == "-":
        sys.stdout.write(text)
        if text and not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else f"{text}\n", encoding="utf-8")


def write_json_output(destination: str, payload: dict[str, Any]) -> None:
    """Write a JSON payload to disk."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
