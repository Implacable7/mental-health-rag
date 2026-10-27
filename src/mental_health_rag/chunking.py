"""Deterministic Chinese-aware text chunking for cleaned documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ("\n", "。", "！", "？", "；", "，", " ", "")
_CHUNKING_METHOD = "recursive_character"
_CHUNKING_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configure character-based recursive splitting."""

    chunk_size: int = 500
    chunk_overlap: int = 80
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be at least 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.separators or self.separators[-1] != "":
            raise ValueError("separators must end with an empty-string fallback")


@dataclass(frozen=True, slots=True)
class DocumentChunkSummary:
    """Summarize the chunks produced for one source document."""

    document_id: str
    source: str | None
    chunk_count: int


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    """Contain a stable batch of chunks and document-level counts."""

    chunks: tuple[Document, ...]
    documents: tuple[DocumentChunkSummary, ...]
    input_count: int

    @property
    def chunk_count(self) -> int:
        """Return the total number of chunks in the batch."""
        return len(self.chunks)


@dataclass(frozen=True, slots=True)
class ChunkBuildResult:
    """Describe files and stable identities produced by one chunk build."""

    input_count: int
    chunk_count: int
    output_path: Path
    report_path: Path
    input_sha256: str
    output_sha256: str
    build_id: str


def split_document(
    document: Document,
    config: ChunkingConfig | None = None,
) -> tuple[Document, ...]:
    """Split one cleaned document while preserving source lineage.

    The splitter prefers line and Chinese sentence boundaries, then falls back
    to smaller separators. Chunk text is copied from the cleaned source;
    it is never summarized or rewritten.
    """
    if not isinstance(document, Document):
        raise TypeError("document must be a LangChain Document")

    content = document.page_content
    if not content.strip():
        raise ValueError("document content must not be empty")

    cleaned_content_hash = document.metadata.get("cleaned_content_hash")
    if not isinstance(cleaned_content_hash, str) or not cleaned_content_hash:
        raise ValueError("document has no cleaned_content_hash; clean it before chunking")

    calculated_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if cleaned_content_hash != calculated_hash:
        raise ValueError("document has a stale cleaned_content_hash")

    document_id = document.metadata.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document metadata must contain a non-empty document_id")

    resolved_config = config or ChunkingConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=resolved_config.chunk_size,
        chunk_overlap=resolved_config.chunk_overlap,
        separators=list(resolved_config.separators),
        keep_separator="end",
        length_function=len,
        add_start_index=True,
        strip_whitespace=True,
    )
    split_documents = splitter.create_documents([content], [dict(document.metadata)])
    chunk_count = len(split_documents)
    chunks: list[Document] = []

    for chunk_index, split_chunk in enumerate(split_documents):
        start_index = split_chunk.metadata.pop("start_index")
        chunk_content = split_chunk.page_content
        chunk_hash = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()
        chunk_id = hashlib.sha256(
            f"{document_id}\0{chunk_index}\0{chunk_hash}".encode()
        ).hexdigest()

        metadata = dict(split_chunk.metadata)
        metadata.update(
            {
                "parent_document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "start_index": start_index,
                "end_index": start_index + len(chunk_content),
                "chunk_char_count": len(chunk_content),
                "chunk_content_hash": chunk_hash,
                "chunking_method": _CHUNKING_METHOD,
                "chunking_version": _CHUNKING_VERSION,
                "chunk_size": resolved_config.chunk_size,
                "chunk_overlap": resolved_config.chunk_overlap,
            }
        )
        chunks.append(Document(page_content=chunk_content, metadata=metadata))

    return tuple(chunks)


def split_documents(
    documents: Iterable[Document],
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """Split cleaned documents with one shared configuration and stable order.

    Duplicate document IDs are rejected because they would make chunk lineage
    ambiguous and can produce duplicate chunk IDs in a vector index.
    """
    resolved_config = config or ChunkingConfig()
    all_chunks: list[Document] = []
    summaries: list[DocumentChunkSummary] = []
    seen_document_ids: set[str] = set()

    for index, document in enumerate(documents):
        try:
            chunks = split_document(document, resolved_config)
        except (TypeError, ValueError) as error:
            raise type(error)(f"documents[{index}]: {error}") from error

        document_id = chunks[0].metadata["parent_document_id"]
        if document_id in seen_document_ids:
            raise ValueError(f"documents[{index}] has duplicate document_id: {document_id}")
        seen_document_ids.add(document_id)

        source = document.metadata.get("source")
        summaries.append(
            DocumentChunkSummary(
                document_id=document_id,
                source=None if source is None else str(source),
                chunk_count=len(chunks),
            )
        )
        all_chunks.extend(chunks)

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in all_chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("batch contains duplicate chunk_id values")

    return ChunkingResult(
        chunks=tuple(all_chunks),
        documents=tuple(summaries),
        input_count=len(summaries),
    )


def build_chunk_corpus(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    config: ChunkingConfig | None = None,
) -> ChunkBuildResult:
    """Read cleaned documents, split them, and atomically write JSONL and a report."""
    source_path = Path(input_path)
    chunk_path = Path(output_path)
    audit_path = Path(report_path)
    _validate_build_paths(source_path, chunk_path, audit_path)

    input_bytes = source_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    documents = _load_documents_jsonl(input_bytes, source_path)
    resolved_config = config or ChunkingConfig()
    result = split_documents(documents, resolved_config)

    output_text = _documents_to_jsonl(result.chunks)
    output_sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    build_id = _build_id(input_sha256, resolved_config)
    chunk_lengths = [len(chunk.page_content) for chunk in result.chunks]
    chunks_per_document = [summary.chunk_count for summary in result.documents]
    report: dict[str, Any] = {
        "schema_version": "1",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "build_id": build_id,
        "input_path": source_path.as_posix(),
        "output_path": chunk_path.as_posix(),
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "configuration": {
            "chunking_method": _CHUNKING_METHOD,
            "chunking_version": _CHUNKING_VERSION,
            "chunk_size": resolved_config.chunk_size,
            "chunk_overlap": resolved_config.chunk_overlap,
            "length_function": "len",
            "keep_separator": "end",
            "separators": list(resolved_config.separators),
        },
        "results": {
            "input_count": result.input_count,
            "chunk_count": result.chunk_count,
            "min_chunk_chars": min(chunk_lengths),
            "max_chunk_chars": max(chunk_lengths),
            "min_chunks_per_document": min(chunks_per_document),
            "max_chunks_per_document": max(chunks_per_document),
            "all_chunk_ids_unique": len(
                {chunk.metadata["chunk_id"] for chunk in result.chunks}
            )
            == result.chunk_count,
        },
        "documents": [asdict(summary) for summary in result.documents],
    }

    _atomic_write_text(chunk_path, output_text)
    _atomic_write_text(
        audit_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return ChunkBuildResult(
        input_count=result.input_count,
        chunk_count=result.chunk_count,
        output_path=chunk_path,
        report_path=audit_path,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        build_id=build_id,
    )


def _validate_build_paths(input_path: Path, output_path: Path, report_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {input_path}")
    if input_path.suffix.lower() != ".jsonl":
        raise ValueError(f"input must be a .jsonl file: {input_path}")
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"output must be a .jsonl file: {output_path}")
    if report_path.suffix.lower() != ".json":
        raise ValueError(f"report must be a .json file: {report_path}")

    resolved_paths = [path.resolve() for path in (input_path, output_path, report_path)]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("input, output, and report paths must be different")


def _load_documents_jsonl(content: bytes, path: Path) -> tuple[Document, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise UnicodeError(f"input JSONL must use UTF-8 encoding: {path}") from error

    if not text.strip():
        raise ValueError(f"input JSONL is empty: {path}")

    documents: list[Document] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"input JSONL contains a blank line at line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON at line {line_number}: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")

        page_content = record.get("page_content")
        metadata = record.get("metadata")
        if not isinstance(page_content, str) or not page_content.strip():
            raise ValueError(f"line {line_number}.page_content must be a non-empty string")
        if not isinstance(metadata, dict):
            raise ValueError(f"line {line_number}.metadata must be a JSON object")
        documents.append(Document(page_content=page_content, metadata=metadata))

    return tuple(documents)


def _documents_to_jsonl(documents: tuple[Document, ...]) -> str:
    lines = [
        json.dumps(
            {"page_content": document.page_content, "metadata": document.metadata},
            ensure_ascii=False,
            sort_keys=True,
        )
        for document in documents
    ]
    return "\n".join(lines) + "\n"


def _build_id(input_sha256: str, config: ChunkingConfig) -> str:
    payload = {
        "input_sha256": input_sha256,
        "chunking_method": _CHUNKING_METHOD,
        "chunking_version": _CHUNKING_VERSION,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "separators": list(config.separators),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/corpus_v1.jsonl")
    parser.add_argument("--output", default="data/processed/chunks_v1.jsonl")
    parser.add_argument("--report", default="data/reports/chunks_v1_build.json")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    """Build the configured chunk corpus and print its measured summary."""
    args = _parse_args()
    result = build_chunk_corpus(
        args.input,
        args.output,
        args.report,
        ChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        ),
    )
    print(json.dumps(asdict(result), ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
