"""Deterministic Chinese-aware text chunking for cleaned documents."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

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
