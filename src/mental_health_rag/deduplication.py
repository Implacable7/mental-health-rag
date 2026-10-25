"""Stable, auditable deduplication for cleaned documents."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    """Describe one removed duplicate and the document that was kept."""

    content_hash: str
    kept_index: int
    duplicate_index: int
    kept_source: str | None
    duplicate_source: str | None
    kept_document_id: str | None
    duplicate_document_id: str | None


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Contain unique documents and an audit trail for removed duplicates."""

    unique_documents: tuple[Document, ...]
    duplicates: tuple[DuplicateRecord, ...]
    input_count: int

    @property
    def unique_count(self) -> int:
        """Return the number of documents kept."""
        return len(self.unique_documents)

    @property
    def duplicate_count(self) -> int:
        """Return the number of duplicate documents removed."""
        return len(self.duplicates)


def deduplicate_documents(documents: Iterable[Document]) -> DeduplicationResult:
    """Keep the first copy of each cleaned text and record later duplicates.

    Every input must come from the cleaning stage. The stored cleaned-content
    hash is recomputed before use so stale metadata cannot silently remove data.
    """
    unique_documents: list[Document] = []
    duplicate_records: list[DuplicateRecord] = []
    first_by_hash: dict[str, tuple[int, Document]] = {}
    input_count = 0

    for index, document in enumerate(documents):
        input_count += 1
        if not isinstance(document, Document):
            raise TypeError(f"documents[{index}] must be a LangChain Document")

        stored_hash = document.metadata.get("cleaned_content_hash")
        if not isinstance(stored_hash, str) or not stored_hash:
            raise ValueError(
                f"documents[{index}] has no cleaned_content_hash; clean it before deduplication"
            )

        calculated_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
        if stored_hash != calculated_hash:
            raise ValueError(f"documents[{index}] has a stale cleaned_content_hash")

        first = first_by_hash.get(stored_hash)
        if first is None:
            first_by_hash[stored_hash] = (index, document)
            unique_documents.append(document)
            continue

        kept_index, kept_document = first
        if kept_document.page_content != document.page_content:
            raise ValueError(f"SHA-256 collision detected at documents[{index}]")

        duplicate_records.append(
            DuplicateRecord(
                content_hash=stored_hash,
                kept_index=kept_index,
                duplicate_index=index,
                kept_source=_metadata_text(kept_document, "source"),
                duplicate_source=_metadata_text(document, "source"),
                kept_document_id=_metadata_text(kept_document, "document_id"),
                duplicate_document_id=_metadata_text(document, "document_id"),
            )
        )

    return DeduplicationResult(
        unique_documents=tuple(unique_documents),
        duplicates=tuple(duplicate_records),
        input_count=input_count,
    )


def _metadata_text(document: Document, key: str) -> str | None:
    value = document.metadata.get(key)
    return None if value is None else str(value)
