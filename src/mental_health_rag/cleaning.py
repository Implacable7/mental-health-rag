"""Conservative, deterministic cleaning for loaded documents."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from langchain_core.documents import Document

_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\u00a0\u3000]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_CLEANING_VERSION = "1"


def clean_document(document: Document) -> Document:
    """Return a cleaned copy while preserving the source document's lineage.

    Cleaning is intentionally conservative: normalize Unicode and line endings,
    remove common zero-width artifacts, normalize horizontal whitespace, and
    keep at most one blank line between paragraphs.
    """
    if not isinstance(document, Document):
        raise TypeError("document must be a LangChain Document")

    content = unicodedata.normalize("NFC", document.page_content)
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = content.replace("\ufeff", "").replace("\u200b", "")

    cleaned_lines = [
        _HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in content.split("\n")
    ]
    cleaned_content = _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(cleaned_lines)).strip()

    if not cleaned_content:
        raise ValueError("Document is empty after cleaning")

    metadata = dict(document.metadata)
    metadata.setdefault("original_char_count", len(document.page_content))
    metadata.update(
        {
            "char_count": len(cleaned_content),
            "cleaned_content_hash": hashlib.sha256(
                cleaned_content.encode("utf-8")
            ).hexdigest(),
            "cleaning_version": _CLEANING_VERSION,
        }
    )

    return Document(page_content=cleaned_content, metadata=metadata)
