"""Load source documents into the RAG pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document


def load_text_document(path: str | Path) -> Document:
    """Load one non-empty UTF-8 text file as a LangChain document.

    The document content is kept unchanged. Metadata is limited to stable,
    traceable fields that later stages can copy to chunks and citations.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Text file does not exist: {file_path}")
    if file_path.suffix.lower() != ".txt":
        raise ValueError(f"Only .txt files are supported: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise UnicodeError(f"Text file must use UTF-8 encoding: {file_path}") from error

    if not content.strip():
        raise ValueError(f"Text file is empty: {file_path}")

    resolved_path = file_path.resolve()
    project_root = Path.cwd().resolve()
    try:
        source = resolved_path.relative_to(project_root).as_posix()
    except ValueError:
        source = resolved_path.name

    document_id = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return Document(
        page_content=content,
        metadata={
            "source": source,
            "source_name": file_path.name,
            "file_type": "txt",
            "document_id": document_id,
            "char_count": len(content),
        },
    )
