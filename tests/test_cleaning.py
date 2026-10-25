import hashlib
import unicodedata

import pytest
from langchain_core.documents import Document

from mental_health_rag.cleaning import clean_document


def test_clean_document_normalizes_text_without_removing_meaningful_terms() -> None:
    document = Document(
        page_content=(
            "\ufeff焦虑\u00a0反应  \r\n\r\n\r\n"
            "  不进行诊断，保留数字 123 和自杀安全关键词。\t\r"
        ),
        metadata={"source": "data/raw/example.txt", "document_id": "source-id"},
    )

    cleaned = clean_document(document)

    assert cleaned.page_content == (
        "焦虑 反应\n\n不进行诊断，保留数字 123 和自杀安全关键词。"
    )


def test_clean_document_uses_nfc_unicode_normalization() -> None:
    decomposed_text = "Cafe\u0301 与心理健康"

    cleaned = clean_document(Document(page_content=decomposed_text))

    assert cleaned.page_content == unicodedata.normalize("NFC", decomposed_text)


def test_clean_document_preserves_lineage_and_records_cleaning_metadata() -> None:
    content = "  睡眠   科普  "
    original_metadata = {
        "source": "data/raw/sleep.txt",
        "document_id": "source-id",
        "char_count": len(content),
    }
    document = Document(page_content=content, metadata=original_metadata)

    cleaned = clean_document(document)

    assert cleaned.metadata == {
        "source": "data/raw/sleep.txt",
        "document_id": "source-id",
        "char_count": len("睡眠 科普"),
        "original_char_count": len(content),
        "cleaned_content_hash": hashlib.sha256("睡眠 科普".encode()).hexdigest(),
        "cleaning_version": "1",
    }
    assert document.page_content == content
    assert document.metadata == original_metadata


def test_clean_document_is_idempotent() -> None:
    document = Document(page_content="  压力\t管理\r\n\r\n\r\n建议  ")

    cleaned_once = clean_document(document)
    cleaned_twice = clean_document(cleaned_once)

    assert cleaned_twice == cleaned_once


def test_clean_document_rejects_content_that_becomes_empty() -> None:
    document = Document(page_content="\ufeff \t\r\n\u200b")

    with pytest.raises(ValueError, match="empty after cleaning"):
        clean_document(document)


def test_clean_document_rejects_non_document_input() -> None:
    with pytest.raises(TypeError, match="LangChain Document"):
        clean_document("not a document")  # type: ignore[arg-type]
