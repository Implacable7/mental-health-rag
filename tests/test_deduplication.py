import pytest
from langchain_core.documents import Document

from mental_health_rag.cleaning import clean_document
from mental_health_rag.deduplication import DuplicateRecord, deduplicate_documents


def make_cleaned_document(content: str, source: str, document_id: str) -> Document:
    return clean_document(
        Document(
            page_content=content,
            metadata={"source": source, "document_id": document_id},
        )
    )


def test_deduplicate_documents_keeps_first_copy_and_preserves_order() -> None:
    first = make_cleaned_document("规律睡眠有助于维持身心健康。", "first.txt", "raw-1")
    duplicate = make_cleaned_document(
        "规律睡眠有助于维持身心健康。", "duplicate.txt", "raw-2"
    )
    different = make_cleaned_document("适度运动可能帮助缓解压力。", "third.txt", "raw-3")

    result = deduplicate_documents([first, duplicate, different])

    assert result.unique_documents == (first, different)
    assert result.input_count == 3
    assert result.unique_count == 2
    assert result.duplicate_count == 1
    assert result.duplicates == (
        DuplicateRecord(
            content_hash=first.metadata["cleaned_content_hash"],
            kept_index=0,
            duplicate_index=1,
            kept_source="first.txt",
            duplicate_source="duplicate.txt",
            kept_document_id="raw-1",
            duplicate_document_id="raw-2",
        ),
    )


def test_formatting_only_differences_are_duplicates_after_cleaning() -> None:
    first = make_cleaned_document("睡眠\t建议", "first.txt", "raw-1")
    second = make_cleaned_document("  睡眠   建议  ", "second.txt", "raw-2")

    result = deduplicate_documents([first, second])

    assert result.unique_count == 1
    assert result.duplicate_count == 1


def test_each_removed_copy_has_its_own_audit_record() -> None:
    documents = [
        make_cleaned_document("压力管理", f"source-{index}.txt", f"raw-{index}")
        for index in range(3)
    ]

    result = deduplicate_documents(documents)

    assert [record.duplicate_index for record in result.duplicates] == [1, 2]
    assert [record.duplicate_source for record in result.duplicates] == [
        "source-1.txt",
        "source-2.txt",
    ]


def test_deduplicate_documents_does_not_modify_inputs() -> None:
    first = make_cleaned_document("焦虑科普", "first.txt", "raw-1")
    second = make_cleaned_document("焦虑科普", "second.txt", "raw-2")
    first_metadata = dict(first.metadata)
    second_metadata = dict(second.metadata)

    deduplicate_documents([first, second])

    assert first.metadata == first_metadata
    assert second.metadata == second_metadata


def test_empty_input_returns_empty_result() -> None:
    result = deduplicate_documents([])

    assert result.input_count == 0
    assert result.unique_count == 0
    assert result.duplicate_count == 0
    assert result.unique_documents == ()
    assert result.duplicates == ()


def test_document_must_be_cleaned_before_deduplication() -> None:
    document = Document(page_content="尚未清洗", metadata={"source": "raw.txt"})

    with pytest.raises(ValueError, match="clean it before deduplication"):
        deduplicate_documents([document])


def test_stale_cleaned_content_hash_is_rejected() -> None:
    document = make_cleaned_document("原始正文", "source.txt", "raw-1")
    document.page_content = "正文已被修改"

    with pytest.raises(ValueError, match="stale cleaned_content_hash"):
        deduplicate_documents([document])


def test_non_document_input_is_rejected_with_its_position() -> None:
    with pytest.raises(TypeError, match=r"documents\[0\]"):
        deduplicate_documents(["not a document"])  # type: ignore[list-item]
