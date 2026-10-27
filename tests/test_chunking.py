import pytest
from langchain_core.documents import Document

from mental_health_rag.chunking import (
    ChunkingConfig,
    DocumentChunkSummary,
    split_document,
    split_documents,
)
from mental_health_rag.cleaning import clean_document


def make_cleaned_document(content: str, document_id: str = "document-1") -> Document:
    return clean_document(
        Document(
            page_content=content,
            metadata={
                "source": "example.txt",
                "title": "示例资料",
                "document_id": document_id,
            },
        )
    )


def test_short_document_remains_one_traceable_chunk() -> None:
    document = make_cleaned_document("规律睡眠有助于维持身心健康。")

    chunks = split_document(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.page_content == document.page_content
    assert chunk.metadata["source"] == "example.txt"
    assert chunk.metadata["parent_document_id"] == "document-1"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["chunk_count"] == 1
    assert chunk.metadata["start_index"] == 0
    assert chunk.metadata["end_index"] == len(document.page_content)
    assert chunk.metadata["chunk_char_count"] == len(chunk.page_content)
    assert chunk.metadata["chunk_size"] == 500
    assert chunk.metadata["chunk_overlap"] == 80


def test_chinese_sentence_boundaries_are_preferred() -> None:
    document = make_cleaned_document("第一句介绍背景。第二句说明方法。第三句给出结论。")
    config = ChunkingConfig(chunk_size=12, chunk_overlap=0)

    chunks = split_document(document, config)

    assert [chunk.page_content for chunk in chunks] == [
        "第一句介绍背景。",
        "第二句说明方法。",
        "第三句给出结论。",
    ]
    assert all(len(chunk.page_content) <= 12 for chunk in chunks)


def test_short_title_is_not_left_as_a_standalone_chunk() -> None:
    document = make_cleaned_document(
        "压力\n\n重要事实\n" + "识别压力来源并寻求支持有助于维护心理健康。" * 8
    )
    config = ChunkingConfig(chunk_size=80, chunk_overlap=10)

    chunks = split_document(document, config)

    assert len(chunks) > 1
    assert chunks[0].page_content.startswith("压力\n\n重要事实")
    assert chunks[0].page_content != "压力"


def test_character_fallback_creates_configured_overlap() -> None:
    document = make_cleaned_document("一二三四五六七八九十一二三四五六七八九十")
    config = ChunkingConfig(chunk_size=10, chunk_overlap=3)

    chunks = split_document(document, config)

    assert len(chunks) == 3
    assert chunks[0].page_content[-3:] == chunks[1].page_content[:3]
    assert chunks[1].page_content[-3:] == chunks[2].page_content[:3]
    assert [chunk.metadata["start_index"] for chunk in chunks] == [0, 7, 14]


def test_chunk_metadata_is_complete_unique_and_deterministic() -> None:
    document = make_cleaned_document("心理健康很重要。" * 20)
    config = ChunkingConfig(chunk_size=30, chunk_overlap=5)

    first_run = split_document(document, config)
    second_run = split_document(document, config)

    assert first_run == second_run
    assert len({chunk.metadata["chunk_id"] for chunk in first_run}) == len(first_run)
    for index, chunk in enumerate(first_run):
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_count"] == len(first_run)
        assert chunk.metadata["end_index"] == (
            chunk.metadata["start_index"] + len(chunk.page_content)
        )
        assert document.page_content[
            chunk.metadata["start_index"] : chunk.metadata["end_index"]
        ] == chunk.page_content


def test_split_document_does_not_modify_input() -> None:
    document = make_cleaned_document("压力管理包括识别压力来源和寻求支持。" * 10)
    original_content = document.page_content
    original_metadata = dict(document.metadata)

    split_document(document, ChunkingConfig(chunk_size=40, chunk_overlap=8))

    assert document.page_content == original_content
    assert document.metadata == original_metadata


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_overlap": -1}, "chunk_overlap"),
        ({"chunk_size": 10, "chunk_overlap": 10}, "smaller"),
        ({"separators": ("。",)}, "empty-string fallback"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ChunkingConfig(**kwargs)  # type: ignore[arg-type]


def test_document_must_be_cleaned_before_chunking() -> None:
    document = Document(page_content="尚未清洗", metadata={"document_id": "document-1"})

    with pytest.raises(ValueError, match="clean it before chunking"):
        split_document(document)


def test_stale_cleaned_content_hash_is_rejected() -> None:
    document = make_cleaned_document("原始正文")
    document.page_content = "正文已被修改"

    with pytest.raises(ValueError, match="stale cleaned_content_hash"):
        split_document(document)


def test_document_id_is_required_for_chunk_lineage() -> None:
    document = clean_document(Document(page_content="有效正文"))

    with pytest.raises(ValueError, match="document_id"):
        split_document(document)


def test_non_document_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="LangChain Document"):
        split_document("not a document")  # type: ignore[arg-type]


def test_split_documents_preserves_document_and_chunk_order() -> None:
    first = make_cleaned_document("第一篇资料。" * 12, "document-1")
    second = make_cleaned_document("第二篇资料。" * 8, "document-2")
    config = ChunkingConfig(chunk_size=30, chunk_overlap=5)

    result = split_documents([first, second], config)

    first_chunks = split_document(first, config)
    second_chunks = split_document(second, config)
    assert result.input_count == 2
    assert result.chunk_count == len(first_chunks) + len(second_chunks)
    assert result.chunks == first_chunks + second_chunks
    assert result.documents == (
        DocumentChunkSummary("document-1", "example.txt", len(first_chunks)),
        DocumentChunkSummary("document-2", "example.txt", len(second_chunks)),
    )


def test_split_documents_applies_one_shared_configuration() -> None:
    documents = [
        make_cleaned_document("压力管理。" * 10, "document-1"),
        make_cleaned_document("规律睡眠。" * 10, "document-2"),
    ]
    config = ChunkingConfig(chunk_size=24, chunk_overlap=4)

    result = split_documents(documents, config)

    assert result.chunk_count > 2
    assert all(chunk.metadata["chunk_size"] == 24 for chunk in result.chunks)
    assert all(chunk.metadata["chunk_overlap"] == 4 for chunk in result.chunks)


def test_split_documents_accepts_a_generator() -> None:
    documents = (
        make_cleaned_document(f"第{index}篇资料。", f"document-{index}")
        for index in range(3)
    )

    result = split_documents(documents)

    assert result.input_count == 3
    assert result.chunk_count == 3


def test_split_documents_empty_input_returns_empty_result() -> None:
    result = split_documents([])

    assert result.input_count == 0
    assert result.chunk_count == 0
    assert result.chunks == ()
    assert result.documents == ()


def test_split_documents_rejects_duplicate_document_ids() -> None:
    documents = [
        make_cleaned_document("第一篇正文。", "same-id"),
        make_cleaned_document("第二篇正文。", "same-id"),
    ]

    with pytest.raises(ValueError, match=r"documents\[1\].*duplicate document_id"):
        split_documents(documents)


def test_split_documents_reports_invalid_document_position() -> None:
    valid = make_cleaned_document("有效正文。", "document-1")

    with pytest.raises(TypeError, match=r"documents\[1\].*LangChain Document"):
        split_documents([valid, "not a document"])  # type: ignore[list-item]
