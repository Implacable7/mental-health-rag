import hashlib
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from mental_health_rag.chunking import ChunkingConfig, build_chunk_corpus
from mental_health_rag.cleaning import clean_document


def _write_cleaned_documents(path: Path) -> None:
    documents = [
        clean_document(
            Document(
                page_content="规律睡眠有助于维护心理健康。" * 8,
                metadata={
                    "document_id": "document-1",
                    "source": "first.txt",
                    "title": "睡眠",
                },
            )
        ),
        clean_document(
            Document(
                page_content="识别压力来源并在需要时寻求支持。" * 8,
                metadata={
                    "document_id": "document-2",
                    "source": "second.txt",
                    "title": "压力",
                },
            )
        ),
    ]
    lines = [
        json.dumps(
            {"page_content": document.page_content, "metadata": document.metadata},
            ensure_ascii=False,
            sort_keys=True,
        )
        for document in documents
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def test_build_chunk_corpus_writes_jsonl_and_report(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    output_path = tmp_path / "nested" / "chunks.jsonl"
    report_path = tmp_path / "reports" / "build.json"
    _write_cleaned_documents(input_path)

    result = build_chunk_corpus(
        input_path,
        output_path,
        report_path,
        ChunkingConfig(chunk_size=50, chunk_overlap=10),
    )

    records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.input_count == 2
    assert result.chunk_count == len(records)
    assert report["results"]["input_count"] == 2
    assert report["results"]["chunk_count"] == len(records)
    assert report["configuration"]["chunk_size"] == 50
    assert report["configuration"]["chunk_overlap"] == 10
    assert report["build_id"] == result.build_id
    assert report["output_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert [item["document_id"] for item in report["documents"]] == [
        "document-1",
        "document-2",
    ]
    assert all("chunk_id" in record["metadata"] for record in records)
    assert not output_path.with_suffix(".jsonl.tmp").exists()
    assert not report_path.with_suffix(".json.tmp").exists()


def test_repeated_build_has_identical_chunks_and_build_id(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    _write_cleaned_documents(input_path)
    config = ChunkingConfig(chunk_size=60, chunk_overlap=10)

    first = build_chunk_corpus(input_path, first_output, tmp_path / "first.json", config)
    second = build_chunk_corpus(input_path, second_output, tmp_path / "second.json", config)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first.output_sha256 == second.output_sha256
    assert first.build_id == second.build_id


def test_configuration_change_changes_build_id(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_cleaned_documents(input_path)

    first = build_chunk_corpus(
        input_path,
        tmp_path / "first.jsonl",
        tmp_path / "first.json",
        ChunkingConfig(chunk_size=50, chunk_overlap=10),
    )
    second = build_chunk_corpus(
        input_path,
        tmp_path / "second.jsonl",
        tmp_path / "second.json",
        ChunkingConfig(chunk_size=60, chunk_overlap=10),
    )

    assert first.build_id != second.build_id


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "empty"),
        ("not-json\n", "invalid JSON at line 1"),
        ('{"page_content": "正文", "metadata": []}\n', "metadata"),
        ('{"page_content": "", "metadata": {}}\n', "page_content"),
        ('{"page_content": "正文", "metadata": {}}\n\n', "blank line at line 2"),
    ],
)
def test_invalid_input_jsonl_is_rejected(
    tmp_path: Path, content: str, message: str
) -> None:
    input_path = tmp_path / "corpus.jsonl"
    input_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_chunk_corpus(
            input_path,
            tmp_path / "chunks.jsonl",
            tmp_path / "report.json",
        )


def test_build_paths_must_be_different(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    _write_cleaned_documents(input_path)

    with pytest.raises(ValueError, match="must be different"):
        build_chunk_corpus(input_path, input_path, tmp_path / "report.json")


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build_chunk_corpus(
            tmp_path / "missing.jsonl",
            tmp_path / "chunks.jsonl",
            tmp_path / "report.json",
        )
