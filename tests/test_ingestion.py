import hashlib

import pytest

from mental_health_rag.ingestion import load_text_document


def test_load_text_document_returns_content_and_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "data" / "raw"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "sleep.txt"
    content = "规律作息有助于维持稳定的睡眠节律。\n"
    source_file.write_text(content, encoding="utf-8")

    document = load_text_document(source_file)

    assert document.page_content == content
    assert document.metadata == {
        "source": "data/raw/sleep.txt",
        "source_name": "sleep.txt",
        "file_type": "txt",
        "document_id": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "char_count": len(content),
    }


def test_document_id_depends_on_content_not_file_name(tmp_path) -> None:
    content = "这是仅用于测试文档读取功能的示例文本。"
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    first_file.write_text(content, encoding="utf-8")
    second_file.write_text(content, encoding="utf-8")

    first = load_text_document(first_file)
    second = load_text_document(second_file)

    assert first.metadata["document_id"] == second.metadata["document_id"]


def test_missing_file_is_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_text_document(tmp_path / "missing.txt")


def test_non_txt_file_is_rejected(tmp_path) -> None:
    source_file = tmp_path / "document.md"
    source_file.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Only \.txt files"):
        load_text_document(source_file)


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_or_whitespace_only_file_is_rejected(tmp_path, content) -> None:
    source_file = tmp_path / "empty.txt"
    source_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="is empty"):
        load_text_document(source_file)


def test_non_utf8_file_is_rejected(tmp_path) -> None:
    source_file = tmp_path / "gbk.txt"
    source_file.write_bytes("中文内容".encode("gbk"))

    with pytest.raises(UnicodeError, match="UTF-8"):
        load_text_document(source_file)
