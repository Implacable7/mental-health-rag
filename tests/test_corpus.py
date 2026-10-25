import json
from pathlib import Path
from typing import Any

import pytest

from mental_health_rag import corpus


def _source(**overrides: Any) -> dict[str, Any]:
    value = {
        "source_id": "who_example",
        "title": "心理健康示例",
        "publisher": "世界卫生组织",
        "topic": "基础知识",
        "source_type": "html",
        "extractor": "who_article",
        "url": "https://www.who.int/zh/example",
        "license_note": "仅用于非商业学习。",
        "required_phrases": ["心理健康示例", "重要事实", "寻求专业帮助"],
    }
    value.update(overrides)
    return value


def _write_catalog(path: Path, sources: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps({"corpus_id": "test_v1", "sources": sources}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_real_catalog_contains_approved_authoritative_sources() -> None:
    corpus_id, sources = corpus.load_source_catalog("data/sources/corpus_v1.json")

    assert corpus_id == "mental_health_zh_v1"
    assert len(sources) == 16
    assert {source.publisher for source in sources} == {
        "世界卫生组织",
        "中国政府网转载国家卫生健康委",
        "中国政府网转载国家卫生健康委等部门",
    }
    assert all(source.url.startswith("https://") for source in sources)


def test_catalog_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    catalog_path = tmp_path / "sources.json"
    _write_catalog(
        catalog_path,
        [_source(), _source(url="https://www.who.int/zh/another")],
    )

    with pytest.raises(ValueError, match="duplicate source_id"):
        corpus.load_source_catalog(catalog_path)


def test_catalog_rejects_unapproved_host(tmp_path: Path) -> None:
    catalog_path = tmp_path / "sources.json"
    _write_catalog(catalog_path, [_source(url="https://example.com/article")])

    with pytest.raises(ValueError, match="approved HTTPS host"):
        corpus.load_source_catalog(catalog_path)


def test_extracts_only_who_article_body() -> None:
    raw_html = """
    <html><body><nav>导航噪声</nav>
    <article class="sf-detail-body-wrapper">
      <h2>重要事实</h2><p>心理健康正文。</p>
      <script>javascript noise</script><ul><li>第一项</li><li>第二项</li></ul>
    </article><footer>页脚噪声</footer></body></html>
    """.encode()

    text = corpus.extract_html_text(raw_html, "who_article", "测试标题")

    assert text.startswith("测试标题\n\n重要事实")
    assert "心理健康正文" in text
    assert "- 第一项" in text
    assert "导航噪声" not in text
    assert "页脚噪声" not in text
    assert "javascript noise" not in text


def test_extracts_government_article_layout() -> None:
    raw_html = """
    <html><body><article class="artcielcont" id="articleins">
    <h1>政策标题</h1><p>心理健康服务正文。</p>
    </article></body></html>
    """.encode()

    text = corpus.extract_html_text(raw_html, "gov_article", "预期标题")

    assert "政策标题" in text
    assert "心理健康服务正文" in text


def test_extracts_only_first_matching_article_root() -> None:
    raw_html = """
    <div class="pages_content"><p>唯一正文。</p></div>
    <div class="pages_content"><p>重复的移动版正文。</p></div>
    """.encode()

    text = corpus.extract_html_text(raw_html, "gov_trs", "政策标题")

    assert "唯一正文" in text
    assert "重复的移动版正文" not in text


def test_extracts_who_question_and_answer_layout() -> None:
    raw_html = """
    <html><body><div id="sf-accordion">
    <div class="sf-accordion__panel"><h2>什么是压力</h2>
    <div class="sf-accordion__content"><p>压力是困难处境引起的担忧或紧张状态。</p></div>
    </div></div></body></html>
    """.encode()

    text = corpus.extract_html_text(raw_html, "who_qa", "压力")

    assert "什么是压力" in text
    assert "担忧或紧张状态" in text


def test_html_extraction_fails_when_article_root_is_missing() -> None:
    with pytest.raises(ValueError, match="article root was not found"):
        corpus.extract_html_text(b"<html><body>no article</body></html>", "who_article", "标题")


def test_semantic_verification_rejects_missing_or_reordered_phrases() -> None:
    source = corpus.CorpusSource(**_source())

    with pytest.raises(ValueError, match="missing or out of order"):
        corpus._verify_required_phrases(
            source,
            "心理健康示例\n寻求专业帮助\n重要事实",
        )


def test_prepare_corpus_writes_cleaned_jsonl_and_audit_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "sources.json"
    _write_catalog(catalog_path, [_source()])
    repeated_text = "心理健康需要科学认识，并在需要时寻求专业帮助。" * 40
    raw_html = (
        '<article class="sf-detail-body-wrapper"><h2>重要事实</h2><p>'
        + repeated_text
        + "</p></article>"
    ).encode()
    monkeypatch.setattr(corpus, "_download", lambda *_args: raw_html)

    result = corpus.prepare_corpus(
        catalog_path,
        tmp_path / "raw",
        tmp_path / "processed.jsonl",
        tmp_path / "report.json",
    )

    assert result.input_count == 1
    assert result.unique_count == 1
    assert result.duplicate_count == 0
    record = json.loads((tmp_path / "processed.jsonl").read_text(encoding="utf-8"))
    assert record["metadata"]["source_id"] == "who_example"
    assert record["metadata"]["cleaning_version"] == "1"
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["sources"][0]["cleaned_chars"] == len(record["page_content"])
    assert report["semantic_verification"]["required_phrase_count"] == 3
    assert report["safety_term_counts"]["求助"] == 0
