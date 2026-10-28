"""Build a versioned, auditable corpus from an approved source catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.documents import Document
from pypdf import PdfReader

from mental_health_rag.cleaning import clean_document
from mental_health_rag.deduplication import deduplicate_documents

_ALLOWED_HOSTS = {
    "www.who.int",
    "www.nhc.gov.cn",
    "www.gov.cn",
    "app.www.gov.cn",
    "www.zlq.gov.cn",
}
_SOURCE_TYPES = {"html", "pdf"}
_EXTRACTORS = {
    "who_article",
    "who_qa",
    "gov_article",
    "gov_trs",
    "gov_doc_content",
    "pdf_text",
}
_BLOCK_TAGS = {"br", "div", "h1", "h2", "h3", "h4", "li", "p", "section"}
_IGNORED_TAGS = {"script", "style", "noscript", "svg"}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}
_CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_MULTIPLE_SPACES = re.compile(r"[ \t\u00a0\u3000]+")
_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """Describe one approved source and how its main text is extracted."""

    source_id: str
    title: str
    publisher: str
    topic: str
    source_type: str
    extractor: str
    url: str
    license_note: str
    required_phrases: tuple[str, ...]
    landing_url: str | None = None


@dataclass(frozen=True, slots=True)
class SourceResult:
    """Record the observed result for one downloaded source."""

    source_id: str
    title: str
    publisher: str
    topic: str
    url: str
    source_type: str
    raw_path: str
    raw_sha256: str
    raw_bytes: int
    extracted_chars: int
    cleaned_chars: int
    document_id: str
    cleaned_content_hash: str
    verified_phrase_count: int


@dataclass(frozen=True, slots=True)
class CorpusBuildResult:
    """Summarize one complete corpus build."""

    corpus_id: str
    input_count: int
    unique_count: int
    duplicate_count: int
    total_cleaned_chars: int
    processed_path: Path
    report_path: Path


class _ArticleTextParser(HTMLParser):
    def __init__(self, extractor: str) -> None:
        super().__init__(convert_charrefs=True)
        self.extractor = extractor
        self.root_seen = False
        self.capture_depth = 0
        self.ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if (
            self.capture_depth == 0
            and not self.root_seen
            and self._matches_root(tag, attributes)
        ):
            self.root_seen = True
            self.capture_depth = 1
            return
        if self.capture_depth == 0:
            return

        if tag in _VOID_TAGS:
            if not self.ignored_depth and tag in _BLOCK_TAGS:
                self.parts.append("\n")
            return

        self.capture_depth += 1
        if self.ignored_depth:
            self.ignored_depth += 1
            return
        if tag in _IGNORED_TAGS:
            self.ignored_depth = 1
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if self.capture_depth and not self.ignored_depth and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.capture_depth == 0:
            return
        if self.ignored_depth:
            self.ignored_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        self.capture_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.capture_depth and not self.ignored_depth:
            self.parts.append(data)

    def _matches_root(self, tag: str, attrs: dict[str, str]) -> bool:
        classes = set(attrs.get("class", "").split())
        if self.extractor == "who_article":
            return tag == "article" and "sf-detail-body-wrapper" in classes
        if self.extractor == "who_qa":
            return tag == "div" and attrs.get("id") == "sf-accordion"
        if self.extractor == "gov_article":
            return tag == "article" and attrs.get("id") == "articleins"
        if self.extractor == "gov_trs":
            return tag == "div" and (
                "TRS_Editor" in classes or "pages_content" in classes
            )
        if self.extractor == "gov_doc_content":
            return tag == "div" and attrs.get("id") == "docContent"
        return False


def load_source_catalog(path: str | Path) -> tuple[str, tuple[CorpusSource, ...]]:
    """Load and validate a JSON source catalog, including an optional base catalog."""
    return _load_source_catalog(Path(path), seen_paths=set())


def _load_source_catalog(
    catalog_path: Path, *, seen_paths: set[Path]
) -> tuple[str, tuple[CorpusSource, ...]]:
    """Resolve one catalog and merge its reviewed additions with its base."""
    resolved_path = catalog_path.resolve()
    if resolved_path in seen_paths:
        raise ValueError(f"catalog inheritance cycle: {catalog_path}")
    seen_paths.add(resolved_path)

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    corpus_id = _required_text(payload, "corpus_id", "catalog")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("catalog.sources must be a non-empty list")

    sources: list[CorpusSource] = []
    base_catalog = _optional_text(payload, "base_catalog", "catalog")
    if base_catalog is not None:
        base_path = (catalog_path.parent / base_catalog).resolve()
        _, base_sources = _load_source_catalog(base_path, seen_paths=seen_paths)
        sources.extend(base_sources)

    source_ids = {source.source_id for source in sources}
    urls = {source.url for source in sources}
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(f"catalog.sources[{index}] must be an object")
        label = f"catalog.sources[{index}]"
        source = CorpusSource(
            source_id=_required_text(raw_source, "source_id", label),
            title=_required_text(raw_source, "title", label),
            publisher=_required_text(raw_source, "publisher", label),
            topic=_required_text(raw_source, "topic", label),
            source_type=_required_text(raw_source, "source_type", label),
            extractor=_required_text(raw_source, "extractor", label),
            url=_required_text(raw_source, "url", label),
            license_note=_required_text(raw_source, "license_note", label),
            required_phrases=_required_text_list(
                raw_source, "required_phrases", label
            ),
            landing_url=_optional_text(raw_source, "landing_url", label),
        )
        _validate_source(source, label)
        if source.source_id in source_ids:
            raise ValueError(f"duplicate source_id: {source.source_id}")
        if source.url in urls:
            raise ValueError(f"duplicate source URL: {source.url}")
        source_ids.add(source.source_id)
        urls.add(source.url)
        sources.append(source)

    seen_paths.remove(resolved_path)
    return corpus_id, tuple(sources)


def extract_html_text(raw_html: bytes, extractor: str, title: str) -> str:
    """Extract readable article text from one supported official HTML layout."""
    if extractor not in {
        "who_article",
        "who_qa",
        "gov_article",
        "gov_trs",
        "gov_doc_content",
    }:
        raise ValueError(f"unsupported HTML extractor: {extractor}")
    parser = _ArticleTextParser(extractor)
    parser.feed(raw_html.decode("utf-8"))
    parser.close()
    if not parser.parts:
        raise ValueError(f"article root was not found for extractor: {extractor}")

    lines = []
    for raw_line in "".join(parser.parts).splitlines():
        line = _MULTIPLE_SPACES.sub(" ", raw_line).strip()
        if line and line != "点击打开":
            lines.append(line)
    body = _MULTIPLE_NEWLINES.sub("\n\n", "\n".join(lines)).strip()
    return f"{title}\n\n{body}"


def extract_pdf_text(raw_pdf: bytes, title: str) -> str:
    """Extract text from a text-based PDF while preserving page boundaries."""
    reader = PdfReader(BytesIO(raw_pdf))
    page_texts = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            page_texts.append(f"[第 {page_number} 页]\n{text}")
    if not page_texts:
        raise ValueError("PDF contains no extractable text")
    return f"{title}\n\n" + "\n\n".join(page_texts)


def prepare_corpus(
    catalog_path: str | Path,
    raw_dir: str | Path,
    processed_path: str | Path,
    report_path: str | Path,
    *,
    timeout_seconds: int = 30,
    max_bytes: int = 20_000_000,
    min_characters: int = 500,
    refresh: bool = False,
) -> CorpusBuildResult:
    """Download, extract, clean, deduplicate, and save an approved corpus."""
    corpus_id, sources = load_source_catalog(catalog_path)
    raw_directory = Path(raw_dir)
    output_path = Path(processed_path)
    audit_path = Path(report_path)
    raw_directory.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    build_time = datetime.now(UTC).replace(microsecond=0).isoformat()
    cleaned_documents: list[Document] = []
    source_results: list[SourceResult] = []

    for source in sources:
        suffix = ".pdf" if source.source_type == "pdf" else ".html"
        raw_path = raw_directory / f"{source.source_id}{suffix}"
        if refresh or not raw_path.exists():
            raw_content = _download(source.url, timeout_seconds, max_bytes)
            _atomic_write_bytes(raw_path, raw_content)
        else:
            raw_content = raw_path.read_bytes()
            if len(raw_content) > max_bytes:
                raise ValueError(f"cached source exceeds max_bytes: {source.source_id}")

        extracted_text = (
            extract_pdf_text(raw_content, source.title)
            if source.source_type == "pdf"
            else extract_html_text(raw_content, source.extractor, source.title)
        )
        _validate_extracted_text(source, extracted_text, min_characters)
        _verify_required_phrases(source, extracted_text)

        raw_hash = hashlib.sha256(raw_content).hexdigest()
        document_id = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()
        document = Document(
            page_content=extracted_text,
            metadata={
                "source": source.url,
                "source_id": source.source_id,
                "source_name": source.title,
                "title": source.title,
                "publisher": source.publisher,
                "topic": source.topic,
                "language": "zh-CN",
                "file_type": source.source_type,
                "extractor": source.extractor,
                "landing_url": source.landing_url,
                "license_note": source.license_note,
                "accessed_at_utc": build_time,
                "raw_path": raw_path.as_posix(),
                "raw_sha256": raw_hash,
                "document_id": document_id,
                "char_count": len(extracted_text),
            },
        )
        cleaned = clean_document(document)
        cleaned_documents.append(cleaned)
        source_results.append(
            SourceResult(
                source_id=source.source_id,
                title=source.title,
                publisher=source.publisher,
                topic=source.topic,
                url=source.url,
                source_type=source.source_type,
                raw_path=raw_path.as_posix(),
                raw_sha256=raw_hash,
                raw_bytes=len(raw_content),
                extracted_chars=len(extracted_text),
                cleaned_chars=len(cleaned.page_content),
                document_id=document_id,
                cleaned_content_hash=str(cleaned.metadata["cleaned_content_hash"]),
                verified_phrase_count=len(source.required_phrases),
            )
        )

    deduplication = deduplicate_documents(cleaned_documents)
    _write_jsonl(output_path, deduplication.unique_documents)

    topic_counts = Counter(
        str(document.metadata["topic"])
        for document in deduplication.unique_documents
    )
    publisher_counts = Counter(
        str(document.metadata["publisher"])
        for document in deduplication.unique_documents
    )
    corpus_text = "\n".join(
        document.page_content for document in deduplication.unique_documents
    )
    total_cleaned_chars = sum(
        len(document.page_content) for document in deduplication.unique_documents
    )
    report = {
        "corpus_id": corpus_id,
        "generated_at_utc": build_time,
        "catalog_path": Path(catalog_path).as_posix(),
        "processed_path": output_path.as_posix(),
        "input_count": deduplication.input_count,
        "unique_count": deduplication.unique_count,
        "duplicate_count": deduplication.duplicate_count,
        "total_cleaned_chars": total_cleaned_chars,
        "topic_counts": dict(sorted(topic_counts.items())),
        "publisher_counts": dict(sorted(publisher_counts.items())),
        "semantic_verification": {
            "method": "人工选定的原文短语必须全部保留并按正文顺序出现",
            "required_phrase_count": sum(
                result.verified_phrase_count for result in source_results
            ),
            "all_required_phrases_verified": True,
        },
        "safety_term_counts": {
            term: corpus_text.count(term)
            for term in ("自杀", "自伤", "自残", "求助", "专业", "治疗", "不建议")
        },
        "duplicates": [asdict(item) for item in deduplication.duplicates],
        "sources": [asdict(item) for item in source_results],
    }
    _atomic_write_text(audit_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    return CorpusBuildResult(
        corpus_id=corpus_id,
        input_count=deduplication.input_count,
        unique_count=deduplication.unique_count,
        duplicate_count=deduplication.duplicate_count,
        total_cleaned_chars=total_cleaned_chars,
        processed_path=output_path,
        report_path=audit_path,
    )


def _validate_source(source: CorpusSource, label: str) -> None:
    parsed = urlparse(source.url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(f"{label}.url must use an approved HTTPS host")
    if source.source_type not in _SOURCE_TYPES:
        raise ValueError(f"{label}.source_type is unsupported: {source.source_type}")
    if source.extractor not in _EXTRACTORS:
        raise ValueError(f"{label}.extractor is unsupported: {source.extractor}")
    if source.source_type == "pdf" and source.extractor != "pdf_text":
        raise ValueError(f"{label} PDF source must use pdf_text")
    if source.source_type == "html" and source.extractor == "pdf_text":
        raise ValueError(f"{label} HTML source cannot use pdf_text")
    if not re.fullmatch(r"[a-z0-9_]+", source.source_id):
        raise ValueError(f"{label}.source_id must use lowercase letters, numbers, and underscores")


def _validate_extracted_text(
    source: CorpusSource, text: str, min_characters: int
) -> None:
    if len(text) < min_characters:
        raise ValueError(
            f"extracted text is too short for {source.source_id}: {len(text)} characters"
        )
    cjk_count = len(_CJK_CHARACTER.findall(text))
    if cjk_count / len(text) < 0.2:
        raise ValueError(f"extracted text has too little Chinese content: {source.source_id}")
    if "<html" in text.lower() or "javascript" in text.lower():
        raise ValueError(f"HTML noise remains in extracted text: {source.source_id}")


def _verify_required_phrases(source: CorpusSource, text: str) -> None:
    """Require reviewed phrases to survive extraction and remain in source order."""
    previous_position = -1
    for phrase in source.required_phrases:
        position = text.find(phrase, previous_position + 1)
        if position < 0:
            raise ValueError(
                f"required phrase missing or out of order for {source.source_id}: {phrase}"
            )
        previous_position = position


def _download(url: str, timeout_seconds: int, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mental-health-rag-learning-project/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"download exceeds max_bytes: {url}")
    if not content:
        raise ValueError(f"download is empty: {url}")
    return content


def _write_jsonl(path: Path, documents: tuple[Document, ...]) -> None:
    lines = [
        json.dumps(
            {"page_content": document.page_content, "metadata": document.metadata},
            ensure_ascii=False,
            sort_keys=True,
        )
        for document in documents
    ]
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _required_text(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str, label: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be null or a non-empty string")
    return value.strip()


def _required_text_list(
    payload: dict[str, Any], key: str, label: str
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{label}.{key} must contain at least three strings")
    phrases = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(phrases) != len(value):
        raise ValueError(f"{label}.{key} must contain only non-empty strings")
    return phrases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/sources/corpus_v1.json")
    parser.add_argument("--raw-dir", default="data/raw/corpus_v1")
    parser.add_argument("--output", default="data/processed/corpus_v1.jsonl")
    parser.add_argument("--report", default="data/reports/corpus_v1_report.json")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the configured corpus and print its measured summary."""
    args = _parse_args()
    result = prepare_corpus(
        args.catalog,
        args.raw_dir,
        args.output,
        args.report,
        refresh=args.refresh,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
