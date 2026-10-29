"""Build a standalone curated catalog from an audit and reviewed additions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mental_health_rag.corpus import load_source_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="data/sources/corpus_v7.json")
    parser.add_argument(
        "--audit", default="data/reports/corpus_v7_relevance_audit.json"
    )
    parser.add_argument(
        "--additions",
        default="data/sources/corpus_mental_health_v1_additions.json",
    )
    parser.add_argument(
        "--output", default="data/sources/corpus_mental_health_v1.json"
    )
    args = parser.parse_args()

    _, base_sources = load_source_catalog(args.base)
    _, additions = load_source_catalog(args.additions)
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    active_ids = {
        item["source_id"]
        for item in audit["sources"]
        if item["relevance_label"] == "core"
    }
    selected = [source for source in base_sources if source.source_id in active_ids]
    all_sources = selected + list(additions)
    source_ids = [source.source_id for source in all_sources]
    urls = [source.url for source in all_sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("curated catalog contains duplicate source IDs")
    if len(urls) != len(set(urls)):
        raise ValueError("curated catalog contains duplicate source URLs")

    payload = {
        "corpus_id": "mental_health_zh_curated_v1",
        "purpose": (
            "面向心理健康科普问答的严格语料目录；基础目录仅保留核心心理健康资料，"
            "再合并逐篇审核通过的官方补充资料。"
        ),
        "language": "zh-CN",
        "selection_policy": (
            "来源权威、与心理健康直接相关、正文可稳定提取且通过顺序语义锚点；"
            "支持性背景资料和一般医学资料均不进入默认检索池。"
        ),
        "sources": [asdict(source) for source in all_sources],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "retained_from_base": len(selected),
                "reviewed_additions": len(additions),
                "total": len(all_sources),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
