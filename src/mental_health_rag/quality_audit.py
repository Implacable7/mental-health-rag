"""Audit source relevance independently from source authority and extraction quality."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mental_health_rag.corpus import load_source_catalog


def build_relevance_audit(
    catalog_path: str | Path,
    policy_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Classify every catalog source and fail if the policy is inconsistent."""
    corpus_id, sources = load_source_catalog(catalog_path)
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    core_ids = _source_id_set(policy, "core_source_ids")
    supporting_ids = _source_id_set(policy, "supporting_source_ids")
    catalog_ids = {source.source_id for source in sources}

    overlap = core_ids & supporting_ids
    if overlap:
        raise ValueError(f"source IDs appear in multiple relevance labels: {sorted(overlap)}")
    unknown = (core_ids | supporting_ids) - catalog_ids
    if unknown:
        raise ValueError(f"policy contains unknown source IDs: {sorted(unknown)}")

    labels = policy["labels"]
    records = []
    for source in sources:
        if source.source_id in core_ids:
            relevance = "core"
        elif source.source_id in supporting_ids:
            relevance = "supporting"
        else:
            relevance = "quarantined"
        records.append(
            {
                "source_id": source.source_id,
                "title": source.title,
                "publisher": source.publisher,
                "url": source.url,
                "host": urlparse(source.url).hostname,
                "topic": source.topic,
                "authority_tier": source.authority_tier,
                "risk_level": source.risk_level,
                "relevance_label": relevance,
                "decision_reason": labels[relevance],
            }
        )

    counts = Counter(record["relevance_label"] for record in records)
    result = {
        "audit_id": policy["policy_id"],
        "corpus_id": corpus_id,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "catalog_source_count": len(sources),
        "default_retrieval_source_count": counts["core"] + counts["supporting"],
        "label_counts": dict(sorted(counts.items())),
        "important_limit": (
            "该审计判断主题准入，不等于逐句医学审校；医学正确性仍依赖权威来源、"
            "版本时效检查和高风险内容人工复核。"
        ),
        "sources": records,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return result


def _source_id_set(policy: dict[str, Any], key: str) -> set[str]:
    value = policy.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of source IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{key} contains duplicate source IDs")
    return set(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/sources/corpus_v7.json")
    parser.add_argument(
        "--policy", default="data/policies/corpus_relevance_v1.json"
    )
    parser.add_argument(
        "--output", default="data/reports/corpus_v7_relevance_audit.json"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_relevance_audit(args.catalog, args.policy, args.output)
    print(
        json.dumps(
            {
                "catalog_source_count": result["catalog_source_count"],
                "default_retrieval_source_count": result[
                    "default_retrieval_source_count"
                ],
                "label_counts": result["label_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
