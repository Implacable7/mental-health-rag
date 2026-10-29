"""Report source- and chunk-level balance for a curated corpus.

This report deliberately separates source authority from retrieval suitability.
An authoritative long report may be valuable as a professional reference while
still being unsuitable for the default public-education retrieval pool.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_number}") from exc
    return records


def _counts(items: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def build_report(
    catalog_path: Path,
    chunks_path: Path,
    top_n: int,
) -> dict[str, Any]:
    catalog = _read_json(catalog_path)
    sources = catalog["sources"]
    chunks = _read_jsonl(chunks_path)
    sources_by_id = {source["source_id"]: source for source in sources}
    if len(sources_by_id) != len(sources):
        raise ValueError("catalog contains duplicate source_id values")

    chunk_counts: Counter[str] = Counter()
    missing_source_ids: set[str] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source_id = metadata.get("source_id")
        if source_id not in sources_by_id:
            missing_source_ids.add(str(source_id))
            continue
        chunk_counts[source_id] += 1

    source_layers = _counts(
        [source.get("content_layer", "unspecified") for source in sources]
    )
    chunk_layers = Counter()
    for source_id, count in chunk_counts.items():
        layer = sources_by_id[source_id].get("content_layer", "unspecified")
        chunk_layers[layer] += count

    total_chunks = len(chunks)
    ranked = sorted(
        (
            {
                "source_id": source_id,
                "title": sources_by_id[source_id]["title"],
                "content_layer": sources_by_id[source_id].get(
                    "content_layer", "unspecified"
                ),
                "chunk_count": count,
                "chunk_share": round(count / total_chunks, 6)
                if total_chunks
                else 0.0,
            }
            for source_id, count in chunk_counts.items()
        ),
        key=lambda item: (-item["chunk_count"], item["source_id"]),
    )
    top_three_share = (
        sum(item["chunk_count"] for item in ranked[:3]) / total_chunks
        if total_chunks
        else 0.0
    )
    public_chunks = chunk_layers.get("public_education", 0)

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "catalog_path": catalog_path.as_posix(),
        "chunks_path": chunks_path.as_posix(),
        "source_count": len(sources),
        "chunk_count": total_chunks,
        "integrity": {
            "catalog_source_ids_unique": len(sources_by_id) == len(sources),
            "chunk_source_ids_all_in_catalog": not missing_source_ids,
            "missing_source_ids": sorted(missing_source_ids),
        },
        "source_counts_by_content_layer": source_layers,
        "chunk_counts_by_content_layer": dict(sorted(chunk_layers.items())),
        "chunk_share_by_content_layer": {
            layer: round(count / total_chunks, 6) if total_chunks else 0.0
            for layer, count in sorted(chunk_layers.items())
        },
        "authority_tier_source_counts": _counts(
            [source.get("authority_tier", "unspecified") for source in sources]
        ),
        "risk_level_source_counts": _counts(
            [source.get("risk_level", "unspecified") for source in sources]
        ),
        "concentration": {
            "top_3_source_chunk_share": round(top_three_share, 6),
            "largest_source_chunk_share": ranked[0]["chunk_share"] if ranked else 0.0,
            "top_sources": ranked[:top_n],
        },
        "target_status": {
            "all_layers_at_least_1200_chunks": total_chunks >= 1200,
            "public_education_at_least_1200_chunks": public_chunks >= 1200,
        },
        "retrieval_policy_recommendation": {
            "default_pool": "public_education",
            "routed_pools": ["policy_service", "professional_reference"],
            "reason": (
                "默认公众问答应优先检索科普层；政策服务和专业参考资料按问题意图路由，"
                "并对单一来源设置候选片段上限，避免长文档片段数量偏置。"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", default="data/sources/corpus_mental_health_v1.json"
    )
    parser.add_argument(
        "--chunks", default="data/processed/chunks_mental_health_v1.jsonl"
    )
    parser.add_argument(
        "--output", default="data/reports/corpus_mental_health_v1_balance.json"
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")

    report = build_report(Path(args.catalog), Path(args.chunks), args.top_n)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
