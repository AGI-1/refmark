"""Build and evaluate an article-level BGB concern-navigation index.

The region demo indexes individual BGB paragraphs/Absatz regions. This script
aggregates those regions to article-level refs such as ``bgb:S_437`` and can
inject curated concern aliases into the retrieval view. That makes layperson
motivation queries measurable without changing the source legal text.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from refmark.search_index import (
    PortableBM25Index,
    RetrievalView,
    SearchHit,
    SearchRegion,
    export_browser_search_index,
    load_search_index,
    tokenize,
)


DEFAULT_INPUT_INDEX = "examples/bgb_browser_search/output_full_qwen_turbo/bgb_openrouter_index.json"
DEFAULT_ALIAS_PATH = "examples/bgb_browser_search/concern_aliases.json"
DEFAULT_BREAK_QUERIES = "examples/bgb_browser_search/break_queries.json"
ARTICLE_SUFFIX_RE = re.compile(r"_A\d+$")


@dataclass(frozen=True)
class ConcernAlias:
    id: str
    category: str
    expected_prefixes: list[str]
    queries: list[str]
    aliases: list[str]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an article-level BGB concern-navigation index.")
    parser.add_argument("--input-index", default=DEFAULT_INPUT_INDEX)
    parser.add_argument("--aliases", default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--break-queries", default=DEFAULT_BREAK_QUERIES)
    parser.add_argument("--output-dir", default="examples/bgb_browser_search/output_article_nav")
    parser.add_argument("--without-aliases", action="store_true", help="Build a raw article index without concern aliases.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concern_aliases = load_aliases(args.aliases)
    injected_aliases = [] if args.without_aliases else concern_aliases
    source_index = load_search_index(args.input_index)
    article_regions = build_article_regions(source_index.regions, aliases=injected_aliases)

    portable_path = output_dir / "bgb_article_index.json"
    write_portable_index(
        article_regions,
        portable_path,
        source_index=args.input_index,
        aliases_path=None if args.without_aliases else args.aliases,
    )
    browser_path = output_dir / "bgb_article_browser_index.json"
    browser_payload = export_browser_search_index(portable_path, browser_path, max_text_chars=1300)
    data_path = output_dir / "bgb_article_demo_data.js"
    data_path.write_text(
        "window.BGB_REFMARK_INDEX = "
        + json.dumps(browser_payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )

    query_rows = load_break_queries(args.break_queries) + alias_query_rows(concern_aliases)
    report = evaluate_queries(article_regions, query_rows, top_k=args.top_k)
    report.update(
        {
            "schema": "refmark.bgb_article_navigation.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_index": str(args.input_index),
            "aliases": None if args.without_aliases else str(args.aliases),
            "article_regions": len(article_regions),
            "portable_index": str(portable_path),
            "browser_index": str(browser_path),
            "data_js": str(data_path),
        }
    )
    report_path = output_dir / "bgb_article_navigation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def build_article_regions(regions: list[SearchRegion], *, aliases: list[ConcernAlias]) -> list[SearchRegion]:
    grouped: dict[tuple[str, str], list[SearchRegion]] = defaultdict(list)
    for region in regions:
        grouped[(region.doc_id, article_id_for(region.region_id))].append(region)

    alias_by_ref = _alias_text_by_ref(aliases)
    ordered_groups = sorted(grouped.items(), key=lambda item: min(region.ordinal for region in item[1]))
    article_regions: list[SearchRegion] = []
    for ordinal, ((doc_id, article_id), items) in enumerate(ordered_groups):
        items = sorted(items, key=lambda region: region.ordinal)
        stable_ref = f"{doc_id}:{article_id}"
        text = "\n\n".join(region.text for region in items)
        source_path = items[0].source_path
        view = merge_views(items, alias_text=alias_by_ref.get(stable_ref, []))
        article_regions.append(
            SearchRegion(
                doc_id=doc_id,
                region_id=article_id,
                text=text,
                hash=hash_text(text),
                source_path=source_path,
                ordinal=ordinal,
                prev_region_id=None,
                next_region_id=None,
                view=view,
            )
        )
    return [
        SearchRegion(
            doc_id=region.doc_id,
            region_id=region.region_id,
            text=region.text,
            hash=region.hash,
            source_path=region.source_path,
            ordinal=region.ordinal,
            prev_region_id=article_regions[index - 1].region_id if index > 0 else None,
            next_region_id=article_regions[index + 1].region_id if index + 1 < len(article_regions) else None,
            view=region.view,
        )
        for index, region in enumerate(article_regions)
    ]


def merge_views(items: list[SearchRegion], *, alias_text: list[str]) -> RetrievalView:
    first = items[0]
    summaries = unique([region.view.summary for region in items if region.view.summary])
    questions = unique([question for region in items for question in region.view.questions] + alias_text)
    keywords = unique([keyword for region in items for keyword in region.view.keywords] + keyword_phrases(alias_text))
    return RetrievalView(
        summary=summaries[0] if summaries else first.text.splitlines()[0][:160],
        questions=questions[:80],
        keywords=keywords[:80],
    )


def evaluate_queries(regions: list[SearchRegion], query_rows: list[dict[str, object]], *, top_k: int) -> dict[str, object]:
    index = PortableBM25Index(regions, include_source=True)
    rows = []
    for query_row in query_rows:
        hits = index.search(str(query_row["query"]), top_k=top_k)
        expected = [str(item) for item in query_row.get("expected_prefixes", [])]
        rank = hit_rank(hits, expected)
        rows.append(
            {
                "category": str(query_row.get("category", "")),
                "query": str(query_row["query"]),
                "expected_prefixes": expected,
                "rank": rank,
                "note": str(query_row.get("note", "")),
                "top_hits": [
                    {
                        "stable_ref": hit.stable_ref,
                        "score": hit.score,
                        "summary": hit.summary,
                    }
                    for hit in hits
                ],
            }
        )
    expected_rows = [row for row in rows if row["category"] == "expected"]
    return {
        "queries": len(rows),
        "expected_queries": len(expected_rows),
        "expected_hit_at_1": rate(expected_rows, lambda row: row["rank"] == 1),
        "expected_hit_at_3": rate(expected_rows, lambda row: row["rank"] is not None and row["rank"] <= 3),
        "expected_hit_at_5": rate(expected_rows, lambda row: row["rank"] is not None and row["rank"] <= 5),
        "rows": rows,
    }


def load_aliases(path: str | Path) -> list[ConcernAlias]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    aliases: list[ConcernAlias] = []
    for item in payload:
        aliases.append(
            ConcernAlias(
                id=str(item["id"]),
                category=str(item.get("category", "expected")),
                expected_prefixes=[str(ref) for ref in item.get("expected_prefixes", [])],
                queries=[str(query) for query in item.get("queries", [])],
                aliases=[str(alias) for alias in item.get("aliases", [])],
                note=str(item.get("note", "")),
            )
        )
    return aliases


def load_break_queries(path: str | Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [dict(item) for item in payload]


def alias_query_rows(aliases: list[ConcernAlias]) -> list[dict[str, object]]:
    rows = []
    for alias in aliases:
        for query in alias.queries:
            rows.append(
                {
                    "category": alias.category,
                    "query": query,
                    "expected_prefixes": alias.expected_prefixes,
                    "note": f"{alias.id}: {alias.note}",
                }
            )
    return rows


def write_portable_index(
    regions: list[SearchRegion],
    path: Path,
    *,
    source_index: str,
    aliases_path: str | None,
) -> None:
    payload = {
        "schema": "refmark.portable_search_index.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_corpus": source_index,
        "settings": {
            "view_source": "bgb-article-concern-navigation" if aliases_path else "bgb-article",
            "model": "curated-concern-aliases" if aliases_path else "none",
            "chunker": "bgb-article",
            "include_source_in_index": True,
            "aliases_path": aliases_path,
        },
        "stats": {
            "documents": len({region.doc_id for region in regions}),
            "regions": len(regions),
            "approx_input_tokens": sum(max(1, len(region.text) // 4) for region in regions),
            "approx_output_tokens": sum(
                max(1, len(region.view.summary) // 4)
                + sum(max(1, len(question) // 4) for question in region.view.questions)
                + sum(max(1, len(keyword) // 4) for keyword in region.view.keywords)
                for region in regions
            ),
        },
        "regions": [region.to_dict() for region in regions],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def article_id_for(region_id: str) -> str:
    return ARTICLE_SUFFIX_RE.sub("", region_id)


def hit_rank(hits: Iterable[SearchHit], expected_prefixes: list[str]) -> int | None:
    if not expected_prefixes:
        return None
    for index, hit in enumerate(hits, start=1):
        if any(hit.stable_ref.startswith(prefix) for prefix in expected_prefixes):
            return index
    return None


def _alias_text_by_ref(aliases: list[ConcernAlias]) -> dict[str, list[str]]:
    by_ref: dict[str, list[str]] = defaultdict(list)
    for alias in aliases:
        text = alias.aliases
        for expected_ref in alias.expected_prefixes:
            by_ref[expected_ref].extend(text)
    return {ref: unique(values) for ref, values in by_ref.items()}


def keyword_phrases(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        tokens = [token for token in tokenize(value) if len(token) > 2]
        output.extend(tokens)
    return unique(output)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rate(rows: list[dict[str, object]], predicate) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4)


def unique(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            output.append(clean)
            seen.add(clean)
    return output


if __name__ == "__main__":
    main()
