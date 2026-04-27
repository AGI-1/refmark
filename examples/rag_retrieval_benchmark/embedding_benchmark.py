"""Evaluate refmark retrieval with cached embeddings and an in-process vector store."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rag_retrieval_benchmark.common import (
    DEFAULT_DATA_DIR,
    OUTPUT_DIR,
    RetrievalUnit,
    anchor_units,
    enriched_anchor_units,
    expanded_anchor_units,
    fixed_window_units,
    load_jsonl,
    view_anchor_units,
)
from refmark_train.synthetic import tokenize


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark refmark retrieval with embeddings.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", choices=["valid", "reformulated"], default="valid")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sample-mode", choices=["first", "even", "random"], default="even")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument(
        "--unit-mode",
        choices=["anchors", "expanded", "enriched", "naive", "views"],
        default="anchors",
    )
    parser.add_argument("--expand", type=int, default=1)
    parser.add_argument("--chunk-tokens", type=int, default=220)
    parser.add_argument("--chunk-stride", type=int, default=110)
    parser.add_argument("--views-jsonl")
    parser.add_argument("--view-source-only", action="store_true")
    parser.add_argument("--source", choices=["hashing", "openrouter"], default="hashing")
    parser.add_argument("--hash-dim", type=int, default=512)
    parser.add_argument("--model", default="perplexity/pplx-embed-v1-0.6b")
    parser.add_argument("--endpoint", default=OPENROUTER_EMBEDDINGS_URL)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--cache", default=str(OUTPUT_DIR / "embedding_cache.jsonl"))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "embedding_benchmark.json"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    anchors = load_jsonl(data_dir / "anchors.jsonl")
    train = load_jsonl(data_dir / "train.jsonl")
    examples = _select_examples(
        load_jsonl(data_dir / f"{args.split}.jsonl"),
        limit=args.limit,
        mode=args.sample_mode,
        seed=args.seed,
    )
    units = _units_for_mode(args, anchors, train)
    top_ks = tuple(int(part) for part in args.top_ks.split(",") if part.strip())

    embedder = Embedder(args)
    unit_embeddings = embedder.embed_many([unit.text for unit in units])
    query_embeddings = embedder.embed_many([str(row["question"]) for row in examples])
    report = _evaluate_vectors(units, examples, unit_embeddings, query_embeddings, top_ks=top_ks)

    payload = {
        "data_dir": str(data_dir),
        "split": args.split,
        "examples": len(examples),
        "anchors": len(anchors),
        "unit_mode": args.unit_mode,
        "source": args.source,
        "model": args.model if args.source == "openrouter" else f"hashing-{args.hash_dim}",
        "settings": {
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "top_ks": top_ks,
            "batch_size": args.batch_size,
            "cache": args.cache,
        },
        "report": report,
        "interpretation": [
            "This is a lightweight vector-store benchmark: embeddings are cached, normalized, and searched by cosine similarity.",
            "Use unit_mode=anchors for precise refmark retrieval, expanded for child-to-parent context, and enriched/views for metadata-aided retrieval.",
            "The hashing source is a no-cost deterministic smoke baseline; openrouter source measures actual embedding models.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote embedding benchmark to {output}")


class Embedder:
    def __init__(self, args) -> None:
        self.args = args
        self.cache_path = Path(args.cache)
        self.cache: dict[str, list[float]] = {}
        with _cache_lock(self.cache_path):
            if self.cache_path.exists():
                for row in load_jsonl(self.cache_path):
                    self.cache[str(row["key"])] = [float(value) for value in row["embedding"]]

    def embed_many(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(text) for text in texts]
        missing = [(key, text) for key, text in zip(keys, texts, strict=True) if key not in self.cache]
        if missing:
            if self.args.source == "openrouter":
                self._fill_openrouter(missing)
            else:
                self._fill_hashing(missing)
        vectors = np.array([self.cache[key] for key in keys], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def _key(self, text: str) -> str:
        payload = f"{self.args.source}:{self.args.model}:{self.args.hash_dim}:{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _fill_hashing(self, missing: list[tuple[str, str]]) -> None:
        rows = []
        for key, text in missing:
            vector = _hash_embedding(text, dim=self.args.hash_dim)
            self.cache[key] = vector
            rows.append({"key": key, "source": "hashing", "embedding": vector})
        self._append_rows(rows)

    def _fill_openrouter(self, missing: list[tuple[str, str]]) -> None:
        api_key = os.environ.get(self.args.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.args.api_key_env} is not set.")
        for start in range(0, len(missing), self.args.batch_size):
            batch = missing[start : start + self.args.batch_size]
            payload = {
                "model": self.args.model,
                "input": [text for _key, text in batch],
                "encoding_format": "float",
            }
            request = Request(
                self.args.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            rows = []
            for (key, _text), item in zip(batch, data["data"], strict=True):
                vector = [float(value) for value in item["embedding"]]
                self.cache[key] = vector
                rows.append({"key": key, "source": self.args.model, "embedding": vector})
            self._append_rows(rows)
            print(f"embedded {min(start + len(batch), len(missing))}/{len(missing)} missing texts")
            if self.args.sleep:
                time.sleep(self.args.sleep)

    def _append_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with _cache_lock(self.cache_path):
            with self.cache_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _units_for_mode(args, anchors: list[dict], train: list[dict]) -> list[RetrievalUnit]:
    if args.unit_mode == "anchors":
        return anchor_units(anchors)
    if args.unit_mode == "expanded":
        return expanded_anchor_units(anchors, margin=args.expand)
    if args.unit_mode == "enriched":
        return enriched_anchor_units(anchors, train)
    if args.unit_mode == "naive":
        return fixed_window_units(anchors, chunk_tokens=args.chunk_tokens, stride=args.chunk_stride)
    if not args.views_jsonl:
        raise RuntimeError("--views-jsonl is required with --unit-mode views.")
    return view_anchor_units(anchors, load_jsonl(Path(args.views_jsonl)), include_source=not args.view_source_only)


def _evaluate_vectors(
    units: list[RetrievalUnit],
    examples: list[dict],
    unit_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    top_ks: tuple[int, ...],
) -> dict[str, object]:
    max_k = max(top_ks)
    hits = {k: 0 for k in top_ks}
    reciprocal_sum = 0.0
    token_cost = {k: 0 for k in top_ks}
    refs_returned = {k: 0 for k in top_ks}
    misses: list[dict[str, object]] = []
    scores = query_embeddings @ unit_embeddings.T
    for row, row_scores in zip(examples, scores, strict=True):
        gold = str(row["refmark"])
        order = np.argsort(-row_scores, kind="stable")[:max_k]
        rank = None
        for idx, unit_index in enumerate(order, start=1):
            if gold in units[int(unit_index)].refs:
                rank = idx
                break
        if rank is not None:
            reciprocal_sum += 1.0 / rank
        else:
            misses.append(
                {
                    "question": row["question"],
                    "gold": gold,
                    "top_units": [units[int(index)].unit_id for index in order[:3]],
                    "top_refs": [list(units[int(index)].refs)[:6] for index in order[:3]],
                }
            )
        for k in top_ks:
            selected = [units[int(index)] for index in order[:k]]
            returned_refs = {ref for unit in selected for ref in unit.refs}
            if gold in returned_refs:
                hits[k] += 1
            token_cost[k] += sum(unit.token_count for unit in selected)
            refs_returned[k] += len(returned_refs)
    total = max(len(examples), 1)
    return {
        "unit_count": len(units),
        "avg_unit_tokens": round(sum(unit.token_count for unit in units) / max(len(units), 1), 2),
        "mrr": round(reciprocal_sum / total, 4),
        "hit_at_k": {str(k): round(hits[k] / total, 4) for k in top_ks},
        "avg_token_cost_at_k": {str(k): round(token_cost[k] / total, 2) for k in top_ks},
        "avg_refs_returned_at_k": {str(k): round(refs_returned[k] / total, 2) for k in top_ks},
        "sample_misses": misses[:8],
    }


def _hash_embedding(text: str, *, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * (1.0 + math.log1p(len(token)))
    return vector


@contextmanager
def _cache_lock(cache_path: Path):
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    while not acquired:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _select_examples(examples: list[dict], *, limit: int, mode: str, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(examples):
        return examples
    if mode == "first":
        return examples[:limit]
    if mode == "random":
        rng = random.Random(seed)
        indexes = sorted(rng.sample(range(len(examples)), limit))
        return [examples[index] for index in indexes]
    if limit == 1:
        return [examples[0]]
    indexes = sorted({round(index * (len(examples) - 1) / (limit - 1)) for index in range(limit)})
    return [examples[index] for index in indexes[:limit]]


if __name__ == "__main__":
    main()
