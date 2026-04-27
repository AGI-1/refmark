"""Fetch public documentation corpora for portable search-index evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib import request


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output" / "online_corpora"


SOURCES = [
    {
        "id": "fastapi_docs",
        "category": "api_framework",
        "repo": "fastapi/fastapi",
        "ref": "master",
        "prefixes": ["docs/en/docs/"],
        "suffixes": [".md"],
        "limit": 220,
    },
    {
        "id": "django_docs",
        "category": "web_framework",
        "repo": "django/django",
        "ref": "main",
        "prefixes": ["docs/"],
        "suffixes": [".txt"],
        "limit": 260,
    },
    {
        "id": "kubernetes_docs",
        "category": "platform_ops",
        "repo": "kubernetes/website",
        "ref": "main",
        "prefixes": ["content/en/docs/concepts/", "content/en/docs/tasks/"],
        "suffixes": [".md"],
        "limit": 260,
    },
    {
        "id": "rust_book",
        "category": "language_book",
        "repo": "rust-lang/book",
        "ref": "main",
        "prefixes": ["src/"],
        "suffixes": [".md"],
        "limit": 80,
    },
    {
        "id": "typescript_handbook",
        "category": "language_reference",
        "repo": "microsoft/TypeScript-Website",
        "ref": "v2",
        "prefixes": ["packages/documentation/copy/en/handbook-v2/"],
        "suffixes": [".md"],
        "limit": 120,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public docs for Refmark search evaluation.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source", action="append", default=None, help="Optional source id to fetch. Repeatable.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = [source for source in SOURCES if args.source is None or source["id"] in set(args.source)]
    manifest = []
    for source in selected:
        print(f"fetching {source['id']} from {source['repo']}")
        rows = fetch_source(source, output_dir)
        manifest.extend(rows)
        write_combined(source, output_dir, rows)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_category_sets(output_dir, manifest)
    print(f"wrote {manifest_path}")


def fetch_source(source: dict, output_dir: Path) -> list[dict[str, object]]:
    owner_repo = str(source["repo"])
    ref = str(source["ref"])
    tree_url = f"https://api.github.com/repos/{owner_repo}/git/trees/{ref}?recursive=1"
    tree = json.loads(fetch_bytes(tree_url).decode("utf-8"))
    paths = []
    for item in tree.get("tree", []):
        path = str(item.get("path", ""))
        if item.get("type") != "blob":
            continue
        if not any(path.startswith(prefix) for prefix in source["prefixes"]):
            continue
        if not any(path.endswith(suffix) for suffix in source["suffixes"]):
            continue
        if _skip_path(path):
            continue
        paths.append(path)
    paths = sorted(paths)[: int(source["limit"])]
    rows: list[dict[str, object]] = []
    source_dir = output_dir / str(source["id"])
    source_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        raw_url = f"https://raw.githubusercontent.com/{owner_repo}/{ref}/{quote(path)}"
        payload = fetch_bytes(raw_url)
        text = normalize_text(payload.decode("utf-8", errors="replace"))
        doc_id = slugify(path)
        text_path = source_dir / f"{doc_id}.txt"
        text_path.write_text(text, encoding="utf-8")
        rows.append(
            {
                "source_id": source["id"],
                "category": source["category"],
                "repo": owner_repo,
                "ref": ref,
                "path": path,
                "url": raw_url,
                "text_path": str(text_path),
                "text_sha256": sha256_text(text),
                "chars": len(text),
                "words": len(text.split()),
            }
        )
    (source_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def write_combined(source: dict, output_dir: Path, rows: list[dict[str, object]]) -> None:
    parts = []
    for row in rows:
        text = Path(str(row["text_path"])).read_text(encoding="utf-8")
        parts.append(f"# Source: {row['source_id']}\n# Path: {row['path']}\n# URL: {row['url']}\n\n{text}")
    combined = "\n\n".join(parts)
    path = output_dir / f"{source['id']}_combined.txt"
    path.write_text(combined, encoding="utf-8")


def write_category_sets(output_dir: Path, rows: list[dict[str, object]]) -> None:
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(row)
    set_rows = []
    for category, category_rows in sorted(by_category.items()):
        parts = []
        for row in category_rows:
            text = Path(str(row["text_path"])).read_text(encoding="utf-8")
            parts.append(f"# Source: {row['source_id']}\n# Path: {row['path']}\n# URL: {row['url']}\n\n{text}")
        combined = "\n\n".join(parts)
        path = output_dir / f"{category}_combined.txt"
        path.write_text(combined, encoding="utf-8")
        set_rows.append(
            {
                "category": category,
                "text_path": str(path),
                "documents": len(category_rows),
                "words": sum(int(row["words"]) for row in category_rows),
                "text_sha256": sha256_text(combined),
            }
        )
    (output_dir / "sets_manifest.json").write_text(json.dumps(set_rows, indent=2), encoding="utf-8")


def fetch_bytes(url: str) -> bytes:
    req = request.Request(url, headers={"User-Agent": "refmark-online-corpus-eval/0.1"})
    with request.urlopen(req, timeout=120) as response:
        return response.read()


def normalize_text(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{<.*?>\}\}", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + "\n"


def slugify(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_").lower()[:140]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _skip_path(path: str) -> bool:
    lowered = path.lower()
    skip_parts = ["/img/", "/images/", "/assets/", "/includes/", "/_partials/", "/translations/"]
    return any(part in lowered for part in skip_parts)


if __name__ == "__main__":
    main()
