"""Small library-facing refmark interface with optional shadow persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from refmark.edit import _detect_premarked
from refmark.pipeline import RegionRecord, build_region_manifest
from refmark.regions import _parse_blocks_with_mode
from refmark.workflow_config import WorkflowConfig, resolve_workflow_config

RefmarkMode = Literal["auto", "live", "shadow"]


@dataclass(frozen=True)
class RefmarkResult:
    """Result of refmarking content for either embedded or shadow use."""

    doc_id: str
    content: str
    marked_view: str
    records: list[RegionRecord]
    namespace_mode: str
    source_hash: str
    config_fingerprint: str
    registry_path: str | None = None
    session_reset: bool = False
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.to_dict() for record in self.records]
        return payload

    @property
    def is_live(self) -> bool:
        return self.namespace_mode == "live"

    @property
    def is_shadow(self) -> bool:
        return self.namespace_mode == "shadow"


class RefmarkRegistry:
    """Filesystem registry for shadow-marked views and their region metadata."""

    def __init__(self, root: str | Path = ".refmark/registry"):
        self.root = Path(root)

    def load(self, key: str) -> RefmarkResult | None:
        path = self._entry_path(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return replace(_result_from_dict(payload), session_reset=False, registry_path=str(path))

    def save(self, key: str, result: RefmarkResult) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._entry_path(key)
        payload = result.to_dict()
        payload["registry_path"] = str(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._write_index(key, payload)
        return path

    def key_for(self, *, doc_id: str, source_hash: str, config_fingerprint: str) -> str:
        digest = hashlib.sha256(f"{doc_id}\0{source_hash}\0{config_fingerprint}".encode("utf-8")).hexdigest()
        return digest[:24]

    def _entry_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def _write_index(self, key: str, payload: dict[str, Any]) -> None:
        index_path = self.root / "index.jsonl"
        line = json.dumps(
            {
                "key": key,
                "doc_id": payload["doc_id"],
                "source_hash": payload["source_hash"],
                "config_fingerprint": payload["config_fingerprint"],
            },
            ensure_ascii=True,
        )
        existing = index_path.read_text(encoding="utf-8").splitlines() if index_path.exists() else []
        retained = [row for row in existing if not row.startswith(f'{{"key": "{key}"')]
        index_path.write_text("\n".join(retained + [line]) + "\n", encoding="utf-8")


class Refmarker:
    """Pass-through refmarker for application code.

    Live mode embeds markers into returned content. Shadow mode leaves content unchanged
    and persists the marked view plus manifest in a registry.
    """

    def __init__(
        self,
        *,
        config: WorkflowConfig | None = None,
        registry: RefmarkRegistry | None = None,
        registry_path: str | Path | None = None,
        mode: RefmarkMode = "shadow",
    ):
        self.config = resolve_workflow_config(config)
        self.registry = registry or RefmarkRegistry(registry_path or ".refmark/registry")
        self.mode = mode

    def mark_text(
        self,
        content: str,
        *,
        file_ext: str = ".txt",
        doc_id: str = "document",
        source_path: str | None = None,
        mode: RefmarkMode | None = None,
        **overrides: Any,
    ) -> RefmarkResult:
        resolved = resolve_workflow_config(self.config, **overrides)
        chosen_mode = mode or self.mode
        if chosen_mode == "auto":
            chosen_mode = "live" if _detect_premarked(content, marker_format=resolved.marker_format) else "shadow"
        if chosen_mode not in {"live", "shadow"}:
            raise ValueError(f"Unknown refmark mode '{chosen_mode}'.")

        source_hash = _source_hash(content)
        config_fingerprint = _config_fingerprint(resolved)

        if _detect_premarked(content, marker_format=resolved.marker_format):
            records = _records_from_marked(
                content,
                marker_format=resolved.marker_format,
                doc_id=doc_id,
                source_path=source_path,
            )
            return RefmarkResult(
                doc_id=doc_id,
                content=content,
                marked_view=content,
                records=records,
                namespace_mode="live",
                source_hash=source_hash,
                config_fingerprint=config_fingerprint,
                session_reset=False,
                warnings=[],
            )

        key = self.registry.key_for(
            doc_id=doc_id,
            source_hash=source_hash,
            config_fingerprint=config_fingerprint,
        )
        if chosen_mode == "shadow":
            cached = self.registry.load(key)
            if cached is not None:
                return cached

        marked_view, records = build_region_manifest(
            content,
            file_ext,
            doc_id=doc_id,
            source_path=source_path,
            marker_format=resolved.marker_format,
            chunker=resolved.chunker,
            chunker_kwargs=_chunker_kwargs(resolved),
            min_words=resolved.min_words,
        )
        namespace_mode = "shadow" if chosen_mode == "shadow" else "live"
        result = RefmarkResult(
            doc_id=doc_id,
            content=content if namespace_mode == "shadow" else marked_view,
            marked_view=marked_view,
            records=records,
            namespace_mode=namespace_mode,
            source_hash=source_hash,
            config_fingerprint=config_fingerprint,
            session_reset=True,
            warnings=[],
        )
        if namespace_mode == "shadow":
            registry_entry = self.registry.save(key, result)
            return replace(result, registry_path=str(registry_entry))
        return result

    def mark_path(
        self,
        path: str | Path,
        *,
        doc_id: str | None = None,
        mode: RefmarkMode | None = None,
        **overrides: Any,
    ) -> RefmarkResult:
        source = Path(path)
        return self.mark_text(
            source.read_text(encoding="utf-8-sig"),
            file_ext=source.suffix or ".txt",
            doc_id=doc_id or source.stem,
            source_path=str(source),
            mode=mode,
            **overrides,
        )


def _records_from_marked(
    marked: str,
    *,
    marker_format: str,
    doc_id: str,
    source_path: str | None,
) -> list[RegionRecord]:
    blocks = _parse_blocks_with_mode(marked, marker_format, line_mode="marked")
    ordered = sorted(blocks.items(), key=lambda item: int(item[1]["ordinal"]))
    records: list[RegionRecord] = []
    for index, (region_id, block) in enumerate(ordered):
        text = str(block.get("text", ""))
        records.append(
            RegionRecord(
                doc_id=doc_id,
                region_id=region_id,
                text=text,
                start_line=int(block["line_start"]),
                end_line=int(block["line_end"]),
                ordinal=int(block["ordinal"]),
                hash=_source_hash(text)[:16],
                source_path=source_path,
                prev_region_id=ordered[index - 1][0] if index > 0 else None,
                next_region_id=ordered[index + 1][0] if index + 1 < len(ordered) else None,
            )
        )
    return records


def _result_from_dict(payload: dict[str, Any]) -> RefmarkResult:
    records = [
        RegionRecord(
            doc_id=str(record["doc_id"]),
            region_id=str(record["region_id"]),
            text=str(record["text"]),
            start_line=int(record["start_line"]),
            end_line=int(record["end_line"]),
            ordinal=int(record["ordinal"]),
            hash=str(record["hash"]),
            source_path=record.get("source_path"),
            prev_region_id=record.get("prev_region_id"),
            next_region_id=record.get("next_region_id"),
            parent_region_id=record.get("parent_region_id"),
        )
        for record in payload.get("records", [])
    ]
    return RefmarkResult(
        doc_id=str(payload["doc_id"]),
        content=str(payload["content"]),
        marked_view=str(payload["marked_view"]),
        records=records,
        namespace_mode=str(payload["namespace_mode"]),
        source_hash=str(payload["source_hash"]),
        config_fingerprint=str(payload["config_fingerprint"]),
        registry_path=payload.get("registry_path"),
        session_reset=bool(payload.get("session_reset", False)),
        warnings=list(payload.get("warnings") or []),
    )


def _chunker_kwargs(config: WorkflowConfig) -> dict | None:
    if config.chunker == "line" and config.lines_per_chunk:
        return {"lines_per_chunk": config.lines_per_chunk}
    if config.chunker == "token" and config.tokens_per_chunk:
        return {"tokens_per_chunk": config.tokens_per_chunk}
    return None


def _source_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _config_fingerprint(config: WorkflowConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
