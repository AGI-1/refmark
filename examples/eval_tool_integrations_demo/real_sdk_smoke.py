from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from refmark import export_deepeval_cases, export_ragas_rows, export_trace_events

from examples.eval_tool_integrations_demo.run import OUTPUT, build_demo


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suite, run, answers = build_demo()
    report = {
        "schema": "refmark.real_sdk_smoke.v1",
        "note": "Optional smoke test for installed third-party SDKs; hosted ingestion is skipped unless env/server config is present.",
        "tools": {
            "ragas": _check_ragas(suite, run, answers),
            "deepeval": _check_deepeval(suite, run, answers),
            "langfuse": _check_langfuse(suite, run, answers),
            "phoenix_openinference": _check_phoenix_openinference(suite, run, answers),
        },
    }
    path = OUTPUT / "real_sdk_smoke.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def _check_ragas(suite: Any, run: Any, answers: dict[str, str]) -> dict[str, Any]:
    rows = export_ragas_rows(suite, run, answers=answers)

    def construct() -> dict[str, Any]:
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

        samples = [
            SingleTurnSample(
                user_input=row["user_input"],
                response=row["response"],
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["reference"],
                retrieved_context_ids=row["refmark"]["context_refs"],
                reference_context_ids=row["refmark"]["gold_refs"],
            )
            for row in rows
        ]
        dataset = EvaluationDataset(samples=samples, name="refmark-smoke")
        first = dataset.samples[0]
        return {
            "native_class": type(dataset).__name__,
            "sample_class": type(first).__name__,
            "sample_count": len(dataset.samples),
            "refs_preserved": first.retrieved_context_ids == rows[0]["refmark"]["context_refs"],
            "source_hashes_available": bool(rows[0]["refmark"]["source_hashes"]),
        }

    return _run_optional("ragas", construct)


def _check_deepeval(suite: Any, run: Any, answers: dict[str, str]) -> dict[str, Any]:
    rows = export_deepeval_cases(suite, run, answers=answers)

    def construct() -> dict[str, Any]:
        from deepeval.test_case import LLMTestCase

        cases = [
            LLMTestCase(
                input=row["input"],
                actual_output=row["actual_output"],
                expected_output=row["expected_output"],
                retrieval_context=row["retrieval_context"],
                context=row["context"],
                metadata={"refmark": row["refmark"]},
                tags=["refmark-smoke", row["refmark"].get("query_style") or "unknown"],
            )
            for row in rows
        ]
        first = cases[0]
        return {
            "native_class": type(first).__name__,
            "case_count": len(cases),
            "refs_preserved": first.metadata["refmark"]["context_refs"] == rows[0]["refmark"]["context_refs"],
            "source_hashes_available": bool(first.metadata["refmark"]["source_hashes"]),
        }

    return _run_optional("deepeval", construct)


def _check_langfuse(suite: Any, run: Any, answers: dict[str, str]) -> dict[str, Any]:
    events = export_trace_events(suite, run, tool="langfuse", answers=answers)

    def construct() -> dict[str, Any]:
        from langfuse import Langfuse

        event = events[0]
        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-refmark-smoke"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-refmark-smoke"),
            host=os.getenv("LANGFUSE_HOST", "http://127.0.0.1:1"),
            tracing_enabled=False,
        )
        with client.start_as_current_observation(
            name=event["span_name"],
            as_type="retriever",
            input=event["input"],
            output=event["output"],
            metadata=event["attributes"],
        ):
            pass
        client.shutdown()
        return {
            "native_class": type(client).__name__,
            "event_count": len(events),
            "refs_preserved": event["attributes"]["refmark.context_refs"] == run.examples[0].context_refs,
            "fingerprints_preserved": bool(event["attributes"]["refmark.corpus_fingerprint"]),
            "hosted_ingestion": "skipped; set LANGFUSE_* and enable tracing in a caller-owned run",
        }

    return _run_optional("langfuse", construct)


def _check_phoenix_openinference(suite: Any, run: Any, answers: dict[str, str]) -> dict[str, Any]:
    events = export_trace_events(suite, run, tool="phoenix", answers=answers)

    def construct() -> dict[str, Any]:
        from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        event = events[0]
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("refmark-smoke")
        with tracer.start_as_current_span(event["span_name"]) as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
            span.set_attribute("input.value", event["input"])
            for key, value in event["attributes"].items():
                span.set_attribute(key, _otel_value(value))
        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        provider.shutdown()
        return {
            "native_class": type(spans[0]).__name__,
            "span_count": len(spans),
            "span_kind": attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND),
            "refs_preserved": json.loads(attrs["refmark.context_refs"]) == run.examples[0].context_refs,
            "fingerprints_preserved": bool(attrs["refmark.corpus_fingerprint"]),
            "hosted_ingestion": "skipped; Phoenix server/OTLP endpoint not required for this in-memory span check",
        }

    return _run_optional("phoenix.otel", construct)


def _run_optional(import_name: str, func: Any) -> dict[str, Any]:
    try:
        __import__(import_name)
    except Exception as exc:
        return {"status": "not_installed", "error": f"{type(exc).__name__}: {exc}"}
    try:
        detail = func()
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc().splitlines()[-5:],
        }
    return {"status": "ok", **detail}


def _otel_value(value: Any) -> str | bool | int | float:
    if isinstance(value, str | bool | int | float):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
