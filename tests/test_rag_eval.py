from refmark.pipeline import RegionRecord
from refmark.provenance import build_eval_provenance, validate_provenance
from refmark.rag_eval import CorpusMap, EvalExample, EvalSuite


def _record(region_id: str, text: str, ordinal: int, *, doc_id: str = "policy") -> RegionRecord:
    return RegionRecord(
        doc_id=doc_id,
        region_id=region_id,
        text=text,
        start_line=ordinal,
        end_line=ordinal,
        ordinal=ordinal,
        hash=f"h-{region_id}-{len(text)}",
    )


def test_eval_suite_compares_arbitrary_retrievers_with_context_refs():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 30 days.", 1),
            _record("P02", "Expedited shipping is non-refundable.", 2),
            _record("P03", "Audit logs are retained for 180 days.", 3),
        ]
    )
    suite = EvalSuite(
        examples=[
            EvalExample("Which clause covers expedited shipping?", ["policy:P02"]),
            EvalExample("Which clauses describe refunds and shipping?", ["policy:P01-P02"]),
        ],
        corpus=corpus,
    )

    runs = suite.compare(
        {
            "direct": lambda query: ["policy:P02"] if "expedited" in query else ["policy:P01", "policy:P02"],
            "expanded": lambda _query: [{"stable_ref": "policy:P01", "context_refs": ["policy:P01", "policy:P02"]}],
        },
        k=2,
    )

    assert runs["direct"].metrics["hit_at_1"] == 1.0
    assert runs["direct"].metrics["gold_coverage"] == 1.0
    assert runs["expanded"].metrics["hit_at_k"] == 1.0
    assert runs["expanded"].metrics["avg_context_refs"] == 2.0


def test_eval_suite_detects_stale_examples_from_region_hashes():
    original = CorpusMap.from_records([_record("P01", "Old retention text.", 1)])
    current = CorpusMap.from_records([_record("P01", "New retention text with changed hash.", 1)])
    example = EvalExample("How long are logs retained?", ["policy:P01"]).with_source_hashes(original)
    suite = EvalSuite(examples=[example], corpus=current)

    stale = suite.stale_examples()

    assert len(stale) == 1
    assert stale[0].changed_refs == ["policy:P01"]
    assert stale[0].missing_refs == []


def test_corpus_map_reports_added_removed_and_changed_refs():
    previous = CorpusMap.from_records(
        [
            _record("P01", "Alpha", 1),
            _record("P02", "Beta", 2),
        ]
    )
    current = CorpusMap.from_records(
        [
            _record("P01", "Alpha changed", 1),
            _record("P03", "Gamma", 3),
        ]
    )

    diff = current.changed_refs(previous)

    assert diff["changed"] == ["policy:P01"]
    assert diff["removed"] == ["policy:P02"]
    assert diff["added"] == ["policy:P03"]


def test_corpus_map_uses_strict_citation_parser_for_ranges_and_doc_ids():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Alpha", 1, doc_id="handbook"),
            _record("P02", "Beta", 2, doc_id="handbook"),
            _record("P03", "Gamma", 3, doc_id="handbook"),
        ]
    )

    assert corpus.expand_refs(["[handbook:P01-handbook:P03]"]) == [
        "handbook:P01",
        "handbook:P02",
        "handbook:P03",
    ]


def test_context_pack_orders_refs_and_includes_headers():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Alpha evidence.", 1),
            _record("P02", "Beta evidence.", 2),
        ]
    )

    pack = corpus.context_pack(["policy:P01-P02"])

    assert pack.refs == ["policy:P01", "policy:P02"]
    assert "[policy:P01]" in pack.text
    assert "Beta evidence." in pack.text
    assert pack.token_estimate > 0


def test_eval_run_reports_heatmap_and_confidence_gating():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 30 days.", 1),
            _record("P02", "Expedited shipping is non-refundable.", 2),
        ]
    )
    suite = EvalSuite(
        examples=[
            EvalExample("refund window?", ["policy:P01"]),
            EvalExample("shipping refund?", ["policy:P02"]),
        ],
        corpus=corpus,
    )

    run = suite.evaluate(
        lambda query: (
            [{"stable_ref": "policy:P01", "score": 4.0}, {"stable_ref": "policy:P02", "score": 3.0}]
            if "refund window" in query
            else [{"stable_ref": "policy:P01", "score": 5.0}, {"stable_ref": "policy:P02", "score": 4.9}]
        ),
        k=2,
    )

    assert run.examples[0].score_margin == 1.0
    assert run.diagnostics["heatmap"]["missed_queries"] == 0
    assert run.diagnostics["heatmap"]["confusions"][0] == {
        "gold_ref": "policy:P02",
        "top_ref": "policy:P01",
        "count": 1,
    }
    assert run.diagnostics["selective_jump"]["confidence_field"] == "score_margin_ratio"
    assert run.diagnostics["adaptation"][0]["action"] == "add_hard_negative_or_disambiguating_signature"


def test_eval_provenance_detects_input_changes(tmp_path):
    index_path = tmp_path / "index.json"
    examples_path = tmp_path / "examples.jsonl"
    index_path.write_text('{"schema":"refmark.portable_search_index.v1","regions":[]}', encoding="utf-8")
    examples_path.write_text('{"query":"q","gold_refs":["doc:P01"]}\n', encoding="utf-8")

    first = build_eval_provenance(
        index_path=index_path,
        examples_path=examples_path,
        settings={"top_k": 10},
    )
    examples_path.write_text('{"query":"changed","gold_refs":["doc:P01"]}\n', encoding="utf-8")
    second = build_eval_provenance(
        index_path=index_path,
        examples_path=examples_path,
        settings={"top_k": 10},
    )

    check = validate_provenance(first, second)

    assert not check["ok"]
    assert check["mismatches"][0]["path"] == "artifacts.examples.sha256"
