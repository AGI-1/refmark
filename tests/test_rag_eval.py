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


def test_eval_suite_loads_jsonl_and_writes_run_report(tmp_path):
    corpus = CorpusMap.from_records([_record("P01", "Refunds are available within 30 days.", 1)])
    examples_path = tmp_path / "examples.jsonl"
    report_path = tmp_path / "run.json"
    examples_path.write_text(
        '{"query":"refund window?","gold_refs":["policy:P01"]}\n',
        encoding="utf-8",
    )

    suite = EvalSuite.from_jsonl(examples_path, corpus=corpus, attach_source_hashes=True)
    run = suite.evaluate(lambda _query: [{"stable_ref": "policy:P01", "score": 1.0}], k=1)
    run.write_json(report_path)

    assert suite.examples[0].source_hashes == {"policy:P01": "h-P01-37"}
    assert report_path.exists()
    assert '"hit_at_k": 1.0' in report_path.read_text(encoding="utf-8")


def test_eval_suite_builds_comparable_run_artifact():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 30 days.", 1),
            _record("P02", "Expedited shipping is non-refundable.", 2),
        ],
        revision_id="git:abc123",
        metadata={"source_path": "docs/policy.md"},
    )
    suite = EvalSuite(
        examples=[
            EvalExample("refund window?", ["policy:P01"], metadata={"query_style": "direct"}).with_source_hashes(corpus),
            EvalExample("shipping refund?", ["policy:P02"], metadata={"query_style": "concern"}).with_source_hashes(corpus),
        ],
        corpus=corpus,
    )

    run = suite.evaluate(lambda query: ["policy:P01"] if "window" in query else ["policy:P02"], k=1)
    artifact = suite.run_artifact(run, settings={"strategy": "test", "top_k": 1}, artifacts={"examples": "eval.jsonl"})

    assert artifact["schema"] == "refmark.eval_run_artifact.v1"
    assert artifact["corpus"]["fingerprint"] == corpus.fingerprint
    assert artifact["corpus"]["revision_id"] == "git:abc123"
    assert artifact["eval_suite"]["fingerprint"] == suite.fingerprint
    assert artifact["eval_suite"]["query_styles"] == {"concern": 1, "direct": 1}
    assert artifact["eval_suite"]["source_hash_coverage"] == 1.0
    assert artifact["run_fingerprint"] == run.fingerprint
    assert artifact["comparison_key"] == suite.run_artifact(
        run,
        settings={"strategy": "test", "top_k": 1},
        artifacts={"examples": "other.jsonl"},
    )["comparison_key"]
    assert artifact["comparison_key"] != suite.run_artifact(run, settings={"strategy": "other", "top_k": 1})["comparison_key"]


def test_eval_suite_preserves_saved_source_hashes_for_stale_checks(tmp_path):
    original = CorpusMap.from_records([_record("P01", "Original refund text.", 1)])
    current = CorpusMap.from_records([_record("P01", "Changed refund text.", 1)])
    examples_path = tmp_path / "examples.jsonl"
    saved_path = tmp_path / "saved.jsonl"
    suite = EvalSuite(
        examples=[EvalExample("refund?", ["policy:P01"]).with_source_hashes(original)],
        corpus=original,
    )
    suite.to_jsonl(examples_path)

    loaded = EvalSuite.from_jsonl(examples_path, corpus=current, attach_source_hashes=True)
    loaded.to_jsonl(saved_path)

    assert loaded.examples[0].source_hashes == {"policy:P01": "h-P01-21"}
    assert loaded.stale_examples()[0].changed_refs == ["policy:P01"]
    assert '"source_hashes"' in saved_path.read_text(encoding="utf-8")


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


def test_corpus_map_snapshots_external_revision_metadata():
    corpus = CorpusMap.from_records(
        [
            _record("P02", "Beta", 2),
            _record("P01", "Alpha", 1),
        ],
        revision_id="git:abc123",
        metadata={"source_path": "docs/policy.md"},
    )

    snapshot = corpus.snapshot()

    assert snapshot.revision_id == "git:abc123"
    assert snapshot.region_count == 2
    assert snapshot.stable_refs == ["policy:P01", "policy:P02"]
    assert snapshot.metadata == {"source_path": "docs/policy.md"}
    assert snapshot.to_dict()["fingerprint"] == corpus.fingerprint


def test_corpus_revision_diff_marks_examples_stale():
    previous = CorpusMap.from_records(
        [
            _record("P01", "Alpha", 1),
            _record("P02", "Beta", 2),
        ],
        revision_id="rev-a",
    )
    current = CorpusMap.from_records(
        [
            _record("P01", "Alpha changed", 1),
            _record("P03", "Gamma", 3),
        ],
        revision_id="rev-b",
    )
    changed_example = EvalExample("alpha?", ["policy:P01"]).with_source_hashes(previous)
    removed_example = EvalExample("beta?", ["policy:P02"]).with_source_hashes(previous)
    unaffected_example = EvalExample("new?", ["policy:P03"])

    diff = current.diff_revision(previous)
    stale = diff.stale_examples([changed_example, removed_example, unaffected_example])

    assert diff.previous_revision_id == "rev-a"
    assert diff.current_revision_id == "rev-b"
    assert diff.has_changes
    assert diff.affected_refs() == ["policy:P03", "policy:P02", "policy:P01"]
    assert [item.changed_refs for item in stale] == [["policy:P01"], []]
    assert [item.missing_refs for item in stale] == [[], ["policy:P02"]]
    assert diff.to_dict()["has_changes"] is True


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
    assert run.examples[0].gold_mode == "single"
    assert run.diagnostics["heatmap"]["missed_queries"] == 0
    assert run.diagnostics["by_gold_mode"]["single"]["count"] == 2.0
    assert run.diagnostics["heatmap"]["confusions"][0] == {
        "gold_ref": "policy:P02",
        "top_ref": "policy:P01",
        "count": 1,
    }
    assert run.diagnostics["selective_jump"]["confidence_field"] == "score_margin_ratio"
    assert run.diagnostics["adaptation"][0]["adaptation_type"] == "confusion_mapping"
    assert run.diagnostics["adaptation"][0]["action"] == "record_confusion_pair_and_review_resolution"
    assert "add_hard_negative_or_disambiguating_signature" in run.diagnostics["adaptation"][0]["candidate_actions"]


def test_eval_run_reports_query_style_gap():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Refunds are available within 30 days.", 1),
            _record("P02", "Expedited shipping is non-refundable.", 2),
        ]
    )
    suite = EvalSuite(
        examples=[
            EvalExample("refund window?", ["policy:P01"], metadata={"variant": "direct"}),
            EvalExample("my package cost was not returned", ["policy:P02"], metadata={"query_style": "concern"}),
        ],
        corpus=corpus,
    )

    run = suite.evaluate(
        lambda query: (
            [{"stable_ref": "policy:P01", "score": 4.0}]
            if "refund window" in query
            else [{"stable_ref": "policy:P01", "score": 4.0}]
        ),
        k=1,
    )

    assert [result.query_style for result in run.examples] == ["direct", "concern"]
    assert run.diagnostics["by_query_style"]["direct"]["hit_at_1"] == 1.0
    assert run.diagnostics["by_query_style"]["concern"]["hit_at_1"] == 0.0
    assert run.diagnostics["query_style_gap"]["hit_at_1_gap"] == 1.0
    assert run.diagnostics["query_style_gap"]["weakest_by_hit_at_1"] == {
        "style": "concern",
        "value": 0.0,
    }


def test_eval_run_reports_range_and_distributed_gold_modes():
    corpus = CorpusMap.from_records(
        [
            _record("P01", "Alpha evidence.", 1),
            _record("P02", "Beta evidence.", 2),
            _record("P03", "Gamma evidence.", 3),
        ]
    )
    suite = EvalSuite(
        examples=[
            EvalExample("range?", ["policy:P01-policy:P02"]),
            EvalExample("distributed?", ["policy:P01", "policy:P03"]),
        ],
        corpus=corpus,
    )

    run = suite.evaluate(lambda _query: ["policy:P01", "policy:P02", "policy:P03"], k=3)

    assert [result.gold_mode for result in run.examples] == ["range", "distributed"]
    assert run.diagnostics["by_gold_mode"]["range"]["gold_coverage"] == 1.0
    assert run.diagnostics["by_gold_mode"]["distributed"]["gold_coverage"] == 1.0


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
