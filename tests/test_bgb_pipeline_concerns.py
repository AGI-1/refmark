from examples.bgb_browser_search.build_bgb_article_navigation import ConcernAlias
from examples.bgb_browser_search.run_bgb_pipeline import _apply_concern_aliases_to_views, _concern_questions
from refmark.pipeline import RegionRecord
from refmark.search_index import RetrievalView


def record(region_id: str, text: str, ordinal: int) -> RegionRecord:
    return RegionRecord(
        doc_id="bgb",
        region_id=region_id,
        text=text,
        start_line=ordinal + 1,
        end_line=ordinal + 1,
        ordinal=ordinal,
        hash=f"h{ordinal}",
        source_path=None,
        prev_region_id=None,
        next_region_id=None,
    )


def test_concern_aliases_become_training_loop_questions():
    records = [
        record("S_437_A01", "Rights of buyer in case of defects.", 0),
        record("S_439", "Supplementary performance by repair or replacement.", 1),
        record("S_965", "Finder duties for lost property.", 2),
    ]
    aliases = [
        ConcernAlias(
            id="broken_phone",
            category="expected",
            expected_prefixes=["bgb:S_437", "bgb:S_439"],
            queries=["I bought a cellphone and found it is broken. What should I do?"],
            aliases=["defective phone"],
            note="",
        )
    ]

    questions = _concern_questions(records, aliases, source="curated", model="concern-aliases")

    assert len(questions) == 1
    assert questions[0].gold_mode == "concern"
    assert questions[0].query.startswith("I bought a cellphone")
    assert questions[0].gold_refs == ["bgb:S_437_A01", "bgb:S_439"]


def test_concern_alias_injection_keeps_eval_queries_held_out():
    records = [
        record("S_437_A01", "Rights of buyer in case of defects.", 0),
        record("S_965", "Finder duties for lost property.", 1),
    ]
    views = {
        ("bgb", "S_437_A01"): RetrievalView(summary="", questions=[], keywords=[]),
        ("bgb", "S_965"): RetrievalView(summary="", questions=[], keywords=[]),
    }
    aliases = [
        ConcernAlias(
            id="broken_phone",
            category="expected",
            expected_prefixes=["bgb:S_437"],
            queries=["I bought a cellphone and found it is broken. What should I do?"],
            aliases=["defective phone repair replacement refund"],
            note="",
        )
    ]

    updated = _apply_concern_aliases_to_views(records, views, aliases)

    target_view = updated[("bgb", "S_437_A01")]
    assert "defective phone repair replacement refund" in target_view.questions
    assert aliases[0].queries[0] not in target_view.questions
