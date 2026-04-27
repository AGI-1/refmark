from examples.bgb_browser_search.run_bgb_stress_eval import build_article_blocks, evaluate
from refmark.search_index import RetrievalView, SearchRegion, PortableBM25Index


def region(region_id: str, text: str, ordinal: int) -> SearchRegion:
    return SearchRegion(
        doc_id="bgb",
        region_id=region_id,
        text=text,
        hash=f"h{ordinal}",
        source_path=None,
        ordinal=ordinal,
        prev_region_id=None,
        next_region_id=None,
        view=RetrievalView(summary="", questions=[], keywords=[]),
    )


class Question:
    query = "broken purchased phone repair"
    block_id = "bgb:S_437"
    gold_refs = ["bgb:S_437_A01", "bgb:S_437_A02"]
    generator_model = "test-model"
    language = "en"
    style = "concern"


def test_build_article_blocks_groups_absatz_regions():
    blocks = build_article_blocks(
        [
            region("S_437_A01", "Buyer rights for defects.", 0),
            region("S_437_A02", "Repair replacement withdrawal.", 1),
            region("S_965", "Finder duties for lost property.", 2),
        ]
    )

    assert [block.block_id for block in blocks] == ["bgb:S_437", "bgb:S_965"]
    assert blocks[0].stable_refs == ["bgb:S_437_A01", "bgb:S_437_A02"]


def test_stress_eval_scores_article_hits():
    index = PortableBM25Index(
        [
            region("S_965", "lost property finder duties", 0),
            region("S_437_A01", "broken purchased phone repair replacement", 1),
        ]
    )

    report = evaluate(index, [Question()], top_ks=(1, 3))

    assert report["article_hit_at_k"]["hit_at_k"]["1"] == 1.0
    assert report["weakness_heatmap"]["misses"] == 0
