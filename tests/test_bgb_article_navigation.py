import importlib.util
from pathlib import Path
import sys

from refmark.search_index import RetrievalView, SearchRegion


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "bgb_browser_search" / "build_bgb_article_navigation.py"
SPEC = importlib.util.spec_from_file_location("build_bgb_article_navigation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bgb_nav = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bgb_nav
SPEC.loader.exec_module(bgb_nav)


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
        view=RetrievalView(summary=region_id, questions=[], keywords=[]),
    )


def test_article_id_for_groups_absatz_regions():
    assert bgb_nav.article_id_for("S_437_A01") == "S_437"
    assert bgb_nav.article_id_for("S_437") == "S_437"
    assert bgb_nav.article_id_for("SS_1_bis_20_A03") == "SS_1_bis_20"


def test_concern_aliases_make_layperson_query_measurable():
    aliases = [
        bgb_nav.ConcernAlias(
            id="broken_phone",
            category="expected",
            expected_prefixes=["bgb:S_437"],
            queries=["I bought a cellphone and found it is broken. What should I do?"],
            aliases=["broken cellphone after purchase", "defective phone repair replacement refund"],
            note="",
        )
    ]
    regions = [
        region("S_965", "Lost property finder duties and return obligations.", 0),
        region("S_437_A01", "Rights of the buyer in case of defects.", 1),
    ]

    article_regions = bgb_nav.build_article_regions(regions, aliases=aliases)
    report = bgb_nav.evaluate_queries(
        article_regions,
        [
            {
                "category": "expected",
                "query": "I bought a cellphone and found it is broken. What should I do?",
                "expected_prefixes": ["bgb:S_437"],
            }
        ],
        top_k=1,
    )

    assert report["expected_hit_at_1"] == 1.0
    buyer_view = next(region.view for region in article_regions if region.stable_ref == "bgb:S_437")
    assert "broken cellphone after purchase" in buyer_view.questions
    assert aliases[0].queries[0] not in buyer_view.questions
