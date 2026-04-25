from refmark.metrics import citation_reward, expand_refs, normalize_ref, score_ref_range, summarize_scores


def test_score_ref_range_detects_overcite_and_undercite():
    overcite = score_ref_range(["F02", "F03"], ["F03"])
    undercite = score_ref_range(["F05"], ["F05", "F06"])

    assert overcite.data_smell == "overcite"
    assert overcite.cover == 1.0
    assert overcite.precision == 0.5
    assert undercite.data_smell == "undercite"
    assert undercite.cover == 0.5


def test_score_ref_range_detects_wrong_location():
    score = score_ref_range(["F10"], ["F03"])

    assert score.wrong_location is True
    assert score.data_smell == "wrong_location"
    assert score.overlap == 0.0


def test_expand_refs_uses_address_space_order_for_ranges():
    assert expand_refs(["F03-F01"], address_space=["F01", "F02", "F03"]) == ["F01", "F02", "F03"]


def test_refs_preserve_existing_numeric_width():
    assert normalize_ref("d00284") == "D00284"
    assert expand_refs(["D00283-D00285"]) == ["D00283", "D00284", "D00285"]


def test_summarize_scores_reports_rates():
    scores = [
        score_ref_range(["F01"], ["F01"]),
        score_ref_range(["F01", "F02"], ["F02"]),
        score_ref_range(["F09"], ["F03"]),
    ]
    summary = summarize_scores(scores)

    assert summary["count"] == 3
    assert round(summary["exact_match"], 3) == 0.333
    assert round(summary["overcite_rate"], 3) == 0.333
    assert round(summary["wrong_location_rate"], 3) == 0.333


def test_citation_reward_is_continuous_and_judge_free():
    exact = citation_reward(score_ref_range(["F03"], ["F03"]))
    overcite = citation_reward(score_ref_range(["F02", "F03"], ["F03"]))
    undercite = citation_reward(score_ref_range(["F05"], ["F05", "F06"]))
    wrong = citation_reward(score_ref_range(["F10"], ["F03"]))

    assert exact == 1.0
    assert exact > overcite > wrong
    assert exact > undercite > wrong
