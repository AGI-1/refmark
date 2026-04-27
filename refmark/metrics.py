"""Deterministic metrics for refmark locate-only citation evaluation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from refmark.citations import parse_citation_refs


REF_RE = re.compile(r"([A-Za-z]+)(\d+)")


@dataclass(frozen=True)
class RefRangeScore:
    predicted_refs: list[str]
    gold_refs: list[str]
    exact_match: float
    overlap: float
    cover: float
    precision: float
    recall: float
    f1: float
    breadth_ratio: float
    excess_ref_count: int
    overcite: bool
    undercite: bool
    wrong_location: bool
    overbroad_2x: bool
    data_smell: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RewardConfig:
    exact_bonus: float = 0.2
    f1_weight: float = 0.55
    cover_weight: float = 0.35
    breadth_penalty: float = 0.15
    wrong_location_penalty: float = 1.0
    undercite_penalty: float = 0.2



def normalize_ref(value: str) -> str:
    text = value.strip()
    match = REF_RE.fullmatch(text)
    if not match:
        return text.upper()
    prefix, digits = match.groups()
    width = max(2, len(digits))
    return f"{prefix.upper()}{int(digits):0{width}d}"


def expand_refs(refs: Iterable[str], address_space: Iterable[str] | None = None) -> list[str]:
    """Expand refs and simple ranges such as F03-F05 into normalized ref ids."""
    ordered = [normalize_ref(ref) for ref in address_space or []]
    index = {ref: idx for idx, ref in enumerate(ordered)}
    expanded: list[str] = []

    for citation in parse_citation_refs(refs):
        if citation.is_range:
            start = normalize_ref(citation.ref)
            end = normalize_ref(citation.end_ref or "")
            if start in index and end in index:
                lo = min(index[start], index[end])
                hi = max(index[start], index[end])
                expanded.extend(ordered[lo : hi + 1])
                continue
            expanded.extend(_expand_numeric_range(start, end))
            continue
        expanded.append(normalize_ref(citation.ref))

    return _dedupe(expanded)


def score_ref_range(
    predicted_refs: Iterable[str],
    gold_refs: Iterable[str],
    *,
    address_space: Iterable[str] | None = None,
) -> RefRangeScore:
    """Score one locate-only ref prediction and label common data smells."""
    predicted = expand_refs(predicted_refs, address_space)
    gold = expand_refs(gold_refs, address_space)
    predicted_set = set(predicted)
    gold_set = set(gold)

    if not predicted_set or not gold_set:
        return RefRangeScore(
            predicted_refs=predicted,
            gold_refs=gold,
            exact_match=0.0,
            overlap=0.0,
            cover=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            breadth_ratio=float(len(predicted_set)) / max(len(gold_set), 1),
            excess_ref_count=max(len(predicted_set) - len(gold_set), 0),
            overcite=bool(predicted_set and not gold_set.issuperset(predicted_set)),
            undercite=bool(gold_set and predicted_set and predicted_set < gold_set),
            wrong_location=not bool(predicted_set & gold_set),
            overbroad_2x=len(predicted_set) >= 2 * max(len(gold_set), 1),
            data_smell="missing_prediction" if not predicted_set else "missing_gold",
        )

    intersection = predicted_set & gold_set
    union = predicted_set | gold_set
    precision = len(intersection) / len(predicted_set)
    recall = len(intersection) / len(gold_set)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    breadth_ratio = len(predicted_set) / max(len(gold_set), 1)
    overcite = bool(intersection) and not predicted_set.issubset(gold_set)
    undercite = bool(intersection) and not gold_set.issubset(predicted_set)
    wrong_location = not bool(intersection)
    overbroad_2x = breadth_ratio >= 2.0

    if predicted_set == gold_set:
        smell = "exact"
    elif wrong_location:
        smell = "wrong_location"
    elif overcite and undercite:
        smell = "boundary_mismatch"
    elif overcite:
        smell = "overcite"
    elif undercite:
        smell = "undercite"
    else:
        smell = "partial_overlap"

    return RefRangeScore(
        predicted_refs=predicted,
        gold_refs=gold,
        exact_match=1.0 if predicted_set == gold_set else 0.0,
        overlap=len(intersection) / len(union),
        cover=recall,
        precision=precision,
        recall=recall,
        f1=f1,
        breadth_ratio=breadth_ratio,
        excess_ref_count=max(len(predicted_set) - len(gold_set), 0),
        overcite=overcite,
        undercite=undercite,
        wrong_location=wrong_location,
        overbroad_2x=overbroad_2x,
        data_smell=smell,
    )


def summarize_scores(scores: Iterable[RefRangeScore]) -> dict[str, float]:
    items = list(scores)
    count = len(items)
    if count == 0:
        return {
            "count": 0,
            "exact_match": 0.0,
            "overlap": 0.0,
            "cover": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "breadth_ratio": 0.0,
            "overcite_rate": 0.0,
            "undercite_rate": 0.0,
            "wrong_location_rate": 0.0,
            "overbroad_2x_rate": 0.0,
        }
    return {
        "count": count,
        "exact_match": _mean(score.exact_match for score in items),
        "overlap": _mean(score.overlap for score in items),
        "cover": _mean(score.cover for score in items),
        "precision": _mean(score.precision for score in items),
        "recall": _mean(score.recall for score in items),
        "f1": _mean(score.f1 for score in items),
        "breadth_ratio": _mean(score.breadth_ratio for score in items),
        "overcite_rate": _mean(1.0 if score.overcite else 0.0 for score in items),
        "undercite_rate": _mean(1.0 if score.undercite else 0.0 for score in items),
        "wrong_location_rate": _mean(1.0 if score.wrong_location else 0.0 for score in items),
        "overbroad_2x_rate": _mean(1.0 if score.overbroad_2x else 0.0 for score in items),
    }


def citation_reward(score: RefRangeScore, config: RewardConfig | None = None) -> float:
    """Return a deterministic continuous reward for citation-range training.

    This is suitable for local preference/reward experiments where the model
    emits refs and the reward should not require an LLM judge.
    """
    config = config or RewardConfig()
    if score.wrong_location:
        return 0.0
    reward = (
        config.f1_weight * score.f1
        + config.cover_weight * score.cover
        + (config.exact_bonus if score.exact_match else 0.0)
    )
    reward -= config.breadth_penalty * max(score.breadth_ratio - 1.0, 0.0)
    if score.undercite:
        reward -= config.undercite_penalty
    reward -= config.wrong_location_penalty if score.wrong_location else 0.0
    return max(0.0, min(1.0, reward))


def _expand_numeric_range(start: str, end: str) -> list[str]:
    start_match = REF_RE.fullmatch(start)
    end_match = REF_RE.fullmatch(end)
    if not start_match or not end_match:
        return [start, end]
    start_prefix, start_digits = start_match.groups()
    end_prefix, end_digits = end_match.groups()
    if start_prefix != end_prefix:
        return [start, end]
    start_num = int(start_digits)
    end_num = int(end_digits)
    step = 1 if end_num >= start_num else -1
    width = max(2, len(start_digits), len(end_digits))
    return [f"{start_prefix}{num:0{width}d}" for num in range(start_num, end_num + step, step)]


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
