from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9_]+")

DOMAINS = [
    "archive",
    "garden",
    "harbor",
    "forge",
    "observatory",
]

SUBJECTS = [
    "lumet",
    "brindle",
    "kestral",
    "morrow",
    "talon",
    "verin",
    "cinder",
    "rill",
    "sable",
    "thorn",
]

OBJECTS = [
    "bridge",
    "gate",
    "canal",
    "signal",
    "token",
    "ledger",
    "cargo",
    "lantern",
    "chamber",
    "path",
]

ACTIONS = [
    "enter",
    "cross",
    "carry",
    "open",
    "activate",
    "record",
    "store",
    "signal",
    "guard",
    "inspect",
]

CONDITIONS = [
    "during blue tide",
    "after two cold nights",
    "before sunrise",
    "when the brass bell sounds",
    "unless marked by ash",
    "only with a copper seal",
    "when river mist is present",
    "if the eastern lamp is lit",
    "after the second watch",
    "unless the cargo is iron",
]

TYPES = [
    "rule",
    "exception",
    "condition",
    "restriction",
]

QUESTION_PATTERNS = {
    "isolated": [
        "Which refmark says that the {subject} may {action} the {object} {condition}?",
        "Where is the rule about a {subject} using the {object} {condition}?",
        "Which anchor defines when a {subject} can {action} the {object}?",
        "Find the refmark for the {subject} {action} {object} rule.",
        "Identify the anchor covering how a {subject} may {action} the {object}.",
        "Which citation contains the {subject} policy for the {object}?",
        "What refmark governs a {subject} that needs to {action} the {object}?",
        "Find the exact anchor for the {subject}, {action}, and {object} requirement.",
        "Which refmark should be cited for a {subject} and the {object} clause?",
        "Locate the rule that lets the {subject} {action} the {object} {condition}.",
    ],
    "expanded": [
        "Which refmark covers the operating rule for the {subject} and the {object} under the timing constraint {condition}?",
        "Find the anchor that governs whether the {subject} may {action} the {object} once the rule {condition} applies.",
        "What citation describes the policy for a {subject} to {action} the {object} in the case {condition}?",
        "Which refmark is the best match for the broader procedure involving the {subject}, the {object}, and the clause {condition}?",
    ],
    "reformulated": [
        "Which citation states the condition under which the {subject} is permitted to {action} the {object}?",
        "Locate the anchor describing the {subject}'s {object}-related {action} rule tied to {condition}.",
        "What refmark contains the governing clause for {subject} {action} on the {object}?",
        "Which semantic anchor best matches the policy combining {subject}, {action}, {object}, and {condition}?",
    ],
}


@dataclass(frozen=True)
class Anchor:
    refmark: str
    domain: str
    subject: str
    action: str
    object: str
    condition: str
    anchor_type: str
    text: str
    title: str | None = None
    region_start: int | None = None
    region_end: int | None = None


@dataclass(frozen=True)
class Example:
    split: str
    prompt_style: str
    question: str
    refmark: str
    title: str | None = None
    answer_text: str | None = None
    answer_start: int | None = None
    answer_end: int | None = None
    gold_start_refmark: str | None = None
    gold_end_refmark: str | None = None


@dataclass(frozen=True)
class CorpusBundle:
    anchors: list[Anchor]
    train: list[Example]
    valid: list[Example]
    reformulated: list[Example]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _anchor_text(anchor: Anchor) -> str:
    return (
        f"[{anchor.refmark}] In the {anchor.domain}, a {anchor.subject} may "
        f"{anchor.action} the {anchor.object} {anchor.condition}. This entry is "
        f"classified as a {anchor.anchor_type}."
    )


def build_corpus(anchor_count: int = 100, seed: int = 13) -> CorpusBundle:
    rng = random.Random(seed)
    seen: set[tuple[str, str, str, str, str]] = set()
    anchors: list[Anchor] = []
    attempts = 0
    while len(anchors) < anchor_count:
        attempts += 1
        if attempts > anchor_count * 100:
            raise RuntimeError("failed to build a sufficiently distinct corpus")

        candidate = (
            rng.choice(DOMAINS),
            rng.choice(SUBJECTS),
            rng.choice(ACTIONS),
            rng.choice(OBJECTS),
            rng.choice(CONDITIONS),
        )
        if candidate in seen:
            continue
        seen.add(candidate)
        domain, subject, action, object_name, condition = candidate
        anchor = Anchor(
            refmark=f"R{len(anchors) + 1:03d}",
            domain=domain,
            subject=subject,
            action=action,
            object=object_name,
            condition=condition,
            anchor_type=rng.choice(TYPES),
            text="",
        )
        anchor = Anchor(**{**anchor.__dict__, "text": _anchor_text(anchor)})
        anchors.append(anchor)

    train: list[Example] = []
    valid: list[Example] = []
    reformulated: list[Example] = []

    for anchor in anchors:
        fields = {
            "subject": anchor.subject,
            "action": anchor.action,
            "object": anchor.object,
            "condition": anchor.condition,
        }
        for template in QUESTION_PATTERNS["isolated"]:
            train.append(
                Example(
                    split="train",
                    prompt_style="isolated",
                    question=template.format(**fields),
                    refmark=anchor.refmark,
                )
            )
        for template in QUESTION_PATTERNS["expanded"]:
            valid.append(
                Example(
                    split="valid",
                    prompt_style="expanded",
                    question=template.format(**fields),
                    refmark=anchor.refmark,
                )
            )
        for template in QUESTION_PATTERNS["reformulated"]:
            reformulated.append(
                Example(
                    split="reformulated",
                    prompt_style="reformulated",
                    question=template.format(**fields),
                    refmark=anchor.refmark,
                )
            )

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(reformulated)
    return CorpusBundle(
        anchors=anchors,
        train=train,
        valid=valid,
        reformulated=reformulated,
    )


def preview_lines(bundle: CorpusBundle, limit: int = 5) -> Iterable[str]:
    yield "Anchors:"
    for anchor in bundle.anchors[:limit]:
        yield f"  {anchor.refmark}: {anchor.text}"
    yield ""
    yield "Train examples:"
    for example in bundle.train[:limit]:
        yield f"  {example.refmark}: {example.question}"


def save_bundle(bundle: CorpusBundle, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    anchors_path = output_dir / "anchors.jsonl"
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    reform_path = output_dir / "reformulated.jsonl"
    manifest_path = output_dir / "manifest.json"

    anchors_path.write_text(
        "\n".join(json.dumps(anchor.__dict__) for anchor in bundle.anchors) + "\n",
        encoding="utf-8",
    )
    train_path.write_text(
        "\n".join(json.dumps(example.__dict__) for example in bundle.train) + "\n",
        encoding="utf-8",
    )
    valid_path.write_text(
        "\n".join(json.dumps(example.__dict__) for example in bundle.valid) + "\n",
        encoding="utf-8",
    )
    reform_path.write_text(
        "\n".join(json.dumps(example.__dict__) for example in bundle.reformulated) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "anchor_count": len(bundle.anchors),
        "train_examples": len(bundle.train),
        "valid_examples": len(bundle.valid),
        "reformulated_examples": len(bundle.reformulated),
        "files": {
            "anchors": str(anchors_path),
            "train": str(train_path),
            "valid": str(valid_path),
            "reformulated": str(reform_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "anchors": str(anchors_path),
        "train": str(train_path),
        "valid": str(valid_path),
        "reformulated": str(reform_path),
        "manifest": str(manifest_path),
    }
