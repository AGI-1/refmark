"""Prompt helpers for using refmarked documents with general chat models."""

from __future__ import annotations

from dataclasses import dataclass

from refmark.core import inject


@dataclass(frozen=True)
class EnrichedPrompt:
    """A self-contained prompt package with an addressable document."""

    prompt: str
    marked_document: str
    marker_count: int


def build_reference_prompt(
    content: str,
    file_ext: str,
    *,
    question: str | None = None,
    marker_format: str = "typed_bracket",
    chunker: str = "paragraph",
) -> EnrichedPrompt:
    """Build a prompt that asks a chat model to cite concrete refmark ranges."""
    marked_document, marker_count = inject(
        content,
        file_ext,
        marker_format=marker_format,
        chunker=chunker,
    )
    prompt_parts = [
        "You are given a document with Refmark region markers.",
        "",
        "Markers look like [@P01]. The citeable region id is the label inside the marker, for example P01.",
        "When answering, cite concrete regions using [P01], contiguous ranges using [P01-P03], or non-contiguous sets using [P01,P04].",
        "Use only the marked document as evidence. Do not cite line numbers or uncited prose.",
        "If the document does not support an answer, say that the document does not provide enough support and cite the closest relevant region if one exists.",
        "",
        "MARKED DOCUMENT:",
        marked_document.rstrip(),
    ]
    if question:
        prompt_parts.extend(["", "QUESTION:", question.strip()])
    prompt_parts.extend(
        [
            "",
            "Return the answer with region citations attached to the claims they support.",
        ]
    )
    return EnrichedPrompt(
        prompt="\n".join(prompt_parts) + "\n",
        marked_document=marked_document,
        marker_count=marker_count,
    )
