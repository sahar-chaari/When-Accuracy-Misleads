#!/usr/bin/env python3
"""Answer post-processing helpers for Phase 1 stored outputs."""

from __future__ import annotations

import json
import re
import string

PUBMEDQA_LABELS = ("yes", "no", "maybe")


def decode_json_answer(value) -> str:
    """Mirror IRCoT's common string/list JSON answer convention."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return " ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    text = value.strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if isinstance(parsed, list):
        return " ".join(str(item) for item in parsed)
    if isinstance(parsed, str):
        return parsed.strip()
    return str(parsed)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc((text or "").lower())))


def _strip_answer_noise(text: str) -> str:
    text = re.sub(r"\[[A-Za-z]?\d+\]", "", text or "")
    text = text.strip().strip("\"'` ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def postprocess_hotpot_answer(value: str) -> tuple[str, str]:
    """Extract a concise HotpotQA-style answer span from verbose LLM text.

    This is intentionally conservative. It mainly handles the formats produced
    by instruction-tuned causal LMs when given the official IRCoT few-shot
    prompts, while preserving the original text when no robust cue is present.
    """
    original = decode_json_answer(value)
    text = _strip_answer_noise(original)
    if not text:
        return "", "empty"

    text = re.split(r"\n\s*(?:Q:|Question:|Context:|Wikipedia Title:)\s*", text, maxsplit=1)[0].strip()
    text = re.sub(r"^(?:A|Answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    answer_markers = [
        r"(?:so|thus|therefore|hence)[,\s]+(?:the\s+)?answer\s+is\s*:?\s+",
        r"(?:the\s+)?final\s+answer\s+is\s*:?\s+",
        r"(?:the\s+)?answer\s+is\s*:?\s+",
        r"answer\s*:\s+",
    ]
    lowered = text.lower()
    best_start = -1
    best_marker = ""
    for pattern in answer_markers:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            if match.end() > best_start:
                best_start = match.end()
                best_marker = pattern

    if best_start >= 0:
        candidate = text[best_start:].strip()
        candidate = re.split(r"\s+(?:because|since|as)\s+", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
        candidate = re.split(r"\n", candidate, maxsplit=1)[0]
        candidate = candidate.rstrip(" .")
        candidate = _strip_answer_noise(candidate)
        if candidate:
            return candidate, "answer_marker"

    short_prefixes = [
        r"^it\s+is\s+",
        r"^it\s+was\s+",
        r"^they\s+are\s+",
        r"^the\s+answer\s+would\s+be\s+",
    ]
    candidate = text
    for pattern in short_prefixes:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()

    sentences = re.split(r"(?<=[.!?])\s+", candidate)
    if len(sentences) > 1 and len(sentences[0].split()) <= 10:
        candidate = sentences[0].rstrip(" .")
        return _strip_answer_noise(candidate), "short_first_sentence"

    return _strip_answer_noise(candidate), "unchanged"


MCQ_LABELS = ("A", "B", "C", "D")


def parse_mcq_label(text: str) -> str:
    """Extract A/B/C/D from structured MCQ model output."""
    text = text or ""
    match = re.search(r"\b(?:final\s+answer|answer)\s*:\s*([ABCD])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    first_line = text.splitlines()[0] if text.splitlines() else text
    match = re.search(r"\b([ABCD])\b", first_line)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1).upper() if match else "unknown"


def parse_pubmedqa_label(text: str) -> str:
    """Parse PubMedQA yes/no/maybe labels from structured or legacy text."""
    lowered = (text or "").lower()
    match = re.search(r"\b(?:final\s+answer|answer)\s*:\s*(yes|no|maybe)\b", lowered)
    if match:
        return match.group(1)
    first_line = lowered.splitlines()[0] if lowered.splitlines() else lowered
    match = re.search(r"\b(yes|no|maybe)\b", first_line)
    if match:
        return match.group(1)
    match = re.search(r"\b(yes|no|maybe)\b", lowered)
    return match.group(1) if match else "unknown"
