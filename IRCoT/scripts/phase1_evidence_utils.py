#!/usr/bin/env python3
"""Utilities for verifying model-provided evidence against retrieved passages."""

from __future__ import annotations

import re


NONE_MARKERS = {
    "none",
    "n/a",
    "na",
    "not available",
    "no supporting evidence",
    "no relevant evidence",
    "no exact quote",
}


def _clean_text(text: str) -> str:
    text = (text or "").strip().strip("\"'` ")
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\bPMID:\s*\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _loose_text(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def verify_supporting_evidence(evidence: str, retrieved_passages: list[dict]) -> dict:
    """Check whether supporting evidence is copied from retrieved passages.

    This is a string-level compliance check, not a semantic entailment scorer.
    It helps distinguish model-provided explanation from actual cited evidence.
    """
    raw_evidence = (evidence or "").strip()
    clean_evidence = _clean_text(raw_evidence)
    loose_evidence = _loose_text(raw_evidence)

    if not raw_evidence:
        return {
            "quote_match": False,
            "status": "missing_supporting_evidence",
            "method": "none",
            "verified_passage_ids": [],
            "verified_passage_titles": [],
        }

    if clean_evidence in NONE_MARKERS or loose_evidence in NONE_MARKERS:
        return {
            "quote_match": False,
            "status": "no_supporting_quote_declared",
            "method": "none_marker",
            "verified_passage_ids": [],
            "verified_passage_titles": [],
        }

    exact_matches = []
    loose_matches = []
    for passage in retrieved_passages or []:
        text = passage.get("text", "")
        clean_passage = _clean_text(text)
        loose_passage = _loose_text(text)
        if clean_evidence and clean_evidence in clean_passage:
            exact_matches.append(passage)
        elif loose_evidence and loose_evidence in loose_passage:
            loose_matches.append(passage)

    matches = exact_matches or loose_matches
    if not matches:
        return {
            "quote_match": False,
            "status": "not_found_in_retrieved_passages",
            "method": "normalized_substring",
            "verified_passage_ids": [],
            "verified_passage_titles": [],
        }

    return {
        "quote_match": True,
        "status": "found_in_retrieved_passages",
        "method": "case_whitespace_substring" if exact_matches else "loose_alnum_substring",
        "verified_passage_ids": [str(item.get("passage_id", "")) for item in matches],
        "verified_passage_titles": [str(item.get("title", "")) for item in matches],
    }

