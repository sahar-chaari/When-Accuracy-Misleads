#!/usr/bin/env python3
"""Backward-compatible parsing for confidence-aware IRCoT outputs."""

from __future__ import annotations

import json
import re
from typing import Any


def _as_float(value: Any) -> tuple[float | None, str]:
    if value in (None, ""):
        return None, "missing"
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None, "invalid"
    if 0.0 <= confidence <= 1.0:
        return confidence, "parsed"
    if 1.0 < confidence <= 100.0:
        return confidence / 100.0, "parsed_percent"
    return None, "out_of_range"


def parse_confidence_detailed(text: str) -> dict:
    """Parse explicit confidence from generated text without inventing it."""
    if not text:
        return {
            "value": None,
            "raw": "",
            "status": "missing_in_legacy_output",
            "parse_error": "confidence_missing",
        }

    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        for key in ("confidence", "confidence_score", "uncertainty_score"):
            raw_value = obj.get(key)
            confidence, status = _as_float(raw_value)
            if confidence is not None or status not in {"missing"}:
                return {
                    "value": confidence,
                    "raw": "" if raw_value is None else str(raw_value),
                    "status": f"json_{status}",
                    "parse_error": "" if confidence is not None else f"confidence_{status}",
                }

    patterns = [
        r"\bconfidence\s*(?:score)?\s*[:=]\s*([01](?:\.\d+)?|100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\s*%?",
        r"\bconfidence\s+is\s+([01](?:\.\d+)?|100(?:\.0+)?|\d{1,2}(?:\.\d+)?)\s*%?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            confidence, status = _as_float(match.group(1))
            return {
                "value": confidence,
                "raw": match.group(1),
                "status": f"text_{status}",
                "parse_error": "" if confidence is not None else f"confidence_{status}",
            }
    return {
        "value": None,
        "raw": "",
        "status": "missing_in_legacy_output",
        "parse_error": "confidence_missing",
    }


def parse_confidence(text: str) -> tuple[float | None, str]:
    """Parse explicit confidence from generated text without inventing it."""
    parsed = parse_confidence_detailed(text)
    return parsed["value"], parsed["status"]


def parse_supporting_evidence_detailed(text: str) -> dict:
    """Extract model-cited/supporting evidence when future prompts include it."""
    if not text:
        return {
            "value": "",
            "raw": "",
            "status": "missing_in_legacy_output",
            "parse_error": "supporting_evidence_missing",
        }

    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        for key in ("supporting_evidence", "evidence", "reference", "citation"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return {
                    "value": value.strip(),
                    "raw": value,
                    "status": f"json_{key}",
                    "parse_error": "",
                }

    match = re.search(
        r"(?:supporting evidence|evidence|reference)\s*:\s*(.+?)(?:\n\s*confidence\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        raw_value = match.group(1)
        return {
            "value": raw_value.strip(),
            "raw": raw_value,
            "status": "text_supporting_evidence",
            "parse_error": "",
        }
    return {
        "value": "",
        "raw": "",
        "status": "missing_in_legacy_output",
        "parse_error": "supporting_evidence_missing",
    }


def parse_supporting_evidence(text: str) -> tuple[str, str]:
    """Extract model-cited/supporting evidence when future prompts include it."""
    parsed = parse_supporting_evidence_detailed(text)
    return parsed["value"], parsed["status"]


def parse_final_answer_detailed(text: str) -> dict:
    """Extract future structured final answers when present."""
    if not text:
        return {"value": "", "raw": "", "status": "missing", "parse_error": "final_answer_missing"}

    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        for key in ("final_answer", "answer", "predicted_answer"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return {
                    "value": value.strip(),
                    "raw": value,
                    "status": f"json_{key}",
                    "parse_error": "",
                }

    match = re.search(
        r"(?:final answer|answer)\s*:\s*(.+?)(?:\n\s*(?:supporting evidence|evidence|reference|confidence)\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        raw_value = match.group(1)
        return {
            "value": raw_value.strip(),
            "raw": raw_value,
            "status": "text_final_answer",
            "parse_error": "",
        }
    return {"value": text.strip(), "raw": text, "status": "raw_text", "parse_error": ""}


def parse_final_answer(text: str) -> tuple[str, str]:
    """Extract future structured final answers when present."""
    parsed = parse_final_answer_detailed(text)
    return parsed["value"], parsed["status"]


def confidence_from_row(row: dict) -> tuple[float | None, str]:
    """Return confidence from explicit row fields or parseable generations."""
    for field in ("confidence", "parsed_confidence", "model_confidence", "self_reported_confidence"):
        confidence, status = _as_float(row.get(field))
        if confidence is not None or status not in {"missing"}:
            return confidence, f"field_{field}_{status}"

    for field in ("raw_predicted_answer", "predicted_answer", "raw_generation"):
        confidence, status = parse_confidence(str(row.get(field, "")))
        if confidence is not None or status != "missing_in_legacy_output":
            return confidence, f"{field}_{status}"
    return None, "missing_in_legacy_output"
