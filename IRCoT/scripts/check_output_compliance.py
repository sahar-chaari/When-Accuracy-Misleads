#!/usr/bin/env python3
"""Check corrected IRCoT output schema and evidence quote compliance."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from phase1_evidence_utils import verify_supporting_evidence


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    text = text.strip("\"'` ")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    verifications = [
        verify_supporting_evidence(row.get("supporting_evidence", ""), row.get("retrieved_passages", []))
        for row in rows
    ]
    quote_matches = [item["quote_match"] for item in verifications]
    confidence_values = [row.get("self_reported_confidence") for row in rows]
    confidence_available = [value for value in confidence_values if isinstance(value, (int, float))]
    result = {
        "input_jsonl": str(args.input_jsonl),
        "rows": len(rows),
        "confidence_status_counts": dict(collections.Counter(row.get("confidence_status") for row in rows)),
        "supporting_evidence_status_counts": dict(collections.Counter(row.get("supporting_evidence_status") for row in rows)),
        "parsed_label_counts": dict(collections.Counter(row.get("parsed_label") for row in rows)),
        "abstention_status_counts": dict(collections.Counter(row.get("abstention_status") for row in rows)),
        "error_count": sum(bool(row.get("errors")) for row in rows),
        "evidence_exact_quote_count": sum(quote_matches),
        "evidence_exact_quote_rate": sum(quote_matches) / len(rows) if rows else None,
        "evidence_verification_status_counts": dict(
            collections.Counter(item["status"] for item in verifications)
        ),
        "evidence_verification_method_counts": dict(
            collections.Counter(item["method"] for item in verifications)
        ),
        "avg_confidence": sum(confidence_available) / len(confidence_available) if confidence_available else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    misses = [row for row, ok in zip(rows, quote_matches) if not ok]
    if misses:
        print("\nExamples where supporting_evidence was not an exact retrieved-text substring:")
        for row in misses[: args.examples]:
            print("=" * 80)
            print("question_id:", row.get("question_id"))
            print("gold:", row.get("gold_short_answer") or row.get("gold_answers"))
            print("pred:", row.get("predicted_answer"))
            print("confidence:", row.get("self_reported_confidence"))
            print("supporting_evidence:", row.get("supporting_evidence"))


if __name__ == "__main__":
    main()
