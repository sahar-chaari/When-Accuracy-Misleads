#!/usr/bin/env python3
"""Validate HippoRAG wrapper JSONL output compliance."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    confidence_status = Counter()
    evidence_status = Counter()
    evidence_found = Counter()
    parse_errors = Counter()

    for row in rows:
        confidence_status["present" if row.get("self_reported_confidence") is not None else "missing"] += 1
        evidence_status["present" if row.get("supporting_evidence") else "missing"] += 1
        evidence_found[str(bool(row.get("supporting_evidence_found_in_retrieved")))] += 1
        for error in row.get("parse_errors", []):
            parse_errors[error] += 1

    summary = {
        "input_jsonl": str(path),
        "rows": len(rows),
        "confidence_status_counts": dict(confidence_status),
        "supporting_evidence_status_counts": dict(evidence_status),
        "supporting_evidence_found_counts": dict(evidence_found),
        "parse_error_counts": dict(parse_errors),
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
