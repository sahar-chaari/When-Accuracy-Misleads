#!/usr/bin/env python3
"""Faithfulness scorer interface scaffold for stored Phase 1 outputs.

This intentionally does not implement the final metric. The final scorer should
use NLI or an LLM judge over the retrieved passages and be validated against
hand labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scaffold_score(row: dict) -> dict:
    return {
        "question_id": row.get("question_id", ""),
        "benchmark": row.get("benchmark", ""),
        "system": row.get("system", ""),
        "faithfulness_label": "not_scored",
        "unsupported_claims": [],
        "contradiction_claims": [],
        "evidence_ids_used": [],
        "judge_model": "",
        "rationale": "Scaffold only. Use NLI or LLM-as-judge over retrieved_passages after hand-label validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    rows = [scaffold_score(row) for row in load_jsonl(args.input_jsonl)]
    write_jsonl(args.output_jsonl, rows)
    print(f"Wrote {len(rows)} faithfulness scaffold rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
