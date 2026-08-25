#!/usr/bin/env python3
"""Convert the prepared PubMedQA subset into IRCoT's MultiParaRC JSONL shape."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_numbered_context(text: str) -> tuple[str, str]:
    match = re.match(r"^\[P(\d+)\]\s*PMID:([^\s]+)\s+(.*)$", text.strip())
    if not match:
        return "", text.strip()
    local_pid = "P" + match.group(1)
    rest = match.group(3).strip()
    section_match = re.match(r"^([A-Z /-]+)\s+-\s+(.*)$", rest)
    if section_match:
        section = section_match.group(1).strip()
        body = section_match.group(2).strip()
        return local_pid, f"{section}: {body}"
    return local_pid, rest


def convert_row(row: dict) -> dict:
    pmid = str(row["id"])
    contexts = []
    for index, context in enumerate(row.get("numbered_context", []), start=1):
        local_pid, paragraph_text = strip_numbered_context(context)
        local_pid = local_pid or f"P{index}"
        contexts.append(
            {
                "id": f"{pmid}::{local_pid}",
                "title": f"PMID:{pmid} {local_pid}",
                "paragraph_text": paragraph_text,
                "is_supporting": True,
            }
        )

    return {
        "question_id": pmid,
        "question_text": row["question"],
        "answers_objects": [
            {
                "spans": [row["gold_short_answer"]],
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
            }
        ],
        "contexts": contexts,
        "pinned_contexts": [],
        "level": "pubmedqa",
        "type": "yes_no_maybe",
        "answer_type": row["gold_short_answer"],
        "gold_long_answer": row.get("gold_long_answer", ""),
        "split": row.get("split", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/pubmedqa_150_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/phase1/pubmedqa/pubmedqa_150_ircot.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    converted = [convert_row(row) for row in rows]
    write_jsonl(args.output, converted)
    print(f"Wrote {len(converted)} PubMedQA examples to {args.output}")


if __name__ == "__main__":
    main()
