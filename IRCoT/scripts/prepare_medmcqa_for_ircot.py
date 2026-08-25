#!/usr/bin/env python3
"""Convert the prepared MedMCQA fixed subset into IRCoT's MultiParaRC JSONL shape.

MedMCQA rows have no per-question gold contexts, so the contexts list is empty
and the system retrieves from the general MedRAG corpus.
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


def convert_row(row: dict) -> dict:
    options = row.get("options", {})
    # Embed the options in the question text so IRCoT's reasoning steps see them.
    options_text = "\n".join(
        f"{letter}) {text}" for letter, text in sorted(options.items()) if text
    )
    full_question = row["question_stem"].strip() + "\n" + options_text if options else row["question"].strip()

    return {
        "question_id": row["question_id"],
        "question_text": full_question,
        "answers_objects": [
            {
                "spans": [row["gold_short_answer"]],
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
            }
        ],
        "contexts": [],
        "pinned_contexts": [],
        "level": "medmcqa",
        "type": "multiple_choice",
        "answer_type": row["gold_short_answer"],
        "gold_answer_text": row.get("gold_answer_text", ""),
        "options": options,
        "subject_name": row.get("subject_name", ""),
        "topic_name": row.get("topic_name", ""),
        "split": row.get("split", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/fixed/medmcqa_60_balanced_seed42_questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase1/medmcqa/medmcqa_60_ircot.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    converted = [convert_row(row) for row in rows]
    write_jsonl(args.output, converted)
    print(f"Wrote {len(converted)} MedMCQA examples to {args.output}")


if __name__ == "__main__":
    main()
