#!/usr/bin/env python3
"""Build a hand-labeling file for Phase 1 faithfulness validation.

Each row asks whether the system's answer is supported by the evidence it was
actually given. This is intentionally independent of whether the answer matches
the gold label.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


LABELS = ["supported", "unsupported", "contradicted", "abstained"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]) + " ..."


def select_evidence(row: dict, max_passages: int, max_words: int) -> list[dict]:
    passages = row.get("retrieved_passages", [])[:max_passages]
    evidence = []
    for passage in passages:
        evidence.append(
            {
                "rank": passage.get("rank", ""),
                "passage_id": passage.get("passage_id", ""),
                "title": passage.get("title", ""),
                "source_doc_id": passage.get("source_doc_id", ""),
                "corpus": passage.get("corpus", ""),
                "text": truncate_words(passage.get("text", ""), max_words),
            }
        )
    return evidence


def is_correct(row: dict) -> bool | None:
    if row.get("benchmark") == "pubmedqa":
        gold = str(row.get("gold_short_answer") or (row.get("gold_answers") or [""])[0]).lower()
        pred = str(row.get("parsed_label") or row.get("predicted_answer", "")).lower()
        return gold in pred.split() or pred.strip() == gold
    return None


def build_item(row: dict, max_passages: int, max_words: int) -> dict:
    return {
        "question_id": row.get("question_id", ""),
        "benchmark": row.get("benchmark", ""),
        "system": row.get("system", ""),
        "config_id": row.get("config_id", ""),
        "model": row.get("model", ""),
        "question": row.get("question", ""),
        "predicted_answer": row.get("predicted_answer", ""),
        "raw_predicted_answer": row.get("raw_predicted_answer", row.get("predicted_answer", "")),
        "gold_answers": row.get("gold_answers", []),
        "gold_short_answer": row.get("gold_short_answer", ""),
        "retrieval_recall": row.get("retrieval_recall", None),
        "answer_correct_if_available": is_correct(row),
        "evidence_policy": "retrieved_passages_top_k",
        "evidence": select_evidence(row, max_passages, max_words),
        "manual_faithfulness_label": "",
        "manual_supported_evidence_ids": [],
        "manual_unsupported_claims": [],
        "manual_notes": "",
        "allowed_labels": LABELS,
        "labeling_instruction": (
            "Label supported only if the retrieved evidence shown here supports the system answer. "
            "If the answer is correct from outside knowledge but not supported by this evidence, label unsupported. "
            "Use contradicted when evidence says the answer is false. Use abstained for explicit no-answer/unknown."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-passages", type=int, default=8)
    parser.add_argument("--max-words-per-passage", type=int, default=180)
    args = parser.parse_args()

    rows = []
    for path in args.input_jsonl:
        rows.extend(load_jsonl(path))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: args.limit]
    output_rows = [build_item(row, args.max_passages, args.max_words_per_passage) for row in selected]
    write_jsonl(args.output_jsonl, output_rows)
    print(f"Wrote {len(output_rows)} annotation rows to {args.output_jsonl}")
    print("Fill manual_faithfulness_label with one of: " + ", ".join(LABELS))


if __name__ == "__main__":
    main()
