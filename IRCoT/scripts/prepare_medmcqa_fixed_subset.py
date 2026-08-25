#!/usr/bin/env python3
"""Create a fixed balanced MedMCQA subset with a tune/eval split and a manifest.

MedMCQA is a multiple-choice medical QA dataset from Indian medical exams.
We sample 60 questions (15 per answer choice A/B/C/D) from the validation
split, using only single-answer complete questions. The resulting rows use
the same canonical schema as the PubMedQA fixed subset so all downstream
runners work without modification.

Usage:
    python prepare_medmcqa_fixed_subset.py \
        --output data/fixed/medmcqa_60_balanced_seed42_questions.jsonl \
        --seed 42 --per-choice 15
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

CHOICE_LETTERS = ("A", "B", "C", "D")
CHOICE_INDEX = {letter: idx for idx, letter in enumerate(CHOICE_LETTERS)}
INDEX_TO_LETTER = {idx: letter for idx, letter in enumerate(CHOICE_LETTERS)}


def load_medmcqa_validation() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Install the HuggingFace 'datasets' package: pip install datasets"
        )
    ds = load_dataset("medmcqa", split="validation", trust_remote_code=True)
    return [dict(row) for row in ds]


def filter_usable(rows: list[dict]) -> list[dict]:
    usable = []
    for row in rows:
        if row.get("choice_type", "single") != "single":
            continue
        if row.get("is_incomplete_flag", False):
            continue
        cop = row.get("cop")
        if cop not in (0, 1, 2, 3):
            continue
        for field in ("opa", "opb", "opc", "opd"):
            if not str(row.get(field, "")).strip():
                break
        else:
            usable.append(row)
    return usable


def convert_row(raw: dict, split: str) -> dict:
    cop = int(raw["cop"])
    gold_letter = INDEX_TO_LETTER[cop]
    options = {
        "A": str(raw.get("opa", "")).strip(),
        "B": str(raw.get("opb", "")).strip(),
        "C": str(raw.get("opc", "")).strip(),
        "D": str(raw.get("opd", "")).strip(),
    }
    question_with_choices = (
        raw["question"].strip()
        + "\nA) " + options["A"]
        + "\nB) " + options["B"]
        + "\nC) " + options["C"]
        + "\nD) " + options["D"]
    )
    return {
        "question_id": str(raw.get("id", "")),
        "benchmark": "medmcqa",
        "split": split,
        "question": question_with_choices,
        "question_stem": raw["question"].strip(),
        "options": options,
        "gold_short_answer": gold_letter,
        "gold_answer_text": options[gold_letter],
        "gold_long_answer": str(raw.get("exp", "") or "").strip(),
        "subject_name": str(raw.get("subject_name", "") or "").strip(),
        "topic_name": str(raw.get("topic_name", "") or "").strip(),
        "choice_type": str(raw.get("choice_type", "single")),
        "metadata": {
            "gold_ids": [],
            "gold_titles": [],
            "gold_answer_index": cop,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fixed/medmcqa_60_balanced_seed42_questions.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-choice", type=int, default=15,
                        help="Questions per answer choice (A/B/C/D). Total = 4 * per-choice.")
    parser.add_argument(
        "--tune-fraction", type=float, default=0.5,
        help="Fraction of selected questions assigned to 'tune' split (rest go to 'eval').",
    )
    args = parser.parse_args()

    print("Loading MedMCQA validation split from HuggingFace...")
    raw_rows = load_medmcqa_validation()
    print(f"  Loaded {len(raw_rows)} raw rows.")
    usable = filter_usable(raw_rows)
    print(f"  {len(usable)} usable after filtering (single-choice, complete).")

    grouped: dict[str, list[dict]] = {letter: [] for letter in CHOICE_LETTERS}
    for row in usable:
        letter = INDEX_TO_LETTER[int(row["cop"])]
        grouped[letter].append(row)

    for letter in CHOICE_LETTERS:
        if len(grouped[letter]) < args.per_choice:
            raise SystemExit(
                f"Not enough usable rows for choice {letter}: "
                f"have {len(grouped[letter])}, need {args.per_choice}."
            )
        print(f"  Choice {letter}: {len(grouped[letter])} available, sampling {args.per_choice}.")

    rng = random.Random(args.seed)
    selected_raw = []
    for letter in CHOICE_LETTERS:
        selected_raw.extend(rng.sample(grouped[letter], args.per_choice))
    rng.shuffle(selected_raw)

    total = len(selected_raw)
    n_tune = round(total * args.tune_fraction)
    splits = ["tune"] * n_tune + ["eval"] * (total - n_tune)
    rng.shuffle(splits)

    rows = [convert_row(raw, split) for raw, split in zip(selected_raw, splits)]
    write_jsonl(args.output, rows)

    label_counts = dict(
        collections.Counter(row["gold_short_answer"] for row in rows)
    )
    split_counts = dict(collections.Counter(row["split"] for row in rows))
    manifest = {
        "source": "HuggingFace medmcqa validation",
        "output": str(args.output),
        "seed": args.seed,
        "per_choice": args.per_choice,
        "total": total,
        "choice_counts": label_counts,
        "split_counts": split_counts,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {total} rows to {args.output}")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
