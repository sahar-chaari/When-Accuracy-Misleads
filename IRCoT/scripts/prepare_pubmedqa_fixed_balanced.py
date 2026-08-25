#!/usr/bin/env python3
"""Create a fixed balanced PubMedQA subset with a manifest."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


LABELS = ("yes", "no", "maybe")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("../data/processed/pubmedqa_150_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/fixed/pubmedqa_60_balanced_seed42_questions.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-label", type=int, default=20)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    grouped: dict[str, list[dict]] = {label: [] for label in LABELS}
    for row in rows:
        label = str(row.get("gold_short_answer", "")).lower()
        if label in grouped:
            grouped[label].append(row)

    missing = {label: args.per_label - len(items) for label, items in grouped.items() if len(items) < args.per_label}
    if missing:
        raise ValueError(f"Not enough rows for requested balance: {missing}")

    rng = random.Random(args.seed)
    selected = []
    for label in LABELS:
        selected.extend(rng.sample(grouped[label], args.per_label))
    rng.shuffle(selected)

    write_jsonl(args.output, selected)
    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "per_label": args.per_label,
        "total": len(selected),
        "label_counts": dict(collections.Counter(str(row.get("gold_short_answer", "")).lower() for row in selected)),
        "question_ids": [str(row.get("id", "")) for row in selected],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

