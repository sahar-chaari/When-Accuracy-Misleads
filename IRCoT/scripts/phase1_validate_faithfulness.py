#!/usr/bin/env python3
"""Validate automatic faithfulness labels against hand labels."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


VALID_LABELS = ("supported", "unsupported", "contradicted", "abstained")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def key(row: dict) -> tuple[str, str]:
    return (str(row.get("benchmark", "")), str(row.get("question_id", "")))


def binary(label: str) -> str:
    return "faithful" if label == "supported" else "unfaithful"


def f1_for_label(golds: list[str], preds: list[str], label: str) -> float:
    tp = sum(g == label and p == label for g, p in zip(golds, preds))
    fp = sum(g != label and p == label for g, p in zip(golds, preds))
    fn = sum(g == label and p != label for g, p in zip(golds, preds))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def macro_f1(golds: list[str], preds: list[str], labels: tuple[str, ...]) -> float:
    return sum(f1_for_label(golds, preds, label) for label in labels) / len(labels)


def confusion(golds: list[str], preds: list[str], labels: tuple[str, ...]) -> dict:
    return {
        gold: {pred: sum(g == gold and p == pred for g, p in zip(golds, preds)) for pred in labels}
        for gold in labels
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-jsonl", type=Path, required=True)
    parser.add_argument("--auto-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manual_rows = load_jsonl(args.manual_jsonl)
    auto_rows = {key(row): row for row in load_jsonl(args.auto_jsonl)}

    pairs = []
    skipped = []
    for row in manual_rows:
        gold = str(row.get("manual_faithfulness_label", "")).strip().lower()
        if not gold:
            skipped.append({"question_id": row.get("question_id", ""), "reason": "missing_manual_label"})
            continue
        if gold not in VALID_LABELS:
            skipped.append({"question_id": row.get("question_id", ""), "reason": f"invalid_label:{gold}"})
            continue
        auto = auto_rows.get(key(row))
        if not auto:
            skipped.append({"question_id": row.get("question_id", ""), "reason": "missing_auto_label"})
            continue
        pred = str(auto.get("faithfulness_label", "")).strip().lower()
        if pred not in VALID_LABELS:
            pred = "unsupported"
        pairs.append((gold, pred))

    golds = [gold for gold, _ in pairs]
    preds = [pred for _, pred in pairs]
    bin_golds = [binary(label) for label in golds]
    bin_preds = [binary(label) for label in preds]
    binary_labels = ("faithful", "unfaithful")

    result = {
        "manual_jsonl": str(args.manual_jsonl),
        "auto_jsonl": str(args.auto_jsonl),
        "validated_examples": len(pairs),
        "skipped_examples": len(skipped),
        "exact_label_accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else 0.0,
        "macro_f1_4way": macro_f1(golds, preds, VALID_LABELS) if pairs else 0.0,
        "binary_faithful_accuracy": (
            sum(g == p for g, p in zip(bin_golds, bin_preds)) / len(bin_golds) if bin_golds else 0.0
        ),
        "binary_macro_f1": macro_f1(bin_golds, bin_preds, binary_labels) if pairs else 0.0,
        "manual_label_counts": dict(collections.Counter(golds)),
        "auto_label_counts": dict(collections.Counter(preds)),
        "confusion_4way": confusion(golds, preds, VALID_LABELS),
        "confusion_binary": confusion(bin_golds, bin_preds, binary_labels),
        "skipped": skipped,
    }
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
