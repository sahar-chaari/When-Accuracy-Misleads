#!/usr/bin/env python3
"""Compute Phase 1 classical metrics from stored JSONL outputs."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from phase1_answer_utils import MCQ_LABELS, normalize_answer, parse_mcq_label, parse_pubmedqa_label, postprocess_hotpot_answer

PUBMEDQA_LABELS = ("yes", "no", "maybe")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def best_over_golds(metric, prediction: str, golds: list[str]) -> float:
    return max(metric(prediction, gold) for gold in golds) if golds else 0.0


def macro_f1(golds: list[str], preds: list[str]) -> float:
    scores = []
    for label in PUBMEDQA_LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def score_hotpotqa(rows: list[dict], postprocess: bool = False) -> dict:
    ems = []
    f1s = []
    changed = 0
    for row in rows:
        prediction = row.get("predicted_answer", "")
        if postprocess:
            cleaned, _ = postprocess_hotpot_answer(prediction)
            changed += int(cleaned != prediction)
            prediction = cleaned
        golds = [str(item) for item in row.get("gold_answers", [])]
        ems.append(best_over_golds(exact_match, prediction, golds))
        f1s.append(best_over_golds(token_f1, prediction, golds))
    metrics = {
        "benchmark": "hotpotqa",
        "num_examples": len(rows),
        "exact_match": sum(ems) / len(ems) if ems else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "avg_retrieval_recall": average(row.get("retrieval_recall") for row in rows),
        "error_count": sum(bool(row.get("errors")) for row in rows),
    }
    if postprocess:
        metrics["postprocess_hotpot_answer"] = True
        metrics["postprocessed_prediction_count"] = changed
    return metrics


def score_pubmedqa(rows: list[dict]) -> dict:
    golds = []
    preds = []
    for row in rows:
        gold = str(row.get("gold_short_answer") or (row.get("gold_answers") or [""])[0]).lower()
        pred = parse_pubmedqa_label(row.get("predicted_answer", ""))
        row["parsed_label"] = pred
        golds.append(gold)
        preds.append(pred)
    return {
        "benchmark": "pubmedqa",
        "num_examples": len(rows),
        "accuracy": sum(g == p for g, p in zip(golds, preds)) / len(rows) if rows else 0.0,
        "macro_f1": macro_f1(golds, preds) if rows else 0.0,
        "unknown_prediction_count": sum(pred == "unknown" for pred in preds),
        "gold_counts": dict(collections.Counter(golds)),
        "prediction_counts": dict(collections.Counter(preds)),
        "confusion": confusion(golds, preds),
        "avg_retrieval_recall": average(row.get("retrieval_recall") for row in rows),
        "avg_retrieval_recall_pmid": average(row.get("retrieval_recall_pmid") for row in rows),
        "error_count": sum(bool(row.get("errors")) for row in rows),
    }


def macro_f1_mcq(golds: list[str], preds: list[str]) -> float:
    scores = []
    for label in MCQ_LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def score_medmcqa(rows: list[dict]) -> dict:
    golds = []
    preds = []
    for row in rows:
        gold = str(row.get("gold_short_answer") or (row.get("gold_answers") or [""])[0]).upper()
        pred = parse_mcq_label(row.get("predicted_answer", ""))
        row["parsed_label"] = pred
        golds.append(gold)
        preds.append(pred)
    labels = list(MCQ_LABELS) + ["unknown"]
    mcq_confusion = {
        gold: {pred: sum(g == gold and p == pred for g, p in zip(golds, preds)) for pred in labels}
        for gold in MCQ_LABELS
    }
    return {
        "benchmark": "medmcqa",
        "num_examples": len(rows),
        "accuracy": sum(g == p for g, p in zip(golds, preds)) / len(rows) if rows else 0.0,
        "macro_f1": macro_f1_mcq(golds, preds) if rows else 0.0,
        "unknown_prediction_count": sum(pred == "unknown" for pred in preds),
        "gold_counts": dict(collections.Counter(golds)),
        "prediction_counts": dict(collections.Counter(preds)),
        "confusion": mcq_confusion,
        "error_count": sum(bool(row.get("errors")) for row in rows),
    }


def average(values) -> float | None:
    nums = [float(value) for value in values if value not in (None, "")]
    return sum(nums) / len(nums) if nums else None


def confusion(golds: list[str], preds: list[str]) -> dict:
    labels = list(PUBMEDQA_LABELS) + ["unknown"]
    return {
        gold: {pred: sum(g == gold and p == pred for g, p in zip(golds, preds)) for pred in labels}
        for gold in PUBMEDQA_LABELS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["hotpotqa", "pubmedqa", "medmcqa"], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--postprocess-hotpot-answer",
        action="store_true",
        help="Apply conservative span extraction before scoring HotpotQA outputs.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    if args.benchmark == "hotpotqa":
        metrics = score_hotpotqa(rows, postprocess=args.postprocess_hotpot_answer)
    elif args.benchmark == "medmcqa":
        metrics = score_medmcqa(rows)
    else:
        metrics = score_pubmedqa(rows)
    metrics["input_jsonl"] = str(args.input_jsonl)
    write_json(args.output_json, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
