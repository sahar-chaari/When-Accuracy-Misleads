#!/usr/bin/env python3
"""Compute risk/coverage metrics from confidence-aware stored outputs."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from phase1_confidence_utils import confidence_from_row
from phase1_answer_utils import MCQ_LABELS, parse_mcq_label, parse_pubmedqa_label
from phase1_score_outputs import (
    PUBMEDQA_LABELS,
    best_over_golds,
    exact_match,
    macro_f1,
    macro_f1_mcq,
    token_f1,
)
# parse_pubmedqa_label and MCQ_LABELS/parse_mcq_label imported from phase1_answer_utils above


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_thresholds(value: str) -> list[float]:
    if value == "default":
        return [round(i / 10, 1) for i in range(11)]
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1]: {threshold}")
    return thresholds


def pubmedqa_gold_pred(row: dict) -> tuple[str, str]:
    gold = str(row.get("gold_short_answer") or (row.get("gold_answers") or [""])[0]).lower()
    pred = row.get("parsed_label") or parse_pubmedqa_label(row.get("predicted_answer", ""))
    return gold, pred


def hotpot_scores(row: dict) -> tuple[float, float]:
    prediction = str(row.get("predicted_answer", ""))
    golds = [str(item) for item in row.get("gold_answers", [])]
    return best_over_golds(exact_match, prediction, golds), best_over_golds(token_f1, prediction, golds)


def row_confidence(row: dict, missing_policy: str) -> tuple[float | None, str]:
    confidence, status = confidence_from_row(row)
    if confidence is None and missing_policy == "answer_at_zero":
        return 0.0, "missing_treated_as_zero"
    return confidence, status


def pubmedqa_threshold_metrics(rows: list[dict], threshold: float, missing_policy: str) -> dict:
    answered = []
    missing_confidence = 0
    for row in rows:
        confidence, _ = row_confidence(row, missing_policy)
        if confidence is None:
            missing_confidence += 1
            continue
        if confidence >= threshold:
            answered.append(row)

    golds = []
    preds = []
    for row in answered:
        gold, pred = pubmedqa_gold_pred(row)
        golds.append(gold)
        preds.append(pred)
    correct = sum(g == p for g, p in zip(golds, preds))
    labels = list(PUBMEDQA_LABELS) + ["unknown"]
    confusion = {
        gold: {pred: sum(g == gold and p == pred for g, p in zip(golds, preds)) for pred in labels}
        for gold in PUBMEDQA_LABELS
    }
    return {
        "threshold": threshold,
        "total_count": len(rows),
        "answered_count": len(answered),
        "abstained_count": len(rows) - len(answered),
        "missing_confidence_count": missing_confidence,
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "accuracy_answered": correct / len(answered) if answered else None,
        "error_rate_answered": 1.0 - (correct / len(answered)) if answered else None,
        "macro_f1_answered": macro_f1(golds, preds) if answered else None,
        "prediction_counts_answered": dict(collections.Counter(preds)),
        "confusion_answered": confusion,
    }


def hotpotqa_threshold_metrics(rows: list[dict], threshold: float, missing_policy: str) -> dict:
    answered = []
    missing_confidence = 0
    for row in rows:
        confidence, _ = row_confidence(row, missing_policy)
        if confidence is None:
            missing_confidence += 1
            continue
        if confidence >= threshold:
            answered.append(row)

    ems = []
    f1s = []
    for row in answered:
        em, f1 = hotpot_scores(row)
        ems.append(em)
        f1s.append(f1)
    exact = sum(ems) / len(ems) if answered else None
    return {
        "threshold": threshold,
        "total_count": len(rows),
        "answered_count": len(answered),
        "abstained_count": len(rows) - len(answered),
        "missing_confidence_count": missing_confidence,
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "exact_match_answered": exact,
        "accuracy_answered": exact,
        "error_rate_answered": 1.0 - exact if exact is not None else None,
        "f1_answered": sum(f1s) / len(f1s) if answered else None,
    }


def medmcqa_threshold_metrics(rows: list[dict], threshold: float, missing_policy: str) -> dict:
    answered = []
    missing_confidence = 0
    for row in rows:
        confidence, _ = row_confidence(row, missing_policy)
        if confidence is None:
            missing_confidence += 1
            continue
        if confidence >= threshold:
            answered.append(row)

    golds = []
    preds = []
    for row in answered:
        gold = str(row.get("gold_short_answer") or (row.get("gold_answers") or [""])[0]).upper()
        pred = row.get("parsed_label") or parse_mcq_label(row.get("predicted_answer", ""))
        golds.append(gold)
        preds.append(pred)
    correct = sum(g == p for g, p in zip(golds, preds))
    labels = list(MCQ_LABELS) + ["unknown"]
    confusion = {
        gold: {pred: sum(g == gold and p == pred for g, p in zip(golds, preds)) for pred in labels}
        for gold in MCQ_LABELS
    }
    return {
        "threshold": threshold,
        "total_count": len(rows),
        "answered_count": len(answered),
        "abstained_count": len(rows) - len(answered),
        "missing_confidence_count": missing_confidence,
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "accuracy_answered": correct / len(answered) if answered else None,
        "error_rate_answered": 1.0 - (correct / len(answered)) if answered else None,
        "macro_f1_answered": macro_f1_mcq(golds, preds) if answered else None,
        "prediction_counts_answered": dict(collections.Counter(preds)),
        "confusion_answered": confusion,
    }


def confidence_summary(rows: list[dict], missing_policy: str) -> dict:
    values = []
    status_counts = collections.Counter()
    for row in rows:
        confidence, status = row_confidence(row, missing_policy)
        status_counts[status] += 1
        if confidence is not None:
            values.append(confidence)
    return {
        "confidence_available_count": len(values),
        "confidence_missing_count": len(rows) - len(values),
        "confidence_status_counts": dict(status_counts),
        "min_confidence": min(values) if values else None,
        "max_confidence": max(values) if values else None,
        "avg_confidence": sum(values) / len(values) if values else None,
    }


def select_threshold(tune_rows: list[dict], thresholds: list[float], metric_fn, missing_policy: str) -> float:
    """Return the threshold that maximises accuracy on the tune split."""
    best_threshold = thresholds[0]
    best_acc = -1.0
    for threshold in thresholds:
        m = metric_fn(tune_rows, threshold, missing_policy)
        acc = m.get("accuracy_answered") or 0.0
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["hotpotqa", "pubmedqa", "medmcqa"], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--thresholds", default="default")
    parser.add_argument(
        "--missing-confidence-policy",
        choices=["abstain", "answer_at_zero"],
        default="abstain",
        help="Default is abstain: legacy missing confidence is not invented.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help=(
            "If set, use only rows with split==<value> for reported metrics. "
            "Threshold selection is done on all rows whose split is NOT <value>. "
            "Use 'eval' when the JSONL rows carry a split field (tune/eval)."
        ),
    )
    args = parser.parse_args()

    all_rows = load_jsonl(args.input_jsonl)
    thresholds = parse_thresholds(args.thresholds)
    if args.benchmark == "pubmedqa":
        metric_fn = pubmedqa_threshold_metrics
    elif args.benchmark == "medmcqa":
        metric_fn = medmcqa_threshold_metrics
    else:
        metric_fn = hotpotqa_threshold_metrics

    # Partition into tune/eval if --eval-split is given.
    if args.eval_split:
        eval_rows = [r for r in all_rows if r.get("split") == args.eval_split]
        tune_rows = [r for r in all_rows if r.get("split") != args.eval_split]
        if not eval_rows:
            raise SystemExit(
                f"No rows with split='{args.eval_split}' found. "
                "Check that the JSONL rows have a 'split' field."
            )
        selected_threshold = select_threshold(tune_rows, thresholds, metric_fn, args.missing_confidence_policy)
        report_rows = eval_rows
        split_note = (
            f"Threshold {selected_threshold} selected on {len(tune_rows)} tune rows; "
            f"metrics reported on {len(eval_rows)} eval rows only."
        )
    else:
        report_rows = all_rows
        selected_threshold = None
        split_note = "No split used; thresholds swept over all rows."

    summary = confidence_summary(report_rows, args.missing_confidence_policy)
    unavailable = (
        args.missing_confidence_policy == "abstain"
        and report_rows
        and summary["confidence_available_count"] == 0
    )
    risk_coverage = [] if unavailable else [
        metric_fn(report_rows, threshold, args.missing_confidence_policy) for threshold in thresholds
    ]
    result = {
        "benchmark": args.benchmark,
        "input_jsonl": str(args.input_jsonl),
        "missing_confidence_policy": args.missing_confidence_policy,
        "eval_split": args.eval_split,
        "selected_threshold": selected_threshold,
        "split_note": split_note,
        **summary,
        "status": "unavailable_missing_confidence" if unavailable else "computed",
        "risk_coverage": risk_coverage,
        "note": (
            "Risk coverage unavailable because confidence was not collected."
            if unavailable
            else "Rows without explicit confidence are marked missing; confidence is never guessed."
        ),
    }
    write_json(args.output_json, result)
    if args.output_csv is not None:
        write_csv(args.output_csv, risk_coverage)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
