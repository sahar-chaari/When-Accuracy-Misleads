#!/usr/bin/env python3
"""Compute IEC 62304-inspired harm cost directly from severity ledgers."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from shared_inputs import FILES, BASE


ROOT = Path(BASE).resolve()
LABEL_FILES = {
    "MedMCQA": ROOT / "annotations" / "harm_cost" / "severity_labels_medmcqa_60_seed42.jsonl",
    "PubMedQA": ROOT / "annotations" / "harm_cost" / "severity_labels_pubmedqa_60_seed42.jsonl",
}
OUT_JSON = ROOT / "results" / "analysis" / "harm_cost_direct_20260615.json"
OUT_CSV = ROOT / "results" / "analysis" / "harm_cost_direct_20260615.csv"
OUT_TEX = ROOT / "results" / "analysis" / "harm_cost_table_rows_20260615.tex"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_mcq_label(text: object) -> str:
    value = str(text or "").strip().upper()
    if value in {"A", "B", "C", "D"}:
        return value
    match = re.search(r"\b(?:FINAL\s+ANSWER|ANSWER)\s*[:\-]?\s*([ABCD])\b", value)
    if match:
        return match.group(1)
    first_line = value.splitlines()[0] if value.splitlines() else value
    match = re.search(r"\b([ABCD])\b", first_line)
    return match.group(1) if match else "unknown"


def parse_pubmedqa_label(text: object) -> str:
    value = str(text or "").strip().lower()
    if value in {"yes", "no", "maybe"}:
        return value
    match = re.search(r"\b(?:final\s+answer|answer)\s*[:\-]?\s*(yes|no|maybe)\b", value)
    if match:
        return match.group(1)
    match = re.search(r"\b(yes|no|maybe)\b", value)
    return match.group(1) if match else "unknown"


def gold(row: dict[str, Any], benchmark: str) -> str:
    value = row.get("gold_short_answer") or (row.get("gold_answers") or [""])[0]
    text = str(value).strip()
    return text.upper() if benchmark == "MedMCQA" else text.lower()


def pred(row: dict[str, Any], benchmark: str) -> str:
    value = row.get("parsed_label") or row.get("predicted_answer")
    return parse_mcq_label(value) if benchmark == "MedMCQA" else parse_pubmedqa_label(value)


def load_labels() -> dict[str, dict[str, dict[str, Any]]]:
    labels: dict[str, dict[str, dict[str, Any]]] = {}
    for benchmark, path in LABEL_FILES.items():
        rows = load_jsonl(path)
        labels[benchmark] = {str(row["question_id"]): row for row in rows}
        if len(labels[benchmark]) != 60:
            raise RuntimeError(f"{benchmark} severity ledger has {len(labels[benchmark])} rows, expected 60")
    return labels


def tex_rows(results: dict[str, Any]) -> str:
    lines = []
    for family, systems in [
        ("IRCoT", ["IRCoT 7B", "IRCoT 32B"]),
        ("CRAG", ["CRAG 7B", "CRAG 32B"]),
        ("HippoRAG~2", ["HippoRAG 7B", "HippoRAG 32B"]),
    ]:
        for i, system in enumerate(systems):
            row_name = rf"\multirow{{2}}{{*}}{{{family}}}" if i == 0 else " " * 28
            size = "7B" if system.endswith("7B") else "32B"
            med = results[f"{system}|MedMCQA"]["H_rounded_3"]
            pub = results[f"{system}|PubMedQA"]["H_rounded_3"]
            lines.append(f"{row_name} & {size} & {med:.3f} & {pub:.3f} " + r"\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    labels = load_labels()
    outputs: dict[str, Any] = {}
    rows_for_csv: list[dict[str, Any]] = []
    label_summary: dict[str, Any] = {}

    for benchmark, rows in labels.items():
        weights = [int(row["severity_weight"]) for row in rows.values()]
        classes = [str(row["severity_class"]) for row in rows.values()]
        label_summary[benchmark] = {
            "n_questions": len(rows),
            "class_counts": dict(Counter(classes)),
            "total_severity_weight": sum(weights),
        }

    for system, benchmark, rel_path in FILES:
        path = ROOT / rel_path
        source_rows = load_jsonl(path)
        ledger = labels[benchmark]
        total_weight = sum(int(row["severity_weight"]) for row in ledger.values())
        errors = []
        weighted_error = 0
        for row in source_rows:
            qid = str(row["question_id"])
            severity = ledger[qid]
            correct = gold(row, benchmark) == pred(row, benchmark)
            if not correct:
                weighted_error += int(severity["severity_weight"])
                errors.append(
                    {
                        "question_id": qid,
                        "gold": gold(row, benchmark),
                        "pred": pred(row, benchmark),
                        "severity_class": severity["severity_class"],
                        "severity_weight": int(severity["severity_weight"]),
                    }
                )
        key = f"{system}|{benchmark}"
        h_value = weighted_error / total_weight
        outputs[key] = {
            "system": system,
            "benchmark": benchmark,
            "n": len(source_rows),
            "errors": len(errors),
            "weighted_error": weighted_error,
            "total_severity_weight": total_weight,
            "H": h_value,
            "H_rounded_3": round(h_value, 3),
            "error_class_counts": dict(Counter(error["severity_class"] for error in errors)),
            "input_jsonl": rel_path,
        }
        rows_for_csv.append(outputs[key])

    result = {
        "method": (
            "Direct IEC 62304-inspired harm cost from released question-level "
            "severity ledgers; H=sum(error weights)/sum(question weights)."
        ),
        "weights": {"A": 1, "B": 4, "C": 10},
        "label_summary": label_summary,
        "results": outputs,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system",
                "benchmark",
                "n",
                "errors",
                "weighted_error",
                "total_severity_weight",
                "H",
                "H_rounded_3",
                "error_class_counts",
                "input_jsonl",
            ],
        )
        writer.writeheader()
        for row in rows_for_csv:
            writer.writerow(row)
    OUT_TEX.write_text(tex_rows(outputs), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
