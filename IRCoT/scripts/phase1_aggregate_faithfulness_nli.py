#!/usr/bin/env python3
"""Aggregate NLI faithfulness scores (v2) from multiple output files into a summary table.

v2 adds: unsupported_not_found rate (fabricated/absent citation signal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ALL_LABELS = ("supported", "unsupported", "no_evidence_admitted", "contradicted", "abstained")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(rows: list[dict]) -> dict:
    labels = [r.get("faithfulness_label", "abstained") for r in rows]
    n = len(rows)
    counts = {label: sum(1 for l in labels if l == label) for label in ALL_LABELS}
    entailments = [r["max_entailment"] for r in rows if r.get("max_entailment") is not None]
    contradictions = [r["max_contradiction"] for r in rows if r.get("max_contradiction") is not None]
    presence_methods = [r.get("evidence_present_method", "none") for r in rows]
    method_counts = {}
    for m in ("substring", "fragment", "fuzzy", "none"):
        method_counts[m] = sum(1 for pm in presence_methods if pm == m)
    return {
        "n": n,
        "label_counts": counts,
        "supported_rate": round(counts["supported"] / n, 4) if n else 0.0,
        "unsupported_rate": round(counts["unsupported"] / n, 4) if n else 0.0,
        "no_evidence_admitted_rate": round(counts["no_evidence_admitted"] / n, 4) if n else 0.0,
        "contradicted_rate": round(counts["contradicted"] / n, 4) if n else 0.0,
        "abstained_rate": round(counts["abstained"] / n, 4) if n else 0.0,
        "avg_max_entailment": round(sum(entailments) / len(entailments), 4) if entailments else None,
        "avg_max_contradiction": round(sum(contradictions) / len(contradictions), 4) if contradictions else None,
        "presence_method_counts": method_counts,
        "judge_model": rows[0].get("judge_model", "") if rows else "",
        "evidence_policy": rows[0].get("evidence_policy", "") if rows else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    results = {}
    for path_str, label in zip(args.inputs, args.labels):
        path = Path(path_str)
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue
        rows = load_jsonl(path)
        results[label] = aggregate(rows)
        c = results[label]["label_counts"]
        print(
            f"{label}: n={results[label]['n']}  "
            f"supported={c['supported']}  unsupported={c['unsupported']}  "
            f"no_evidence={c['no_evidence_admitted']}  "
            f"contradicted={c['contradicted']}  abstained={c['abstained']}  "
            f"supported_rate={results[label]['supported_rate']}"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
