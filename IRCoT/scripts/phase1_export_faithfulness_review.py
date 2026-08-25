#!/usr/bin/env python3
"""Export a compact faithfulness review file for manual spot-checking.

The export combines:
- stored system outputs,
- LLM-assisted faithfulness labels,
- NLI faithfulness labels,
- retrieved evidence shown to the model.

It does not overwrite or relabel any original result file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path | None) -> list[dict]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def key(row: dict) -> tuple[str, str]:
    return (str(row.get("benchmark", "")), str(row.get("question_id", "")))


def by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {key(row): row for row in rows}


def truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]) + " ..."


def compact_evidence(row: dict, max_passages: int, max_words: int) -> list[dict]:
    passages = row.get("evidence") or row.get("retrieved_passages") or []
    compact = []
    for passage in passages[:max_passages]:
        compact.append(
            {
                "rank": passage.get("rank", ""),
                "passage_id": passage.get("passage_id", ""),
                "title": passage.get("title", ""),
                "source_doc_id": passage.get("source_doc_id", ""),
                "corpus": passage.get("corpus", ""),
                "text": truncate_words(passage.get("text", ""), max_words),
            }
        )
    return compact


def confidence_value(row: dict) -> tuple[float | None, str]:
    for field in ("confidence", "parsed_confidence", "model_confidence"):
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None, f"invalid_{field}"
        if 0.0 <= confidence <= 1.0:
            return confidence, field
        return None, f"out_of_range_{field}"
    return None, "missing_in_legacy_output"


def priority(llm_label: str, nli_label: str) -> str:
    if llm_label == "unsupported" and nli_label == "supported":
        return "high_nli_false_faithful_candidate"
    if llm_label and nli_label and llm_label != nli_label:
        return "medium_label_disagreement"
    if not llm_label or not nli_label:
        return "medium_missing_label"
    return "normal"


def make_review_row(source: dict, llm: dict | None, nli: dict | None, args: argparse.Namespace) -> dict:
    llm = llm or {}
    nli = nli or {}
    confidence, confidence_status = confidence_value(source)
    llm_label = str(llm.get("manual_faithfulness_label", "")).strip()
    nli_label = str(nli.get("faithfulness_label", "")).strip()
    cited_or_supporting = source.get("supporting_evidence") or source.get("citations") or []
    return {
        "question_id": source.get("question_id", ""),
        "benchmark": source.get("benchmark", ""),
        "system": source.get("system", ""),
        "config_id": source.get("config_id", ""),
        "model": source.get("model", ""),
        "question": source.get("question", ""),
        "gold_answers": source.get("gold_answers", []),
        "gold_short_answer": source.get("gold_short_answer", ""),
        "predicted_answer": source.get("predicted_answer", ""),
        "raw_predicted_answer": source.get("raw_predicted_answer", source.get("predicted_answer", "")),
        "confidence": confidence,
        "confidence_status": confidence_status,
        "model_cited_or_supporting_evidence": cited_or_supporting,
        "model_cited_or_supporting_evidence_status": (
            "present" if cited_or_supporting else "missing_in_legacy_output"
        ),
        "supporting_evidence_quote_match": source.get("supporting_evidence_quote_match", None),
        "supporting_evidence_verification_status": source.get("supporting_evidence_verification_status", ""),
        "supporting_evidence_verification_method": source.get("supporting_evidence_verification_method", ""),
        "supporting_evidence_verified_passage_ids": source.get("supporting_evidence_verified_passage_ids", []),
        "supporting_evidence_verified_passage_titles": source.get("supporting_evidence_verified_passage_titles", []),
        "retrieved_evidence_for_review": compact_evidence(llm or source, args.max_passages, args.max_words_per_passage),
        "llm_assisted_label": llm_label,
        "llm_assisted_notes": llm.get("manual_notes", ""),
        "nli_label": nli_label,
        "nli_max_entailment": nli.get("max_entailment", None),
        "nli_max_contradiction": nli.get("max_contradiction", None),
        "nli_best_evidence_id": nli.get("best_evidence_id", ""),
        "nli_best_evidence_title": nli.get("best_evidence_title", ""),
        "review_priority": priority(llm_label, nli_label),
        "manual_review_label": "",
        "manual_review_notes": "",
        "manual_review_supported_evidence_ids": [],
        "allowed_manual_labels": ["supported", "unsupported", "contradicted", "abstained"],
        "review_instruction": (
            "Judge whether the model's answer is supported by its cited/supporting evidence when present. "
            "For legacy rows with no cited evidence, review the retrieved evidence and mark the cited-evidence "
            "gap in notes. Do not use outside knowledge."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--llm-labels-jsonl", type=Path, default=None)
    parser.add_argument("--nli-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-passages", type=int, default=8)
    parser.add_argument("--max-words-per-passage", type=int, default=160)
    parser.add_argument("--only-review-set", action="store_true")
    args = parser.parse_args()

    source_rows = load_jsonl(args.source_jsonl)
    llm_rows = load_jsonl(args.llm_labels_jsonl)
    nli_rows = load_jsonl(args.nli_jsonl)
    source_map = by_key(source_rows)
    llm_map = by_key(llm_rows)
    nli_map = by_key(nli_rows)

    keys = sorted(llm_map) if args.only_review_set and llm_map else sorted(source_map)
    review_rows = []
    missing_source = []
    for item_key in keys:
        source = source_map.get(item_key)
        if not source:
            missing_source.append(item_key)
            continue
        review_rows.append(make_review_row(source, llm_map.get(item_key), nli_map.get(item_key), args))

    write_jsonl(args.output_jsonl, review_rows)
    counts = {}
    for row in review_rows:
        counts[row["review_priority"]] = counts.get(row["review_priority"], 0) + 1
    print(
        json.dumps(
            {
                "rows_written": len(review_rows),
                "output_jsonl": str(args.output_jsonl),
                "priority_counts": counts,
                "missing_source_rows": missing_source,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
