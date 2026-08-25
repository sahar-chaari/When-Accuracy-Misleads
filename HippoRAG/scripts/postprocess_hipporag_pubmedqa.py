#!/usr/bin/env python3
"""Post-process HippoRAG PubMedQA outputs for reliability metrics.

This script reads a completed HippoRAG JSONL run and creates the same
downstream artifacts expected for IRCoT comparison:

- confidence risk-coverage curves,
- evidence compliance summary,
- faithfulness scaffold and manual-review file,
- RAGChecker standard/cited exports,
- harm-cost scaffold and A/B/C severity labeling template.

It does not invent missing confidence or severity labels.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path
from typing import Any


PUBMEDQA_LABELS = ("yes", "no", "maybe")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def normalize_for_match(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def parse_pubmedqa_label(value: Any) -> str:
    text = clean_text(value).lower()
    match = re.search(r"\b(?:final\s+answer|answer)\s*:\s*(yes|no|maybe)\b", text)
    if match:
        return match.group(1)
    first_line = text.splitlines()[0] if text.splitlines() else text
    match = re.search(r"\b(yes|no|maybe)\b", first_line)
    if match:
        return match.group(1)
    match = re.search(r"\b(yes|no|maybe)\b", text)
    return match.group(1) if match else "unknown"


def row_gold_pred(row: dict[str, Any]) -> tuple[str, str]:
    gold_answers = row.get("gold_answers") or []
    gold = clean_text(row.get("gold_short_answer") or (gold_answers[0] if gold_answers else "")).lower()
    pred = clean_text(row.get("parsed_label") or parse_pubmedqa_label(row.get("predicted_answer"))).lower()
    return gold, pred


def row_confidence(row: dict[str, Any]) -> tuple[float | None, str]:
    for field in ("self_reported_confidence", "confidence"):
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None, f"invalid_{field}"
        if 0.0 <= confidence <= 1.0:
            return confidence, f"present_{field}"
        return None, f"out_of_range_{field}"
    return None, "missing"


def macro_f1(golds: list[str], preds: list[str]) -> float:
    scores = []
    for label in PUBMEDQA_LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def risk_coverage(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    confidence_values = []
    status_counts = collections.Counter()
    for row in rows:
        confidence, status = row_confidence(row)
        status_counts[status] += 1
        if confidence is not None:
            confidence_values.append(confidence)

    curve = []
    for threshold in [round(index / 10, 1) for index in range(11)]:
        answered = []
        missing = 0
        for row in rows:
            confidence, _ = row_confidence(row)
            if confidence is None:
                missing += 1
            elif confidence >= threshold:
                answered.append(row)

        golds, preds = [], []
        for row in answered:
            gold, pred = row_gold_pred(row)
            golds.append(gold)
            preds.append(pred)
        correct = sum(g == p for g, p in zip(golds, preds))
        curve.append(
            {
                "threshold": threshold,
                "total_count": len(rows),
                "answered_count": len(answered),
                "abstained_count": len(rows) - len(answered),
                "missing_confidence_count": missing,
                "coverage": len(answered) / len(rows) if rows else 0.0,
                "accuracy_answered": correct / len(answered) if answered else None,
                "error_rate_answered": 1.0 - (correct / len(answered)) if answered else None,
                "macro_f1_answered": macro_f1(golds, preds) if answered else None,
                "prediction_counts_answered": dict(collections.Counter(preds)),
            }
        )

    summary = {
        "confidence_available_count": len(confidence_values),
        "confidence_missing_count": len(rows) - len(confidence_values),
        "confidence_status_counts": dict(status_counts),
        "min_confidence": min(confidence_values) if confidence_values else None,
        "max_confidence": max(confidence_values) if confidence_values else None,
        "avg_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else None,
        "status": "computed" if confidence_values else "unavailable_missing_confidence",
        "note": "Rows without explicit confidence are treated as abstained; confidence is never guessed.",
    }
    return summary, curve


def passage_id(passage: dict[str, Any], fallback: int) -> str:
    return clean_text(
        passage.get("passage_id")
        or passage.get("doc_id")
        or passage.get("id")
        or passage.get("pmid")
        or f"passage_{fallback}"
    )


def passage_text(passage: dict[str, Any]) -> str:
    parts = []
    for label, field in (("Title", "title"), ("PMID", "pmid"), ("Section", "section")):
        value = clean_text(passage.get(field))
        if value:
            parts.append(f"{label}: {value}")
    text = clean_text(passage.get("text") or passage.get("hipporag_doc_text"))
    if text:
        parts.append(f"Text: {text}")
    return ". ".join(parts)


def verified_evidence_passages(row: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = clean_text(row.get("supporting_evidence"))
    if not evidence:
        return []
    normalized_evidence = normalize_for_match(evidence)
    if not normalized_evidence:
        return []
    matches = []
    for index, passage in enumerate(row.get("retrieved_passages") or []):
        text = passage_text(passage)
        if normalized_evidence in normalize_for_match(text):
            enriched = dict(passage)
            enriched["verified_passage_id"] = passage_id(passage, index)
            matches.append(enriched)
    return matches


def evidence_compliance(rows: list[dict[str, Any]], input_jsonl: Path) -> dict[str, Any]:
    present = 0
    verified = 0
    exact_flag_true = 0
    for row in rows:
        if clean_text(row.get("supporting_evidence")):
            present += 1
        if row.get("supporting_evidence_found_in_retrieved") is True:
            exact_flag_true += 1
        if verified_evidence_passages(row):
            verified += 1
    total = len(rows)
    return {
        "input_jsonl": str(input_jsonl),
        "rows": total,
        "supporting_evidence_present_count": present,
        "supporting_evidence_present_rate": present / total if total else 0.0,
        "supporting_evidence_found_flag_count": exact_flag_true,
        "supporting_evidence_found_flag_rate": exact_flag_true / total if total else 0.0,
        "supporting_evidence_verified_by_exporter_count": verified,
        "supporting_evidence_verified_by_exporter_rate": verified / total if total else 0.0,
    }


def canonical_gold_answer(row: dict[str, Any]) -> str:
    short = clean_text(row.get("gold_short_answer"))
    long = clean_text(row.get("gold_long_answer"))
    if long and short:
        return f"Short answer: {short}. Long answer: {long}"
    return short or clean_text((row.get("gold_answers") or [""])[0])


def ragchecker_response(row: dict[str, Any]) -> str:
    answer = clean_text(row.get("predicted_answer") or row.get("parsed_label"))
    evidence = clean_text(row.get("supporting_evidence"))
    confidence, _ = row_confidence(row)
    confidence_text = "" if confidence is None else str(confidence)
    return "\n".join(
        [
            f"Final answer: {answer}",
            f"Supporting evidence: {evidence}",
            f"Confidence: {confidence_text}",
        ]
    ).strip()


def ragchecker_export(rows: list[dict[str, Any]], input_jsonl: Path, mode: str, max_contexts: int | None) -> dict[str, Any]:
    exported = []
    for row in rows:
        passages = row.get("retrieved_passages") or []
        if mode == "cited":
            passages = verified_evidence_passages(row)
        if max_contexts is not None:
            passages = passages[:max_contexts]
        context = [
            {"doc_id": passage_id(passage, index), "text": passage_text(passage)}
            for index, passage in enumerate(passages)
            if passage_text(passage)
        ]
        exported.append(
            {
                "query_id": clean_text(row.get("question_id")),
                "query": clean_text(row.get("question")),
                "gt_answer": canonical_gold_answer(row),
                "response": ragchecker_response(row),
                "retrieved_context": context,
            }
        )

    empty_context_count = sum(not row["retrieved_context"] for row in exported)
    return {
        "results": exported,
        "export_metadata": {
            "source_jsonl": str(input_jsonl),
            "system": rows[0].get("system") if rows else "",
            "context_mode": mode,
            "response_mode": "answer_evidence_confidence",
            "max_contexts": max_contexts,
            "rows": len(exported),
            "empty_context_count": empty_context_count,
            "note": (
                "Standard mode uses all retrieved passages."
                if mode == "standard"
                else "Cited mode uses only retrieved passages that contain the model supporting evidence."
            ),
        },
    }


def faithfulness_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scaffold = []
    review = []
    for row in rows:
        gold, pred = row_gold_pred(row)
        confidence, _ = row_confidence(row)
        evidence_matches = verified_evidence_passages(row)
        wrong = gold != pred
        priority_reasons = []
        if not evidence_matches:
            priority_reasons.append("supporting_evidence_not_verified")
        if wrong and confidence is not None and confidence >= 0.8:
            priority_reasons.append("high_confidence_wrong_answer")
        if row.get("parse_errors"):
            priority_reasons.append("parse_error")
        priority = "high" if priority_reasons else "normal"

        scaffold.append(
            {
                "question_id": row.get("question_id", ""),
                "benchmark": row.get("benchmark", ""),
                "system": row.get("system", ""),
                "run_id": row.get("run_id", ""),
                "faithfulness_label": "not_scored",
                "unsupported_claims": [],
                "contradiction_claims": [],
                "evidence_ids_used": [passage_id(passage, idx) for idx, passage in enumerate(evidence_matches)],
                "judge_model": "",
                "rationale": "Scaffold only. Score with NLI, LLM-as-judge, RAGChecker, and manual review.",
            }
        )
        review.append(
            {
                "question_id": row.get("question_id", ""),
                "benchmark": row.get("benchmark", ""),
                "system": row.get("system", ""),
                "run_id": row.get("run_id", ""),
                "question": row.get("question", ""),
                "gold_answer": gold,
                "predicted_answer": row.get("predicted_answer", ""),
                "parsed_label": pred,
                "self_reported_confidence": confidence,
                "supporting_evidence": row.get("supporting_evidence", ""),
                "supporting_evidence_verified_passage_ids": [
                    passage_id(passage, idx) for idx, passage in enumerate(evidence_matches)
                ],
                "retrieved_passages": row.get("retrieved_passages", []),
                "llm_assisted_label": "",
                "nli_label": "",
                "ragchecker_standard_label": "",
                "ragchecker_cited_label": "",
                "manual_faithfulness_label": "",
                "allowed_manual_labels": ["supported", "unsupported", "contradicted", "abstained"],
                "priority": priority,
                "priority_reasons": priority_reasons,
                "manual_notes": "",
            }
        )
    return scaffold, review


def criticality(question: str, terms: list[str]) -> float:
    lowered = clean_text(question).lower()
    count = sum(1 for term in terms if term.lower() in lowered)
    return min(1.0, count / 3.0)


def base_cost(gold: str, pred: str, config: dict[str, Any]) -> float:
    costs = config["base_costs"]
    if pred == gold:
        return float(costs["correct"])
    if pred == "unknown":
        return float(costs["unknown"])
    if pred == "maybe" and gold in {"yes", "no"}:
        return float(costs["abstain_when_confident_answer_reasonable"])
    if gold == "maybe" and pred in {"yes", "no"}:
        return float(costs["wrong_on_maybe"])
    return float(costs["confident_wrong"])


def severity_class(row: dict[str, Any]) -> tuple[str | None, str]:
    for field in ("severity_class", "medical_severity_class", "iec62304_severity_class"):
        value = clean_text(row.get(field)).upper()
        if value in {"A", "B", "C"}:
            return value, f"field_{field}"
        if value:
            return None, f"invalid_{field}:{value}"
    return None, "missing_needs_label"


def harm_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scored = []
    severity_templates = []
    for row in rows:
        gold, pred = row_gold_pred(row)
        confidence, _ = row_confidence(row)
        severity, severity_status = severity_class(row)
        cost = base_cost(gold, pred, config)
        crit = criticality(row.get("question", ""), config["criticality"]["terms"])
        criticality_weighted = cost * (1.0 + float(config["criticality"]["multiplier"]) * crit)
        severity_weighted = {}
        confidence_severity_weighted = {}
        for name, weights in config.get("severity_weight_schemes", {}).items():
            if severity is None:
                severity_weighted[name] = None
                confidence_severity_weighted[name] = None
            else:
                weighted = cost * float(weights[severity])
                severity_weighted[name] = round(weighted, 4)
                confidence_severity_weighted[name] = (
                    round(weighted * confidence, 4) if confidence is not None else None
                )
        scored.append(
            {
                "question_id": row.get("question_id", ""),
                "benchmark": row.get("benchmark", ""),
                "system": row.get("system", ""),
                "run_id": row.get("run_id", ""),
                "gold": gold,
                "pred": pred,
                "self_reported_confidence": confidence,
                "base_harm_cost": round(cost, 4),
                "confidence_weighted_base_harm_cost": (
                    round(cost * confidence, 4) if confidence is not None else None
                ),
                "criticality_score": round(crit, 4),
                "criticality_weighted_harm_cost": round(criticality_weighted, 4),
                "severity_class": severity,
                "severity_label_status": severity_status,
                "severity_weighted_harm_costs": severity_weighted,
                "confidence_weighted_severity_harm_costs": confidence_severity_weighted,
                "config_note": "Scaffold defaults; severity labels and weights require paper-level justification.",
            }
        )
        if severity is None:
            severity_templates.append(
                {
                    "question_id": row.get("question_id", ""),
                    "benchmark": row.get("benchmark", ""),
                    "question": row.get("question", ""),
                    "gold_answer": gold,
                    "severity_class_A_B_C": "",
                    "allowed_severity_classes": ["A", "B", "C"],
                    "severity_class_definitions": config.get("severity_classes", {}),
                    "severity_rationale": "",
                    "manual_checked": False,
                    "notes": "",
                }
            )

    aggregate = {
        "rows": len(scored),
        "severity_labeled_count": sum(row["severity_class"] is not None for row in scored),
        "severity_missing_count": sum(row["severity_class"] is None for row in scored),
        "avg_base_harm_cost": sum(row["base_harm_cost"] for row in scored) / len(scored) if scored else None,
        "avg_confidence_weighted_base_harm_cost": (
            sum(row["confidence_weighted_base_harm_cost"] for row in scored if row["confidence_weighted_base_harm_cost"] is not None)
            / sum(row["confidence_weighted_base_harm_cost"] is not None for row in scored)
            if any(row["confidence_weighted_base_harm_cost"] is not None for row in scored)
            else None
        ),
        "note": "Severity-weighted aggregate is unavailable until A/B/C severity labels are filled.",
    }
    return scored, severity_templates, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--harm-config", type=Path, default=Path("configs/harm_cost_default.json"))
    parser.add_argument("--max-ragchecker-contexts", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    run_id = args.run_id or (rows[0].get("run_id") if rows else args.input_jsonl.stem)
    row_count = len(rows)
    root = args.output_root
    harm_config = load_json(args.harm_config)

    risk_summary, risk_rows = risk_coverage(rows)
    risk_payload = {
        "benchmark": "pubmedqa",
        "input_jsonl": str(args.input_jsonl),
        "run_id": run_id,
        **risk_summary,
        "risk_coverage": risk_rows,
    }
    risk_json = root / "results" / "risk_coverage" / f"risk_pubmedqa_{row_count}_{run_id}.json"
    risk_csv = root / "results" / "risk_coverage" / f"risk_pubmedqa_{row_count}_{run_id}.csv"
    write_json(risk_json, risk_payload)
    write_csv(risk_csv, risk_rows)

    compliance = evidence_compliance(rows, args.input_jsonl)
    compliance_json = (
        root
        / "results"
        / "faithfulness"
        / "evidence_compliance"
        / f"evidence_compliance_pubmedqa_{row_count}_{run_id}.json"
    )
    write_json(compliance_json, compliance)

    faith_scaffold, review_rows = faithfulness_rows(rows)
    faith_jsonl = (
        root
        / "results"
        / "faithfulness"
        / "scaffolds"
        / f"faithfulness_scaffold_pubmedqa_{row_count}_{run_id}.jsonl"
    )
    review_jsonl = (
        root
        / "annotations"
        / "faithfulness"
        / f"manual_review_pubmedqa_{row_count}_{run_id}.jsonl"
    )
    write_jsonl(faith_jsonl, faith_scaffold)
    write_jsonl(review_jsonl, review_rows)

    ragchecker_dir = root / "results" / "faithfulness" / "ragchecker_exports"
    ragchecker_outputs = []
    for mode in ("standard", "cited"):
        payload = ragchecker_export(rows, args.input_jsonl, mode, args.max_ragchecker_contexts)
        output = ragchecker_dir / f"{args.input_jsonl.stem}_ragchecker_{mode}.json"
        write_json(output, payload)
        ragchecker_outputs.append(
            {
                "mode": mode,
                "path": str(output),
                "rows": len(payload["results"]),
                "empty_context_count": payload["export_metadata"]["empty_context_count"],
            }
        )
    ragchecker_manifest = ragchecker_dir / f"{args.input_jsonl.stem}_ragchecker_export_manifest.json"
    write_json(
        ragchecker_manifest,
        {
            "input_jsonl": str(args.input_jsonl),
            "run_id": run_id,
            "outputs": ragchecker_outputs,
        },
    )

    harm_scaffold, severity_template, harm_aggregate = harm_rows(rows, harm_config)
    harm_jsonl = root / "results" / "harm_cost" / f"harm_cost_scaffold_pubmedqa_{row_count}_{run_id}.jsonl"
    severity_jsonl = (
        root
        / "annotations"
        / "harm_cost"
        / f"severity_template_pubmedqa_{row_count}_{run_id}.jsonl"
    )
    harm_summary_json = root / "results" / "harm_cost" / f"harm_cost_summary_pubmedqa_{row_count}_{run_id}.json"
    write_jsonl(harm_jsonl, harm_scaffold)
    write_jsonl(severity_jsonl, severity_template)
    write_json(harm_summary_json, harm_aggregate)

    manifest = {
        "input_jsonl": str(args.input_jsonl),
        "run_id": run_id,
        "rows": row_count,
        "outputs": {
            "risk_json": str(risk_json),
            "risk_csv": str(risk_csv),
            "evidence_compliance_json": str(compliance_json),
            "faithfulness_scaffold_jsonl": str(faith_jsonl),
            "manual_faithfulness_review_jsonl": str(review_jsonl),
            "ragchecker_manifest_json": str(ragchecker_manifest),
            "harm_cost_scaffold_jsonl": str(harm_jsonl),
            "severity_template_jsonl": str(severity_jsonl),
            "harm_cost_summary_json": str(harm_summary_json),
        },
        "risk_summary": risk_summary,
        "evidence_compliance": compliance,
        "harm_cost_summary": harm_aggregate,
    }
    manifest_path = root / "manifests" / f"{run_id}_postprocess_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
