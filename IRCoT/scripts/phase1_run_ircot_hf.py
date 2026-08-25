#!/usr/bin/env python3
"""Run official IRCoT with the patched hf_local backend and save JSONL rows."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from phase1_answer_utils import parse_mcq_label, parse_pubmedqa_label, postprocess_hotpot_answer
from phase1_confidence_utils import (
    parse_confidence_detailed,
    parse_final_answer_detailed,
    parse_supporting_evidence_detailed,
)
from phase1_evidence_utils import verify_supporting_evidence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRCOT_ROOT = PROJECT_ROOT / "external" / "ircot"


def configure_imports() -> None:
    sys.path.insert(0, str(IRCOT_ROOT))


def load_jsonnet_config(config_path: Path) -> dict:
    import _jsonnet

    ext_vars = {key: value for key, value in os.environ.items() if isinstance(value, str)}
    return json.loads(_jsonnet.evaluate_file(str(config_path), ext_vars=ext_vars))


def make_reader(config_map: dict):
    from commaqa.inference.constants import READER_NAME_CLASS

    reader_config = dict(config_map["reader"])
    reader_name = reader_config.pop("name")
    return READER_NAME_CLASS[reader_name](**reader_config)


def final_answer_from_state(final_state) -> str:
    if final_state is None:
        return ""
    data = final_state._data
    final_answer = data.get_last_answer()
    try:
        json_answer = json.loads(final_answer)
        if isinstance(json_answer, (list, str)):
            final_answer = json_answer
    except ValueError:
        pass
    if isinstance(final_answer, list):
        final_answer = " ".join(str(item) for item in final_answer)
    return str(final_answer)


def extract_citations(text: str) -> list[str]:
    return re.findall(r"\[(?:P|R|D)\d+\]|\bPMID:\d+\b", text or "")


def normalize_retrieved_passages(data: dict) -> list[dict]:
    trace = data.get("retrieval_trace") or []
    if trace:
        out = []
        for idx, item in enumerate(trace, start=1):
            out.append(
                {
                    "rank": item.get("rank", idx),
                    "passage_id": item.get("passage_id", ""),
                    "source_doc_id": item.get("source_doc_id", ""),
                    "corpus": item.get("corpus", ""),
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                    "score": item.get("score", ""),
                    "retrieval_step": item.get("retrieval_step", ""),
                    "query": item.get("query", ""),
                }
            )
        return out

    titles = data.get("titles", [])
    paras = data.get("paras", [])
    return [
        {
            "rank": idx,
            "passage_id": "",
            "source_doc_id": title,
            "corpus": "",
            "title": title,
            "text": para,
            "score": "",
            "retrieval_step": "",
            "query": "",
        }
        for idx, (title, para) in enumerate(zip(titles, paras), start=1)
    ]


def gold_from_example(example: dict) -> tuple[list[str], str, str, list[str]]:
    answer = example.get("answer", "")
    if isinstance(answer, (list, tuple)):
        gold_answers = [str(item) for item in answer]
    else:
        gold_answers = [str(answer)]
    metadata = example.get("metadata", {})
    gold_ids = [str(item) for item in metadata.get("gold_ids", []) if item]
    return gold_answers, gold_answers[0] if gold_answers else "", example.get("gold_long_answer", ""), gold_ids


def extract_pmids(text: str) -> set[str]:
    return set(re.findall(r"\b(?:PMID:)?(\d{5,9})\b", text or ""))


def retrieval_recall(example: dict, retrieved: list[dict], benchmark: str) -> tuple[float | None, float | None]:
    metadata = example.get("metadata", {})
    gold_ids = {str(item) for item in metadata.get("gold_ids", []) if item}
    retrieved_ids = {str(item.get("passage_id", "")) for item in retrieved if item.get("passage_id")}
    if gold_ids and retrieved_ids:
        return len(gold_ids & retrieved_ids) / len(gold_ids), None

    gold_titles = {str(item).lower() for item in metadata.get("gold_titles", []) if item}
    retrieved_titles = {str(item.get("title", "")).lower() for item in retrieved if item.get("title")}
    title_recall = len(gold_titles & retrieved_titles) / len(gold_titles) if gold_titles else None

    if benchmark != "pubmedqa":
        return title_recall, None

    # question_id holds the PMID in the IRCoT PubMedQA format; fall back to qid for compatibility.
    qid_str = example.get("question_id", "") or example.get("qid", "")
    gold_pmids = extract_pmids(qid_str) | extract_pmids(" ".join(gold_titles))
    # Also collect PMIDs from gold context IDs (format: "PMID::P1").
    for ctx in example.get("contexts", []):
        gold_pmids |= extract_pmids(str(ctx.get("id", "")))
        gold_pmids |= extract_pmids(str(ctx.get("title", "")))

    retrieved_pmids = set()
    for item in retrieved:
        retrieved_pmids |= extract_pmids(item.get("passage_id", ""))
        retrieved_pmids |= extract_pmids(item.get("source_doc_id", ""))
        retrieved_pmids |= extract_pmids(item.get("title", ""))
    pmid_recall = len(gold_pmids & retrieved_pmids) / len(gold_pmids) if gold_pmids else None
    return title_recall, pmid_recall


def classify_abstention_status(answer: str, errors: list[str]) -> str:
    if errors:
        return "error"
    if not (answer or "").strip():
        return "missing_answer"
    if re.search(
        r"\b(abstain|cannot answer|can't answer|unknown|not enough information|insufficient evidence|not provided)\b",
        answer,
        flags=re.IGNORECASE,
    ):
        return "model_abstained"
    return "answered"


def make_output_row(example: dict, final_state, benchmark: str, args: argparse.Namespace, seconds: float, errors: list[str]):
    data = final_state._data if final_state is not None else {}
    raw_predicted = final_answer_from_state(final_state)
    final_answer = parse_final_answer_detailed(raw_predicted)
    supporting_evidence = parse_supporting_evidence_detailed(raw_predicted)
    confidence = parse_confidence_detailed(raw_predicted)
    answer_for_metrics = (
        final_answer["value"]
        if final_answer["status"] not in {"missing", "raw_text"} and final_answer["value"]
        else raw_predicted
    )
    predicted = answer_for_metrics
    answer_postprocess = None
    if benchmark == "hotpotqa":
        predicted, rule = postprocess_hotpot_answer(answer_for_metrics)
        answer_postprocess = {
            "method": "phase1_hotpot_conservative_span_extraction",
            "rule": rule,
            "changed": predicted != answer_for_metrics,
        }
    if benchmark == "pubmedqa":
        parsed_label = parse_pubmedqa_label(predicted)
    elif benchmark == "medmcqa":
        parsed_label = parse_mcq_label(predicted)
    else:
        parsed_label = ""
    retrieved = normalize_retrieved_passages(data)
    evidence_verification = verify_supporting_evidence(supporting_evidence["value"], retrieved)
    gold_answers, gold_short, gold_long, gold_ids = gold_from_example(example)
    recall, pmid_recall = retrieval_recall(example, retrieved, benchmark)
    raw_chain = data.get_printable_reasoning_chain() if final_state is not None else ""
    return {
        "question_id": example.get("qid", ""),
        "benchmark": benchmark,
        "split": example.get("split", ""),
        "system": args.system,
        "system_version": args.system_version,
        "config_id": args.config_id,
        "model": args.model,
        "backend": "hf_local",
        "question": example.get("question", ""),
        "gold_answers": gold_answers,
        "gold_short_answer": gold_short,
        "gold_long_answer": gold_long,
        "raw_predicted_answer": raw_predicted,
        "predicted_answer": predicted,
        "structured_final_answer": final_answer["value"],
        "structured_final_answer_raw": final_answer["raw"],
        "structured_final_answer_status": final_answer["status"],
        "structured_final_answer_parse_error": final_answer["parse_error"],
        "supporting_evidence_raw": supporting_evidence["raw"],
        "supporting_evidence": supporting_evidence["value"],
        "supporting_evidence_status": supporting_evidence["status"],
        "supporting_evidence_parse_error": supporting_evidence["parse_error"],
        "supporting_evidence_quote_match": evidence_verification["quote_match"],
        "supporting_evidence_verification_status": evidence_verification["status"],
        "supporting_evidence_verification_method": evidence_verification["method"],
        "supporting_evidence_verified_passage_ids": evidence_verification["verified_passage_ids"],
        "supporting_evidence_verified_passage_titles": evidence_verification["verified_passage_titles"],
        "self_reported_confidence": confidence["value"],
        "confidence_raw": confidence["raw"],
        "confidence_parse_error": confidence["parse_error"],
        "confidence": confidence["value"],
        "confidence_status": confidence["status"],
        "abstention_status": classify_abstention_status(predicted, errors),
        "answer_postprocess": answer_postprocess,
        "parsed_label": parsed_label,
        "raw_generation": raw_chain or predicted,
        "reasoning_trace": data.get("generated_sentences", []) if final_state is not None else [],
        "retrieved_passages": retrieved,
        "citations": extract_citations(raw_predicted) + extract_citations(supporting_evidence["value"]),
        "gold_evidence_ids": gold_ids,
        "retrieval_recall": recall,
        "retrieval_recall_pmid": pmid_recall,
        "timing": {
            "retrieval_seconds": None,
            "generation_seconds": None,
            "total_seconds": round(seconds, 4),
        },
        "errors": errors,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["hotpotqa", "pubmedqa", "medmcqa"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--system", default="ircot_official_hf_local")
    parser.add_argument("--system-version", default="")
    parser.add_argument("--config-id", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_imports()
    from commaqa.inference.configurable_inference import build_decomposer_and_models

    config_path = args.config.resolve()
    input_path = args.input.resolve()
    output_path = args.output_jsonl.resolve()

    os.environ.setdefault("PROJECT_DIR", str(PROJECT_ROOT))
    os.environ["MODEL"] = args.model
    os.chdir(IRCOT_ROOT)
    config_map = load_jsonnet_config(config_path)
    decomposer, _ = build_decomposer_and_models(config_map)
    reader = make_reader(config_map)

    rows = []
    examples = list(reader.read_examples(str(input_path)))
    if args.limit is not None:
        examples = examples[: args.limit]

    for index, example in enumerate(examples, start=1):
        started = time.time()
        errors = []
        final_state = None
        try:
            final_state, _ = decomposer.find_answer_decomp(example, debug=args.debug)
            if final_state is None:
                errors.append("ircot_failed_no_final_state")
        except Exception as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
        row = make_output_row(example, final_state, args.benchmark, args, time.time() - started, errors)
        rows.append(row)
        write_jsonl(output_path, rows)
        print(f"[{index}/{len(examples)}] {row['question_id']} pred={row['predicted_answer'][:80]!r} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
