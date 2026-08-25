#!/usr/bin/env python3
"""Run official HippoRAG on PubMedQA with evidence/confidence output."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_hipporag_smoke import (
    add_official_repo_to_path,
    build_answer_messages,
    exact_evidence_found,
    parse_generation,
)


SYSTEM_LABEL = "hipporag_official_hf_local"
CONFIG_ID = "hipporag_qwen25_7b_pubmedqa_evidence_conf_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--official-root", default="external/official_hipporag")
    parser.add_argument("--run-id", default=time.strftime("pubmedqa_%Y%m%d_%H%M%S"))
    parser.add_argument("--questions-jsonl", required=True)
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--restrict-corpus-to-question-pmids", action="store_true")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--manifest-json", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--llm-name", default=os.environ.get("HIPPORAG_LLM_NAME", "gpt-4o-mini"))
    parser.add_argument("--llm-base-url", default=os.environ.get("HIPPORAG_LLM_BASE_URL"))
    parser.add_argument("--embedding-name", default=os.environ.get("HIPPORAG_EMBEDDING_NAME", "text-embedding-3-small"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("HIPPORAG_EMBEDDING_BASE_URL"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true",
                        help="Skip indexing/NER if save-dir already has graph+openie artifacts.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_doc_text(row: dict[str, Any]) -> str:
    section = str(row.get("section") or "PASSAGE").strip()
    passage_id = str(row.get("passage_id") or "").strip()
    pmid = str(row.get("pmid") or "").strip()
    text = str(row.get("text") or "").strip()
    return f"[passage_id: {passage_id}; pmid: {pmid}; section: {section}] {text}"


def parse_pubmedqa_label(answer: str | None) -> str:
    if not answer:
        return "unknown"
    normalized = str(answer).strip().lower()
    normalized = normalized.replace("final answer:", "").strip()
    for char in ".:,;!?'\"`":
        normalized = normalized.replace(char, " ")
    tokens = normalized.split()
    for label in ("yes", "no", "maybe"):
        if tokens and tokens[0] == label:
            return label
    for label in ("yes", "no", "maybe"):
        if label in tokens:
            return label
    return "unknown"


def summarize_openie(save_dir: Path, hipporag: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "graph_nodes": None,
        "graph_edges": None,
        "graph_active": None,
        "fact_embedding_count": None,
        "graph_extracted_fact_count": None,
        "graph_total_triples": None,
        "openie_json_files": [],
        "openie_rows": None,
        "openie_extracted_triples": None,
        "openie_extracted_entities": None,
        "notes": [],
    }

    try:
        summary["graph_nodes"] = int(hipporag.graph.vcount())
        summary["graph_edges"] = int(hipporag.graph.ecount())
    except Exception as exc:  # pragma: no cover - diagnostic only
        summary["graph_error"] = repr(exc)

    try:
        summary["fact_embedding_count"] = len(hipporag.fact_embedding_store.hash_ids)
    except Exception as exc:  # pragma: no cover - diagnostic only
        summary["fact_embedding_error"] = repr(exc)

    rows = []
    for path in sorted(save_dir.glob("openie_results*.json")):
        summary["openie_json_files"].append(str(path))
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary.setdefault("openie_read_errors", {})[str(path)] = repr(exc)
            continue
        if isinstance(obj, list):
            rows.extend(obj)
        elif isinstance(obj, dict) and "docs" in obj:
            rows.extend(obj["docs"])

    if rows:
        summary["openie_rows"] = len(rows)
        summary["openie_extracted_triples"] = sum(len(row.get("extracted_triples") or []) for row in rows)
        summary["openie_extracted_entities"] = sum(len(row.get("extracted_entities") or []) for row in rows)
    else:
        summary["openie_rows"] = 0
        summary["openie_extracted_triples"] = 0
        summary["openie_extracted_entities"] = 0

    graph_edges = summary.get("graph_edges")
    fact_count = summary.get("fact_embedding_count")
    summary["graph_active"] = bool((graph_edges or 0) > 0 or (fact_count or 0) > 0)
    summary["graph_extracted_fact_count"] = fact_count
    summary["graph_total_triples"] = graph_edges
    if summary["graph_active"] and summary["openie_extracted_triples"] == 0:
        summary["notes"].append(
            "Raw openie_results JSON rows were not counted, but HippoRAG graph/fact stores are non-empty."
        )

    return summary


def macro_f1(golds: list[str], preds: list[str]) -> float:
    scores = []
    for label in ("yes", "no", "maybe"):
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def write_metrics(rows: list[dict[str, Any]], output_json: Path) -> dict[str, Any]:
    golds = [str(row.get("gold_short_answer", "")).lower() for row in rows]
    preds = [str(row.get("parsed_label", "unknown")).lower() for row in rows]
    labels = ["yes", "no", "maybe", "unknown"]
    metrics = {
        "benchmark": "pubmedqa",
        "num_examples": len(rows),
        "accuracy": sum(g == p for g, p in zip(golds, preds)) / len(rows) if rows else 0.0,
        "macro_f1": macro_f1(golds, preds) if rows else 0.0,
        "unknown_prediction_count": sum(p == "unknown" for p in preds),
        "gold_counts": dict(Counter(golds)),
        "prediction_counts": dict(Counter(preds)),
        "confusion": {
            gold: {pred: sum(g == gold and p == pred for g, p in zip(golds, preds)) for pred in labels}
            for gold in ("yes", "no", "maybe")
        },
        "avg_retrieval_recall_pmid": (
            sum(float(row.get("retrieval_recall_pmid") or 0.0) for row in rows) / len(rows) if rows else None
        ),
        "error_count": sum(bool(row.get("errors")) for row in rows),
    }
    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    args.llm_base_url = args.llm_base_url or None
    args.embedding_base_url = args.embedding_base_url or None
    project_dir = Path(args.project_dir).resolve()
    official_path = add_official_repo_to_path(project_dir, args.official_root)

    from hipporag import HippoRAG as HippoRAGImport
    from hipporag.utils.config_utils import BaseConfig

    HippoRAGClass = getattr(HippoRAGImport, "HippoRAG", HippoRAGImport)

    questions = load_jsonl(Path(args.questions_jsonl))
    if args.question_limit is not None:
        questions = questions[: args.question_limit]

    corpus_rows = load_jsonl(Path(args.corpus_jsonl))
    question_pmids = {str(row.get("id")) for row in questions}
    if args.restrict_corpus_to_question_pmids:
        corpus_rows = [row for row in corpus_rows if str(row.get("pmid")) in question_pmids]
    if not corpus_rows:
        raise ValueError("No corpus rows selected.")

    doc_texts = [make_doc_text(row) for row in corpus_rows]
    doc_to_meta = {doc: row for doc, row in zip(doc_texts, corpus_rows)}

    output_jsonl = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else project_dir / "results" / "pubmedqa" / f"{args.run_id}.jsonl"
    )
    manifest_json = (
        Path(args.manifest_json)
        if args.manifest_json
        else project_dir / "manifests" / f"{args.run_id}.manifest.json"
    )
    metrics_json = output_jsonl.with_name(output_jsonl.stem + ".metrics.json")
    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else project_dir / "results" / "pubmedqa" / f"{args.run_id}_hipporag_workdir"
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = BaseConfig(
        save_dir=str(save_dir),
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_name,
        embedding_base_url=args.embedding_base_url,
        openie_mode="online",
        retrieval_top_k=args.top_k,
        qa_top_k=args.top_k,
        temperature=0,
        max_new_tokens=args.max_answer_tokens,
        force_index_from_scratch=not args.resume,
        force_openie_from_scratch=not args.resume,
    )

    prompt_template = (project_dir / "prompts" / "pubmedqa_evidence_confidence_qa.txt").read_text(encoding="utf-8")
    started_at = time.time()

    hipporag = HippoRAGClass(global_config=config)
    hipporag.index(docs=doc_texts)
    openie_summary = summarize_openie(save_dir, hipporag)

    queries = [str(row["question"]) for row in questions]
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=args.top_k)

    rows = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for question_row, retrieval in zip(questions, retrieval_results):
            retrieved_passages = []
            gold_passage_ids = [
                str(row.get("passage_id"))
                for row in corpus_rows
                if str(row.get("pmid")) == str(question_row.get("id"))
            ]
            for rank, doc in enumerate(retrieval.docs[: args.top_k], start=1):
                meta = doc_to_meta.get(doc, {})
                retrieved_passages.append(
                    {
                        "rank": rank,
                        "passage_id": meta.get("passage_id"),
                        "pmid": meta.get("pmid"),
                        "section": meta.get("section"),
                        "text": meta.get("text", doc),
                        "hipporag_doc_text": doc,
                        "score": (
                            float(retrieval.doc_scores[rank - 1])
                            if getattr(retrieval, "doc_scores", None) is not None
                            else None
                        ),
                    }
                )

            messages = build_answer_messages(str(question_row["question"]), retrieved_passages, prompt_template)
            raw_generation, metadata, cache_hit = hipporag.llm_model.infer(
                messages,
                max_completion_tokens=args.max_answer_tokens,
            )
            parsed = parse_generation(raw_generation)
            parsed_label = parse_pubmedqa_label(parsed["predicted_answer"])
            retrieved_ids = {str(row.get("passage_id")) for row in retrieved_passages}
            retrieval_recall_pmid = (
                len(set(gold_passage_ids) & retrieved_ids) / len(gold_passage_ids) if gold_passage_ids else None
            )
            row = {
                "question_id": str(question_row.get("id")),
                "benchmark": "pubmedqa",
                "split": question_row.get("split"),
                "system": SYSTEM_LABEL,
                "config_id": CONFIG_ID,
                "run_id": args.run_id,
                "model": args.llm_name,
                "embedding_model": args.embedding_name,
                "backend": "official_hipporag_transformers_local",
                "question": question_row.get("question"),
                "gold_answers": [question_row.get("gold_short_answer")],
                "gold_short_answer": question_row.get("gold_short_answer"),
                "gold_long_answer": question_row.get("gold_long_answer"),
                "predicted_answer": parsed["predicted_answer"],
                "parsed_label": parsed_label,
                "raw_generation": raw_generation,
                "supporting_evidence": parsed["supporting_evidence"],
                "self_reported_confidence": parsed["self_reported_confidence"],
                "parse_errors": parsed["parse_errors"],
                "supporting_evidence_found_in_retrieved": exact_evidence_found(
                    parsed["supporting_evidence"], retrieved_passages
                ),
                "retrieved_passages": retrieved_passages,
                "gold_evidence_ids": gold_passage_ids,
                "retrieval_recall_pmid": retrieval_recall_pmid,
                "llm_metadata": metadata,
                "llm_cache_hit": cache_hit,
                "errors": [],
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{row['question_id']} gold={row['gold_short_answer']!r} "
                f"pred={row['predicted_answer']!r} label={row['parsed_label']!r} "
                f"conf={row['self_reported_confidence']}"
            )

    metrics = write_metrics(rows, metrics_json)
    manifest = {
        "run_id": args.run_id,
        "system": SYSTEM_LABEL,
        "config_id": CONFIG_ID,
        "official_root": str(official_path),
        "questions_jsonl": str(args.questions_jsonl),
        "corpus_jsonl": str(args.corpus_jsonl),
        "restrict_corpus_to_question_pmids": args.restrict_corpus_to_question_pmids,
        "output_jsonl": str(output_jsonl),
        "metrics_json": str(metrics_json),
        "save_dir": str(save_dir),
        "llm_name": args.llm_name,
        "embedding_name": args.embedding_name,
        "num_corpus_passages": len(corpus_rows),
        "num_questions": len(questions),
        "gold_counts": dict(Counter(str(row.get("gold_short_answer")) for row in questions)),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "rows": len(rows),
        "parse_error_count": sum(1 for row in rows if row["parse_errors"]),
        "evidence_found_count": sum(1 for row in rows if row["supporting_evidence_found_in_retrieved"]),
        "openie_summary": openie_summary,
        "metrics": metrics,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
