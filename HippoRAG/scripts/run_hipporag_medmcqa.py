#!/usr/bin/env python3
"""Run official HippoRAG on MedMCQA with evidence/confidence MCQ output.

MedMCQA has no per-question gold corpus; retrieval is open-domain over the
supplied MedRAG textbooks/StatPearls JSONL corpus. The prompt asks for a
single letter answer (A/B/C/D), a supporting evidence quote, and a confidence
score in the same format used for PubMedQA (so downstream scorers work unchanged).
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
CONFIG_ID = "hipporag_qwen25_7b_medmcqa_evidence_conf_v1"
MCQ_LABELS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--official-root", default="external/official_hipporag")
    parser.add_argument("--run-id", default=time.strftime("medmcqa_%Y%m%d_%H%M%S"))
    parser.add_argument("--questions-jsonl", required=True)
    parser.add_argument("--corpus-jsonl", required=True,
                        help="MedRAG textbooks/StatPearls JSONL (passage_id, title, paragraph_text, ...).")
    parser.add_argument("--corpus-limit", type=int, default=None,
                        help="Limit corpus to this many passages (for debugging).")
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--manifest-json", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--llm-name", default=os.environ.get("HIPPORAG_LLM_NAME", "gpt-4o-mini"))
    parser.add_argument("--llm-base-url", default=os.environ.get("HIPPORAG_LLM_BASE_URL"))
    parser.add_argument("--embedding-name",
                        default=os.environ.get("HIPPORAG_EMBEDDING_NAME", "text-embedding-3-small"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("HIPPORAG_EMBEDDING_BASE_URL"))
    parser.add_argument("--openie-workers", type=int, default=int(os.environ.get("HIPPORAG_OPENIE_WORKERS", "0") or 0),
                        help="Override online OpenIE thread count for API-backed runs.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing index/openie in --save-dir (skip reindexing)")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_doc_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or row.get("passage_id") or "").strip()
    text = str(row.get("paragraph_text") or row.get("text") or "").strip()
    passage_id = str(row.get("passage_id") or "").strip()
    return f"[passage_id: {passage_id}; title: {title}] {text}"


def parse_mcq_label(text: str) -> str:
    text = text or ""
    match = re.search(r"\b(?:final\s+answer|answer)\s*:\s*([ABCD])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    first_line = text.splitlines()[0] if text.splitlines() else text
    match = re.search(r"\b([ABCD])\b", first_line)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1).upper() if match else "unknown"


def macro_f1_mcq(golds: list[str], preds: list[str]) -> float:
    scores = []
    for label in MCQ_LABELS:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def summarize_openie(save_dir: Path, hipporag: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "graph_nodes": None, "graph_edges": None, "graph_active": None,
        "fact_embedding_count": None, "graph_extracted_fact_count": None,
        "graph_total_triples": None,
        "openie_json_files": [], "openie_rows": None,
        "openie_extracted_triples": None, "openie_extracted_entities": None,
    }
    try:
        summary["graph_nodes"] = int(hipporag.graph.vcount())
        summary["graph_edges"] = int(hipporag.graph.ecount())
    except Exception as exc:
        summary["graph_error"] = repr(exc)
    try:
        summary["fact_embedding_count"] = len(hipporag.fact_embedding_store.hash_ids)
    except Exception as exc:
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

    summary["openie_rows"] = len(rows)
    summary["openie_extracted_triples"] = sum(len(r.get("extracted_triples") or []) for r in rows)
    summary["openie_extracted_entities"] = sum(len(r.get("extracted_entities") or []) for r in rows)
    graph_edges = summary.get("graph_edges")
    fact_count = summary.get("fact_embedding_count")
    summary["graph_active"] = bool((graph_edges or 0) > 0 or (fact_count or 0) > 0)
    summary["graph_extracted_fact_count"] = fact_count
    summary["graph_total_triples"] = graph_edges
    return summary


def write_metrics(rows: list[dict[str, Any]], output_json: Path) -> dict[str, Any]:
    golds = [str(row.get("gold_short_answer", "")).upper() for row in rows]
    preds = [str(row.get("parsed_label", "unknown")).upper() for row in rows]
    labels = list(MCQ_LABELS) + ["unknown"]
    metrics = {
        "benchmark": "medmcqa",
        "num_examples": len(rows),
        "accuracy": sum(g == p for g, p in zip(golds, preds)) / len(rows) if rows else 0.0,
        "macro_f1": macro_f1_mcq(golds, preds) if rows else 0.0,
        "unknown_prediction_count": sum(p == "unknown" for p in preds),
        "gold_counts": dict(Counter(golds)),
        "prediction_counts": dict(Counter(preds)),
        "confusion": {
            gold: {pred: sum(g == gold and p == pred for g, p in zip(golds, preds)) for pred in labels}
            for gold in MCQ_LABELS
        },
        "confidence_present_count": sum(row.get("self_reported_confidence") is not None for row in rows),
        "evidence_found_count": sum(bool(row.get("supporting_evidence_found_in_retrieved")) for row in rows),
        "error_count": sum(bool(row.get("errors")) for row in rows),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def configure_openie_workers(worker_count: int) -> None:
    if worker_count <= 0:
        return
    from concurrent.futures import ThreadPoolExecutor as OriginalThreadPoolExecutor
    import hipporag.information_extraction.openie_openai as openie_openai

    def thread_pool_executor_factory(*args, **kwargs):
        kwargs["max_workers"] = worker_count
        return OriginalThreadPoolExecutor(*args, **kwargs)

    openie_openai.ThreadPoolExecutor = thread_pool_executor_factory


def main() -> None:
    args = parse_args()
    args.llm_base_url = args.llm_base_url or None
    args.embedding_base_url = args.embedding_base_url or None
    project_dir = Path(args.project_dir).resolve()
    official_path = add_official_repo_to_path(project_dir, args.official_root)

    from hipporag import HippoRAG as HippoRAGImport
    from hipporag.utils.config_utils import BaseConfig

    configure_openie_workers(args.openie_workers)

    HippoRAGClass = getattr(HippoRAGImport, "HippoRAG", HippoRAGImport)

    questions = load_jsonl(Path(args.questions_jsonl))
    if args.question_limit is not None:
        questions = questions[: args.question_limit]
    if not questions:
        raise ValueError("No MedMCQA questions selected.")

    corpus_rows = load_jsonl(Path(args.corpus_jsonl))
    if args.corpus_limit is not None:
        corpus_rows = corpus_rows[: args.corpus_limit]
    if not corpus_rows:
        raise ValueError("No corpus rows loaded. Check --corpus-jsonl path.")

    doc_texts = [make_doc_text(row) for row in corpus_rows]
    doc_to_meta = {doc: row for doc, row in zip(doc_texts, corpus_rows)}

    output_jsonl = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else project_dir / "results" / "medmcqa" / f"{args.run_id}.jsonl"
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
        else project_dir / "results" / "medmcqa" / f"{args.run_id}_hipporag_workdir"
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
        seed=1,
        temperature=0,
        max_new_tokens=args.max_answer_tokens,
        force_index_from_scratch=not args.resume,
        force_openie_from_scratch=not args.resume,
    )

    prompt_template = (
        project_dir / "prompts" / "medmcqa_evidence_confidence_qa.txt"
    ).read_text(encoding="utf-8")
    started_at = time.time()
    api_backed = bool(args.llm_base_url) and not str(args.llm_name).startswith("Transformers/")
    system_label = "hipporag_official_remote" if api_backed else SYSTEM_LABEL
    backend_label = (
        "official_hipporag_openai_compatible"
        if api_backed
        else "official_hipporag_transformers_local"
    )

    hipporag = HippoRAGClass(global_config=config)
    hipporag.index(docs=doc_texts)
    openie_summary = summarize_openie(save_dir, hipporag)

    queries = [str(row["question"]) for row in questions]
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=args.top_k)

    rows = []
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for question_row, retrieval in zip(questions, retrieval_results):
            retrieved_passages = []
            for rank, doc in enumerate(retrieval.docs[: args.top_k], start=1):
                meta = doc_to_meta.get(doc, {})
                retrieved_passages.append(
                    {
                        "rank": rank,
                        "passage_id": meta.get("passage_id"),
                        "title": meta.get("title"),
                        "text": meta.get("paragraph_text") or meta.get("text", doc),
                        "hipporag_doc_text": doc,
                        "score": (
                            float(retrieval.doc_scores[rank - 1])
                            if getattr(retrieval, "doc_scores", None) is not None
                            else None
                        ),
                    }
                )

            messages = build_answer_messages(
                str(question_row["question"]), retrieved_passages, prompt_template
            )
            raw_generation, metadata, cache_hit = hipporag.llm_model.infer(
                messages,
                max_completion_tokens=args.max_answer_tokens,
            )
            parsed = parse_generation(raw_generation)
            parsed_label = parse_mcq_label(parsed["predicted_answer"])
            row = {
                "question_id": str(question_row.get("question_id", "")),
                "benchmark": "medmcqa",
                "split": question_row.get("split", ""),
                "system": system_label,
                "config_id": CONFIG_ID,
                "run_id": args.run_id,
                "model": args.llm_name,
                "embedding_model": args.embedding_name,
                "backend": backend_label,
                "question": question_row.get("question"),
                "options": question_row.get("options", {}),
                "gold_answers": [question_row.get("gold_short_answer")],
                "gold_short_answer": question_row.get("gold_short_answer"),
                "gold_answer_text": question_row.get("gold_answer_text", ""),
                "predicted_answer": parsed["predicted_answer"],
                "parsed_label": parsed_label,
                "raw_generation": raw_generation,
                "supporting_evidence": parsed["supporting_evidence"],
                "self_reported_confidence": parsed["self_reported_confidence"],
                "confidence": parsed["self_reported_confidence"],
                "parse_errors": parsed["parse_errors"],
                "supporting_evidence_found_in_retrieved": exact_evidence_found(
                    parsed["supporting_evidence"], retrieved_passages
                ),
                "retrieved_passages": retrieved_passages,
                "llm_metadata": metadata,
                "llm_cache_hit": cache_hit,
                "errors": [],
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{row['question_id']} gold={row['gold_short_answer']!r} "
                f"pred={parsed_label!r} conf={row['self_reported_confidence']}",
                flush=True,
            )

    metrics = write_metrics(rows, metrics_json)
    manifest = {
        "run_id": args.run_id,
        "system": system_label,
        "config_id": CONFIG_ID,
        "official_root": str(official_path),
        "questions_jsonl": str(args.questions_jsonl),
        "corpus_jsonl": str(args.corpus_jsonl),
        "corpus_limit": args.corpus_limit,
        "output_jsonl": str(output_jsonl),
        "metrics_json": str(metrics_json),
        "save_dir": str(save_dir),
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "embedding_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "backend": backend_label,
        "openie_workers": args.openie_workers or None,
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
