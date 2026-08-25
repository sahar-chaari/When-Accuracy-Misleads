#!/usr/bin/env python3
"""Tiny smoke test for the official HippoRAG wrapper.

This script intentionally uses a synthetic 3-question corpus. It is only meant
to verify that the official HippoRAG package can index, retrieve, and produce
answer/evidence/confidence rows in our shared output schema.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


SYSTEM_LABEL = "hipporag_official_hf_local"
CONFIG_ID = "hipporag_official_smoke_evidence_conf_v1"


SMOKE_DOCS = [
    "Greyia is a plant genus containing three species.",
    "Calibanus is a plant genus containing one known species.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Biofeedback training teaches people to control physiological responses.",
    "A small clinical note reports that biofeedback training reduced migraine frequency in a pilot cohort.",
]


SMOKE_QUESTIONS = [
    {
        "question_id": "smoke_001",
        "question": "Which genus has more species, Greyia or Calibanus?",
        "gold_answers": ["Greyia"],
    },
    {
        "question_id": "smoke_002",
        "question": "What object did the prince use to search the kingdom?",
        "gold_answers": ["the lost glass slipper", "glass slipper"],
    },
    {
        "question_id": "smoke_003",
        "question": "What intervention reduced migraine frequency in the pilot cohort?",
        "gold_answers": ["biofeedback training"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--official-root", default="external/official_hipporag")
    parser.add_argument("--run-id", default=time.strftime("smoke_%Y%m%d_%H%M%S"))
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--manifest-json", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--llm-name", default=os.environ.get("HIPPORAG_LLM_NAME", "gpt-4o-mini"))
    parser.add_argument("--llm-base-url", default=os.environ.get("HIPPORAG_LLM_BASE_URL"))
    parser.add_argument("--embedding-name", default=os.environ.get("HIPPORAG_EMBEDDING_NAME", "text-embedding-3-small"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("HIPPORAG_EMBEDDING_BASE_URL"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-answer-tokens", type=int, default=256)
    return parser.parse_args()


def add_official_repo_to_path(project_dir: Path, official_root: str) -> Path:
    official_path = Path(official_root)
    if not official_path.is_absolute():
        official_path = project_dir / official_path
    src_path = official_path / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    return official_path


def read_prompt(project_dir: Path) -> str:
    return (project_dir / "prompts" / "evidence_confidence_qa.txt").read_text(encoding="utf-8")


def build_answer_messages(question: str, retrieved_docs: list[dict[str, Any]], prompt_template: str) -> list[dict[str, str]]:
    context_lines = []
    for doc in retrieved_docs:
        context_lines.append(f"[{doc['rank']}] {doc['text']}")
    user = (
        f"{prompt_template}\n\n"
        f"Retrieved context:\n" + "\n\n".join(context_lines) + "\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": "You are a careful QA system that only answers from retrieved context."},
        {"role": "user", "content": user},
    ]


def parse_generation(text: str) -> dict[str, Any]:
    final_answer = None
    evidence = None
    confidence = None
    errors = []

    answer_matches = list(re.finditer(
        r"Final answer\s*:\s*(.*?)(?:\n\s*Supporting evidence\s*:|\Z)",
        text,
        flags=re.I | re.S,
    ))
    answer_match = answer_matches[-1] if answer_matches else None
    if answer_match:
        final_answer = answer_match.group(1).strip().strip('"')
    else:
        errors.append("missing_final_answer")

    evidence_matches = list(re.finditer(
        r"Supporting evidence\s*:\s*(.*?)(?:\n\s*Confidence\s*:|\Z)",
        text,
        flags=re.I | re.S,
    ))
    evidence_match = evidence_matches[-1] if evidence_matches else None
    if evidence_match:
        evidence = evidence_match.group(1).strip().strip('"')
    else:
        errors.append("missing_supporting_evidence")

    confidence_matches = list(re.finditer(r"Confidence\s*:\s*([01](?:\.\d+)?)", text, flags=re.I))
    confidence_match = confidence_matches[-1] if confidence_matches else None
    if confidence_match:
        confidence = float(confidence_match.group(1))
        if not 0.0 <= confidence <= 1.0:
            errors.append("confidence_out_of_range")
            confidence = None
    else:
        errors.append("missing_confidence")

    return {
        "predicted_answer": final_answer,
        "supporting_evidence": evidence,
        "self_reported_confidence": confidence,
        "parse_errors": errors,
    }


def exact_evidence_found(evidence: str | None, passages: list[dict[str, Any]]) -> bool:
    if not evidence:
        return False
    normalized_evidence = " ".join(evidence.lower().split())
    for passage in passages:
        normalized_passage = " ".join(str(passage["text"]).lower().split())
        if normalized_evidence in normalized_passage:
            return True
    return False


def main() -> None:
    args = parse_args()
    args.llm_base_url = args.llm_base_url or None
    args.embedding_base_url = args.embedding_base_url or None
    project_dir = Path(args.project_dir).resolve()
    official_path = add_official_repo_to_path(project_dir, args.official_root)

    from hipporag import HippoRAG as HippoRAGImport
    from hipporag.utils.config_utils import BaseConfig

    HippoRAGClass = getattr(HippoRAGImport, "HippoRAG", HippoRAGImport)

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else project_dir / "results" / "smoke" / f"{args.run_id}.jsonl"
    manifest_json = Path(args.manifest_json) if args.manifest_json else project_dir / "manifests" / f"{args.run_id}.manifest.json"
    save_dir = Path(args.save_dir) if args.save_dir else project_dir / "results" / "smoke" / f"{args.run_id}_hipporag_workdir"
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
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
    )

    prompt_template = read_prompt(project_dir)
    started_at = time.time()

    hipporag = HippoRAGClass(global_config=config)
    hipporag.index(docs=SMOKE_DOCS)

    rows = []
    queries = [q["question"] for q in SMOKE_QUESTIONS]
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=args.top_k)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for question_row, retrieval in zip(SMOKE_QUESTIONS, retrieval_results):
            retrieved_passages = [
                {
                    "rank": rank,
                    "text": doc,
                    "score": float(retrieval.doc_scores[rank - 1]) if getattr(retrieval, "doc_scores", None) is not None else None,
                }
                for rank, doc in enumerate(retrieval.docs[: args.top_k], start=1)
            ]
            messages = build_answer_messages(question_row["question"], retrieved_passages, prompt_template)
            raw_generation, metadata, cache_hit = hipporag.llm_model.infer(
                messages,
                max_completion_tokens=args.max_answer_tokens,
            )
            parsed = parse_generation(raw_generation)
            row = {
                "question_id": question_row["question_id"],
                "benchmark": "hipporag_smoke",
                "system": SYSTEM_LABEL,
                "config_id": CONFIG_ID,
                "run_id": args.run_id,
                "model": args.llm_name,
                "embedding_model": args.embedding_name,
                "question": question_row["question"],
                "gold_answers": question_row["gold_answers"],
                "predicted_answer": parsed["predicted_answer"],
                "raw_generation": raw_generation,
                "supporting_evidence": parsed["supporting_evidence"],
                "self_reported_confidence": parsed["self_reported_confidence"],
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
            print(f"{row['question_id']} pred={row['predicted_answer']!r} conf={row['self_reported_confidence']}")

    manifest = {
        "run_id": args.run_id,
        "system": SYSTEM_LABEL,
        "config_id": CONFIG_ID,
        "official_root": str(official_path),
        "output_jsonl": str(output_jsonl),
        "save_dir": str(save_dir),
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "embedding_name": args.embedding_name,
        "embedding_base_url": args.embedding_base_url,
        "num_docs": len(SMOKE_DOCS),
        "num_questions": len(SMOKE_QUESTIONS),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "rows": len(rows),
        "parse_error_count": sum(1 for row in rows if row["parse_errors"]),
        "evidence_found_count": sum(1 for row in rows if row["supporting_evidence_found_in_retrieved"]),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
