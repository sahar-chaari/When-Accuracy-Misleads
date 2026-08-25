#!/usr/bin/env python3
"""
Corrective RAG (CRAG) pipeline for biomedical QA.

Paper: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
arXiv: 2401.15884

Adaptations from the original paper (all documented in manifests/crag_architecture.json):
  1. Retrieval evaluator: Qwen (zero-shot LLM judge) instead of fine-tuned T5-large.
  2. Web-search fallback: second BM25/ES pass over the same StatPearls+textbooks corpus
     with keyword-extracted queries. Web search is fully disabled.
  3. Knowledge refinement: passage-level filtering (not sentence-level decomposition).
  4. Reader LLM: Qwen2.5-{7,32}B-Instruct (local HF). No OpenAI/GPT used anywhere.

Output schema: canonical per-row JSONL identical to IRCoT/HippoRAG so all existing
scorers (phase1_score_outputs, phase1_risk_coverage, phase1_score_faithfulness_nli,
phase1_scaffold_harm_cost) consume the output unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# ES retrieval
# ──────────────────────────────────────────────────────────────────────────────

def make_es_client(host: str, port: int):
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        sys.exit("pip install elasticsearch==7.10.1")
    es = Elasticsearch(f"http://{host}:{port}", timeout=30)
    if not es.ping():
        sys.exit(f"Cannot reach Elasticsearch at {host}:{port}")
    return es


def bm25_retrieve(es, index: str, query: str, top_k: int) -> list[dict]:
    try:
        resp = es.search(
            index=index,
            body={"query": {"match": {"paragraph_text": {"query": query, "operator": "or"}}},
                  "size": top_k},
        )
    except Exception as exc:
        print(f"[WARN] ES retrieve failed: {exc}", file=sys.stderr)
        return []
    passages = []
    for rank, hit in enumerate(resp["hits"]["hits"], start=1):
        src = hit.get("_source", {})
        text = src.get("paragraph_text") or src.get("contents") or src.get("text") or ""
        title = src.get("title", "")
        pid = src.get("id") or src.get("passage_id") or hit.get("_id") or f"doc_{rank}"
        passages.append({
            "rank": rank,
            "passage_id": pid,
            "title": title,
            "text": text,
            "score": hit.get("_score", 0.0),
            "query": query,
        })
    return passages


def keyword_query(question: str) -> str:
    """Cheap keyword extraction: strip stop words and punctuation."""
    STOPWORDS = {
        "a","an","the","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "shall","can","need","dare","ought","used","to","of","in","on","at",
        "by","for","with","about","against","between","into","through","during",
        "before","after","above","below","from","up","down","out","off","over",
        "under","then","once","and","but","or","nor","not","so","yet","both",
        "either","neither","whether","if","though","although","because","since",
        "while","as","than","that","which","who","whom","whose","what","where",
        "when","why","how","all","any","each","every","no","some","such","only",
        "same","than","too","very","just","because","it","its","this","these",
        "those","they","them","their","we","us","our","you","your","he","she",
        "him","her","his","i","me","my","also","does","did","could","would",
    }
    tokens = question.lower().translate(str.maketrans("", "", string.punctuation)).split()
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(keywords) if keywords else question


# ──────────────────────────────────────────────────────────────────────────────
# Model loading + generation
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, torch_dtype: str = "auto", device_map: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[CRAG] Loading model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype if torch_dtype != "auto" else "auto",
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[CRAG] Model loaded.", flush=True)
    return tokenizer, model


def generate(tokenizer, model, messages: list[dict], max_new_tokens: int = 256,
             temperature: float = 0.0) -> str:
    import torch
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# CRAG evaluator
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_passage_relevance(tokenizer, model, question: str, passage: dict,
                                prompt_template: str) -> str:
    """
    Ask Qwen whether a passage is Correct / Incorrect / Ambiguous for the question.
    Returns one of: 'correct', 'incorrect', 'ambiguous'.
    """
    text = passage.get("text", "")[:800]  # truncate to keep prompt short
    title = passage.get("title", "")
    user_msg = prompt_template.format(
        question=question,
        title=title,
        text=text,
    )
    messages = [{"role": "user", "content": user_msg}]
    response = generate(tokenizer, model, messages, max_new_tokens=8, temperature=0.0)
    response_lower = response.lower().strip()
    if "incorrect" in response_lower:
        return "incorrect"
    if "ambiguous" in response_lower:
        return "ambiguous"
    if "correct" in response_lower:
        return "correct"
    # Default to ambiguous if parsing fails
    return "ambiguous"


def decide_action(labels: list[str]) -> str:
    """
    CRAG decision rule over passage-level evaluator labels.
      - Any 'correct'   → action = 'correct'
      - All 'incorrect' → action = 'incorrect'
      - Otherwise       → action = 'ambiguous'
    """
    if "correct" in labels:
        return "correct"
    if all(l == "incorrect" for l in labels):
        return "incorrect"
    return "ambiguous"


# ──────────────────────────────────────────────────────────────────────────────
# Output parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_pubmedqa_label(text: str) -> str:
    t = (text or "").lower()
    for label in ("yes", "no", "maybe"):
        if label in t:
            return label
    return "unknown"


def parse_mcq_label(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", text or "")
    return match.group(1) if match else "unknown"


def parse_confidence(text: str) -> float | None:
    match = re.search(
        r"confidence\s*(?:score)?\s*[:=]\s*([01](?:\.\d+)?|\d{1,2}(?:\.\d+)?)\s*%?",
        text or "", flags=re.IGNORECASE
    )
    if match:
        try:
            val = float(match.group(1))
            return val / 100.0 if val > 1.0 else val
        except ValueError:
            pass
    return None


def parse_supporting_evidence(text: str) -> str:
    match = re.search(
        r"supporting evidence\s*[:=]\s*(.+?)(?:\nconfidence\s*[:=]|\Z)",
        text or "", flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""


def verify_evidence_in_passages(evidence: str, passages: list[dict]) -> tuple[list[str], list[str]]:
    """Check if the cited evidence quote appears in any retrieved passage."""
    if not evidence:
        return [], []
    ev_norm = " ".join(evidence.lower().split())
    found_ids, found_titles = [], []
    for p in passages:
        p_norm = " ".join(p.get("text", "").lower().split())
        # substring match after collapsing whitespace
        if len(ev_norm) > 20 and ev_norm[:40] in p_norm:
            found_ids.append(p["passage_id"])
            found_titles.append(p.get("title", ""))
    return found_ids, found_titles


# ──────────────────────────────────────────────────────────────────────────────
# CRAG per-question pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_crag_question(
    row: dict,
    es,
    tokenizer,
    model,
    config: dict,
    evaluator_prompt: str,
    reader_prompt_template: str,
    benchmark: str,
) -> dict:
    t0 = time.time()
    question = row.get("question", "")
    qid = str(row.get("question_id", row.get("id", "")))
    gold_short = str(row.get("gold_short_answer") or "")
    gold_answers = [str(a) for a in (row.get("gold_answers") or row.get("answers") or [])]
    if not gold_answers and gold_short:
        gold_answers = [gold_short]
    gold_evidence_ids = list(row.get("gold_evidence_ids") or [])
    # PubMedQA: derive gold passage IDs from numbered_context (format "[P1] PMID:xxx ...")
    if not gold_evidence_ids and benchmark == "pubmedqa":
        for ctx in (row.get("numbered_context") or []):
            m = re.match(r'\[P(\d+)\]\s+PMID:(\d+)', ctx)
            if m:
                gold_evidence_ids.append(f"{m.group(2)}::P{m.group(1)}")
    split = row.get("split", "")

    index = config["es_index"]
    top_k = config.get("top_k", 5)
    top_k_second = config.get("top_k_second_pass", 5)

    errors: list[str] = []

    # ── Step 1: Initial BM25 retrieval ──────────────────────────────────────
    initial_passages = bm25_retrieve(es, index, question, top_k)

    # ── Step 2: Evaluate each passage ───────────────────────────────────────
    eval_labels: list[str] = []
    for p in initial_passages:
        try:
            label = evaluate_passage_relevance(tokenizer, model, question, p, evaluator_prompt)
        except Exception as exc:
            label = "ambiguous"
            errors.append(f"eval_error: {exc}")
        eval_labels.append(label)

    for p, lbl in zip(initial_passages, eval_labels):
        p["crag_eval_label"] = lbl

    # ── Step 3: Corrective action ────────────────────────────────────────────
    action = decide_action(eval_labels)
    second_pass_used = False
    second_pass_query = ""

    if action == "correct":
        # Keep only passages judged correct
        context_passages = [p for p, l in zip(initial_passages, eval_labels) if l == "correct"]
        if not context_passages:
            context_passages = initial_passages  # fallback: keep all

    elif action == "incorrect":
        # Second ES pass with keyword query (replaces web search)
        second_pass_query = keyword_query(question)
        second_passages = bm25_retrieve(es, index, second_pass_query, top_k_second)
        for p in second_passages:
            p["crag_eval_label"] = "second_pass"
            p["retrieval_pass"] = 2
        context_passages = second_passages
        second_pass_used = True

    else:  # ambiguous
        # Keep correct passages + second pass
        correct_passages = [p for p, l in zip(initial_passages, eval_labels) if l == "correct"]
        second_pass_query = keyword_query(question)
        second_passages = bm25_retrieve(es, index, second_pass_query, top_k_second)
        for p in second_passages:
            p["crag_eval_label"] = "second_pass"
            p["retrieval_pass"] = 2
        # Deduplicate by passage_id
        seen = {p["passage_id"] for p in correct_passages}
        new_passages = [p for p in second_passages if p["passage_id"] not in seen]
        context_passages = correct_passages + new_passages
        second_pass_used = True

    if not context_passages:
        context_passages = initial_passages  # last-resort fallback

    # ── Step 4: Format context for reader ────────────────────────────────────
    context_str = ""
    for i, p in enumerate(context_passages[:8], start=1):
        title = p.get("title", "")
        text = p.get("text", "")[:600]
        context_str += f"[{i}] {title}\n{text}\n\n"

    if benchmark == "medmcqa":
        choices = row.get("options") or row.get("choices") or {}
        if isinstance(choices, dict):
            choices_str = "\n".join(f"  {k}: {v}" for k, v in sorted(choices.items()))
        else:
            choices_str = str(choices)
        user_msg = reader_prompt_template.format(
            question=question, context=context_str.strip(), choices=choices_str
        )
    else:
        user_msg = reader_prompt_template.format(
            question=question, context=context_str.strip()
        )

    # ── Step 5: Generate answer ───────────────────────────────────────────────
    raw_generation = ""
    try:
        messages = [{"role": "user", "content": user_msg}]
        raw_generation = generate(
            tokenizer, model, messages,
            max_new_tokens=config.get("max_answer_tokens", 256),
            temperature=0.0,
        )
    except Exception as exc:
        errors.append(f"generation_error: {exc}")
        raw_generation = ""

    # ── Step 6: Parse structured output ──────────────────────────────────────
    if benchmark == "medmcqa":
        predicted_answer = parse_mcq_label(raw_generation)
    else:
        predicted_answer = parse_pubmedqa_label(raw_generation)

    confidence = parse_confidence(raw_generation)
    supporting_evidence = parse_supporting_evidence(raw_generation)

    all_passages = initial_passages + [
        p for p in context_passages if p.get("retrieval_pass") == 2
    ]
    ev_ids, ev_titles = verify_evidence_in_passages(supporting_evidence, all_passages)

    # Retrieval recall against gold evidence IDs (PubMedQA only)
    retrieved_ids = {p["passage_id"] for p in initial_passages}
    if gold_evidence_ids:
        hits = sum(1 for gid in gold_evidence_ids if gid in retrieved_ids)
        retrieval_recall = hits / len(gold_evidence_ids)
    else:
        retrieval_recall = None

    elapsed = time.time() - t0

    return {
        "question_id": qid,
        "benchmark": benchmark,
        "split": split,
        "system": "crag",
        "config_id": config.get("config_id", "crag"),
        "model": config.get("model_name", ""),
        "question": question,
        "gold_answers": gold_answers,
        "gold_short_answer": gold_short,
        "gold_evidence_ids": gold_evidence_ids,
        # ── canonical scorer fields ──────────────────────────────────────────
        "predicted_answer": predicted_answer,
        "confidence": confidence,
        "supporting_evidence": supporting_evidence,
        "supporting_evidence_verified_passage_ids": ev_ids,
        "supporting_evidence_verified_passage_titles": ev_titles,
        "retrieved_passages": all_passages,
        "raw_generation": raw_generation,
        # ── CRAG-specific metadata ───────────────────────────────────────────
        "crag_eval_labels": eval_labels,
        "crag_action": action,
        "crag_second_pass_used": second_pass_used,
        "crag_second_pass_query": second_pass_query,
        # ── provenance ───────────────────────────────────────────────────────
        "retrieval_recall": retrieval_recall,
        "timing": {"total_seconds": round(elapsed, 3)},
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Health-check summary
# ──────────────────────────────────────────────────────────────────────────────

def health_check(rows: list[dict], benchmark: str) -> dict:
    n = len(rows)
    zero_retrieval = sum(1 for r in rows if not r.get("retrieved_passages"))
    with_evidence = sum(1 for r in rows if r.get("supporting_evidence"))
    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    predictions = [r.get("predicted_answer", "") for r in rows]

    from collections import Counter
    pred_dist = dict(Counter(predictions))
    recalls = [r["retrieval_recall"] for r in rows if r.get("retrieval_recall") is not None]
    action_dist = dict(Counter(r.get("crag_action", "") for r in rows))

    hc = {
        "n_total": n,
        "zero_retrieval_rows": zero_retrieval,
        "evidence_present_pct": round(100 * with_evidence / n, 1) if n else 0,
        "confidence_coverage_pct": round(100 * len(confidences) / n, 1) if n else 0,
        "confidence_min": round(min(confidences), 3) if confidences else None,
        "confidence_max": round(max(confidences), 3) if confidences else None,
        "confidence_avg": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "prediction_distribution": pred_dist,
        "crag_action_distribution": action_dist,
    }
    if recalls:
        hc["avg_retrieval_recall"] = round(sum(recalls) / len(recalls), 4)
        hc["retrieval_recall_note"] = (
            "PubMedQA gold abstracts are not in the MedRAG corpus; "
            "low recall expected — treat this arm as a calibration probe."
            if benchmark == "pubmedqa" else ""
        )
    collapse_threshold = 0.80
    for label, count in pred_dist.items():
        if count / n >= collapse_threshold:
            hc["WARNING_label_collapse"] = (
                f"Label '{label}' accounts for {count}/{n} = "
                f"{count/n:.0%} of predictions — possible collapse."
            )
    return hc


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["pubmedqa", "medmcqa"], required=True)
    parser.add_argument("--es-host", default="localhost")
    parser.add_argument("--es-port", type=int, default=9200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Run only 3 questions for a smoke test")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    evaluator_prompt = Path(config["evaluator_prompt_file"]).read_text(encoding="utf-8").strip()
    reader_prompt_template = Path(config["reader_prompt_file"]).read_text(encoding="utf-8").strip()

    es = make_es_client(args.es_host, args.es_port)
    tokenizer, model = load_model(config["model_name"])

    rows = load_jsonl(args.input_jsonl)
    if args.smoke:
        rows = rows[:3]
    elif args.limit is not None:
        rows = rows[:args.limit]

    print(f"[CRAG] Running on {len(rows)} questions, benchmark={args.benchmark}", flush=True)

    results = []
    for i, row in enumerate(rows):
        print(f"[CRAG] Q{i+1}/{len(rows)} id={row.get('question_id', row.get('id','?'))}", flush=True)
        try:
            result = run_crag_question(
                row, es, tokenizer, model, config,
                evaluator_prompt, reader_prompt_template, args.benchmark
            )
        except Exception as exc:
            result = {
                "question_id": str(row.get("question_id", "")),
                "benchmark": args.benchmark,
                "split": row.get("split", ""),
                "system": "crag",
                "config_id": config.get("config_id", "crag"),
                "model": config.get("model_name", ""),
                "question": row.get("question", ""),
                "gold_answers": row.get("gold_answers", []),
                "gold_short_answer": row.get("gold_short_answer", ""),
                "gold_evidence_ids": row.get("gold_evidence_ids", []),
                "predicted_answer": "unknown",
                "confidence": None,
                "supporting_evidence": "",
                "supporting_evidence_verified_passage_ids": [],
                "retrieved_passages": [],
                "raw_generation": "",
                "crag_eval_labels": [],
                "crag_action": "error",
                "crag_second_pass_used": False,
                "crag_second_pass_query": "",
                "retrieval_recall": None,
                "timing": {"total_seconds": 0},
                "errors": [f"pipeline_error: {exc}"],
            }
        results.append(result)

    write_jsonl(args.output_jsonl, results)
    print(f"[CRAG] Wrote {len(results)} rows → {args.output_jsonl}")

    hc = health_check(results, args.benchmark)
    hc_path = args.output_jsonl.parent / (args.output_jsonl.stem + "_health.json")
    write_json(hc_path, hc)
    print(json.dumps(hc, indent=2))


if __name__ == "__main__":
    main()
