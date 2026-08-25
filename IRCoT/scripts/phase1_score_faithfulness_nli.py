#!/usr/bin/env python3
"""Score cited-evidence faithfulness with an NLI model.

The scorer implements the paper's cited-only evidence policy. It first detects
explicit no-evidence admissions in the model's supporting-evidence field, then
checks whether the cited text appears in retrieved passages using exact and
fuzzy span matching. Present evidence is scored with NLI into supported,
contradicted, or unsupported. Empty answers are labelled abstained.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    return text or "" if len(words) <= max_words else " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# Hypothesis construction
# ---------------------------------------------------------------------------

def _parse_medmcqa_options(row: dict) -> dict[str, str]:
    opts = row.get("options")
    if opts and isinstance(opts, dict) and len(opts) >= 2:
        return {k: str(v).strip() for k, v in opts.items()}
    question = row.get("question", "")
    parts = re.split(r"\n([A-D])[)\.]\s*", question)
    result = {}
    for i in range(1, len(parts) - 1, 2):
        letter = parts[i].strip()
        text = parts[i + 1].strip()
        text = re.split(r"\n[A-D][)\.]", text)[0].strip()
        result[letter] = text
    return result


def _medmcqa_stem(row: dict) -> str:
    question = row.get("question", "")
    return re.split(r"\n[A-D][)\.]\s*", question)[0].strip()


_AUX_RE = re.compile(
    r"^(Can|Could|Does|Do|Is|Are|Has|Have|Will|Would|Should|Were|Was|May|Might|Did)\s+",
    re.IGNORECASE,
)


def _pubmedqa_declarative(question: str, answer: str) -> str:
    q = question.strip().rstrip("?").strip()
    m_there = re.match(r"^Is\s+there\s+(.+)$", q, re.IGNORECASE)
    if m_there:
        rest = m_there.group(1).strip()
        if answer == "yes":
            return f"There is {rest}."
        elif answer == "no":
            return f"There is no {rest}."
        else:
            return f"The evidence is inconclusive about whether there is {rest}."
    bare = _AUX_RE.sub("", q).strip()
    lower_bare = bare.lower()
    if answer == "yes":
        return f"Research confirms: {lower_bare}."
    elif answer == "no":
        return f"Research does not support: {lower_bare}."
    else:
        return f"The evidence is inconclusive about whether {lower_bare}."


def make_hypothesis(row: dict) -> str:
    benchmark = row.get("benchmark", "")
    predicted = (row.get("predicted_answer") or "").strip()
    if not predicted:
        return ""
    if benchmark == "medmcqa":
        options = _parse_medmcqa_options(row)
        option_text = options.get(predicted, "").strip() or predicted
        stem = _medmcqa_stem(row)
        return f"{stem}: {option_text}."
    question = (row.get("question") or "").strip().rstrip("?").strip()
    return _pubmedqa_declarative(question, predicted)


# ---------------------------------------------------------------------------
# Fix 2 — Admission detection (anywhere in supporting_evidence)
# ---------------------------------------------------------------------------

# Detects honest "I have no grounding" admissions anywhere in the SE text.
# Deliberately does NOT match "does not directly address" (meta-relevance comment)
# so that rows with a real quote + Note: about relevance still route to NLI.
_ADMISSION_RE = re.compile(
    r"("
    # "not found/provided/mentioned/..." anywhere in text
    r"\bnot\s+(?:directly\s+)?(?:found|provided|mentioned|given|stated|quoted|discussed|available)\b"
    # "evidence/context/information/passages DOES NOT mention/contain/discuss/include"
    # Requires a context-related subject to avoid matching medical statements like
    # "Etching of dentin does not include…"
    r"|\b(?:evidence|context|information|passages?)\s+(?:\w+\s+)?(?:does|do|did|is|are|was|were)\s+not\s+(?:directly\s+)?(?:mention|contain|discuss|include)\b"
    # "none of the options/appliances/provided …"
    r"|\bnone\s+of\s+the\s+(?:provided|given|options|appliances|alternatives)\b"
    # Model explicitly says it is using knowledge outside the retrieved passages
    r"|\bbased\s+on\s+(?:general|external|common)\s+(?:medical\s+|clinical\s+)?knowledge\b"
    # Bracketed disclaimers (IRCoT / HippoRAG style)
    r"|\[not\s+directly\s+quoted"
    r"|\bimplied\s+from\s+the\s+context\b"
    r")",
    re.IGNORECASE,
)


def is_admission(se: str) -> bool:
    """Return True if the SE admits no grounding anywhere in its text."""
    return bool(_ADMISSION_RE.search(se or ""))


# ---------------------------------------------------------------------------
# Fix 1 — Improved presence matcher with span extraction
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


# Sentence-initial phrases that mark framing/meta-commentary (not real quotes).
_FRAMING_SENT_RE = re.compile(
    r"^(?:Note\s*[:\.]|However[,\.]|Therefore[,\.]|Based on|Since the|Although|"
    r"Given the|Thus[,\.]|Hence[,\.]|The question asks|The answer is|"
    r"The correct answer|The most (?:plausible|appropriate|likely)|"
    r"Confidence\s*[:\.])",
    re.IGNORECASE,
)

# Boundaries that introduce meta-commentary mid-text.
_META_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:Note\s*[:\.]|However[,\.]|Therefore[,\.]|Based on|Since the|"
    r"Given the|Thus[,\.]|Hence[,\.]|The question asks|The answer is|"
    r"The correct answer|Confidence\s*[:\.]))",
    re.IGNORECASE,
)


def _candidate_spans(evidence: str) -> list[str]:
    """
    Return a list of text spans to try matching against retrieved passages.

    Order: full text first, then progressively smaller / cleaner spans so
    that the first successful match wins (caller stops at first hit).
    """
    spans: list[str] = [evidence]

    # 1. Everything before the first meta-commentary boundary (e.g., "Note:").
    pre_meta = _META_BOUNDARY_RE.split(evidence, maxsplit=1)[0].strip()
    if pre_meta and pre_meta != evidence and len(pre_meta.split()) >= 8:
        spans.append(pre_meta)

    # 2. Verbatim quoted spans ≥ 8 words (straight and curly quotes).
    for m in re.finditer(
        r'["“”‘’](.*?)["”“’‘]',
        evidence,
        re.DOTALL,
    ):
        span = m.group(1).strip()
        if len(span.split()) >= 8:
            spans.append(span)

    # 3. Individual substantive sentences ≥ 10 words (excluding framing ones).
    raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(\"])", evidence)
    substantive = [
        s.strip()
        for s in raw_sents
        if len(s.strip().split()) >= 10 and not _FRAMING_SENT_RE.match(s.strip())
    ]
    for s in substantive:
        if s not in spans:
            spans.append(s)

    # 4. Consecutive pairs of substantive sentences.
    for i in range(len(substantive) - 1):
        pair = substantive[i] + " " + substantive[i + 1]
        if pair not in spans:
            spans.append(pair)

    return spans


def check_evidence_presence(
    evidence: str,
    retrieved_passages: list[dict],
    fuzzy_threshold: int = 90,
) -> dict:
    """
    Check whether any candidate span derived from `evidence` is present
    in the text of `retrieved_passages`.

    Tries each span in order: exact substring, then rapidfuzz partial_ratio.
    Returns on the first hit.
    """
    if not evidence or not evidence.strip():
        return {"evidence_present": False, "evidence_present_method": "none"}

    passage_texts = []
    for p in retrieved_passages:
        t = p.get("text") or p.get("hipporag_doc_text") or ""
        passage_texts.append(_normalize(t))
    haystack = " ".join(passage_texts)

    try:
        from rapidfuzz import fuzz as _fuzz
        _fuzzy_available = True
    except ImportError:
        _fuzzy_available = False

    for span in _candidate_spans(evidence):
        needle = _normalize(span)
        if not needle or len(needle.split()) < 4:
            continue

        if needle in haystack:
            return {"evidence_present": True, "evidence_present_method": "substring"}

        # Ellipsis fragment check
        if "..." in span or "…" in span:
            fragments = re.split(r"\.{2,}|…", needle)
            non_trivial = [f.strip() for f in fragments if len(f.strip().split()) >= 5]
            if non_trivial and all(f in haystack for f in non_trivial):
                return {"evidence_present": True, "evidence_present_method": "fragment"}

        if _fuzzy_available:
            score = _fuzz.partial_ratio(needle, haystack)
            if score >= fuzzy_threshold:
                return {"evidence_present": True, "evidence_present_method": "fuzzy"}

    return {"evidence_present": False, "evidence_present_method": "none"}


# ---------------------------------------------------------------------------
# NLI model helpers
# ---------------------------------------------------------------------------

def label_index(id2label: dict, needle: str) -> int | None:
    for idx, label in id2label.items():
        if needle in str(label).lower():
            return int(idx)
    return None


def load_nli_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def nli_scores(tokenizer, model, device: str, premise: str, hypothesis: str) -> dict:
    import torch

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    id2label = model.config.id2label
    entail_idx = label_index(id2label, "entail")
    contra_idx = label_index(id2label, "contrad")
    neutral_idx = label_index(id2label, "neutral")
    return {
        "entailment": probs[entail_idx] if entail_idx is not None else None,
        "contradiction": probs[contra_idx] if contra_idx is not None else None,
        "neutral": probs[neutral_idx] if neutral_idx is not None else None,
    }


def cited_evidence_passage(row: dict, max_words: int) -> dict | None:
    evidence = (row.get("supporting_evidence") or "").strip()
    if not evidence:
        return None
    return {
        "rank": 0,
        "passage_id": (row.get("supporting_evidence_verified_passage_ids") or [None])[0] or "cited",
        "title": "",
        "text": truncate_words(evidence, max_words),
    }


def evidence_passages(row: dict, max_passages: int, max_words: int) -> list[dict]:
    passages = row.get("retrieved_passages") or row.get("evidence") or []
    return [
        {
            "rank": p.get("rank", ""),
            "passage_id": p.get("passage_id", ""),
            "title": p.get("title", ""),
            "text": truncate_words(p.get("text", ""), max_words),
        }
        for p in passages[:max_passages]
    ]


# ---------------------------------------------------------------------------
# Fix 3 — Core scoring with new routing
# ---------------------------------------------------------------------------

def score_row(row: dict, tokenizer, model, device: str, config: dict) -> dict:
    hypothesis_text = make_hypothesis(row)
    policy = config.get("evidence_policy", "top_retrieved")

    base = {
        "question_id": row.get("question_id", ""),
        "benchmark": row.get("benchmark", ""),
        "system": row.get("system", ""),
        "config_id": row.get("config_id", ""),
        "split": row.get("split", ""),
        "predicted_answer": row.get("predicted_answer", ""),
        "gold_short_answer": row.get("gold_short_answer", ""),
        "evidence_policy": policy,
        "judge_model": config["model_name"],
        "hypothesis_text": hypothesis_text,
    }

    # Gate 0: empty predicted answer → abstained
    if not hypothesis_text:
        return {**base,
                "faithfulness_label": "abstained",
                "evidence_present": None,
                "evidence_present_method": "none",
                "max_entailment": None,
                "max_contradiction": None,
                "best_evidence_id": "",
                "passage_scores": [],
                "rationale": "Empty predicted answer."}

    supporting_evidence = (row.get("supporting_evidence") or "").strip()
    retrieved = row.get("retrieved_passages") or []

    # Gate 1: no supporting evidence cited → abstained
    if not supporting_evidence:
        return {**base,
                "faithfulness_label": "abstained",
                "evidence_present": False,
                "evidence_present_method": "none",
                "max_entailment": None,
                "max_contradiction": None,
                "best_evidence_id": "",
                "passage_scores": [],
                "rationale": "No supporting evidence cited by the model."}

    # Always compute presence (needed for the output field even if routed elsewhere).
    presence = check_evidence_presence(
        supporting_evidence,
        retrieved,
        fuzzy_threshold=config.get("fuzzy_threshold", 90),
    )

    # Fix 2: admission check — routes before NLI.
    if is_admission(supporting_evidence):
        return {**base,
                "faithfulness_label": "no_evidence_admitted",
                "evidence_present": presence["evidence_present"],
                "evidence_present_method": presence["evidence_present_method"],
                "max_entailment": None,
                "max_contradiction": None,
                "best_evidence_id": "",
                "passage_scores": [],
                "rationale": "Model admitted no grounding in retrieved passages."}

    # Fix 1: if evidence not present and no admission → unsupported (no NLI).
    if not presence["evidence_present"]:
        return {**base,
                "faithfulness_label": "unsupported",
                "evidence_present": False,
                "evidence_present_method": presence["evidence_present_method"],
                "max_entailment": None,
                "max_contradiction": None,
                "best_evidence_id": "",
                "passage_scores": [],
                "rationale": "Cited text not found in retrieved passages and no admission detected."}

    # NLI scoring (only reached when evidence is present and no admission).
    if policy == "cited_only":
        cited = cited_evidence_passage(row, config["max_words_per_passage"])
        if cited is None:
            return {**base,
                    "faithfulness_label": "abstained",
                    "evidence_present": True,
                    "evidence_present_method": presence["evidence_present_method"],
                    "max_entailment": None,
                    "max_contradiction": None,
                    "best_evidence_id": "",
                    "passage_scores": [],
                    "rationale": "Supporting evidence empty after truncation."}
        premise = cited["text"].strip()
        raw_scores = nli_scores(tokenizer, model, device, premise, hypothesis_text)
        passage_scores = [{**cited, **raw_scores}]
    else:
        passages = evidence_passages(row, config["max_passages"], config["max_words_per_passage"])
        passage_scores = []
        for passage in passages:
            premise = f"{passage.get('title', '')}\n{passage.get('text', '')}".strip()
            if not premise:
                continue
            raw_scores = nli_scores(tokenizer, model, device, premise, hypothesis_text)
            passage_scores.append({**passage, **raw_scores})

    if not passage_scores:
        return {**base,
                "faithfulness_label": "unsupported",
                "evidence_present": True,
                "evidence_present_method": presence["evidence_present_method"],
                "max_entailment": 0.0,
                "max_contradiction": 0.0,
                "best_evidence_id": "",
                "passage_scores": [],
                "rationale": "No usable evidence passages after policy filter."}

    max_entailment = max(
        (item["entailment"] for item in passage_scores if item["entailment"] is not None),
        default=0.0,
    )
    max_contradiction = max(
        (item["contradiction"] for item in passage_scores if item["contradiction"] is not None),
        default=0.0,
    )
    best = max(passage_scores, key=lambda item: item.get("entailment") or 0.0, default={})

    entail_thresh = config["entailment_threshold"]
    contra_thresh = config["contradiction_threshold"]

    if max_entailment >= entail_thresh and max_entailment >= max_contradiction:
        label = "supported"
    elif max_contradiction >= contra_thresh:
        label = "contradicted"
    else:
        label = "unsupported"

    return {**base,
            "faithfulness_label": label,
            "evidence_present": True,
            "evidence_present_method": presence["evidence_present_method"],
            "max_entailment": round(max_entailment, 6),
            "max_contradiction": round(max_contradiction, 6),
            "best_evidence_id": best.get("passage_id", ""),
            "best_evidence_title": best.get("title", ""),
            "passage_scores": passage_scores,
            "rationale": (
                "NLI over model-cited supporting evidence (cited_only policy)."
                if policy == "cited_only"
                else "NLI over retrieved passages."
            )}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/faithfulness_nli_default.json"))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    config = load_json(args.config)
    if args.model_name:
        config["model_name"] = args.model_name

    rows = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer, model, device = load_nli_model(config["model_name"], args.device)
    scored = [score_row(row, tokenizer, model, device, config) for row in rows]
    write_jsonl(args.output_jsonl, scored)

    counts: dict[str, int] = {}
    for r in scored:
        counts[r["faithfulness_label"]] = counts.get(r["faithfulness_label"], 0) + 1
    print(json.dumps({"rows": len(scored), "device": device, "label_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
