#!/usr/bin/env python3
"""Build fixed question-level severity ledgers for harm-cost scoring.

The ledgers are the reproducibility source of truth for Axis 4. Once generated,
the committed JSONL ledgers are sufficient for downstream harm scoring and do
not require API access. PubMedQA labels can be regenerated with the fixed
rubric through an OpenAI-compatible chat endpoint; MedMCQA regeneration uses
the optional local score-prior file when available.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "annotations" / "harm_cost"
MED_PRIOR = ROOT / "results" / "analysis" / "medmcqa_severity_score_prior.jsonl"
MED_QUESTIONS = ROOT / "data" / "fixed" / "medmcqa_60_balanced_seed42_questions.jsonl"
PUBMEDQA_QUESTIONS = ROOT / "HippoRAG" / "data" / "fixed" / "pubmedqa_60_balanced_seed42_questions.jsonl"

PROTOCOL_ID = "iec62304_question_severity_v1"
WEIGHTS = {"A": 1, "B": 4, "C": 10}
CLASS_DEFINITIONS = {
    "A": "No plausible injury or damage to health from an incorrect answer.",
    "B": "Non-serious injury, delay, unnecessary follow-up, or moderate clinical mismanagement possible.",
    "C": "Serious injury or death possible through missed emergency, dangerous treatment, contraindication, or life-critical diagnosis.",
}
MODEL = os.environ.get("SEVERITY_ANNOTATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
BASE_URL = os.environ.get("SEVERITY_ANNOTATOR_BASE_URL", "https://api.example.invalid/v1").rstrip("/")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def class_to_weight(label: str) -> int:
    label = str(label).strip().upper()
    if label not in WEIGHTS:
        raise ValueError(f"Invalid severity class: {label!r}")
    return WEIGHTS[label]


def extract_api_key() -> str:
    if os.environ.get("SEVERITY_ANNOTATOR_API_KEY"):
        return os.environ["SEVERITY_ANNOTATOR_API_KEY"].strip()
    raise RuntimeError("Set SEVERITY_ANNOTATOR_API_KEY in the environment.")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def prompt_for_pubmedqa(row: dict[str, Any]) -> list[dict[str, str]]:
    contexts = "\n".join(str(x) for x in row.get("numbered_context", [])[:4])
    content = f"""Classify the clinical consequence of answering this biomedical QA item incorrectly.

Use this IEC 62304-inspired rubric:
A = no plausible injury or damage to health from an incorrect answer. Examples: administrative, methodological, epidemiological, athletic performance, basic-science, or non-clinical recall.
B = non-serious injury, delayed care, unnecessary follow-up, or moderate clinical mismanagement possible.
C = serious injury or death possible. Examples: emergency diagnosis, life-threatening condition, dangerous treatment, drug contraindication/dosing, invasive procedure decision, or missed critical deterioration.

Score 1-10 should be consistent with the class: A=1-2, B=3-6, C=7-10.
Return exactly one JSON object with keys severity_class, severity_score_1_to_10, rationale.
The rationale must be <= 35 words.

Question: {row.get("question", "")}
Gold answer: {row.get("gold_short_answer", "")}
Gold conclusion: {row.get("gold_long_answer", "")}
Evidence snippets:
{contexts}
"""
    return [
        {"role": "system", "content": "You are a conservative clinical risk annotator. Return valid JSON only."},
        {"role": "user", "content": content},
    ]


def chat_json(messages: list[dict[str, str]], api_key: str, retries: int = 4) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 180,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
            return parse_json_object(body["choices"][0]["message"]["content"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts: {last_error}")


def normalize_pubmedqa_label(question_row: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    severity_class = str(annotation.get("severity_class", "")).strip().upper()
    if severity_class not in WEIGHTS:
        raise ValueError(f"Bad severity_class for {question_row.get('id')}: {annotation}")
    score = float(annotation.get("severity_score_1_to_10", 0))
    low, high = {"A": (1, 2), "B": (3, 6), "C": (7, 10)}[severity_class]
    if not low <= score <= high:
        score = float(low)
    return {
        "question_id": str(question_row["id"]),
        "benchmark": "pubmedqa",
        "question": question_row.get("question", ""),
        "gold_short_answer": question_row.get("gold_short_answer", ""),
        "gold_long_answer": question_row.get("gold_long_answer", ""),
        "severity_class": severity_class,
        "severity_weight": class_to_weight(severity_class),
        "severity_score_1_to_10": score,
        "rationale": str(annotation.get("rationale", "")).strip(),
        "protocol_id": PROTOCOL_ID,
        "label_source": "fixed_zero_temperature_rubric_annotator",
        "annotator_model": MODEL,
        "temperature": 0,
    }


def build_medmcqa(force: bool) -> list[dict[str, Any]]:
    out_path = OUT_DIR / "severity_labels_medmcqa_60_seed42.jsonl"
    if out_path.exists() and (not force or not MED_PRIOR.exists()):
        return load_jsonl(out_path)
    if not MED_PRIOR.exists():
        raise RuntimeError(f"Cannot regenerate MedMCQA labels; optional source file missing: {MED_PRIOR}")

    question_by_id = {str(row["question_id"]): row for row in load_jsonl(MED_QUESTIONS)}
    rows = []
    for row in load_jsonl(MED_PRIOR):
        severity_class = str(row["class_hint"]).strip().upper()
        question_row = question_by_id.get(str(row["question_id"]), {})
        rows.append(
            {
                "question_id": str(row["question_id"]),
                "benchmark": "medmcqa",
                "question": row.get("question", ""),
                "gold_short_answer": question_row.get("gold_short_answer", ""),
                "gold_answer_text": question_row.get("gold_answer_text", ""),
                "severity_class": severity_class,
                "severity_weight": class_to_weight(severity_class),
                "severity_score_1_to_10": float(row["severity_score_1_to_10"]),
                "rationale": row.get("rationale", ""),
                "protocol_id": PROTOCOL_ID,
                "label_source": "fixed_zero_temperature_rubric_annotator",
                "annotator_model": row.get("model", MODEL),
                "temperature": 0,
            }
        )
    return rows


def build_pubmedqa(max_workers: int, force: bool) -> list[dict[str, Any]]:
    out_path = OUT_DIR / "severity_labels_pubmedqa_60_seed42.jsonl"
    if out_path.exists() and not force:
        return load_jsonl(out_path)

    api_key = extract_api_key()
    question_rows = load_jsonl(PUBMEDQA_QUESTIONS)
    labels: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(chat_json, prompt_for_pubmedqa(row), api_key): row
            for row in question_rows
        }
        for future in as_completed(futures):
            row = futures[future]
            labels.append(normalize_pubmedqa_label(row, future.result()))
            print(f"labeled pubmedqa {row['id']}", flush=True)

    order = {str(row["id"]): i for i, row in enumerate(question_rows)}
    labels.sort(key=lambda item: order[item["question_id"]])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--force-medmcqa", action="store_true")
    parser.add_argument("--force-pubmedqa", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    med = build_medmcqa(args.force_medmcqa)
    pub = build_pubmedqa(args.max_workers, args.force_pubmedqa)

    write_jsonl(OUT_DIR / "severity_labels_medmcqa_60_seed42.jsonl", med)
    write_jsonl(OUT_DIR / "severity_labels_pubmedqa_60_seed42.jsonl", pub)

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "weights": WEIGHTS,
        "class_definitions": CLASS_DEFINITIONS,
        "label_source": "fixed_zero_temperature_rubric_annotator",
        "annotator_model": MODEL,
        "temperature": 0,
        "n_medmcqa": len(med),
        "n_pubmedqa": len(pub),
        "outputs": [
            "annotations/harm_cost/severity_labels_medmcqa_60_seed42.jsonl",
            "annotations/harm_cost/severity_labels_pubmedqa_60_seed42.jsonl",
        ],
    }
    (OUT_DIR / "severity_labeling_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
