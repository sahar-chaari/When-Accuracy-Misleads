#!/usr/bin/env python3
"""Build a large PubMedQA retrieval corpus from all 1,000 PQA-L questions.

Reads ori_pqal.json and extracts every context paragraph from every question,
producing a JSONL corpus in the same format as pubmedqa_150_corpus.jsonl but
covering all 1,000 PMIDs (~3,358 passages) instead of only the 60 evaluation
questions' PMIDs (517 passages).

The 60 evaluation questions are unchanged — only the retrieval corpus grows.
Gold passages for each question still exist in the index but now compete
against ~3,300 distractor passages from other questions' PMIDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_corpus(ori_pqal: dict) -> list[dict]:
    rows = []
    seen = set()
    for pmid, entry in ori_pqal.items():
        contexts = entry.get("CONTEXTS", [])
        labels = entry.get("LABELS", [])
        for i, text in enumerate(contexts):
            passage_id = f"{pmid}::P{i + 1}"
            if passage_id in seen:
                continue
            seen.add(passage_id)
            local_pid = f"P{i + 1}"
            section = labels[i] if i < len(labels) else local_pid
            rows.append({
                "passage_id": passage_id,
                "pmid": pmid,
                "local_pid": local_pid,
                "section": section,
                "text": text.strip(),
                "doc_idx": len(rows),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori-pqal", type=Path,
                        default=Path("data/pubmedqa/data/ori_pqal.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/pubmedqa_1000_corpus.jsonl"))
    args = parser.parse_args()

    ori_pqal = load_json(args.ori_pqal)
    print(f"Loaded {len(ori_pqal)} questions from {args.ori_pqal}")

    rows = build_corpus(ori_pqal)
    write_jsonl(args.output, rows)

    unique_pmids = len({r["pmid"] for r in rows})
    print(f"Wrote {len(rows)} passages from {unique_pmids} unique PMIDs → {args.output}")


if __name__ == "__main__":
    main()
