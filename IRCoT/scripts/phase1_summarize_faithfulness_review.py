#!/usr/bin/env python3
"""Summarize high-priority faithfulness review cases as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_block(evidence: list[dict]) -> str:
    lines = []
    for passage in evidence:
        lines.append(
            f"- `{passage.get('passage_id', '')}` {passage.get('title', '')}: "
            f"{passage.get('text', '')}"
        )
    return "\n".join(lines) if lines else "- No retrieved evidence included."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--priority", default="high_nli_false_faithful_candidate")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = [
        row
        for row in load_jsonl(args.input_jsonl)
        if str(row.get("review_priority", "")) == args.priority
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    parts = [
        "# High-Priority Faithfulness Review Cases",
        "",
        f"Priority filter: `{args.priority}`",
        f"Cases: {len(rows)}",
        "",
        "Manual label choices: `supported`, `unsupported`, `contradicted`, `abstained`.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        parts.extend(
            [
                f"## Case {index}: {row.get('question_id', '')}",
                "",
                f"- Benchmark: `{row.get('benchmark', '')}`",
                f"- LLM-assisted label: `{row.get('llm_assisted_label', '')}`",
                f"- NLI label: `{row.get('nli_label', '')}`",
                f"- NLI entailment: `{row.get('nli_max_entailment', None)}`",
                f"- NLI contradiction: `{row.get('nli_max_contradiction', None)}`",
                "",
                f"**Question:** {row.get('question', '')}",
                "",
                f"**Gold:** {row.get('gold_answers') or row.get('gold_short_answer', '')}",
                "",
                f"**Predicted:** {row.get('predicted_answer', '')}",
                "",
                f"**Model cited/supporting evidence status:** `{row.get('model_cited_or_supporting_evidence_status', '')}`",
                "",
                "**Retrieved evidence for review:**",
                "",
                evidence_block(row.get("retrieved_evidence_for_review", [])),
                "",
                "**Manual spot-check:**",
                "",
                "- manual_review_label:",
                "- manual_review_notes:",
                "",
            ]
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {len(rows)} cases to {args.output_md}")


if __name__ == "__main__":
    main()
