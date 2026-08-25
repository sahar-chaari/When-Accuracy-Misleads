#!/usr/bin/env python3
"""Index the PubMedQA PMID-restricted corpus into Elasticsearch for IRCoT.

This creates a small index (207 passages for the 60-question set, 517 for all
150 questions) so IRCoT retrieves from the same documents that HippoRAG indexes.
The existing medrag_pubmed_textbooks_statpearls index is not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "english"},
            "paragraph_index": {"type": "integer"},
            "paragraph_text": {"type": "text", "analyzer": "english"},
            "url": {"type": "text"},
            "is_abstract": {"type": "boolean"},
            "source_corpus": {"type": "keyword"},
        }
    }
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_question_pmids(questions_jsonl: Path) -> set[str]:
    rows = load_jsonl(questions_jsonl)
    pmids: set[str] = set()
    for row in rows:
        # 'id' field holds the PMID in the fixed-60 question format
        pmid = str(row.get("id") or row.get("pmid") or "").strip()
        if pmid:
            pmids.add(pmid)
        # Also collect any passage_id-style IDs like "26200172::P1"
        for gid in row.get("gold_evidence_ids") or []:
            pmids.add(str(gid).split("::")[0])
    return pmids


def iter_actions(corpus_rows: list[dict], source_corpus: str):
    for idx, row in enumerate(corpus_rows, start=1):
        passage_id = str(row.get("passage_id") or f"passage:{idx}")
        pmid = str(row.get("pmid") or "").strip()
        section = str(row.get("section") or "").strip()
        text = str(row.get("text") or "").strip()
        local_pid = str(row.get("local_pid") or "").strip()

        # paragraph_index from local_pid ("P1" -> 1) or fallback to enumeration
        try:
            para_idx = int(local_pid.lstrip("P")) if local_pid.startswith("P") else idx
        except ValueError:
            para_idx = idx

        # Title carries PMID + section so BM25 can match entity/concept terms
        title = f"PMID:{pmid} {section}".strip() if pmid else section

        yield {
            "_op_type": "index",
            "_index": source_corpus,
            "_id": idx,
            "_source": {
                "id": passage_id,
                "title": title,
                "paragraph_index": para_idx,
                "paragraph_text": text,
                "url": "",
                "is_abstract": True,
                "source_corpus": source_corpus,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index the PubMedQA PMID-restricted corpus for IRCoT BM25 retrieval."
    )
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=Path("data/processed/pubmedqa_150_corpus.jsonl"),
        help="Path to the PubMedQA corpus JSONL (default: HippoRAG processed corpus).",
    )
    parser.add_argument(
        "--questions-jsonl",
        type=Path,
        default=None,
        help="If given, restrict indexing to PMIDs that appear in these questions.",
    )
    parser.add_argument(
        "--index-name",
        default="pubmedqa_pmid_restricted_60",
        help="Elasticsearch index name (used as source_corpus in the IRCoT config).",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--force", action="store_true", help="Delete existing index before creating.")
    args = parser.parse_args()

    corpus_rows = load_jsonl(args.corpus_jsonl)
    print(f"Loaded {len(corpus_rows)} passages from {args.corpus_jsonl}")

    if args.questions_jsonl:
        pmid_filter = load_question_pmids(args.questions_jsonl)
        print(f"Restricting to {len(pmid_filter)} question PMIDs from {args.questions_jsonl}")
        corpus_rows = [
            r for r in corpus_rows
            if str(r.get("pmid") or "").strip() in pmid_filter
        ]
        print(f"Passages after PMID filter: {len(corpus_rows)}")

    es = Elasticsearch(
        [{"host": args.host, "port": args.port}],
        timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )

    if es.indices.exists(index=args.index_name):
        if not args.force:
            raise SystemExit(
                f"Index '{args.index_name}' already exists. Pass --force to recreate it."
            )
        print(f"Deleting existing index '{args.index_name}' ...")
        es.indices.delete(index=args.index_name)

    print(f"Creating index '{args.index_name}' ...")
    es.indices.create(index=args.index_name, body=INDEX_SETTINGS)

    actions = list(iter_actions(corpus_rows, args.index_name))
    print(f"Indexing {len(actions)} passages ...")
    success, errors = bulk(
        es,
        tqdm(actions, desc="indexing"),
        stats_only=True,
        chunk_size=200,
        request_timeout=120,
    )
    es.indices.refresh(index=args.index_name)
    count = es.count(index=args.index_name)["count"]
    result = {
        "index": args.index_name,
        "indexed": success,
        "errors": errors,
        "count_after_refresh": count,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(f"Indexing had {errors} errors.")


if __name__ == "__main__":
    main()
