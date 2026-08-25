#!/usr/bin/env python3
"""Index MedRAG chunk corpora into the Elasticsearch schema expected by IRCoT."""

from __future__ import annotations

import argparse
import json
import sys
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
            "url": {"type": "text", "analyzer": "english"},
            "is_abstract": {"type": "boolean"},
            "source_corpus": {"type": "keyword"},
        }
    }
}


def iter_chunk_rows(db_dir: Path, corpora: list[str], limit: int | None, limit_per_corpus: int | None):
    emitted = 0
    total_malformed = 0
    for corpus in corpora:
        corpus_emitted = 0
        corpus_malformed = 0
        chunk_dir = db_dir / corpus / "chunk"
        if not chunk_dir.exists():
            raise FileNotFoundError(
                f"Missing MedRAG chunk directory: {chunk_dir}. "
                "Download/chunk the corpus before indexing."
            )
        for path in sorted(chunk_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line_index, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        corpus_malformed += 1
                        total_malformed += 1
                        if corpus_malformed <= 5:
                            print(
                                f"Skipping malformed JSON line in {path}:{line_index + 1}",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                    if not isinstance(item, dict):
                        corpus_malformed += 1
                        total_malformed += 1
                        if corpus_malformed <= 5:
                            print(
                                f"Skipping non-object JSON row in {path}:{line_index + 1}",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                    content = item.get("content") or item.get("contents") or ""
                    title = item.get("title") or corpus
                    doc_id = item.get("id") or f"{corpus}:{path.stem}:{line_index}"
                    yield {
                        "id": doc_id,
                        "title": title,
                        "paragraph_index": line_index,
                        "paragraph_text": content,
                        "url": "",
                        "is_abstract": True,
                        "source_corpus": corpus,
                    }
                    emitted += 1
                    corpus_emitted += 1
                    if limit is not None and emitted >= limit:
                        return
                    if limit_per_corpus is not None and corpus_emitted >= limit_per_corpus:
                        break
            if limit_per_corpus is not None and corpus_emitted >= limit_per_corpus:
                break
        if corpus_malformed:
            print(
                f"Skipped {corpus_malformed} malformed/non-object rows while indexing {corpus}.",
                file=sys.stderr,
                flush=True,
            )
    if total_malformed:
        print(f"Skipped {total_malformed} malformed/non-object rows in total.", file=sys.stderr, flush=True)


def iter_es_actions(index_name: str, db_dir: Path, corpora: list[str], limit: int | None, limit_per_corpus: int | None):
    for idx, row in enumerate(iter_chunk_rows(db_dir, corpora, limit, limit_per_corpus), start=1):
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": idx,
            "_source": row,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--index-name", default="medrag_pubmed_textbooks_statpearls")
    parser.add_argument("--corpora", nargs="+", default=["pubmed", "textbooks", "statpearls"])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--limit", type=int, default=None, help="optional cap for smoke indexing")
    parser.add_argument("--limit-per-corpus", type=int, default=None, help="optional per-corpus cap for smoke indexing")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    es = Elasticsearch([{"host": args.host, "port": args.port}], timeout=500, max_retries=2, retry_on_timeout=True)
    if es.indices.exists(index=args.index_name):
        if not args.force:
            raise SystemExit(f"Index {args.index_name} already exists. Pass --force to recreate it.")
        es.indices.delete(index=args.index_name)

    es.indices.create(index=args.index_name, body=INDEX_SETTINGS)
    success, errors = bulk(
        es,
        tqdm(
            iter_es_actions(args.index_name, args.db_dir, args.corpora, args.limit, args.limit_per_corpus),
            desc="indexing",
        ),
        stats_only=True,
        chunk_size=500,
        request_timeout=500,
    )
    es.indices.refresh(index=args.index_name)
    count = es.count(index=args.index_name)["count"]
    print(json.dumps({"indexed": success, "errors": errors, "count": count, "index": args.index_name}, indent=2))


if __name__ == "__main__":
    main()
