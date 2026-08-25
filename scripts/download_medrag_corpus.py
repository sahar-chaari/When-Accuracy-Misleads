#!/usr/bin/env python3
"""Build the JSONL corpus expected by the HippoRAG medical runners.

The runner expects one JSON object per passage with at least:
  passage_id, title, paragraph_text

This helper downloads public MedRAG textbooks chunks from Hugging Face and can
reconstruct the StatPearls chunks from the public NCBI Bookshelf tarball using
the same parsing rules as MedRAG's `src/data/statpearls.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

STATPEARLS_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["textbooks", "statpearls"],
        choices=["textbooks", "statpearls"],
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke/debug runs.")
    parser.add_argument(
        "--limit-strategy",
        choices=["balanced", "first"],
        default="balanced",
        help="How to cap rows when multiple sources are requested.",
    )
    parser.add_argument("--statpearls-url", default=STATPEARLS_URL)
    return parser.parse_args()


def coerce_passage(obj: dict[str, Any], source: str, fallback_id: str) -> dict[str, Any] | None:
    text = (
        obj.get("paragraph_text")
        or obj.get("text")
        or obj.get("content")
        or obj.get("contents")
        or obj.get("chunk")
        or obj.get("passage")
    )
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return None

    title = (
        obj.get("title")
        or obj.get("source_doc_id")
        or obj.get("book")
        or obj.get("source")
        or source
    )
    passage_id = obj.get("passage_id") or obj.get("id") or obj.get("_id") or fallback_id
    return {
        "passage_id": str(passage_id),
        "title": str(title or "").strip(),
        "paragraph_text": text,
        "source": source,
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def download_textbooks(cache_dir: Path | None) -> list[dict[str, Any]]:
    from huggingface_hub import snapshot_download

    local_dir = Path(
        snapshot_download(
            repo_id="MedRAG/textbooks",
            repo_type="dataset",
            allow_patterns=["chunk/*.jsonl"],
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    rows: list[dict[str, Any]] = []
    for path in sorted((local_dir / "chunk").glob("*.jsonl")):
        source = f"textbooks:{path.stem}"
        for line_no, obj in iter_jsonl(path):
            row = coerce_passage(obj, source, f"{path.stem}_{line_no}")
            if row:
                rows.append(row)
    return rows


def ends_with_ending_punctuation(text: str) -> bool:
    return any(text.endswith(char) for char in (".", "?", "!"))


def concat_title_content(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    if ends_with_ending_punctuation(title):
        return f"{title} {content}"
    return f"{title}. {content}"


def extract_text(element: ET.Element) -> str:
    text = (element.text or "").strip()
    for child in element:
        child_text = extract_text(child)
        if child_text:
            text += (" " if text else "") + child_text
        if child.tail and child.tail.strip():
            text += (" " if text else "") + child.tail.strip()
    return text.strip()


def is_subtitle(element: ET.Element) -> bool:
    children = list(element)
    return (
        element.tag == "p"
        and len(children) == 1
        and children[0].tag == "bold"
        and not (children[0].tail and children[0].tail.strip())
    )


def extract_statpearls_file(path: Path) -> list[dict[str, Any]]:
    fname = path.name.replace(".nxml", "")
    tree = ET.parse(path)
    title_node = tree.getroot().find(".//title")
    if title_node is None or not title_node.text:
        return []

    article_title = title_node.text.strip()
    saved: list[dict[str, Any]] = []
    index = 0
    for section in tree.getroot().findall(".//sec"):
        section_title_node = section.find("./title")
        if section_title_node is None or not section_title_node.text:
            continue
        section_title = section_title_node.text.strip()
        prefix = " -- ".join([article_title, section_title])
        last_text: str | None = None
        last_row: dict[str, Any] | None = None
        last_node: ET.Element | None = None

        for child in section:
            if is_subtitle(child):
                last_text = None
                last_row = None
                subtitle = extract_text(child)
                prefix = " -- ".join(prefix.split(" -- ")[:2] + [subtitle])
            elif child.tag == "p":
                curr_text = extract_text(child)
                if not curr_text:
                    continue
                if len(curr_text) < 200 and last_text is not None and last_row is not None and len(last_text + curr_text) < 1000:
                    last_text = " ".join([last_row["content"], curr_text])
                    last_row.update({"content": last_text, "contents": concat_title_content(last_row["title"], last_text)})
                    saved[-1] = last_row
                else:
                    last_text = curr_text
                    last_row = {"id": "_".join([fname, str(index)]), "title": prefix, "content": curr_text}
                    last_row["contents"] = concat_title_content(last_row["title"], last_row["content"])
                    saved.append(last_row)
                    index += 1
            elif child.tag == "list":
                list_text = [extract_text(item) for item in child]
                list_text = [item for item in list_text if item]
                joined = " ".join(list_text)
                if not joined:
                    continue
                if last_text is not None and last_row is not None and len(joined + last_text) < 1000:
                    last_text = " ".join([last_row["content"]] + list_text)
                    last_row.update({"content": last_text, "contents": concat_title_content(last_row["title"], last_text)})
                    saved[-1] = last_row
                elif len(joined) < 1000:
                    last_text = joined
                    last_row = {"id": "_".join([fname, str(index)]), "title": prefix, "content": joined}
                    last_row["contents"] = concat_title_content(last_row["title"], last_row["content"])
                    saved.append(last_row)
                    index += 1
                else:
                    last_text = None
                    last_row = None
                    for item in list_text:
                        saved.append(
                            {
                                "id": "_".join([fname, str(index)]),
                                "title": prefix,
                                "content": item,
                                "contents": concat_title_content(prefix, item),
                            }
                        )
                        index += 1
                if last_node is not None and is_subtitle(last_node):
                    prefix = " -- ".join([article_title, section_title])
            last_node = child
    return saved


def safe_extract_tar(tar_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not str(member_path).startswith(str(target_dir)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(target_dir)


def download_statpearls(work_dir: Path, url: str) -> list[dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    tar_path = work_dir / "statpearls_NBK430685.tar.gz"
    extracted_dir = work_dir / "statpearls_NBK430685"

    if not tar_path.exists():
        print(f"Downloading StatPearls from {url}", flush=True)
        urllib.request.urlretrieve(url, tar_path)

    if not extracted_dir.exists():
        print(f"Extracting {tar_path}", flush=True)
        safe_extract_tar(tar_path, work_dir)

    rows: list[dict[str, Any]] = []
    nxml_files = sorted(extracted_dir.glob("*.nxml"))
    if not nxml_files:
        raise RuntimeError(f"No .nxml files found under {extracted_dir}")
    for path in nxml_files:
        for obj in extract_statpearls_file(path):
            row = coerce_passage(obj, f"statpearls:{path.stem}", str(obj.get("id") or path.stem))
            if row:
                rows.append(row)
    return rows


def apply_limit(source_batches: list[tuple[str, list[dict[str, Any]]]], limit: int | None, strategy: str) -> list[dict[str, Any]]:
    if limit is None:
        return [row for _, rows in source_batches for row in rows]

    if strategy == "first" or len(source_batches) <= 1:
        rows = [row for _, batch in source_batches for row in batch]
        return rows[:limit]

    take = max(1, limit // len(source_batches))
    rows: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for _, batch in source_batches:
        rows.extend(batch[:take])
        leftovers.extend(batch[take:])
    if len(rows) < limit:
        rows.extend(leftovers[: limit - len(rows)])
    return rows[:limit]


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir or (args.hf_cache_dir / "medrag_sources" if args.hf_cache_dir else args.output.parent / ".medrag_sources")

    source_batches: list[tuple[str, list[dict[str, Any]]]] = []
    if "textbooks" in args.sources:
        rows = download_textbooks(args.hf_cache_dir)
        print(json.dumps({"source": "textbooks", "rows": len(rows)}), flush=True)
        source_batches.append(("textbooks", rows))

    if "statpearls" in args.sources:
        rows = download_statpearls(work_dir / "statpearls", args.statpearls_url)
        print(json.dumps({"source": "statpearls", "rows": len(rows)}), flush=True)
        source_batches.append(("statpearls", rows))

    rows = apply_limit(source_batches, args.limit, args.limit_strategy)
    if not rows:
        raise SystemExit("No passages were loaded; cannot write corpus.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
