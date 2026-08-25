"""
Shared input definitions for post-hoc analysis scripts.
All three analysis scripts import from here.
"""
import json, os

BASE = os.path.join(os.path.dirname(__file__), "..", "..")

# ── File map: (system_label, benchmark, path) ──────────────────────────────
FILES = [
    # MedMCQA
    ("IRCoT 7B",    "MedMCQA",
     "IRCoT/results/medmcqa/confidence_runs/ircot_qwen25_7b_medmcqa_60_ircot_qwen25_7b_evidence_conf_v1_medmcqa_20260613_075253.jsonl"),
    ("IRCoT 32B",   "MedMCQA",
     "IRCoT/results/medmcqa/confidence_runs/ircot_qwen25_32b_medmcqa_60_ircot_qwen25_32b_evidence_conf_v1_medmcqa_20260613_154532.jsonl"),
    ("CRAG 7B",     "MedMCQA",
     "CRAG/results/medmcqa/confidence_runs/crag_qwen25_7b_medmcqa_60_crag_qwen25_7b_medmcqa_20260613_172254.jsonl"),
    ("CRAG 32B",    "MedMCQA",
     "CRAG/results/medmcqa/confidence_runs/crag_qwen25_32b_medmcqa_60_crag_qwen25_32b_medmcqa_20260613_175948.jsonl"),
    ("HippoRAG 7B", "MedMCQA",
     "HippoRAG/results/medmcqa/hipporag_qwen25_7b_medmcqa_60_20260615.jsonl"),
    ("HippoRAG 32B","MedMCQA",
     "HippoRAG/results/medmcqa/hipporag_qwen25_32b_medmcqa_60_20260614_042849.jsonl"),
    # PubMedQA (full-1000 corpus)
    ("IRCoT 7B",    "PubMedQA",
     "IRCoT/results/pubmedqa/confidence_runs/ircot_qwen25_7b_pubmedqa_60_full1000_ircot_qwen25_7b_evidence_conf_v2_20260614_084634.jsonl"),
    ("IRCoT 32B",   "PubMedQA",
     "IRCoT/results/pubmedqa/confidence_runs/ircot_qwen25_32b_pubmedqa_60_full1000_ircot_qwen25_32b_evidence_conf_v2_20260614_084634.jsonl"),
    ("CRAG 7B",     "PubMedQA",
     "CRAG/results/pubmedqa/confidence_runs/crag_qwen25_7b_pubmedqa_60_full1000_20260614_084944.jsonl"),
    ("CRAG 32B",    "PubMedQA",
     "CRAG/results/pubmedqa/confidence_runs/crag_qwen25_32b_pubmedqa_60_full1000_20260614_084634.jsonl"),
    ("HippoRAG 7B", "PubMedQA",
     "HippoRAG/results/pubmedqa/hipporag_qwen25_7b_pubmedqa_60_full1000_20260614_161142.jsonl"),
    ("HippoRAG 32B","PubMedQA",
     "HippoRAG/results/pubmedqa/hipporag_qwen25_32b_pubmedqa_60_full1000_resume_20260614_104710.jsonl"),
]


def _gold(row):
    ga = row.get("gold_answers")
    if isinstance(ga, list) and ga:
        return str(ga[0]).lower().strip()
    gs = row.get("gold_short_answer", "")
    return str(gs).lower().strip()


def _pred(row):
    p = row.get("predicted_answer") or row.get("parsed_label", "")
    return str(p).lower().strip()


def _conf(row):
    c = row.get("confidence")
    if c is None:
        c = row.get("self_reported_confidence")
    if c is None:
        return None
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


def load_rows(rel_path):
    """Load JSONL rows from BASE/rel_path and extract (qid, correct, conf).
    Returns empty list if the file does not exist (pending run)."""
    path = os.path.join(BASE, rel_path)
    if not os.path.exists(path):
        return []
    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for r in raw:
        qid  = str(r["question_id"])
        corr = int(_gold(r) == _pred(r))
        conf = _conf(r)
        out.append((qid, corr, conf))
    return out


def available_files():
    """Return FILES entries whose data file exists on disk."""
    return [(sys, bench, rel) for sys, bench, rel in FILES
            if os.path.exists(os.path.join(BASE, rel))]


def validate_inputs():
    """Check file availability and question-ID consistency. Warns on missing."""
    print("=" * 70)
    print("INPUT FILE MAP")
    print("=" * 70)
    bench_ids = {}
    id_mismatch = False
    for sys, bench, rel in FILES:
        abs_path = os.path.join(BASE, rel)
        exists = os.path.exists(abs_path)
        rows = load_rows(rel)
        qids = [r[0] for r in rows]
        n_no_conf = sum(1 for r in rows if r[2] is None)
        tag = "OK" if exists else "PENDING"
        print(f"  {tag:7s}  {sys:<14} {bench:<9} n={len(rows)}  "
              f"no_conf={n_no_conf}  {os.path.basename(rel)}")
        if exists and qids:
            if bench not in bench_ids:
                bench_ids[bench] = set(qids)
            elif bench_ids[bench] != set(qids):
                print(f"  *** MISMATCH in question IDs for {bench} / {sys} ***")
                id_mismatch = True
    print()
    for bench, ids in bench_ids.items():
        print(f"  {bench}: {len(ids)} unique question IDs  OK")
    pending = [(s, b) for s, b, r in FILES
               if not os.path.exists(os.path.join(BASE, r))]
    if pending:
        print(f"\n  PENDING ({len(pending)} run(s) not yet available):")
        for s, b in pending:
            print(f"    {s} / {b}")
    print()
    if id_mismatch:
        raise RuntimeError("Question-ID mismatch - see above.")
    print(f"Inputs ready: {len(FILES) - len(pending)}/{len(FILES)} configurations.\n")


if __name__ == "__main__":
    validate_inputs()
