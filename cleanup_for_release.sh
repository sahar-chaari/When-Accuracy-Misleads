#!/usr/bin/env bash
# Run from the repository root, BEFORE `git init`.
# Removes files not used by any paper table, figure, or script.
# Verified against shared_inputs.py, analysis_01_risk_coverage.py,
# compute_harm_cost_from_labels.py, and run_faithfulness_scoring_all.sh.
set -euo pipefail

echo ">> internal notes"
rm -f  HippoRAG/TODO.md
rm -rf IRCoT/archive_reference

echo ">> superseded harm-cost scaffolds and severity templates"
echo "   (canonical ledgers live in annotations/harm_cost/)"
rm -rf CRAG/results/annotations
rm -rf CRAG/results/harm_cost
rm -rf HippoRAG/annotations
rm -rf HippoRAG/results/harm_cost
rm -rf HippoRAG/results/pubmedqa/harm_cost
rm -rf IRCoT/results/medmcqa/harm_cost
rm -rf IRCoT/results/pubmedqa/harm_cost


echo ">> superseded PubMedQA runs (restricted/PMID corpus; paper uses full1000)"
echo "   comment this block out if you prefer to archive them instead"
rm -f IRCoT/results/pubmedqa/confidence_runs/ircot_qwen25_32b_pubmedqa_60_ircot_qwen25_32b_evidence_conf_v2_20260613_154532.jsonl
rm -f IRCoT/results/pubmedqa/confidence_runs/ircot_qwen25_7b_pubmedqa_60_ircot_qwen25_7b_evidence_conf_v2_20260613_013038.jsonl
rm -f CRAG/results/pubmedqa/confidence_runs/crag_qwen25_32b_pubmedqa_60_crag_qwen25_32b_pubmedqa_pmid_20260613_172253.jsonl
rm -f CRAG/results/pubmedqa/confidence_runs/crag_qwen25_32b_pubmedqa_60_crag_qwen25_32b_pubmedqa_pmid_20260613_172253_health.json
rm -f CRAG/results/pubmedqa/confidence_runs/crag_qwen25_7b_pubmedqa_60_crag_qwen25_7b_pubmedqa_pmid_20260613_175948.jsonl
rm -f CRAG/results/pubmedqa/confidence_runs/crag_qwen25_7b_pubmedqa_60_crag_qwen25_7b_pubmedqa_pmid_20260613_175948_health.json
rm -f HippoRAG/results/pubmedqa/hipporag_pubmedqa60_20260612_095925.jsonl
rm -f HippoRAG/results/pubmedqa/hipporag_pubmedqa60_20260612_095925.metrics.json
rm -f HippoRAG/results/pubmedqa/hipporag_qwen25_32b_pubmedqa_60_20260613_142719.jsonl
rm -f HippoRAG/results/pubmedqa/hipporag_qwen25_32b_pubmedqa_60_20260613_142719.metrics.json
rm -f HippoRAG/results/pubmedqa/hipporag_qwen25_32b_pubmedqa_60_restricted_20260614_161918.jsonl
rm -f HippoRAG/results/pubmedqa/hipporag_qwen25_32b_pubmedqa_60_restricted_20260614_161918.metrics.json
rm -f HippoRAG/results/pubmedqa/hipporag_qwen25_7b_pubmedqa_60_restricted_20260614_161142.jsonl

echo ">> stray caches"
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

echo
echo "Done. Now verify nothing load-bearing was removed:"
echo "  python scripts/analysis/shared_inputs.py"
