#!/usr/bin/env bash
# Run cited-evidence faithfulness scoring for the 12 paper configurations.

set -euo pipefail

SCRIPTS="IRCoT/scripts"
CONFIG="IRCoT/configs/faithfulness_nli_default.json"
OUT="results/faithfulness_nli"

mkdir -p "$OUT"

run_one() {
    local input="$1"
    local outname="$2"
    local start
    start=$(date +%s)
    echo ""
    echo "======================================================================"
    echo "SCORING: $outname"
    echo "  input: $input"
    echo "  $(date)"
    echo "======================================================================"
    python3 "$SCRIPTS/phase1_score_faithfulness_nli.py" \
        --input-jsonl  "$input" \
        --output-jsonl "$OUT/$outname.jsonl" \
        --config       "$CONFIG" \
        --device       cpu
    local end
    end=$(date +%s)
    echo "  done in $(( end - start ))s; rows=$(wc -l < "$OUT/$outname.jsonl")"
}

run_one \
  "IRCoT/results/medmcqa/confidence_runs/ircot_qwen25_7b_medmcqa_60_ircot_qwen25_7b_evidence_conf_v1_medmcqa_20260613_075253.jsonl" \
  "faithfulness_ircot_7b_medmcqa"

run_one \
  "IRCoT/results/pubmedqa/confidence_runs/ircot_qwen25_7b_pubmedqa_60_full1000_ircot_qwen25_7b_evidence_conf_v2_20260614_084634.jsonl" \
  "faithfulness_ircot_7b_pubmedqa_full1000"

run_one \
  "IRCoT/results/medmcqa/confidence_runs/ircot_qwen25_32b_medmcqa_60_ircot_qwen25_32b_evidence_conf_v1_medmcqa_20260613_154532.jsonl" \
  "faithfulness_ircot_32b_medmcqa"

run_one \
  "IRCoT/results/pubmedqa/confidence_runs/ircot_qwen25_32b_pubmedqa_60_full1000_ircot_qwen25_32b_evidence_conf_v2_20260614_084634.jsonl" \
  "faithfulness_ircot_32b_pubmedqa_full1000"

run_one \
  "CRAG/results/medmcqa/confidence_runs/crag_qwen25_7b_medmcqa_60_crag_qwen25_7b_medmcqa_20260613_172254.jsonl" \
  "faithfulness_crag_7b_medmcqa"

run_one \
  "CRAG/results/pubmedqa/confidence_runs/crag_qwen25_7b_pubmedqa_60_full1000_20260614_084944.jsonl" \
  "faithfulness_crag_7b_pubmedqa_full1000"

run_one \
  "CRAG/results/medmcqa/confidence_runs/crag_qwen25_32b_medmcqa_60_crag_qwen25_32b_medmcqa_20260613_175948.jsonl" \
  "faithfulness_crag_32b_medmcqa"

run_one \
  "CRAG/results/pubmedqa/confidence_runs/crag_qwen25_32b_pubmedqa_60_full1000_20260614_084634.jsonl" \
  "faithfulness_crag_32b_pubmedqa_full1000"

run_one \
  "HippoRAG/results/medmcqa/hipporag_qwen25_7b_medmcqa_60_20260615.jsonl" \
  "faithfulness_hipporag_7b_medmcqa"

run_one \
  "HippoRAG/results/pubmedqa/hipporag_qwen25_7b_pubmedqa_60_full1000_20260614_161142.jsonl" \
  "faithfulness_hipporag_7b_pubmedqa_full1000"

run_one \
  "HippoRAG/results/medmcqa/hipporag_qwen25_32b_medmcqa_60_20260614_042849.jsonl" \
  "faithfulness_hipporag_32b_medmcqa"

run_one \
  "HippoRAG/results/pubmedqa/hipporag_qwen25_32b_pubmedqa_60_full1000_resume_20260614_104710.jsonl" \
  "faithfulness_hipporag_32b_pubmedqa_full1000"

echo ""
echo "======================================================================"
echo "AGGREGATING faithfulness results"
echo "======================================================================"

python3 "$SCRIPTS/phase1_aggregate_faithfulness_nli.py" \
  --inputs \
    "$OUT/faithfulness_ircot_7b_medmcqa.jsonl" \
    "$OUT/faithfulness_ircot_7b_pubmedqa_full1000.jsonl" \
    "$OUT/faithfulness_ircot_32b_medmcqa.jsonl" \
    "$OUT/faithfulness_ircot_32b_pubmedqa_full1000.jsonl" \
    "$OUT/faithfulness_crag_7b_medmcqa.jsonl" \
    "$OUT/faithfulness_crag_7b_pubmedqa_full1000.jsonl" \
    "$OUT/faithfulness_crag_32b_medmcqa.jsonl" \
    "$OUT/faithfulness_crag_32b_pubmedqa_full1000.jsonl" \
    "$OUT/faithfulness_hipporag_7b_medmcqa.jsonl" \
    "$OUT/faithfulness_hipporag_7b_pubmedqa_full1000.jsonl" \
    "$OUT/faithfulness_hipporag_32b_medmcqa.jsonl" \
    "$OUT/faithfulness_hipporag_32b_pubmedqa_full1000.jsonl" \
  --labels \
    ircot_7b_medmcqa \
    ircot_7b_pubmedqa \
    ircot_32b_medmcqa \
    ircot_32b_pubmedqa \
    crag_7b_medmcqa \
    crag_7b_pubmedqa \
    crag_32b_medmcqa \
    crag_32b_pubmedqa \
    hipporag_7b_medmcqa \
    hipporag_7b_pubmedqa \
    hipporag_32b_medmcqa \
    hipporag_32b_pubmedqa \
  --output-json "$OUT/summary.json"

echo ""
echo "======================================================================"
echo "ALL DONE"
echo "======================================================================"
