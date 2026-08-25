# When Accuracy Misleads: Calibration, Faithfulness, and Harm in Medical RAG

Experimental artifact for a controlled, multi-axis evaluation of medical
retrieval-augmented generation (RAG).

> Sahar Chaari, Omar Abdelhedi, Diaa Azzam, Manar Hamed, Omid Reza Heidari,
> Xiang Chen Zhu, and Yassine Yaakoubi. "When Accuracy Misleads: Calibration,
> Faithfulness, and Harm in Medical RAG." IEEE AIDIST, 2026.


The harness scores any schema-conforming RAG system along four axes from a
**single saved inference run**: answer correctness, cited-evidence
faithfulness, selective-prediction quality (AUARC), and an IEC 62304-inspired
severity-weighted harm cost. Because every system writes the same JSONL schema,
all four axes are recomputed offline and deterministically, without re-running
inference.

This repository contains the fixed benchmark subsets, saved system outputs,
scoring scripts, faithfulness labels, AUARC summaries, severity ledgers, and
paper figures for the 12 system-benchmark configurations reported in the paper.

---

## What Is Evaluated

Three published RAG families, each at two reader scales, on two medical
benchmarks — 12 configurations in total. All systems share one reader
(Qwen2.5-Instruct, 7B and 32B, decoded deterministically) and one closed
retrieval corpus per benchmark.

| System | Retrieval | Note |
|--------|-----------|------|
| IRCoT | BM25 interleaved with chain-of-thought | 3 steps, 6 passages per step |
| CRAG | Retrieval-quality grader with corrective fallback | **Adapted closed-corpus variant** — see below |
| HippoRAG 2 | Knowledge-graph retrieval over OpenIE triples | all-MiniLM-L6-v2 embeddings |

**CRAG adaptation.** The evaluated CRAG is not the deployed pipeline. Two
changes were made for a controlled closed-corpus comparison: the original
web-search fallback is replaced by a second retrieval pass over the same
corpus, and the fine-tuned T5-large grader is replaced by the same Qwen reader
prompted zero-shot to label passages Correct / Incorrect / Ambiguous, with
knowledge refinement applied at passage level rather than by sentence-level
decomposition.

**Benchmarks.** PubMedQA (60 questions, balanced 20 yes / 20 no / 20 maybe,
seed 42) and MedMCQA (60 questions, 15 per answer choice A–D, seed 42).

---

## Evaluation Axes

| Axis | Metric | Main script |
|------|--------|-------------|
| Answer correctness | Accuracy, macro-F1 | `IRCoT/scripts/phase1_score_outputs.py` |
| Cited-evidence faithfulness | Four-label NLI | `IRCoT/scripts/phase1_score_faithfulness_nli.py` |
| Selective prediction | Accuracy-coverage AUARC | `IRCoT/scripts/phase1_risk_coverage.py` |
| Severity-weighted harm | IEC 62304-inspired weighted error | `scripts/analysis/compute_harm_cost_from_labels.py` |

The four-label faithfulness scheme is `supported` / `unsupported` /
`no-evidence-admitted` / `contradicted`. The `no-evidence-admitted` label
separates ungrounded generation from a model's explicit statement that the
retrieved passages do not contain the answer.

---

## Quick Start

```bash
pip install -r requirements.txt

# Verify all 12 canonical inputs are present and question IDs align.
# Must report: Inputs ready: 12/12 configurations.
python scripts/analysis/shared_inputs.py
```

### Recompute the paper analyses

```bash
# Risk-coverage curves, AUARC, bootstrap CIs, rank correlations
python scripts/analysis/analysis_01_risk_coverage.py
python scripts/analysis/analysis_02_bootstrap_auarc.py
python scripts/analysis/analysis_03_rank_correlation.py

# Severity-weighted harm cost
python scripts/analysis/compute_harm_cost_from_labels.py

# Robustness variant (abstention handling)
python scripts/analysis_skip_none/generate_skip_none_results.py
```

### Recompute faithfulness labels

Requires a GPU and downloads a DeBERTa-v3-large NLI checkpoint.

```bash
bash run_faithfulness_scoring_all.sh
```

---

## Paper Table Sources

| Paper content | Source |
|---------------|--------|
| Accuracy and macro-F1 | Per-system `metrics*.json` under `IRCoT/`, `CRAG/`, `HippoRAG/` |
| AUARC and rank tables | `results/analysis/analysis_01_risk_coverage.json` |
| Bootstrap AUARC intervals | `results/analysis/analysis_02_bootstrap_auarc.json` |
| Cross-axis rank correlations | `results/analysis/analysis_03_rank_correlation.json` |
| Faithfulness distribution | `results/faithfulness_nli/faithfulness_*.jsonl`, `summary.json` |
| Harm-cost table | `results/analysis/harm_cost_direct_20260615.csv` and `.json` |
| Accuracy-coverage figure | `figures/risk_coverage_medmcqa.pdf`, `figures/risk_coverage_pubmedqa.pdf` |
| Abstention-handling robustness | `results/analysis_skip_none/` and `figures_skip_none/` |

`results/faithfulness_nli/` contains exactly the 12 faithfulness files used by
the paper, plus `summary.json`.

---

## Canonical Inputs

All reported numbers derive from exactly twelve saved inference runs. The
authoritative list is `FILES` in `scripts/analysis/shared_inputs.py`.

| System | MedMCQA | PubMedQA |
|--------|---------|----------|
| IRCoT 7B | `IRCoT/results/medmcqa/confidence_runs/ircot_7b_medmcqa.jsonl` | `IRCoT/results/pubmedqa/confidence_runs/ircot_7b_pubmedqa_full1000.jsonl` |
| IRCoT 32B | `IRCoT/results/medmcqa/confidence_runs/ircot_32b_medmcqa.jsonl` | `IRCoT/results/pubmedqa/confidence_runs/ircot_32b_pubmedqa_full1000.jsonl` |
| CRAG 7B | `CRAG/results/medmcqa/confidence_runs/crag_7b_medmcqa.jsonl` | `CRAG/results/pubmedqa/confidence_runs/crag_7b_pubmedqa_full1000.jsonl` |
| CRAG 32B | `CRAG/results/medmcqa/confidence_runs/crag_32b_medmcqa.jsonl` | `CRAG/results/pubmedqa/confidence_runs/crag_32b_pubmedqa_full1000.jsonl` |
| HippoRAG 7B | `HippoRAG/results/medmcqa/hipporag_7b_medmcqa.jsonl` | `HippoRAG/results/pubmedqa/hipporag_7b_pubmedqa_full1000.jsonl` |
| HippoRAG 32B | `HippoRAG/results/medmcqa/hipporag_32b_medmcqa.jsonl` | `HippoRAG/results/pubmedqa/hipporag_32b_pubmedqa_full1000.jsonl` |

PubMedQA runs use the full 1,000-question PQA-L retrieval corpus, so gold
passages compete with distractors beyond the question-specific PMIDs.

Each run file records, per question: question identifier, gold label, predicted
label, parsed answer, verbatim supporting evidence, self-reported confidence,
and all retrieved passages. Rows with no parseable confidence are treated as
abstentions by the AUARC scorer.

---

## Repository Layout

```
├── IRCoT/ CRAG/ HippoRAG/   # per-system configs, prompts, scripts, saved runs
├── scripts/analysis/        # cross-system offline scorers
├── data/fixed/              # fixed 60-question benchmark subsets
├── annotations/harm_cost/   # question-level severity ledgers
├── results/                 # aggregated analysis outputs
└── figures/                 # paper figures
```

Two conventions are historical and are documented rather than changed, because
the analysis scripts reference these paths directly:

1. **`IRCoT/results/risk_coverage/` holds risk-coverage summaries for all three
   systems**, not only IRCoT. `analysis_01_risk_coverage.py` reads the CRAG and
   HippoRAG-32B curves from there, and the HippoRAG-7B curves from
   `HippoRAG/results/risk_coverage/`.
2. `data/processed/pubmedqa_150_*` are Phase-1 intermediates retained because
   the PubMedQA corpus-preparation scripts consume them. The paper evaluates
   only the fixed 60-question subsets in `data/fixed/` and
   `HippoRAG/data/fixed/`.

Relocating these files requires updating the hardcoded paths in
`scripts/analysis/shared_inputs.py` and
`scripts/analysis/analysis_01_risk_coverage.py`.

---

## Data Not Included

The fixed question subsets are included. Two large corpora are not duplicated
in git:

```bash
# MedRAG textbooks + StatPearls, for MedMCQA retrieval
python scripts/download_medrag_corpus.py
# -> data/corpus/medrag_textbooks_statpearls_corpus.jsonl

# PubMedQA 1,000-question corpus (~3,358 passages)
python IRCoT/scripts/prepare_pubmedqa_large_corpus.py
```

Neither is needed to recompute the paper tables from saved outputs — only to
re-run inference.

---

## Severity Ledgers and Their Limits

`annotations/harm_cost/severity_labels_{medmcqa,pubmedqa}_60_seed42.jsonl` are
the only severity inputs used by `compute_harm_cost_from_labels.py`. Each
question is assigned one IEC 62304 class — A (no injury, weight 1), B
(non-serious injury, weight 4), C (serious injury or death, weight 10) —
independently of any system's predictions and fixed before system scoring.

| Benchmark | A / B / C | Total weight |
|-----------|-----------|--------------|
| MedMCQA | 41 / 15 / 4 | 141 |
| PubMedQA | 14 / 45 / 1 | 204 |

**These labels are rubric-generated by an automated annotator, not
clinician-adjudicated.** The harm cost H should be read as an exploratory proxy
for clinical cost. The ledgers are released in this fixed format precisely so
clinician review can replace them later while preserving the same scoring
interface, and so H can be recomputed under an alternative labelling without
re-running inference.

Regenerating the ledgers (not needed to reproduce the paper) requires an
annotator endpoint:

```bash
export SEVERITY_ANNOTATOR_API_KEY=...
python scripts/analysis/build_severity_ledgers.py
```

---

## Known Limitations

Carried over from the paper, and relevant to anyone reusing this artifact:

- **Scale.** 60 questions per benchmark. Bootstrap intervals are wide; only the
  MedMCQA HippoRAG-32B − IRCoT-7B AUARC gap excludes zero. Individual rank
  reversals are not significant at this scale.
- **Confidence signal.** AUARC uses coarse self-reported verbal confidence, not
  token probabilities. The high-confidence tail is sparsely populated. The
  elicitation prompts are not byte-identical across systems.
- **Judges.** Faithfulness uses an automated NLI model with fixed thresholds and
  a lexical no-evidence cue list, none human-validated here. Severity labels
  come from a rubric annotator in the same model family as the evaluated
  readers, with no inter-annotator agreement reported.
- **Coverage.** One reader family at two scales, two English-language
  benchmarks. Generalisation to other readers, languages, and clinical settings
  is untested.

---
