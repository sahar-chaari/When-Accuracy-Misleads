# CRAG Baseline

**Paper:** Corrective Retrieval Augmented Generation (Yan et al., 2024) — arXiv:2401.15884

This workspace implements CRAG as the third baseline in the multi-axis RAG evaluation harness, alongside IRCoT and HippoRAG.

## Architecture

```
Question
   │
   ▼
BM25/ES retrieve top-5 passages
   │
   ▼
Retrieval Evaluator (Qwen zero-shot)
  → Correct / Incorrect / Ambiguous per passage
   │
   ├─ Any Correct  → use Correct passages only
   ├─ All Incorrect → second BM25 pass (keyword query)  ← replaces web search
   └─ Ambiguous    → Correct passages + second BM25 pass
   │
   ▼
Reader (Qwen): generate Answer + Confidence + Supporting evidence
```

## Adaptations from the published paper

| Component | Paper | This implementation |
|---|---|---|
| Retrieval evaluator | Fine-tuned T5-large | Qwen zero-shot (same LLM as reader) |
| Fallback on Incorrect/Ambiguous | Web search (Google) | Second BM25 pass over same StatPearls+textbooks corpus |
| Knowledge refinement | Sentence-level decomposition | Passage-level filtering |

All deviations are recorded in `manifests/crag_architecture.json`.

## Why these adaptations

- **Evaluator**: No fine-tuning data available; zero-shot Qwen avoids a second model family.
- **Fallback**: Web search is disabled per experiment constraints. The same shared corpus keeps the comparison fair — every baseline sees the same evidence pool.
- **Knowledge refinement**: Sentence-level scoring would triple evaluator calls; passage-level is adequate for the ~200-word passages in this corpus.

## PubMedQA note

The gold PubMedQA abstracts are **not** in the MedRAG StatPearls+textbooks corpus. Low retrieval recall on PubMedQA is expected and **explicitly flagged** in the health check output. The PubMedQA arm is treated as a calibration/overconfidence probe, not an accuracy benchmark.

## Running

```bash
# Smoke test (3 questions) on Nibi
sbatch slurm/crag_pubmedqa60_7b.sbatch  # add --smoke to run_crag.py call for 3-question test

# Full runs
sbatch slurm/crag_pubmedqa60_7b.sbatch
sbatch slurm/crag_medmcqa60_7b.sbatch
sbatch slurm/crag_pubmedqa60_32b.sbatch
sbatch slurm/crag_medmcqa60_32b.sbatch
```

## Output schema compatibility

CRAG emits the same canonical JSONL fields as IRCoT/HippoRAG:
`predicted_answer`, `confidence`, `supporting_evidence`, `supporting_evidence_verified_passage_ids`, `retrieved_passages`, `raw_generation`, `split`, `benchmark`.

All existing scorers (`phase1_score_outputs`, `phase1_risk_coverage`, `phase1_score_faithfulness_nli`, `phase1_scaffold_harm_cost`) consume CRAG output unchanged.

Extra CRAG-only fields (`crag_action`, `crag_eval_labels`, `crag_second_pass_used`) are present but ignored by scorers.
