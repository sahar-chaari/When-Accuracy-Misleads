# Clean HippoRAG Wrapper Workspace

This folder is the clean wrapper workspace for the second published RAG system candidate: official HippoRAG.

The paper evaluates whether published RAG methods that perform well on Wikipedia-style QA remain reliable on medical QA. This workspace must therefore keep HippoRAG separate from IRCoT while using the same fixed medical questions and the same answer/evidence/confidence output contract.

## System Identity

- System label: `hipporag_official_hf_local`
- Baseline: official OSU NLP HippoRAG repository
- Official clone location on Narval: `external/official_hipporag`
- Intended comparison: same fixed questions and same reader prompt contract as IRCoT
- Output contract:

```text
Final answer: <answer>
Supporting evidence: <exact evidence sentence or passage>
Confidence: <number between 0 and 1>
```

Use `hipporag_official_hf_local` only when the official HippoRAG package is used with minimal environment/output-wrapper patches. If the algorithm is rewritten, use `hipporag_reimplementation` instead.

## Current Scope

The immediate goal is not PubMedQA 60 yet. First we run a tiny 2-5 question smoke test to confirm:

- the official HippoRAG package imports,
- indexing works,
- retrieval works,
- the wrapper can save retrieved passages,
- the same answer/evidence/confidence schema can be parsed.

After the smoke test passes, we can adapt the wrapper to the same fixed PubMedQA 60 questions used for IRCoT.

## Notes

The official HippoRAG repository supports OpenAI-compatible LLM and embedding endpoints. The smoke Slurm script defaults to OpenAI-compatible online mode because it is the lightest way to validate the wrapper before using larger local models or heavier GPU jobs.

Do not copy IRCoT results or old debug outputs into this workspace.
