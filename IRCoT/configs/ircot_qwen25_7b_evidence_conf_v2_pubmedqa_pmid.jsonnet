# PubMedQA config using the PMID-restricted corpus (same 207 passages as HippoRAG).
# Use with --config-id ircot_qwen25_7b_evidence_conf_v2.
# Difference from v1: source_corpus_name points to the small PMID-restricted index
# instead of the general MedRAG index that does not contain gold PubMedQA articles.
local model_name = std.extVar("MODEL");
local retrieval_corpus_name = "pubmedqa_pmid_restricted_60";
local project_dir = std.extVar("PROJECT_DIR");
local max_reasoning_steps = 3;
local prompt_reader_args = {
    "estimated_generation_length": 220,
    "shuffle": false,
    "model_length_limit": 1000000,
    "tokenizer_model_name": model_name,
};
local bm25_retrieval_count = 6;

{
    "start_state": "step_by_step_bm25_retriever",
    "end_state": "[EOQ]",
    "models": {
        "step_by_step_bm25_retriever": {
            "name": "retrieve_and_reset_paragraphs",
            "next_model": "step_by_step_cot_reasoning_gen",
            "retrieval_type": "bm25",
            "retriever_host": std.extVar("RETRIEVER_HOST"),
            "retriever_port": std.extVar("RETRIEVER_PORT"),
            "retrieval_count": bm25_retrieval_count,
            "global_max_num_paras": 15,
            "query_source": "question_or_last_generated_sentence",
            "source_corpus_name": retrieval_corpus_name,
            "document_type": "title_paragraph_text",
            "return_pids": false,
            "cumulate_titles": true,
            "end_state": "[EOQ]",
        },
        "step_by_step_cot_reasoning_gen": {
            "name": "step_by_step_cot_gen",
            "next_model": "step_by_step_exit_controller",
            "prompt_file": project_dir + "/prompts/pubmedqa/medical_context_cot_qa_qwen.txt",
            "question_prefix": "Answer the following biomedical question by reasoning step-by-step.\n",
            "prompt_reader_args": prompt_reader_args,
            "generation_type": "sentences",
            "reset_queries_as_sentences": false,
            "add_context": true,
            "shuffle_paras": false,
            "terminal_return_type": null,
            "disable_exit": true,
            "end_state": "[EOQ]",
            "gen_model": "hf_local",
            "model_name": model_name,
            "model_tokens_limit": 8192,
            "max_length": 160,
            "eos_text": "\n",
            "torch_dtype": "auto",
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "use_chat_template": true,
        },
        "step_by_step_exit_controller": {
            "name": "step_by_step_exit_controller",
            "next_model": "step_by_step_bm25_retriever",
            "answer_extractor_regex": ".* answer is:? (.*)\\.?",
            "answer_extractor_remove_last_fullstop": true,
            "terminal_state_next_model": "generate_main_question",
            "terminal_return_type": "pids",
            "max_num_sentences": max_reasoning_steps,
            "global_max_num_paras": 15,
            "end_state": "[EOQ]",
        },
        "generate_main_question": {
            "name": "copy_question",
            "next_model": "answer_main_question",
            "eoq_after_n_calls": 1,
            "end_state": "[EOQ]",
        },
        "answer_main_question": {
            "name": "llmqa",
            "next_model": null,
            "prompt_file": project_dir + "/prompts/pubmedqa/medical_yes_no_maybe_evidence_conf_qwen.txt",
            "question_prefix": "",
            "prompt_reader_args": prompt_reader_args,
            "end_state": "[EOQ]",
            "gen_model": "hf_local",
            "model_name": model_name,
            "model_tokens_limit": 8192,
            "max_length": 220,
            "eos_text": "\n\n",
            "add_context": true,
            "torch_dtype": "auto",
            "device_map": "auto",
            "attn_implementation": "sdpa",
            "use_chat_template": true,
        },
    },
    "reader": {
        "name": "multi_para_rc",
        "add_paras": false,
        "add_gold_paras": false,
        "add_pinned_paras": false,
    },
    "prediction_type": "answer",
}
