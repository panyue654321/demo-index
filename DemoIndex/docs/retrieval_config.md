# DemoIndex Retrieval Config

This document is the consolidated configuration summary for the current retrieval stack.
It covers Stage 1 through Stage 5 in one place.

## Entry Points

Python:

- `parse_query(...)`
- `retrieve_candidates(...)`
- `localize_sections(...)`
- `retrieve_tree_candidates(...)`
- `expand_localized_sections(...)`
- `package_evidence(...)`
- `retrieve_evidence(...)`

CLI:

- `python -m DemoIndex.run retrieve ...`
- `python -m DemoIndex.run retrieve-tree ...`
- `python -m DemoIndex.run retrieve-evidence ...`

## Shared Debug Controls

All retrieval entrypoints support:

- `debug_log`
- `debug_log_dir`

CLI options:

- `--debug-log`
- `--debug-log-dir`

## Stage 1: Query Understanding

API options:

- `use_llm_parse`
- `parse_model`
- `parse_fallback_model`
- `retrieval_profile_path`

CLI options:

- `--disable-llm-parse`
- `--parse-model`
- `--parse-fallback-model`
- `--retrieval-profile-path`

Notes:

- `retrieval_profile_path` overrides `DEMOINDEX_RETRIEVAL_PROFILE_PATH`
- Stage 1 stays generic in core code and only uses an external profile when explicitly configured

Current defaults:

- `use_llm_parse=True`
- `parse_model="dashscope/qwen3.6-plus"`
- `parse_fallback_model="dashscope/qwen3.5-plus"`

## Stage 2: Global Candidate Recall

API options:

- `top_k_dense`
- `top_k_lexical`
- `top_k_fused_chunks`
- `top_k_docs`
- `top_k_sections_per_doc`
- `top_k_chunks_per_section`
- `embedding_model`
- `rrf_k`
- `lexical_score_threshold`
- `doc_score_chunk_limit`
- `section_score_chunk_limit`

CLI options:

- `--top-k-dense`
- `--top-k-lexical`
- `--top-k-fused-chunks`
- `--top-k-docs`
- `--top-k-sections-per-doc`
- `--top-k-chunks-per-section`
- `--embedding-model`
- `--rrf-k`
- `--lexical-score-threshold`
- `--doc-score-chunk-limit`
- `--section-score-chunk-limit`

Current defaults:

- `top_k_dense=60`
- `top_k_lexical=60`
- `top_k_fused_chunks=80`
- `top_k_docs=10`
- `top_k_sections_per_doc=3`
- `top_k_chunks_per_section=2`
- `embedding_model="text-embedding-v4"`
- `rrf_k=60`
- `lexical_score_threshold=0.18`
- `doc_score_chunk_limit=5`
- `section_score_chunk_limit=3`

## Stage 3: Tree Localization

API options:

- `mode` on `localize_sections(...)`
- `stage3_mode` on `retrieve_tree_candidates(...)` and `retrieve_evidence(...)`
- `top_k_tree_sections_per_doc`
- `top_k_anchor_sections_per_doc`
- `whole_doc_fallback`
- `rerank_model`
- `rerank_fallback_model`
- `stage3_shortlist_size`
- `stage3_relation_priors`

CLI options:

- `--stage3-mode`
- `--top-k-tree-sections-per-doc`
- `--top-k-anchor-sections-per-doc`
- `--disable-whole-doc-fallback`
- `--stage3-rerank-model`
- `--stage3-rerank-fallback-model`
- `--stage3-shortlist-size`
- `--stage3-relation-priors-json`

Current defaults:

- `stage3_mode="hybrid"`
- `top_k_tree_sections_per_doc=5`
- `top_k_anchor_sections_per_doc=3`
- `whole_doc_fallback=True`
- `rerank_model="dashscope/qwen3.6-plus"`
- `rerank_fallback_model="dashscope/qwen3.5-plus"`
- `stage3_shortlist_size=8`

`stage3_relation_priors` and `--stage3-relation-priors-json` use the same key set:

- `anchor`
- `descendant`
- `ancestor`
- `sibling`
- `doc_fallback`

Example:

```json
{
  "anchor": 4.0,
  "descendant": 2.75,
  "ancestor": 2.1,
  "sibling": 1.45,
  "doc_fallback": 0.55
}
```

## Stage 4: Context Expansion

API options:

- `top_k_focus_sections_per_doc`
- `max_ancestor_hops`
- `max_descendant_depth`
- `max_siblings_per_focus`
- `chunk_neighbor_window`
- `max_evidence_chunks_per_focus`
- `context_char_budget`

CLI options on `retrieve-evidence`:

- `--top-k-focus-sections-per-doc`
- `--max-ancestor-hops`
- `--max-descendant-depth`
- `--max-siblings-per-focus`
- `--chunk-neighbor-window`
- `--max-evidence-chunks-per-focus`
- `--context-char-budget`

Current defaults:

- `top_k_focus_sections_per_doc=3`
- `max_ancestor_hops=2`
- `max_descendant_depth=1`
- `max_siblings_per_focus=2`
- `chunk_neighbor_window=1`
- `max_evidence_chunks_per_focus=6`
- `context_char_budget=6000`

Notes:

- Stage 4 is deterministic
- It expands around Stage 3 focus sections with ancestors, descendants, siblings, and chunk neighbors
- It does not call a new LLM

## Stage 5: Evidence Packaging

API options:

- `stage5_relation_mode`
- `top_k_evidence_per_doc`
- `top_k_total_evidence`
- `stage5_relation_model`
- `stage5_relation_fallback_model`
- `stage5_relation_shortlist_size`

CLI options on `retrieve-evidence`:

- `--stage5-relation-mode`
- `--top-k-evidence-per-doc`
- `--top-k-total-evidence`
- `--stage5-relation-model`
- `--stage5-relation-fallback-model`
- `--stage5-relation-shortlist-size`

Current defaults:

- `stage5_relation_mode="heuristic"`
- `top_k_evidence_per_doc=3`
- `top_k_total_evidence=8`
- `stage5_relation_model="dashscope/qwen3.6-plus"`
- `stage5_relation_fallback_model="dashscope/qwen3.5-plus"`
- `stage5_relation_shortlist_size=8`

Notes:

- `heuristic` packages evidence only and leaves `relationship_label="unlabeled"`
- `hybrid` keeps the same heuristic package and then runs one global relation-labeling pass over the shortlisted evidence items

## Environment Variables

- `DATABASE_URL`
- `DASHSCOPE_API_KEY`
- `OPENAI_API_KEY`
- `DEMOINDEX_RETRIEVAL_PROFILE_PATH`
