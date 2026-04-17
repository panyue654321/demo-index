# DemoIndex Search Plan B: Doc Hit Only + Whole-Tree Entry

## Goal

This plan is the simpler fallback design.

It assumes that the global layer only needs to answer one question:

- which documents are likely relevant

After that, every selected document is searched from the whole document tree, without using section anchors as the primary entry point.

## Design Summary

This plan still uses two index layers:

- a global chunk vector index for cross-document recall
- a per-document tree index for single-document search

This plan is the closest match to the public PageIndex flow:

- `Document Search by Semantics`: use chunks to score documents across the corpus
- then enter the chosen document with one whole-tree search entry

It can also reuse `Hybrid Tree Search` ideas inside a document, but without making `section anchors` the main entry policy.

But the handoff is simpler:

1. understand the query and extract constraints
2. run global chunk retrieval across all reports
3. aggregate hits only to the document level
4. choose top-N documents
5. for each selected document, enter from the whole document tree
6. let tree search decide the relevant sections from scratch

In shorthand, the flow is:

- PageIndex multi-document pattern: `chunk -> doc`
- DemoIndex fallback plan: `chunk -> doc -> whole-tree search`

## What Is Indexed

### Global Vector Index

The main global index is still built on body chunks.

Each chunk should keep:

- `chunk_id`
- `doc_id`
- `section_id`
- `chunk_index`
- `chunk_text`
- `embedding`
- `title`
- `title_path`
- `page_index`
- report metadata such as date, region, platform, genre, source

Embedding input should be:

- `title_path + title + chunk_body`

### Document Tree Index

The existing `document_sections` table remains the single-document reasoning index.

Recommended fields:

- `section_id`
- `parent_id`
- `doc_id`
- `node_id`
- `title`
- `depth`
- `summary`
- `title_path`
- `preorder`
- `subtree_end`

## Retrieval Flow

### Stage 1: Query Understanding

Extract:

- topic
- time scope
- intent type such as trend, benchmark, diagnosis, strategy, comparison

Implementation note:

- Stage 1 should stay generic in core code
- domain-specific fields such as metrics, region, platform, and genre should only come from an optional external retrieval profile or a fallback LLM parse for sparse queries
- already-informative Chinese queries should usually skip query-time LLM enrichment

### Stage 2: Global Candidate Recall

Run:

- dense chunk ANN retrieval
- lexical retrieval over `title`, `title_path`, and `search_text`

Implementation note:

- lexical recall should remain PostgreSQL-based
- Chinese-heavy queries should rely on derived search terms plus weighted title/title-path/body matching
- `pg_trgm` should be used as a ranking aid, not as a strict whole-query filter

The goal is only to find relevant documents, not relevant sections.

This stage is effectively the same handoff style as public PageIndex `Document Search by Semantics`.

### Stage 3: Document Selection

Aggregate all chunk hits to `doc_id`.

Possible signals:

- max chunk score
- average of top chunk scores
- number of matched chunks
- metadata match score

Then choose top-N documents.

No section anchor is retained as a required entry point.

Section-level information can still be logged for debugging, but it is not used as the formal tree-entry prior.

### Stage 4: Whole-Tree Search

For each selected document:

1. load the whole document tree
2. run tree search on the entire structure
3. identify relevant nodes
4. read the needed pages or sections
5. synthesize evidence

The tree search itself stays section-based.
The difference is only the entry policy.

This is why the plan is closer to public PageIndex examples than Plan A:
the selected document is treated as one retrieval entry, and the tree search reasons over the full structure from the beginning.

## Why This Plan Exists

This plan is designed to minimize risk from bad local anchors.

If anchor selection is noisy, subtree-first entry may over-focus on the wrong branch.
This fallback plan avoids that by letting the document tree search reason over the full structure from the beginning.

It is a useful baseline precisely because it stays close to the public PageIndex idea of:

- first choose the document
- then search the document tree

## Strengths

- Simpler implementation
- Lower risk of anchor-induced search bias
- Closer to the public PageIndex-style "one doc as one entry" pattern
- Easier to debug and explain

## Risks

- More expensive per selected document
- Harder to scale when many documents are selected
- Slower when a long report contains many partially related sections
- Weaker localization from the global layer

## Safety Rails

- Keep the top-N document count conservative
- Add strong metadata filters before tree search
- Prefer documents with multiple global hits over weak single-hit documents
- Use caching for loaded trees and section summaries

## Recommended Use

Use this plan when:

- you want the simplest reliable baseline
- anchor quality is not yet trusted
- the number of documents passed to stage two is still small
- you want a PageIndex-style whole-tree reasoning entry

## Recommendation

This should be kept as the baseline fallback plan.

It is safer and simpler than the anchor-subtree approach, but it will likely cost more at retrieval time and may waste work inside very long reports.
