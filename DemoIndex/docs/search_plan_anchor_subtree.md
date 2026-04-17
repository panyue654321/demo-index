# DemoIndex Search Plan A: Doc + Section Anchors + Subtree-First Search

## Goal

This plan is designed for a corpus of hundreds or thousands of long game-industry research reports.

The target problem is not just "find one similar paragraph", but:

- find the most relevant reports across the whole corpus
- enter the right place inside a report quickly
- preserve contradictory or conditional conclusions
- keep enough structural context for answer generation

## Design Summary

This plan uses two index layers:

- a global chunk vector index for cross-document recall
- a per-document tree index for reasoning-style navigation inside one report

This plan is intentionally built on top of two public PageIndex ideas:

- `Document Search by Semantics`: use chunks to score documents across a corpus
- `Hybrid Tree Search`: use chunks to score tree nodes inside one document

The extra step added by DemoIndex is explicit `section anchor` handoff.

The retrieval flow is:

1. understand the query and extract constraints
2. run global chunk retrieval across all reports
3. aggregate hits by `doc_id` and `section_id`
4. keep each selected document together with a small number of section anchors
5. enter the matched document tree from the anchor subtrees first
6. expand to ancestors, siblings, or the full tree only when needed
7. aggregate evidence, including agreement and disagreement across reports

In shorthand, the flow is:

- PageIndex multi-document pattern: `chunk -> doc`
- PageIndex single-document hybrid pattern: `chunk -> node`
- DemoIndex current plan: `chunk -> doc + section anchor -> subtree-first tree search`

## What Is Indexed

### Global Vector Index

Main indexed object:

- body chunks, not only summaries

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

The existing `document_sections` table remains the single-document structural index.

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
- domain-specific fields such as metrics, region, platform, and genre should only be filled by an optional external retrieval profile or by one fallback LLM parse when the query is genuinely sparse
- already-informative Chinese queries should usually skip query-time LLM enrichment

### Stage 2: Global Candidate Recall

Run:

- dense chunk ANN retrieval
- lexical retrieval over `title`, `title_path`, and `search_text`

Implementation note:

- lexical recall should stay PostgreSQL-native
- Chinese-heavy queries should use generic derived search terms plus weighted title/title-path/body hits
- `pg_trgm` should be used as a soft scoring signal, not as a strict whole-query gate

The goal of this stage is not to answer directly.
It is only to find:

- which reports are likely relevant
- which sections inside those reports are likely entry points

This stage is closest to the public PageIndex `Document Search by Semantics` flow, except that DemoIndex keeps `section_id` information instead of collapsing immediately to document-only state.

### Stage 3: Anchor Selection

Aggregate results by:

- `doc_id`
- `section_id`

For each selected document:

- keep the best 1-3 section anchors
- keep the best 1-2 chunks inside each anchor section

This is not for final answer display.
It is only for tree-entry localization.

This is the key difference from public PageIndex examples.
PageIndex clearly exposes `chunk -> doc` and `chunk -> node` scoring, but it does not publicly describe a `section-anchor-first` document entry policy.
DemoIndex adds that policy on purpose.

### Stage 4: Subtree-First Tree Search

For each selected document:

1. start from the matched anchor section
2. inspect the anchor subtree first
3. if evidence is weak or incomplete, expand to:
   ancestor chain
4. if still insufficient, expand to:
   sibling sections
5. if still insufficient, fall back to:
   whole-document tree search

The tree search unit remains the section node, not the chunk.

This stage still follows the PageIndex principle that the final retrieval unit is the tree node or section.
Chunks are only the evidence used to score and localize nodes.

### Stage 5: Evidence Aggregation

Group retrieved evidence by:

- claim
- metric
- region
- platform
- time period

Then preserve:

- supporting evidence
- conflicting evidence
- scope conditions

## Why This Plan Exists

This plan is meant to solve a specific problem:

- one report can be 40-100 pages
- many reports say similar things
- some reports disagree because of time range, sample, region, or methodology

If we only know that a report was hit, we still do not know where to enter the tree.
Section anchors reduce that ambiguity.

In other words, this plan does not replace the PageIndex hybrid idea.
It extends it with a stronger cross-document handoff rule for long, repetitive report corpora.

## Strengths

- Faster and more focused entry into a document tree
- Better localization when one report mentions the same topic in many sections
- Lower chance of missing contradictory evidence from other reports
- Better fit for large multi-report question answering

## Risks

- If the anchor is wrong, subtree-first search may enter the wrong branch
- More system complexity than doc-only entry
- Requires careful fallback rules to avoid over-trusting local hits

## Safety Rails

- Never stop at the first anchor subtree by default
- Always allow ancestor expansion and whole-tree fallback
- Use document-level confidence after anchor aggregation
- Keep contradictory evidence from different reports instead of collapsing them early

## Recommended Use

Use this plan when:

- the corpus is large
- reports are long
- the same topic appears in multiple places in the same report
- cross-report contradictions matter
- answer quality depends on precise evidence placement

## Recommendation

This should be the primary plan for DemoIndex.

It keeps the PageIndex-like section-based reasoning flow, but adds a stronger cross-document entry mechanism for large report corpora.
