"""Unit tests for DemoIndex retrieval logic."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from DemoIndex.retrieval import (
    ContextChunk,
    ContextSection,
    DocCandidate,
    EvidenceItem,
    ExpandedContext,
    ExpandedDoc,
    LocalizedSection,
    QueryUnderstanding,
    RetrievalStage12Result,
    RetrievalChunkHit,
    SectionCandidate,
    _Stage4ChunkRow,
    _assemble_stage4_answer_context_text,
    _Stage3TreeSection,
    _aggregate_candidates,
    _build_stage3_candidate_pool,
    _build_stage4_context_sections,
    _build_evidence_items,
    _derive_search_terms,
    _fuse_chunk_hits,
    _label_evidence_items_with_llm,
    _rerank_stage3_sections_with_llm,
    _score_stage3_candidates,
    _select_stage4_evidence_chunks,
    localize_sections,
    parse_query,
)


class RetrievalUnitTests(unittest.TestCase):
    """Cover parsing, fusion, aggregation, and Stage 3 helpers."""

    def test_parse_query_rule_based_mixed_language(self) -> None:
        """Generic parsing should extract language, intent, terms, and time scope."""
        result = parse_query("2024 全球手游 CPI 和 retention 趋势", use_llm=False)
        self.assertEqual(result.language, "mixed")
        self.assertEqual(result.intent, "trend")
        self.assertIn("全球手游", result.terms)
        self.assertIn("CPI", result.terms)
        self.assertIn(2024, result.time_scope["years"])
        self.assertEqual(result.metrics, [])
        self.assertEqual(result.regions, [])
        self.assertFalse(result.llm_enriched)

    def test_parse_query_llm_failure_falls_back(self) -> None:
        """LLM enrichment errors should not break the rule-based parse."""
        with patch("DemoIndex.retrieval._enrich_query_with_llm", side_effect=RuntimeError("boom")):
            result = parse_query("留存率", use_llm=True)
        self.assertEqual(result.intent, "general")
        self.assertEqual(result.terms, ["留存率", "留存", "存率"])
        self.assertEqual(result.metrics, [])
        self.assertEqual(result.genres, [])
        self.assertFalse(result.llm_enriched)

    def test_parse_query_can_use_external_profile_aliases(self) -> None:
        """Optional external retrieval profiles should populate domain fields."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(
                {
                    "metrics": {"Retention": ["留存", "留存率"]},
                    "platforms": {"Mobile": ["手游"]},
                },
                handle,
                ensure_ascii=False,
            )
            profile_path = handle.name
        try:
            with patch.dict(os.environ, {"DEMOINDEX_RETRIEVAL_PROFILE_PATH": profile_path}, clear=False):
                result = parse_query("手游留存率", use_llm=False)
            self.assertIn("Retention", result.metrics)
            self.assertIn("Mobile", result.platforms)
        finally:
            os.unlink(profile_path)

    def test_parse_query_skips_llm_for_informative_query(self) -> None:
        """Informative generic queries should not trigger query-time LLM enrichment."""
        with patch("DemoIndex.retrieval._enrich_query_with_llm") as mock_enrich:
            result = parse_query("2024 全球手游 CPI 和留存趋势", use_llm=True)
        mock_enrich.assert_not_called()
        self.assertFalse(result.llm_enriched)
        self.assertIn(2024, result.time_scope["years"])

    def test_rrf_fusion_merges_dense_and_lexical_hits(self) -> None:
        """RRF fusion should preserve both branches and reward overlap."""
        dense_hits = [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "section_id": "s1",
                "node_id": "n1",
                "title": "T1",
                "title_path": "A > T1",
                "page_index": 1,
                "chunk_index": 0,
                "chunk_text": "alpha",
                "dense_rank": 1,
                "dense_score": 0.99,
            },
            {
                "chunk_id": "c2",
                "doc_id": "d2",
                "section_id": "s2",
                "node_id": "n2",
                "title": "T2",
                "title_path": "B > T2",
                "page_index": 2,
                "chunk_index": 0,
                "chunk_text": "beta",
                "dense_rank": 2,
                "dense_score": 0.85,
            },
        ]
        lexical_hits = [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "section_id": "s1",
                "node_id": "n1",
                "title": "T1",
                "title_path": "A > T1",
                "page_index": 1,
                "chunk_index": 0,
                "chunk_text": "alpha",
                "lexical_rank": 1,
                "lexical_score": 0.77,
            },
            {
                "chunk_id": "c3",
                "doc_id": "d3",
                "section_id": "s3",
                "node_id": "n3",
                "title": "T3",
                "title_path": "C > T3",
                "page_index": 3,
                "chunk_index": 0,
                "chunk_text": "gamma",
                "lexical_rank": 2,
                "lexical_score": 0.55,
            },
        ]
        fused = _fuse_chunk_hits(
            dense_hits=dense_hits,
            lexical_hits=lexical_hits,
            top_k_fused_chunks=10,
            rrf_k=60,
        )
        self.assertEqual(fused[0].chunk_id, "c1")
        self.assertIsNotNone(fused[0].dense_rank)
        self.assertIsNotNone(fused[0].lexical_rank)
        self.assertEqual({item.chunk_id for item in fused}, {"c1", "c2", "c3"})

    def test_derive_search_terms_keeps_chinese_subterms(self) -> None:
        """Lexical search terms should retain useful Chinese fragments and ASCII tokens."""
        understanding = QueryUnderstanding(
            raw_query="2024 全球手游 CPI 和留存趋势",
            normalized_query="2024 全球手游 CPI 和留存趋势",
            language="mixed",
            intent="trend",
            terms=["全球手游", "CPI", "留存趋势"],
            metrics=[],
            regions=[],
            platforms=[],
            genres=[],
            time_scope={"years": [2024], "quarters": [], "raw_mentions": ["2024"]},
            llm_enriched=False,
        )
        search_terms = _derive_search_terms(understanding)
        self.assertIn("2024", search_terms)
        self.assertIn("cpi", search_terms)
        self.assertIn("全球", search_terms)
        self.assertIn("手游", search_terms)
        self.assertIn("留存", search_terms)

    def test_doc_and_section_aggregation_limits(self) -> None:
        """Aggregation should limit sections and supporting chunks per doc."""
        fused_hits = [
            RetrievalChunkHit("c1", "d1", "s1", "n1", "T1", "P > T1", 1, 0, "a", 1, 0.9, 1, 0.8, 0.03),
            RetrievalChunkHit("c2", "d1", "s1", "n1", "T1", "P > T1", 1, 1, "b", 2, 0.8, None, None, 0.02),
            RetrievalChunkHit("c3", "d1", "s2", "n2", "T2", "P > T2", 2, 0, "c", None, None, 2, 0.7, 0.015),
            RetrievalChunkHit("c4", "d2", "s3", "n3", "T3", "Q > T3", 3, 0, "d", 3, 0.7, 3, 0.6, 0.014),
        ]
        metadata = {
            "s1": {"section_id": "s1", "node_id": "n1", "title": "T1", "depth": 1, "summary": "S1"},
            "s2": {"section_id": "s2", "node_id": "n2", "title": "T2", "depth": 2, "summary": "S2"},
            "s3": {"section_id": "s3", "node_id": "n3", "title": "T3", "depth": 1, "summary": "S3"},
        }
        with patch("DemoIndex.retrieval._load_section_metadata", return_value=metadata):
            docs, sections = _aggregate_candidates(
                fused_hits=fused_hits,
                database_url="postgresql://unused",
                top_k_docs=1,
                top_k_sections_per_doc=1,
                top_k_chunks_per_section=1,
                doc_score_chunk_limit=5,
                section_score_chunk_limit=3,
            )
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].doc_id, "d1")
        self.assertEqual(len(docs[0].section_candidates), 1)
        self.assertEqual(len(docs[0].section_candidates[0].supporting_chunks), 1)
        self.assertEqual(len(sections), 1)

    def test_stage3_candidate_pool_expands_anchor_relations(self) -> None:
        """Stage 3 candidate-pool expansion should include tree relations around anchors."""
        doc_sections = {
            "root": _Stage3TreeSection("root", None, "d1", "n0", "Root", 0, "root summary", "Root"),
            "s1": _Stage3TreeSection("s1", "root", "d1", "n1", "Anchor", 1, "anchor summary", "Root > Anchor"),
            "s2": _Stage3TreeSection("s2", "s1", "d1", "n2", "Child", 2, "child summary", "Root > Anchor > Child"),
            "s3": _Stage3TreeSection("s3", "root", "d1", "n3", "Sibling", 1, "sibling summary", "Root > Sibling"),
        }
        children_map = {
            None: ["root"],
            "root": ["s1", "s3"],
            "s1": ["s2"],
        }
        anchor_sections = [
            SectionCandidate(
                doc_id="d1",
                section_id="s1",
                node_id="n1",
                title="Anchor",
                depth=1,
                summary="anchor summary",
                section_score=0.03,
                matched_chunk_count=1,
                supporting_chunks=[{"chunk_id": "c1"}],
            )
        ]
        stage2_lookup = {("d1", "s1"): anchor_sections[0]}
        candidate_pool, used_whole_doc_fallback = _build_stage3_candidate_pool(
            doc_id="d1",
            doc_sections=doc_sections,
            children_map=children_map,
            anchor_sections=anchor_sections,
            stage2_section_lookup=stage2_lookup,
            top_k_tree_sections_per_doc=5,
            whole_doc_fallback=False,
        )
        self.assertFalse(used_whole_doc_fallback)
        self.assertEqual(candidate_pool["s1"]["relation_to_anchor"], "anchor")
        self.assertEqual(candidate_pool["s2"]["relation_to_anchor"], "descendant")
        self.assertEqual(candidate_pool["root"]["relation_to_anchor"], "ancestor")
        self.assertEqual(candidate_pool["s3"]["relation_to_anchor"], "sibling")

    def test_stage3_heuristic_prefers_anchor_over_doc_fallback(self) -> None:
        """Heuristic localization should keep strong anchor matches above fallback nodes."""
        understanding = QueryUnderstanding(
            raw_query="2024 CPI 趋势",
            normalized_query="2024 CPI 趋势",
            language="mixed",
            intent="trend",
            terms=["2024", "CPI"],
            metrics=[],
            regions=[],
            platforms=[],
            genres=[],
            time_scope={"years": [2024], "quarters": [], "raw_mentions": ["2024"]},
            llm_enriched=False,
        )
        candidate_pool = {
            "anchor": {
                "section": _Stage3TreeSection("anchor", None, "d1", "n1", "2024 CPI", 1, "CPI summary", "2024 CPI"),
                "anchor_section_id": "anchor",
                "relation_to_anchor": "anchor",
                "reason_codes": ["relation:anchor"],
                "stage2_section_score": 0.03,
                "supporting_chunks": [{"chunk_id": "c1"}],
            },
            "fallback": {
                "section": _Stage3TreeSection("fallback", None, "d1", "n2", "Overview", 1, "generic summary", "Overview"),
                "anchor_section_id": None,
                "relation_to_anchor": "doc_fallback",
                "reason_codes": ["relation:doc_fallback"],
                "stage2_section_score": 0.0,
                "supporting_chunks": [],
            },
        }
        localized = _score_stage3_candidates(
            query_understanding=understanding,
            search_terms=_derive_search_terms(understanding),
            candidate_pool=candidate_pool,
            top_k_tree_sections_per_doc=5,
            stage3_relation_priors={
                "anchor": 4.0,
                "descendant": 2.75,
                "ancestor": 2.1,
                "sibling": 1.45,
                "doc_fallback": 0.55,
            },
        )
        self.assertEqual(localized[0].section_id, "anchor")
        self.assertGreater(localized[0].localization_score, localized[1].localization_score)

    def test_stage3_hybrid_rerank_accepts_valid_ids_and_ignores_invalid(self) -> None:
        """Hybrid rerank should reorder valid shortlist ids and ignore invalid ids."""
        shortlisted = [
            LocalizedSection("d1", "s1", "n1", None, "A", 1, "summary a", "Root > A", 4.0, 0.03, "s1", "anchor", ["relation:anchor"], [{"chunk_id": "c1"}]),
            LocalizedSection("d1", "s2", "n2", None, "B", 1, "summary b", "Root > B", 3.0, 0.0, "s1", "sibling", ["relation:sibling"], [{"chunk_id": "c1"}]),
        ]

        class _FakeChatClient:
            def completion(self, _model: str, _prompt: str) -> str:
                return json.dumps(
                    {
                        "ranked_sections": [
                            {"section_id": "s2", "reason": "more direct"},
                            {"section_id": "missing", "reason": "ignored"},
                        ]
                    },
                    ensure_ascii=False,
                )

        reranked = _rerank_stage3_sections_with_llm(
            query_understanding=QueryUnderstanding(
                raw_query="query",
                normalized_query="query",
                language="en",
                intent="general",
                terms=["query"],
                metrics=[],
                regions=[],
                platforms=[],
                genres=[],
                time_scope={"years": [], "quarters": [], "raw_mentions": []},
                llm_enriched=False,
            ),
            anchor_sections=[],
            shortlisted_sections=shortlisted,
            top_k_tree_sections_per_doc=5,
            llm_client=_FakeChatClient(),
            rerank_model="dashscope/qwen3.6-plus",
            debug_recorder=None,
            doc_id="d1",
        )
        self.assertIsNotNone(reranked)
        self.assertEqual(reranked[0].section_id, "s2")
        self.assertIn("llm_rerank_selected", reranked[0].reason_codes)

    def test_stage3_hybrid_falls_back_when_rerank_fails(self) -> None:
        """Stage 3 should fall back to heuristic when hybrid rerank fails."""
        stage12 = RetrievalStage12Result(
            query_understanding=QueryUnderstanding(
                raw_query="2024 CPI 趋势",
                normalized_query="2024 CPI 趋势",
                language="mixed",
                intent="trend",
                terms=["2024", "CPI"],
                metrics=[],
                regions=[],
                platforms=[],
                genres=[],
                time_scope={"years": [2024], "quarters": [], "raw_mentions": ["2024"]},
                llm_enriched=False,
            ),
            chunk_hits=[],
            doc_candidates=[
                DocCandidate(
                    doc_id="d1",
                    doc_score=0.1,
                    matched_chunk_count=1,
                    matched_section_count=1,
                    top_section_ids=["s1"],
                    section_candidates=[
                        SectionCandidate(
                            doc_id="d1",
                            section_id="s1",
                            node_id="n1",
                            title="Anchor",
                            depth=1,
                            summary="anchor summary",
                            section_score=0.03,
                            matched_chunk_count=1,
                            supporting_chunks=[{"chunk_id": "c1"}],
                        )
                    ],
                )
            ],
            section_candidates=[
                SectionCandidate(
                    doc_id="d1",
                    section_id="s1",
                    node_id="n1",
                    title="Anchor",
                    depth=1,
                    summary="anchor summary",
                    section_score=0.03,
                    matched_chunk_count=1,
                    supporting_chunks=[{"chunk_id": "c1"}],
                )
            ],
            metadata={"counts": {}, "settings": {}},
        )
        tree_sections = {
            "d1": {
                "root": _Stage3TreeSection("root", None, "d1", "n0", "Root", 0, "root summary", "Root"),
                "s1": _Stage3TreeSection("s1", "root", "d1", "n1", "Anchor", 1, "anchor summary", "Root > Anchor"),
            }
        }
        children_map = {"d1": {None: ["root"], "root": ["s1"]}}

        with (
            patch("DemoIndex.retrieval._load_tree_sections_for_docs", return_value=(tree_sections, children_map)),
            patch("DemoIndex.retrieval.resolve_database_url", return_value="postgresql://unused"),
            patch("DemoIndex.retrieval.load_dashscope_api_key"),
            patch("DemoIndex.retrieval.QwenChatClient") as mock_client_cls,
        ):
            mock_client_cls.return_value.completion.side_effect = RuntimeError("boom")
            result = localize_sections(stage12, mode="hybrid", top_k_tree_sections_per_doc=2)
        self.assertTrue(result.localized_docs)
        self.assertEqual(result.localized_docs[0].mode_used, "heuristic")
        self.assertTrue(result.localized_sections)

    def test_stage4_context_expansion_includes_tree_relations_and_neighbors(self) -> None:
        """Stage 4 helpers should include ancestors, descendants, siblings, and neighboring chunks."""
        focus = LocalizedSection(
            doc_id="d1",
            section_id="focus",
            node_id="n1",
            parent_id="root",
            title="Focus",
            depth=1,
            summary="focus summary with query",
            title_path="Root > Focus",
            localization_score=5.0,
            stage2_section_score=0.03,
            anchor_section_id="focus",
            relation_to_anchor="anchor",
            reason_codes=["relation:anchor"],
            supporting_chunks=[{"chunk_id": "c2", "chunk_index": 1, "page_index": 2, "chunk_text": "support query chunk"}],
        )
        doc_sections = {
            "root": _Stage3TreeSection("root", None, "d1", "n0", "Root", 0, "root summary", "Root"),
            "focus": _Stage3TreeSection("focus", "root", "d1", "n1", "Focus", 1, "focus summary with query", "Root > Focus"),
            "child": _Stage3TreeSection("child", "focus", "d1", "n2", "Child", 2, "child summary query", "Root > Focus > Child"),
            "sib1": _Stage3TreeSection("sib1", "root", "d1", "n3", "Query Sibling", 1, "sibling summary", "Root > Query Sibling"),
            "sib2": _Stage3TreeSection("sib2", "root", "d1", "n4", "Other Sibling", 1, "other", "Root > Other Sibling"),
        }
        children_map = {
            None: ["root"],
            "root": ["focus", "sib1", "sib2"],
            "focus": ["child"],
        }
        context_sections = _build_stage4_context_sections(
            focus_section=focus,
            doc_sections=doc_sections,
            children_map=children_map,
            search_terms=["query"],
            years=[],
            quarters=[],
            max_ancestor_hops=2,
            max_descendant_depth=1,
            max_siblings_per_focus=1,
        )
        self.assertEqual([item.role for item in context_sections], ["ancestor", "focus", "descendant", "sibling"])
        self.assertEqual(context_sections[-1].section_id, "sib1")

        focus_chunks = [
            _Stage4ChunkRow("c1", "d1", "focus", "n1", "Focus", "Root > Focus", 2, 0, "neighbor before"),
            _Stage4ChunkRow("c2", "d1", "focus", "n1", "Focus", "Root > Focus", 2, 1, "support query chunk"),
            _Stage4ChunkRow("c3", "d1", "focus", "n1", "Focus", "Root > Focus", 3, 2, "neighbor after"),
        ]
        evidence_chunks = _select_stage4_evidence_chunks(
            focus_section=focus,
            focus_chunks=focus_chunks,
            chunk_neighbor_window=1,
            max_evidence_chunks_per_focus=3,
        )
        self.assertEqual([item.role for item in evidence_chunks], ["supporting", "neighbor", "neighbor"])
        self.assertEqual({item.chunk_id for item in evidence_chunks}, {"c1", "c2", "c3"})

    def test_stage4_answer_context_truncation_preserves_focus_summary(self) -> None:
        """Stage 4 truncation should preserve the focus summary while dropping lower-priority parts first."""
        text = _assemble_stage4_answer_context_text(
            title_path="Root > Focus",
            context_sections=[
                ContextSection("a1", "na1", "Ancestor", 0, "Root", "ancestor summary", "ancestor"),
                ContextSection("f1", "nf1", "Focus", 1, "Root > Focus", "focus summary must stay", "focus"),
                ContextSection("s1", "ns1", "Sibling", 1, "Root > Sibling", "sibling summary should drop", "sibling"),
            ],
            evidence_chunks=[
                ContextChunk("c1", "f1", 0, 2, "supporting chunk should stay if possible", "supporting"),
                ContextChunk("c2", "f1", 1, 2, "neighbor chunk should drop first", "neighbor"),
            ],
            context_char_budget=140,
        )
        self.assertIn("focus summary must stay", text)
        self.assertNotIn("neighbor chunk should drop first", text)

    def test_stage5_packaging_deduplicates_and_applies_caps(self) -> None:
        """Stage 5 heuristic packaging should deduplicate focus sections and honor caps."""
        expanded_docs = [
            ExpandedDoc(
                doc_id="d1",
                doc_score=1.0,
                expanded_contexts=[
                    ExpandedContext(
                        doc_id="d1",
                        focus_section_id="s1",
                        focus_node_id="n1",
                        focus_title="Focus A",
                        focus_localization_score=4.0,
                        context_sections=[
                            ContextSection("s1", "n1", "Focus A", 1, "Root > Focus A", "summary alpha", "focus")
                        ],
                        evidence_chunks=[
                            ContextChunk("c1", "s1", 0, 1, "alpha", "supporting"),
                            ContextChunk("c2", "s1", 1, 2, "beta", "neighbor"),
                        ],
                        answer_context_text="context a",
                    ),
                    ExpandedContext(
                        doc_id="d1",
                        focus_section_id="s1",
                        focus_node_id="n1",
                        focus_title="Focus A duplicate",
                        focus_localization_score=3.0,
                        context_sections=[
                            ContextSection("s1", "n1", "Focus A", 1, "Root > Focus A", "summary alpha", "focus")
                        ],
                        evidence_chunks=[],
                        answer_context_text="context a duplicate",
                    ),
                ],
            ),
            ExpandedDoc(
                doc_id="d2",
                doc_score=0.8,
                expanded_contexts=[
                    ExpandedContext(
                        doc_id="d2",
                        focus_section_id="s2",
                        focus_node_id="n2",
                        focus_title="Focus B",
                        focus_localization_score=2.0,
                        context_sections=[
                            ContextSection("s2", "n2", "Focus B", 1, "Root > Focus B", "summary beta", "focus")
                        ],
                        evidence_chunks=[ContextChunk("c3", "s2", 0, 3, "gamma", "supporting")],
                        answer_context_text="context b",
                    )
                ],
            ),
        ]
        evidence_items, evidence_docs = _build_evidence_items(
            query_understanding=QueryUnderstanding(
                raw_query="alpha",
                normalized_query="alpha",
                language="en",
                intent="general",
                terms=["alpha"],
                metrics=[],
                regions=[],
                platforms=[],
                genres=[],
                time_scope={"years": [], "quarters": [], "raw_mentions": []},
                llm_enriched=False,
            ),
            expanded_docs=expanded_docs,
            top_k_evidence_per_doc=1,
            top_k_total_evidence=1,
        )
        self.assertEqual(len(evidence_items), 1)
        self.assertEqual(len(evidence_docs), 1)
        self.assertEqual(evidence_items[0].relationship_label, "unlabeled")
        self.assertEqual(evidence_items[0].page_indexes, [1, 2])
        self.assertEqual(evidence_items[0].supporting_chunk_ids, ["c1"])

    def test_stage5_hybrid_labeling_accepts_valid_ids_and_ignores_invalid(self) -> None:
        """Stage 5 hybrid labeling should only apply labels to known shortlisted evidence IDs."""
        evidence_items = [
            EvidenceItem(
                evidence_id="e1",
                doc_id="d1",
                focus_section_id="s1",
                focus_node_id="n1",
                title="Focus A",
                title_path="Root > Focus A",
                evidence_score=3.0,
                page_indexes=[1],
                supporting_chunk_ids=["c1"],
                context_section_ids=["s1"],
                answer_context_text="context a",
                relationship_label="unlabeled",
                relationship_reason="",
            ),
            EvidenceItem(
                evidence_id="e2",
                doc_id="d2",
                focus_section_id="s2",
                focus_node_id="n2",
                title="Focus B",
                title_path="Root > Focus B",
                evidence_score=2.0,
                page_indexes=[2],
                supporting_chunk_ids=["c2"],
                context_section_ids=["s2"],
                answer_context_text="context b",
                relationship_label="unlabeled",
                relationship_reason="",
            ),
        ]

        class _FakeStage5ChatClient:
            def completion(self, _model: str, _prompt: str) -> str:
                return json.dumps(
                    {
                        "labeled_evidence": [
                            {"evidence_id": "e1", "relationship_label": "supports", "relationship_reason": "fits"},
                            {"evidence_id": "missing", "relationship_label": "conflicts", "relationship_reason": "ignore"},
                        ]
                    },
                    ensure_ascii=False,
                )

        labeled = _label_evidence_items_with_llm(
            query_understanding=QueryUnderstanding(
                raw_query="query",
                normalized_query="query",
                language="en",
                intent="general",
                terms=["query"],
                metrics=[],
                regions=[],
                platforms=[],
                genres=[],
                time_scope={"years": [], "quarters": [], "raw_mentions": []},
                llm_enriched=False,
            ),
            evidence_items=evidence_items,
            relation_shortlist_size=2,
            llm_client=_FakeStage5ChatClient(),
            relation_model="dashscope/qwen3.6-plus",
            debug_recorder=None,
        )
        self.assertEqual(labeled[0].relationship_label, "supports")
        self.assertEqual(labeled[0].relationship_reason, "fits")
        self.assertEqual(labeled[1].relationship_label, "unlabeled")


if __name__ == "__main__":
    unittest.main()
