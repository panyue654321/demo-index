"""Integration tests for DemoIndex retrieval against PostgreSQL."""

from __future__ import annotations

import ast
import json
import os
import unittest
from io import StringIO
from unittest.mock import patch

from DemoIndex import expand_localized_sections, package_evidence, retrieve_candidates, retrieve_evidence, retrieve_tree_candidates
from DemoIndex.run import main as run_main


DEFAULT_DATABASE_URL = "postgresql://demoindex:demoindex@127.0.0.1:5432/demoindex"


def _load_one_embedding(database_url: str) -> list[float]:
    """Load one stored embedding vector from PostgreSQL for deterministic testing."""
    import psycopg

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT embedding::text FROM section_chunks ORDER BY chunk_id LIMIT 1")
            row = cursor.fetchone()
    if not row:
        raise unittest.SkipTest("section_chunks is empty; build the global index before running retrieval tests.")
    return [float(value) for value in ast.literal_eval(row[0])]


class _FakeEmbeddingClient:
    """Return one known embedding vector to avoid network calls in tests."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._vector = _load_one_embedding(os.environ["DATABASE_URL"])

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Return the same valid-dimension vector for every query."""
        return [self._vector for _ in texts]


class _FakeStage3ChatClient:
    """Return a deterministic Stage 3 rerank payload."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def completion(self, _model: str, prompt: str) -> str:
        """Select the first two shortlist entries from the prompt JSON payload."""
        shortlist_marker = "Shortlisted sections:\n"
        shortlist_payload = prompt.split(shortlist_marker, 1)[1]
        shortlist = json.loads(shortlist_payload)
        ranked_sections = [
            {"section_id": item["section_id"], "reason": "test rerank"}
            for item in shortlist[:2]
        ]
        return json.dumps({"ranked_sections": ranked_sections}, ensure_ascii=False)


class _FakeStage3And5ChatClient:
    """Return deterministic payloads for both Stage 3 and Stage 5 chat calls."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def completion(self, _model: str, prompt: str) -> str:
        """Inspect the prompt marker and return the corresponding JSON payload."""
        if "Shortlisted sections:\n" in prompt:
            shortlist_payload = prompt.split("Shortlisted sections:\n", 1)[1]
            shortlist = json.loads(shortlist_payload)
            ranked_sections = [
                {"section_id": item["section_id"], "reason": "test rerank"}
                for item in shortlist[:2]
            ]
            return json.dumps({"ranked_sections": ranked_sections}, ensure_ascii=False)
        shortlist_payload = prompt.split("Evidence shortlist:\n", 1)[1]
        shortlist = json.loads(shortlist_payload)
        labeled_evidence = [
            {
                "evidence_id": item["evidence_id"],
                "relationship_label": "related",
                "relationship_reason": "test relation",
            }
            for item in shortlist[:2]
        ]
        return json.dumps({"labeled_evidence": labeled_evidence}, ensure_ascii=False)


class RetrievalIntegrationTests(unittest.TestCase):
    """Exercise Stage 1 + Stage 2 retrieval against the local PostgreSQL."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare the local database URL or skip when PostgreSQL is unavailable."""
        cls.database_url = os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
        os.environ["DATABASE_URL"] = cls.database_url

    def test_retrieve_candidates_returns_ranked_results(self) -> None:
        """The retrieval API should return chunk, doc, and section results with lexical support."""
        with patch("DemoIndex.retrieval.DashScopeEmbeddingClient", _FakeEmbeddingClient):
            result = retrieve_candidates(
                "2024 全球手游 CPI 和留存趋势",
                top_k_dense=10,
                top_k_lexical=10,
                top_k_fused_chunks=12,
                top_k_docs=3,
                top_k_sections_per_doc=2,
                top_k_chunks_per_section=1,
                use_llm_parse=False,
            )
        payload = result.to_dict()
        self.assertTrue(payload["chunk_hits"])
        self.assertTrue(payload["doc_candidates"])
        self.assertTrue(payload["section_candidates"])
        self.assertIn("summary", payload["section_candidates"][0])
        self.assertGreater(payload["metadata"]["counts"]["lexical_hits"], 0)
        self.assertTrue(any(item["lexical_rank"] is not None for item in payload["chunk_hits"]))

    def test_retrieve_cli_prints_json(self) -> None:
        """The retrieve CLI should print structured JSON output."""
        stdout = StringIO()
        argv = [
            "DemoIndex.run",
            "retrieve",
            "--query",
            "混合休闲 合作伙伴",
            "--top-k-dense",
            "8",
            "--top-k-lexical",
            "8",
            "--top-k-fused-chunks",
            "10",
            "--top-k-docs",
            "2",
            "--top-k-sections-per-doc",
            "1",
            "--top-k-chunks-per-section",
            "1",
            "--disable-llm-parse",
        ]
        with (
            patch("DemoIndex.retrieval.DashScopeEmbeddingClient", _FakeEmbeddingClient),
            patch("sys.argv", argv),
            patch("sys.stdout", stdout),
        ):
            exit_code = run_main()
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["doc_candidates"])
        self.assertGreater(payload["metadata"]["counts"]["lexical_hits"], 0)

    def test_retrieve_tree_candidates_returns_localized_sections(self) -> None:
        """Stage 3 retrieval should return localized docs and sections."""
        with (
            patch("DemoIndex.retrieval.DashScopeEmbeddingClient", _FakeEmbeddingClient),
            patch("DemoIndex.retrieval.QwenChatClient") as mock_chat_cls,
        ):
            result = retrieve_tree_candidates(
                "2024 全球手游 CPI 和留存趋势",
                top_k_dense=10,
                top_k_lexical=10,
                top_k_fused_chunks=12,
                top_k_docs=2,
                top_k_sections_per_doc=2,
                top_k_chunks_per_section=1,
                stage3_mode="heuristic",
                top_k_tree_sections_per_doc=3,
                use_llm_parse=False,
            )
        mock_chat_cls.assert_not_called()
        payload = result.to_dict()
        self.assertTrue(payload["localized_docs"])
        self.assertTrue(payload["localized_sections"])
        self.assertEqual(payload["localized_docs"][0]["mode_used"], "heuristic")
        self.assertIn("stage3", payload["metadata"])

    def test_retrieve_tree_cli_prints_json(self) -> None:
        """The retrieve-tree CLI should print the enriched Stage 3 JSON envelope."""
        stdout = StringIO()
        argv = [
            "DemoIndex.run",
            "retrieve-tree",
            "--query",
            "2024 全球手游 CPI 和留存趋势",
            "--top-k-dense",
            "8",
            "--top-k-lexical",
            "8",
            "--top-k-fused-chunks",
            "10",
            "--top-k-docs",
            "2",
            "--top-k-sections-per-doc",
            "2",
            "--top-k-chunks-per-section",
            "1",
            "--top-k-tree-sections-per-doc",
            "3",
            "--stage3-mode",
            "hybrid",
            "--disable-llm-parse",
        ]
        with (
            patch("DemoIndex.retrieval.DashScopeEmbeddingClient", _FakeEmbeddingClient),
            patch("DemoIndex.retrieval.QwenChatClient", _FakeStage3ChatClient),
            patch("DemoIndex.retrieval.load_dashscope_api_key"),
            patch("sys.argv", argv),
            patch("sys.stdout", stdout),
        ):
            exit_code = run_main()
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["localized_docs"])
        self.assertTrue(payload["localized_sections"])
        self.assertIn(payload["localized_docs"][0]["mode_used"], {"heuristic", "hybrid"})

    def test_stage4_and_stage5_public_apis_return_evidence(self) -> None:
        """Stage 4 and Stage 5 public APIs should build expanded contexts and evidence items."""
        with (
            patch("DemoIndex.retrieval.DashScopeEmbeddingClient", _FakeEmbeddingClient),
            patch("DemoIndex.retrieval.QwenChatClient", _FakeStage3ChatClient),
        ):
            stage3_result = retrieve_tree_candidates(
                "2024 全球手游 CPI 和留存趋势",
                top_k_dense=10,
                top_k_lexical=10,
                top_k_fused_chunks=12,
                top_k_docs=2,
                top_k_sections_per_doc=2,
                top_k_chunks_per_section=1,
                stage3_mode="heuristic",
                top_k_tree_sections_per_doc=3,
                use_llm_parse=False,
            )
            stage4_result = expand_localized_sections(
                stage3_result,
                top_k_focus_sections_per_doc=2,
                max_ancestor_hops=2,
                max_descendant_depth=1,
                max_siblings_per_focus=1,
                chunk_neighbor_window=1,
                max_evidence_chunks_per_focus=4,
                context_char_budget=3000,
            )
            stage5_result = package_evidence(
                stage4_result,
                relation_mode="heuristic",
                top_k_evidence_per_doc=2,
                top_k_total_evidence=4,
            )
        self.assertTrue(stage4_result.expanded_contexts)
        self.assertTrue(stage5_result.evidence_items)
        self.assertIn("stage4", stage5_result.metadata)
        self.assertIn("stage5", stage5_result.metadata)

    def test_retrieve_evidence_returns_full_stage15_envelope(self) -> None:
        """The end-to-end evidence API should return packaged evidence items."""
        with (
            patch("DemoIndex.retrieval.DashScopeEmbeddingClient", _FakeEmbeddingClient),
            patch("DemoIndex.retrieval.QwenChatClient", _FakeStage3And5ChatClient),
        ):
            result = retrieve_evidence(
                "2024 全球手游 CPI 和留存趋势",
                top_k_dense=10,
                top_k_lexical=10,
                top_k_fused_chunks=12,
                top_k_docs=2,
                top_k_sections_per_doc=2,
                top_k_chunks_per_section=1,
                stage3_mode="hybrid",
                top_k_tree_sections_per_doc=3,
                use_llm_parse=False,
                stage5_relation_mode="hybrid",
                top_k_evidence_per_doc=2,
                top_k_total_evidence=4,
            )
        payload = result.to_dict()
        self.assertTrue(payload["expanded_contexts"])
        self.assertTrue(payload["evidence_docs"])
        self.assertTrue(payload["evidence_items"])
        self.assertIn("stage4", payload["metadata"])
        self.assertIn("stage5", payload["metadata"])

    def test_retrieve_evidence_cli_prints_json(self) -> None:
        """The retrieve-evidence CLI should print the full Stage 1 through Stage 5 JSON envelope."""
        stdout = StringIO()
        argv = [
            "DemoIndex.run",
            "retrieve-evidence",
            "--query",
            "2024 全球手游 CPI 和留存趋势",
            "--top-k-dense",
            "8",
            "--top-k-lexical",
            "8",
            "--top-k-fused-chunks",
            "10",
            "--top-k-docs",
            "2",
            "--top-k-sections-per-doc",
            "2",
            "--top-k-chunks-per-section",
            "1",
            "--top-k-tree-sections-per-doc",
            "3",
            "--top-k-focus-sections-per-doc",
            "2",
            "--top-k-evidence-per-doc",
            "2",
            "--top-k-total-evidence",
            "4",
            "--stage3-mode",
            "heuristic",
            "--stage5-relation-mode",
            "heuristic",
            "--disable-llm-parse",
        ]
        with (
            patch("DemoIndex.retrieval.DashScopeEmbeddingClient", _FakeEmbeddingClient),
            patch("DemoIndex.retrieval.QwenChatClient", _FakeStage3And5ChatClient),
            patch("sys.argv", argv),
            patch("sys.stdout", stdout),
        ):
            exit_code = run_main()
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["expanded_contexts"])
        self.assertTrue(payload["evidence_items"])
        self.assertIn("stage4", payload["metadata"])
        self.assertIn("stage5", payload["metadata"])


if __name__ == "__main__":
    unittest.main()
