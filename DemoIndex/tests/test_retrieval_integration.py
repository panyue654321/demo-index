"""Integration tests for DemoIndex retrieval against PostgreSQL."""

from __future__ import annotations

import ast
import json
import os
import unittest
from io import StringIO
from unittest.mock import patch

from DemoIndex import retrieve_candidates
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


if __name__ == "__main__":
    unittest.main()
