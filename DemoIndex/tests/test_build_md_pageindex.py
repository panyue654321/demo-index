"""Unit tests for Markdown PageIndex building and unified build routing."""

from __future__ import annotations

import textwrap
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from DemoIndex.build_md_pageindex import (
    PageIndexOptions,
    build_forest_from_markdown,
    build_forest_page_per_page_with_doc_root,
    compute_line_count,
    normalize_display_title,
    parse_page_comments,
    sync_build_pageindex_payload,
    sync_build_pageindex_payload_from_lines,
)
from DemoIndex.pipeline import build_pageindex_tree


class MarkdownBuildTests(unittest.TestCase):
    """Cover Markdown builder helpers and unified build input routing."""

    def test_compute_line_count(self) -> None:
        """Markdown line counting should follow the standard newline-plus-one rule."""
        self.assertEqual(compute_line_count(""), 1)
        self.assertEqual(compute_line_count("a"), 1)
        self.assertEqual(compute_line_count("a\nb\nc"), 3)

    def test_parse_page_comments_tracks_page_switch_for_following_lines(self) -> None:
        """Page comments should affect subsequent lines without re-tagging the comment line."""
        lines = ["line0", "<!-- page:3 -->", "after break"]
        self.assertEqual(parse_page_comments(lines), [1, 1, 3])

    def test_normalize_display_title_cleans_prefix_and_year_spacing(self) -> None:
        """Display titles should remove the qianyan prefix and collapse year spacing."""
        self.assertEqual(normalize_display_title("前言：移动游戏", 2), "移动游戏")
        self.assertEqual(normalize_display_title("2025 年报告", 1), "2025年报告")

    def test_build_forest_from_markdown_creates_two_h1_roots(self) -> None:
        """The h1_forest layout should split the document into multiple H1-rooted trees."""
        markdown = textwrap.dedent(
            """\
            # Alpha

            intro

            ## Beta

            beta body

            <!-- page:2 -->

            # Gamma

            gamma only
            """
        )
        lines = markdown.strip().split("\n")
        forest = build_forest_from_markdown(lines, parse_page_comments(lines))
        self.assertEqual(len(forest), 2)
        self.assertEqual(forest[0]["title"], "Alpha")
        self.assertEqual(forest[0]["nodes"][0]["title"], "Beta")
        self.assertEqual(forest[1]["title"], "Gamma")

    def test_build_forest_page_per_page_with_doc_root_creates_page_children(self) -> None:
        """The page_per_page layout should create one doc root plus one child per logical page."""
        markdown = textwrap.dedent(
            """\
            <!-- page:1 -->
            # RootTitle

            intro

            <!-- page:2 -->
            ## PageTwoOnly
            body
            """
        ).strip()
        lines = markdown.split("\n")
        forest = build_forest_page_per_page_with_doc_root(lines, parse_page_comments(lines))
        self.assertEqual(len(forest), 1)
        root = forest[0]
        self.assertEqual(root["title"], "RootTitle")
        self.assertEqual(len(root["nodes"]), 2)
        self.assertEqual(root["nodes"][0]["title"], "RootTitle")
        self.assertEqual(root["nodes"][1]["title"], "PageTwoOnly")

    def test_sync_build_pageindex_payload_from_lines_supports_page_per_page_layout(self) -> None:
        """The line-based builder should preserve page-per-page output shape."""
        lines = [
            "<!-- page:1 -->",
            "# OnlyH1",
            "",
            "x",
            "<!-- page:2 -->",
            "## PageTwo",
            "no heading line",
        ]
        payload = sync_build_pageindex_payload_from_lines(
            lines,
            PageIndexOptions(if_add_summary=False, doc_id="00000000-0000-0000-0000-000000000001"),
            llm_factory=None,
            layout="page_per_page",
        )
        self.assertEqual(len(payload["result"]), 1)
        root = payload["result"][0]
        self.assertEqual(root["title"], "OnlyH1")
        self.assertEqual(len(root["nodes"]), 2)
        self.assertEqual(root["nodes"][1]["title"], "PageTwo")

    def test_sync_build_pageindex_payload_summary_off_clears_summaries(self) -> None:
        """Turning summaries off should clear both doc and node summaries."""
        with self.subTest("summary_off"):
            tmp_path = Path(self.id().replace(".", "_") + ".md")
            try:
                tmp_path.write_text("# One\n\nHello\n", encoding="utf-8")
                payload = sync_build_pageindex_payload(
                    tmp_path,
                    PageIndexOptions(if_add_summary=False, doc_id=str(uuid.uuid4())),
                    llm_factory=None,
                )
                self.assertEqual(payload["summary"], "")
                self.assertEqual(payload["result"][0]["summary"], "")
            finally:
                tmp_path.unlink(missing_ok=True)

    def test_sync_build_pageindex_payload_uses_heuristic_summary_without_llm(self) -> None:
        """Markdown payload building should produce summaries without an LLM when enabled."""
        tmp_path = Path(self.id().replace(".", "_") + ".md")
        try:
            tmp_path.write_text("# One\n\nHello world.\n\n## Sub\n\nShort.\n", encoding="utf-8")
            fixed_id = "11111111-1111-1111-1111-111111111111"
            payload = sync_build_pageindex_payload(
                tmp_path,
                PageIndexOptions(if_add_summary=True, summary_char_threshold=600, doc_id=fixed_id),
                llm_factory=None,
            )
            self.assertEqual(payload["doc_id"], fixed_id)
            self.assertTrue(payload["summary"])
            self.assertIn("One", payload["summary"])
            self.assertTrue(payload["result"][0]["summary"])
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_build_pageindex_tree_routes_markdown_input_and_auto_layout(self) -> None:
        """The unified build entrypoint should route Markdown files through the Markdown builder."""
        tmp_path = Path(self.id().replace(".", "_") + ".md")
        try:
            tmp_path.write_text(
                textwrap.dedent(
                    """\
                    <!-- page:1 -->
                    # RootTitle

                    intro

                    <!-- page:2 -->
                    ## PageTwo
                    body
                    """
                ),
                encoding="utf-8",
            )
            result = build_pageindex_tree(input_path=str(tmp_path), markdown_layout="auto")
            self.assertTrue(result["doc_id"])
            self.assertGreaterEqual(result["line_count"], 1)
            self.assertEqual(len(result["result"]), 1)
            self.assertEqual(result["result"][0]["title"], "RootTitle")
            self.assertEqual(len(result["result"][0]["nodes"]), 2)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_build_pageindex_tree_routes_pdf_input_to_pdf_builder(self) -> None:
        """The unified build entrypoint should continue to route PDF files to the PDF builder."""
        tmp_path = Path(self.id().replace(".", "_") + ".pdf")
        try:
            tmp_path.write_bytes(b"%PDF-1.4\n")
            expected = {"doc_id": "doc-1", "status": "completed", "retrieval_ready": False, "result": []}
            with patch("DemoIndex.pipeline._build_pdf_output", return_value=expected) as mock_pdf_builder:
                result = build_pageindex_tree(input_path=str(tmp_path))
            self.assertEqual(result, expected)
            mock_pdf_builder.assert_called_once()
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_build_pageindex_tree_rejects_multiple_input_arguments(self) -> None:
        """The unified build entrypoint should reject simultaneous input_path and pdf_path usage."""
        tmp_path = Path(self.id().replace(".", "_") + ".pdf")
        try:
            tmp_path.write_bytes(b"%PDF-1.4\n")
            with self.assertRaisesRegex(ValueError, "Only one of input_path or pdf_path"):
                build_pageindex_tree(input_path=str(tmp_path), pdf_path=str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
