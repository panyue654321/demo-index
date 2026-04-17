"""Tests for ``md_combined_pageindex`` (markdown forest, page comments, payloads)."""

from __future__ import annotations

import textwrap
import uuid
from pathlib import Path

import pytest

from md_combined_pageindex import (
    PageIndexOptions,
    assign_node_ids_preorder,
    build_forest_from_markdown,
    build_tree_from_flat_nodes,
    compute_line_count,
    find_h1_line_indices,
    iter_atx_headers,
    normalize_display_title,
    parse_page_comments,
    strip_internal_fields,
    sync_build_pageindex_payload,
    _heuristic_doc_summary,
    _heuristic_node_summary,
)


def test_compute_line_count() -> None:
    assert compute_line_count("") == 1
    assert compute_line_count("a") == 1
    assert compute_line_count("a\nb") == 2
    assert compute_line_count("a\nb\nc") == 3


def test_parse_page_comments_tracks_current_page() -> None:
    """分页行本身仍记「切换前」的页码，其后行使用新页码（与模块内文档一致）。"""
    lines = [
        "line0",
        "<!-- page:3 -->",
        "after break",
    ]
    pages = parse_page_comments(lines)
    assert pages == [1, 1, 3]


def test_iter_atx_headers_ignores_headers_inside_fenced_code() -> None:
    lines = textwrap.dedent(
        """\
        # Real

        ```text
        # Not a header
        ```

        ## Also real
        """
    ).splitlines()
    headers = iter_atx_headers(lines)
    assert [h["level"] for h in headers] == [1, 2]
    assert headers[0]["raw_title"] == "Real"
    assert headers[1]["raw_title"] == "Also real"


def test_find_h1_line_indices() -> None:
    lines = ["# A", "## b", "# C"].copy()
    h = iter_atx_headers(lines)
    assert find_h1_line_indices(h) == [0, 2]


def test_normalize_display_title_qianyan_and_year_spacing() -> None:
    assert normalize_display_title("前言：移动游戏", 2) == "移动游戏"
    assert normalize_display_title("2025 年报告", 1) == "2025年报告"


def test_build_tree_from_flat_nodes_nested() -> None:
    flat = [
        {"_line_idx": 1, "level": 2, "title": "H2", "page_index": 1, "text": "t2"},
        {"_line_idx": 2, "level": 3, "title": "H3", "page_index": 1, "text": "t3"},
        {"_line_idx": 3, "level": 2, "title": "H2b", "page_index": 1, "text": "t2b"},
    ]
    roots = build_tree_from_flat_nodes(flat)
    assert len(roots) == 2
    assert roots[0]["title"] == "H2"
    assert len(roots[0]["nodes"]) == 1
    assert roots[0]["nodes"][0]["title"] == "H3"
    assert roots[1]["title"] == "H2b"


def test_build_forest_from_markdown_two_h1_sections() -> None:
    md = textwrap.dedent(
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
    lines = md.strip().split("\n")
    forest = build_forest_from_markdown(lines, parse_page_comments(lines))
    assert len(forest) == 2
    assert forest[0]["title"] == "Alpha"
    assert forest[0]["text"].startswith("# Alpha")
    assert len(forest[0]["nodes"]) == 1
    assert forest[0]["nodes"][0]["title"] == "Beta"
    assert forest[1]["title"] == "Gamma"


def test_build_forest_from_markdown_raises_without_h1() -> None:
    lines = ["## Only h2", "text"]
    with pytest.raises(ValueError, match="No H1"):
        build_forest_from_markdown(lines, parse_page_comments(lines))


def test_assign_node_ids_preorder() -> None:
    md = textwrap.dedent(
        """\
        # R

        ## A

        ### A1

        ## B
        """
    )
    lines = md.strip().split("\n")
    forest = build_forest_from_markdown(lines, parse_page_comments(lines))
    assign_node_ids_preorder(forest)
    assert forest[0]["node_id"] == "0000"
    assert forest[0]["nodes"][0]["node_id"] == "0001"
    assert forest[0]["nodes"][0]["nodes"][0]["node_id"] == "0002"
    assert forest[0]["nodes"][1]["node_id"] == "0003"


def test_strip_internal_fields() -> None:
    tree = {
        "title": "t",
        "node_id": "0000",
        "page_index": 1,
        "text": "body",
        "summary": "s",
        "_line_idx": 9,
        "nodes": [],
    }
    out = strip_internal_fields(tree)
    assert out == {
        "title": "t",
        "node_id": "0000",
        "page_index": 1,
        "text": "body",
        "summary": "s",
    }
    assert "_line_idx" not in out


def test_heuristic_node_summary_short_uses_body() -> None:
    assert _heuristic_node_summary("T", "hello") == "hello"


def test_heuristic_node_summary_long_truncates() -> None:
    body = "x" * 400
    s = _heuristic_node_summary("T", body.replace("\n", " "), max_len=50)
    assert len(s) <= 50
    assert s.endswith("…")


def test_heuristic_doc_summary_lists_roots() -> None:
    forest = [
        {"title": "第一章"},
        {"title": "第二章"},
    ]
    s = _heuristic_doc_summary(forest, line_count=10)
    assert "10" in s
    assert "第一章" in s and "第二章" in s


def test_sync_build_pageindex_payload_summary_off_clears_nodes(tmp_path: Path) -> None:
    md = tmp_path / "x.md"
    md.write_text("# One\n\nHello\n", encoding="utf-8")
    opt = PageIndexOptions(if_add_summary=False, doc_id=str(uuid.uuid4()))
    payload = sync_build_pageindex_payload(md, opt, llm_factory=None)
    assert payload["summary"] == ""
    assert payload["result"][0]["summary"] == ""


def test_sync_build_pageindex_payload_summary_on_heuristic_without_llm(tmp_path: Path) -> None:
    md = tmp_path / "x.md"
    md.write_text("# One\n\nHello world.\n\n## Sub\n\nShort.\n", encoding="utf-8")
    fixed_id = "11111111-1111-1111-1111-111111111111"
    opt = PageIndexOptions(
        if_add_summary=True,
        summary_char_threshold=600,
        doc_id=fixed_id,
    )
    payload = sync_build_pageindex_payload(md, opt, llm_factory=None)
    assert payload["doc_id"] == fixed_id
    assert payload["summary"]
    assert "One" in payload["summary"]
    root = payload["result"][0]
    assert root["summary"]
    assert root["title"] == "One"
