"""Unit tests for ``retrieve`` (keyword search, tree navigation, page-index fetch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieve import (
    _SEMANTIC_INDEX_KEY,
    _character_hash_embedding,
    _extract_search_terms,
    _get_md_page_content,
    _get_pdf_page_content,
    _parse_pages,
    _score_node_for_terms,
    get_document,
    get_document_structure,
    get_page_content,
    index_document_semantic,
    normalize_document_payload,
    remove_fields,
    search_content_by_query,
    search_content_hybrid,
    search_content_semantic,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMBINED_JSON = (
    _REPO_ROOT
    / "docs/results_with_vlm/game2025report_v7/combined_document_output.json"
)


def test_remove_fields_strips_text_recursively() -> None:
    tree = [
        {
            "title": "A",
            "node_id": "0000",
            "text": "secret",
            "summary": "s",
            "nodes": [{"title": "B", "text": "inner", "nodes": []}],
        }
    ]
    out = remove_fields(tree, fields=["text"])
    assert out[0]["title"] == "A"
    assert "text" not in out[0]
    assert "text" not in out[0]["nodes"][0]


def test_normalize_document_payload_maps_result_to_structure() -> None:
    raw = {
        "doc_id": "abc",
        "result": [{"title": "R", "node_id": "0000", "page_index": 1, "text": "t"}],
        "line_count": 10,
        "summary": "doc sum",
        "status": "completed",
    }
    info = normalize_document_payload(raw, doc_name="My Doc", doc_description="d")
    assert info["doc_id"] == "abc"
    assert info["structure"] is raw["result"]
    assert info["line_count"] == 10
    assert info["type"] == "markdown"
    assert info["doc_name"] == "My Doc"
    assert info["doc_description"] == "d"


def test_parse_pages_comma_range_and_dedup() -> None:
    assert _parse_pages("12") == [12]
    assert _parse_pages("3,8") == [3, 8]
    assert _parse_pages("5-7") == [5, 6, 7]
    assert _parse_pages("2-4,3,1") == [1, 2, 3, 4]


def test_parse_pages_invalid_range_raises() -> None:
    with pytest.raises(ValueError, match="start must be"):
        _parse_pages("7-5")


def test_get_document_markdown() -> None:
    documents = {
        "d1": normalize_document_payload(
            {
                "doc_id": "d1",
                "result": [],
                "line_count": 522,
                "summary": "x",
            },
            doc_name="Combined",
        )
    }
    meta = json.loads(get_document(documents, "d1"))
    assert meta["doc_id"] == "d1"
    assert meta["type"] == "markdown"
    assert meta["line_count"] == 522
    assert meta["doc_name"] == "Combined"
    assert "page_count" not in meta


def test_get_document_pdf_page_count() -> None:
    documents = {
        "p1": {
            "type": "pdf",
            "page_count": 42,
            "doc_name": "pdf doc",
            "doc_description": "",
            "structure": [],
        }
    }
    meta = json.loads(get_document(documents, "p1"))
    assert meta["page_count"] == 42
    assert "line_count" not in meta


def test_get_document_missing() -> None:
    err = json.loads(get_document({}, "nope"))
    assert "error" in err


def test_get_md_page_content_interval_and_fields() -> None:
    doc_info = {
        "structure": [
            {
                "title": "Root",
                "node_id": "0000",
                "page_index": 1,
                "text": "r",
                "nodes": [
                    {
                        "title": "Child",
                        "node_id": "0001",
                        "page_index": 3,
                        "text": "c-text",
                        "nodes": [],
                    },
                    {
                        "title": "Also p3",
                        "node_id": "0002",
                        "page_index": 3,
                        "text": "d-text",
                        "nodes": [],
                    },
                ],
            }
        ]
    }
    hits = _get_md_page_content(doc_info, [2, 4])
    assert len(hits) == 2
    assert hits[0]["node_id"] == "0001"
    assert hits[0]["title"] == "Child"
    assert hits[0]["page"] == 3
    assert hits[0]["content"] == "c-text"
    assert hits[1]["node_id"] == "0002"
    assert hits[1]["content"] == "d-text"


def test_get_md_page_content_single_page() -> None:
    doc_info = {
        "structure": [
            {
                "title": "Only",
                "node_id": "0005",
                "page_index": 6,
                "text": "body",
                "nodes": [],
            }
        ]
    }
    hits = _get_md_page_content(doc_info, [6])
    assert len(hits) == 1
    assert hits[0]["page"] == 6
    assert hits[0]["node_id"] == "0005"
    assert hits[0]["title"] == "Only"


def test_get_pdf_page_content_cached_no_node_fields() -> None:
    doc_info = {
        "pages": [
            {"page": 1, "content": "A"},
            {"page": 2, "content": "B"},
        ]
    }
    rows = _get_pdf_page_content(doc_info, [2, 1])
    assert rows == [
        {"page": 2, "content": "B"},
        {"page": 1, "content": "A"},
    ]
    for r in rows:
        assert "node_id" not in r
        assert "title" not in r


def test_get_page_content_markdown_via_public_api() -> None:
    documents = {
        "x": normalize_document_payload(
            {
                "doc_id": "x",
                "result": [
                    {
                        "title": "N1",
                        "node_id": "0001",
                        "page_index": 10,
                        "text": "alpha",
                        "summary": "s1",
                        "nodes": [],
                    }
                ],
                "line_count": 100,
                "summary": "",
            }
        )
    }
    data = json.loads(get_page_content(documents, "x", "10"))
    assert len(data) == 1
    assert data[0]["node_id"] == "0001"
    assert data[0]["title"] == "N1"
    assert data[0]["page"] == 10
    assert data[0]["content"] == "alpha"


def test_get_page_content_pdf_only_page_and_content() -> None:
    documents = {
        "pdf1": {
            "type": "pdf",
            "pages": [{"page": 1, "content": "x"}],
            "structure": [],
        }
    }
    data = json.loads(get_page_content(documents, "pdf1", "1"))
    assert data == [{"page": 1, "content": "x"}]


def test_extract_search_terms_chinese_and_english() -> None:
    zh = _extract_search_terms("游戏, 销量", locale="zh")
    assert "游戏" in zh
    assert "销量" in zh

    eng = _extract_search_terms("RPG games downloads", locale="en")
    assert "rpg" in eng
    assert "games" in eng
    assert "downloads" in eng


def test_extract_search_terms_single_cjk_from_segment() -> None:
    terms = _extract_search_terms("美, 欧", locale="zh")
    assert "美" in terms
    assert "欧" in terms


def test_extract_search_terms_fallback_bigrams() -> None:
    """无显式有效词条时回退到 2 字滑窗。"""
    terms = _extract_search_terms("的的的了", locale="zh")
    assert len(terms) >= 1


def test_score_node_title_weight_higher_than_text() -> None:
    terms = ["foobar"]
    s_title, m1 = _score_node_for_terms(
        {"title": "foobar", "summary": "", "text": "", "node_id": "a"}, terms
    )
    s_text, m2 = _score_node_for_terms(
        {"title": "", "summary": "", "text": "foobar", "node_id": "b"}, terms
    )
    assert m1 == m2 == ["foobar"]
    assert s_title == 3.0
    assert s_text == 1.0


def test_score_node_summary_between_title_and_text() -> None:
    terms = ["x"]
    s_sum, _ = _score_node_for_terms(
        {"title": "", "summary": "x", "text": "", "node_id": "a"}, terms
    )
    assert s_sum == 2.0


def test_search_content_by_query_ranks_and_fields() -> None:
    documents = {
        "d": normalize_document_payload(
            {
                "doc_id": "d",
                "result": [
                    {
                        "title": "冷门小节",
                        "node_id": "0001",
                        "page_index": 1,
                        "text": "无关内容",
                        "summary": "",
                        "nodes": [],
                    },
                    {
                        "title": "下载量 TOP",
                        "node_id": "0002",
                        "page_index": 2,
                        "text": "全球下载量 TOP 10 游戏榜单",
                        "summary": "摘要也含下载量",
                        "nodes": [],
                    },
                ],
                "line_count": 10,
                "summary": "",
            }
        )
    }
    rows = json.loads(
        search_content_by_query(documents, "d", "下载量, 排行", top_k=4)
    )
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert rows[0]["node_id"] == "0002"
    assert rows[0]["page"] == 2
    assert "score" in rows[0]
    assert "matched_terms" in rows[0]
    assert "下载量" in rows[0]["content"] or "下载量" in rows[0]["title"]


def test_search_content_by_query_missing_doc() -> None:
    err = json.loads(search_content_by_query({}, "x", "hello"))
    assert "error" in err


def test_search_content_by_query_pdf_rejected() -> None:
    documents = {"p": {"type": "pdf", "structure": []}}
    err = json.loads(search_content_by_query(documents, "p", "test"))
    assert "error" in err
    low = err["error"].lower()
    assert "markdown" in low or "pdf" in low


def test_search_content_by_query_empty_terms() -> None:
    documents = {
        "d": normalize_document_payload(
            {
                "doc_id": "d",
                "result": [{"title": "A", "node_id": "0", "text": "b"}],
                "line_count": 1,
                "summary": "",
            }
        )
    }
    err = json.loads(search_content_by_query(documents, "d", "   ", locale="zh"))
    assert "error" in err


def test_get_page_content_invalid_pages() -> None:
    documents = {
        "x": normalize_document_payload(
            {"doc_id": "x", "result": [], "line_count": 1, "summary": ""}
        )
    }
    err = json.loads(get_page_content(documents, "x", "7-5"))
    assert "error" in err


@pytest.mark.skipif(not _COMBINED_JSON.is_file(), reason="fixture JSON not present")
def test_combined_document_output_no_text_in_structure() -> None:
    raw = json.loads(_COMBINED_JSON.read_text(encoding="utf-8"))
    doc_id = raw["doc_id"]
    documents = {doc_id: normalize_document_payload(raw)}
    struct_str = get_document_structure(documents, doc_id)
    struct = json.loads(struct_str)

    def assert_no_text(obj: object) -> None:
        if isinstance(obj, dict):
            assert "text" not in obj
            for v in obj.values():
                assert_no_text(v)
        elif isinstance(obj, list):
            for item in obj:
                assert_no_text(item)

    assert_no_text(struct)
    assert isinstance(struct, list)
    assert len(struct) >= 1


@pytest.mark.skipif(not _COMBINED_JSON.is_file(), reason="fixture JSON not present")
def test_combined_document_search_by_query_finds_content() -> None:
    raw = json.loads(_COMBINED_JSON.read_text(encoding="utf-8"))
    doc_id = raw["doc_id"]
    documents = {doc_id: normalize_document_payload(raw)}
    rows = json.loads(search_content_by_query(documents, doc_id, "游戏, 下载量", top_k=5))
    assert isinstance(rows, list)
    assert len(rows) >= 1
    for row in rows:
        assert "node_id" in row
        assert "title" in row
        assert "content" in row
        assert "score" in row
        assert row["score"] > 0


@pytest.mark.skipif(not _COMBINED_JSON.is_file(), reason="fixture JSON not present")
def test_character_hash_embedding_stable_and_normalized() -> None:
    a = _character_hash_embedding("hello", dim=64)
    b = _character_hash_embedding("hello", dim=64)
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6 or sum(x * x for x in a) == 0


def test_index_document_semantic_and_search_lazy() -> None:
    doc = normalize_document_payload(
        {
            "doc_id": "s1",
            "result": [
                {
                    "title": "Alpha",
                    "node_id": "0001",
                    "page_index": 1,
                    "text": "alpha beta gamma",
                    "summary": "",
                    "nodes": [],
                },
                {
                    "title": "Zeta",
                    "node_id": "0002",
                    "page_index": 2,
                    "text": "unrelated zzz",
                    "summary": "",
                    "nodes": [],
                },
            ],
            "line_count": 5,
            "summary": "",
        }
    )
    assert _SEMANTIC_INDEX_KEY not in doc
    documents = {"s1": doc}
    rows = json.loads(
        search_content_semantic(documents, "s1", "alpha, beta", top_k=2)
    )
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert rows[0]["node_id"] == "0001"
    assert "similarity" in rows[0]
    assert _SEMANTIC_INDEX_KEY in doc


def test_search_content_semantic_empty_keywords_error() -> None:
    documents = {
        "d": normalize_document_payload(
            {
                "doc_id": "d",
                "result": [{"title": "A", "node_id": "0", "text": "x"}],
                "line_count": 1,
                "summary": "",
            }
        )
    }
    err = json.loads(search_content_semantic(documents, "d", "  "))
    assert "error" in err


def test_search_content_semantic_pdf_rejected() -> None:
    documents = {"p": {"type": "pdf", "structure": []}}
    err = json.loads(search_content_semantic(documents, "p", "x"))
    assert "error" in err


def test_custom_embed_fn_synonym_style() -> None:
    """Inject embed_fn so query and node need not share exact substrings."""

    def embed(t: str) -> list[float]:
        v = [0.0, 0.0, 0.0]
        if "营收" in t or "收入" in t:
            v[0] = 1.0
        if "成本" in t:
            v[1] = 1.0
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n else v

    doc = normalize_document_payload(
        {
            "doc_id": "c1",
            "result": [
                {
                    "title": "财务",
                    "node_id": "n1",
                    "page_index": 1,
                    "text": "本季度公司收入同比增长。",
                    "summary": "",
                    "nodes": [],
                },
                {
                    "title": "其它",
                    "node_id": "n2",
                    "page_index": 2,
                    "text": "成本结构优化。",
                    "summary": "",
                    "nodes": [],
                },
            ],
            "line_count": 2,
            "summary": "",
        }
    )
    index_document_semantic(doc, embed_fn=embed)
    documents = {"c1": doc}
    rows = json.loads(search_content_semantic(documents, "c1", "营收", embed_fn=embed))
    assert rows[0]["node_id"] == "n1"
    assert rows[0]["similarity"] > rows[1]["similarity"]


def test_search_content_hybrid_rrf() -> None:
    doc = normalize_document_payload(
        {
            "doc_id": "h1",
            "result": [
                {
                    "title": "仅词面",
                    "node_id": "a",
                    "page_index": 1,
                    "text": "uniquelexfoo",
                    "summary": "",
                    "nodes": [],
                },
                {
                    "title": "仅语义近邻",
                    "node_id": "b",
                    "page_index": 2,
                    "text": "uniquelexbar",
                    "summary": "",
                    "nodes": [],
                },
            ],
            "line_count": 2,
            "summary": "",
        }
    )
    documents = {"h1": doc}
    rows = json.loads(
        search_content_hybrid(documents, "h1", "uniquelexfoo", top_k=4)
    )
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert "rrf_score" in rows[0]


@pytest.mark.skipif(not _COMBINED_JSON.is_file(), reason="fixture JSON not present")
def test_combined_document_semantic_returns_results() -> None:
    raw = json.loads(_COMBINED_JSON.read_text(encoding="utf-8"))
    doc_id = raw["doc_id"]
    doc = normalize_document_payload(raw)
    if _SEMANTIC_INDEX_KEY in doc:
        del doc[_SEMANTIC_INDEX_KEY]
    documents = {doc_id: doc}
    rows = json.loads(
        search_content_semantic(documents, doc_id, "下载量", top_k=3)
    )
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert "similarity" in rows[0]


def test_combined_document_output_page_range_has_node_ids() -> None:
    raw = json.loads(_COMBINED_JSON.read_text(encoding="utf-8"))
    doc_id = raw["doc_id"]
    documents = {doc_id: normalize_document_payload(raw)}
    items = json.loads(get_page_content(documents, doc_id, "6-7"))
    assert len(items) >= 1
    for item in items:
        assert "node_id" in item
        assert "title" in item
        assert "content" in item
        assert "page" in item
        assert 6 <= item["page"] <= 7
