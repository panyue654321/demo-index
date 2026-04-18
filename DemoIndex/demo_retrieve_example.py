"""README 中「检索 Demo」的可运行脚本：在 DemoIndex 目录下执行 python demo_retrieve_example.py。"""

from __future__ import annotations

import json
from pathlib import Path

from retrieve import (
    get_page_content,
    normalize_document_payload,
    search_content_by_query,
    search_content_hybrid,
)


def main() -> None:
    demo_index_dir = Path(__file__).resolve().parent
    raw_path = (
        demo_index_dir
        / "docs/results_with_vlm/game2025report_v7/combined_document_output.json"
    )
    with raw_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    doc_id = raw["doc_id"]
    documents = {
        doc_id: normalize_document_payload(raw, doc_name="示例报告"),
    }

    keywords = "游戏, 下载量"
    lex_raw = search_content_by_query(documents, doc_id, keywords, top_k=5)
    lex_hits = json.loads(lex_raw)
    if isinstance(lex_hits, dict) and "error" in lex_hits:
        print("词面检索失败:", lex_hits)
        return

    print("--- 词面 Top ---")
    for row in lex_hits:
        title = (row.get("title") or "")[:50]
        print(
            row.get("node_id"),
            row.get("page"),
            title,
            "matched_terms=",
            row.get("matched_terms"),
        )

    hybrid_raw = search_content_hybrid(documents, doc_id, keywords, top_k=5)
    hybrid_hits = json.loads(hybrid_raw)
    if isinstance(hybrid_hits, list):
        print("--- 混合 Top（含 rrf_score 等）---")
        for row in hybrid_hits:
            print(row.get("node_id"), row.get("rrf_score"), row.get("title", "")[:40])

    if lex_hits:
        first_page = lex_hits[0].get("page")
        if first_page is not None:
            pages_raw = get_page_content(documents, doc_id, str(first_page))
            segments = json.loads(pages_raw)
            if isinstance(segments, list) and segments:
                print("--- 按页精读（首条命中所在页）---")
                print(
                    "node_id",
                    segments[0].get("node_id"),
                    "len(content)=",
                    len(segments[0].get("content") or ""),
                )


if __name__ == "__main__":
    main()
