"""CLI for DemoIndex."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _maybe_reexec_into_pageindex_venv() -> None:
    """Re-exec into PageIndex's virtualenv when required dependencies are missing."""
    if os.environ.get("DEMOINDEX_BOOTSTRAPPED") == "1":
        return
    try:
        import openai  # noqa: F401
        import pymupdf  # noqa: F401
        if (
            "--write-postgres" in sys.argv
            or "--write-global-index" in sys.argv
            or (len(sys.argv) > 1 and sys.argv[1] in {"retrieve", "retrieve-tree", "retrieve-evidence"})
        ):
            import psycopg  # noqa: F401
    except Exception:
        venv_python = REPO_ROOT / "PageIndex" / ".venv" / "bin" / "python"
        if not venv_python.exists():
            return
        env = os.environ.copy()
        env["DEMOINDEX_BOOTSTRAPPED"] = "1"
        cmd = [str(venv_python), "-m", "DemoIndex.run", *sys.argv[1:]]
        raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT), env=env))


_maybe_reexec_into_pageindex_venv()

from .pipeline import build_pageindex_tree, compare_tree
from .retrieval import retrieve_candidates, retrieve_evidence, retrieve_tree_candidates


def _parse_json_object_arg(value: str | None, *, arg_name: str) -> dict[str, float] | None:
    """Parse one optional CLI JSON object argument into a float mapping."""
    if not value:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must be a JSON object.")
    return {str(key): float(item) for key, item in payload.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and compare DemoIndex trees.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Build a PageIndex-style tree from a PDF.")
    run_parser.add_argument("--pdf-path", required=True, help="Path to the PDF file.")
    run_parser.add_argument("--output-json", default=None, help="Optional output JSON path.")
    run_parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional artifact directory. Defaults to DemoIndex/artifacts/<pdf_stem>.",
    )
    run_parser.add_argument(
        "--model",
        default="dashscope/qwen3.6-plus",
        help="Primary DashScope model for page transcription.",
    )
    run_parser.add_argument(
        "--fallback-model",
        default="dashscope/qwen3.5-plus",
        help="Fallback DashScope model when the primary model fails.",
    )
    run_parser.add_argument(
        "--include-summary",
        action="store_true",
        help="Generate PageIndex-style node summaries and include them in the output.",
    )
    run_parser.add_argument(
        "--write-postgres",
        action="store_true",
        help="Persist the final document tree into PostgreSQL using DATABASE_URL.",
    )
    run_parser.add_argument(
        "--write-global-index",
        action="store_true",
        help="Build and persist global chunk vectors into PostgreSQL using DATABASE_URL.",
    )
    run_parser.add_argument(
        "--global-index-model",
        default="text-embedding-v4",
        help="DashScope embedding model used for global chunk indexing.",
    )
    run_parser.add_argument(
        "--debug-log",
        action="store_true",
        help="Write structured debug logs, API usage, and stage timings under the artifact directory.",
    )
    run_parser.add_argument(
        "--debug-log-dir",
        default=None,
        help="Optional directory for structured debug logs. Defaults to <artifacts-dir>/debug.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare two tree JSON files.")
    compare_parser.add_argument("--actual-json", required=True, help="Generated tree JSON path.")
    compare_parser.add_argument("--expected-json", required=True, help="Expected tree JSON path.")
    compare_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for saving the comparison report as JSON.",
    )

    retrieve_parser = subparsers.add_parser("retrieve", help="Run Stage 1 and Stage 2 retrieval.")
    retrieve_parser.add_argument("--query", required=True, help="Search query text.")
    retrieve_parser.add_argument("--output-json", default=None, help="Optional output JSON path.")
    retrieve_parser.add_argument("--top-k-dense", type=int, default=60, help="Dense ANN recall limit.")
    retrieve_parser.add_argument("--top-k-lexical", type=int, default=60, help="Lexical recall limit.")
    retrieve_parser.add_argument(
        "--top-k-fused-chunks",
        type=int,
        default=80,
        help="Final fused chunk candidate limit.",
    )
    retrieve_parser.add_argument("--top-k-docs", type=int, default=10, help="Document candidate limit.")
    retrieve_parser.add_argument(
        "--top-k-sections-per-doc",
        type=int,
        default=3,
        help="Section anchor limit per selected document.",
    )
    retrieve_parser.add_argument(
        "--top-k-chunks-per-section",
        type=int,
        default=2,
        help="Supporting chunk limit per selected section.",
    )
    retrieve_parser.add_argument(
        "--disable-llm-parse",
        action="store_true",
        help="Disable optional query-time LLM enrichment.",
    )
    retrieve_parser.add_argument(
        "--parse-model",
        default="dashscope/qwen3.6-plus",
        help="DashScope chat model used for query-time LLM parsing.",
    )
    retrieve_parser.add_argument(
        "--parse-fallback-model",
        default="dashscope/qwen3.5-plus",
        help="Fallback DashScope chat model for query-time parsing.",
    )
    retrieve_parser.add_argument(
        "--embedding-model",
        default="text-embedding-v4",
        help="Embedding model used for dense retrieval.",
    )
    retrieve_parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal rank fusion constant.",
    )
    retrieve_parser.add_argument(
        "--lexical-score-threshold",
        type=float,
        default=0.18,
        help="Minimum lexical similarity threshold for candidate generation.",
    )
    retrieve_parser.add_argument(
        "--doc-score-chunk-limit",
        type=int,
        default=5,
        help="How many top fused chunks contribute to each doc score.",
    )
    retrieve_parser.add_argument(
        "--section-score-chunk-limit",
        type=int,
        default=3,
        help="How many top fused chunks contribute to each section score.",
    )
    retrieve_parser.add_argument(
        "--retrieval-profile-path",
        default=None,
        help="Optional retrieval profile JSON path overriding DEMOINDEX_RETRIEVAL_PROFILE_PATH.",
    )
    retrieve_parser.add_argument(
        "--debug-log",
        action="store_true",
        help="Write structured retrieval debug logs and timings.",
    )
    retrieve_parser.add_argument(
        "--debug-log-dir",
        default=None,
        help="Optional directory for retrieval debug logs.",
    )

    retrieve_tree_parser = subparsers.add_parser(
        "retrieve-tree",
        help="Run Stage 1 + Stage 2 + Stage 3 tree localization.",
    )
    retrieve_tree_parser.add_argument("--query", required=True, help="Search query text.")
    retrieve_tree_parser.add_argument("--output-json", default=None, help="Optional output JSON path.")
    retrieve_tree_parser.add_argument("--top-k-dense", type=int, default=60, help="Dense ANN recall limit.")
    retrieve_tree_parser.add_argument("--top-k-lexical", type=int, default=60, help="Lexical recall limit.")
    retrieve_tree_parser.add_argument(
        "--top-k-fused-chunks",
        type=int,
        default=80,
        help="Final fused chunk candidate limit.",
    )
    retrieve_tree_parser.add_argument("--top-k-docs", type=int, default=10, help="Document candidate limit.")
    retrieve_tree_parser.add_argument(
        "--top-k-sections-per-doc",
        type=int,
        default=3,
        help="Section anchor limit per selected document.",
    )
    retrieve_tree_parser.add_argument(
        "--top-k-chunks-per-section",
        type=int,
        default=2,
        help="Supporting chunk limit per selected section.",
    )
    retrieve_tree_parser.add_argument(
        "--stage3-mode",
        choices=("heuristic", "hybrid"),
        default="hybrid",
        help="Stage 3 localization mode.",
    )
    retrieve_tree_parser.add_argument(
        "--top-k-tree-sections-per-doc",
        type=int,
        default=5,
        help="Final localized section limit per selected document.",
    )
    retrieve_tree_parser.add_argument(
        "--top-k-anchor-sections-per-doc",
        type=int,
        default=3,
        help="Anchor section limit per selected document.",
    )
    retrieve_tree_parser.add_argument(
        "--disable-whole-doc-fallback",
        action="store_true",
        help="Disable whole-document fallback when the anchor-local pool is too small.",
    )
    retrieve_tree_parser.add_argument(
        "--disable-llm-parse",
        action="store_true",
        help="Disable optional query-time LLM enrichment.",
    )
    retrieve_tree_parser.add_argument(
        "--parse-model",
        default="dashscope/qwen3.6-plus",
        help="DashScope chat model used for query-time LLM parsing.",
    )
    retrieve_tree_parser.add_argument(
        "--parse-fallback-model",
        default="dashscope/qwen3.5-plus",
        help="Fallback DashScope chat model for query-time parsing.",
    )
    retrieve_tree_parser.add_argument(
        "--embedding-model",
        default="text-embedding-v4",
        help="Embedding model used for dense retrieval.",
    )
    retrieve_tree_parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal rank fusion constant.",
    )
    retrieve_tree_parser.add_argument(
        "--lexical-score-threshold",
        type=float,
        default=0.18,
        help="Minimum lexical similarity threshold for candidate generation.",
    )
    retrieve_tree_parser.add_argument(
        "--doc-score-chunk-limit",
        type=int,
        default=5,
        help="How many top fused chunks contribute to each doc score.",
    )
    retrieve_tree_parser.add_argument(
        "--section-score-chunk-limit",
        type=int,
        default=3,
        help="How many top fused chunks contribute to each section score.",
    )
    retrieve_tree_parser.add_argument(
        "--retrieval-profile-path",
        default=None,
        help="Optional retrieval profile JSON path overriding DEMOINDEX_RETRIEVAL_PROFILE_PATH.",
    )
    retrieve_tree_parser.add_argument(
        "--debug-log",
        action="store_true",
        help="Write structured retrieval debug logs and timings.",
    )
    retrieve_tree_parser.add_argument(
        "--debug-log-dir",
        default=None,
        help="Optional directory for retrieval debug logs.",
    )
    retrieve_tree_parser.add_argument(
        "--stage3-rerank-model",
        default="dashscope/qwen3.6-plus",
        help="DashScope chat model used for Stage 3 hybrid rerank.",
    )
    retrieve_tree_parser.add_argument(
        "--stage3-rerank-fallback-model",
        default="dashscope/qwen3.5-plus",
        help="Fallback DashScope chat model for Stage 3 rerank.",
    )
    retrieve_tree_parser.add_argument(
        "--stage3-shortlist-size",
        type=int,
        default=8,
        help="Shortlist size per document before Stage 3 hybrid rerank.",
    )
    retrieve_tree_parser.add_argument(
        "--stage3-relation-priors-json",
        default=None,
        help="Optional JSON object overriding Stage 3 relation priors.",
    )

    retrieve_evidence_parser = subparsers.add_parser(
        "retrieve-evidence",
        help="Run Stage 1 through Stage 5 retrieval and package final evidence.",
    )
    retrieve_evidence_parser.add_argument("--query", required=True, help="Search query text.")
    retrieve_evidence_parser.add_argument("--output-json", default=None, help="Optional output JSON path.")
    retrieve_evidence_parser.add_argument("--top-k-dense", type=int, default=60, help="Dense ANN recall limit.")
    retrieve_evidence_parser.add_argument("--top-k-lexical", type=int, default=60, help="Lexical recall limit.")
    retrieve_evidence_parser.add_argument(
        "--top-k-fused-chunks",
        type=int,
        default=80,
        help="Final fused chunk candidate limit.",
    )
    retrieve_evidence_parser.add_argument("--top-k-docs", type=int, default=10, help="Document candidate limit.")
    retrieve_evidence_parser.add_argument(
        "--top-k-sections-per-doc",
        type=int,
        default=3,
        help="Section anchor limit per selected document.",
    )
    retrieve_evidence_parser.add_argument(
        "--top-k-chunks-per-section",
        type=int,
        default=2,
        help="Supporting chunk limit per selected section.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage3-mode",
        choices=("heuristic", "hybrid"),
        default="hybrid",
        help="Stage 3 localization mode.",
    )
    retrieve_evidence_parser.add_argument(
        "--top-k-tree-sections-per-doc",
        type=int,
        default=5,
        help="Final localized section limit per selected document.",
    )
    retrieve_evidence_parser.add_argument(
        "--top-k-anchor-sections-per-doc",
        type=int,
        default=3,
        help="Anchor section limit per selected document.",
    )
    retrieve_evidence_parser.add_argument(
        "--disable-whole-doc-fallback",
        action="store_true",
        help="Disable whole-document fallback when the anchor-local pool is too small.",
    )
    retrieve_evidence_parser.add_argument(
        "--disable-llm-parse",
        action="store_true",
        help="Disable optional query-time LLM enrichment.",
    )
    retrieve_evidence_parser.add_argument(
        "--parse-model",
        default="dashscope/qwen3.6-plus",
        help="DashScope chat model used for query-time LLM parsing.",
    )
    retrieve_evidence_parser.add_argument(
        "--parse-fallback-model",
        default="dashscope/qwen3.5-plus",
        help="Fallback DashScope chat model for query-time parsing.",
    )
    retrieve_evidence_parser.add_argument(
        "--embedding-model",
        default="text-embedding-v4",
        help="Embedding model used for dense retrieval.",
    )
    retrieve_evidence_parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal rank fusion constant.",
    )
    retrieve_evidence_parser.add_argument(
        "--lexical-score-threshold",
        type=float,
        default=0.18,
        help="Minimum lexical similarity threshold for candidate generation.",
    )
    retrieve_evidence_parser.add_argument(
        "--doc-score-chunk-limit",
        type=int,
        default=5,
        help="How many top fused chunks contribute to each doc score.",
    )
    retrieve_evidence_parser.add_argument(
        "--section-score-chunk-limit",
        type=int,
        default=3,
        help="How many top fused chunks contribute to each section score.",
    )
    retrieve_evidence_parser.add_argument(
        "--retrieval-profile-path",
        default=None,
        help="Optional retrieval profile JSON path overriding DEMOINDEX_RETRIEVAL_PROFILE_PATH.",
    )
    retrieve_evidence_parser.add_argument(
        "--debug-log",
        action="store_true",
        help="Write structured retrieval debug logs and timings.",
    )
    retrieve_evidence_parser.add_argument(
        "--debug-log-dir",
        default=None,
        help="Optional directory for retrieval debug logs.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage3-rerank-model",
        default="dashscope/qwen3.6-plus",
        help="DashScope chat model used for Stage 3 hybrid rerank.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage3-rerank-fallback-model",
        default="dashscope/qwen3.5-plus",
        help="Fallback DashScope chat model for Stage 3 rerank.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage3-shortlist-size",
        type=int,
        default=8,
        help="Shortlist size per document before Stage 3 hybrid rerank.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage3-relation-priors-json",
        default=None,
        help="Optional JSON object overriding Stage 3 relation priors.",
    )
    retrieve_evidence_parser.add_argument(
        "--top-k-focus-sections-per-doc",
        type=int,
        default=3,
        help="How many localized sections per doc enter Stage 4 expansion.",
    )
    retrieve_evidence_parser.add_argument(
        "--max-ancestor-hops",
        type=int,
        default=2,
        help="Maximum ancestor hops to include for each focus section.",
    )
    retrieve_evidence_parser.add_argument(
        "--max-descendant-depth",
        type=int,
        default=1,
        help="Maximum descendant depth to include for each focus section.",
    )
    retrieve_evidence_parser.add_argument(
        "--max-siblings-per-focus",
        type=int,
        default=2,
        help="Maximum sibling sections to include for each focus section.",
    )
    retrieve_evidence_parser.add_argument(
        "--chunk-neighbor-window",
        type=int,
        default=1,
        help="Neighbor window around supporting chunks inside the focus section.",
    )
    retrieve_evidence_parser.add_argument(
        "--max-evidence-chunks-per-focus",
        type=int,
        default=6,
        help="Maximum evidence chunks kept for each focus section.",
    )
    retrieve_evidence_parser.add_argument(
        "--context-char-budget",
        type=int,
        default=6000,
        help="Character budget for each Stage 4 answer-ready context.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage5-relation-mode",
        choices=("heuristic", "hybrid"),
        default="heuristic",
        help="Stage 5 relation-labeling mode.",
    )
    retrieve_evidence_parser.add_argument(
        "--top-k-evidence-per-doc",
        type=int,
        default=3,
        help="Maximum evidence items kept per document in Stage 5.",
    )
    retrieve_evidence_parser.add_argument(
        "--top-k-total-evidence",
        type=int,
        default=8,
        help="Maximum evidence items kept overall in Stage 5.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage5-relation-model",
        default="dashscope/qwen3.6-plus",
        help="DashScope chat model used for Stage 5 hybrid relation labeling.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage5-relation-fallback-model",
        default="dashscope/qwen3.5-plus",
        help="Fallback DashScope chat model for Stage 5 relation labeling.",
    )
    retrieve_evidence_parser.add_argument(
        "--stage5-relation-shortlist-size",
        type=int,
        default=8,
        help="How many evidence items enter the Stage 5 hybrid relation-labeling pass.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the DemoIndex CLI."""
    args = _parse_args()
    if args.command == "run":
        artifact_root = (
            Path(args.artifacts_dir).expanduser().resolve()
            if args.artifacts_dir
            else REPO_ROOT / "DemoIndex" / "artifacts" / Path(args.pdf_path).stem
        )
        output_path = (
            Path(args.output_json).expanduser().resolve()
            if args.output_json
            else artifact_root / f"{Path(args.pdf_path).stem}_pageindex_tree.json"
        )
        result = build_pageindex_tree(
            pdf_path=args.pdf_path,
            output_json=args.output_json,
            artifacts_dir=args.artifacts_dir,
            model=args.model,
            fallback_model=args.fallback_model,
            include_summary=args.include_summary,
            write_postgres=args.write_postgres,
            write_global_index=args.write_global_index,
            global_index_model=args.global_index_model,
            debug_log=args.debug_log,
            debug_log_dir=args.debug_log_dir,
        )
        if not result:
            return 1
        print(output_path)
        return 0

    if args.command == "compare":
        report = compare_tree(args.actual_json, args.expected_json)
        output_path = getattr(args, "output_json", None)
        if output_path:
            Path(output_path).expanduser().resolve().write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(output_path)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "retrieve":
        result = retrieve_candidates(
            query=args.query,
            top_k_dense=args.top_k_dense,
            top_k_lexical=args.top_k_lexical,
            top_k_fused_chunks=args.top_k_fused_chunks,
            top_k_docs=args.top_k_docs,
            top_k_sections_per_doc=args.top_k_sections_per_doc,
            top_k_chunks_per_section=args.top_k_chunks_per_section,
            use_llm_parse=not args.disable_llm_parse,
            parse_model=args.parse_model,
            parse_fallback_model=args.parse_fallback_model,
            embedding_model=args.embedding_model,
            rrf_k=args.rrf_k,
            lexical_score_threshold=args.lexical_score_threshold,
            doc_score_chunk_limit=args.doc_score_chunk_limit,
            section_score_chunk_limit=args.section_score_chunk_limit,
            retrieval_profile_path=args.retrieval_profile_path,
            debug_log=args.debug_log,
            debug_log_dir=args.debug_log_dir,
        )
    elif args.command == "retrieve-tree":
        result = retrieve_tree_candidates(
            query=args.query,
            top_k_dense=args.top_k_dense,
            top_k_lexical=args.top_k_lexical,
            top_k_fused_chunks=args.top_k_fused_chunks,
            top_k_docs=args.top_k_docs,
            top_k_sections_per_doc=args.top_k_sections_per_doc,
            top_k_chunks_per_section=args.top_k_chunks_per_section,
            use_llm_parse=not args.disable_llm_parse,
            parse_model=args.parse_model,
            parse_fallback_model=args.parse_fallback_model,
            embedding_model=args.embedding_model,
            rrf_k=args.rrf_k,
            lexical_score_threshold=args.lexical_score_threshold,
            doc_score_chunk_limit=args.doc_score_chunk_limit,
            section_score_chunk_limit=args.section_score_chunk_limit,
            retrieval_profile_path=args.retrieval_profile_path,
            stage3_mode=args.stage3_mode,
            top_k_tree_sections_per_doc=args.top_k_tree_sections_per_doc,
            top_k_anchor_sections_per_doc=args.top_k_anchor_sections_per_doc,
            whole_doc_fallback=not args.disable_whole_doc_fallback,
            rerank_model=args.stage3_rerank_model,
            rerank_fallback_model=args.stage3_rerank_fallback_model,
            stage3_shortlist_size=args.stage3_shortlist_size,
            stage3_relation_priors=_parse_json_object_arg(
                args.stage3_relation_priors_json,
                arg_name="--stage3-relation-priors-json",
            ),
            debug_log=args.debug_log,
            debug_log_dir=args.debug_log_dir,
        )
    else:
        result = retrieve_evidence(
            query=args.query,
            top_k_dense=args.top_k_dense,
            top_k_lexical=args.top_k_lexical,
            top_k_fused_chunks=args.top_k_fused_chunks,
            top_k_docs=args.top_k_docs,
            top_k_sections_per_doc=args.top_k_sections_per_doc,
            top_k_chunks_per_section=args.top_k_chunks_per_section,
            use_llm_parse=not args.disable_llm_parse,
            parse_model=args.parse_model,
            parse_fallback_model=args.parse_fallback_model,
            embedding_model=args.embedding_model,
            rrf_k=args.rrf_k,
            lexical_score_threshold=args.lexical_score_threshold,
            doc_score_chunk_limit=args.doc_score_chunk_limit,
            section_score_chunk_limit=args.section_score_chunk_limit,
            retrieval_profile_path=args.retrieval_profile_path,
            stage3_mode=args.stage3_mode,
            top_k_tree_sections_per_doc=args.top_k_tree_sections_per_doc,
            top_k_anchor_sections_per_doc=args.top_k_anchor_sections_per_doc,
            whole_doc_fallback=not args.disable_whole_doc_fallback,
            rerank_model=args.stage3_rerank_model,
            rerank_fallback_model=args.stage3_rerank_fallback_model,
            stage3_shortlist_size=args.stage3_shortlist_size,
            stage3_relation_priors=_parse_json_object_arg(
                args.stage3_relation_priors_json,
                arg_name="--stage3-relation-priors-json",
            ),
            top_k_focus_sections_per_doc=args.top_k_focus_sections_per_doc,
            max_ancestor_hops=args.max_ancestor_hops,
            max_descendant_depth=args.max_descendant_depth,
            max_siblings_per_focus=args.max_siblings_per_focus,
            chunk_neighbor_window=args.chunk_neighbor_window,
            max_evidence_chunks_per_focus=args.max_evidence_chunks_per_focus,
            context_char_budget=args.context_char_budget,
            stage5_relation_mode=args.stage5_relation_mode,
            top_k_evidence_per_doc=args.top_k_evidence_per_doc,
            top_k_total_evidence=args.top_k_total_evidence,
            stage5_relation_model=args.stage5_relation_model,
            stage5_relation_fallback_model=args.stage5_relation_fallback_model,
            stage5_relation_shortlist_size=args.stage5_relation_shortlist_size,
            debug_log=args.debug_log,
            debug_log_dir=args.debug_log_dir,
        )
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    output_path = getattr(args, "output_json", None)
    if output_path:
        Path(output_path).expanduser().resolve().write_text(payload, encoding="utf-8")
        print(output_path)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
