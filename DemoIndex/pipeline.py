"""PageIndex-aligned pipeline for DemoIndex."""

from __future__ import annotations

import asyncio
import json
import hashlib
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .build_md_pageindex import PageIndexOptions, sync_build_pageindex_payload
from .debug import DebugRecorder
from .env import PAGEINDEX_ROOT, REPO_ROOT, ensure_pageindex_import_path
from .global_index import (
    DEFAULT_GLOBAL_INDEX_MODEL,
    build_global_chunk_records,
)
from .llm import DashScopeEmbeddingClient, QwenChatClient
from .pdf import extract_outline_entries, extract_page_artifacts, layout_heading_candidates, normalize_text
from .postgres_store import persist_document_sections, persist_section_chunks, resolve_database_url


def build_pageindex_tree(
    input_path: str | None = None,
    pdf_path: str | None = None,
    output_json: str | None = None,
    artifacts_dir: str | None = None,
    model: str = "dashscope/qwen3.6-plus",
    fallback_model: str = "dashscope/qwen3.5-plus",
    include_summary: bool = False,
    write_postgres: bool = False,
    write_global_index: bool = False,
    global_index_model: str = DEFAULT_GLOBAL_INDEX_MODEL,
    markdown_layout: str = "auto",
    debug_log: bool = False,
    debug_log_dir: str | None = None,
) -> dict[str, Any]:
    """Build a target-format tree from one PDF or Markdown input."""
    ensure_pageindex_import_path()
    _load_pageindex_env()
    effective_write_postgres = write_postgres or write_global_index
    effective_include_summary = include_summary or effective_write_postgres
    if effective_write_postgres:
        resolve_database_url()

    resolved_input_path = _resolve_input_path(input_path=input_path, pdf_path=pdf_path)
    input_kind = _detect_input_kind(resolved_input_path)

    artifact_root = (
        Path(artifacts_dir).expanduser().resolve()
        if artifacts_dir
        else REPO_ROOT / "DemoIndex" / "artifacts" / resolved_input_path.stem
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    debug_recorder = (
        DebugRecorder(Path(debug_log_dir).expanduser().resolve() if debug_log_dir else artifact_root / "debug")
        if debug_log
        else None
    )
    if debug_recorder is not None:
        debug_recorder.set_run_metadata(
            input_path=str(resolved_input_path),
            input_kind=input_kind,
            artifact_root=str(artifact_root),
            output_json=str(Path(output_json).expanduser().resolve()) if output_json else None,
            model=model,
            fallback_model=fallback_model,
            include_summary=effective_include_summary,
            write_postgres=effective_write_postgres,
            write_global_index=write_global_index,
            global_index_model=global_index_model,
            markdown_layout=markdown_layout,
        )

    target_output_path: Path | None = None

    try:
        if input_kind == "pdf":
            output = _build_pdf_output(
                resolved_input_path=resolved_input_path,
                artifact_root=artifact_root,
                model=model,
                fallback_model=fallback_model,
                include_summary=effective_include_summary,
                debug_recorder=debug_recorder,
            )
        else:
            output = _build_markdown_output(
                resolved_input_path=resolved_input_path,
                artifact_root=artifact_root,
                model=model,
                fallback_model=fallback_model,
                include_summary=effective_include_summary,
                markdown_layout=markdown_layout,
                debug_recorder=debug_recorder,
            )
        if effective_write_postgres:
            with _debug_stage(debug_recorder, "persist_document_sections"):
                persistence_report = persist_document_sections(output)
            _save_json(artifact_root / "postgres_write.json", persistence_report)
        if write_global_index:
            utils_module = _load_pageindex_utils()
            with _debug_stage(debug_recorder, "build_global_chunk_records"):
                embedding_client = DashScopeEmbeddingClient(
                    model_name=global_index_model,
                    debug_recorder=debug_recorder,
                )
                chunk_records, chunk_report = build_global_chunk_records(
                    output,
                    count_tokens=utils_module.count_tokens,
                    embedding_client=embedding_client,
                    embedding_model=global_index_model,
                )
            with _debug_stage(debug_recorder, "persist_section_chunks"):
                chunk_persistence_report = persist_section_chunks(
                    chunk_records,
                    doc_id=str(output.get("doc_id") or ""),
                )
            _save_json(
                artifact_root / "global_index_write.json",
                {
                    **chunk_report,
                    "table_name": chunk_persistence_report["table_name"],
                    "doc_id": chunk_persistence_report["doc_id"],
                    "row_count": chunk_persistence_report["row_count"],
                    "records": chunk_persistence_report["records"],
                },
            )

        target_output_path = (
            Path(output_json).expanduser().resolve()
            if output_json
            else artifact_root / f"{resolved_input_path.stem}_pageindex_tree.json"
        )
        with _debug_stage(debug_recorder, "save_final_output"):
            _save_json(target_output_path, output)
        return output
    except Exception as exc:
        if debug_recorder is not None:
            debug_recorder.log_event(
                "run_error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        raise
    finally:
        if debug_recorder is not None:
            debug_recorder.write_summary(
                target_output_path=str(target_output_path) if target_output_path else None,
                pageindex_raw_path=str(artifact_root / "pageindex_raw.json"),
                outline_entries_path=str(artifact_root / "outline_entries.json"),
                seeded_outline_path=str(artifact_root / "seeded_outline.json"),
            )


def _build_pdf_output(
    *,
    resolved_input_path: Path,
    artifact_root: Path,
    model: str,
    fallback_model: str | None,
    include_summary: bool,
    debug_recorder: DebugRecorder | None,
) -> dict[str, Any]:
    """Build the final output payload for one PDF input."""
    pageindex_module, utils_module, config_loader = _patch_pageindex_llm(
        model=model,
        fallback_model=fallback_model,
        debug_recorder=debug_recorder,
    )
    with _debug_stage(debug_recorder, "load_pageindex_config"):
        opt = config_loader.load(
            {
                "model": model,
                "if_add_node_id": "yes",
                "if_add_node_summary": "yes" if include_summary else "no",
                "if_add_doc_description": "no",
                "if_add_node_text": "yes",
            }
        )
    with _debug_stage(debug_recorder, "get_page_tokens"):
        page_list = utils_module.get_page_tokens(str(resolved_input_path), model=model)
    with _debug_stage(debug_recorder, "extract_page_artifacts"):
        page_artifacts = extract_page_artifacts(resolved_input_path, artifact_root)
    with _debug_stage(debug_recorder, "extract_outline_entries"):
        toc_page_number, outline_entries = extract_outline_entries(page_artifacts)
    _save_json(artifact_root / "outline_entries.json", [asdict(entry) for entry in outline_entries])

    with _debug_stage(debug_recorder, "build_seeded_outline"):
        seeded_outline = _build_seeded_outline(
            page_artifacts=page_artifacts,
            outline_entries=outline_entries,
            toc_page_number=toc_page_number,
        )
    _save_json(artifact_root / "seeded_outline.json", seeded_outline)

    with _debug_stage(debug_recorder, "build_tree_from_seeded_outline"):
        tree = asyncio.run(
            _build_tree_from_seeded_outline(
                seeded_outline=seeded_outline,
                page_list=page_list,
                pageindex_module=pageindex_module,
                utils_module=utils_module,
                opt=opt,
                debug_recorder=debug_recorder,
            )
        )
    with _debug_stage(debug_recorder, "write_node_id"):
        utils_module.write_node_id(tree)
    with _debug_stage(debug_recorder, "add_node_text"):
        utils_module.add_node_text(tree, page_list)
    if include_summary:
        with _debug_stage(debug_recorder, "generate_node_summaries"):
            asyncio.run(utils_module.generate_summaries_for_structure(tree, model=opt.model))

    raw_result = {
        "doc_name": resolved_input_path.name,
        "structure": tree,
    }
    _save_json(artifact_root / "pageindex_raw.json", raw_result)

    with _debug_stage(debug_recorder, "convert_output_tree"):
        return {
            "doc_id": _stable_doc_id(resolved_input_path),
            "status": "completed",
            "retrieval_ready": False,
            "result": _convert_pageindex_structure(
                raw_result.get("structure") or [],
                include_summary=include_summary,
            ),
        }


def _build_markdown_output(
    *,
    resolved_input_path: Path,
    artifact_root: Path,
    model: str,
    fallback_model: str | None,
    include_summary: bool,
    markdown_layout: str,
    debug_recorder: DebugRecorder | None,
) -> dict[str, Any]:
    """Build the final output payload for one Markdown input."""
    resolved_layout = _resolve_markdown_layout(resolved_input_path, markdown_layout)
    if debug_recorder is not None:
        debug_recorder.log_event(
            "markdown_layout_selected",
            input_path=str(resolved_input_path),
            requested_layout=markdown_layout,
            resolved_layout=resolved_layout,
        )
    with _debug_stage(debug_recorder, "build_markdown_pageindex"):
        payload = sync_build_pageindex_payload(
            resolved_input_path,
            PageIndexOptions(
                doc_id=_stable_doc_id(resolved_input_path),
                status="completed",
                retrieval_ready=False,
                if_add_summary=include_summary,
                model=model,
            ),
            llm_factory=lambda: QwenChatClient(
                primary_model=model,
                fallback_model=fallback_model,
                debug_recorder=debug_recorder,
            ),
            layout=resolved_layout,
        )
    _save_json(artifact_root / "pageindex_raw.json", payload)
    return payload


def _resolve_input_path(*, input_path: str | None, pdf_path: str | None) -> Path:
    """Resolve the one allowed build input path and validate its existence."""
    provided_paths = [value for value in (input_path, pdf_path) if value]
    if not provided_paths:
        raise ValueError("Either input_path or pdf_path must be provided.")
    if input_path and pdf_path:
        raise ValueError("Only one of input_path or pdf_path may be provided.")
    resolved_path = Path(provided_paths[0]).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input file not found: {resolved_path}")
    return resolved_path


def _detect_input_kind(input_path: Path) -> str:
    """Return the normalized input kind for one build input path."""
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    raise ValueError(f"Unsupported input extension for DemoIndex build: {input_path.suffix}")


def _resolve_markdown_layout(input_path: Path, requested_layout: str) -> str:
    """Resolve the effective Markdown layout for one Markdown input."""
    if requested_layout != "auto":
        return requested_layout
    content = input_path.read_text(encoding="utf-8")
    if "<!-- page:" in content:
        return "page_per_page"
    return "h1_forest"


async def _build_tree_from_seeded_outline(
    *,
    seeded_outline: list[dict[str, Any]],
    page_list: list[tuple[str, int]],
    pageindex_module,
    utils_module,
    opt,
    debug_recorder: DebugRecorder | None = None,
) -> list[dict[str, Any]]:
    """Build a tree from a seeded top-level outline using PageIndex recursion."""
    logger = _DebugLogger(debug_recorder) if debug_recorder is not None else _NullLogger()
    outline_with_start_flags = await pageindex_module.check_title_appearance_in_start_concurrent(
        seeded_outline,
        page_list,
        model=opt.model,
        logger=logger,
    )
    valid_outline = [item for item in outline_with_start_flags if item.get("physical_index") is not None]
    tree = utils_module.post_processing(valid_outline, len(page_list))
    tasks = [
        pageindex_module.process_large_node_recursively(node, page_list, opt=opt, logger=logger)
        for node in tree
    ]
    await asyncio.gather(*tasks)
    return tree


def compare_tree(actual_json: str, expected_json: str) -> dict[str, Any]:
    """Compare two PageIndex-style tree JSON files."""
    actual = json.loads(Path(actual_json).expanduser().resolve().read_text(encoding="utf-8"))
    expected = json.loads(Path(expected_json).expanduser().resolve().read_text(encoding="utf-8"))

    actual_nodes = _flatten_tree(actual.get("result") or [])
    expected_nodes = _flatten_tree(expected.get("result") or [])
    actual_titles = [node["title"] for node in actual_nodes]
    expected_titles = [node["title"] for node in expected_nodes]

    actual_title_set = set(actual_titles)
    expected_title_set = set(expected_titles)
    matched_titles = sorted(actual_title_set & expected_title_set)

    page_matches = 0
    page_examples: list[dict[str, Any]] = []
    expected_by_title = {node["title"]: node for node in expected_nodes}
    for title in matched_titles:
        actual_page = next(node["page_index"] for node in actual_nodes if node["title"] == title)
        expected_page = expected_by_title[title]["page_index"]
        is_match = (
            actual_page is not None
            and expected_page is not None
            and abs(int(actual_page) - int(expected_page)) <= 1
        )
        if is_match:
            page_matches += 1
        if len(page_examples) < 20 and not is_match:
            page_examples.append(
                {
                    "title": title,
                    "actual_page_index": actual_page,
                    "expected_page_index": expected_page,
                }
            )

    return {
        "actual_root_titles": [node["title"] for node in actual.get("result") or []],
        "expected_root_titles": [node["title"] for node in expected.get("result") or []],
        "top_level_schema_match": list(actual.keys()) == list(expected.keys()),
        "actual_node_count": len(actual_nodes),
        "expected_node_count": len(expected_nodes),
        "matched_title_count": len(matched_titles),
        "title_precision": _ratio(len(matched_titles), len(actual_title_set)),
        "title_recall": _ratio(len(matched_titles), len(expected_title_set)),
        "page_match_ratio_within_one": _ratio(page_matches, len(matched_titles)),
        "missing_titles": sorted(expected_title_set - actual_title_set)[:30],
        "unexpected_titles": sorted(actual_title_set - expected_title_set)[:30],
        "page_mismatches": page_examples,
        "text_length_examples": [
            {
                "title": title,
                "actual_text_length": len(next(node["text"] for node in actual_nodes if node["title"] == title) or ""),
                "expected_text_length": len(expected_by_title[title]["text"] or ""),
            }
            for title in matched_titles[:15]
        ],
    }


def _load_pageindex_env() -> None:
    """Load environment variables from PageIndex's local `.env` file."""
    env_path = PAGEINDEX_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _patch_pageindex_llm(
    model: str,
    fallback_model: str | None,
    debug_recorder: DebugRecorder | None = None,
):
    """Patch PageIndex modules to use the local qwen-compatible client."""
    import importlib

    pageindex_module = importlib.import_module("pageindex.page_index")
    utils_module = importlib.import_module("pageindex.utils")

    client = QwenChatClient(
        primary_model=model,
        fallback_model=fallback_model,
        timeout_seconds=min(120.0, float(utils_module._get_llm_timeout_seconds())),
        max_retries=min(3, int(utils_module._get_llm_max_retries())),
        max_concurrency=int(utils_module._get_llm_max_concurrency()),
        debug_recorder=debug_recorder,
    )
    utils_module.llm_completion = client.completion
    utils_module.llm_acompletion = client.acompletion
    pageindex_module.llm_completion = client.completion
    pageindex_module.llm_acompletion = client.acompletion
    return pageindex_module, utils_module, utils_module.ConfigLoader()


def _load_pageindex_utils():
    """Load the PageIndex utilities module for shared helpers such as token counting."""
    import importlib

    return importlib.import_module("pageindex.utils")


def _build_seeded_outline(
    *,
    page_artifacts,
    outline_entries,
    toc_page_number: int | None,
) -> list[dict[str, Any]]:
    """Build a seeded outline that mirrors PageIndex's flat TOC item format."""
    seeded: list[dict[str, Any]] = []
    cover_title = _extract_cover_title(page_artifacts)
    include_cover = bool(cover_title and toc_page_number and toc_page_number > 1)
    include_toc = bool(include_cover and toc_page_number is not None)

    if include_cover:
        seeded.append({"structure": "0", "title": cover_title, "physical_index": 1})
    if include_toc and toc_page_number is not None:
        seeded.append({"structure": "0.1", "title": "目录", "physical_index": toc_page_number})

    counters: list[int] = []
    top_level_seed = 1 if include_toc else 0
    for entry in outline_entries:
        if entry.physical_page is None:
            continue
        level = _effective_outline_level(entry, page_artifacts)
        while len(counters) < level:
            counters.append(0)
        counters = counters[:level]
        if level == 1 and counters[0] == 0:
            counters[0] = top_level_seed
        counters[-1] += 1
        structure_parts = [str(value) for value in counters]
        if include_cover:
            structure = ".".join(["0", *structure_parts])
        else:
            structure = ".".join(structure_parts)
        seeded.append(
            {
                "structure": structure,
                "title": _resolved_entry_title(entry, page_artifacts),
                "physical_index": entry.physical_page,
            }
        )
    return seeded


def _extract_cover_title(page_artifacts) -> str | None:
    """Extract a likely cover title from the first page."""
    if not page_artifacts:
        return None
    prominent_lines = [
        " ".join(line_text.split()).strip()
        for line_text, bbox in page_artifacts[0].lines
        if bbox[1] < 260 and len(" ".join(line_text.split()).strip()) <= 20
    ]
    if len(prominent_lines) >= 2:
        combined = "".join(prominent_lines[:2]).strip()
        if len(combined) >= 8:
            return combined
    candidates = layout_heading_candidates(page_artifacts[0], limit=3)
    if candidates:
        return str(candidates[0]["title"]).strip()
    for line_text, _bbox in page_artifacts[0].lines:
        cleaned = " ".join(line_text.split()).strip()
        if len(cleaned) >= 4:
            return cleaned
    return None


def _effective_outline_level(entry, page_artifacts) -> int:
    """Adjust TOC indentation levels using the actual page heading scale."""
    level = max(1, int(entry.level_hint))
    if entry.physical_page is None or entry.physical_page > len(page_artifacts):
        return level
    page = page_artifacts[int(entry.physical_page) - 1]
    dominant_title, dominant_size = _page_dominant_heading(page)
    if (
        dominant_title
        and dominant_size >= 32.0
        and (
            normalize_text(entry.title) in normalize_text(dominant_title)
            or normalize_text(dominant_title) in normalize_text(entry.title)
        )
    ):
        return 1
    return level


def _page_dominant_heading(page_artifact) -> tuple[str | None, float]:
    """Return the dominant heading text and size for one page."""
    candidates = layout_heading_candidates(page_artifact, limit=5)
    if not candidates:
        return None, 0.0
    max_size = max(float(item["size"]) for item in candidates)
    dominant_parts = [
        str(item["title"]).strip()
        for item in candidates
        if float(item["size"]) >= max_size * 0.95
    ]
    return "".join(dominant_parts).strip() or None, max_size


def _convert_pageindex_structure(
    nodes: list[dict[str, Any]],
    *,
    include_summary: bool = False,
) -> list[dict[str, Any]]:
    """Convert PageIndex's PDF structure into the target JSON schema."""
    converted: list[dict[str, Any]] = []
    for node in nodes:
        item = {
            "title": _sanitize_output_title(str(node.get("title") or "")),
            "node_id": node.get("node_id"),
            "page_index": node.get("start_index"),
            "text": node.get("text", ""),
        }
        if include_summary and node.get("summary"):
            item["summary"] = node.get("summary")
        children = _convert_pageindex_structure(
            node.get("nodes") or [],
            include_summary=include_summary,
        )
        if children:
            item["nodes"] = children
        converted.append(item)
    return _reshape_root_nodes(converted)


def _flatten_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten a tree into a list while preserving title, page, and text."""
    flattened: list[dict[str, Any]] = []
    for node in nodes:
        flattened.append(
            {
                "title": node.get("title"),
                "page_index": node.get("page_index"),
                "text": node.get("text", ""),
            }
        )
        flattened.extend(_flatten_tree(node.get("nodes") or []))
    return flattened


def _save_json(path: Path, payload: Any) -> None:
    """Write a JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _debug_stage(debug_recorder: DebugRecorder | None, stage_name: str):
    """Return a no-op or structured debug stage context manager."""
    if debug_recorder is None:
        return _NoOpContextManager()
    return debug_recorder.stage(stage_name)


def _stable_doc_id(pdf_path: Path) -> str:
    """Build a stable document id from PDF content."""
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _ratio(numerator: int, denominator: int) -> float:
    """Return a safe ratio."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _sanitize_output_title(title: str) -> str:
    """Apply lightweight title cleanup for final tree output."""
    cleaned = " ".join(title.split())
    cleaned = _collapse_cjk_spaces(cleaned)
    if cleaned.startswith("前言："):
        return cleaned.removeprefix("前言：").strip()
    if cleaned.startswith("前言:"):
        return cleaned.removeprefix("前言:").strip()
    return cleaned


def _collapse_cjk_spaces(text: str) -> str:
    """Remove spaces inside titles when both neighbors are CJK, digits, or punctuation."""
    collapsed: list[str] = []
    for index, char in enumerate(text):
        if char != " ":
            collapsed.append(char)
            continue
        prev_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if _is_cjkish(prev_char) and _is_cjkish(next_char):
            continue
        collapsed.append(char)
    return "".join(collapsed).strip()


def _is_cjkish(char: str) -> bool:
    """Return whether a character belongs to a CJK-ish title token set."""
    if not char:
        return False
    return "\u4e00" <= char <= "\u9fff" or char.isdigit() or char in {"年", "月", "日", "！", "：", ":", "（", "）", "(", ")", "·"}


def _reshape_root_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Promote main sections out of the cover root while keeping TOC and preface nested."""
    if not nodes:
        return nodes
    first_root = nodes[0]
    children = first_root.get("nodes") or []
    if first_root.get("page_index") != 1 or len(children) < 3:
        return nodes
    if children[0].get("title") != "目录":
        return nodes
    first_root["nodes"] = children[:2]
    return [first_root, *children[2:], *nodes[1:]]


def _resolved_entry_title(entry, page_artifacts) -> str:
    """Prefer the page-visible heading text when it clearly matches the TOC title."""
    if entry.physical_page is None or entry.physical_page > len(page_artifacts):
        return entry.title
    dominant_title, dominant_size = _page_dominant_heading(page_artifacts[int(entry.physical_page) - 1])
    if dominant_title and dominant_size >= 22.0 and _titles_share_prefix(entry.title, dominant_title):
        return dominant_title
    return entry.title


def _titles_share_prefix(left_title: str, right_title: str) -> bool:
    """Return whether two titles likely refer to the same visible heading."""
    left = normalize_text(left_title)
    right = normalize_text(right_title)
    shared = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        shared += 1
    return shared >= 4


class _NullLogger:
    """A tiny logger compatible with the PageIndex call sites."""

    def info(self, _message: Any) -> None:
        """Ignore info logs."""
        return None

    def error(self, _message: Any) -> None:
        """Ignore error logs."""
        return None


class _DebugLogger:
    """Bridge PageIndex logger calls into DemoIndex debug events."""

    def __init__(self, debug_recorder: DebugRecorder) -> None:
        self._debug_recorder = debug_recorder

    def info(self, message: Any) -> None:
        """Record an info-level PageIndex log line."""
        self._debug_recorder.log_event("pageindex_log", level="info", message=str(message))

    def error(self, message: Any) -> None:
        """Record an error-level PageIndex log line."""
        self._debug_recorder.log_event("pageindex_log", level="error", message=str(message))


class _NoOpContextManager:
    """Provide a no-op context manager for disabled debug logging."""

    def __enter__(self) -> None:
        """Enter the no-op context."""
        return None

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        """Exit the no-op context without suppressing exceptions."""
        return False
