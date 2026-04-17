"""从带 `<!-- page:N -->` 注释与 ATX 标题的合并 Markdown 构建 PageIndex 风格的 JSON。

本模块将全文按一级标题 `#` 切成多棵树的森林，段内用 `##`～`######` 建树；
可选调用 LLM 生成文档级与各节点摘要。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PAGE_COMMENT_RE = re.compile(r"^\s*<!--\s*page:(\d+)\s*-->\s*$")
ATX_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
PREFIX_QIANYAN_RE = re.compile(r"^前言[：:]\s*(.+)$")


@dataclass
class PageIndexOptions:
    """构建 PageIndex 载荷时的可选项。

    属性:
        doc_id: 输出 JSON 中的文档 ID；为 None 时用文件路径的 UUID5 生成。
        status: 处理状态字符串，写入顶层 ``status``。
        retrieval_ready: 是否写入顶层 ``retrieval_ready``。
        if_add_summary: 为 True 时填充顶层与各节点的 ``summary``（可走 LLM 或启发式）。
        summary_char_threshold: 节点正文短于此字符数时，摘要直接采用正文或标题，不调模型。
        model: 调用 LLM 时使用的模型名；具体含义由 ``llm_factory`` 返回的客户端决定。
    """

    doc_id: str | None = None
    status: str = "completed"
    retrieval_ready: bool = False
    if_add_summary: bool = True
    summary_char_threshold: int = 600
    model: str | None = None


def compute_line_count(content: str) -> int:
    """统计 Markdown 全文行数（与常见 ``count('\\n') + 1`` 语义一致）。

    入参:
        content: 完整文件文本（UTF-8 解码后的字符串）。

    返回:
        行数，用于顶层 ``line_count`` 字段。
    """
    return content.count("\n") + 1


def parse_page_comments(lines: list[str]) -> list[int]:
    """按行解析 ``<!-- page:N -->``，得到每一行「行首」所处的逻辑页码。

    入参:
        lines: 按行切分后的文本行列表（不含换行符）。

    返回:
        与 ``lines`` 等长的整数列表；``page_by_line[i]`` 表示第 ``i`` 行开始时生效的页码
        （本行若为页码注释，先记入当前页，再在本行末尾更新后续页码）。
    """
    current = 1
    out: list[int] = []
    for line in lines:
        out.append(current)
        m = PAGE_COMMENT_RE.match(line)
        if m:
            current = int(m.group(1))
    return out


def iter_atx_headers(lines: list[str]) -> list[dict[str, Any]]:
    """扫描 ATX 标题（``#``～``######``），跳过围栏代码块内的行。

    入参:
        lines: 按行切分后的文本行列表。

    返回:
        标题字典列表，每项含 ``line_idx``（0 起始行号）、``level``（1～6）、
        ``raw_title``（去掉 ``#`` 后的标题原文）。
    """
    headers: list[dict[str, Any]] = []
    in_code = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = ATX_HEADER_RE.match(line.rstrip("\n"))
        if m:
            level = len(m.group(1))
            raw = m.group(2).strip()
            headers.append({"line_idx": i, "level": level, "raw_title": raw})
    return headers


def find_h1_line_indices(headers: list[dict[str, Any]]) -> list[int]:
    """从标题列表中取出所有一级标题所在行号。

    入参:
        headers: ``iter_atx_headers`` 的返回值。

    返回:
        每个 ``#``（且非 ``##``）标题的 ``line_idx``，按文档顺序排列。
    """
    return [h["line_idx"] for h in headers if h["level"] == 1]


def normalize_display_title(raw_title: str, level: int) -> str:
    """将 Markdown 标题转为对外展示的 ``title``（如去掉「前言：」、压缩「2025 年」）。

    入参:
        raw_title: 标题行去掉 ``#`` 后的字符串。
        level: 标题层级 1～6。

    返回:
        规范化后的展示标题。
    """
    t = raw_title.strip()
    if level >= 2:
        m = PREFIX_QIANYAN_RE.match(t)
        if m:
            t = m.group(1).strip()
    t = re.sub(r"(\d)\s+年", r"\1年", t)
    return t


def _join_lines(lines: list[str], start: int, end: int) -> str:
    """将 ``lines[start:end+1]`` 用换行拼接为一段文本，并去掉首尾多余换行。

    入参:
        lines: 全文行列表。
        start: 起始行下标（含）。
        end: 结束行下标（含）。

    返回:
        拼接后的字符串。
    """
    chunk = lines[start : end + 1]
    return "\n".join(chunk).strip("\n")


def _headers_in_range(
    all_headers: list[dict[str, Any]], start: int, end: int, min_level: int = 2
) -> list[dict[str, Any]]:
    """筛选落在闭区间 ``[start, end]`` 内、且层级 ``>= min_level`` 的标题。

    入参:
        all_headers: 全文标题列表。
        start: 起始行下标（含）。
        end: 结束行下标（含）。
        min_level: 最小标题层级，默认 2（即段内从 ``##`` 起）。

    返回:
        满足条件的标题字典子列表，顺序与文中出现顺序一致。
    """
    return [h for h in all_headers if start <= h["line_idx"] <= end and h["level"] >= min_level]


def build_section_root_and_flat_nodes(
    lines: list[str],
    page_by_line: list[int],
    all_headers: list[dict[str, Any]],
    h1_line: int,
    section_end: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """为单个 H1 段构建根节点（无 ``node_id`` / 子树）与段内 ``##+`` 的扁平节点列表。

    根节点正文为 H1 行起至第一个 ``##`` 之前；各子标题正文至「段内下一个任意级标题」之前。

    入参:
        lines: 全文行列表。
        page_by_line: ``parse_page_comments`` 返回值。
        all_headers: ``iter_atx_headers`` 返回值。
        h1_line: 当前段 H1 标题行的 0 起始下标。
        section_end: 当前段最后一行的 0 起始下标（含）。

    返回:
        ``(root, flat)``：``root`` 含 ``_line_idx``、``title``、``page_index``、``text``；
        ``flat`` 为段内二级及以下标题的扁平列表，每项含 ``_line_idx``、``level``、
        ``title``、``page_index``、``text``。
    """
    h1_header = next(h for h in all_headers if h["line_idx"] == h1_line and h["level"] == 1)
    raw_h1 = h1_header["raw_title"]
    title = normalize_display_title(raw_h1, 1)
    subs = _headers_in_range(all_headers, h1_line + 1, section_end, min_level=2)

    if not subs:
        text = _join_lines(lines, h1_line, section_end)
        root = {
            "_line_idx": h1_line,
            "title": title,
            "page_index": page_by_line[h1_line],
            "text": text,
        }
        return root, []

    first_sub_line = subs[0]["line_idx"]
    root_text = _join_lines(lines, h1_line, first_sub_line - 1)
    root = {
        "_line_idx": h1_line,
        "title": title,
        "page_index": page_by_line[h1_line],
        "text": root_text,
    }

    # Each subsection's text runs until the immediately following header in this section
    # (any level). This matches VLM PageIndex samples: parent holds only intro before
    # the first child heading; deeper content lives under child nodes.
    flat: list[dict[str, Any]] = []
    for j, h in enumerate(subs):
        lvl = h["level"]
        line_i = h["line_idx"]
        if j + 1 < len(subs):
            next_boundary = subs[j + 1]["line_idx"]
        else:
            next_boundary = section_end + 1
        body = _join_lines(lines, line_i, next_boundary - 1)
        flat.append(
            {
                "_line_idx": line_i,
                "level": lvl,
                "title": normalize_display_title(h["raw_title"], lvl),
                "page_index": page_by_line[line_i],
                "text": body,
            }
        )
    return root, flat


def build_tree_from_flat_nodes(flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将按文档顺序排列的扁平节点（含 ``level``）用栈算法挂成若干子树根列表。

    入参:
        flat: ``build_section_root_and_flat_nodes`` 返回的扁平列表。

    返回:
        森林中「仅子树部分」的根节点列表；每项含 ``_line_idx``、``title``、``page_index``、
        ``text``、``nodes``（可能嵌套）。
    """
    if not flat:
        return []
    stack: list[tuple[dict[str, Any], int]] = []
    roots: list[dict[str, Any]] = []
    for node in flat:
        tree_node: dict[str, Any] = {
            "_line_idx": node["_line_idx"],
            "title": node["title"],
            "page_index": node["page_index"],
            "text": node["text"],
            "nodes": [],
        }
        lvl = node["level"]
        while stack and stack[-1][1] >= lvl:
            stack.pop()
        if not stack:
            roots.append(tree_node)
        else:
            stack[-1][0]["nodes"].append(tree_node)
        stack.append((tree_node, lvl))
    return roots


def _merge_root_and_children(root: dict[str, Any], child_trees: list[dict[str, Any]]) -> dict[str, Any]:
    """把 H1 根字典与段内子树列表合并为一棵完整树节点。

    入参:
        root: 含 ``_line_idx``、``title``、``page_index``、``text`` 的根片段。
        child_trees: ``build_tree_from_flat_nodes`` 的返回值。

    返回:
        带 ``nodes`` 字段的完整树节点（仍含内部字段 ``_line_idx``）。
    """
    out = {
        "_line_idx": root["_line_idx"],
        "title": root["title"],
        "page_index": root["page_index"],
        "text": root["text"],
        "nodes": child_trees,
    }
    return out


def build_forest_from_markdown(lines: list[str], page_by_line: list[int]) -> list[dict[str, Any]]:
    """根据全文行与页码映射，按每个 H1 切段并建树，得到多棵树的列表。

    入参:
        lines: 全文行列表。
        page_by_line: ``parse_page_comments`` 返回值。

    返回:
        森林：每个元素是一棵 H1 为根的树（含 ``_line_idx``、``title``、``page_index``、
        ``text``、``nodes``），尚未分配 ``node_id``。

    异常:
        ValueError: 文中没有 H1 标题时抛出。
    """
    all_h = iter_atx_headers(lines)
    h1s = find_h1_line_indices(all_h)
    if not h1s:
        raise ValueError("No H1 (#) headings found in markdown.")
    forest: list[dict[str, Any]] = []
    for i, h1_line in enumerate(h1s):
        section_end = (h1s[i + 1] - 1) if i + 1 < len(h1s) else (len(lines) - 1)
        root, flat = build_section_root_and_flat_nodes(lines, page_by_line, all_h, h1_line, section_end)
        child_trees = build_tree_from_flat_nodes(flat)
        forest.append(_merge_root_and_children(root, child_trees))
    return forest


def assign_node_ids_preorder(forest: list[dict[str, Any]]) -> None:
    """对整片森林做深度优先先序遍历，依次为节点写入 ``node_id``（``0000``、``0001``、…）。

    入参:
        forest: ``build_forest_from_markdown`` 等返回的树列表；原地修改各节点。

    返回:
        无（``None``）。
    """
    counter = 0

    def visit(n: dict[str, Any]) -> None:
        """单节点递归访问，分配递增 id。"""
        nonlocal counter
        n["node_id"] = str(counter).zfill(4)
        counter += 1
        for c in n.get("nodes") or []:
            visit(c)

    for root in forest:
        visit(root)


def strip_internal_fields(tree: dict[str, Any]) -> dict[str, Any]:
    """去掉内部字段 ``_line_idx``，保留对外 JSON 所需字段（含 ``summary``）。

    入参:
        tree: 已含 ``node_id``、``summary`` 等字段的树节点。

    返回:
        新字典，键为 ``title``、``node_id``、``page_index``、``text``、``summary``；
        若有非空子节点则含 ``nodes``。
    """
    out: dict[str, Any] = {
        "title": tree["title"],
        "node_id": tree["node_id"],
        "page_index": tree["page_index"],
        "text": tree["text"],
        "summary": tree.get("summary", ""),
    }
    nodes = tree.get("nodes") or []
    if nodes:
        out["nodes"] = [strip_internal_fields(ch) for ch in nodes]
    return out


def strip_forest(forest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对森林中每棵树调用 ``strip_internal_fields``。

    入参:
        forest: 内存中的树列表。

    返回:
        用于序列化到 ``result`` 字段的树列表副本。
    """
    return [strip_internal_fields(t) for t in forest]


def _heuristic_doc_summary(forest: list[dict[str, Any]], line_count: int) -> str:
    """无 LLM 时，用行数与各根标题拼一段简短文档摘要。

    入参:
        forest: 内存中的树森林（取每棵根的 ``title``）。
        line_count: 全文行数。

    返回:
        一段中文说明性字符串。
    """
    titles = [t["title"] for t in forest]
    return (
        f"本文档共 {line_count} 行，包含以下主要部分："
        + "、".join(titles[:12])
        + ("等。" if len(titles) > 12 else "。")
    )


async def generate_doc_summary(
    forest: list[dict[str, Any]],
    full_text_sample: str,
    line_count: int,
    *,
    model: str | None,
    llm: Any | None,
    use_llm: bool,
) -> str:
    """生成文档级摘要（顶层 ``summary``）。

    入参:
        forest: 内存树森林，用于提取章节标题列表。
        full_text_sample: 全文或摘录，供 LLM 提示使用（会截断）。
        line_count: 全文行数；无 LLM 时参与启发式摘要。
        model: LLM 模型名。
        llm: 具备 ``acompletion(model, prompt) -> str`` 的异步客户端；可为 None。
        use_llm: 为 True 且 ``llm`` 非空时走模型，否则走 ``_heuristic_doc_summary``。

    返回:
        文档级摘要字符串。
    """
    if not use_llm or llm is None:
        return _heuristic_doc_summary(forest, line_count)
    titles = [t["title"] for t in forest]
    prompt = (
        "请用 2～4 句中文概括以下报告的目的、主要章节要点和结论导向。"
        "不要编造数据，仅根据给出的目录与摘录推断。\n\n"
        f"主要章节标题：{json.dumps(titles, ensure_ascii=False)}\n\n"
        f"正文摘录（可能截断）：\n{full_text_sample[:8000]}"
    )
    return (await llm.acompletion(model=model, prompt=prompt)).strip()


def _heuristic_node_summary(title: str, text: str, max_len: int = 320) -> str:
    """无 LLM 时，将正文压成短摘要（过长则截断加省略号）。

    入参:
        title: 节点标题（正文为空时可作回退）。
        text: 节点正文。
        max_len: 摘要最大字符数。

    返回:
        启发式摘要字符串。
    """
    body = text.replace("\n", " ").strip()
    if len(body) <= max_len:
        return body if body else title
    return (body[: max_len - 1] + "…").strip()


async def generate_node_summaries(
    forest: list[dict[str, Any]],
    *,
    model: str | None,
    llm: Any | None,
    use_llm: bool,
    char_threshold: int,
) -> None:
    """为森林中每个节点原地写入 ``summary``（并发异步调用 LLM 或启发式）。

    入参:
        forest: 内存树森林；原地修改。
        model: LLM 模型名。
        llm: 异步补全客户端；可为 None。
        use_llm: 是否调用模型（否则用截断/全文作摘要）。
        char_threshold: 正文短于此长度则摘要不调模型，直接采用正文或标题。

    返回:
        无（``None``）。
    """
    nodes_flat: list[dict[str, Any]] = []

    def collect(n: dict[str, Any]) -> None:
        """先序收集所有节点到扁平列表。"""
        nodes_flat.append(n)
        for c in n.get("nodes") or []:
            collect(c)

    for r in forest:
        collect(r)

    async def one(n: dict[str, Any]) -> None:
        """为单个节点生成 ``summary``。"""
        text = n.get("text") or ""
        title = n.get("title") or ""
        if len(text) < char_threshold:
            n["summary"] = text if text else title
            return
        if not use_llm or llm is None:
            n["summary"] = _heuristic_node_summary(title, text)
            return
        prompt = (
            "用1～3 句中文概括下面小节内容，保留关键术语与数字；不要添加小节中没有的信息。\n"
            f"标题：{title}\n\n正文：\n{text[:12000]}"
        )
        n["summary"] = (await llm.acompletion(model=model, prompt=prompt)).strip()

    await asyncio.gather(*[one(n) for n in nodes_flat])


async def build_pageindex_payload(
    md_path: str | Path,
    opt: PageIndexOptions | None = None,
    llm_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """从 Markdown 文件路径构建完整 API 载荷（含 ``doc_id``、``line_count``、``summary``、``result``）。

    入参:
        md_path: ``combined_document.md`` 等文件路径。
        opt: 选项；默认 ``PageIndexOptions()``。
        llm_factory: 无参可调用，返回 LLM 客户端；为 None 或创建失败时摘要走启发式/空串。

    返回:
        可 ``json.dumps`` 的字典，键含 ``doc_id``、``status``、``retrieval_ready``、
        ``line_count``、``summary``、``result``。
    """
    opt = opt or PageIndexOptions()
    path = Path(md_path)
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")
    line_count = compute_line_count(content)
    page_by_line = parse_page_comments(lines)

    forest = build_forest_from_markdown(lines, page_by_line)
    assign_node_ids_preorder(forest)

    llm = None
    use_llm = opt.if_add_summary and llm_factory is not None
    if use_llm:
        try:
            llm = llm_factory()
        except Exception:
            llm = None
            use_llm = False

    if opt.if_add_summary:
        await generate_node_summaries(
            forest,
            model=opt.model,
            llm=llm,
            use_llm=use_llm,
            char_threshold=opt.summary_char_threshold,
        )
        doc_summary = await generate_doc_summary(
            forest,
            content,
            line_count,
            model=opt.model,
            llm=llm,
            use_llm=use_llm,
        )
    else:
        for r in forest:
            _clear_summaries(r)
        doc_summary = ""

    doc_id = opt.doc_id
    if not doc_id:
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))

    result_trees = strip_forest(forest)
    return {
        "doc_id": doc_id,
        "status": opt.status,
        "retrieval_ready": opt.retrieval_ready,
        "line_count": line_count,
        "summary": doc_summary,
        "result": result_trees,
    }


def _clear_summaries(n: dict[str, Any]) -> None:
    """递归将节点 ``summary`` 置为空串（关闭摘要生成时使用）。

    入参:
        n: 树节点；原地修改。

    返回:
        无（``None``）。
    """
    n["summary"] = ""
    for c in n.get("nodes") or []:
        _clear_summaries(c)


def sync_build_pageindex_payload(
    md_path: str | Path,
    opt: PageIndexOptions | None = None,
    llm_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """``build_pageindex_payload`` 的同步包装，在内部 ``asyncio.run`` 一次。

    入参:
        md_path: Markdown 文件路径。
        opt: 构建选项。
        llm_factory: LLM 工厂，语义同 ``build_pageindex_payload``。

    返回:
        与 ``build_pageindex_payload`` 相同的字典。
    """
    return asyncio.run(build_pageindex_payload(md_path, opt, llm_factory))


def main_argv(argv: list[str] | None = None) -> None:
    """命令行入口：解析参数，生成 JSON 并写入默认或指定的输出路径。

    入参:
        argv: 参数列表；为 None 时使用 ``sys.argv``。

    返回:
        无；成功时打印写出路径。
    """
    parser = argparse.ArgumentParser(description="Build PageIndex JSON from combined_document.md")
    parser.add_argument("--md", dest="md_path", type=str, default=None)
    parser.add_argument("--out", dest="out_path", type=str, default=None)
    parser.add_argument("--doc-id", type=str, default=None)
    parser.add_argument("--if-add-summary", type=str, default="yes", choices=("yes", "no"))
    parser.add_argument("--summary-threshold", type=int, default=600)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--retrieval-ready", type=str, default="no", choices=("yes", "no"))
    args = parser.parse_args(argv)

    base = Path(__file__).resolve().parent
    md_path = Path(args.md_path) if args.md_path else base / "docs/results_with_vlm/game2025report_v7/combined_document.md"
    out_path = Path(args.out_path) if args.out_path else base / "docs/results_with_vlm/game2025report_v7/out_generated.json"

    def llm_factory() -> Any:
        """构造本工程使用的 Qwen/DashScope 兼容客户端。"""
        from llm import QwenChatClient

        return QwenChatClient()

    opt = PageIndexOptions(
        doc_id=args.doc_id,
        retrieval_ready=args.retrieval_ready.lower() == "yes",
        if_add_summary=args.if_add_summary.lower() == "yes",
        summary_char_threshold=args.summary_threshold,
        model=args.model,
    )

    payload = sync_build_pageindex_payload(md_path, opt, llm_factory=llm_factory)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main_argv()
