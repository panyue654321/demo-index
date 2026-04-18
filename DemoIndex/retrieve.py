"""从 PageIndex 风格 JSON（森林结构）做检索。

与 [docs/design/design_retrieve.md](docs/design/design_retrieve.md) 对齐：

- **词面主路径**：``search_content_by_query`` — 关键词在 ``title`` / ``summary`` / ``text``
  上子串加权打分（``keywords``）。
- **语义扩展（§5）**：``index_document_semantic`` + ``search_content_semantic`` — 默认使用
  **字符 n-gram 哈希嵌入**（无外部模型、可复现）；可注入 ``embed_fn`` 对接真实 embedding。
- **混合**：``search_content_hybrid`` — RRF 融合词面与语义两路排序。
- **辅助**：``get_page_content`` — 按逻辑 ``page_index`` 区间取节点正文。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

try:
    import PyPDF2
except ImportError:  # pragma: no cover - 可选依赖
    PyPDF2 = None

# 关键词场景：中文**不**做停用词过滤；英文仅过滤极常见功能词（避免误伤用户明确给出的检索词）。
_EN_KEYWORD_STOP_MINIMAL: frozenset[str] = frozenset({"the", "a", "an", "or", "and"})

# 分隔符：空格、中英文逗号、顿号、分号、竖线等（设计文档 §4.3.1）。
_KEYWORD_SPLIT_RE = re.compile(r"[\s,，、;；|/\\]+")


def remove_fields(structure: list | dict, fields: list[str]) -> list | dict:
    """递归复制树或森林，并删除指定字段键（如 ``text``），用于导航阶段省 token。

    入参:
        structure: 节点字典、或根节点组成的列表（森林）。
        fields: 要从各层 dict 中剔除的键名列表。

    返回:
        与 ``structure`` 同形的浅层重建结构；列表与 dict 中的嵌套同样递归处理。
    """
    if isinstance(structure, dict):
        return {
            k: remove_fields(v, fields)
            for k, v in structure.items()
            if k not in fields
        }
    if isinstance(structure, list):
        return [remove_fields(item, fields) for item in structure]
    return structure


def normalize_document_payload(
    raw: dict,
    doc_name: str | None = None,
    doc_description: str | None = None,
) -> dict:
    """将 DemoIndex 导出的顶层 JSON 规范为检索用的 ``doc_info``。

    入参:
        raw: 含 ``doc_id``、``result``（森林）、``line_count``、``summary`` 等字段的字典。
        doc_name: 可选展示名；缺省为空字符串。
        doc_description: 可选文档描述；缺省为空字符串。

    返回:
        可供 ``documents[doc_id] = ...`` 使用的 ``doc_info``：含 ``structure``（来自
        ``result``）、``type='markdown'``、``line_count`` 等。
    """
    return {
        "doc_id": raw["doc_id"],
        "structure": raw["result"],
        "line_count": raw.get("line_count", 0),
        "summary": raw.get("summary", ""),
        "type": "markdown",
        "status": raw.get("status", "completed"),
        "retrieval_ready": raw.get("retrieval_ready", False),
        "doc_name": doc_name or "",
        "doc_description": doc_description or "",
    }


def _parse_pages(pages: str) -> list[int]:
    """将 ``pages`` 字符串解析为去重、升序的整数页码列表。

    入参:
        pages: 形如 ``"12"``、``"3,8"``、``"5-7"`` 或组合字符串。

    返回:
        排序后的唯一整数列表。

    抛出:
        ValueError: 区间起点大于终点，或片段无法解析为整数。
    """
    result: list[int] = []
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _count_pages(doc_info: dict) -> int:
    """返回 PDF 文档的总页数（与 PageIndex 语义一致）。

    入参:
        doc_info: 可能含 ``page_count``、缓存 ``pages`` 列表、或 ``path`` 指向 PDF。

    返回:
        总页数；无法解析时返回 ``0``。
    """
    if doc_info.get("page_count") is not None:
        return int(doc_info["page_count"])
    cached = doc_info.get("pages")
    if cached:
        return len(cached)
    path = doc_info.get("path")
    if path and PyPDF2 is not None:
        try:
            with open(path, "rb") as f:
                return len(PyPDF2.PdfReader(f).pages)
        except OSError:
            return 0
    return 0


def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict[str, Any]]:
    """按 1-based 物理页码从 PDF 或缓存中取文本（无 ``node_id`` / ``title``）。

    入参:
        doc_info: 含 ``path`` 或 ``pages`` 缓存（每项 ``page`` + ``content``）。
        page_nums: 目标页码列表。

    返回:
        ``[{"page": int, "content": str}, ...]``，按 ``page_nums`` 中合法页顺序输出。
    """
    cached_pages = doc_info.get("pages")
    if cached_pages:
        page_map = {p["page"]: p["content"] for p in cached_pages}
        return [
            {"page": p, "content": page_map[p]}
            for p in page_nums
            if p in page_map
        ]
    path = doc_info.get("path")
    if not path:
        return []
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is required to read PDF from path")
    with open(path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {
                "page": p,
                "content": pdf_reader.pages[p - 1].extract_text() or "",
            }
            for p in valid_pages
        ]


def _iter_tree_nodes(structure: list | dict) -> Iterator[dict[str, Any]]:
    """深度优先遍历森林中的每个节点 dict。

    入参:
        structure: 根节点列表或单个根 dict。

    生成:
        每个节点字典（含 ``page_index``、``text``、``nodes`` 等）。
    """
    if isinstance(structure, dict):
        yield structure
        children = structure.get("nodes") or []
        for child in children:
            yield from _iter_tree_nodes(child)
        return
    for root in structure:
        yield from _iter_tree_nodes(root)


def _normalize_match_text(s: str) -> str:
    """转为用于子串匹配的规范化文本（英文小写，空白压缩）。"""
    if not s:
        return ""
    t = s.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _count_substring(haystack: str, needle: str) -> int:
    """统计 ``needle`` 在 ``haystack`` 中的非重叠出现次数。"""
    if not needle or not haystack:
        return 0
    return haystack.count(needle)


def _extract_search_terms(keywords: str, *, locale: str = "zh") -> list[str]:
    """将用户输入的**关键词字符串**规范为 search_terms（向量无关、无 embedding）。

    策略：先用 ``空格 / 逗号 / 顿号 / 分号 / 竖线`` 等切分为显式片段（允许单字中文）；
    再从全文抽取英文 ``[a-zA-Z]{2,}`` 与连续汉字块（块长 ``>=4`` 时附加2 字滑窗）。
    中文不做停用词过滤；英文词条仅剔除 ``_EN_KEYWORD_STOP_MINIMAL``。若仍为空，对全部
    汉字做 2 字滑窗回退（设计文档 §4.3.1）。

    入参:
        keywords: 用户提交的一个或多个关键词（可为 ``游戏,下载量`` 等短串）。
        locale: 预留语言提示；当前仅影响文档说明，英文停用词规则对 ``locale`` 均可适用。

    返回:
        去重且保序的词条列表；英文统一小写存入列表以便与正文匹配。
    """
    _ = locale  # 保留签名供设计文档扩展；逻辑上对中英文均用同一套保守英文停用词。
    if not keywords or not str(keywords).strip():
        return []

    raw = str(keywords).strip()
    out: list[str] = []
    seen: set[str] = set()

    def add_term(term: str, *, min_cjk_len: int) -> None:
        t = term.strip()
        if not t:
            return
        if t.isascii():
            tl = t.lower()
            if len(tl) < 2:
                return
            if tl in _EN_KEYWORD_STOP_MINIMAL:
                return
            if tl in seen:
                return
            seen.add(tl)
            out.append(tl)
            return
        if len(t) < min_cjk_len:
            return
        if t in seen:
            return
        seen.add(t)
        out.append(t)

    for seg in _KEYWORD_SPLIT_RE.split(raw):
        add_term(seg, min_cjk_len=1)

    for w in re.findall(r"[a-zA-Z]{2,}", raw.lower()):
        add_term(w, min_cjk_len=1)

    for block in re.findall(r"[\u4e00-\u9fff]{2,}", raw):
        add_term(block, min_cjk_len=2)
        if len(block) >= 4:
            for i in range(len(block) - 1):
                add_term(block[i : i + 2], min_cjk_len=2)

    if not out:
        all_cjk = "".join(re.findall(r"[\u4e00-\u9fff]", raw))
        if len(all_cjk) >= 2:
            for i in range(min(len(all_cjk) - 1, 48)):
                bg = all_cjk[i : i + 2]
                if bg not in seen:
                    seen.add(bg)
                    out.append(bg)

    return out[:64]


def _score_node_for_terms(
    node: dict[str, Any], terms: list[str]
) -> tuple[float, list[str]]:
    """对单个节点按词面命中计分（``title`` 权重 3、``summary`` 权重 2、``text`` 权重 1）。

    每个词条在三个字段上分别做子串计数： ``score = 3 * hits(title) + 2 * hits(summary) + 1 * hits(text)``。

    入参:
        node: 含 ``title``、``summary``、``text`` 等字段的树节点。
        terms: ``_extract_search_terms`` 对用户关键词的输出。

    返回:
        ``(score, matched_terms)``；``matched_terms`` 为至少在某一字段出现过一次的词条，
        顺序与 ``terms`` 一致。
    """
    title = _normalize_match_text(str(node.get("title") or ""))
    summary = _normalize_match_text(str(node.get("summary") or ""))
    text = _normalize_match_text(str(node.get("text") or ""))

    score = 0.0
    matched: list[str] = []
    for term in terms:
        nt = term.lower() if term.isascii() else term
        c_title = _count_substring(title, nt)
        c_summary = _count_substring(summary, nt)
        c_text = _count_substring(text, nt)
        if c_title or c_summary or c_text:
            matched.append(term)
        score += 3.0 * c_title + 2.0 * c_summary + 1.0 * c_text

    return score, matched


def search_content_by_query(
    documents: dict[str, Any],
    doc_id: str,
    keywords: str,
    *,
    top_k: int = 8,
    min_score: float = 0.0,
    locale: str = "zh",
) -> str:
    """关键词检索主接口：遍历 ``structure`` 全树，按词面相关度返回 Top-K 节点正文。

    仅支持 ``type == 'markdown'`` 且含 ``structure`` 的文档；纯 PDF 无树时返回错误 JSON。

    入参:
        documents: ``doc_id -> doc_info``。
        doc_id: 目标文档 ID。
        keywords: 用户输入的关键词字符串（可多词用分隔符分开，不必为完整问句）。
        top_k: 返回条数上限。
        min_score: 低于该得分的节点丢弃。
        locale: 传给 ``_extract_search_terms`` 的语言提示（预留）。

    返回:
        成功时为 JSON 数组字符串，每项含 ``node_id``、``title``、``content``、``page``、
        ``score``、``matched_terms``；失败为 ``{"error": ...}``。
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"}, ensure_ascii=False)

    if doc_info.get("type") == "pdf":
        return json.dumps(
            {
                "error": (
                    "search_content_by_query supports markdown documents with a tree "
                    "structure; PDF-only payloads are not supported."
                )
            },
            ensure_ascii=False,
        )

    terms = _extract_search_terms(keywords, locale=locale)
    if not terms:
        return json.dumps(
            {"error": "No search terms extracted from keywords"},
            ensure_ascii=False,
        )

    structure = doc_info.get("structure") or []
    ranked: list[tuple[float, str, dict[str, Any], list[str]]] = []

    for node in _iter_tree_nodes(structure):
        score, matched = _score_node_for_terms(node, terms)
        if score < min_score:
            continue
        nid = str(node.get("node_id", ""))
        ranked.append((score, nid, node, matched))

    ranked.sort(key=lambda x: (-x[0], x[1]))

    out_rows: list[dict[str, Any]] = []
    for score, _nid, node, matched in ranked[: max(0, top_k)]:
        pi = node.get("page_index")
        row: dict[str, Any] = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "content": node.get("text") or "",
            "score": round(score, 6),
            "matched_terms": matched,
        }
        if pi is not None:
            row["page"] = int(pi)
        out_rows.append(row)

    return json.dumps(out_rows, ensure_ascii=False)


# --- §5 语义检索：默认字符哈希向量（L2 归一化），余弦相似度 =点积 -----------------

_SEMANTIC_INDEX_KEY = "_semantic_index"
_DEFAULT_SEMANTIC_DIM = 256
_SEMANTIC_EMBED_KIND = "char_hash_v1"


def _make_node_passage(node: dict[str, Any]) -> str:
    """将节点拼成一段 passage，供嵌入编码（设计文档 §5.3）。

    入参:
        node: 树节点 dict。

    返回:
        ``title`` / ``summary`` / ``text`` 按固定标签拼接的字符串。
    """
    title = str(node.get("title") or "").strip()
    summary = str(node.get("summary") or "").strip()
    text = str(node.get("text") or "").strip()
    parts: list[str] = []
    if title:
        parts.append(f"[title]\n{title}")
    if summary:
        parts.append(f"[summary]\n{summary}")
    if text:
        parts.append(f"[text]\n{text}")
    return "\n\n".join(parts)


def _character_hash_embedding(text: str, *, dim: int = _DEFAULT_SEMANTIC_DIM) -> list[float]:
    """默认嵌入：字符 + 二字节组的哈希桶计数，再 L2 归一化（无外部模型、可复现）。

    与真实语义模型相比为**弱语义**；便于离线测试与无依赖部署。生产可换用
    ``index_document_semantic(..., embed_fn=...)`` 注入 API / 本地模型向量。

    入参:
        text: 待编码文本。
        dim: 桶维度，须与索引阶段一致。

    返回:
        长度为 ``dim`` 的浮点列表，L2 范数为 1（零向量除外）。
    """
    vec = [0.0] * dim
    if not text:
        return vec
    t = _normalize_match_text(text)

    def _bucket(key: str) -> int:
        h = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).digest()
        return int.from_bytes(h[:4], "big") % dim

    for c in t:
        vec[_bucket(c)] += 1.0
    for i in range(len(t) - 1):
        bg = t[i : i + 2]
        vec[_bucket(bg)] += 1.5
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _cosine_similarity_unit(a: Sequence[float], b: Sequence[float]) -> float:
    """对已 L2 归一化向量计算余弦相似度（点积）。"""
    if len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def _node_index_lookup(structure: list | dict) -> dict[str, dict[str, Any]]:
    """``node_id`` -> 节点 dict。"""
    out: dict[str, dict[str, Any]] = {}
    for node in _iter_tree_nodes(structure):
        nid = str(node.get("node_id", ""))
        if nid:
            out[nid] = node
    return out


def index_document_semantic(
    doc_info: dict[str, Any],
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    dim: int = _DEFAULT_SEMANTIC_DIM,
) -> None:
    """为 ``doc_info`` 构建内存语义索引，写入 ``doc_info["_semantic_index"]``。

    对每个树节点：``passage = _make_node_passage(node)``，``embedding = embed_fn(passage)``。
    默认 ``embed_fn`` 为 ``_character_hash_embedding``（``dim`` 维）。

    入参:
        doc_info: 须含 ``structure``；会被**原地**修改。
        embed_fn: 可选；``text -> 向量``，长度须恒为 ``dim``。
        dim: 向量维度；仅在使用默认嵌入时传入 ``_character_hash_embedding``。

    返回:
       无；索引存在 ``doc_info[_SEMANTIC_INDEX_KEY]``。
    """
    structure = doc_info.get("structure") or []
    nodes = list(_iter_tree_nodes(structure))

    if embed_fn is None:

        def _default_embed(s: str) -> list[float]:
            return _character_hash_embedding(s, dim=dim)

        embed_fn = _default_embed
        used_dim = dim
    else:
        probe = _make_node_passage(nodes[0]) if nodes else ""
        used_dim = len(embed_fn(probe))

    items: list[dict[str, Any]] = []
    for node in nodes:
        passage = _make_node_passage(node)
        vec = embed_fn(passage)
        if len(vec) != used_dim:
            raise ValueError(
                f"Inconsistent embedding length {len(vec)} vs expected {used_dim}"
            )
        items.append(
            {
                "node_id": node.get("node_id", ""),
                "page_index": node.get("page_index"),
                "embedding": list(vec),
            }
        )

    doc_info[_SEMANTIC_INDEX_KEY] = {
        "kind": _SEMANTIC_EMBED_KIND,
        "dim": used_dim if not items else len(items[0]["embedding"]),
        "items": items,
    }


def search_content_semantic(
    documents: dict[str, Any],
    doc_id: str,
    keywords: str,
    *,
    top_k: int = 8,
    min_similarity: float = 0.0,
    embed_fn: Callable[[str], list[float]] | None = None,
    dim: int = _DEFAULT_SEMANTIC_DIM,
) -> str:
    """语义检索：查询与节点 passage 嵌入的余弦相似度 Top-K（设计文档 §5）。

    若 ``doc_info`` 尚无 ``_semantic_index``，则先用默认嵌入**惰性建索引**。
    仅支持 ``type == 'markdown'``。

    入参:
        documents: ``doc_id -> doc_info``。
        doc_id: 目标文档 ID。
        keywords: 与词面检索相同形态的关键词串；整体编码为查询向量。
        top_k: 返回条数上限。
        min_similarity: 余弦相似度阈值 ``[0,1]``（单位向量点积）。
        embed_fn: 若提供且当前无索引，则用于 ``index_document_semantic``；已有索引时
            查询向量须与索引同空间，默认仍用 ``_character_hash_embedding``。
        dim: 仅在建索引用默认嵌入时有效。

    返回:
        JSON 数组，每项含 ``node_id``、``title``、``content``、``page``（若有）、
        ``similarity``；为与 §4 对齐可含 ``score``（等于 ``similarity``）、
        ``matched_terms`` 空列表。失败为 ``{"error": ...}``。
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"}, ensure_ascii=False)

    if doc_info.get("type") == "pdf":
        return json.dumps(
            {
                "error": (
                    "search_content_semantic supports markdown documents with a tree "
                    "structure; PDF-only payloads are not supported."
                )
            },
            ensure_ascii=False,
        )

    kw = str(keywords).strip()
    if not kw:
        return json.dumps(
            {"error": "No keywords provided for semantic search"},
            ensure_ascii=False,
        )

    if doc_info.get(_SEMANTIC_INDEX_KEY) is None:
        index_document_semantic(doc_info, embed_fn=embed_fn, dim=dim)

    idx = doc_info[_SEMANTIC_INDEX_KEY]
    index_dim: int = int(idx.get("dim", _DEFAULT_SEMANTIC_DIM))

    if embed_fn is None:
        qvec = _character_hash_embedding(kw, dim=index_dim)
    else:
        qvec = embed_fn(kw)
        if len(qvec) != index_dim:
            return json.dumps(
                {
                    "error": (
                        f"Query embedding dim {len(qvec)} != index dim {index_dim}; "
                        "rebuild index with the same embed_fn."
                    )
                },
                ensure_ascii=False,
            )

    lookup = _node_index_lookup(doc_info.get("structure") or [])
    ranked: list[tuple[float, str, dict[str, Any]]] = []

    for item in idx["items"]:
        vec = item["embedding"]
        sim = _cosine_similarity_unit(qvec, vec)
        if sim < min_similarity:
            continue
        nid = str(item.get("node_id", ""))
        node = lookup.get(nid, {})
        ranked.append((sim, nid, node))

    ranked.sort(key=lambda x: (-x[0], x[1]))

    out_rows: list[dict[str, Any]] = []
    for sim, _nid, node in ranked[: max(0, top_k)]:
        pi = node.get("page_index")
        row: dict[str, Any] = {
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "content": node.get("text") or "",
            "similarity": round(float(sim), 6),
            "score": round(float(sim), 6),
            "matched_terms": [],
        }
        if pi is not None:
            row["page"] = int(pi)
        out_rows.append(row)

    return json.dumps(out_rows, ensure_ascii=False)


def search_content_hybrid(
    documents: dict[str, Any],
    doc_id: str,
    keywords: str,
    *,
    top_k: int = 8,
    min_score: float = 0.0,
    min_similarity: float = 0.0,
    rrf_k: int = 60,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
    locale: str = "zh",
) -> str:
    """混合检索：对词面 Top与语义 Top 做 RRF 融合（设计文档 §5.6）。

    入参:
        documents: ``doc_id -> doc_info``。
        doc_id: 目标文档 ID。
        keywords: 关键词串。
        top_k: 返回条数。
        min_score / min_similarity: 分别传给词面、语义检索。
        rrf_k: RRF 平滑常数（常见 60）。
        lexical_weight / semantic_weight: 两路 RRF 贡献权重。
        locale: 词面检索语言提示。

    返回:
        JSON 数组；每项含 ``node_id``、``title``、``content``、``page``、``rrf_score``、
        ``lexical_score``、``similarity``（无则 null）。一路失败时尽量返回另一路结果。
    """
    lex_raw = search_content_by_query(
        documents,
        doc_id,
        keywords,
        top_k=max(top_k * 3, 16),
        min_score=min_score,
        locale=locale,
    )
    sem_raw = search_content_semantic(
        documents,
        doc_id,
        keywords,
        top_k=max(top_k * 3, 16),
        min_similarity=min_similarity,
    )

    lex_list: list[dict[str, Any]] | dict[str, Any] = json.loads(lex_raw)
    sem_list: list[dict[str, Any]] | dict[str, Any] = json.loads(sem_raw)

    lex_err = isinstance(lex_list, dict) and "error" in lex_list
    sem_err = isinstance(sem_list, dict) and "error" in sem_list

    if lex_err and sem_err:
        return json.dumps(
            {
                "error": {
                    "lexical": lex_list.get("error"),
                    "semantic": sem_list.get("error"),
                }
            },
            ensure_ascii=False,
        )
    if lex_err:
        sem_list_typed: list[dict[str, Any]] = (
            sem_list if isinstance(sem_list, list) else []
        )
        return json.dumps(sem_list_typed[:top_k], ensure_ascii=False)
    if sem_err:
        lex_list_typed: list[dict[str, Any]] = (
            lex_list if isinstance(lex_list, list) else []
        )
        return json.dumps(lex_list_typed[:top_k], ensure_ascii=False)

    assert isinstance(lex_list, list) and isinstance(sem_list, list)

    rrf: dict[str, float] = {}
    detail: dict[str, dict[str, Any]] = {}

    for r, row in enumerate(lex_list, start=1):
        nid = str(row.get("node_id", ""))
        if not nid:
            continue
        rrf[nid] = rrf.get(nid, 0.0) + lexical_weight / (rrf_k + r)
        if nid not in detail:
            detail[nid] = {
                "node_id": row.get("node_id", ""),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "page": row.get("page"),
                "lexical_score": row.get("score"),
                "similarity": None,
            }

    for r, row in enumerate(sem_list, start=1):
        nid = str(row.get("node_id", ""))
        if not nid:
            continue
        rrf[nid] = rrf.get(nid, 0.0) + semantic_weight / (rrf_k + r)
        d = detail.setdefault(
            nid,
            {
                "node_id": row.get("node_id", ""),
                "title": row.get("title", ""),
                "content": row.get("text", row.get("content", "")),
                "page": row.get("page"),
                "lexical_score": None,
                "similarity": row.get("similarity"),
            },
        )
        d["similarity"] = row.get("similarity")
        if not d.get("content"):
            d["content"] = row.get("content", "")

    merged: list[dict[str, Any]] = []
    for nid, score in sorted(rrf.items(), key=lambda x: (-x[1], x[0]))[:top_k]:
        base = detail[nid]
        merged.append(
            {
                "node_id": base["node_id"],
                "title": base["title"],
                "content": base["content"],
                "page": base.get("page"),
                "rrf_score": round(score, 6),
                "lexical_score": base.get("lexical_score"),
                "similarity": base.get("similarity"),
                "matched_terms": next(
                    (
                        x.get("matched_terms", [])
                        for x in lex_list
                        if str(x.get("node_id")) == nid
                    ),
                    [],
                ),
            }
        )

    return json.dumps(merged, ensure_ascii=False)


def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict[str, Any]]:
    """在 Markdown 树中按 ``page_index`` 闭区间收集节点 ``text``，并带上引用字段。

    入参:
        doc_info: 须含 ``structure``（由 ``result`` 规范化而来）。
        page_nums: 解析后的整数列表；使用 ``min`` 与 ``max`` 形成闭区间。

    返回:
        ``[{"page", "content", "node_id", "title"}, ...]``，按 ``(page, node_id)`` 排序。
    """
    if not page_nums:
        return []
    min_p, max_p = min(page_nums), max(page_nums)
    structure = doc_info.get("structure") or []
    hits: list[dict[str, Any]] = []
    for node in _iter_tree_nodes(structure):
        pi = node.get("page_index")
        if pi is None:
            continue
        if not (min_p <= int(pi) <= max_p):
            continue
        hits.append(
            {
                "page": int(pi),
                "content": node.get("text") or "",
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
            }
        )
    hits.sort(key=lambda x: (x["page"], x["node_id"]))
    return hits


def get_document(documents: dict, doc_id: str) -> str:
    """返回文档元数据的 JSON 字符串（供 Agent 工具使用）。

    入参:
        documents: ``doc_id -> doc_info``。
        doc_id: 目标文档 ID。

    返回:
        JSON 字符串：``doc_id``、``doc_name``、``doc_description``、``type``、``status``；
        PDF 含 ``page_count``，其它类型含 ``line_count``。未找到时为 ``{"error": ...}``。
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    result: dict[str, Any] = {
        "doc_id": doc_id,
        "doc_name": doc_info.get("doc_name", ""),
        "doc_description": doc_info.get("doc_description", ""),
        "type": doc_info.get("type", "markdown"),
        "status": doc_info.get("status", "completed"),
    }
    if doc_info.get("type") == "pdf":
        result["page_count"] = _count_pages(doc_info)
    else:
        result["line_count"] = doc_info.get("line_count", 0)
    return json.dumps(result, ensure_ascii=False)


def get_document_structure(documents: dict, doc_id: str) -> str:
    """返回去掉 ``text`` 后的结构树 JSON 字符串。

    入参:
        documents: ``doc_id -> doc_info``。
        doc_id: 目标文档 ID。

    返回:
        森林或树的 JSON（``ensure_ascii=False``）；错误同 ``get_document``。
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    structure = doc_info.get("structure", [])
    stripped = remove_fields(structure, fields=["text"])
    return json.dumps(stripped, ensure_ascii=False)


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """按 ``pages`` 解析结果拉取 PDF 页文本或 Markdown 节点正文。

    入参:
        documents: ``doc_id -> doc_info``。
        doc_id: 目标文档 ID。
        pages: ``"5-7"``、``"3,8"``、``"12"`` 等形式。

    返回:
        成功时为 JSON 列表字符串；Markdown 每项含 ``page``、``content``、``node_id``、
        ``title``；PDF 每项仅 ``page``、``content``。失败时为 ``{"error": ...}``。
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})

    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps(
            {
                "error": (
                    f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". '
                    f"Error: {e}"
                )
            },
            ensure_ascii=False,
        )

    try:
        if doc_info.get("type") == "pdf":
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps(
            {"error": f"Failed to read page content: {e}"},
            ensure_ascii=False,
        )

    return json.dumps(content, ensure_ascii=False)
