# 设计文档索引（建树与检索）

本目录包含 Markdown 建树脚本说明与检索用法摘要；完整检索协议见 [`design_retrieve.md`](design_retrieve.md)。

---

## `build_md_pageindex.py` 命令行说明

在 **`DemoIndex`** 目录下执行（与 `build_md_pageindex.py` 同级）：

```bash
python build_md_pageindex.py [选项]
```

脚本会从 Markdown 生成 PageIndex 用的 JSON，并写入 `--out` 指定的路径（或下方默认路径）。

---

## 输入与布局

- **`--input-md`**：输入 Markdown 文件路径。  
  **省略** `--input-md` 时，默认使用仓库内  
  `docs/results_with_vlm/game2025report_v7/combined_document.md`。
- **`--layout`**：
  - `h1-forest`（默认）：按一级标题 `#` 分段建树。
  - `page-per-page`：文档根节点 + 每个 `<!-- page:N -->` 逻辑页一个节点（**MinerU 等分页导出稿**用此模式）。每页标题优先页内第一个 `#`，否则第一个 `##`，否则由 LLM/启发式根据正文生成短标题（见 `--page-title-max-chars`）。
- **省略 `--out`** 时，默认写出：  
  `docs/results_with_vlm/game2025report_v7/out_generated.json`。

---

## 常用选项

| 选项 | 说明 |
|------|------|
| `--out` | 输出 JSON 路径（覆盖上述默认）。 |
| `--doc-id` | 写入 JSON 的文档 ID；不填则按输入文件路径自动生成确定性 ID。 |
| `--if-add-summary` | `yes`（默认）或 `no`，是否生成文档级与节点级摘要。 |
| `--summary-threshold` | 摘要相关字符阈值，默认 `600`。 |
| `--page-title-max-chars` | 仅 **`page-per-page`**：某一页没有 `#` 和 `##` 时，用 LLM 生成页标题的最大长度（默认 `50`）；无可用模型时对正文做截断式启发式标题。 |
| `--model` | 调用 LLM 时的模型名（可选，取决于 `llm` 封装）。 |
| `--retrieval-ready` | `yes` 或 `no`（默认），是否标记为检索就绪。 |

---

## 示例

```bash
# 默认：仓库内 combined_document.md → out_generated.json，h1-forest
python build_md_pageindex.py

# 指定合并稿与输出
python build_md_pageindex.py --input-md path/to/combined_document.md --out path/to/out.json

# 分页稿（MinerU 等）：按页建树
python build_md_pageindex.py --input-md path/to/paged_export.md --layout page-per-page --out pageindex.json

# 关闭摘要（节点/文档 summary 为空；若使用 page-per-page 且某页无 #/##，仍会尝试用 LLM 只生成该页标题，除非未配置 llm_factory）
python build_md_pageindex.py --input-md path/to/doc.md --if-add-summary no
```

更完整的设计与字段约定见同目录下的 [`design_md_pageindex.md`](design_md_pageindex.md)。

---

## 用户如何检索（`retrieve.py`）

检索逻辑在仓库根目录下的 [`retrieve.py`](../../retrieve.py)（与 `build_md_pageindex.py` 同级）。**用户输入形态以「关键词串」为主**（单个词或多词，可用空格、中英文逗号、顿号等分隔），不要求先提供页码。

### 1. 注册文档

1. 读入建树产物或合并后的顶层 JSON（含 `doc_id`、`result` 等）。
2. 调用 **`normalize_document_payload(raw, doc_name=..., doc_description=...)`** 得到 `doc_info`。
3. 写入内存字典：**`documents[doc_id] = doc_info`**。

后续所有检索接口均传入同一 **`documents`** 与目标 **`doc_id`**。

### 2. 主路径：关键词词面检索（§4）

- **函数**：`search_content_by_query(documents, doc_id, keywords, top_k=8, min_score=0.0, locale="zh")`
- **返回**：JSON **字符串**；成功时为数组，每项通常含 `node_id`、`title`、`content`（节点正文）、`page`（逻辑页 `page_index`）、`score`、`matched_terms`。
- **适用范围**：仅支持 **`type == 'markdown'`** 且带树形 `structure`；纯 PDF 无树会返回错误 JSON。关键词规范化后若无有效检索词，亦返回错误。

编排层与 Agent **应默认走此接口**，无需用户先猜页码。

### 3. 辅助：元数据、目录、按页精读

| 场景 | 函数 |
|------|------|
| 文档元数据（类型、行数/页数等） | `get_document(documents, doc_id)` |
| 只要目录、不要正文（省 token） | `get_document_structure(documents, doc_id)` |
| **已知**逻辑页表达式，拉取对应节点正文 | `get_page_content(documents, doc_id, pages)`，`pages` 如 `"5-7"`、`"3,8"`、`"12"` |

`get_page_content` 在 Markdown 下成功项一般含 `page`、`content`、`node_id`、`title`；与关键词主路径分工见 [`design_retrieve.md`](design_retrieve.md) §4.5。

### 4. 可选：语义检索与混合（§5）

- **`search_content_semantic(...)`**：对 `keywords` 整体编码为查询向量，与节点 passage 向量做余弦相似度排序；无索引时会惰性构建 **`_semantic_index`**。默认使用无外部模型的字符哈希向量；生产可传入 **`embed_fn`** 对接真实 embedding。
- **`search_content_hybrid(...)`**：并行词面 + 语义，**RRF** 融合（可调 `lexical_weight`、`semantic_weight`、`rrf_k` 等）；一路失败时尽量退回另一路。

若切换 **`embed_fn`** 或模型维度，需清空该文档的 **`doc_info["_semantic_index"]`** 后重建索引，避免查询向量与索引维度不一致。

### 5. 引用与推荐流程

- 引用建议组合：**`doc_id` + `node_id` + `title` + `page`**。
- **推荐流程**：**关键词 → `search_content_by_query`（或 `search_content_hybrid`）→ 需要时再根据结果中的 `page` 收窄并调用 `get_page_content`**。

### 检索 Demo（Python）

在 **`DemoIndex`** 目录下执行：`python demo_retrieve_example.py`（脚本与 `retrieve.py` 同级，已随仓库提供）。亦可复制下方代码自行保存后运行。

下面示例从仓库内的合并稿 JSON 加载文档，依次演示词面检索、混合检索、以及按首条命中节点的逻辑页精读。

```python
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
                print("node_id", segments[0].get("node_id"), "len(content)=", len(segments[0].get("content") or ""))


if __name__ == "__main__":
    main()
```

若使用自建 JSON，在脚本或副本中将 `raw_path` 改为 `build_md_pageindex.py --out` 的产物路径即可。

细则、字段表与混合策略见 [`design_retrieve.md`](design_retrieve.md)。
