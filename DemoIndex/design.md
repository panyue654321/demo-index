# DemoIndex：从 `combined_document.md` 生成 PageIndex 风格 JSON

本文档描述在 `DemoIndex` 工程中，如何由 `docs/results_with_vlm/game2025report_v7/combined_document.md` 生成与现有 `out.json` **兼容并扩展** 的结构化索引。实现思路**参考** PageIndex-main 的 `page_index_md.py`（标题解析、按层级建树、可选摘要），但**不照搬**其仅适用于纯 MD 目录树的输出形状；需同时支持 VLM 合并稿中的 `<!-- page:N -->` 分页注释与「按一级标题分段的森林」布局。

---

## 1. 目标输出：顶层外壳（envelope）

生成一个 JSON 对象，字段如下。

| 字段 | 类型 | 说明 |
|------|------|------|
| `doc_id` | string (UUID) | 文档标识；可配置为固定值或与路径/内容绑定的确定性 UUID，便于重复生成一致。 |
| `status` | string | 处理状态，例如 `"completed"`。 |
| `retrieval_ready` | boolean | 是否已完成可供下游检索链路消费的校验/后处理。 |
| **`line_count`** | **integer** | **整份 Markdown 的行数**（与 `page_index_md.py` 中 `line_count = markdown_content.count('\n') + 1` 语义一致），表示源 `combined_document.md` 的规模。 |
| **`summary`** | **string** | **整份文档的摘要**（见第 4 节「文档级 summary」），与 PageIndex-main 中可选的 `doc_description` 角色类似，但字段名统一为 `summary`。 |
| `result` | array | 森林：每个元素是一棵树的根节点（见第 2 节）。 |

说明：

- 相比最初只含 `doc_id` / `status` / `retrieval_ready` / `result` 的样例，本设计**显式增加**顶层 **`line_count`** 与 **`summary`**，以满足与 PageIndex-main 对齐的元信息需求。
- 顶层 **`summary`** 表示**全文**；节点上的 **`summary`** 表示**该节**（见第 2 节与第 4.3 节），二者层级不同，互不替代。

---

## 2. `result` 中每个节点的字段

每个节点为对象，字段如下。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `title` | string | 是 | 展示用标题（可对原始 ATX 标题做规范化，如去掉「前言：」等前缀）。 |
| `node_id` | string | 是 | 如 `0000`、`0001`，按约定遍历顺序分配。 |
| `page_index` | integer | 是 | 该节起始位置对应的页码，由正文中最近一次出现的 `<!-- page:N -->` 推导。 |
| `text` | string | 是 | 该节正文片段（从对应标题行起至下一同级/更高级标题前，含中间的 page 注释等）。 |
| **`summary`** | **string** | **条件** | **该节内容的摘要**（见第 4 节「节点级 summary」）；若关闭摘要生成则可设为空字符串或省略（实现阶段二选一并在实现中固定）。 |
| `nodes` | array | 否 | 子节点列表；叶子节点可无此字段或为空数组。 |

与 PageIndex-main 的对应关系：

- `line_num`：本设计**不输出**到 JSON（实现内部仍可用行号）；对外用 `page_index` +结构表达位置。
- **`summary`**：与 `page_index_md.py` 中叶子 `summary`、内部节点 `prefix_summary` 的意图一致；为简化对外 JSON，**统一使用字段名 `summary`**：内部节点可为「本节引言/压缩概括」，叶子为「整段压缩」；若需与原版完全一致的可区分语义，可在后续版本增加 `prefix_summary`，当前设计以单一 `summary` 为主。

---

## 3. 文档结构算法（与实现对齐的约束）

1. **分页映射**  
   扫描 `<!-- page:N -->`，为每一行绑定「当前页码」，供填写 `page_index`。

2. **按一级标题分段（森林）**  
   每个单独成段的 `# ...`（ATX 一级标题，非 `##`）对应 `result` 数组中的**一个根节点**。第一段通常对应主报告；后续 `# 研究方法`、`# 关键看点` 等为并列顶层项。

3. **段内层级树**  
   在每段内，用 `##` / `###` / `####` 等构建父子关系；算法与 `page_index_md.py` 的栈式建树同类：每个标题覆盖从其标题行到「下一个同级或更高级标题」之前的行范围。

4. **标题规范化**  
   `normalize_display_title`：例如 `## 前言：移动游戏勇攀新巅峰` → `title` 为 `移动游戏勇攀新巅峰`；根标题可与正文空格规则对齐（如 `2025年游戏应用洞察报告`）。

5. **`node_id` 分配**  
   对整片森林按约定顺序（深度优先先序、子节点按文档出现顺序）分配 `0000`、`0001`、…，与现有 `out.json` 可对比回归。

6. **代码块**  
   若未来 MD 含 fenced code，标题识别需跳过代码块（与 `page_index_md.py` 一致）。

---

## 4. `summary` 与 `line_count` 的生成约定

### 4.1 `line_count`（顶层）

- 在读取 `combined_document.md` 后计算：  
  `line_count = content.count('\n') + 1`（若文件不以换行结尾，仍与 Python 常见行计数一致）。
- 写入 JSON **根对象**，与 `doc_id`、`status` 等并列。

### 4.2 文档级 `summary`（顶层）

- **输入**：整份 MD 全文或「仅标题树 + 各节短摘录」以控制 token（实现可选）。
- **输出**：一段中文（或与文档语言一致的）短文，概括报告目的、主要章节与结论导向。
- **生成方式**（实现可选其一或组合）：
  - 调用 DemoIndex 已有 LLM 封装（如 `llm.py`）；
  - 或占位策略：首段 H1 下前 N 字 + 各 `result` 根节点 `title` 拼接的启发式摘要（无 API 时降级）。
- 字段名固定为根上的 **`summary`**。

### 4.3 节点级 `summary`（`result` 树中每个节点）

- **叶子节点**：对 `text` 做压缩摘要；若 `text` 短于某 token 阈值，可直接使用 `text` 或略去调用模型（与 `get_node_summary` 思路一致）。
- **内部节点**：可对「标题 + 子节点标题列表 + 首节若干字」生成概括，或对合并后的子摘要再摘要；实现可与 `generate_node_summary` / `prefix_summary` 行为对齐，但 **对外只写 `summary` 一个键**。
- **异步**：若走 LLM，实现可采用 `asyncio.gather` 批量生成（参考 `generate_summaries_for_structure_md`），在 CLI 入口用 `asyncio.run` 统一调度。

### 4.4 与「无 summary」模式的兼容

- 配置项建议：`--if-add-summary`（`yes`/`no`）、`--if-add-doc-summary`（或与顶层 summary 绑定同一开关）。
- 当关闭时：顶层 `summary` 可为 `""`，节点 `summary` 为 `""` 或省略；**`line_count` 仍始终写入**。

---

## 5. 建议文件与函数（更新版）

以下与先前规划一致，仅补充与 **`line_count` / `summary`** 相关的职责。

### 5.1 `md_combined_pageindex.py`（核心库）

| 函数 | 作用 |
|------|------|
| `parse_page_comments(lines)` | 建立行号 → 页码。 |
| `iter_atx_headers(lines)` | 扫描 `#`～`######`，跳过代码块。 |
| `split_h1_sections(...)` | 按一级标题切 `result` 多根。 |
| `normalize_display_title(raw_title, level)` | 生成对外 `title`。 |
| `flat_nodes_for_section(...)` | 段内扁平节点 + `text` +起始 `page_index`。 |
| `build_tree_from_flat_nodes(...)` | 栈式建树。 |
| `assign_node_ids_preorder(forest)` | 分配 `node_id`。 |
| `strip_internal_fields_for_output(tree)` | 仅保留对外字段；**保留 `summary` 键**（由后续步骤填入）。 |
| **`compute_line_count(content) -> int`** | **计算顶层 `line_count`。** |
| **`async generate_doc_summary(...)`** | **生成顶层 `summary`（文档级）。** |
| **`async generate_node_summaries(tree, ...)`** | **为每个节点填 `summary`（可批量异步）。** |
| `build_pageindex_payload(md_path, options) -> dict` | 编排全流程，返回含 **`doc_id`, `status`, `retrieval_ready`, `line_count`, `summary`, `result`** 的字典。 |

### 5.2 `build_game2025_pageindex.py`（CLI）

- 解析 `--md`、`--out`、`--doc-id`、摘要相关开关、模型名等。
- 调用 `build_pageindex_payload`（或 `asyncio.run` 包装），写入 UTF-8 JSON。

---

## 6. 验收要点

- JSON 根级包含 **`line_count`**、**`summary`**，且 **`result` 中节点在开启摘要时含 `summary`**。
- 不开启摘要时：**`line_count` 仍有**；`summary` 行为按第 4.4 节固定一种策略。
- 与现有 `out.json` 对比：**结构、标题、`page_index`、`text`、`nodes` 一致**；新增字段 **`line_count`、顶层 `summary`、节点 `summary`** 为扩展，不破坏既有消费方时，应以后向兼容方式解析（忽略未知键的客户端仍可工作）。

---

## 7. 修订记录

- **2026-04-16**：在信封与节点 schema 中增加 **`line_count`**（顶层）与 **`summary`**（顶层文档摘要 + 每节点摘要）；补充生成约定与验收要点。
