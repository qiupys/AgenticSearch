# 召回模块（Recall）

> **负责专家**：召回专家 —— 检索与语料建设
> **代码位置**：`search/engine/corpus.py`（数据契约）+ `search/engine/recall.py`（召回逻辑）
> **独立升级承诺**：只要输入 `(query, qp, corpus, idx, search_cfg)` 输出 `(scored, best_chunk_of_doc, emb_client, diag)` 契约不变，排序模块与 API 无需任何改动。

## 1. 职责

从本地内容库把「可能相关」的文档捞出来，并做召回侧的清洗：

```
BM25 词面路 ──────────┐
chunk emb 语义路 ─────┤
                      ├─ RRF 融合 ─ 意图软加权 ─ 过滤1 信息词噪声
意图专用通道 ─────────┘                        （链指命中豁免）
（QP 每个意图一条独立       ─ 过滤2 召回相关性阈值（emb，链指豁免）
  排序路，通道分=BM25×      ─ 过滤3 核心实体链指硬过滤
  内容类型加成×情绪加成      ─ 链指标记（ent_link，仅展示不参与排序）
  ×实体链指加成）            ─ 过滤4 时效窗软过滤（time_starved 上报）
```

意图通道（v2.0）：`configs/recall_linkage.json` 的 channels 定义每个意图组的
内容类型与时效特征；召回引擎为每个 weight>0 的意图条件生成独立排序路参与
RRF——意图直接驱动召回，而非仅作展示标签。通道命中上报 `diag.channels`。

## 2. 数据契约（corpus.py）

| 数据 | 文件 | 说明 |
|---|---|---|
| corpus | `data/content/corpus.jsonl` | 语料主档；行 = doc（id/title/snippet/url/content_type/sentiment/linked_entities/date） |
| 索引 | `data/content/index.pkl` | `{"bm25": BM25Okapi, "doc_ids": [...]}`（TF-IDF 路已删除） |
| chunks | `data/content/chunks.jsonl` | doc 切块（句边界+overlap 280/50），chunk 侧链指 |
| chunk 向量缓存 | `data/content/chunk_embeddings.npz` | embedding 增量缓存（新 chunk 自动补齐） |

改语料格式 = 改 `corpus.py` = 两个模块同时感知；这是唯一共享点。

## 3. 输入 / 输出契约

```python
# 输入
query       str          原始查询
qp          QPOutput     QP 模块产物（用 entities/time/recall×channel）
corpus/idx  load_all()   数据契约
search_cfg  dict         见下

# 输出（交给排序模块）
scored             list[(score, doc, sub)]      sub: {bm25, info, ent_link?}
best_chunk_of_doc  dict[doc_index -> {chunk_id, doc_id, seq, text, emb}]
emb_client         LLM client 或 None           排序模块复用其 rerank
diag               dict                          见下
```

## 4. 配置项（search_cfg 键）

| 键 | 默认 | 说明 |
|---|---|---|
| `min_recall_score` | 0.4 | 召回相关性阈值（chunk emb 余弦）；**链指命中豁免** |
| `info_idf_quantile` | 0.5 | 信息词 IDF 分位（泛词噪声过滤强度） |
| `min_token_coverage`* | 0.3 | 兜底触发阈值（serving 层使用，非召回过滤） |

\* serving 层组合召回诊断与该配置决定是否触发兜底爬虫。

前端开发者面板：`召回模块 · 相关性阈值` 滑杆 → `min_recall` 请求参数 → `min_recall_score`。

## 5. diag 键（观测契约，前端漏斗依赖）

| 键 | 含义 |
|---|---|
| `embedding` | on / off / fail |
| `token_coverage` | IDF 加权查询词覆盖率（0~1） |
| `missing_entities` | 库内完全缺失的 token（兜底爬虫的补库目标） |
| `n_filtered` | 信息词噪声剔除数 |
| `n_recall_filtered` | 召回相关性低分剔除数 |
| `n_entity_filtered` | 实体硬过滤剔除数 |
| `entity_boost_hits` | 链指命中 doc 数 |
| `n_time_filtered` / `time_starved` | 时窗剔除数 / 时效无命中标志 |
| `core_entity_coverage` | 库内是否有链指命中（0=无，供兜底） |
| `info_tokens` | 本轮信息词（调试用） |
| `channels` | 意图通道命中：{channel: {intent, top[3]}} |

## 6. 升级指南（召回专家必读）

1. **加一路召回**（如向量库/结构化过滤）：在 `recall_candidates` 内新增 rank list 进
   RRF 即可；新路的分数如需展示，加进 `sub` dict（前端 mini bar 自动渲染需同步）。
2. **换 embedding 模型**：删除 `chunk_embeddings.npz` 让缓存重建即可（增量补齐兼容
   不同模型会静默错位——**换模型必须清缓存**）。
3. **改融合算法**（RRF→加权线性等）：只动 `_rrf`；score 语义变化需同步排序模块文档
   （排序输入的 score 仅用于 rerank 前的候选排序，rerank on 时最终顺序与它无关）。
4. **语料重建**：`tools/build_content_library.py`（标题清洗 clean_title 已内置）；
   增量入库走 `tools/inline_crawl.py`（兜底自愈，180s TTL 防重爬）。
5. **回归底线**：`pytest tests/ -q` + 冒烟：一次正常检索（漏斗各级数字合理）+ 一次
   触发兜底（`missing_entities` → 入库 → 复检命中）。

## 7. 已知边界

- 库规模 ~300 docs，单进程内存索引足够；上万 docs 时需迁移向量库（届时只动 corpus.py）。
- chunk emb 缓存与模型强绑定（无模型指纹），换模型必须手动清缓存。
- 时效窗依赖 doc 的 `date` 字段质量（兜底入库时从 HTML/URL 提取，缺失时豁免过滤）。
