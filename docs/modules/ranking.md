# 排序模块（Ranking）

> **负责专家**：排序专家 —— 相关性建模
> **代码位置**：`content_search/ranking.py`
> **独立升级承诺**：只要输入 `(query, scored, corpus, best_chunk_of_doc, emb_client, search_cfg)` 输出 `(scored, diag)` 契约不变，召回模块与 API 无需任何改动。

## 1. 职责

把召回候选变成用户看到的最终排序结果：

```
召回候选 (scored, best_chunk_of_doc, emb_client)
  │
  ├─ 1. rerank 全候选重排（≤64，输入 = doc 最相关 chunk）
  │     · 展示分 = 排序分 = rerank 原始分（铁律：展示什么就按什么排）
  │     · 链指/频道加成不参与排序，仅作为 sub 标记
  ├─ 2. rerank 相关性阈值过滤（含 API 漏返回的无分文档，一律丢弃；
  │     vec-only 文档——词面零命中仅 embedding 召回——按更高门槛丢弃导航页噪声）
  ├─ 3. 同文去重（URL 参数归一必去重；标题相似：同域 ≥0.6 / 跨域 ≥0.85）
  └─ 4. finalize：截断 top，每条附 best_chunk
```

## 2. 核心设计决定（改动前必读）

| 决定 | 理由 |
|---|---|
| **rerank 覆盖全部候选**（`RERANK_CAND_LIMIT=64`），而非只重排头部 | 只重排头部会让无 rerank 分的尾部文档绕过阈值过滤，兜底场景（高分不足）混入低分结果 |
| **无 rerank 分 = 丢弃**，不回退召回分 | 展示分与排序分必须同源；rerank 引擎挂掉时整体降级（`rerank=fail`），不混两种分数 |
| **排序分 = rerank 原始分** | 链指/频道加成曾导致「显示 0.71 的排在 1.0 前面」的观感错乱；加成仅作 `ent_link` 标记展示 |
| **vec-only 文档用更高门槛**（`vec_only_min_rerank=0.5`） | bm25=0 的纯向量召回噪声率极高：导航页 boilerplate（如"资金流向首页"含"房价/行情"词）、正文侧栏词都可能向量凑近（实测"上海楼市"query 混入东方财富导航页与含楼市侧栏的海底捞文章）；词面零命中 + 中低语义分 ≈ 噪声 |
| **去重在 rerank 之后** | 同文变体 rerank 分相近，去重保留排序最靠前的一条 |

## 3. 输入 / 输出契约

```python
# 输入（来自召回模块）
query              str
scored             list[(score, doc, sub)]
corpus             list[dict]
best_chunk_of_doc  dict[doc_index -> chunk dict]   rerank 输入文本来源
emb_client         LLM client 或 None              None 时 rerank=off 直通
search_cfg         dict

# 输出
scored   list[(rerank_score, doc, sub)]            sub 新增 rerank 键
diag     {rerank: on/off/fail, n_below_threshold, n_vec_only_filtered, n_dup_filtered}
results  finalize() -> [(score, doc, sub, best_chunk), ...]
```

## 4. 配置项

| 键 | 默认 | 前端入口 |
|---|---|---|
| `min_rerank_score` | 0.3 | 开发者面板 `排序模块 · 相关性阈值` 滑杆（`min_score` 请求参数） |
| `vec_only_min_rerank` | 0.5 | 代码默认（bm25=0 文档的 rerank 门槛；调低会放进导航页噪声） |
| `RERANK_CAND_LIMIT` | 64 | 代码常量（改需评估 rerank API 延迟） |

## 5. 升级指南（排序专家必读）

1. **换 rerank 模型**：只动 `common/llm_client.py` 的 rerank 端点配置（provider 无关：
   api_key + api_base + model）；本模块无需改动。
2. **调阈值**：运行时走前端滑杆 / `search_cfg`；默认值改动需同步本文档与前端默认值。
3. **加分排序需求**（如运营加权、时效加权）：如确需引入非 rerank 信号，必须以
   **独立 tiebreaker** 形式加在 `reranked.sort` 的次级键上，禁止乘进主分数——保持
   「展示分 = 排序分」铁律。
4. **去重规则调整**：`n_dup_filtered` 语义不变即可；阈值配置化前先与召回专家确认
   语料侧近重复拦截（`inline_crawl/build_content_library` 的 have_titles）口径一致。
5. **回归底线**：`pytest tests/ -q` + 冒烟：结果 rerank 严格降序、无低于阈值结果、
   漏斗 `n_below_threshold/n_dup_filtered` 数字合理。

## 6. 已知边界

- rerank 输入截断 512 字符（chunk 文本 240 + fallback title+snippet）。
- rerank 引擎不可用时 `rerank=fail`：结果按召回分直通，**不做过滤**（可用性优先），
  前端漏斗引擎状态显示 fail 供排查。
- 候选 >64 时超出部分直接丢弃（当前库规模 ~300、召回截断每路 50，实际触顶概率低）。
