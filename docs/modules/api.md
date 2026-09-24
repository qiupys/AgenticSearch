# API 契约（serving 层）

> **代码位置**：`serving/app.py`
> **定位**：模块编排 + HTTP 契约。serving 不实现检索/QP 逻辑，只做：参数解析 → 模块调用 → 诊断组合 → 兜底自愈。前端仅依赖本文档字段，禁止绕过 API 直接耦合模块。

## 端点总览

| 端点 | 方法 | 职责 | 背后模块 |
|---|---|---|---|
| `/predict` | POST | 单条 QP | QP |
| `/batch` | POST | 批量 NER+链指评测 | QP |
| `/search/content` | POST | 端到端内容检索（含兜底自愈） | 召回 + 排序 |
| `/hot` | GET | 今日热点问句（Trends→LLM 筛选，分级缓存） | 独立 |
| `/health` | GET | 存活 + 语料数 + 并发闸门状态 | 状态 |
| `/config/*` | GET/POST | 链路开关与参数（落盘 llm.yaml） | QP/排序 |
| `/metrics` | GET | QP 延迟/层级统计 | QP |
| `/` | GET | 前端单页 | — |

## POST /search/content

**请求**
```json
{
  "query": "AI眼镜 出货量 预测",
  "top": 8,
  "min_score": 0.7,      // 可选：rerank 阈值（排序模块）
  "min_recall": 0.4      // 可选：召回阈值（召回模块）
}
```

**响应（前端依赖的字段，向后兼容）**
```json
{
  "qp": { ... QPOutput.to_dict() ... },
  "results": [{
    "score": 0.89,                    // 排序分 = rerank 原始分（rerank on 时）
    "title": "...", "url": "...", "snippet": "...",
    "content_type": "...", "sentiment": "...", "lang": "zh",
    "date": "2026-09-20",
    "sub_scores": {"bm25":..., "info":..., "ent_link":true, "rerank":0.89},
    "best_chunk": {"chunk_id","doc_id","seq","text","emb"}
  }],
  "diag": {                           // 透传召回+排序诊断（漏斗渲染依据）
    "embedding","token_coverage","missing_entities",
    "n_filtered","n_recall_filtered","n_entity_filtered","entity_boost_hits",
    "n_time_filtered","time_starved","core_entity_coverage",
    "rerank","n_below_threshold","n_dup_filtered"
  },
  "fallback_crawl": {"attempted","new_docs","weak_reason","reason","error"},
  "latency": {"qp_ms","search_ms","crawl_ms","total_ms"}   // 互斥可加
}
```

**兜底自愈逻辑**（serving 编排，非模块职责）：
1. 弱命中判定：无结果 / 覆盖率 < `min_token_coverage` / 有 `missing_entities` /
   `core_entity_coverage=0` / `time_starved`
2. 触发 `scripts/inline_crawl.crawl_and_ingest`（30min TTL 防重爬，进程级锁）
3. `new_docs>0`：失效 lib 计数缓存 → 重跑检索

**并发**：`/search/content` 由 `BoundedSemaphore(8)` 闸门限流（LLM+EMB+rerank 重链路）。

## POST /predict

```json
请求 {"query": "..."}     响应 QPOutput.to_dict()（含 latency_ms.total_ms）
```

## GET /hot

```json
{"source": "google_trends+llm|google_trends+keyword|builtin", "items": ["...", "...", "..."]}
```
分级缓存：真实热点 1h / builtin 回落 10min（自愈）。最多 3 条、每条 ≤14 字。

## GET /health

```json
{"status","tier1_models","ner_weights_loaded","docs","search_active","search_gate","version"}
```
`docs` 为语料行数（60s 缓存；兜底入库后立即失效）。

## 升级守则（API 层）

1. **新增字段自由、删改字段必须过前端**：前端按本文档字段渲染，改字段 = 同步
   `frontend.md` + `serving/static/index.html`。
2. **diag 键是漏斗契约**：召回/排序新增过滤环节上报 `n_xxx_filtered`，漏斗自动多一级
   需前端同步分组；删除环节需同步清理。
3. **分段计时**：新增重阶段需加入 `latency` 互斥分段（可加和 = total）。
4. serving 内部模块导入只允许 `content_search` / `pipeline` / `scripts.inline_crawl`
   三个入口，不深入模块内部文件。
