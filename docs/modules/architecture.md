# 架构总览（Agentic Search）

> 模块化目标：**QP / 召回 / 排序三个核心模块完全解耦，可独立升级**。
> 每个模块有独立文档（本目录）、独立配置项、清晰的接口契约；前端独立，仅通过 API 契约联动。

## 模块划分与负责专家

| 模块 | 代码位置 | 负责专家 | 文档 |
|---|---|---|---|
| QP（v2.0 三路：意图+实体+时间 → 召回通道） | `rule_engine/ intent/ ner/ time/ recall/ pipeline/ common/` | **QP 专家**（意图体系与级联架构） | [qp.md](qp.md) |
| 召回（多路召回 + 意图通道 RRF + 召回侧过滤） | `search/engine/corpus.py search/engine/recall.py` | **召回专家**（检索与语料建设） | [recall.md](recall.md) |
| 排序（Rerank 精排 + 阈值 + 去重） | `search/engine/ranking.py` | **排序专家**（相关性建模） | [ranking.md](ranking.md) |
| API 服务层（模块编排 + HTTP 契约） | `search/service/app.py` | 全栈（随模块升级同步） | [api.md](api.md) |
| Agent 循环（ReAct：Planner 决策 + 工具调用 + 引用综合作答） | `search/agent/` | Agent 专家（自主决策与工具编排） | [agent.md](agent.md) |
| 前端 Web 控制台 | `search/service/static/index.html` | 前端 | [frontend.md](frontend.md) |

## 端到端链路

```
query
  │
  ▼
┌────────────── QP 模块（v2.0 三路，LLM 优先）──────────┐
│ 路1 意图：Tier0 规则 → LLM → 轻量模型 → 兜底          │
│ 路2 实体：LLM NER → 规则 NER 兜底 → 链指 → 市场定向   │
│ 路三时间：LLM（随路2 合并调用）→ 规则时间路兜底        │
│ → recall/ 翻译：意图 × 专用召回通道（channel）         │
│ → pipeline 组装 QPOutput                              │
└──────────────┬────────────────────────────────────────┘
               │ QPOutput（intents/entities/time/recall×channel）
               ▼
┌────────────── 召回模块 ───────────────────────────────┐
│ corpus.load_all()                                     │
│ BM25 词面路 + chunk emb 语义路                        │
│ + 意图专用通道（每意图一条独立排序路）                │
│ → RRF 融合 → 意图软加权（内容类型/情绪）              │
│ → 信息词噪声过滤（链指豁免）                          │
│ → 召回相关性阈值（emb，链指豁免）                     │
│ → 核心实体链指硬过滤 → 市场定向                       │
│ → 时效窗软过滤（time_starved 上报）                   │
└──────────────┬────────────────────────────────────────┘
               │ (scored, best_chunk_of_doc, emb_client, diag)
               ▼
┌────────────── 排序模块 ───────────────────────────────┐
│ rerank 全候选重排（≤64）                              │
│ → rerank 相关性阈值过滤                               │
│ → 同文去重（URL/标题相似）                            │
│ → finalize: 截断 top + 附 best_chunk                  │
└──────────────┬────────────────────────────────────────┘
               │ results[(score, doc, sub, best_chunk)] + diag
               ▼
┌────────────── API 服务层 ─────────────────────────────┐
│ search/service/app.py                                 │
│ /predict(QP) /search/content(检索) /hot /health …     │
│ /agent/search：Agentic ReAct 循环（可选增强路）       │
│   Planner(hy3) ⇄ 工具(库内检索/web兜底) → 引用综合作答 │
│ 兜底爬虫自愈（tools/inline_crawl）                    │
└──────────────┬────────────────────────────────────────┘
               ▼
        前端 index.html（单文件 SPA，仅依赖 API 契约）
```

## 模块间契约（解耦的关键）

跨模块**只允许**通过以下函数签名耦合，升级任一模块保持签名与 diag 键不变：

```python
# QP -> 召回
from qp.pipeline.app import process_query
qp = process_query(query)                 # QPOutput v2.0

# 召回（search/engine/recall.py）
scored, best_chunk_of_doc, emb_client, diag = recall_candidates(
    query, qp, corpus, idx, search_cfg)

# 排序（search/engine/ranking.py）
scored, diag_add = rank_candidates(
    query, scored, corpus, best_chunk_of_doc, emb_client, search_cfg)
results = finalize(scored, corpus, best_chunk_of_doc, top)
```

- **QP → 召回**：`QPOutput`（intents/entities/time/recall×channel）。QP 升级
  （换意图模型）不影响召回，只要 QPOutput 字段兼容。
- **召回 → 排序**：`list[(score, doc, sub)]` + `best_chunk_of_doc` + `emb_client` +
  `diag`。召回升级（加一路召回、改融合算法）不影响排序。
- **排序 → API**：`results` + `diag`。排序升级（换 rerank 模型、调阈值）不影响 API。

## 独立升级守则

1. **改前读文档**：每个模块的行为、参数、diag 键以自己的 md 为准。
2. **保持契约**：函数签名、diag 键、API 字段向后兼容；新增字段允许，删除/改名需同步
   下游模块文档与前端。
3. **模块内自治**：阈值、开关、缓存等参数只写进自己的 `search_cfg` 键 / `configs/*.yaml`。
4. **回归底线**：`pytest tests/ -q` 全绿 + 手工冒烟（一次正常检索 + 一次触发兜底）。
5. **diag 是观测契约**：前端漏斗按 diag 键渲染，新增过滤环节必须上报 `n_xxx_filtered`；
   意图通道命中上报 `diag.channels`。

## 目录速查

```
qp/               查询理解（v2.0 三路）
  rule_engine/      路1: Tier0 规则快通道
  intent/           路1: 意图级联（LLM 优先 → 轻量模型 → 规则）
  ner/              路2: 实体识别（LLM 优先 → 规则兜底）+ 链指
  time/             路三: 时间锚点与时效分桶（LLM 随路2 产出，规则兜底）
  recall/           意图→召回通道翻译（注意与 search/engine/recall 区分：
                    前者产出"要什么+走哪个通道"，后者执行"去哪拿"）
  pipeline/         统一组装 QPOutput
  common/           共享：llm_client(所有 LLM 调用唯一入口)/schema/normalize
search/engine/    检索子系统：corpus(数据) + recall(召回) + ranking(排序)
search/service/   API + 前端静态资源
tools/            数据/训练/兜底爬虫(inline_crawl)/内容库构建/评测
configs/          llm.yaml(开关+模型) service.yaml(服务+检索参数)
                  rule_patterns.yaml recall_linkage.json(意图×通道映射)
```
