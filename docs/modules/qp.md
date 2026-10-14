# QP 模块（Query Processing，v2.0 三路架构）

> **负责专家**：QP 专家 —— 意图体系与级联架构
> **代码位置**：`rule_engine/ intent/ ner/ time/ recall/ pipeline/ common/`
> **独立升级承诺**：只要 `QPOutput` 字段向后兼容，QP 内部任何改动（换模型/调级联）不影响召回与排序模块。

## 1. 职责：三路并行产出

一条 query 一次推理，三路各自 LLM 优先、失败回落规则：

| 路 | 职责 | 主路径（LLM 优先） | 兜底路径 | 关键文件 |
|---|---|---|---|---|
| 路1 意图 | 8 域 × 42 二级意图 | LLM（llm_first + llm_intent） | Tier0 规则 → 轻量模型 → fallback | `intent/cascade.py` `rule_engine/fast_path.py` |
| 路2 实体 | 11 类 NER + 词表链指 + 市场定向 | LLM（llm_ner，与路3 合并一次调用） | 规则 NER（核心类型互补合并） | `ner/predictor.py` `ner/llm_ner.py` `ner/entity_linker.py` |
| 路三 时间 | 时效分桶 + 时间锚点 | LLM（随路2 同一次调用产出） | 规则时间路（`time/extractor.py`） | `time/extractor.py` |

三路产物由 `recall/linkage.py` 编译为带**专用召回 channel** 的召回条件
（意图不再只是打标签，直接驱动召回——见 §3）。

> v2.0 变更：`slots/` 模块已删除。modality/output_form/trigger/complexity 槽位
> 无消费方，移除；object_type 由 NER 实体内联推导（`recall/linkage.py: object_type_of`）；
> time 抽出为独立三路 `time/`。

## 2. 输出契约（QPOutput v2.0）

下游（召回模块）依赖的字段——**改名/删除 = 破坏契约**：

```
QPOutput (version 2.0)
├── intents      [Intent(intent_id, domain, confidence, source)]
│                · 8 域 × 42 二级；source ∈ rule/model/llm/fallback
├── entities     [Entity(text, type, start, end, source, eid, canonical, market)]
│                · type ∈ COMPANY/STOCK/STOCK_CODE/INDEX/INDUSTRY/FUND/MACRO/ORG/PRODUCT/TIME/METRIC
│                · eid 非空 = 可链指核心实体（召回模块据此做硬过滤/豁免）
│                · market ∈ HK/US/A（两地上市实体消歧后写入）
├── time         TimeInfo(anchor, raw, normalized, bucket)
│                · anchor ∈ 最新/历史时点/区间/未来/无
│                · bucket ∈ in_one_day/one_week/one_month/one_year/no_limit
├── recall       [RecallCondition]  意图×召回通道条件
│                · channel ∈ news/disclosure/research/forecast/screener/education/multimedia/none
│                · role / weight / sentiment / content_type_pref / industry /
│                  time_window_days / time_channel / time_bucket
├── tier / escalated                        级联层级（评测/展示用）
├── stats_category_16                       产品 16 类统计口径
└── latency_ms   {intent_ms, ner_ms, time_ms, recall_ms, total_ms}
```

## 3. 意图 → 召回通道（v2.0 核心）

每个意图映射一个专用召回 channel（`configs/recall_linkage.json` channels 定义），
召回引擎为每个 channel 生成**独立排序路**参与 RRF 融合：

| channel | 意图组 | 内容类型 | 时效特征 |
|---|---|---|---|
| news | INFO-01~07（资讯/动态） | news, article | freshness 短窗 |
| disclosure | INFO-04/08（公告/财报） | announcement, report | 长窗 90d |
| research | ANALYZE-01~07（分析/点评） | report, article, review | 30d, neutral 优先 |
| forecast | PREDICT-01~06（预测/展望） | news, article | 30d, 多样性 |
| screener | SELECT-01~04（选品/受益） | report, review, news | 30d, 正向优先 |
| education | SERVICE/QUANT/TRADE（知识/策略） | tutorial, qa, encyclopedia | 不限时效 |
| multimedia | INFO-09（视频/音频） | video_*, audio, gallery | 不限时效 |
| none | SYSTEM-01~04 | 不走内容召回 | — |

## 4. 级联与开关

```
Tier0 规则 ──SYSTEM 域命中──► 直出
   │ 其余命中留作回落候选
   ▼
llm_first=on 且 llm_intent=on ──► LLM 意图（qwen3.8-flash）
   │ 失败/低置信(<0.55)            ▲
   ▼                               │ llm_first=off 时 LLM 退到这里
轻量判别模型（tier1, accept_threshold）│
   │ 不足                          │
   ▼                               │
规则回落 ──────────────────────────┘
   ▼ 仍无 → fallback_intent（escalated）
```

LLM 开关（`configs/llm.yaml` switches，运行时经 `/config/switch` 切换）：

| 开关 | 语义 | 关闭效果 |
|---|---|---|
| `llm_intent` / `llm_first` | 路1：LLM 参与意图 + 调度顺序 | 级联只剩规则+轻量模型 |
| `llm_ner` | 路2+路3：LLM 实体+时间合一调用 | 规则 NER + 规则时间路 |

## 5. 配置

| 配置 | 位置 | 说明 |
|---|---|---|
| 意图体系（8 域 × 42 二级） | `configs/intent_taxonomy.json` | 改意图 = 改 taxonomy，需重训轻量模型 |
| 规则模式库 | `configs/rule_patterns.yaml` | Tier0/兜底规则 |
| 级联参数 | `configs/service.yaml` cascade | 各层置信门槛 |
| 意图×召回通道映射 | `configs/recall_linkage.json` | channels + intent_to_recall |
| 开关 | `configs/llm.yaml` switches | llm_intent / llm_first / llm_ner |

## 6. 升级指南

1. **换意图模型**：只动 `intent/`；保持 `process_query` 签名与 QPOutput 字段。
   新模型先过 `tools/run_eval.py`（L1≥96% / L2≥88%）再替换。
2. **加实体类型**：`ner/` 内自治；新增 entity type 需同步召回模块的核心实体集合
   （`search/engine/recall.py` 的 `q_core_eids`）——这是唯一的跨模块约定。
3. **调意图通道**：`configs/recall_linkage.json`（channels + intent_to_recall），
   热加载（lru_cache 进程级）；新增 channel 需同步召回引擎的通道加成逻辑
   （`recall.py` 路3 段）。
4. **训练链路**：`tools/ingest_data.py → build_silver.py → train_lightweight.py →
   build_evalset.py → run_eval.py`。
5. **回归底线**：`pytest tests/ -q`（35 用例）+ 冒烟：`/predict` 三路输出齐全
   （intents/entities/time/recall 带 channel）。

## 7. 已知边界

- 轻量模型为字符 n-gram LR，冷启动快但精度有限，靠级联兜底。
- NER 权重未训练时 `ner_weights_loaded=false`，实体识别退化到词典+LLM NER。
- 三路 LLM 优先意味着 QP 延迟 ≈ 2 次 LLM 调用（意图 1 次 + NER/时间 1 次），
  约 2-4s；关闭 llm_* 开关可整体降级为纯规则（<10ms）。
- 覆盖率/缺失 token 诊断（token_coverage、missing_entities）由**召回模块**计算，
  QP 不负责——兜底触发条件在 serving 层组合两者输出。
