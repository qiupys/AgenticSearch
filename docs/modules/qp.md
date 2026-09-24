# QP 模块（Query Processing）

> **负责专家**：QP 专家 —— 意图体系与级联架构
> **代码位置**：`rule_engine/ intent/ ner/ slots/ recall/ pipeline/ common/ evaluation/`
> **独立升级承诺**：只要 `QPOutput` 字段向后兼容，QP 内部任何改动（换模型/调级联/加槽位）不影响召回与排序模块。

## 1. 职责

把一条自然语言 query 变成结构化的检索意图：

| 子模块 | 职责 | 关键文件 |
|---|---|---|
| rule_engine | Tier0 规则快通道（<1ms，高频/SYSTEM 意图直出） | `rule_engine/fast_path.py` |
| intent | 三级级联：规则 → 轻量判别模型 → LLM；顺序由开关控制 | `intent/cascade.py` `intent/lightweight.py` `intent/llm_intent.py` |
| ner | 实体识别 + 词表链指（正则弱标注 / BiLSTM-CRF / LLM NER 兜底） | `ner/predictor.py` `ner/entity_linker.py` |
| slots | 正交槽位推断（object_type / time_anchor / composite） | `slots/extractor.py` |
| recall/ | 意图 → 召回条件翻译（情绪/行业/内容类型/时间窗，配置表驱动） | `recall/linkage.py` |
| pipeline | 统一组装 QPOutput | `pipeline/app.py` |

## 2. 输出契约（QPOutput）

下游（召回模块）依赖的字段——**改名/删除 = 破坏契约**：

```
QPOutput
├── intents      [(intent_id, domain)]      8 域 × 42 二级
├── slots        {object_type, time_anchor, composite, ...}
├── entities     [Entity(text, type, source, eid, canonical)]
│                · type ∈ COMPANY/STOCK/STOCK_CODE/INDEX/INDUSTRY/FUND/MACRO/ORG/PRODUCT
│                · eid 非空 = 可链指核心实体（召回模块据此做硬过滤/豁免）
├── recall       [RecallCond]  召回条件：role / content_type_pref / sentiment /
│                              time_window_days / time_channel
├── time         {bucket, anchor}          显式时效桶
├── tier / escalated                        级联层级（评测/展示用）
└── latency_ms   {total_ms}                 QP 阶段自报耗时（API 分段计时用）
```

## 3. 级联与开关

```
Tier0 规则 ──SYSTEM 域命中──► 直出
   │ 其余命中留作回落候选
   ▼
llm_first=on 且 llm_intent=on ──► LLM 意图（qwen3.8-27b, tier2）
   │ 失败/低置信(<0.55)            ▲
   ▼                               │ llm_first=off 时 LLM 退到这里
轻量判别模型（tier1, accept_threshold）│
   │ 不足                          │
   ▼                               │
规则回落 ──────────────────────────┘
   ▼ 仍无 → fallback_intent（tier2, escalated）
```

两个 LLM 开关（`configs/llm.yaml` switches，运行时经 `/config/switch` 切换）：

| 开关 | 语义 | 关闭效果 |
|---|---|---|
| `llm_intent` | **总闸**：LLM 能否参与意图判定 | 级联只剩规则+轻量模型 |
| `llm_first` | **调度顺序**：on=LLM 最前；off=LLM 只做兜底 | 改变的是顺序，不是能力 |

## 4. 配置

| 配置 | 位置 | 说明 |
|---|---|---|
| 意图体系（8 域 × 42 二级 + 槽位） | `configs/` 意图体系表 | 改意图 = 改 taxonomy，需重训轻量模型 |
| 规则模式库 | `configs/rule_patterns.yaml` | Tier0/兜底规则，热加载 |
| 级联参数（rule_min_conf / accept_threshold / llm_min_confidence） | `configs/llm.yaml` cascade | 各层置信门槛 |
| 开关 | `configs/llm.yaml` switches | llm_intent / llm_first / llm_ner |

## 5. 升级指南（QP 专家必读）

1. **换意图模型**：只动 `intent/`；保持 `process_query` 签名与 QPOutput 字段。新模型
   先过 `scripts/run_eval.py`（L1≥96% / L2≥88%）再替换。
2. **加新槽位/实体类型**：`slots/`、`ner/` 内自治；新增 entity type 需同步召回模块的
   核心实体集合（`content_search/recall.py` 的 `q_core_eids`）——这是唯一的跨模块约定。
3. **调级联顺序/阈值**：`configs/llm.yaml`，无需改代码；调整后跑 NER/意图评测确认。
4. **训练链路**：`scripts/ingest_data.py → build_silver.py → train_lightweight.py →
   build_evalset.py → run_eval.py`（详见 README 快速开始）。
5. **回归底线**：`pytest tests/ -q`（22 用例）+ 冒烟：`/predict` 逐 tier 至少一条样本。

## 6. 已知边界

- 轻量模型为字符 n-gram LR（0.6B 之下），冷启动快但精度有限，靠级联兜底。
- NER 权重未训练时 `ner_weights_loaded=false`，实体识别退化到词典+LLM NER。
- 覆盖率/缺失 token 诊断（token_coverage、missing_entities）由**召回模块**计算，
  QP 不负责——兜底触发条件在 serving 层组合两者输出。
