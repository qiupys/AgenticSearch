# Agent 模块（Agentic 检索循环）

`search/agent/`：在单发检索之上包一层 **ReAct 循环**——QP 理解作为第 0 步，
Planner LLM 每轮决策调用一个工具（库内检索 / web 兜底）或收束作答，
observation（文档 + 检索诊断信号）回传给下一轮决策，最终把编号文档综合成
带 `[n]` 引用的答案。

## 与 QP 主链路的关系（不隔离）

- **第 0 步**：`process_query(query)` 跑一次完整 QP 三路理解，产物三处复用：
  1. 注入 Planner prompt（意图 / 召回通道 / 实体 / 时间摘要），决策有据；
  2. 首轮 `content_search` 检索词与原 query 一致时直接传入 engine（QP 只推理一次）；
  3. 作为返回的 `qp`（前端 QP 面板固定展示原 query 的理解，与子查询无关）。
- **工具即主链路**：`content_search` 走完整 QP→意图通道召回→rerank 链路；
  Planner 改写后的检索词 / 子查询各自走完整 QP（不复用），保证改写被重新理解。
- **反馈修正循环**：observation 携带 `retrieval_signals`（time_starved /
  core_entity_coverage=0 / channels_hit）与文档摘要，Planner 据此改写检索词
  （换别名 / 渠道词 / 拆子查询）或切 web 兜底；与历史完全相同的检索词会被
  拒绝并强制收束（防空转）。

## 接口

```
POST /agent/search   {query, top=6}
→ {answer, citations, steps, results, qp, diag, rounds, degraded, latency, docs_count}
```

- `answer`：综合答案，句末带 `[n]` 引用标注
- `citations`：被引用文档清单（n / title / url / date）
- `results`：**只展示被引用的文档**（每条带 `ref` 编号，与 answer 的 `[n]` 对应，
  字段与 /search/content 对齐——含 sub_scores / best_chunk，前端卡片可展示 chunk）；
  citations 为空（LLM 未标注/摘录兜底）时回落前 top 条。实现：doc_pool 与
  all_docs 首见顺序严格对齐，引用编号 n 直接映射 doc_pool[n-1] 完整文档
- `steps`：推理轨迹（每轮 thought / action / args / observation）
- `degraded`：true 表示 LLM 不可用，已降级为单发检索

## 工具集（双源平权）

| 工具 | 定位 |
|---|---|
| `content_search` | **库内源**：已沉淀内容（公告/研报/新闻存档/公司资料）；完整 QP→通道→rerank 链路；`args.sub_queries` 存在时并行拆查（复合 query 分解）+ URL 去重合并 |
| `web_search` | **web 源**（与库内平权，卡片完全统一）：命中结果过**同一 rerank 模型**按分混排；mini bars 三件套同源——语义（同一 embedding 模型余弦，挂 best_chunk.emb）、关键词（分词覆盖度，web 不在 BM25 索引内）、链指（QP 核心实体命中）；每条附**权威度分级**（3★ 官方权威 / 2★ 专业财经 / 1★ 综合门户 / 0★ 论坛UGC，`tools/inline_crawl.web_authority` 共享词表，非 agent 兜底路同用）、**发布日期**（标题/摘要/URL/正文提取，无则空缺）、**正文开头**（前 4 条并行短超时抓取）；并自动后台清洗入库（自愈沉淀） |
| `answer` | 收束：全部编号文档交 synth prompt 综合作答 |

两个源**无主次之分**，Planner 按 query 特性选择：问"今天/最新/利好哪些板块/
影响分析"→ web 源；问"历史公告/研报观点/已沉淀事件"→ 库内源。某源补充查询
零新增＝该源没有该维度内容 → 立即换另一个源（代码层也有确定性规则：
库内零新增且 web 未用时，免 planner 强制换 web 源补一轮）。

## Planner 决策契约

每轮输出一个 JSON：`{"thought": "...", "action": "工具名", "args": {...}}`。
决策规则：**缺口驱动补全**——每轮先自评现有文档能否回答 query；不能则找出
缺失维度（最新财报/新闻事件/涨跌原因/行业背景）生成针对性补充查询再走
content_search（补充查询是独立 query，各自走完整 QP 与召回通道）；缺时效
内容或改写后仍无结果才 web_search。禁止与历史相同的检索词重复调用（拒绝并
强制收束）；最多 3 轮、循环预算默认 60s（剩余不足以再跑一轮时提前强制收束）。

## LLM 上游（agent 独立 profile）

- `configs/llm.yaml` 的 `agent` 段，`provider: codebuddy` 时走
  `qp/common/llm_client.py::CodeBuddyClient`：CodeBuddy CLI 免费模型（hy3）。
- 协议要点（抓包 codebuddy-cli2api 确认）：
  - 端点 `https://www.codebuddy.cn/v2/chat/completions`，**强制 `stream:true`**
    （非流式 400），SSE chunk 为标准 OpenAI delta 格式，客户端聚合 content；
  - 认证：`Authorization: Bearer` + `X-API-Key` 双头 + CLI 指纹头
    （X-Product: SaaS / X-IDE-Type: CLI 等）；`ck_` 前缀 key 仅 .cn 域有效；
  - hy3 为思考模型：请求带 `enable_thinking: false` 压低思考开销（实测 ~50%），且
    `max_tokens` 必须给足（思考 token 计入额度，400 会全被思考耗尽返回空 content）。
- 总开关：主链路 `switches.agent`（开发者面板可切）+ agent 段配置存在性，
  任一不满足自动降级单发，绝不阻塞主链路。

## 容错设计

- **主链路兜底**：planner 决策与 synth 综合在 agent 路（hy3）失败或解析失败时，
  自动回落主链路 qwen——免费上游单次失败率 ~20-30%，仅两者同时失败才真正降级。
- `CodeBuddyClient.chat`：失败自动重试 1 次（重试前检查剩余时间，不够则快速失败）。
- Planner 某轮彻底失败：已有结果则按已有结果收束；首轮失败降级单发。
- synth 失败（含 qwen 兜底失败）：摘录兜底 `_extractive_answer`（带编号原文摘录，
  仍可提取引用），有文档就保证有答案。
- `LLMClient._extract_json`：容忍 ```json 围栏、前后杂文字，以及**截断 JSON 修复**
  （字符串感知补全引号/括号；最后兜底正则提取 answer 字段）——思考模型把
  max_tokens 耗在 reasoning 上导致 content 截断时，仍能拿到部分可用结果。
- 预算不足：提前强制收束，不丢已有结果；综合（synth）受独立时限约束
  （剩余 <20s 或超上限 45s 时改用毫秒级摘录兜底），端到端最坏 ≈ budget_s + 40s。

## 性能画像与优化路线（2026-10-14 实测）

latency 分解（`latency.planner_ms/tools_ms` + 每步 `ms.{planner,tool}` + `diag.{load,recall,rank,web_wall,web_rerank,web_emb}_ms`）：

| 阶段 | 优化前 | 优化后 | 手段 |
|---|---|---|---|
| QP 第 0 步 | 9~12s（冷启动）/ 3.5s | **~2s**（暖态） | 主链路切 qwen3.8-flash + 服务预热 |
| 工具/轮 | 15~29s | **5~9s** | 见下三条 |
| ├ load_all | 8s/次 | ~0（暖态） | 进程内 mtime 缓存（ingest 落盘自动失效重载） |
| ├ recall 内嵌 | 冷缓存一次 +28s | 后台化 | embed 分批并行（4 线程）+ ingest 线程收尾预热缓存 |
| ├ web 抓正文 | 5s/轮（同步） | 并行隐藏 | content_head 移至后台线程，与 Planner/synth 重叠 |
| └ web 预算 | 15s | 12s | web_wall 实测仅 ~2s，余量为尾部引擎留 |
| Planner | 26~42s（占总时长 36~69%） | **未解决，头号优化项** | step-5-preview 单次决策 9~19s 且方差大 |

**下一步建议（按预期收益排序）**：
1. **Planner 切 qwen3.8-flash**（预期 -25~35s，总时长 → ~25s）：Planner 是纯 JSON
   决策任务，flash 实测单次 ~2s。改法：调换 `_plan_with_fallback` 顺序（flash 优先、
   CodeBuddy 兜底），或 llm.yaml agent 段改 `provider` 走主链路。代价：消耗主链路额度。
2. **同 query 结果缓存（TTL 300s）**：演示场景重复查询直接回放（<100ms）。
3. **QP intent/NER 两次 LLM 并行化**（-1.5s）；启动时预热一发空查询（消冷启动）。

## 已知边界

- planner 决策 9~19s/次（step-5-preview，方差大）、综合 20~30s，当前端到端典型 50~70s；
  免费上游单次失败率 ~20-30%，重试与降级会进一步拉长个别请求。
- 库内无相关文档时 r1 即选 web_search（符合预期）；web 引擎失败则答案缺失、仅透出检索结果。
