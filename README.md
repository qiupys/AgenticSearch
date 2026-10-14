# AgenticSearch — Agent 财经搜索系统

Agent 财经搜索端到端系统：**统一 QP 查询理解（v2.0 三路：意图 / NER / 时间，全部 LLM 优先 + 规则兜底）→ 意图×专用召回通道（每意图一条独立召回路参与 RRF）→ rerank 精排 → 双源检索（库内沉淀 + web 实时平权并行，卡片统一展示）**，并提供 **Agentic 检索循环（ReAct）**：QP 第 0 步理解 → Planner LLM 多轮决策（缺口驱动补查）→ 引用综合作答。

> 技术方案：[docs/01_统一QP技术方案.md](docs/01_统一QP技术方案.md) · 模块契约：[docs/modules/architecture.md](docs/modules/architecture.md)
> 指标目标：统计层面意图 85% / 排序层面 95% / 标的·时间识别 99%；级联架构 P99 < 50ms

## 架构总览

```
query ──▶ QP 查询理解（qp/，三路 LLM 优先）  ──▶ 检索服务（search/）
          ├ 路1 意图  LLM → 轻量模型 → 规则兜底      ├ engine/recall   多路召回（BM25+chunk向量
          ├ 路2 NER   LLM → 规则兜底 → 链指/市场定向 │                  + 意图专用通道 RRF）
          ├ 路三时间  LLM（随路2 合并调用）→ 规则兜底 ├ engine/ranking  rerank 精排 + 阈值
          └ recall    意图 × 专用召回通道翻译        │                  （vec-only 高门槛滤导航页噪声）
                     （news/disclosure/research…）  ├ engine/corpus   语料/索引/chunk（增量构建）
                                                    └ service         FastAPI 服务 + 前端单页
两路搜索：                                              ├ 弱结果触发 web 兜底（同 rerank 通道
  库内沉淀与 web 实时平权并行，                        │  打分合并 + 后台质量清洗入库）
  同一 rerank 模型按分混排；web 文档补齐              └ agent/search    Agentic 循环（见下）
  语义/关键词/链指/权威度/日期，卡片与库内统一
```

## Agentic 检索循环（/agent/search）

QP 理解作为**第 0 步只跑一次**（结果注入 Planner、首轮检索复用），随后 Planner（CodeBuddy
免费模型 step-5-preview，主链路 qwen 兜底）多轮 ReAct 决策：

- **search**：双源并行检索——库内完整链路（可拆子查询）+ web 实时同时执行，URL 去重合并；
  每轮观察含检索诊断信号（时效空窗 / 实体零命中 / 通道命中）
- **answer**：缺口自评足够后收束，编号文档交 LLM 综合作答（hy 与 qwen 双线程竞速），
  句末 `[n]` 引用标注；**results 只返回被引用的文档**（`ref` 与答案编号一一对应）
- LLM 不可用时自动降级单发 + 摘录兜底，保证端到端可用；预算 120s（效果优先）

## 目录结构

```
qp/                    查询理解流水线（v2.0 三路架构，LLM 优先 + 规则兜底）
  ├ common/            统一输出 schema(v2.0) · query 归一化 · LLM client（全站唯一 LLM 入口）· 自学习
  ├ rule_engine/       路1 兜底：Tier0 规则快通道（<1ms，高频意图直出）
  ├ intent/            路1 意图：LLM 优先 → 轻量判别模型 → 规则回落（级联）
  ├ ner/               路2 实体：LLM NER（与路3 合并调用）→ 规则 NER 兜底 → 词表链指（1.6 万实体）→ 市场定向
  ├ time/              路三 时间：时效分桶 + 时间锚点（LLM 随路2 产出，规则兜底）
  ├ recall/            意图 → 专用召回通道翻译（news/disclosure/research/forecast/screener/education/multimedia）
  └ pipeline/          统一 QP 组装（intents + entities + time + recall×channel → QPOutput v2.0）

search/                内容检索服务
  ├ engine/            检索内核：corpus（语料/索引/chunk）+ recall（多路召回+RRF）+ ranking（rerank+阈值）
  └ service/           FastAPI 服务（/predict /search/content /agent/search /search/ingest_status /hot /health）+ 前端单页（Agent 模式 / 推理轨迹 / 统一结果卡片：权威度·日期·chunk·引用标签）

tools/                 工具脚本：数据接入 · 银标注 · 词表构建 · 兜底爬虫（web_search/ingest_hits/quick_fetch/web_authority）· 评测 runner · 连通性检查
evaluation/            指标体系（16 类 / L1 / L2 / 切片 / NER F1 / 时延）
configs/               全部配置（意图体系 / 规则模式库 / 词库 / LLM / 服务）
tests/                 pytest 单元测试（56 用例）
docs/                  技术方案 + 模块文档（架构 / QP / 召回 / 排序 / Agent / API / 前端）
data/                  运行时数据（gitignore，见下方「数据构建」）
```

## 快速开始

**Windows 一键运行**（推荐）：双击或命令行执行 `run.bat`，自动完成依赖检查安装、LLM 配置生成、首次数据构建并启动服务：

```bat
run.bat           REM 一键启动（首次运行自动构建数据，约 10-30 分钟）
run.bat build     REM 强制重建全部数据（幂等）
run.bat test      REM 运行单元测试
```

**手动方式**（Linux/macOS 或需要分步控制时）：

```bash
# 1. 依赖（Python 3.11+）
pip install -r requirements.txt

# 2. 配置：从模板生成 LLM 配置并填入真实 key（唯一含密钥的文件，已 gitignore）
cp configs/llm.example.yaml configs/llm.yaml
#    供应商无关：只需 api_key + api_base + models 三项

# 3. 数据构建（幂等，可重跑；约 10-30 分钟，含 1.5 万股票主数据接入）
python -m tools.build_stock_lexicon      # 股票/指数/行业/宏观词表 + gazetteer
python -m tools.build_topic_lexicon      # 概念/题材/板块指数词表
python -m tools.build_ner_lexicon        # 11 类 NER 词条（COMPANY/ORG/PRODUCT/FUND/METRIC…）
python -m tools.build_content_library    # 内容库抓取入库（BM25 索引）

# 4. 启动服务（默认 :8100）
python -m search.service.app
#    前端单页: http://localhost:8100/
#    API 文档: http://localhost:8100/docs
```

## 核心 API

| 端点 | 说明 |
|---|---|
| `POST /predict` | QP 查询理解：三路输出（意图 + 11 类实体 + 时间要素）+ 意图×召回通道条件 |
| `POST /search/content` | 内容检索：库内多路召回 + rerank；弱结果自动 web 兜底（同一 rerank 通道打分合并，web 文档补齐语义/关键词/链指/权威度/日期，卡片与库内统一） |
| `POST /agent/search` | **Agentic 检索（ReAct 循环）**：QP 第 0 步理解 → Planner 多轮决策（双源并行检索 / 缺口补查）→ 引用综合作答，results 只含被引用文档（见 docs/modules/agent.md） |
| `GET /search/ingest_status?query=` | 兜底入库进度（清洗 → 抓正文 → 重建索引 → 完成） |
| `GET /hot` | 热点榜单（百度热搜 → LLM/关键词筛选） |
| `GET /health` / `GET /metrics` | 健康检查 / 运行指标 |

## 两路搜索兜底机制

库内结果触发弱判定（无结果 / 覆盖率低 / 时效空窗 / 核心实体零命中）时：

1. **web 兜底搜索**（15s 硬预算）与库内结果经**同一 rerank 通道、同一阈值**打分，URL 去重合并透出；
   web 文档按 URL 域名做**权威度分级**（3★ 官方权威 / 2★ 专业财经 / 1★ 综合 / 0★ 论坛UGC，
   `tools/inline_crawl.web_authority`），卡片展示与库内完全统一
2. 搜索结果后台**质量清洗入库**（`search/engine/../clean.py`：滚动新闻条剥离、行情模板页拒收、近重复拦截），完成后重试即可从库内直接命中
3. 入库全程进度可查（`/search/ingest_status`），前端实时展示

## 数据构建

`data/` 全部由脚本幂等生成（corpus / 索引 / chunk / 词表 / 模型权重），不入 git：
- `data/entities/` 词表（gazetteer 1.6 万实体，tools/build_*_lexicon 生成）
- `data/content/` 内容库（corpus.jsonl + index.pkl + chunks.jsonl）
- `data/raw|silver|eval|models/` 训练数据与模型权重

测试：`python -m pytest tests/ -q`（56 用例，约 35s）

## 配置说明

| 文件 | 用途 | 是否入库 |
|---|---|---|
| `configs/service.yaml` | 级联/NER/服务/内容库/评测目标 | ✅ |
| `configs/recall_linkage.json` | 意图×召回通道映射（channels + intent_to_recall） | ✅ |
| `configs/llm.example.yaml` | LLM 配置**脱敏模板** | ✅ |
| `configs/llm.yaml` | 真实 LLM key（从模板复制） | ❌ gitignore |
| `configs/lexicons.yaml` 等 | 词库种子/意图体系/召回联动 | ✅ |

### LLM 底层两种模式（api_key / api_base 接口不变）

1. **阿里云百炼**（默认模板）：api_base 直填 compatible-mode 地址，chat/embedding/rerank 全量可用（chat 现为 qwen3.8-flash，客户端固定 `enable_thinking: false` 关闭思考模式）。
2. **opencode Zen 免费模型**：`api_base: http://127.0.0.1:8180/v1` 指向本地代理 `tools/llm_proxy.py`，代理转发到 `https://opencode.ai/zen/v1`（出站走 v2rayN 代理）。key 自动读 `opencode auth login` 的 auth.json，或 `--api-key`/`OPENCODE_API_KEY` 传入。Zen 无 embeddings/rerank，`embedding_recall`/`rerank` 开关需置 false（自动降级）。`run.bat` 检测到 llm.yaml 指向本地代理时会自动拉起代理。

### Agent 路 LLM（独立 profile，configs/llm.yaml agent 段）

Agentic 循环的 Planner/收束走 CodeBuddy CLI 免费模型（`provider: codebuddy`，www.codebuddy.cn
SSE 流式，国内直连），与主链路 key/额度隔离；主链路 qwen 作为 Planner 兜底与 synth 竞速第二路。
免费上游单次失败率较高，客户端已做截断感知重试（finish=length 自动加倍 max_tokens）。

更多设计细节见 [docs/](docs/)；模块接口契约与升级守则见 [docs/modules/architecture.md](docs/modules/architecture.md)。
