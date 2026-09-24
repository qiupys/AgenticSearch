# AgenticSearch — Agent 财经搜索系统

Agent 财经搜索端到端系统：**统一 QP 查询理解（11 类 NER + 三级级联意图）→ 意图驱动多路召回 → rerank 精排 → 两路搜索兜底（库内检索 + web 实时搜索合并透出）**。

> 技术方案：[docs/01_统一QP技术方案.md](docs/01_统一QP技术方案.md) · 模块契约：[docs/modules/architecture.md](docs/modules/architecture.md)
> 指标目标：统计层面意图 85% / 排序层面 95% / 标的·时间识别 99%；级联架构 P99 < 50ms

## 架构总览

```
query ──▶ QP 查询理解（qp/）                ──▶ 检索服务（search/）
          ├ rule_engine  Tier0 规则快通道 <1ms      ├ engine/recall   多路召回（BM25+chunk向量+RRF）
          ├ intent       三级级联意图（规则→轻量→LLM）├ engine/ranking  rerank 精排 + 阈值 + 去重
          ├ ner          11 类实体识别（规则+LLM 级联）├ engine/corpus   语料/索引/chunk（增量构建）
          ├ slots        正交槽位推断                └ service         FastAPI 服务 + 前端单页
          └ recall       意图→召回条件翻译                    │
                                                    弱结果触发两路兜底：
                                                    web 实时搜索（15s 预算）同阈值合并透出
                                                    + 后台质量清洗入库（进度可查）
```

## 目录结构

```
qp/                    查询理解流水线（Query Processing）
  ├ common/            统一输出 schema · query 归一化 · LLM client（全站唯一 LLM 入口）· 自学习
  ├ rule_engine/       Tier0 规则快通道（<1ms，高频意图直出）
  ├ intent/            意图识别：三级级联（规则 → 轻量模型 → LLM 升级）
  ├ ner/               11 类实体识别：正则弱标注 · 规则 NER（gazetteer 1.6 万实体）· LLM 级联
  ├ slots/             正交槽位推断（object_type / time_anchor / market / composite）
  ├ recall/            意图 → 召回条件翻译层（情绪/行业/内容类型/时间窗，配置表驱动）
  └ pipeline/          统一 QP 组装（intents + slots + entities + time + recall → QPOutput）

search/                内容检索服务
  ├ engine/            检索内核：corpus（语料/索引/chunk）+ recall（多路召回+RRF）+ ranking（rerank+阈值）
  └ service/           FastAPI 服务（/predict /search/content /search/ingest_status /hot /health）+ 前端单页

tools/                 工具脚本：数据接入 · 银标注 · 词表构建 · 兜底爬虫 · 评测 runner · 连通性检查
evaluation/            指标体系（16 类 / L1 / L2 / 切片 / NER F1 / 时延）
configs/               全部配置（意图体系 / 规则模式库 / 词库 / LLM / 服务）
tests/                 pytest 单元测试（34 用例）
docs/                  技术方案 + 模块文档（架构 / QP / 召回 / 排序 / API / 前端）
data/                  运行时数据（gitignore，见下方「数据构建」）
```

## 快速开始

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
| `POST /predict` | QP 查询理解：意图 + 11 类实体 + 槽位 + 召回条件 |
| `POST /search/content` | 内容检索：库内多路召回 + rerank；弱结果自动两路兜底（web 搜索合并透出 + 后台清洗入库） |
| `GET /search/ingest_status?query=` | 兜底入库进度（清洗 → 抓正文 → 重建索引 → 完成） |
| `GET /hot` | 热点榜单（百度热搜 → LLM/关键词筛选） |
| `GET /health` / `GET /metrics` | 健康检查 / 运行指标 |

## 两路搜索兜底机制

库内结果触发弱判定（无结果 / 覆盖率低 / 时效空窗 / 核心实体零命中）时：

1. **web 兜底搜索**（15s 硬预算）与库内结果经**同一 rerank 通道、同一阈值**打分，URL 去重合并透出
2. 搜索结果后台**质量清洗入库**（`search/engine/../clean.py`：滚动新闻条剥离、行情模板页拒收、近重复拦截），完成后重试即可从库内直接命中
3. 入库全程进度可查（`/search/ingest_status`），前端实时展示

## 数据构建

`data/` 全部由脚本幂等生成（corpus / 索引 / chunk / 词表 / 模型权重），不入 git：
- `data/entities/` 词表（gazetteer 1.6 万实体，tools/build_*_lexicon 生成）
- `data/content/` 内容库（corpus.jsonl + index.pkl + chunks.jsonl）
- `data/raw|silver|eval|models/` 训练数据与模型权重

测试：`python -m pytest tests/ -q`（34 用例，约 25s）

## 配置说明

| 文件 | 用途 | 是否入库 |
|---|---|---|
| `configs/service.yaml` | 级联/NER/服务/内容库/评测目标 | ✅ |
| `configs/llm.example.yaml` | LLM 配置**脱敏模板** | ✅ |
| `configs/llm.yaml` | 真实 LLM key（从模板复制） | ❌ gitignore |
| `configs/lexicons.yaml` 等 | 词库种子/意图体系/召回联动 | ✅ |

更多设计细节见 [docs/](docs/)；模块接口契约与升级守则见 [docs/modules/architecture.md](docs/modules/architecture.md)。
