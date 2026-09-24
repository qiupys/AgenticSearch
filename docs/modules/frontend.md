# 前端（Web 控制台）

> **代码位置**：`serving/static/index.html`（单文件 SPA：HTML+CSS+JS，无构建步骤）
> **定位**：纯静态、零依赖框架，**只通过 [api.md](api.md) 的 HTTP 契约与服务联动**。改后端字段不改前端的唯一合法场景是纯新增。

## 页面结构（三 tab）

| Tab | 内容 | 依赖端点 |
|---|---|---|
| 检索 | 搜索框 + 热点引导 + QP 解读 + 结果卡片 + 兜底提示条 | `/search/content` `/hot` |
| 评测 | 意图/检索/实体/时效四个评测器 | `/predict` `/batch` `/search/content` `/eval/*` |
| 说明 | 系统说明（实时参数，进入页面时刷新） | `/config` `/health` |

另有两个**覆盖层**（非页面）：开发者侧栏（`devSide`，含阈值滑杆/过滤漏斗/链路开关）、QP JSON 抽屉。

## 与 API 的联动点

| UI 元素 | 数据来源（api.md 字段） | 备注 |
|---|---|---|
| 结果卡片主分数 | `sub_scores.rerank`（有则标「rerank 相关度」）/ `sub_scores.vec`→召回分（无 rerank 时） | 标签随分数体系切换 |
| mini bars（召回/BM25/实体） | `best_chunk.emb` / `sub_scores.bm25` / `sub_scores.ent_link` | 归一化展示 |
| 过滤漏斗 | `diag.n_*` 系列 + `fallback_crawl` | 按「召回模块/排序模块」分组渲染；新增 `n_xxx_filtered` 键自动多一行需手工归组 |
| 处理耗时 | `latency.{qp_ms,search_ms,crawl_ms,total_ms}` | 互斥可加分段；副题展示明细 |
| 兜底提示条 | `fallback_crawl.{new_docs,reason,error}` + `results` | 按真实原因分流，有结果不报失败 |
| 阈值滑杆 ×2 | 请求参数 `min_recall` / `min_score` | 分别写入召回/排序模块的 search_cfg |
| 链路开关 | `/config/switch` POST | 落盘 llm.yaml；文案区分「总闸 vs 调度」 |
| lib 计数 | `/health.docs` | 检索成功后 `refreshLib()` 同步 |
| 热点 chips | `/hot.{source,items}` | 骨架占位防跳变；builtin 回落标「热门检索」 |

## 约定

1. **单文件原则**：所有样式/脚本内联；改完必须过 `node --check`（抽出 `<script>` 校验）。
2. **骨架占位**：异步数据区先渲染等尺寸骨架，返回后一次性替换，避免跳变。
3. **事件委托**：动态内容（如说明卡片展开）用 document 级委托，防止重复挂监听。
4. **强刷交付**：JS/HTML 有缓存，改动交付一律提示 Ctrl+F5。

## 升级指南（前端）

1. **改渲染逻辑前先读 api.md** 响应字段；新增 diag 键 → 漏斗加行；新增结果字段 → 卡片按需展示。
2. **开发者面板分组与模块文档一一对应**：召回模块/排序模块各有阈值与漏斗分组——模块文档改了，面板同步。
3. **移动端**：≤720px / ≤560px 两档断点已有适配；新增 UI 需检查两档表现。
4. 回归：检索页冒烟（正常 + 兜底两种路径）+ 三个 tab 打开无 JS 报错（控制台）。
