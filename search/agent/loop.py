# -*- coding: utf-8 -*-
"""Agentic 检索循环实现（ReAct）。

契约（search/agent/agent_search）：
  输入  query: str, top: int, max_rounds: int, budget_s: float
  输出  {query, answer, citations, steps, results, qp, diag, rounds,
         degraded, latency}

架构（与 QP 主链路不隔离，QP 是循环的第 0 步）：
  第 0 步  QP 理解（process_query，三路 LLM 优先）——只跑一次；
           理解结果注入 Planner prompt（意图/通道/实体/时间），
           首轮 content_search 复用该 QP（不重复推理），
           并作为返回的 qp（前端 QP 面板固定展示原 query 的理解）。
  循环     Planner 决策 → 工具（库内检索 / web 检索，双源平权）→ observation
           （文档 + 检索诊断信号 time_starved / 实体零命中）回传 → 收束。

工具集（Planner 可调用，按信息特性选源、无主次）：
  content_search(args)  库内沉淀内容：完整 QP→通道召回→rerank 链路（复用第 0 步 QP）；
                        args.sub_queries 存在时并行拆查（复合 query 分解，
                        子查询各自走完整 QP）
  web_search(args)      web 实时检索：最新消息/市场反应/分析解读；结果后台清洗入库
  answer(args)          收束（由主循环触发综合）
"""
from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ACTIONS = ("search", "answer")

_PLANNER_SYSTEM = "你是搜索 Agent 的决策器。只输出一个 JSON 对象，不要输出任何其他文字。"

_PLANNER_PROMPT = """你是财经搜索 Agent。每轮决策：检索（双源并行）或收束作答。

工具：
1. search  双源并行检索——库内沉淀（公告/研报/新闻存档/公司资料）与 web 实时
   （最新消息/市场反应/分析解读）同时执行，结果合并并标注来源（source 字段）。
   args: {"query": "检索词", "top": 6}
   - 对比/多实体类 query 可拆子查询并行：args: {"sub_queries": ["q1", "q2"]}
2. answer  结果足够，收束作答。args: {}

你不需要选择信息源——两个源每轮自动并行；你的职责是生成高质量的检索词
（补实体/补维度/换角度），并根据返回结果判断是否已能回答。

QP 理解（第 0 步已完成，检索会直接复用该理解，不要重复理解 query）：
{qp_block}

决策规则：
- 每轮输出 {{"thought": "一句话分析", "action": "工具名", "args": {{...}}}}
- 每轮先自评：现有文档能否回答 query？
  - 能 → action=answer，不要贪多
  - 不能 → 找出缺失的信息维度（最新动态/财报业绩/涨跌原因/利好板块/行业影响），
    生成针对性检索词再 search（新检索词仍会双源并行）
  - 两个源都零新增（n_new_docs=0）＝该维度两个源都没有 → 换维度，或收束时说明缺口
- 不要与历史完全相同的检索词重复调用（会被拒绝并强制收束）
- 最多 3 轮，轮次用完前必须收束

用户 query: {query}

已执行轮次与观察：
{history}

只输出 JSON 对象:"""

_SYNTH_SYSTEM = "你是财经资讯编辑。只输出一个 JSON 对象。"

_SYNTH_PROMPT = """根据编号文档回答用户 query。要求：
- 简体中文，200~400 字，直接给结论/要点（可用短列表），禁止罗列文档标题
- 引用标注：在对应句子末尾标 [n]（n 为文档编号）；没有依据的不要编造
- 信息不足时明确说明缺口，不要硬凑

query: {query}

文档：
{docs}

输出 {{"answer": "..."}}:"""


# ---------------- 工具实现 ----------------

def _norm_url(u: str) -> str:
    return re.sub(r"^(https?://)?(www\.)?", "", (u or "").split("?")[0]).rstrip("/").lower()


# web 来源权威度分级（0-3）：3 官方权威 / 2 专业财经 / 1 综合门户 / 0 论坛UGC
# 实现在 tools/inline_crawl.web_authority（agent 与非 agent 兜底路共享同一词表）
def _web_authority(url: str) -> int:
    from tools.inline_crawl import web_authority
    return web_authority(url)


_AUTHORITY_LBL = {3: "官方权威", 2: "专业财经", 1: "综合来源", 0: "论坛/UGC"}


def _kw_coverage(query: str, text: str) -> float:
    """字面词覆盖度（0-1）：query 分词（≥2 字）在文本中的命中比例。

    web 文档不在 BM25 索引内，无法取真 BM25 分；用覆盖度近似「关键词匹配强度」，
    与库内卡的 BM25 条语义对应（都是词面匹配，只是度量口径不同）。
    """
    try:
        import jieba
        toks = [t for t in jieba.lcut(query) if len(t.strip()) >= 2]
        if not toks:
            return 0.0
        t = text or ""
        return round(sum(1 for tok in toks if tok in t) / len(toks), 3)
    except Exception:
        return 0.0


def _web_semantic_emb(query: str, web_docs: list) -> None:
    """给 web 文档的 best_chunk 补 emb（与库内同一 embedding 模型的余弦相似度）。

    chunk_embedding_scores 带持久缓存（chunk_id=web_{url哈希}），同 URL 跨查询免重算；
    失败静默（前端 语义 条显示 0，不影响结果）。
    """
    if not web_docs:
        return
    try:
        import hashlib
        from search.engine.corpus import chunk_embedding_scores
        from qp.common.llm_client import default_client
        cli = default_client()
        if not getattr(cli, "enabled", False) or not cli.switch("embedding_recall"):
            return
        chunks = [{"chunk_id": "web_" + hashlib.md5(d["url"].encode()).hexdigest()[:12],
                   "text": f"{d.get('title', '')} {d.get('snippet', '')}"[:1024]}
                  for d in web_docs]
        scores = chunk_embedding_scores(query, chunks, cli)
        if scores is None:
            return
        for d, c, sc in zip(web_docs, chunks, scores):
            if sc is not None and sc >= 0:
                d["best_chunk"] = {"chunk_id": c["chunk_id"], "doc_id": None, "seq": 0,
                                   "text": (d.get("snippet") or "")[:240], "emb": round(float(sc), 4)}
    except Exception:
        pass


def _tool_search_both(args: dict, top: int, svc_cfg: dict,
                      qp0=None, orig_query: str = "") -> dict:
    """双源并行检索：库内（完整 QP→通道→rerank，可拆子查询）+ web 实时，
    同时执行、合并去重、标注来源。Planner 无需选源。

    qp0：第 0 步对原 query 的 QP 理解。检索词与原 query 一致时直接复用
    （QP 只推理一次），改写词/子查询则各自走完整 QP。
    """
    from tools.inline_crawl import web_search as _ws, ingest_hits as _ingest
    proxy = (svc_cfg.get("content") or {}).get("proxy", "http://127.0.0.1:10808")

    sub_queries = args.get("sub_queries") or []
    main_q = str(args.get("query") or "").strip()
    queries = [q for q in sub_queries if isinstance(q, str) and q.strip()] \
        or [main_q]
    queries = [q for q in queries if q.strip()][:3] or [(orig_query or "").strip()]
    web_q = main_q or queries[0]

    def _lib_one(q: str):
        from search.engine import search as _search
        reuse = qp0 if (qp0 is not None and q.strip() == (orig_query or "").strip()) else None
        return _search(q, top, qp=reuse)

    def _web():
        try:
            return _ws(web_q, proxy, 8, budget_s=15.0)
        except Exception:
            return []

    # 库内各子查询 + web 源：一线程一任务，整体并行
    with ThreadPoolExecutor(max_workers=len(queries) + 1) as ex:
        web_f = ex.submit(_web)
        lib_fs = {q: ex.submit(_lib_one, q) for q in queries}
        web_hits = web_f.result()
        lib_outs = [lib_fs[q].result() for q in queries]

    # ---- 库内侧合并（多子查询 URL 去重）----
    if len(lib_outs) == 1:
        qp, results, diag = lib_outs[0]
    else:
        qp, diag = lib_outs[0][0], dict(lib_outs[0][2])
        seen, results = set(), []
        for _qp, _results, _diag in lib_outs:
            for s, d, sub, bc in _results:
                u = _norm_url(d.get("url"))
                if u and u in seen:
                    continue
                if u:
                    seen.add(u)
                results.append((s, d, sub, bc))
        results.sort(key=lambda x: -x[0])
        results = results[:top]
        diag["decomposed"] = queries
    for _s, d, _sub, _bc in results:
        d.setdefault("source", "库内")

    # ---- web 侧：转 doc 格式，与库内去重；补日期/权威度/正文开头；rerank 打分（双源同权）----
    from tools.build_content_library import extract_date, extract_date_from_url
    lib_urls = {_norm_url(d.get("url")) for _s, d, _sub, _bc in results}
    web_docs, seen_w = [], set()
    for h in web_hits:
        u = (h.get("url") or "").strip()
        nu = _norm_url(u)
        if not nu or nu in lib_urls or nu in seen_w:
            continue
        seen_w.add(nu)
        web_docs.append({"title": h.get("title", ""), "url": u,
                         "snippet": (h.get("snippet") or "")[:200],
                         "content_type": "web", "date": h.get("date"),
                         "source": "web"})
    if web_docs:
        # 日期：标题/摘要/URL 提取（搜索引擎结果通常不带日期，"刚刚"是撒谎）
        for d in web_docs:
            if not d.get("date"):
                d["date"] = extract_date(f"{d['title']} {d['snippet']}") \
                    or extract_date_from_url(d["url"])
        # 权威度：域名分级（3 官方权威 / 2 专业财经 / 1 综合 / 0 论坛UGC）
        for d in web_docs:
            d["authority"] = _web_authority(d["url"])
        # 语义分：与库内同一 embedding 模型算余弦（结果挂在 best_chunk.emb）
        _web_semantic_emb(web_q, web_docs)
        # 链指：QP 核心实体在标题/摘要中出现
        ent_texts = []
        if qp0 is not None:
            try:
                ent_texts = [e.text for e in (qp0.entities or [])]
            except Exception:
                ent_texts = []
        for d in web_docs:
            d["ent_link"] = bool(ent_texts) and any(
                e in f"{d.get('title', '')}{d.get('snippet', '')}" for e in ent_texts)
        # 正文开头：前 4 条并行短超时抓取（控延迟，失败留空不阻塞）
        try:
            from tools.inline_crawl import quick_fetch
            with ThreadPoolExecutor(max_workers=4) as ex2:
                futs = {d["url"]: ex2.submit(quick_fetch, d["url"], proxy, 5)
                        for d in web_docs[:4]}
                for d in web_docs[:4]:
                    body, dt = futs[d["url"]].result()
                    if body:
                        d["content_head"] = body[:220]
                    if dt and not d.get("date"):
                        d["date"] = dt
        except Exception:
            pass
    web_scored = []
    if web_docs:
        try:
            from qp.common.llm_client import default_client as _llm
            _cli = _llm()
            _sc = {}
            if getattr(_cli, "enabled", False) and _cli.switch("rerank"):
                _texts = [f"{d.get('title', '')} {d.get('snippet', '')}"[:512]
                          for d in web_docs]
                _rr = _cli.rerank(web_q, _texts) or []
                _sc = {i: float(s) for i, s in _rr}
            for i, d in enumerate(web_docs):
                sub = {"rerank": round(_sc[i], 4)} if i in _sc else {}
                # 与库内 mini bars 同三件套：关键词覆盖 / 链指（语义在 best_chunk.emb）
                sub["kw"] = _kw_coverage(web_q, f"{d.get('title', '')} {d.get('snippet', '')}")
                if d.get("ent_link"):
                    sub["ent_link"] = True
                web_scored.append((_sc.get(i, 0.0), d, sub, d.get("best_chunk")))
        except Exception:
            web_scored = [(0.0, d, {"kw": _kw_coverage(web_q, f"{d.get('title', '')} {d.get('snippet', '')}")},
                           d.get("best_chunk")) for d in web_docs]
    results.extend(web_scored)
    results.sort(key=lambda x: -x[0])   # 双源按同一 rerank 分混排（web 平权）
    results = results[:top]

    diag = dict(diag or {})
    diag["web_hits"] = len(web_hits)
    if web_hits:
        threading.Thread(
            target=lambda: _ingest(web_q, web_hits, proxy=proxy), daemon=True).start()
    return {"queries": queries, "qp": qp, "results": results, "diag": diag}


# ---------------- Planner / 观察构造 ----------------

def _qp_block(qp) -> str:
    """QPOutput → Planner 可读的 QP 理解摘要（意图/通道/实体/时间）。"""
    try:
        intents = "、".join(i.intent_id for i in (qp.intents or [])) or "无"
        chans = "、".join(f"{r.channel}({r.role})" for r in (qp.recall or [])) or "无"
        ents = "、".join(f"{e.text}/{e.type}" for e in (qp.entities or [])[:6]) or "无"
        t = qp.time
        t_txt = f"{t.anchor}/{t.bucket}" if t else "无"
        return f"意图 {intents}；召回通道 {chans}；实体 {ents}；时间 {t_txt}"
    except Exception:
        return "（QP 理解不可用）"


def _obs_from_results(out: dict, seen_urls: set) -> tuple:
    """工具结果 → Planner 观察。返回 (obs, docs裁剪版, docs完整版)。

    docs 与 full 按同一首见顺序排列（同 seen_urls 去重）——all_docs[n-1]
    与 doc_pool[n-1] 严格对齐，synth 的 [n] 引用编号因此能映射回完整文档。
    """
    results = out.get("results") or []
    docs, full, n = [], [], 0
    for s, d, sub, bc in results:
        u = d.get("url", "")
        if _norm_url(u) in seen_urls:
            continue
        if u:
            seen_urls.add(_norm_url(u))
        n += 1
        full.append((s, d, sub, bc))
        docs.append({"n": len(docs) + 1, "title": (d.get("title") or "")[:60],
                     "snippet": (d.get("snippet") or "")[:110],
                     "date": d.get("date"), "type": d.get("content_type"),
                     "source": d.get("source", "库内"),
                     "score": round(float(s), 3), "url": u})
    diag = out.get("diag") or {}
    obs = {"tool": "search", "queries": out.get("queries"),
           "n_new_docs": n, "docs": docs}
    # 检索诊断信号：让 Planner「看得见」漏斗内部发生了什么
    sig = {}
    if diag.get("time_starved"):
        sig["time_starved"] = True          # 时效窗内零命中
    if diag.get("core_entity_coverage") == 0:
        sig["core_entity_coverage"] = 0     # 核心实体库内零链指
    if diag.get("channels"):
        sig["channels_hit"] = list(diag["channels"].keys())
    if sig:
        obs["retrieval_signals"] = sig
    if diag.get("decomposed"):
        obs["decomposed_queries"] = diag["decomposed"]
    if not n:
        reasons = []
        if sig.get("time_starved"):
            reasons.append("库内时效空窗")
        if sig.get("core_entity_coverage") == 0:
            reasons.append("库内核心实体零命中")
        obs["note"] = "两个源都无该维度的新内容" + \
            ("（" + "、".join(reasons) + "）" if reasons else "") + \
            "——换信息维度再查，或收束时如实说明缺口"
    return obs, docs, full


def _history_text(steps: list) -> str:
    if not steps:
        return "（尚无，这是第 1 轮）"
    lines = []
    for st in steps:
        obs = st["observation"]
        lines.append(f"第{st['round']}轮 action={st['action']} args={st.get('args')} "
                     f"→ 新文档 {obs.get('n_new_docs', 0)} 条"
                     + (f"；备注：{obs['note']}" if obs.get("note") else ""))
    return "\n".join(lines)


def _parse_planner(out) -> Optional[dict]:
    if not isinstance(out, dict):
        return None
    action = str(out.get("action", "")).strip()
    if action not in _ACTIONS:
        return None
    args = out.get("args") if isinstance(out.get("args"), dict) else {}
    return {"thought": str(out.get("thought", ""))[:200],
            "action": action, "args": args}


# ---------------- LLM 客户端（agent 独立 profile） ----------------

@lru_cache(maxsize=1)
def _agent_client():
    """agent 路 LLM：优先 configs/llm.yaml 的 agent 段。

    provider=codebuddy → CodeBuddy CLI 免费模型（hy3，SSE 流式）；
    否则按 OpenAI 兼容 profile 构造；均未配置时回落主链路 client。
    """
    from qp.common.llm_client import LLMClient, CodeBuddyClient, default_client
    try:
        import qp.common.llm_client as _m
        cfg = _m.load_config(section="agent")
        if cfg.get("provider") == "codebuddy":
            c = CodeBuddyClient(section="agent")
            if c.enabled:
                return c
        else:
            c = LLMClient(section="agent")
            if c.enabled and c.chat_model:
                return c
    except Exception:
        pass
    return default_client()


def _docs_for_synth(all_docs: list) -> str:
    lines = []
    for i, d in enumerate(all_docs[:10], 1):
        lines.append(f"[{i}] {d['title']}（{d.get('date') or '日期未知'}）"
                     f"\n    {d['snippet']}")
    return "\n".join(lines) if lines else "（无）"


def _synthesize(query: str, all_docs: list,
                timeout: float | None = None, cli=None) -> Optional[str]:
    cli = cli if cli is not None else _agent_client()
    if not getattr(cli, "enabled", False) or not getattr(cli, "chat_model", None):
        return None
    out = cli.chat_json(
        _SYNTH_PROMPT.replace("{query}", query)
                     .replace("{docs}", _docs_for_synth(all_docs)),
        system=_SYNTH_SYSTEM, max_tokens=3000, timeout=timeout)
    if isinstance(out, dict) and str(out.get("answer", "")).strip():
        return str(out["answer"]).strip()
    return None


def _plan_with_fallback(prompt: str, timeout: float) -> Optional[dict]:
    """Planner 决策：agent 路（hy3）失败或解析失败 → 主链路 qwen 兜底。

    免费上游单次失败率 ~20-30%，双尝试全败会浪费 ~40s 且丢失整轮决策；
    qwen 更稳（略慢），仅两者同时失败才真正降级单发。
    """
    try:
        plan = _parse_planner(_agent_client().chat_json(
            prompt, system=_PLANNER_SYSTEM, max_tokens=1200, timeout=timeout))
        if plan:
            return plan
    except Exception:
        pass
    try:
        from qp.common.llm_client import default_client
        d = default_client()
        if d.enabled and d.chat_model:
            return _parse_planner(d.chat_json(prompt, system=_PLANNER_SYSTEM,
                                              max_tokens=900))
    except Exception:
        pass
    return None


def _extractive_answer(all_docs: list) -> Optional[str]:
    """LLM 综合不可用时的诚实兜底：原文摘录（带编号，仍可提取引用）。"""
    lines = []
    for i, d in enumerate(all_docs[:3], 1):
        sn = (d.get("snippet") or "").strip()
        if sn:
            lines.append(f"[{i}] {sn[:120]}（{d.get('date') or '日期未知'}）")
    if not lines:
        return None
    return "（LLM 综合暂不可用，以下为检索资料原文摘录）\n" + "\n".join(lines)


def _citations(answer: str, all_docs: list) -> list:
    used = sorted({int(m) for m in re.findall(r"\[(\d{1,2})\]", answer or "")
                   if 1 <= int(m) <= len(all_docs)})
    return [{"n": i, "title": all_docs[i - 1]["title"], "url": all_docs[i - 1].get("url", ""),
             "date": all_docs[i - 1].get("date")} for i in used]


# ---------------- 主循环 ----------------

def agent_search(query: str, top: int = 6, max_rounds: int = 3,
                 budget_s: float = 120.0) -> dict:
    """budget_s：循环预算（QP + Planner 轮次 + 工具，秒），默认 120（效果优先）。

    综合作答允许在循环预算外再加 ~40s（上限 45s/次）；循环剩余时间不足以
    再跑一轮 planner+工具时提前强制收束，剩余时间不足以 LLM 综合时直接
    毫秒级摘录兜底——保证端到端时长可控（最坏 ≈ budget_s + 40s）。
    """
    t0 = time.perf_counter()
    svc_cfg = {}
    try:
        with open(os.path.join(ROOT, "configs", "service.yaml"), encoding="utf-8") as f:
            svc_cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    proxy = (svc_cfg.get("content") or {}).get("proxy", "http://127.0.0.1:10808")

    # ---- 第 0 步：QP 理解（三路 LLM 优先，全链路共享，只跑一次） ----
    t_qp = time.perf_counter()
    qp0 = None
    try:
        from qp.pipeline.app import process_query
        qp0 = process_query(query)
    except Exception:
        qp0 = None
    qp_ms = round((time.perf_counter() - t_qp) * 1000, 1)
    qp_block = _qp_block(qp0) if qp0 is not None else "（QP 理解不可用）"

    steps: list = []
    all_docs: list = []          # 裁剪版文档（synth 编号 [n] 依据，n-1 即下标）
    doc_pool: list = []          # 完整文档 (score, doc, sub, bc)——首见顺序与 all_docs 对齐
    seen_urls: set = set()
    last_out: Optional[dict] = None
    degraded = False
    rounds = 0
    # 投机综合：首轮出文档后即后台启动 synth（与循环并行），
    # _finish 时按剩余预算等待其完成——隐藏 20~45s 的综合耗时
    synth_state: dict = {"answer": None, "thread": None}

    def _maybe_start_synth():
        if synth_state["thread"] is None and all_docs and planner_ok:
            def _run():
                snapshot = list(all_docs)
                res: dict = {}
                t1 = threading.Thread(target=lambda: res.update(
                    hy=_synthesize(query, snapshot, timeout=30.0)), daemon=True)
                def _qw():
                    try:
                        from qp.common.llm_client import default_client
                        res.update(qw=_synthesize(query, snapshot, timeout=35.0,
                                                  cli=default_client()))
                    except Exception:
                        res.update(qw=None)
                t2 = threading.Thread(target=_qw, daemon=True)
                t1.start(); t2.start()
                t1.join(32); t2.join(37)
                # 并行竞速：hy3（免费）与 qwen（稳定）谁先成功用谁
                synth_state["answer"] = res.get("hy") or res.get("qw")
            synth_state["thread"] = threading.Thread(target=_run, daemon=True)
            synth_state["thread"].start()

    def _finish(answer=None):
        nonlocal last_out
        if answer is None and all_docs:
            elapsed = time.perf_counter() - t0
            synth_left = (budget_s + 40.0) - elapsed
            th = synth_state["thread"]
            if th is not None:
                th.join(timeout=max(0.0, synth_left))
                answer = synth_state["answer"]
            elif synth_left >= 20:   # 无投机线程（如降级路）且时间足够：同步综合
                answer = _synthesize(query, all_docs,
                                     timeout=min(40.0, synth_left))
            if answer is None:
                answer = _extractive_answer(all_docs)   # 毫秒级摘录兜底，保证有答案
        cits = _citations(answer, all_docs) if answer else []
        # results 只展示被引用的文档（all_docs[n-1] ↔ doc_pool[n-1] 严格对齐，
        # 引用编号 n 直接映射回完整文档）；citations 为空时回落 top 条
        ref_ns = {c["n"] for c in cits}
        picked = [(i + 1, t) for i, t in enumerate(doc_pool)
                  if not ref_ns or (i + 1) in ref_ns] \
            or list(enumerate(doc_pool[:top], 1))
        picked.sort(key=lambda x: x[0])
        results = []
        for ref, (s, d, sub, bc) in picked[:top]:
            # 字段与 /search/content 响应对齐（sentiment/lang/sub_scores/best_chunk），
            # 前端结果卡片（含 chunk 信息）才能在两种模式下一致展示
            results.append({"score": round(float(s), 4), "title": d.get("title", ""),
                            "url": d.get("url", ""), "snippet": d.get("snippet", ""),
                            "content_type": d.get("content_type", ""),
                            "sentiment": d.get("sentiment", ""), "lang": d.get("lang", ""),
                            "date": d.get("date"),
                            "authority": d.get("authority",
                                               _web_authority(d.get("url", ""))),
                            "content_head": d.get("content_head"),
                            "sub_scores": sub if isinstance(sub, dict) else {},
                            "best_chunk": bc if isinstance(bc, dict) else None,
                            "ref": ref})
        lat = {"total_ms": round((time.perf_counter() - t0) * 1000, 1),
               "qp_ms": qp_ms, "rounds": rounds}
        return {"query": query, "answer": answer, "citations": cits, "steps": steps,
                "results": results,
                # 原 query 的 QP 理解（前端 QP 面板固定展示它，与子查询无关）
                "qp": qp0 if qp0 is not None else (last_out or {}).get("qp"),
                "diag": last_out.get("diag") if last_out else {},
                "rounds": rounds, "degraded": degraded, "latency": lat,
                "docs_count": len(all_docs)}

    # Planner 可用性：agent 开关（主链路 switches）+ agent 路 LLM 可用 → 循环；否则降级单发
    try:
        from qp.common.llm_client import default_client
        master_on = default_client().switch("agent")
    except Exception:
        master_on = False
    cli = _agent_client()
    planner_ok = master_on and cli.enabled and bool(cli.chat_model)

    if not planner_ok:
        degraded = True
        out = _tool_search_both({"query": query}, top, svc_cfg,
                                qp0=qp0, orig_query=query)
        last_out = out
        obs, docs, full = _obs_from_results(out, seen_urls)
        all_docs.extend(docs)
        doc_pool.extend(full)
        steps.append({"round": 1, "thought": "LLM 不可用，降级双源单发检索",
                      "action": "search", "args": {"query": query},
                      "observation": obs})
        _maybe_start_synth()
        return _finish()

    history = []
    executed_qs: set = set()   # 已执行过的检索词（归一化），防 Planner 空转

    def _norm_q(s: str) -> str:
        return re.sub(r"\s+", "", str(s or "").lower())

    for rnd in range(1, max_rounds + 1):
        rounds = rnd
        elapsed = time.perf_counter() - t0
        if elapsed > budget_s - 20:
            # 剩余不足以完成一轮 planner(~10s)+双源并行(~20s)，提前强制收束
            steps.append({"round": rnd, "thought": "预算不足，强制收束",
                          "action": "answer", "args": {}, "observation": {"n_new_docs": 0}})
            break
        try:
            _e = time.perf_counter() - t0
            plan = _plan_with_fallback(
                _PLANNER_PROMPT.replace("{query}", query)
                               .replace("{qp_block}", qp_block)
                               .replace("{history}", _history_text(steps)),
                timeout=min(20.0, max(8.0, budget_s - _e)))
        except Exception:
            plan = None
        if plan is None:
            if not steps:
                degraded = True
                out = _tool_search_both({"query": query}, top, svc_cfg,
                                        qp0=qp0, orig_query=query)
                last_out = out
                obs, docs, full = _obs_from_results(out, seen_urls)
                all_docs.extend(docs)
                doc_pool.extend(full)
                steps.append({"round": rnd, "thought": "Planner 输出解析失败，降级双源单发",
                              "action": "search", "args": {"query": query},
                              "observation": obs})
            else:
                steps.append({"round": rnd, "thought": "Planner 解析失败，按已有结果收束",
                              "action": "answer", "args": {},
                              "observation": {"n_new_docs": 0}})
            break

        action, args = plan["action"], plan["args"]
        if action == "answer":
            steps.append({"round": rnd, **plan, "observation": {"n_new_docs": 0}})
            break
        # 防空转：检索词与历史完全一致 → 不浪费一轮，直接收束
        qs = [str(q).strip() for q in (args.get("sub_queries") or
                                       [args.get("query") or ""]) if str(q).strip()]
        if qs and all(_norm_q(q) in executed_qs for q in qs):
            steps.append({"round": rnd,
                          "thought": (plan.get("thought", "") + "；检索词与历史重复，"
                                      "跳过并按已有结果收束")[:200],
                          "action": "answer", "args": {},
                          "observation": {"n_new_docs": 0}})
            break
        executed_qs.update(_norm_q(q) for q in qs)
        out = _tool_search_both(args, top, svc_cfg, qp0=qp0, orig_query=query)
        obs, docs, full = _obs_from_results(out, seen_urls)
        all_docs.extend(docs)
        doc_pool.extend(full)
        if out.get("qp") and not last_out:
            last_out = out
        elif out.get("results") and last_out is None:
            last_out = out
        if out.get("results"):
            last_out = out
        steps.append({"round": rnd, **plan, "observation": obs})
        _maybe_start_synth()   # 有文档即投机综合，与后续轮次并行

    return _finish()
