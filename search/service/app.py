# -*- coding: utf-8 -*-
"""QP 统一推理服务（FastAPI）。

接口：
  POST /predict   {"query": "..."}          -> QPOutput 统一结构
  POST /batch     {"queries": ["...", ...]} -> {results: [...]}（上限 50）
  GET  /health                                    -> 健康检查 + 模型状态
  GET  /metrics                                   -> 进程内计数器（分 Tier 流量/缓存命中）

启动：python -m serving.app  （或 uvicorn serving.app:app --port 8100）
"""
from __future__ import annotations

import os
import re
import sys
import time
import threading
from collections import Counter
from datetime import datetime
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from qp.pipeline.app import get_pipeline, process_query

with open(os.path.join(ROOT, "configs", "service.yaml"), encoding="utf-8") as f:
    SVC = yaml.safe_load(f)

app = FastAPI(title="Agentic Search", version="1.2")

# 启动对齐：链指词表 -> jieba 用户词典（NER/链指/BM25 三者同源）
try:
    from qp.common.learn import align_gazetteer_to_jieba
    print(f"[learn] gazetteer->jieba aligned, +{align_gazetteer_to_jieba()} words")
except Exception as _e:
    print(f"[learn] gazetteer->jieba align skipped: {_e}")


def _clean_title_out(t: str) -> str:
    """serve 出口标题清洗（兜住存量库里的脏标题）。"""
    try:
        from qp.common.normalize import clean_title
        return clean_title(t)
    except Exception:
        return t


_STATS = Counter()


class PredictRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)


class BatchRequest(BaseModel):
    queries: list = Field(..., min_items=1, max_items=50)


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    out = process_query(req.query)
    _STATS["total"] += 1
    _STATS[f"tier{out.tier}"] += 1
    if out.escalated:
        _STATS["escalated"] += 1
    _STATS["latency_sum_ms"] += (time.perf_counter() - t0) * 1000
    return out.to_dict()


@app.post("/batch")
def batch(req: BatchRequest):
    results = []
    for q in req.queries:
        if not isinstance(q, str) or not q.strip():
            raise HTTPException(status_code=400, detail="queries must be non-empty strings")
        out = process_query(q)
        _STATS["total"] += 1
        _STATS[f"tier{out.tier}"] += 1
        results.append(out.to_dict())
    return {"results": results}


_DOCS_CACHE = {"ts": 0.0, "docs": 0}


def _corpus_count() -> int:
    """语料行数（60s 缓存，避免 /health 每次全量读文件）。"""
    import json as _json
    now = time.time()
    if now - _DOCS_CACHE["ts"] > 60:
        path = os.path.join(ROOT, "data", "content", "corpus.jsonl")
        n = 0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
        _DOCS_CACHE.update(ts=now, docs=n)
    return _DOCS_CACHE["docs"]


@app.get("/health")
def health():
    p = get_pipeline()
    return {
        "status": "ok",
        "tier1_models": [m.name for m in p.cascade.tier1_models],
        "ner_weights_loaded": p.ner_weights is not None and os.path.exists(str(p.ner_weights)),
        "docs": _corpus_count(),
        "search_active": _SEARCH_ACTIVE["n"],
        "search_gate": 8,
        "version": "1.0",
    }


@app.get("/metrics")
def metrics():
    total = _STATS.get("total", 0)
    return {
        "total": total,
        "tier_dist": {k: v for k, v in _STATS.items() if k.startswith("tier")},
        "escalated": _STATS.get("escalated", 0),
        "escalate_rate": round(_STATS.get("escalated", 0) / total, 4) if total else 0.0,
        "avg_latency_ms": round(_STATS.get("latency_sum_ms", 0.0) / total, 3) if total else 0.0,
    }


# ---------------- 内容库检索 ----------------

# 并发闸门：重型检索（LLM+EMB+rerank）同时最多 8 个在跑，其余排队；
# 线程池（uvicorn 默认 40 线程）承接并发，闸门防内存/下游 API 过载。
import threading as _threading
_SEARCH_GATE = _threading.BoundedSemaphore(8)
_SEARCH_ACTIVE_LOCK = _threading.Lock()
_SEARCH_ACTIVE = {"n": 0}


class ContentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top: int = Field(8, ge=1, le=30)
    min_score: Optional[float] = Field(None, ge=0, le=1,
                                       description="rerank 相关性阈值（开发模式可调，默认取配置）")
    min_recall: Optional[float] = Field(None, ge=0, le=1,
                                        description="召回相关性阈值（chunk 级 emb 余弦，默认取配置）")


@app.get("/search/ingest_status")
def ingest_status(query: str):
    """兜底入库进度查询（前端轮询展示「清洗入库中 → 已入库 N 条」）。"""
    p = _INGEST_PROGRESS.get(query)
    if not p:
        return {"known": False, "query": query}
    return {"known": True, "query": query, **p,
            "age_s": round(time.time() - p.get("ts", time.time()))}


@app.post("/search/content")
def search_content(req: ContentSearchRequest):
    """QP 意图驱动的内容库检索（多路召回 + 链指过滤 + rerank）。

    兜底策略：内容库为空 / 索引缺失 / 无有效结果 / 核心实体库内零命中时，
    实时爬虫兜底抓取该 query 并入库（一律走代理），然后重新检索。
    并发：重型链路（LLM+EMB+rerank）经信号闸门限流（同时最多 8 个），其余排队。
    """
    sys.path.insert(0, ROOT)
    cfg = SVC.get("content", {})
    proxy = cfg.get("proxy", "http://127.0.0.1:10808")
    on_demand = bool(cfg.get("on_demand_crawl", True))
    min_coverage = float(cfg.get("min_token_coverage", 0.3))
    scfg = dict(cfg)
    if req.min_score is not None:
        scfg["min_rerank_score"] = req.min_score
    if req.min_recall is not None:
        scfg["min_recall_score"] = req.min_recall

    from tools.search_content import search as _search

    _SEARCH_GATE.acquire()
    try:
        with _SEARCH_ACTIVE_LOCK:
            _SEARCH_ACTIVE["n"] += 1
        try:
            return _search_content_body(req, cfg, proxy, on_demand, min_coverage, scfg, _search)
        finally:
            with _SEARCH_ACTIVE_LOCK:
                _SEARCH_ACTIVE["n"] -= 1
    finally:
        _SEARCH_GATE.release()


def _learn(qp_out, diag_: dict):
    """自学习：实体/缺失 token 反哺 jieba 分词（词表注册在 pipeline 已做）。"""
    try:
        from qp.common.learn import learn_token
        d = qp_out.to_dict()
        for e in d.get("entities", []):
            if e.get("type") in ("COMPANY", "STOCK", "STOCK_CODE", "INDEX",
                                 "INDUSTRY", "FUND", "MACRO", "ORG", "PRODUCT"):
                learn_token(e.get("text", ""))
                if e.get("canonical"):
                    learn_token(e["canonical"])
        for t in (diag_.get("missing_entities") or []):
            learn_token(t)
    except Exception:
        pass


# ---------------- 兜底爬取后台化 ----------------
_CRAWL_INFLIGHT: set = set()
_CRAWL_LOCK = threading.Lock()
_INGEST_PROGRESS: dict = {}   # query -> {stage, done, total, ts, result}（展示入库进程）
_SEARCH_RECENT: dict = {}     # query -> ts：同 query 180s 内不重复 web 兜底搜索
_SEARCH_TTL = 180.0


def _launch_ingest(query: str, proxy: str, hits: list, core_ents: list) -> dict:
    """两路搜索的「入库路」：把 web 搜索结果在后台线程治理入库（不阻塞响应）。

    进度写入 _INGEST_PROGRESS，前端经 /search/ingest_status 轮询展示。
    """
    with _CRAWL_LOCK:
        if query in _CRAWL_INFLIGHT:
            return {"queued": False, "reason": "ingest already in flight"}
        _CRAWL_INFLIGHT.add(query)

    _INGEST_PROGRESS[query] = {"stage": "queued", "done": 0, "total": len(hits),
                               "ts": time.time()}

    def _progress(stage, done, total):
        _INGEST_PROGRESS[query] = {"stage": stage, "done": done, "total": total,
                                   "ts": time.time()}

    def _worker():
        try:
            from tools.inline_crawl import ingest_hits
            cr = ingest_hits(query, hits, proxy=proxy, core_entities=core_ents,
                             progress=_progress)
            if cr.get("new_docs", 0) > 0:
                _DOCS_CACHE["ts"] = 0.0
            _INGEST_PROGRESS[query] = {
                "stage": "finished", "done": cr.get("new_docs", 0),
                "total": len(hits), "ts": time.time(),
                "rejected": cr.get("rejected", 0),
                "new_docs": cr.get("new_docs", 0)}
        except Exception as e:
            _INGEST_PROGRESS[query] = {"stage": "error", "done": 0,
                                       "total": len(hits), "ts": time.time(),
                                       "error": f"{type(e).__name__}: {str(e)[:80]}"}
        finally:
            with _CRAWL_LOCK:
                _CRAWL_INFLIGHT.discard(query)

    threading.Thread(target=_worker, daemon=True).start()
    return {"queued": True, "hits": len(hits),
            "note": "兜底搜索结果后台清洗入库中，进度见 /search/ingest_status"}


def _web_fallback_scored(query: str, hits: list, scfg: dict) -> list:
    """web 兜底搜索结果经同一 rerank 通道打分 + 阈值过滤，返回可合并的伪 doc 结果。

    分数与库内结果同源（qwen rerank），阈值同一配置（min_rerank_score）——
    两路合并透出时口径一致。rerank 不可用时返回空（不透出未打分的裸结果）。
    """
    if not hits:
        return []
    try:
        from qp.common.llm_client import default_client
        cli = default_client()
        if not cli.switch("rerank"):
            return []
        min_rr = float(scfg.get("min_rerank_score", 0.3))
        texts = [f"{h['title']}。{h['snippet']}"[:512] for h in hits]
        rr = cli.rerank(query, texts)
        if not rr:
            return []
        out = []
        for i, sc in rr:
            if float(sc) < min_rr:
                continue
            h = hits[i]
            doc = {"id": f"web_{i}", "title": h["title"], "url": h["url"],
                   "snippet": h["snippet"], "content_type": "news",
                   "sentiment": "", "lang": "zh", "date": None}
            bc = {"chunk_id": None, "doc_id": doc["id"], "seq": 0,
                  "text": (h["snippet"] or h["title"]).strip()[:240], "emb": None}
            out.append((float(sc), doc, {"rerank": round(float(sc), 4),
                                         "web_fallback": True}, bc))
        return out
    except Exception:
        return []


def _merge_results(lib_results: list, web_scored: list) -> list:
    """两路合并：URL 归一去重（库内优先），按分数降序。"""
    import re as _re
    seen = set()
    merged = []
    for s, d, sub, bc in list(lib_results) + list(web_scored):
        url_n = _re.sub(r"^(https?://)?(www\.)?", "",
                        (d.get("url") or "").split("?")[0]).rstrip("/").lower()
        if url_n and url_n in seen:
            continue
        if url_n:
            seen.add(url_n)
        merged.append((s, d, sub, bc))
    merged.sort(key=lambda x: -x[0])
    return merged


def _search_content_body(req, cfg, proxy, on_demand, min_coverage, scfg, _search):
    _t0 = time.perf_counter()
    _t_search, _t_crawl = 0.0, 0.0
    data, crawled = None, {"attempted": False, "new_docs": 0}
    try:
        _ts = time.perf_counter()
        qp, intents, slots, results, diag = _search(req.query, req.top, search_cfg=scfg)
        _t_search += time.perf_counter() - _ts
        _learn(qp, diag)
        # 两路搜索：库内检索为第 1 路；结果 weak 时第 2 路（web 兜底）同步跑
        # （15s 预算），结果与库内合并透出（同 rerank 阈值口径），高质量文档后台入库。
        weak = (not results) or (diag["token_coverage"] < min_coverage) \
            or bool(diag.get("missing_entities")) \
            or diag.get("core_entity_coverage") == 0 \
            or bool(diag.get("time_starved"))
        if weak and on_demand:
            crawled["attempted"] = True
            crawled["weak_reason"] = {k: diag.get(k) for k in
                                      ("token_coverage", "missing_entities",
                                       "core_entity_coverage", "time_starved")}
            try:
                from tools.inline_crawl import web_search
                alt = _crawl_alt_queries(qp)
                qd = qp.to_dict()
                # 时效桶 -> 引擎 timelimit（d/w/m）；核心实体 -> doc 标题别名挖矿
                tl_map = {"in_one_day": "d", "one_week": "w", "one_month": "m"}
                timelimit = tl_map.get((qd.get("time") or {}).get("bucket") or "")
                core_ents = [{"eid": e.get("eid", ""), "text": e.get("text", "")}
                             for e in qd.get("entities", [])
                             if e.get("type") in ("COMPANY", "STOCK", "STOCK_CODE",
                                                  "INDEX", "FUND") and e.get("eid")]
                _ts = time.perf_counter()
                _ts0 = _SEARCH_RECENT.get(req.query)
                if _ts0 is not None and time.time() - _ts0 < _SEARCH_TTL:
                    hits = []
                    crawled["search_skipped"] = "recently searched (180s TTL)"
                else:
                    hits = web_search(req.query, proxy,
                                      int(cfg.get("max_results", 8)), alt_queries=alt,
                                      timelimit=timelimit,
                                      brave_key=str(cfg.get("brave_api_key") or ""),
                                      budget_s=15.0)
                    _SEARCH_RECENT[req.query] = time.time()
                _t_crawl += time.perf_counter() - _ts
                crawled["web_hits"] = len(hits)
                web_scored = _web_fallback_scored(req.query, hits, scfg)
                crawled["web_kept"] = len(web_scored)
                if web_scored:
                    results = _merge_results(results, web_scored)[:req.top]
                if hits:
                    crawled["ingest"] = _launch_ingest(
                        req.query, proxy, hits, core_ents)
            except Exception as e:
                crawled["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    except FileNotFoundError as e:
        # 索引/语料不存在：直接走兜底爬取
        if not on_demand:
            raise HTTPException(status_code=503, detail=f"content index not built: {e}")
        crawled["attempted"] = True
        try:
            from tools.inline_crawl import crawl_and_ingest
            _ts = time.perf_counter()
            cr = crawl_and_ingest(req.query, proxy, int(cfg.get("max_results", 8)),
                                  budget_s=30.0)
            _t_crawl += time.perf_counter() - _ts
            crawled.update(cr)
        except Exception as e2:
            raise HTTPException(status_code=503,
                                detail=f"content index missing and on-demand crawl failed: {e2}")
        try:
            qp, intents, slots, results, diag = _search(req.query, req.top, search_cfg=scfg)
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="content library still empty after crawl")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {type(e).__name__}: {e}")

    _qd = qp.to_dict()
    _qp_ms = (_qd.get("latency_ms") or {}).get("total_ms") or 0
    # 展示层清洗：老库文档的滚动新闻条/模板句在输出侧剔除（不动索引，立即生效）
    from search.engine.clean import strip_boilerplate
    out_results = []
    for s, d, sub, bc in results:
        d2 = dict(d)
        d2["snippet"] = strip_boilerplate(d.get("snippet", ""))
        bc2 = dict(bc) if isinstance(bc, dict) else bc
        if isinstance(bc2, dict) and bc2.get("text"):
            bc2["text"] = strip_boilerplate(bc2["text"])
        out_results.append((s, d2, sub, bc2))
    return {
        "qp": _qd,
        "fallback_crawl": crawled,
        "diag": diag,
        "latency": {  # 端到端分段耗时，互斥可加总（_search 内部含 QP，故减去）
            "qp_ms": _qp_ms,
            "search_ms": max(0, round(_t_search * 1000) - _qp_ms),
            "crawl_ms": round(_t_crawl * 1000),
            "total_ms": round((time.perf_counter() - _t0) * 1000),
        },
        "results": [{"score": round(s, 4), "title": _clean_title_out(d["title"]), "url": d["url"],
                     "snippet": d["snippet"], "content_type": d["content_type"],
                     "sentiment": d["sentiment"], "lang": d["lang"],
                     "date": d.get("date"),          # doc 发布日期（时效展示）
                     "sub_scores": sub,
                     "best_chunk": bc}   # dict: chunk_id/doc_id/seq/text/emb（开发模式展示）
                    for s, d, sub, bc in out_results],
    }


def _crawl_alt_queries(qp) -> list:
    """用 QP 实体 + 内容渠道构造兜底爬取的补充 query（如「腾讯 公告」）。"""
    ch_map = {"announcement": "公告", "report": "财报 研报", "news": "新闻"}
    try:
        d = qp.to_dict()
        core = next((e["text"] for e in d.get("entities", [])
                     if e["type"] in ("COMPANY", "STOCK", "STOCK_CODE", "INDEX", "INDUSTRY")), None)
        pref = d.get("recall", [{}])[0].get("content_type_pref", []) if d.get("recall") else []
        chan_word = next((ch_map[c] for c in pref if c in ch_map), None)
        if core and chan_word:
            return [f"{core} {chan_word}"]
        if core:
            return [core]
    except Exception:
        pass
    return []


# ---------------- 运行时开关（开发者面板） ----------------

_SWITCH_NAMES = ["llm_intent", "llm_first", "llm_ner", "embedding_recall", "rerank"]


@app.get("/config")
def get_config():
    """LLM 开关、模型信息与模块设计细节（开发者面板 / 系统说明读取）。"""
    from qp.common.llm_client import default_client
    c = default_client()
    out = {"switches": {k: c.switch(k) for k in _SWITCH_NAMES},
           "models": {"chat": c.chat_model, "embedding": c.embed_model,
                      "rerank": c.rerank_model}}
    try:
        from qp.ner.llm_ner import _PROMPT, ENTITY_TYPES
        out["design"] = {"ner_prompt": _PROMPT, "ner_entity_types": ENTITY_TYPES}
    except Exception:
        out["design"] = {}
    try:
        import json as _json
        tax = _json.load(open(os.path.join(ROOT, "configs", "intent_taxonomy.json"),
                              encoding="utf-8"))
        out["design"]["intents"] = [
            {"id": i["id"], "domain": i["domain"], "name": i["name"], "note": i.get("note", "")}
            for i in tax.get("intents", [])]
        out["design"]["domains"] = [
            {"id": d["id"], "name": d["name"]} for d in tax.get("domains", [])]
    except Exception:
        pass
    try:
        import yaml as _yaml
        svc = _yaml.safe_load(open(os.path.join(ROOT, "configs", "service.yaml"), encoding="utf-8")) or {}
        casc = svc.get("cascade", {})
        t0 = casc.get("tier0_rule", {})
        t1 = casc.get("tier1_model", {})
        t2 = casc.get("tier2_escalate", {})
        rl = _yaml.safe_load(open(os.path.join(ROOT, "configs", "recall_linkage.json"),
                                  encoding="utf-8")) or {}
        latest_wins = {iid: r.get("latest_window_days")
                       for iid, r in (rl.get("intent_to_recall") or {}).items()
                       if r.get("latest_window_days") is not None}
        out["design"]["params"] = {
            "cascade": {
                "tier0_min_conf": t0.get("min_confidence"),
                "tier1_accept": t1.get("max_p_threshold"),
                "tier1_lightweight_accept": t1.get("lightweight_max_p"),
                "fallback_intent": t2.get("fallback_intent"),
            },
            "time_anchor_defaults": rl.get("time_anchor_defaults", {}),
            "latest_intent_windows": latest_wins,
            "filter": {"default_min_score": 0.3, "min_rerank_score": 0.3,
                       "dedup_same_site": 0.6, "dedup_cross_site": 0.85,
                       "freshness_boost": 1.1, "entity_boost": 1.25,
                       "channel_match_boost": 1.25, "info_idf_quantile": 0.5},
            "recall": {"rrf_k": 60, "channels": ["bm25", "chunk_emb"],
                       "emb_pool": "doc = max(chunk)"},
            "engines": ["duckduckgo", "bing", "brave", "google", "mojeek",
                        "ddg_html", "bing_html", "brave_api"],
        }
    except Exception:
        pass
    try:
        from tools.build_content_library import load_existing
        _, corpus, _ = load_existing()
        dated = sum(1 for d in corpus if d.get("date"))
        from qp.ner.entity_linker import default_linker
        gaz = default_linker().size()
        dict_path = os.path.join(ROOT, "data", "dict", "jieba_user_dict.txt")
        dict_n = sum(1 for _ in open(dict_path, encoding="utf-8")) if os.path.exists(dict_path) else 0
        out["stats"] = {"docs": len(corpus), "dated": dated, "entities": gaz,
                        "dict_words": dict_n}
    except Exception:
        out["stats"] = {}
    return out


_TRENDS_CACHE = {"ts": 0.0, "source": "", "items": []}

_FIN_FALLBACK = ["小米集团 最新公告", "半导体设备 国产替代",
                 "美联储 加息 影响", "白酒板块 资金流向", "A股 大盘 走势"]

_FIN_KEYWORDS = re.compile(
    r"股|证券|上市|IPO|ipo|财报|业绩|营收|利润|并购|收购|融资|银行|券商|基金|"
    r"加息|降息|汇率|美元|国债|债券|期货|黄金|原油|指数|市值|金融|经济|GDP|CPI|"
    r"通胀|央行|美联储|nvidia|tesla|apple|股票|涨|跌", re.I)


def _baidu_hot() -> list[str]:
    """百度热搜+财经双榜（top.baidu.com，直连 ~0.3s/榜，无需代理），返回候选标题（已简体）。

    财经榜（tab=finance）天然全为财经内容，放前；实时榜（tab=realtime）扩充池子供「换一批」轮转。
    """
    import requests
    out: list[str] = []
    try:
        for tab in ("finance", "realtime"):
            try:
                r = requests.get(f"https://top.baidu.com/api/board?platform=pc&tab={tab}",
                                 timeout=3, proxies={"http": None, "https": None},
                                 headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                for card in r.json().get("data", {}).get("cards", []):
                    for c in card.get("content", []):
                        w = (c.get("word") or c.get("query") or "").strip()
                        if 2 <= len(w) <= 36 and w not in out:
                            out.append(w)
            except Exception:
                continue
    except Exception:
        pass
    return out


def _trends_fetch(proxy: str) -> list[str]:
    """拉取 Google Trends RSS（HK/TW/US），合并、转简体、去重，返回候选标题。"""
    import requests
    import xml.etree.ElementTree as ET
    from zhconv import convert
    cands: list[str] = []
    for geo in ("HK", "TW", "US"):
        try:
            r = requests.get(f"https://trends.google.com/trending/rss?geo={geo}",
                             timeout=8,
                             proxies={"http": proxy, "https": proxy},
                             headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            root = ET.fromstring(r.content)
            titles = [e.text.strip() for e in root.iter() if e.tag.endswith("title") and e.text]
            titles = [t for t in titles[1:] if len(t) >= 2
                      and len(t) <= 36
                      and "prediction market" not in t.lower()
                      and not re.match(r"^[A-Z][a-z]+ \d{1,2}, \d{4}:", t)]
            cands.extend(titles)
        except Exception:
            continue
    seen, out = set(), []
    for t in cands:
        t = convert(t, "zh-cn").strip()
        k = re.sub(r"\s+", "", t).lower()
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _fin_filter_llm(cands: list[str], exclude: list[str] | None = None) -> list[str] | None:
    """LLM 金融筛选：只留强财经相关的热点并改写成简体中文检索问句（≤3 条）。

    exclude：已展示过的问句，「换一批」时要求 LLM 避开同类主题。
    """
    try:
        from qp.common.llm_client import default_client
        c = default_client()
        if not c.enabled or not c.chat_model:
            return None
        ban = ("\n\n以下主题刚展示过，必须排除（含同公司、同事件的近似主题）：\n"
               + "\n".join(f"- {t}" for t in exclude)) if exclude else ""
        out = c.chat_json(
            "从下列今日热搜标题中，挑出与金融财经强相关的（公司、行业、个股、宏观政策、市场行情、"
            "银行证券基金、大宗商品、经济数据等），宁缺毋滥，最多 3 条；没有金融相关的则返回空数组。"
            "把每条改写成一句简体中文搜索问句（保留实体名，去掉口水话，每条不超过 14 个字），"
            "只输出 JSON 数组：\n"
            + "\n".join(f"- {t}" for t in cands[:40])
            + ban,
            system="你是财经资讯编辑，输出严格 JSON。")
        if not isinstance(out, list):
            return None
        items = [str(x).strip() for x in out if str(x).strip()][:3]
        return items or None
    except Exception:
        return None


@app.get("/hot")
def hot_queries(refresh: bool = False):
    """今日金融热点检索问句：百度热搜（直连）→ Google Trends（走代理）→ LLM 金融筛选 ≤3 条；逐级回落。

    refresh=1 时跳过缓存并打乱候选池、排除已展示主题，实现「换一批」。
    """
    now = time.time()
    if not refresh and _TRENDS_CACHE["items"] and \
            now - _TRENDS_CACHE["ts"] < _TRENDS_CACHE.get("ttl", 3600):
        return {"source": _TRENDS_CACHE["source"], "items": _TRENDS_CACHE["items"]}
    proxy = os.getenv("SEARCH_PROXY") or "http://127.0.0.1:10808"
    items, source = [], "builtin"
    # 「换一批」：累积排除最近展示过的问句（含各批），历史清空即重新轮转
    hist = _TRENDS_CACHE.setdefault("history", [])
    exclude = hist[-12:] if refresh else []

    # 首选百度热搜（直连毫秒级）；不可用再走 Google Trends（代理，慢）
    cands, src = _baidu_hot(), "baidu_hot"
    if not cands:
        cands, src = _trends_fetch(proxy), "google_trends"
    if cands:
        import random
        random.shuffle(cands)   # 「换一批」：候选池打乱，每次取到不同批次
        picked = _fin_filter_llm(cands, exclude=exclude)
        if not picked and exclude:
            # 池子里财经条目被历史排除完：清空历史重新轮转，保证每批都有内容
            hist.clear()
            picked = _fin_filter_llm(cands, exclude=None)
        if picked:
            items, source = picked, f"{src}+llm"
        else:
            # LLM 不可用或无金融热点：关键词过滤兜底
            kw = [t for t in cands if _FIN_KEYWORDS.search(t) and t not in exclude][:3]
            if kw:
                # 后缀后总长仍 ≤14，避免截断把「最新动态」剪成半截
                items = [f"{t} 最新动态" if len(t) <= 9 else t for t in kw]
                source = f"{src}+keyword"
    if not items:
        items = _FIN_FALLBACK
    items = [t[:14] for t in items[:3]]  # 安全截断：最多 3 条、每条 ≤14 字
    if source != "builtin":
        hist.extend(items)
        del hist[:-12]          # 只留最近 12 条（4 批），池子轮完自然重新开始
    # 缓存分级：真实热点 1h；builtin 回落只存 10min，尽快自愈恢复「今日热点」
    _TRENDS_CACHE.update(ts=now, source=source, items=items,
                         ttl=3600 if source != "builtin" else 600)
    return {"source": source, "items": items}


class SwitchRequest(BaseModel):
    name: str = Field(..., pattern="^(llm_intent|llm_first|llm_ner|embedding_recall|rerank)$")
    value: bool


@app.post("/config/switch")
def set_switch(req: SwitchRequest):
    """运行时切换 LLM 开关并持久化到 llm.yaml（下一次调用即生效，无需重启）。"""
    from qp.common.llm_client import default_client
    ok = default_client().set_switch(req.name, req.value)
    if not ok:
        raise HTTPException(status_code=500, detail="llm.yaml 不可写或为空")
    return {"ok": True, "name": req.name, "value": req.value}


def _eval_ner(items: list) -> dict:
    """NER 模块评测：逐条跑生产同款 NER+链指（process_query），核对实体识别质量。"""
    rows, type_dist, latencies = [], Counter(), []
    linked = 0
    for it in items:
        q = it["query"]
        t0 = time.perf_counter()
        try:
            qp = process_query(q)
            ents = [{"text": e.text, "type": e.type, "source": e.source,
                     "eid": e.eid, "canonical": e.canonical} for e in qp.entities]
        except Exception as e:
            ents = [{"text": f"<error {type(e).__name__}>", "type": "", "source": "",
                     "eid": "", "canonical": ""}]
        ms = round((time.perf_counter() - t0) * 1000)
        latencies.append(ms)
        for e in ents:
            if e["type"]:
                type_dist[e["type"]] += 1
                if e.get("eid"):
                    linked += 1
        rows.append({"query": q, "entities": ents, "ms": ms})
    n_ent = sum(type_dist.values())
    return {"mode": "ner", "results": rows,
            "metrics": {"n_queries": len(items), "n_entities": n_ent,
                        "linked_pct": round(linked * 100 / max(n_ent, 1), 1),
                        "avg_ms": round(sum(latencies) / max(len(latencies), 1)),
                        "type_dist": dict(type_dist)}}


_TIME_CHANNEL_ZH = {"freshness": "最新", "absolute": "绝对时点", "range": "区间",
                    "forward": "未来", "none": "无"}


def _eval_time(items: list) -> dict:
    """时效模块评测：五桶判定 + 时间锚点 + 窗口/通道（纯规则，无 LLM 开销）。"""
    from qp.slots.extractor import detect_time_bucket, infer_time_anchor
    from qp.recall.linkage import _BUCKET_DAYS
    bucket_dist: Counter = Counter()
    rows = []
    for it in items:
        q = it["query"]
        bucket = detect_time_bucket(q)
        anchor = infer_time_anchor(q).get("anchor", "无")
        window = _BUCKET_DAYS.get(bucket)
        channel = "freshness" if anchor == "最新" else (
            "absolute" if anchor == "历史时点" else "range" if anchor == "区间"
            else "forward" if anchor == "未来" else "none")
        bucket_dist[bucket] += 1
        rows.append({"query": q, "bucket": bucket, "anchor": anchor,
                     "window": window, "channel": channel,
                     "channel_zh": _TIME_CHANNEL_ZH.get(channel, channel)})
    return {"mode": "time", "results": rows,
            "metrics": {"n_queries": len(items), "bucket_dist": dict(bucket_dist)}}


# ---------------- 批量评测 ----------------

class EvalItem(BaseModel):
    query: str
    gold_l2: str | None = None
    gold_product_category: str | None = None


class EvalRunRequest(BaseModel):
    items: list = Field(..., min_items=1, max_items=2000)
    mode: str = Field("qp", pattern="^(qp|retrieval|ner|time)$")
    # qp=意图评测 | retrieval=召回+排序评测 | ner=实体评测 | time=时效评测


def _parse_items(req: EvalRunRequest) -> list:
    items = []
    for it in req.items:
        if isinstance(it, str):
            # 分隔符支持 TAB / ⇥ / | / ｜（textarea 中无法输入真实 TAB）
            parts = re.split(r"[\t\u21e5|｜]", it)
            items.append({"query": parts[0].strip(),
                          "gold_l2": parts[1].strip() if len(parts) > 1 and parts[1].strip() else None,
                          "gold_product_category": parts[2].strip() if len(parts) > 2 and parts[2].strip() else None})
        else:
            items.append(it)
    return items


def _conf_stats(confs: list) -> dict:
    """置信度分布：区分度健康检查（若 p10==p90 说明置信度无区分度）。"""
    if not confs:
        return {"n": 0}
    cs = sorted(confs)
    n = len(cs)

    def pct(p):
        return round(cs[min(n - 1, int(n * p))], 4)

    return {"n": n, "min": cs[0], "p10": pct(0.1), "median": pct(0.5),
            "p90": pct(0.9), "max": cs[-1],
            "distinct": len(set(round(c, 2) for c in cs))}


@app.post("/eval/run")
def eval_run(req: EvalRunRequest):
    """批量评测。items 支持 {query, gold_l2?} 或 "query" 字符串（TAB 分列）。

    mode=qp        : 逐条 QP（意图/槽位），评 L2 准确率 + 置信度分布 + 延迟。
    mode=retrieval : 每条 query 走完整链路 QP -> 内容库检索 -> 相关性标注，
                     评 召回（hit@5 / 库内相关文档数 / token 覆盖率）与
                     排序（MRR / nDCG@5）。相关性代理标注：标题+摘要包含
                     query 的高 IDF 信息词（>=2 个）或 QP 识别的实体串。
    """
    items = _parse_items(req)
    if req.mode == "retrieval":
        return _eval_retrieval(items)
    if req.mode == "ner":
        return _eval_ner(items)
    if req.mode == "time":
        return _eval_time(items)

    results, l2_hit, gold_n, cat_hit, cat_n = [], 0, 0, 0, 0
    latencies, confs = [], []
    tier_dist: Counter = Counter()
    intent_dist: Counter = Counter()
    for it in items:
        out = process_query(it["query"])
        latencies.append(out.latency_ms.get("total_ms", 0))
        tier_dist[f"tier{out.tier}"] += 1
        pred_l2 = out.intents[0].intent_id if out.intents else ""
        conf = out.intents[0].confidence if out.intents else 0
        if out.intents:
            confs.append(conf)
            intent_dist[pred_l2] += 1
        gold_l2 = it.get("gold_l2")
        if gold_l2:
            gold_n += 1
            if pred_l2 == gold_l2:
                l2_hit += 1
        gold_cat = it.get("gold_product_category")
        if gold_cat:
            cat_n += 1
            from qp.common.taxonomy import acceptable_l2
            if pred_l2 in acceptable_l2(gold_cat):
                cat_hit += 1
        results.append({
            "query": it["query"],
            "pred_l2": pred_l2,
            "pred_l1": out.intents[0].domain if out.intents else "",
            "tier": out.tier,
            "confidence": conf,
            "slots": out.slots,
            "gold_l2": gold_l2, "gold_product_category": gold_cat,
            "correct_l2": (pred_l2 == gold_l2) if gold_l2 else None,
        })

    lat_sorted = sorted(latencies)
    n = len(items)

    def pct(p):
        return lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * p))] if lat_sorted else 0

    return {
        "mode": "qp",
        "n": n,
        "tier_dist": dict(tier_dist),
        "escalate_rate": round(tier_dist.get("tier2", 0) / n, 4),
        "intent_dist": intent_dist.most_common(),
        "latency": {"p50_ms": pct(0.5), "p90_ms": pct(0.9), "p99_ms": pct(0.99)},
        "confidence_dist": _conf_stats(confs),
        "accuracy": {
            "l2_acc": round(l2_hit / gold_n, 4) if gold_n else None,
            "l2_n": gold_n,
            "product16_acc": round(cat_hit / cat_n, 4) if cat_n else None,
            "product16_n": cat_n,
        },
        "results": results,
    }


def _eval_retrieval(items: list) -> dict:
    """召回+排序分层评测：每条 query 走 QP -> 内容库检索 -> 代理标注。"""
    import json as _json
    import jieba
    from tools.search_content import load_all, search as _search

    cfg = SVC.get("content", {})
    corpus, idx = load_all()
    jieba.load_userdict(os.path.join(ROOT, "data", "dict", "jieba_user_dict.txt"))
    jieba.initialize()

    idf_map = getattr(idx["bm25"], "idf", {}) or {}
    n_docs = len(corpus)
    import math
    default_idf = math.log(n_docs + 0.5) - math.log(0.5) if n_docs else 0.0
    # 语料级 IDF 中位数：判定「信息词」的阈值
    all_idf = sorted(idf_map.values()) or [0.0]
    median_idf = all_idf[len(all_idf) // 2]

    doc_text = [f"{d.get('title', '')} {d.get('snippet', '')}".lower() for d in corpus]
    doc_toks = [[t for t in jieba.lcut(t) if t.strip()] for t in doc_text]
    id2pos = {d.get("id"): i for i, d in enumerate(corpus)}

    def label_relevant(query: str, entities: list) -> list:
        """代理相关性标注（优先级递减）：
        1. 链指命中：doc linked_entities 与 query 链指实体同 eid（同一词表）
        2. QP 实体串 / 伪实体（最高权重且 df<=50% 的已知 token）出现在正文
        3. 命中 >=2 个高 IDF 信息词"""
        q_eids = {e.get("eid") for e in entities if e.get("eid")}
        toks = set(t for t in jieba.lcut(query.lower()) if t.strip())
        weights = {t: idf_map.get(t, default_idf) for t in toks}
        known = {t: w for t, w in weights.items() if t in idf_map}
        info = {t for t, w in known.items() if w >= median_idf}
        pseudo_ent = None
        df_cache = {}
        cand = [t for t in known
                if (df_cache.setdefault(t, sum(1 for tk in doc_toks if t in tk))) <= 0.5 * n_docs]
        if cand:
            pseudo_ent = max(cand, key=lambda t: known[t])
        labels = []
        for di, text in enumerate(doc_text):
            hit_link = any(le.get("eid") in q_eids
                           for le in (corpus[di].get("linked_entities") or []))
            hit_entity = any(e["text"].lower() in text for e in entities)
            hit_pseudo = pseudo_ent is not None and pseudo_ent in text
            hit_info = sum(1 for t in info if t in text)
            if hit_link or hit_entity or hit_pseudo or hit_info >= 2:
                labels.append(di)
        return labels

    rows, hits, mrrs, ndcgs, covs = [], [], [], [], []
    for it in items:
        q = it["query"]
        try:
            qp, _, _, results, diag = _search(q, 8, search_cfg=cfg)
        except FileNotFoundError:
            qp, results, diag = process_query(q), [], {}
        entities = qp.to_dict().get("entities", [])
        rel_set = set(label_relevant(q, entities))
        top_ids = [id2pos.get(d.get("id"), -1) for _, d, _, _ in results]
        n_rel = len(rel_set)
        hit5 = len(rel_set & set(top_ids[:5])) > 0 if n_rel else None
        mrr = 0.0
        for rank, di in enumerate(top_ids, start=1):
            if di in rel_set:
                mrr = 1.0 / rank
                break
        dcg = sum(1.0 / math.log2(r + 2) for r, di in enumerate(top_ids[:5]) if di in rel_set)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(n_rel, 5))) if n_rel else 1.0
        ndcg = dcg / idcg if n_rel else None
        if hit5 is not None:
            hits.append(hit5)
            mrrs.append(mrr)
            ndcgs.append(ndcg)
        covs.append(diag.get("token_coverage", 0.0))
        rows.append({
            "query": q,
            "pred_l2": qp.intents[0].intent_id if qp.intents else "",
            "confidence": qp.intents[0].confidence if qp.intents else 0,
            "n_relevant": n_rel,
            "hit5": hit5,
            "mrr": round(mrr, 4),
            "ndcg5": round(ndcg, 4) if ndcg is not None else None,
            "token_coverage": diag.get("token_coverage", 0.0),
            "missing_entities": diag.get("missing_entities", []),
            "n_results": len(results),
            # 逐条检索 Top 结果（供前端展开查看）
            "top_results": [{"title": d.get("title", "")[:64], "score": round(s, 4),
                             "rerank": (sub or {}).get("rerank")}
                            for s, d, sub, _bc in results[:5]],
        })

    n = len(items)
    return {
        "mode": "retrieval",
        "n": n,
        "recall_metrics": {
            "hit5_rate": round(sum(hits) / len(hits), 4) if hits else None,
            "avg_token_coverage": round(sum(covs) / n, 4) if n else None,
            "label_note": "相关性=文档含query实体串或>=2个高IDF信息词（代理标注，无人工金标）",
        },
        "ranking_metrics": {
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else None,
            "ndcg5": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else None,
        },
        "confidence_dist": _conf_stats([r["confidence"] for r in rows]),
        "results": rows,
    }


# ---------------- 前端 ----------------

_STATIC = os.path.join(ROOT, "serving", "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"),
                        headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn

    # 日志落盘：终端输出（uvicorn access/error、爬虫错误等）全量 tee 到 logs/service.log
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                except Exception:
                    pass
            return len(s)

        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass

        def isatty(self):
            return False

    _log_dir = os.path.join(ROOT, "logs")
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = open(os.path.join(_log_dir, "service.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)
    print(f"===== service start {datetime.now().isoformat(timespec='seconds')} =====")

    uvicorn.run(app, host=SVC["server"]["host"], port=SVC["server"]["port"])
