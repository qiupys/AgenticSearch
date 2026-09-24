# -*- coding: utf-8 -*-
"""按需实时爬取兜底（Q：内容库空/检索失败时，爬虫兜底并入库）。

被 serving /search/content 调用：单 query 检索 → 多后端爬取 → 治理入库 → 重建索引。
设计约束：
  - 一律走代理（service.yaml: content.proxy）
  - 快速失败：短退避（max_backoff_s=8），最长 ~15s 返回
  - 并发保护：进程级锁，同一时刻只允许一个爬取任务；重复 query 走 LRU 跳过
  - 入库幂等：URL 去重 + 追加写 corpus.jsonl + 全量重建索引（小库秒级）
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path = ROOT
if sys_path not in __import__("sys").path:
    import sys
    sys.path.insert(0, sys_path)

from tools.build_content_library import (
    ResilientSearcher, extract_date, govern_row, load_existing, build_index,
    simplify_query, CORPUS_PATH,
)

_LOCK = threading.Lock()
_RECENT: dict[str, float] = {}   # 最近爬过的 query -> ts（进程级，防重复，TTL 过期）
_RECENT_TTL = 1800.0             # 30 分钟内同 query 不重爬
_RECENT_MAX = 500


def web_search(query: str, proxy: str, max_results: int = 8,
               alt_queries: list | None = None, timelimit: str | None = None,
               brave_key: str = "", budget_s: float = 15.0) -> list[dict]:
    """兜底 web 搜索（两路搜索的「兜底路」，不入库）。

    attempts 链（原 query + alt + 简化降级，上限 2 次尝试）在 budget 内逐个尝试，
    首个命中的后端结果直接返回，供 serving 与库内检索结果合并透出（阈值控制）。
    返回 [{title, url, snippet}]，不写语料。单引擎 timeout=6s 保证预算可守
    （search_one 内部串行多引擎、无法中断，故从入口收紧单引擎耗时）。
    """
    _t0 = time.time()
    searcher = ResilientSearcher(proxy=proxy, max_results=max_results,
                                 sleep_range=(0.2, 0.5), max_retry=0,
                                 max_backoff_s=3, timeout=6,
                                 brave_key=brave_key,
                                 deadline=_t0 + budget_s)
    region = "wt-wt" if re.search(r"[A-Za-z]{3}", query) and not re.search(r"[\u4e00-\u9fff]", query) else "cn-zh"
    attempts = ([(query, region)] + [(a, region) for a in (alt_queries or [])][:1])[:2]
    for attempt_q, rg in attempts:
        if time.time() - _t0 > budget_s * 0.6:
            break   # 剩余预算不足以再跑一轮多引擎尝试
        results = searcher.search_one(attempt_q, rg, timelimit=timelimit)
        if results:
            hits = [{"title": (r.get("title") or "").strip(),
                     "url": (r.get("href") or r.get("url") or "").rstrip("/"),
                     "snippet": (r.get("body") or r.get("description") or "").strip()}
                    for r in results]
            hits = [h for h in hits
                    if h["url"] and len(f"{h['title']} {h['snippet']}") >= 20]
            if hits:
                return hits[:max_results]
        if time.time() - _t0 > budget_s:
            break
    return []


def ingest_hits(query: str, hits: list[dict], proxy: str = "",
                core_entities: list | None = None,
                progress=None) -> dict:
    """把 web 搜索结果治理入库（两路搜索的「入库路」，供后台线程调用）。

    门禁：URL 去重 / 近重复拦截（同域标题>=0.6、跨域>=0.85）/ 过短丢弃 /
    行情模板页与滚动新闻条污染页清洗或拒收（content_search.clean）；
    时效补全：前 4 条无日期的短超时抓正文；入库后重建索引 + 实体别名挖矿。
    progress：可选回调 progress(stage, done, total)，供 serving 记录入库进度。
    """

    def _p(stage, done, total):
        try:
            if progress:
                progress(stage, done, total)
        except Exception:
            pass

    _t0 = time.time()
    acquired = _LOCK.acquire(timeout=0.1)
    if not acquired:
        return {"ok": False, "reason": "another crawl in progress", "new_docs": 0}
    try:
        if not hits:
            return {"ok": True, "new_docs": 0, "total_docs": 0}
        from search.engine.clean import clean_doc_text
        done, corpus, seen_urls = load_existing()
        # 入库合规基线：已有标题（用于近重复拦截，防止同文多版本污染语料）
        import difflib
        from urllib.parse import urlparse as _up
        have_titles = [( _up(d.get("url") or "").netloc.lower().replace("www.", ""),
                         re.sub(r"\s+", "", d.get("title", ""))) for d in corpus]

        new_docs = 0
        new_rows = []
        n_rejected = 0

        def _quick_date_body(url: str):
            """短超时抓正文，提日期 + 补 body（供 chunk/打标）。失败返回 (None, None)。

            质量门禁：WAF 挑战页 / JS 脚本 / mojibake 不入 body——
            snippet 已含搜索引擎摘要，body 脏不如无 body。
            """
            try:
                import requests
                from tools.build_content_library import extract_date_from_html
                from qp.common.normalize import fix_mojibake, is_junk_text
                r = requests.get(url, timeout=6, proxies=(
                    {"http": proxy, "https": proxy} if proxy else None),
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                             "Accept-Language": "zh-CN,zh;q=0.9"})
                r.encoding = r.apparent_encoding or r.encoding   # 防 mojibake
                html = r.text[:60000]
                body = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>"
                              r"|<noscript[\s\S]*?</noscript>|<iframe[\s\S]*?</iframe>"
                              r"|<svg[\s\S]*?</svg>|<[^>]+>", " ", html)
                body = re.sub(r"\s+", " ", body).strip()[:6000]
                body = fix_mojibake(body)
                if is_junk_text(body):
                    return None, extract_date_from_html(html)
                return body, extract_date_from_html(html)
            except Exception:
                return None, None

        with open(CORPUS_PATH, "a", encoding="utf-8") as fout:
            fetched = 0
            for idx, h in enumerate(hits):
                _p("clean", idx, len(hits))
                if time.time() - _t0 > 30.0:
                    break
                url = h["url"]
                if not url or url in seen_urls:
                    continue
                # 质量清洗：模板句/滚动新闻条/噪声剔除；行情/列表/垃圾页整条拒收
                title_c, text_c, ok = clean_doc_text(h.get("title", ""),
                                                     h.get("snippet", ""))
                if not ok:
                    n_rejected += 1
                    continue
                row = {
                    "query": query, "query_intent": "on_demand",
                    "title": title_c,
                    "url": url,
                    "snippet": text_c,
                }
                # 近重复拦截：同域名标题相似>=0.6 或 跨域>=0.85 视为同文，不入库
                title_n = re.sub(r"\s+", "", row["title"])
                domain = _up(url).netloc.lower().replace("www.", "")
                dup = False
                for kd, kt in have_titles:
                    ratio = difflib.SequenceMatcher(None, title_n, kt).ratio()
                    if ratio >= 0.85 or (kd == domain and domain and kt and ratio >= 0.6):
                        dup = True
                        break
                if dup:
                    continue
                have_titles.append((domain, title_n))
                seen_urls.add(url)
                # 时效补全：标题/摘要无日期且未达抓取上限 -> 短超时抓正文提日期
                if fetched < 4 and not extract_date(f"{row['title']} {row['snippet']}"):
                    _p("fetch", fetched, min(4, len(hits)))
                    body, dt = _quick_date_body(url)
                    fetched += 1
                    if body:
                        _bt, _btext, _bok = clean_doc_text(row["title"], row["snippet"], body)
                        if _bok:
                            row["body"] = _btext
                    if dt:
                        row["date"] = dt
                corpus.append(govern_row(row, len(corpus) + 1))
                fout.write(json.dumps(corpus[-1], ensure_ascii=False) + "\n")
                new_rows.append(corpus[-1])
                new_docs += 1
        if new_docs:
            _p("index", 0, new_docs)
            build_index(corpus)
            # 自学习：从新 doc 标题挖核心实体的扩展形式 -> 注册别名
            try:
                from qp.common.learn import learn_alias, mine_alias
                for ce in (core_entities or []):
                    alias = mine_alias(ce.get("text", ""), [r["title"] for r in new_rows])
                    if alias and alias != ce.get("text"):
                        learn_alias(ce.get("eid", ""), alias)
            except Exception:
                pass
        _p("done", new_docs, len(hits))
        return {"ok": True, "new_docs": new_docs, "total_docs": len(corpus),
                "rejected": n_rejected}
    finally:
        _LOCK.release()


def crawl_and_ingest(query: str, proxy: str, max_results: int = 8,
                     alt_queries: list | None = None, timelimit: str | None = None,
                     core_entities: list | None = None, brave_key: str = "",
                     budget_s: float = 45.0) -> dict:
    """对单条 query 实时爬取并入库（web_search + ingest_hits 组合，兼容旧调用）。

    alt_queries：QP 实体+渠道构造的补充爬取 query（如「腾讯 公告」），优先于简化降级。
    timelimit：时效桶映射的引擎参数（d/w/m），透传给支持的后端。
    core_entities：[{eid, text}]，用于从新 doc 标题挖实体别名反哺词表。
    budget_s：总时间预算（搜索 attempts + 逐 URL 抓正文共用），超时即止防失控。
    """
    _t0 = time.time()
    acquired = _LOCK.acquire(timeout=0.1)
    if not acquired:
        return {"ok": False, "reason": "another crawl in progress", "new_docs": 0}
    _LOCK.release()   # web_search 无需锁；入库锁在 ingest_hits 内部获取
    try:
        _ts0 = _RECENT.get(query)
        if _ts0 is not None and time.time() - _ts0 < _RECENT_TTL:
            return {"ok": False, "reason": "recently crawled", "new_docs": 0}

        hits = web_search(query, proxy, max_results, alt_queries=alt_queries,
                          timelimit=timelimit, brave_key=brave_key,
                          budget_s=max(5.0, budget_s * 0.4))
        cr = ingest_hits(query, hits, proxy=proxy, core_entities=core_entities)
        _RECENT[query] = time.time()
        if len(_RECENT) > _RECENT_MAX:   # 淘汰过期项，仍超限则整体清空
            _now = time.time()
            for _q in [q for q, t in _RECENT.items() if _now - t >= _RECENT_TTL]:
                _RECENT.pop(_q, None)
            if len(_RECENT) > _RECENT_MAX:
                _RECENT.clear()
        return {"ok": cr.get("ok", False), "new_docs": cr.get("new_docs", 0),
                "total_docs": cr.get("total_docs", 0)}
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {str(e)[:120]}", "new_docs": 0}
