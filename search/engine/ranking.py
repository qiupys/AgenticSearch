# -*- coding: utf-8 -*-
"""ranking —— 排序模块。

职责：召回候选 -> 最终结果（含排序侧诊断）。
  1. rerank 全候选重排（qwen3.7-text-rerank，输入 = doc 最相关 chunk，上限 64 控延迟）
  2. rerank 相关性阈值过滤（展示分即排序分：链指/频道加成不参与排序，仅作标记；
     vec-only 文档——词面零命中仅 embedding 召回——按更高门槛过滤导航页噪声）
  3. 同文去重（URL 归一必去重；标题相似同域 >=0.6 / 跨域 >=0.85）
  4. 截断 top 并附 best_chunk

输入契约（来自召回模块）：
  scored              list[(score, doc, sub)]
  best_chunk_of_doc   dict[doc_index, chunk dict]
  emb_client          LLM client（复用其 rerank；None 时 rerank=off）
  search_cfg          dict   min_rerank_score / ...

输出契约（交给 serving API）：
  (scored, diag)；results = [(score, doc, sub, best_chunk), ...]
"""
from __future__ import annotations

import difflib
import re
from urllib.parse import urlparse

RERANK_CAND_LIMIT = 64   # rerank 输入上限（控延迟）；超出部分视为低相关丢弃


def rank_candidates(query: str, scored: list, corpus: list, best_chunk_of_doc: dict,
                    emb_client, search_cfg: dict | None = None):
    cfg = search_cfg or {}
    min_rr = float(cfg.get("min_rerank_score", 0.3))
    # vec-only 文档（词面零命中，仅靠 embedding 召回）噪声率极高：
    # 导航页 boilerplate / 正文侧栏词都可能向量凑近，必须用更高 rerank 门槛
    vec_only_min = float(cfg.get("vec_only_min_rerank", 0.5))

    rerank_status = "off"
    if emb_client is not None and emb_client.switch("rerank") and scored:
        rerank_status = "on"
        try:
            # rerank 必须覆盖全部候选：只重排头部会让无 rerank 分的尾部文档
            # 绕过阈值过滤，在兜底场景（高分文档不足）混入结果
            cand = scored[:RERANK_CAND_LIMIT]
            _pos = {dd["id"]: i for i, dd in enumerate(corpus)}
            texts = []
            for _, d, _ in cand:
                bc = best_chunk_of_doc.get(_pos.get(d["id"]))
                texts.append((bc["text"] if bc else
                              f"{d.get('title', '')} {d.get('snippet', '')}")[:512])
            rr = emb_client.rerank(query, texts)
            if rr:
                reranked = []
                for idx_in_cand, rscore in rr:
                    s, d, sub = cand[idx_in_cand]
                    # 展示的就是 rerank 分 -> 排序也用 rerank 原始分；
                    # 链指/频道加成不参与排序，仅作为 sub 标记展示
                    sub = dict(sub, rerank=round(float(rscore), 4))
                    reranked.append((float(rscore), d, sub))
                reranked.sort(key=lambda x: -x[0])
                scored = reranked
                # 阈值过滤：rerank 已覆盖全部候选，无分（API 漏返回）同样视为低相关丢弃；
                # vec-only（bm25=0）文档按更高门槛过滤——词面零命中 + 中低语义分
                # 几乎都是导航页/侧栏噪声（如"资金流向首页"混入"上海楼市"结果）
                kept, n_vec_only = [], 0
                for s, d, sub in scored:
                    rr = sub.get("rerank")
                    if rr is None:
                        continue
                    if not sub.get("bm25"):
                        if rr >= vec_only_min:
                            kept.append((s, d, sub))
                        else:
                            n_vec_only += 1
                    elif rr >= min_rr:
                        kept.append((s, d, sub))
                diag_add = {"n_below_threshold": len(scored) - len(kept) - n_vec_only,
                            "n_vec_only_filtered": n_vec_only}
                scored = kept
            else:
                diag_add = {}
                rerank_status = "fail"
        except Exception:
            diag_add = {}
            rerank_status = "fail"
    else:
        diag_add = {}
    diag = {"rerank": rerank_status, **diag_add}

    # ---- 去重：同一篇文章只保留最优一条 ----
    # 同 URL（参数归一）必去重；标题相似：同域名 >=0.6（同站姊妹页/列表页变体）、
    # 跨域名 >=0.85（转载），均视为同文，避免同文变体占多个结果位。
    n_dup = 0
    seen_urls, kept_titles = [], []   # kept_titles: [(domain, title_n)]
    deduped = []
    for s, d, sub in scored:
        url_n = re.sub(r"^(https?://)?(www\.)?", "",
                       (d.get("url") or "").split("?")[0]).rstrip("/").lower()
        title_n = re.sub(r"\s+", "", d.get("title", ""))
        if url_n and url_n in seen_urls:
            n_dup += 1
            continue
        domain = urlparse(d.get("url") or "").netloc.lower().replace("www.", "")
        dup = False
        if len(title_n) >= 8:
            for kd, kt in kept_titles:
                ratio = difflib.SequenceMatcher(None, title_n, kt).ratio()
                if ratio >= 0.85 or (kd == domain and domain and ratio >= 0.6):
                    dup = True
                    break
        if dup:
            n_dup += 1
            continue
        if url_n:
            seen_urls.append(url_n)
        if len(title_n) >= 8:
            kept_titles.append((domain, title_n))
        deduped.append((s, d, sub))
    diag["n_dup_filtered"] = n_dup

    return deduped, diag


def finalize(scored: list, corpus: list, best_chunk_of_doc: dict, top: int) -> list:
    """截断 top 并附每篇 doc 的最相关 chunk（供前端优先展示）。

    embedding 召回失败/未命中时，兜底用 doc 的正文或 snippet 组装 best_chunk，
    保证前端卡片始终有内容可展示。
    """
    _pos = {dd["id"]: i for i, dd in enumerate(corpus)}
    results = []
    for s, d, sub in scored[:top]:
        di = _pos.get(d["id"])
        bc = best_chunk_of_doc.get(di)
        if not bc or not (bc.get("text") or "").strip():
            txt = (d.get("body") or d.get("snippet") or "").strip()[:240]
            if txt:
                bc = {"chunk_id": None, "doc_id": d["id"], "seq": 0,
                      "text": txt, "emb": None}
        results.append((s, d, sub, bc))
    return results
