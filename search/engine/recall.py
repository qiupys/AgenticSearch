# -*- coding: utf-8 -*-
"""recall —— 召回模块。

职责：query + QP 输出 -> 候选文档列表（含召回侧诊断）。
  1. 多路召回：BM25 词面路 + chunk 级 embedding 语义路（doc = max chunk）
     + 意图专用通道（QP 每个意图一条独立排序路，v2.0 三路架构）
  2. RRF 融合 + 意图驱动软加权（内容类型偏好/情绪）
  3. 召回侧过滤（顺序即漏斗顺序）：
       信息词噪声（泛词 IDF 分位，链指命中豁免）
       -> 召回相关性阈值（chunk emb，链指命中豁免）
       -> 核心实体链指硬过滤
       -> 链指标记（仅标记展示，不参与排序权重）
       -> 时效窗软过滤（剔空则保留并上报 time_starved）

输入契约：
  query       str          原始查询
  qp          QPOutput     QP 模块产物（intents/entities/time/recall 条件）
  corpus/idx  corpus.load_all() 产物
  search_cfg  dict         min_recall_score / info_idf_quantile / ...

输出契约（交给排序模块）：
  (scored, best_chunk_of_doc, emb_client, diag)
  scored        list[(score, doc, sub)]   sub 含 bm25/info 等召回侧分数
  best_chunk_of_doc  dict[doc_index, {chunk_id,doc_id,seq,text,emb}]
  emb_client    LLM client 实例或 None（排序模块复用其 rerank）
  diag          dict    召回侧诊断（embedding/token_coverage/n_* 等）
"""
from __future__ import annotations

from collections import defaultdict

import jieba
import numpy as np

from . import corpus as _corpus


def _rrf(rank_lists, k: int = 60) -> dict:
    """Reciprocal Rank Fusion：各路 rank 融合，稳健于分数口径差异。"""
    agg = defaultdict(float)
    for ranks in rank_lists:
        for r, doc_i in enumerate(ranks):
            agg[doc_i] += 1.0 / (k + r + 1)
    return agg


def recall_candidates(query: str, qp, corpus: list, idx: dict,
                      search_cfg: dict | None = None):
    cfg = search_cfg or {}
    n_docs = len(corpus)

    # ---- 意图召回通道：全部条件参与（不只主意图），每个意图一个专用通道 ----
    conds = [c for c in (qp.recall or []) if c.weight > 0 and c.role != "不走内容召回"]
    recall_cond = qp.recall[0] if qp.recall else None
    try:
        from qp.recall.linkage import _linkage as _linkage_cfg
        ch_cfgs = _linkage_cfg().get("channels", {})
    except Exception:
        ch_cfgs = {}

    # 核心实体集合前置：实体链指命中优先级高于噪声过滤（命中实体 docs 不被泛词过滤误杀）
    q_core_eids = {e.eid for e in qp.entities
                   if getattr(e, "eid", "") and e.type in
                   {"COMPANY", "STOCK", "STOCK_CODE", "INDEX", "INDUSTRY",
                    "FUND", "MACRO", "ORG"}}

    def _doc_eids(d):
        return {le.get("eid") for le in (d.get("linked_entities") or [])}

    core_hit_docs = ({i for i, d in enumerate(corpus) if _doc_eids(d) & q_core_eids}
                     if q_core_eids else set())

    # ---- 路 1：BM25 词面召回 ----
    toks = [t for t in jieba.lcut(query.lower()) if t.strip()]
    bm25_scores = idx["bm25"].get_scores(toks)
    bm25_rank = list(np.argsort(-bm25_scores))[:50]
    rank_lists = [bm25_rank]

    # ---- 路 2：chunk 级 embedding 语义召回（doc 得分 = max chunk）----
    emb_client = None
    emb_status = "off"
    best_chunk_of_doc: dict[int, dict] = {}
    try:
        from qp.common.llm_client import default_client as _llm
        emb_client = _llm()
        if emb_client.switch("embedding_recall"):
            chunks = _corpus.load_or_build_chunks(corpus)
            cscores = _corpus.chunk_embedding_scores(query, chunks, emb_client)
            if cscores is not None:
                pos = {d["id"]: i for i, d in enumerate(corpus)}
                doc_emb = np.full(n_docs, -1.0, dtype=np.float32)
                for ci, ch in enumerate(chunks):
                    di = pos.get(ch["doc_id"])
                    if di is not None and cscores[ci] > doc_emb[di]:
                        doc_emb[di] = cscores[ci]
                        best_chunk_of_doc[di] = {"chunk_id": ch["chunk_id"],
                                                 "doc_id": ch["doc_id"], "seq": ch["seq"],
                                                 "text": ch["text"][:240],
                                                 "emb": round(float(cscores[ci]), 4)}
                emb_rank = list(np.argsort(-doc_emb))[:50]
                rank_lists.append(emb_rank)
                emb_status = "on"
            else:
                emb_status = "fail"
    except Exception:
        emb_status = "fail"
    diag = {"embedding": emb_status}

    # ---- 路 3：意图专用通道召回（每个意图一条独立排序路，参与 RRF）----
    # 通道分 = BM25 主干 ×（1 + 0.3×意图权重）^ 命中通道内容类型
    #        ×（1.15 ^ 正向情绪命中，仅 boost_positive 通道）
    #        ×（1.2 ^ 核心实体链指命中，disclosure/research 类通道加成）
    doc_types = [d.get("content_type", "") for d in corpus]
    diag["channels"] = {}
    for cond in conds:
        ch_types = set(ch_cfgs.get(cond.channel, {}).get("content_types", [])) \
            | set(cond.content_type_pref)
        if not ch_types:
            continue
        ch_score = bm25_scores.astype(np.float32).copy()
        type_hit = np.array([t in ch_types for t in doc_types], dtype=np.float32)
        ch_score *= np.power(1.0 + 0.3 * float(cond.weight), type_hit)
        if cond.sentiment == "boost_positive":
            sent_pos = np.array([d.get("sentiment") == "positive" for d in corpus],
                                dtype=np.float32)
            ch_score *= np.power(1.15, sent_pos)
        if cond.channel in ("disclosure", "research") and core_hit_docs:
            ent_hit = np.zeros(n_docs, dtype=np.float32)
            ent_hit[list(core_hit_docs)] = 1.0
            ch_score *= np.power(1.2, ent_hit)
        ch_rank = list(np.argsort(-ch_score))[:50]
        rank_lists.append(ch_rank)
        diag["channels"][cond.channel] = {"intent": cond.intent_id,
                                          "top": [corpus[i].get("id") for i in ch_rank[:3]]}

    fused = _rrf(rank_lists)

    # ---- 诊断：IDF 加权覆盖率 + 缺失 token（供兜底触发）----
    def _tok(text: str) -> list:
        return [t for t in jieba.lcut(text.lower()) if t.strip()]

    corpus_toks = [_tok(f"{d.get('title', '')} {d.get('snippet', '')}") for d in corpus]
    qtoks = set(toks)
    vocab = set().union(*[set(t) for t in corpus_toks]) if corpus_toks else set()
    import math
    default_idf = math.log(n_docs + 0.5) - math.log(0.5) if n_docs else 0.0
    idf_map = getattr(idx["bm25"], "idf", {}) or {}
    tok_weights = {t: idf_map.get(t, default_idf) for t in qtoks}
    weight_sum = sum(tok_weights.values())
    coverage = (sum(w for t, w in tok_weights.items() if t in vocab) / weight_sum) \
        if weight_sum > 0 else 1.0
    corpus_text = "\n".join(
        f"{d.get('title', '')} {d.get('snippet', '')}" for d in corpus).lower()
    missing_entities = sorted(
        t for t, w in tok_weights.items()
        if t not in vocab and len(t) >= 2 and t not in corpus_text)
    diag.update({"token_coverage": round(coverage, 4),
                 "missing_entities": missing_entities,
                 "max_bm25": float(bm25_scores.max()) if len(bm25_scores) else 0.0})

    # ---- 意图驱动的软加权（RRF 融合阶段）----
    pref_types = set(recall_cond.content_type_pref) if recall_cond else set()
    boost_sent = None
    if recall_cond and recall_cond.sentiment == "boost_positive":
        boost_sent = "positive"

    # ---- 过滤 1：信息量 token BM25（泛词噪声剔除；实体命中 docs 豁免）----
    info_min_idf = float(cfg.get("info_idf_quantile", 0.5))
    pos_w = sorted(w for w in tok_weights.values() if w > 0)
    if pos_w:
        idf_cut = pos_w[min(len(pos_w) - 1, int(len(pos_w) * info_min_idf))]
        info_toks = [t for t, w in tok_weights.items() if w >= idf_cut and t in vocab]
    else:
        info_toks = []
    diag["info_tokens"] = info_toks
    info_scores = idx["bm25"].get_scores(info_toks) if info_toks else np.zeros(n_docs)

    scored = []
    n_filtered = 0
    for doc_i, rrf_score in fused.items():
        d = corpus[doc_i]
        if info_toks and info_scores[doc_i] <= 0 and doc_i not in core_hit_docs:
            n_filtered += 1
            continue
        s = rrf_score
        if pref_types and d["content_type"] in pref_types:
            s *= 1.25
        if boost_sent and d["sentiment"] == boost_sent:
            s *= 1.15
        if recall_cond and recall_cond.sentiment == "mixed" and d["sentiment"] == "neutral":
            s *= 0.95
        sub = {"bm25": round(float(bm25_scores[doc_i]), 4),
               "info": round(float(info_scores[doc_i]), 4)}
        scored.append((s, d, sub))

    # ---- 过滤 2：召回相关性阈值（chunk 级 emb 余弦，doc = max chunk）----
    # 与 rerank 阈值是两个体系：召回分决定「值不值得进 rerank」，rerank 分决定「进不进结果」。
    # 链指命中（核心实体强信号）豁免，不被向量分误杀。
    min_recall = float(cfg.get("min_recall_score", 0.4))
    if min_recall > 0 and scored and best_chunk_of_doc:
        _core_ids = {corpus[i]["id"] for i in (core_hit_docs or set())}
        _pos = {dd["id"]: i for i, dd in enumerate(corpus)}
        kept = [(s, d, sub) for s, d, sub in scored
                if d["id"] in _core_ids
                or (best_chunk_of_doc.get(_pos.get(d["id"])) or {}).get("emb", 0) >= min_recall]
        diag["n_recall_filtered"] = len(scored) - len(kept)
        scored = kept

    # ---- 过滤 3：核心实体链指硬过滤 ----
    # query 链指出核心实体时，库内若存在链指命中的 docs，则其余 docs 全部剔除
    # （平台署名/词面撞车不再混入，如「腾讯新闻」≠「腾讯」）；
    # 库内零命中时不剔除，但上报 core_entity_coverage=0 供兜底爬取。
    n_entity_filtered = 0
    if q_core_eids:
        n_hit_docs = len(core_hit_docs)
        diag["core_entity_coverage"] = 1 if n_hit_docs else 0
        if n_hit_docs:
            kept = [(s, d, sub) for s, d, sub in scored if _doc_eids(d) & q_core_eids]
            n_entity_filtered = len(scored) - len(kept)
            scored = kept
    else:
        diag["core_entity_coverage"] = None
    diag["n_entity_filtered"] = n_entity_filtered

    # ---- 过滤 3.5：市场定向（两地上市实体消歧，如 阿里巴巴 HK(09988) vs US(BABA)）----
    # q 带 market 槽位且实体词表有该市场代码：doc 文本含目标市场代码 -> match，
    # 含其他市场代码且不含目标 -> mismatch。存在 match 时剔除 mismatch（无代码的 neutral 保留）。
    n_market_filtered = 0
    q_market = next((getattr(e, "market", "") for e in (qp.entities or [])
                     if getattr(e, "market", "")), "")
    if q_market and scored:
        try:
            from qp.ner.entity_linker import default_linker
            codes_all: dict = {}
            for e in (qp.entities or []):
                if getattr(e, "eid", "") and getattr(e, "market", "") == q_market:
                    codes_all.update(default_linker().meta_of(e.eid).get("codes") or {})
            tgt = codes_all.get(q_market)
            if tgt:
                tvar = {c for c in (tgt, tgt.lstrip("0")) if c}
                others = {c for m2, c in codes_all.items() if m2 != q_market and c}

                def _mkt_flag(d):
                    txt = f"{d.get('title', '')} {d.get('snippet', '')} {(d.get('body') or '')[:500]}"
                    if any(c in txt for c in tvar):
                        return "match"
                    if any(c in txt for c in others):
                        return "mismatch"
                    return "neutral"

                flags = [(x, _mkt_flag(x[1])) for x in scored]
                if any(f == "match" for _, f in flags):
                    kept_m = [x for x, f in flags if f != "mismatch"]
                    n_market_filtered = len(scored) - len(kept_m)
                    scored = kept_m
            diag["market"] = q_market
        except Exception:
            pass
    diag["n_market_filtered"] = n_market_filtered

    # ---- 链指标记（仅标记，不参与排序权重；排序权属 rerank 模块）----
    n_ent_boost = 0
    if q_core_eids:
        boosted = []
        for s, d, sub in scored:
            hit = bool({le.get("eid") for le in (d.get("linked_entities") or [])} & q_core_eids)
            n_ent_boost += hit
            boosted.append((s, d, dict(sub, ent_link=hit)))
        scored = boosted
    scored.sort(key=lambda x: -x[0])
    diag["n_filtered"] = n_filtered
    diag["entity_boost_hits"] = n_ent_boost

    # ---- 过滤 4：时效窗软过滤 + freshness 加权 ----
    # 窗口来源：显式时效桶（近一周=7d 等，linkage 已覆盖意图默认）或「最新」意图窗
    window = recall_cond.time_window_days if recall_cond else None
    if window and scored:
        import datetime
        cutoff = datetime.date.today() - datetime.timedelta(days=int(window))

        def _dd(d):
            s = (d.get("date") or "").strip()
            try:
                if len(s) >= 10:
                    return datetime.date.fromisoformat(s[:10])
                if len(s) == 7:   # YYYY-MM 月精度 -> 当月 1 日（稍保守）
                    return datetime.date.fromisoformat(s + "-01")
            except ValueError:
                return None
            return None

        kept = [it for it in scored if _dd(it[1]) is None or _dd(it[1]) >= cutoff]
        n_known_fresh = sum(1 for it in kept if _dd(it[1]) is not None)
        # 软过滤：有存活结果则应用过滤；全被剔（清空）或存活者全是未知日期
        # 均视为时效无命中（time_starved）——保留结果但上报，服务端触发搜索兜底补库。
        if kept:
            diag["n_time_filtered"] = len(scored) - len(kept)
            scored = kept
        diag["time_starved"] = n_known_fresh == 0
        if recall_cond and recall_cond.time_channel == "freshness":
            scored = [((s * 1.1, d, sub) if _dd(d) else (s, d, sub))
                      for s, d, sub in scored]
            scored.sort(key=lambda x: -x[0])

    return scored, best_chunk_of_doc, emb_client, diag
