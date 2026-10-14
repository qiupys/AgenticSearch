# -*- coding: utf-8 -*-
"""content_search —— 检索子系统门面。

三个可独立升级的模块：
  corpus.py   语料/索引/chunk 基础设施（共享数据契约）
  recall.py   召回模块：多路召回 + RRF 融合 + 召回侧过滤
  ranking.py  排序模块：rerank + 阈值过滤 + 同文去重

模块编排（唯一 allowed 的跨模块调用点）：
  QP(pipeline.process_query)
    -> recall.recall_candidates(query, qp, corpus, idx, cfg)
       -> (scored, best_chunk_of_doc, emb_client, diag)
    -> ranking.rank_candidates(query, scored, corpus, best_chunk_of_doc, emb_client, cfg)
       -> (scored, diag)
    -> ranking.finalize(...) -> results

模块间只通过上述函数签名耦合；升级任一模块时保持签名与 diag 键即可。
"""
from __future__ import annotations

from .corpus import (CONTENT_DIR, load_all, load_or_build_chunks,       # noqa: F401
                     chunk_embedding_scores, split_chunks)
from . import recall as _recall
from . import ranking as _ranking


def search(query: str, top: int = 10, bm25_weight: float = 1.0,
           vec_weight: float = 1.0, search_cfg: dict | None = None,
           qp=None):
    """端到端检索：QP -> 召回 -> 排序 -> 结果。

    返回 (qp, results, diag)；qp 为完整 QPOutput（intents/entities/time/recall），
    results = [(score, doc, sub, best_chunk), ...]。
    qp 可选传入已算好的 QPOutput（agent 循环复用第 0 步理解，避免重复推理）；
    缺省时内部调用 process_query(query)。
    bm25_weight/vec_weight 为历史兼容参数（TF-IDF 路已删除），当前无效果。
    """
    from qp.pipeline.app import process_query

    import time as _time
    _t0 = _time.perf_counter()
    qp = qp if qp is not None else process_query(query)
    corpus, idx = load_all()
    _t1 = _time.perf_counter()

    # ---- 召回模块 ----
    scored, best_chunk_of_doc, emb_client, diag = _recall.recall_candidates(
        query, qp, corpus, idx, search_cfg)
    _t2 = _time.perf_counter()

    # ---- 排序模块 ----
    scored, rank_diag = _ranking.rank_candidates(
        query, scored, corpus, best_chunk_of_doc, emb_client, search_cfg)
    _t3 = _time.perf_counter()
    diag.update(rank_diag)
    diag["load_ms"] = round((_t1 - _t0) * 1000, 1)
    diag["recall_ms"] = round((_t2 - _t1) * 1000, 1)
    diag["rank_ms"] = round((_t3 - _t2) * 1000, 1)

    # ---- 输出组装 ----
    results = _ranking.finalize(scored, corpus, best_chunk_of_doc, top)
    return qp, results, diag
