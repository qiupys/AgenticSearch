# -*- coding: utf-8 -*-
"""统一 QP Pipeline：归一化 → 级联意图 → NER → 槽位 → 召回翻译 → QPOutput。

对每条 query 输出统一的 QPOutput 结构（common/schema.py），
该结构同时驱动线上服务返回与离线评测消费。
"""
from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional

import yaml

from qp.common.normalize import normalize_query, detect_lang
from qp.common.schema import QPOutput, Intent, Entity, TimeInfo
from qp.common.taxonomy import l2_to_product_categories
from qp.intent.cascade import IntentCascade
from qp.ner import predictor as ner_predictor
from qp.recall.linkage import translate
from qp.slots.extractor import extract_slots

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")


_MARKET_PATTERNS = (
    ("HK", re.compile(r"港股|H股|港股通|恒生指数|恒生科技")),
    ("US", re.compile(r"美股|纳斯达克|纳指|纽交所|道琼斯|标普500")),
    ("A", re.compile(r"A股|A股市场|沪深|沪市|深市|创业板|科创板|北交所")),
)


def _detect_market(query: str, ents: list) -> str:
    """从 query 识别市场限定（HK/US/A），供两地上市实体消歧与召回定向。

    优先级：显式市场词 > 代码归属（6位数字=A、4~5位数字=HK、大写字母 ticker=US）。
    """
    q = query or ""
    for mkt, pat in _MARKET_PATTERNS:
        if pat.search(q):
            return mkt
    for e in ents or []:
        t = (e.get("text") or "").strip()
        if not t:
            continue
        if re.fullmatch(r"\d{6}", t):
            return "A"
        if re.fullmatch(r"\d{4,5}", t) and e.get("type") in ("STOCK_CODE", "STOCK"):
            return "HK"
        if re.fullmatch(r"[A-Z]{1,5}", t) and e.get("type") in ("STOCK", "STOCK_CODE"):
            return "US"
    return ""


class QPPipeline:
    def __init__(self, config_path: Optional[str] = None, ner_weights_path: Optional[str] = None):
        cfg_path = config_path or os.path.join(CONFIG_DIR, "service.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.cascade = IntentCascade(cfg_path)
        ner_cfg = self.cfg.get("ner", {})
        self.ner_weights = ner_weights_path or ner_cfg.get("model_weights_path")
        self.ner_rule_fallback = bool(ner_cfg.get("rule_fallback", True))

    # ------------------------------------------------------------------
    def process(self, query: str) -> QPOutput:
        t_total0 = time.perf_counter()
        timings: Dict[str, float] = {}

        # 1. 归一化 + 语言判定
        t0 = time.perf_counter()
        norm = normalize_query(query)
        lang = detect_lang(query)
        timings["normalize_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # 2. 级联意图识别
        t0 = time.perf_counter()
        casc = self.cascade.predict(query, norm)
        timings["intent_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        intents = [Intent(i["intent_id"], i["domain"], i["confidence"], i.get("source", "rule"))
                   for i in casc["intents"]]

        # 3. NER（原文上做，保留位置）
        #    - 规则 NER 缺核心对象实体（COMPANY/STOCK/PRODUCT）时，LLM 补识别并合并
        #    - 链指：span -> 实体词表（gazetteer），新实体增量注册
        t0 = time.perf_counter()
        ents = ner_predictor.predict(query, self.ner_weights, self.ner_rule_fallback)
        core_types = {"COMPANY", "STOCK", "STOCK_CODE", "PRODUCT",
                      "INDEX", "INDUSTRY", "FUND", "MACRO"}
        if not any(e.get("type") in core_types for e in ents):
            try:
                from qp.ner.llm_ner import predict_llm
                seen_texts = {e["text"] for e in ents}
                for le in predict_llm(query):
                    if le["text"] not in seen_texts:
                        ents.append(le)
                        seen_texts.add(le["text"])
            except Exception:
                pass
        try:
            from qp.ner.entity_linker import default_linker
            linker = default_linker()
            for e in ents:
                et = e.get("type", "")
                if et in {"COMPANY", "STOCK", "STOCK_CODE", "INDEX", "FUND",
                          "INDUSTRY", "MACRO", "ORG", "PRODUCT", "METRIC"}:
                    ref = linker.link(e["text"], et)
                    if ref is None and et in {"COMPANY", "MACRO", "ORG"}:
                        ref = linker.register(e["text"], et, source="query_llm")
                    if ref:
                        e["eid"] = ref["eid"]
                        e["canonical"] = ref["name"]
        except Exception:
            pass
        timings["ner_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # 3.5 市场定向：两地上市实体（阿里巴巴 HK+US）按 query 市场词/代码消歧。
        #     信号优先级：显式市场词 > 代码归属；与核心实体 meta.markets 交集校验，
        #     无交集（矛盾 query，如「茅台港股」）则不启用，交由常规召回。
        market = _detect_market(query, ents)
        if market:
            try:
                from qp.ner.entity_linker import default_linker
                metas = [default_linker().meta_of(e["eid"]) for e in ents if e.get("eid")]
                valid = (not metas) or any(market in (m.get("markets") or []) for m in metas)
                if valid:
                    for e in ents:
                        if e.get("eid") and e.get("type") in ("COMPANY", "STOCK", "STOCK_CODE"):
                            e["market"] = market
            except Exception:
                pass
        entities = [Entity(e["text"], e["type"], e["start"], e["end"],
                           e.get("source", "rule"),
                           eid=e.get("eid", ""), canonical=e.get("canonical", ""),
                           market=e.get("market", ""))
                    for e in ents]

        # 4. 槽位推断
        t0 = time.perf_counter()
        composite = len(intents) > 1
        intent_ids = [i.intent_id for i in intents]
        slots, tinfo = extract_slots(norm, [e.__dict__ for e in entities],
                                     composite=composite, intent_ids=intent_ids)
        if market:
            slots["market"] = market
        timings["slots_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        time_info = TimeInfo(anchor=tinfo["anchor"], raw=tinfo["raw"],
                             normalized=tinfo["normalized"],
                             bucket=tinfo.get("bucket", "no_limit"))

        # 5. 召回翻译
        t0 = time.perf_counter()
        recall = translate(casc["intents"], slots, [e.__dict__ for e in entities], query=query)
        timings["recall_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # 6. 产品 16 类统计口径（反向映射，取代表类）
        stats_16 = None
        if intents:
            cands = l2_to_product_categories(intents[0].intent_id)
            stats_16 = cands[0] if cands else None

        timings["total_ms"] = round((time.perf_counter() - t_total0) * 1000, 3)

        return QPOutput(
            query=query, normalized=norm, lang=lang,
            intents=intents, slots=slots, entities=entities,
            time=time_info, recall=recall, stats_category_16=stats_16,
            tier=casc["tier"], escalated=casc["escalated"],
            latency_ms=timings,
        )


# ---------------- 进程级单例 + LRU 缓存 ----------------
_PIPELINE: Optional[QPPipeline] = None
_CACHE: Dict[str, QPOutput] = {}
_CACHE_MAX = 10000


def get_pipeline() -> QPPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = QPPipeline()
    return _PIPELINE


def process_query(query: str, use_cache: bool = True) -> QPOutput:
    from qp.common.normalize import cache_key
    key = cache_key(query) if use_cache else None
    if use_cache and key in _CACHE:
        return _CACHE[key]
    out = get_pipeline().process(query)
    if use_cache and key:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = out
    return out
