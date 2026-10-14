# -*- coding: utf-8 -*-
"""统一 QP Pipeline（v2.0 三路架构，三路 LLM 优先）。

三路产出，一条 query 一次推理（路2+路3 合并一次 LLM 调用）：
  路1 意图：LLM 优先 → Tier1 模型 → Tier0 规则 → 兜底
  路2 实体：LLM 优先 → 规则 NER 兜底（核心类型互补合并）→ 实体链指 → 市场定向
  路三时间：LLM 优先（与路2 同一次调用）→ 规则时间路兜底
随后意图×召回翻译层把三路产物编译为带专用 channel 的召回条件。

对每条 query 输出统一的 QPOutput 结构（common/schema.py v2.0），
该结构同时驱动线上服务返回与离线评测消费。
"""
from __future__ import annotations

import os
import re
import time
from typing import Dict, Optional

import yaml

from qp.common.normalize import normalize_query, detect_lang
from qp.common.schema import QPOutput, Intent, Entity, TimeInfo
from qp.common.taxonomy import l2_to_product_categories
from qp.intent.cascade import IntentCascade
from qp.ner import predictor as ner_predictor
from qp.recall.linkage import translate
from qp.time.extractor import infer_time_anchor

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

        # 0. 归一化 + 语言判定
        t0 = time.perf_counter()
        norm = normalize_query(query)
        lang = detect_lang(query)
        timings["normalize_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ---- 路1：意图（级联：Tier0 规则 → LLM → Tier1 模型 → 兜底）----
        t0 = time.perf_counter()
        casc = self.cascade.predict(query, norm)
        intents = [Intent(i["intent_id"], i["domain"], i["confidence"], i.get("source", "rule"))
                   for i in casc["intents"]]
        timings["intent_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ---- 路2+路3：NER + 时间（合并一次 LLM 调用，失败各自回落规则）----
        t0 = time.perf_counter()
        core_types = {"COMPANY", "STOCK", "STOCK_CODE", "PRODUCT",
                      "INDEX", "INDUSTRY", "FUND", "MACRO"}
        ents: list = []
        tinfo: dict | None = None
        try:
            from qp.common.llm_client import default_client as _dc
            llm_ner_on = _dc().switch("llm_ner")
        except Exception:
            llm_ner_on = False
        if llm_ner_on:
            try:
                from qp.ner.llm_ner import predict_llm_qp
                ents, llm_time = predict_llm_qp(query)
                if llm_time:
                    # 防御：LLM 时间桶与规则桶取更宽者（硬时间过滤宁宽勿漏；
                    # 「最新」等模糊表达规则路更可靠，防止 LLM 误判窄桶清空召回）
                    try:
                        from qp.time.extractor import detect_time_bucket
                        _order = ["no_limit", "one_year", "one_month", "one_week", "in_one_day"]
                        rule_b = detect_time_bucket(norm)
                        llm_b = llm_time.get("bucket", "no_limit")
                        if rule_b in _order and llm_b in _order \
                                and _order.index(rule_b) < _order.index(llm_b):
                            llm_time = dict(llm_time, bucket=rule_b)
                    except Exception:
                        pass
                    tinfo = llm_time
            except Exception:
                ents, tinfo = [], None
        if not ents:
            # LLM 关闭/失败/空 → 规则 NER 兜底
            ents = ner_predictor.predict(query, self.ner_weights, self.ner_rule_fallback)
        else:
            # LLM 成功：规则 NER 补充 LLM 漏掉的核心类型实体（去重合并）
            have = {(e["text"], e["type"]) for e in ents}
            have_types = {e["type"] for e in ents}
            for e in ner_predictor.predict(query, self.ner_weights, self.ner_rule_fallback):
                if e["type"] in core_types and e["type"] not in have_types \
                        and (e["text"], e["type"]) not in have:
                    ents.append(e)
                    have_types.add(e["type"])
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
        # 市场定向：两地上市实体（阿里巴巴 HK+US）按 query 市场词/代码消歧。
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
        timings["ner_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # ---- 路三：时间（LLM 已随路2产出，耗时计入 ner_ms；缺失时规则兜底）----
        if tinfo is None:
            t0 = time.perf_counter()
            tinfo = infer_time_anchor(norm)
            timings["time_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        else:
            timings["time_ms"] = 0.0   # 随路2 的 LLM 调用产出，无独立耗时
        time_info = TimeInfo(anchor=tinfo["anchor"], raw=tinfo.get("raw", ""),
                             normalized=tinfo.get("normalized", ""),
                             bucket=tinfo.get("bucket", "no_limit"))

        # ---- 意图×召回翻译：三路产物编译为带 channel 的召回条件 ----
        t0 = time.perf_counter()
        recall = translate(casc["intents"], [e.__dict__ for e in entities],
                           time_info, query=query)
        timings["recall_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        # 产品 16 类统计口径（反向映射，取代表类）
        stats_16 = None
        if intents:
            cands = l2_to_product_categories(intents[0].intent_id)
            stats_16 = cands[0] if cands else None

        timings["total_ms"] = round((time.perf_counter() - t_total0) * 1000, 3)

        return QPOutput(
            query=query, normalized=norm, lang=lang,
            intents=intents, entities=entities,
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
