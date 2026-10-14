# -*- coding: utf-8 -*-
"""意图 → 召回翻译层（v2.0 三路架构：意图的输出直接驱动召回通道）。

输入：intents + entities + TimeInfo（时间路产物）+ 原始 query
输出：List[RecallCondition]（主意图置顶，每个意图带专用召回 channel）

核心设计（意图×召回深度绑定）：
  - 每个意图映射一个召回 channel（news/disclosure/research/forecast/
    screener/education/multimedia/none，定义在 recall_linkage.json），
    召回引擎为每个 channel 生成独立排序路参与 RRF 融合
  - 内容频道：query 表面词（公告/财报/研报/新闻）命中时叠加内容类型偏好
  - 时间通道：时间路 TimeInfo → freshness/absolute/range/forward/none，
    「最新」窗口支持意图级覆盖（latest_window_days）
  - 行业过滤：对象类型由 NER 实体内联推导（不再有独立 object_type 槽位）
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

from qp.common.schema import RecallCondition, TimeInfo

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")

# query 表面词 -> 内容频道（content_type）
_CHANNEL_KEYWORDS: List[tuple] = [
    (re.compile(r"公告|通告|披露|澄清|通函"), ["announcement"]),
    (re.compile(r"财报|年报|半年报|季报|业绩|营收|净利润|毛利率"), ["report"]),
    (re.compile(r"研报|研究报告|券商观点|行业分析"), ["report"]),
    (re.compile(r"新闻|资讯|消息|动态"), ["news"]),
]

# time anchor -> time_channel
_TIME_CHANNEL = {"最新": "freshness", "历史时点": "absolute",
                 "区间": "range", "未来": "forward", "无": "none"}

# 显式时效分桶 -> 时间窗（天）：覆盖意图默认窗口
BUCKET_DAYS = {"in_one_day": 1, "one_week": 7, "one_month": 30, "one_year": 365}


@lru_cache(maxsize=1)
def _linkage() -> dict:
    from qp.common.taxonomy import _load
    return _load(os.path.join(CONFIG_DIR, "recall_linkage.json"))


def object_type_of(entities: List[dict], query: str = "") -> str:
    """对象类型：NER 实体类型直接推导（原 slots.object_type，无独立槽位）。

    个股/公司 > 板块指数 > 基金ETF > 行业 > 宏观 > 无特定对象。
    """
    types = {e.get("type") or e.get("type_", "") for e in entities}
    text = query or ""
    if types & {"STOCK", "STOCK_CODE", "COMPANY"}:
        return "个股"
    if "INDEX" in types or re.search(r"板块|概念", text):
        return "板块指数"
    if "FUND" in types:
        return "基金ETF"
    if "INDUSTRY" in types:
        return "行业"
    if "MACRO" in types:
        return "宏观"
    return "无特定对象"


def _industry_condition(entities: List[dict], query: str) -> Optional[Dict]:
    obj = object_type_of(entities, query)
    industry = _linkage()["industry_by_object_type"].get(obj)
    if industry and industry.get("level1") == "entity_link":
        # 行业对象：实体链接（INDUSTRY 实体文本 → 行业分类映射表）
        ind_ents = [e["text"] for e in entities if e["type"] == "INDUSTRY"]
        industry = {"level1": "entity_link", "matched_industries": ind_ents} if ind_ents else None
    return industry


def extract_content_channel(query: str) -> List[str]:
    """query 表面词 -> 内容频道（命中多个取并集，按匹配顺序）。"""
    chans: List[str] = []
    for pat, chans_ in _CHANNEL_KEYWORDS:
        if pat.search(query):
            for c in chans_:
                if c not in chans:
                    chans.append(c)
    return chans


def translate(intents: List[dict], entities: List[dict], time_info: TimeInfo,
              query: str = "") -> List[RecallCondition]:
    """意图驱动的召回条件翻译（v2.0：每个意图输出专用召回 channel）。"""
    if not intents:
        return []
    link = _linkage()
    anchor = time_info.anchor
    bucket = time_info.bucket
    industry = _industry_condition(entities, query)

    # 内容频道：query 关键词命中 > 意图默认
    query_channel = extract_content_channel(query) if query else []
    time_channel = _TIME_CHANNEL.get(anchor, "none")
    bucket_days = BUCKET_DAYS.get(bucket) if bucket else None
    # 默认时效：无显式时间的资讯类 query 按最新处理（freshness + 意图默认窗）
    if time_channel == "none" and bucket in (None, "no_limit") \
            and intents and intents[0].get("domain") != "SYSTEM":
        time_channel = "freshness"

    out: List[RecallCondition] = []
    for it in intents:
        iid = it["intent_id"]
        rule = link["intent_to_recall"].get(iid)
        if rule is None:
            rule = {"channel": "news", "sentiment": "any", "content_type_pref": [],
                    "role": "未配置", "weight": 0.3}
        pref = list(query_channel)
        for t in rule.get("content_type_pref", []):
            if t not in pref:
                pref.append(t)
        if bucket_days:
            window = bucket_days
        elif time_channel == "freshness":
            rule_w = rule.get("latest_window_days")
            window = int(rule_w) if rule_w is not None \
                else (link["time_anchor_defaults"].get("最新", {}).get("window_days") or 30)
        else:
            window = link["time_anchor_defaults"].get(anchor, {}).get("window_days")
        out.append(RecallCondition(
            intent_id=iid,
            channel=rule.get("channel", "news"),
            role=rule["role"],
            weight=float(rule["weight"]),
            sentiment=rule["sentiment"],
            content_type_pref=pref,
            industry=industry if not iid.startswith("SYSTEM-") or iid == "SYSTEM-05" else None,
            time_window_days=window,
            sort=link["time_anchor_defaults"].get(anchor, {}).get("sort"),
            time_channel=time_channel,
            time_bucket=bucket or "no_limit",
        ))
    # 主召回置顶
    out.sort(key=lambda c: -c.weight)
    return out
