# -*- coding: utf-8 -*-
"""意图 → 召回条件翻译层（复用 configs/recall_linkage.json）。

输入：intents + slots + entities + 原始 query
输出：List[RecallCondition]（主意图置顶，含行业过滤/情绪策略/内容类型偏好/时间窗）

渠道建模（对齐「时效性/内容类型走独立 channel」的设计）：
  - content channel：query 表面词 -> 内容类型频道（公告/财报/研报/新闻），
    命中时覆盖意图默认 content_type_pref（频道优先，意图默认补充在后）
  - time channel：time_anchor -> freshness/absolute/range/forward/none；
    「最新」的时间窗支持按意图覆盖（如公告 30d、财报 90d，默认 7d）
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

from qp.common.schema import RecallCondition

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")

# query 表面词 -> 内容频道（content_type）
_CHANNEL_KEYWORDS: List[tuple] = [
    (re.compile(r"公告|通告|披露|澄清|通函"), ["announcement"]),
    (re.compile(r"财报|年报|半年报|季报|业绩|营收|净利润|毛利率"), ["report"]),
    (re.compile(r"研报|研究报告|券商观点|行业分析"), ["report"]),
    (re.compile(r"新闻|资讯|消息|动态"), ["news"]),
]

# time_anchor -> time_channel
_TIME_CHANNEL = {"最新": "freshness", "历史时点": "absolute",
                 "区间": "range", "未来": "forward", "无": "none"}

# 显式时效分桶 -> 时间窗（天）：覆盖意图默认窗口（绝对时间日期过滤在检索端做）
_BUCKET_DAYS = {"in_one_day": 1, "one_week": 7, "one_month": 30, "one_year": 365}


@lru_cache(maxsize=1)
def _linkage() -> dict:
    from qp.common.taxonomy import _load
    return _load(os.path.join(CONFIG_DIR, "recall_linkage.json"))


def _industry_condition(object_type: str) -> Optional[Dict]:
    return _linkage()["industry_by_object_type"].get(object_type)


def _time_channel_window(intent_id: str, anchor: str) -> Optional[int]:
    """「最新」时间窗：意图覆盖（latest_window_days）> 全局默认。"""
    if anchor != "最新":
        return _linkage()["time_anchor_defaults"].get(anchor, {}).get("window_days")
    rule = _linkage()["intent_to_recall"].get(intent_id, {})
    if rule.get("latest_window_days") is not None:
        return int(rule["latest_window_days"])
    return _linkage()["time_anchor_defaults"].get("最新", {}).get("window_days")


def extract_content_channel(query: str) -> List[str]:
    """query 表面词 -> 内容频道（命中多个取并集，按匹配顺序）。"""
    chans: List[str] = []
    for pat, chans_ in _CHANNEL_KEYWORDS:
        if pat.search(query):
            for c in chans_:
                if c not in chans:
                    chans.append(c)
    return chans


def translate(intents: List[dict], slots: Dict, entities: List[dict],
              query: str = "") -> List[RecallCondition]:
    """意图驱动的召回条件翻译。情绪 boost 规则：query/实体中出现利好利空信号时叠加。"""
    if not intents:
        return []
    obj = slots.get("object_type", "无特定对象")
    anchor = slots.get("time_anchor", "无")
    industry = _industry_condition(obj)
    if industry and industry.get("level1") == "entity_link":
        # 行业对象：实体链接（INDUSTRY 实体文本 → 行业分类映射表，生产离线词典）
        ind_ents = [e["text"] for e in entities if e["type"] == "INDUSTRY"]
        industry = {"level1": "entity_link", "matched_industries": ind_ents} if ind_ents else None

    # 内容频道：query 关键词命中 > 意图默认
    query_channel = extract_content_channel(query) if query else []
    time_channel = _TIME_CHANNEL.get(anchor, "none")
    # 显式时效分桶（近一周/今年/2026年8月 等）-> 精确窗口，覆盖意图默认
    bucket = None
    if query:
        from qp.slots.extractor import detect_time_bucket
        bucket = detect_time_bucket(query)
    bucket_days = _BUCKET_DAYS.get(bucket) if bucket else None
    # 默认时效：无显式时间的资讯类 query 按最新处理（freshness + 意图默认窗）
    default_fresh = False
    if time_channel == "none" and bucket in (None, "no_limit") \
            and intents and intents[0].get("domain") != "SYSTEM":
        time_channel = "freshness"
        default_fresh = True

    out: List[RecallCondition] = []
    for it in intents:
        iid = it["intent_id"]
        rule = _linkage()["intent_to_recall"].get(iid)
        if rule is None:
            rule = {"sentiment": "any", "content_type_pref": [], "role": "未配置", "weight": 0.3}
        sentiment = rule["sentiment"]
        pref = list(query_channel)
        for t in rule.get("content_type_pref", []):
            if t not in pref:
                pref.append(t)
        if bucket_days:
            window = bucket_days
        elif time_channel == "freshness":
            rule_w = rule.get("latest_window_days")
            window = int(rule_w) if rule_w is not None \
                else (_time_channel_window(iid, "最新") or 30)
        else:
            window = _time_channel_window(iid, anchor)
        out.append(RecallCondition(
            intent_id=iid,
            role=rule["role"],
            weight=float(rule["weight"]),
            sentiment=sentiment,
            content_type_pref=pref,
            industry=industry if iid not in ("SYSTEM-01", "SYSTEM-02", "SYSTEM-03", "SYSTEM-04") else None,
            time_window_days=window,
            sort=_linkage()["time_anchor_defaults"].get(anchor, {}).get("sort"),
            time_channel=time_channel,
            time_bucket=bucket or "no_limit",
        ))
    # 主召回置顶
    out.sort(key=lambda c: -c.weight)
    return out
