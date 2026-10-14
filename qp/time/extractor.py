# -*- coding: utf-8 -*-
"""时间路（QP 三路之一）：时效分桶 + 时间锚点推断。

独立于意图/NER 的第三路：对 query 的时间表达做归一化与分桶，
产物 TimeInfo（anchor/raw/normalized/bucket）直接驱动召回翻译层
的时间通道（freshness/absolute/range/forward/none）与时间窗过滤。

分桶对齐引擎的 timelimit 与库内日期过滤：
  in_one_day / one_week / one_month / one_year / no_limit
"""
from __future__ import annotations

import datetime
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")


@lru_cache(maxsize=1)
def _lexicons() -> dict:
    with open(os.path.join(CONFIG_DIR, "lexicons.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _has_any(text: str, words: List[str]) -> Optional[str]:
    for w in words:
        if w and w in text:
            return w
    return None


_TIME_ABS = re.compile(r"(\d{4}年(\d{1,2}月)?(\d{1,2}日)?)|(\d{1,2}月\d{1,2}日)|(20\d{2}[-/.]\d{1,2}([-/.]\d{1,2})?)")
_TIME_RANGE = re.compile(r"从.{2,14}到.{2,14}|\d{4}年至\d{4}年|至今|以来")

# 时效分桶：显式相对时间表达 -> 桶（对齐引擎的 timelimit 与库内日期过滤）
_BUCKET_WORDS = [
    ("in_one_day", ["今天", "今日", "当日", "昨日", "昨天", "24小时", "日内", "刚刚", "刚刚发生"]),
    ("one_week", ["本周", "近一周", "一周内", "上周", "7天", "七天", "这一周"]),
    ("one_month", ["本月", "近一月", "近一个月", "一个月内", "上月", "30天", "近30天"]),
    ("one_year", ["今年", "近一年", "一年内", "年内", "365天", "年度"]),
]


def detect_time_bucket(text: str) -> str:
    """query 时效 -> 五桶之一：in_one_day / one_week / one_month / one_year / no_limit。

    显式相对词优先；绝对日期按距今远近归桶；趋势/走势类需要长历史回看 -> one_year；
    「最新」属 freshness 通道，无显式边界 -> no_limit。
    """
    low = text.lower()
    for bucket, words in _BUCKET_WORDS:
        if _has_any(low, [w.lower() for w in words]):
            return bucket
    m = _TIME_ABS.search(text)
    if m:
        y = re.search(r"(20\d{2})", m.group(0))
        mo = re.search(r"[年/.-/](\d{1,2})", m.group(0))
        if y and mo:
            try:
                month = int(mo.group(1))
                if 1 <= month <= 12:
                    dt = datetime.date(int(y.group(1)), month, 1)
                    days = (datetime.date.today() - dt).days
                    if days <= 1:
                        return "in_one_day"
                    if days <= 7:
                        return "one_week"
                    if days <= 31:
                        return "one_month"
                    if days <= 366:
                        return "one_year"
            except ValueError:
                pass
    # 趋势/走势/历史类：需要一年级别的历史数据（股价趋势、历史行情）
    if re.search(r"趋势|走势|历史(行情|数据|价格)|历年", text):
        return "one_year"
    return "no_limit"


def infer_time_anchor(text: str, entities: Optional[List[dict]] = None) -> Dict[str, str]:
    """推断时间锚点（时间路主入口）。

    query 可能同时含多个时间表达（如「最新公告 新闻 2026年8月」）：
    anchor 按优先级 区间 > 未来 > 最新 > 历史时点；raw 保留全部命中（| 分隔），
    normalized 附带绝对时间的归一化结果，供召回时间窗与时效过滤使用。
    """
    lx = _lexicons()
    hits: List[str] = []
    anchor = "无"
    normalized = ""

    if _TIME_RANGE.search(text):
        hits.append(_TIME_RANGE.search(text).group(0))
        anchor, normalized = "区间", "range"
    future = _has_any(text.lower(), [w.lower() for w in lx["time_expressions"]["relative_future"]])
    if future:
        hits.append(future)
        if anchor == "无":
            anchor, normalized = "未来", "future"
    latest = _has_any(text.lower(), [w.lower() for w in lx["time_expressions"]["relative_latest"]])
    abs_m = _TIME_ABS.search(text)
    if latest:
        hits.append(latest)
        if anchor in ("无", "历史时点"):
            anchor, normalized = "最新", (normalized if anchor == "历史时点" else "latest")
    if abs_m:
        hits.append(abs_m.group(0))
        if anchor == "无":
            anchor = "历史时点"
            normalized = _normalize_abs_time(abs_m.group(0))
        elif anchor == "最新" and not normalized:
            normalized = _normalize_abs_time(abs_m.group(0))
    if anchor == "最新" and normalized in ("", "latest"):
        normalized = "latest"

    return {"anchor": anchor, "raw": "|".join(hits), "normalized": normalized,
            "bucket": detect_time_bucket(text)}


def _normalize_abs_time(raw: str) -> str:
    """2026年8月25日 -> 2026-08-25（残缺保持粒度）。"""
    y = re.search(r"(20\d{2})年?", raw)
    m = re.search(r"(\d{1,2})月", raw)
    d = re.search(r"(\d{1,2})日", raw)
    if not y and not m:
        return raw
    parts = []
    if y:
        parts.append(y.group(1))
        if m:
            parts.append(f"{int(m.group(1)):02d}")
            if d:
                parts.append(f"{int(d.group(1)):02d}")
    return "-".join(parts) if parts else raw
