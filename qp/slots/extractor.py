# -*- coding: utf-8 -*-
"""槽位推断：object_type / time_anchor / composite / complexity / modality / output_form / trigger。

槽位不进分类树（对齐 v2.0 体系设计），全部由规则 + NER 实体结果推断。
"""
from __future__ import annotations

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


@lru_cache(maxsize=1)
def _lex() -> dict:
    lx = _lexicons()
    return {
        "index": [re.escape(x) for x in lx["index_names"]],
        "fund": lx["fund_keywords"],
        "industry": lx["industries"],
        "macro": lx["macro_terms"],
    }


def _has_any(text: str, words: List[str]) -> Optional[str]:
    for w in words:
        if w and w in text:
            return w
    return None


# ---------------- time ----------------

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
                import datetime
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
    """推断 time_anchor 槽位。

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


# ---------------- object_type ----------------

def infer_object_type(text: str, entities: List[dict]) -> str:
    """个股/公司 > 板块指数 > 基金ETF > 行业 > 宏观 > 无特定对象（按实体类型优先级）。"""
    types = {e["type"] for e in entities}
    if types & {"STOCK", "STOCK_CODE", "COMPANY"}:
        return "个股"
    if "INDEX" in types or re.search(r"板块|概念", text):
        return "板块指数"
    if "FUND" in types:
        return "基金ETF"
    if "INDUSTRY" in types or _has_any(text, _lex()["industry"]):
        return "行业"
    if "MACRO" in types or _has_any(text, _lex()["macro"]):
        return "宏观"
    return "无特定对象"


# ---------------- 其余槽位 ----------------

def infer_modality(text: str) -> str:
    if re.search(r"图|图片|照片|截图|image|picture|chart|图里的|看图", text, re.I):
        return "图片"
    if re.search(r"文档|pdf|PDF|file|document|附件", text):
        return "文档"
    if re.search(r"网页|链接|url|webpage|website", text, re.I):
        return "网页"
    return "文本"


def infer_output_form(text: str) -> str:
    if re.search(r"可视化|画个图|做个图|图表|chart|graph|visuali[sz]e", text, re.I):
        return "图表"
    if re.search(r"生成(一份|一个|文件)|导出|下载|export|generate a (file|report)", text, re.I):
        return "文件"
    if re.search(r"代码|code|脚本|script|api", text, re.I):
        return "代码"
    return "文本"


def infer_trigger(text: str) -> str:
    if re.search(r"定时|每天|订阅|alert me|notify", text, re.I):
        return "定时"
    if re.search(r"监控|异动|提醒我|track", text, re.I):
        return "事件触发"
    return "即时"


def infer_complexity(text: str) -> Optional[str]:
    """仅 SELECT 域使用：多条件/推理词判复杂。"""
    if not re.search(r"SELECT|筛选|选标的|受益|龙头|名单|概念股|哪些", text):
        return None
    if re.search(r"受益最大|最受益|弹性最大|符合.{2,20}(条件|要求|标准)|股价.{0,4}(低于|不超过|以下).{0,6}(元|块)且|同时", text):
        return "复杂"
    return "简单"


def extract_slots(text: str, entities: List[dict], composite: bool = False,
                  intent_ids: Optional[List[str]] = None) -> Dict[str, object]:
    """槽位推断总入口。intent_ids 用于 complexity 槽位的域判定。"""
    t = infer_time_anchor(text)
    slots = {
        "object_type": infer_object_type(text, entities),
        "time_anchor": t["anchor"],
        "composite": "复合" if composite else "单意图",
        "modality": infer_modality(text),
        "output_form": infer_output_form(text),
        "trigger": infer_trigger(text),
        "complexity": None,
    }
    if intent_ids and any((i or "").startswith("SELECT") for i in intent_ids):
        slots["complexity"] = infer_complexity(text)
    return slots, t
