# -*- coding: utf-8 -*-
"""LLM 辅助 NER + 时间（路2+路3 合一调用）。

v2.0 三路架构：NER 与时间合并为一次 LLM 调用（省一次往返），
LLM 优先产出实体与时间要素；失败/关闭时由 pipeline 回落规则路。
"""
from __future__ import annotations

from qp.common.llm_client import default_client

ENTITY_TYPES = ["COMPANY", "STOCK", "STOCK_CODE", "INDEX", "INDUSTRY", "FUND",
                "MACRO", "ORG", "PRODUCT", "TIME", "METRIC"]

ANCHORS = {"最新", "历史时点", "区间", "未来", "无"}
BUCKETS = {"in_one_day", "one_week", "one_month", "one_year", "no_limit"}

SYSTEM = "你是金融查询理解引擎。只输出一个 JSON 对象，不要输出任何其他文字。"

_PROMPT = """对金融搜索 query 做实体识别 + 时间要素提取。

实体类型定义：
- COMPANY   公司/企业/品牌（腾讯、智谱、瑞幸咖啡）
- STOCK     个股名称（贵州茅台、宁德时代）
- STOCK_CODE 股票/基金代码（001248、0700.HK）
- INDEX     指数/板块指数（沪深300、中证机器人）
- INDUSTRY  行业/板块（半导体、白酒、新能源车）
- FUND      基金/ETF（华夏机器人ETF、562500）
- MACRO     宏观主体与政策（美联储、央行、国务院、加息、CPI、GDP）
- ORG       其他机构/组织（世界银行、证监会、中金公司）
- PRODUCT   产品/业务/技术（换电模式、智驾方案、iPhone）
- TIME      时间表达（2026年8月、近一周、今天）
- METRIC    财务/行情指标（营收、净利润、毛利率、股价）

实体规则：
1. 只输出 query 中出现的实体，不要改写、不要补充；
2. 一个词兼多类型时取最具体的一个（如「华夏机器人ETF」整体为 FUND，不拆 INDUSTRY）；
3. 宏观政策词（加息/降息/缩表）归 MACRO。

时间要素（time）：
- anchor 取值：最新 / 历史时点 / 区间 / 未来 / 无
- bucket 取值：in_one_day / one_week / one_month / one_year / no_limit
- normalized：绝对时间归一化为 2026-08-25 / 2026-08 格式，「最新」填 latest

示例：
query: 华润新能源 001248 最新公告 2026年8月
输出: {"entities": [{"text": "华润新能源", "type": "COMPANY"}, {"text": "001248", "type": "STOCK_CODE"}, {"text": "2026年8月", "type": "TIME"}], "time": {"anchor": "历史时点", "normalized": "2026-08", "bucket": "one_year"}}
query: 美联储 加息 影响
输出: {"entities": [{"text": "美联储", "type": "MACRO"}, {"text": "加息", "type": "MACRO"}], "time": {"anchor": "无", "normalized": "", "bucket": "no_limit"}}
query: 近一周黄金走势
输出: {"entities": [{"text": "黄金", "type": "INDUSTRY"}], "time": {"anchor": "最新", "normalized": "latest", "bucket": "no_limit"}}

query: {query}
只输出 JSON 对象（无实体则 entities 为空数组）:"""


def _parse_ents(out, query: str) -> list[dict]:
    """解析实体数组（容忍旧版纯数组输出）。"""
    items = out.get("entities") if isinstance(out, dict) else out
    if not isinstance(items, list):
        return []
    ents = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        etype = str(item.get("type", "")).strip().upper()
        if not text or etype not in ENTITY_TYPES or text not in query:
            continue
        ents.append({"text": text, "type": etype,
                     "start": query.find(text),
                     "end": query.find(text) + len(text),
                     "source": "llm"})
    return ents


def _parse_time(out) -> dict | None:
    """解析时间要素；字段缺失/非法返回 None（由调用方回落规则）。"""
    if not isinstance(out, dict):
        return None
    t = out.get("time")
    if not isinstance(t, dict):
        return None
    anchor = str(t.get("anchor", "")).strip()
    bucket = str(t.get("bucket", "")).strip().lower()
    if anchor not in ANCHORS or bucket not in BUCKETS:
        return None
    return {"anchor": anchor,
            "normalized": str(t.get("normalized", "")).strip(),
            "bucket": bucket}


def predict_llm_qp(query: str) -> tuple[list[dict], dict | None]:
    """LLM 一次调用产出实体 + 时间要素（路2+路3 合一）。

    返回 (entities, time_hint)；time_hint 为 None 时由调用方回落规则时间路。
    LLM 关闭/失败/解析失败返回 ([], None)。
    """
    client = default_client()
    if not client.switch("llm_ner"):
        return [], None
    # 用 replace 而非 format：prompt 内含 JSON 大括号示例
    prompt = _PROMPT.replace("{query}", query)
    out = client.chat_json(prompt, system=SYSTEM, max_tokens=768)
    if not out:
        return [], None
    return _parse_ents(out, query), _parse_time(out)


def predict_llm(query: str) -> list[dict]:
    """LLM 实体识别（兼容旧接口，仅实体）。失败返回 []。"""
    ents, _ = predict_llm_qp(query)
    return ents
