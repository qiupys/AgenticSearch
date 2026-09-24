# -*- coding: utf-8 -*-
"""LLM 辅助 NER：规则 NER 缺核心实体时用 LLM 补识别（prompt + response 解析）。"""
from __future__ import annotations

from qp.common.llm_client import default_client

ENTITY_TYPES = ["COMPANY", "STOCK", "STOCK_CODE", "INDEX", "INDUSTRY", "FUND",
                "MACRO", "ORG", "PRODUCT", "TIME", "METRIC"]

SYSTEM = "你是金融实体识别引擎。只输出一个 JSON 数组，不要输出任何其他文字。"

_PROMPT = """从金融搜索 query 中识别实体。实体类型定义：

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

规则：
1. 只输出 query 中出现的实体，不要改写、不要补充；
2. 一个词兼多类型时取最具体的一个（如「华夏机器人ETF」整体为 FUND，不拆 INDUSTRY）；
3. 宏观政策词（加息/降息/缩表）归 MACRO。

示例：
query: 华润新能源 001248 最新公告 2026年8月
输出: [{"text": "华润新能源", "type": "COMPANY"}, {"text": "001248", "type": "STOCK_CODE"}, {"text": "2026年8月", "type": "TIME"}]
query: 美联储 加息 影响
输出: [{"text": "美联储", "type": "MACRO"}, {"text": "加息", "type": "MACRO"}]
query: 半导体设备 国产替代 最新进展
输出: [{"text": "半导体设备", "type": "PRODUCT"}, {"text": "半导体", "type": "INDUSTRY"}]

query: {query}
只输出 JSON 数组（无实体则输出 []）:"""


def predict_llm(query: str) -> list[dict]:
    """LLM 实体识别。返回与 ner.predictor 兼容的实体列表；失败返回 []。"""
    client = default_client()
    if not client.switch("llm_ner"):
        return []
    # 用 replace 而非 format：prompt 内含 JSON 大括号示例
    prompt = _PROMPT.replace("{query}", query)
    out = client.chat_json(prompt, system=SYSTEM, max_tokens=512)
    if not isinstance(out, list):
        return []
    ents = []
    for item in out:
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
