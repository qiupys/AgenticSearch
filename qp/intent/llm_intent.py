# -*- coding: utf-8 -*-
"""LLM 意图分类（Tier 2 升级路径）：prompt 构造 + response 解析。

对齐 q qp 大模型需求：0.6B 判别头置信不足时升级到大模型 API 补算。
prompt 内嵌 42 类 L2 清单（id + 名称 + 域），few-shot 2 例，
要求只输出 JSON：{"intent_id": "...", "confidence": 0.xx}
"""
from __future__ import annotations

from qp.common.llm_client import default_client
from qp.common.taxonomy import all_l2_ids, l2_domain, l2_name

SYSTEM = "你是金融搜索查询理解(QP)引擎。只输出一个 JSON 对象，不要输出任何其他文字。"


def _build_class_list() -> str:
    lines = []
    for iid in all_l2_ids():
        lines.append(f"  {iid}  {l2_name(iid)}  [{l2_domain(iid)}]")
    return "\n".join(lines)


_PROMPT_TMPL = """对以下金融搜索 query 做意图分类。

意图清单（intent_id | 名称 | 所属域）：
{classes}

示例：
query: 华润新能源 001248 最新公告 新闻 2026年8月
输出: {{"intent_id": "INFO-07", "confidence": 0.9}}
query: 说说九安医疗后面还有机会涨吗
输出: {{"intent_id": "PREDICT-02", "confidence": 0.85}}

规则：
- intent_id 必须严格取自清单
- confidence 为 0-1 的浮点数，表示你对判断的把握
- 无法判断时输出 {{"intent_id": "SYSTEM-05", "confidence": 0.0}}

query: {query}
只输出 JSON:"""

_class_list_cache = None


def classify_intent(query: str) -> dict | None:
    """LLM 意图分类。返回 {"intent_id", "domain", "confidence", "source": "llm"}
    或 None（服务不可用/解析失败，调用方降级兜底）。"""
    global _class_list_cache
    client = default_client()
    if not client.switch("llm_intent"):
        return None
    if _class_list_cache is None:
        _class_list_cache = _build_class_list()
    prompt = _PROMPT_TMPL.format(classes=_class_list_cache, query=query)
    out = client.chat_json(prompt, system=SYSTEM, max_tokens=256)
    if not isinstance(out, dict):
        return None
    intent_id = out.get("intent_id")
    if not intent_id or intent_id not in set(all_l2_ids()):
        return None
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return {"intent_id": intent_id, "domain": l2_domain(intent_id),
            "confidence": round(conf, 4), "source": "llm"}
