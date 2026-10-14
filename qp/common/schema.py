# -*- coding: utf-8 -*-
"""QP 统一输出 Schema（v2.0 三路架构）。

线上 QP 服务对每条 query 输出一个 QPOutput 结构，三路并行产出：
  intents   —— 路一：多标签意图（1~3 个，级联：规则→模型→LLM）
  entities  —— 路二：NER 实体（规则+LLM 级联 + 实体链指 + 市场定向）
  time      —— 路三：时间锚点与时效分桶（纯规则）
  recall    —— 意图驱动的召回翻译条件（意图×channel 深度绑定）

该结构同时是离线评测的消费对象。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

QP_OUTPUT_VERSION = "2.0"


@dataclass
class Intent:
    intent_id: str                 # e.g. "INFO-07"
    domain: str                    # e.g. "INFO"
    confidence: float              # [0, 1]
    source: str = "rule"           # rule | model | llm | fallback


@dataclass
class Entity:
    text: str
    type: str                      # STOCK / STOCK_CODE / COMPANY / INDEX / FUND / INDUSTRY / MACRO / TIME / PRODUCT
    start: int
    end: int
    source: str = "rule"           # rule | llm | model
    eid: str = ""                  # 实体链指：词表实体 ID（未链指为空）
    canonical: str = ""            # 实体链指：词表规范名（如 智谱 -> 智谱AI）
    market: str = ""               # 市场定向：HK / US / A（两地上市实体 + query 市场词消歧后写入）


@dataclass
class TimeInfo:
    anchor: str = "无"             # 最新 / 历史时点 / 区间 / 未来 / 无
    raw: str = ""                  # 原文时间表达
    normalized: str = ""           # 归一化，如 2026-08 / 2026-08-25 / latest
    bucket: str = "no_limit"       # in_one_day / one_week / one_month / one_year / no_limit


@dataclass
class RecallCondition:
    intent_id: str
    channel: str                   # 意图专用召回通道：news / disclosure / research / forecast / screener / education / multimedia / none
    role: str                      # 主召回 / 数据补充 / 辅助 / 不走内容召回 ...
    weight: float
    sentiment: str                 # any / neutral优先 / mixed / boost_positive / -
    content_type_pref: List[str] = field(default_factory=list)
    industry: Optional[Dict[str, Any]] = None   # {level1: [...], finance_sub: [...], secondary: ...}
    time_window_days: Optional[int] = None
    sort: Optional[str] = None
    time_channel: Optional[str] = None   # freshness / absolute / range / forward / none
    time_bucket: Optional[str] = None    # in_one_day / one_week / one_month / one_year / no_limit


@dataclass
class QPOutput:
    query: str
    normalized: str
    lang: str = "zh"                       # zh / en
    intents: List[Intent] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    time: TimeInfo = field(default_factory=TimeInfo)
    recall: List[RecallCondition] = field(default_factory=list)
    stats_category_16: Optional[str] = None   # 产品 16 类口径（反向映射，供统计对齐）
    tier: int = -1                          # 0 规则 / 1 模型 / 2 LLM/升级
    escalated: bool = False                 # 是否触发升级路径
    latency_ms: Dict[str, float] = field(default_factory=dict)
    version: str = QP_OUTPUT_VERSION
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def primary_intent(self) -> Optional[Intent]:
        return self.intents[0] if self.intents else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def timing(fn, *args, key: str, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        return out, (time.perf_counter() - t0) * 1000, key
