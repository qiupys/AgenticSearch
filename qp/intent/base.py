# -*- coding: utf-8 -*-
"""意图模型抽象接口：规则 / 轻量替代 / 0.6B 判别头（ONNX）可插拔。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class IntentModel(ABC):
    """统一接口：predict(query) -> [{intent_id, domain, confidence, source}]"""

    name: str = "base"

    @abstractmethod
    def predict(self, query: str, topk: int = 3) -> List[dict]:
        ...


class CompositeIntentModel(IntentModel):
    """组合多个模型（按顺序首达可信者胜出），用于级联内 Tier1 内部组合。"""

    def __init__(self, models: List[IntentModel], min_confidence: float = 0.0):
        self.models = models
        self.min_confidence = min_confidence

    def predict(self, query: str, topk: int = 3) -> List[dict]:
        for m in self.models:
            out = m.predict(query, topk)
            if out and out[0]["confidence"] >= self.min_confidence:
                return out
        return out or []
