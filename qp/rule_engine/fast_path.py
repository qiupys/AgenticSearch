# -*- coding: utf-8 -*-
"""Tier 0 规则快通道引擎。

加载 configs/rule_patterns.yaml，对 query 打分并输出：
  - top1 类 + 置信度
  - 复合意图（跨域次高类得分 >= best*ratio 时并列输出）
  - 命中明细（供评测/调试/银标注）

设计对齐 v2.1：高频意图规则直出，<1ms；置信度不足交由级联降级到模型。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import yaml

from qp.common.taxonomy import l2_domain

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")


class RuleEngine:
    def __init__(self, patterns_path: Optional[str] = None, config: Optional[dict] = None):
        path = patterns_path or os.path.join(CONFIG_DIR, "rule_patterns.yaml")
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.engine_cfg = cfg["engine"]
        self.classes: List[dict] = []
        self._compiled: List[Tuple[str, List[Tuple[re.Pattern, float]], List[re.Pattern]]] = []
        for cls_id, cls in cfg["classes"].items():
            pats = [(re.compile(p["re"]), float(p.get("w", 1.0))) for p in cls.get("patterns", [])]
            negs = [re.compile(x) for x in cls.get("not_patterns", [])] if "not_patterns" in cls else []
            self._compiled.append((cls_id, pats, negs))
        self._pats_by_cls: Dict[str, List[Tuple[re.Pattern, float]]] = {
            cls_id: pats for cls_id, pats, _ in self._compiled}
        self.class_names = {cid: c["name"] for cid, c in cfg["classes"].items()}
        self._config = config or {}

    # ------------------------------------------------------------------
    def score(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """返回 (top1_class, score, all_scores)。text 应为归一化后的 query。"""
        scores: Dict[str, float] = {}
        hit_detail: Dict[str, List[str]] = {}
        for cls_id, pats, negs in self._compiled:
            if any(n.search(text) for n in negs):
                continue
            s = 0.0
            for pat, w in pats:
                if pat.search(text):
                    s += w
            if s > 0:
                scores[cls_id] = s
        if not scores:
            return "", 0.0, {}
        # 最高分；同分取文件顺序（self._compiled 顺序）靠前者
        best_id, best_s = "", -1.0
        for cls_id, _, _ in self._compiled:
            s = scores.get(cls_id, 0.0)
            if s > best_s:
                best_id, best_s = cls_id, s
        return best_id, best_s, scores

    # ------------------------------------------------------------------
    def classify(self, query: str) -> dict:
        """完整规则推理：top1 + 置信度 + 复合意图。"""
        best_id, best_s, scores = self.score(query)
        min_score = float(self.engine_cfg.get("min_score", 1.0))
        if not best_id or best_s < min_score:
            return {"hit": False, "intents": [], "scores": scores}

        # 细粒度置信度：信号总分 + 独立命中模式数 + 唯一命中类加成。
        # 旧公式 0.62+0.12*(s-1) 在 s=2 处堆积（绝大多数 query 恒 0.74），
        # 这里按证据强度展开：s=1,n=1 -> ~0.61；s=2,n=2 -> ~0.73；
        # s=3,n=3 且唯一命中 -> ~0.89；上限 0.97。
        pats = self._pats_by_cls[best_id]
        n_pats = sum(1 for pat, _ in pats if pat.search(query))
        conf = min(0.97, 0.58 + 0.10 * (best_s - 1.0) + 0.025 * (n_pats - 1)
                   + (0.03 if len(scores) == 1 else 0.0))

        intents = [{"intent_id": best_id, "domain": l2_domain(best_id), "confidence": round(conf, 4)}]

        # 复合/多标签：次高类得分 >= best*ratio 时并列输出。
        # 跨域 → 复合意图；同域 → 同域多标签（如「公告+研报」→ INFO-07+INFO-08）。
        ratio = float(self.engine_cfg.get("composite_ratio", 0.8))
        second: List[Tuple[str, float]] = []
        for cid, s in scores.items():
            if cid == best_id:
                continue
            if s >= best_s * ratio:
                second.append((cid, s))
        second.sort(key=lambda x: -x[1])
        for cid, s in second[:2]:   # 最多 1+2=3 个意图标签
            c2 = max(0.30, conf * (s / best_s))
            intents.append({"intent_id": cid, "domain": l2_domain(cid), "confidence": round(c2, 4)})

        return {"hit": True, "intents": intents, "scores": scores}


@lru_cache(maxsize=1)
def default_engine() -> RuleEngine:
    return RuleEngine()
