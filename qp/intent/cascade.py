# -*- coding: utf-8 -*-
"""三级级联编排（v2.1）：

  Tier 0 规则快通道   —— 高频意图直出（目标 ~65% 流量，<1ms）
  Tier 1 判别模型     —— 轻量替代 / Qwen 0.6B 判别头（校准后 max-p >= 0.75 接受）
  Tier 2 升级路径     —— 低置信兜底（SYSTEM-05）+ escalate 标记
                        （生产为三选一：大模型API异步补算 / 兜底+追问 / 直接兜底）

返回结构统一为级联结果：{intents, tier, escalated, rule_detail, model_detail}
"""
from __future__ import annotations

import os
from typing import List, Optional

import yaml

from qp.common.taxonomy import l2_domain
from qp.intent.base import IntentModel
from qp.intent.lightweight import LightweightIntentModel
from qp.intent.qwen_head import QwenHeadIntentModel
from qp.rule_engine.fast_path import RuleEngine

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")


class IntentCascade:
    def __init__(self, config_path: Optional[str] = None, rule_engine: Optional[RuleEngine] = None):
        cfg_path = config_path or os.path.join(CONFIG_DIR, "service.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        casc = self.cfg["cascade"]
        self.rule_enabled = casc["tier0_rule"]["enabled"]
        self.rule_min_conf = float(casc["tier0_rule"].get("min_confidence", 0.70))
        self.model_enabled = casc["tier1_model"]["enabled"]
        self.max_p = float(casc["tier1_model"].get("max_p_threshold", 0.75))
        self.fallback_intent = casc["tier2_escalate"].get("fallback_intent", "SYSTEM-05")

        self.engine = rule_engine or RuleEngine()

        # Tier1 模型：Qwen ONNX 优先，轻量模型兜底（组合式）
        models: List[IntentModel] = []
        qwen_path = casc["tier1_model"].get("qwen_onnx_path")
        if qwen_path:
            m = QwenHeadIntentModel(qwen_path)
            if m.is_ready():
                models.append(m)
        light_path = casc["tier1_model"].get("model_path")
        lm = LightweightIntentModel(light_path)
        if lm.model is not None:
            # 轻量模型为未校准 SGD 概率，使用独立（更宽松）门限；
            # 0.6B 校准头用 max_p_threshold（0.75）
            lm.accept_threshold = float(casc["tier1_model"].get("lightweight_max_p", 0.45))
            models.append(lm)
        self.tier1_models = models

    # ------------------------------------------------------------------
    def predict(self, query: str, normalized: Optional[str] = None) -> dict:
        text = normalized or query
        detail: dict = {"rule": None, "model": None}

        # ---- Tier 0：规则（仅 SYSTEM 域短路直出；其余命中留作回落候选）----
        rule_fallback = None
        if self.rule_enabled:
            r = self.engine.classify(text)
            detail["rule"] = {"hit": r["hit"], "intents": r.get("intents"), "scores": r.get("scores")}
            if r["hit"] and r["intents"][0]["confidence"] >= self.rule_min_conf:
                top = r["intents"][0]
                if l2_domain(top["intent_id"]) == "SYSTEM":
                    return {"intents": [dict(i, source="rule") for i in r["intents"]],
                            "tier": 0, "escalated": False, "detail": detail}
                rule_fallback = r

        # ---- LLM 优先（llm_first 开关，默认开）：失败/低置信回落其他层 ----
        from qp.common.llm_client import default_client
        llm_first = default_client().switch("llm_first")
        if llm_first and default_client().switch("llm_intent"):
            out = self._llm_tier(text, detail)
            if out:
                return out

        # ---- Tier 1：轻量判别模型 ----
        if self.model_enabled and self.tier1_models:
            for m in self.tier1_models:
                out = m.predict(text, topk=3)
                thr = getattr(m, "accept_threshold", self.max_p)
                if out and out[0]["confidence"] >= thr:
                    detail["model"] = {"model": m.name, "intents": out}
                    return {"intents": [dict(i, source="model") for i in out],
                            "tier": 1, "escalated": False, "detail": detail}
                if out:
                    detail["model"] = {"model": m.name, "intents": out, "rejected": True}

        # ---- 规则回落（非 SYSTEM 域的高置信规则命中）----
        if rule_fallback:
            return {"intents": [dict(i, source="rule") for i in rule_fallback["intents"]],
                    "tier": 0, "escalated": False, "detail": detail}

        # ---- LLM 兜底尝试（llm_first=false 时的原始顺序）----
        if not llm_first:
            out = self._llm_tier(text, detail)
            if out:
                return out

        fb = {"intent_id": self.fallback_intent, "domain": l2_domain(self.fallback_intent),
              "confidence": 0.0, "source": "fallback"}
        intents = [fb]
        return {"intents": intents, "tier": 2, "escalated": True, "detail": detail}

    def _llm_tier(self, text: str, detail: dict) -> Optional[dict]:
        """LLM 意图判定（qwen3.8-27b）；服务不可用/解析失败/低置信返回 None。"""
        try:
            from qp.intent.llm_intent import classify_intent
            llm_out = classify_intent(text)
        except Exception:
            return None
        if llm_out and llm_out["confidence"] >= float(
                self.cfg.get("cascade", {}).get("tier2_escalate", {})
                .get("llm_min_confidence", 0.55)):
            detail["llm"] = llm_out
            return {"intents": [dict(llm_out)],
                    "tier": 2, "escalated": False, "detail": detail}
        if llm_out:
            detail["llm"] = dict(llm_out, rejected=True)
        return None
