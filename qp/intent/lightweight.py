# -*- coding: utf-8 -*-
"""轻量意图分类模型（本机可训可跑的替代实现）。

- 特征：字符 n-gram（1~4）哈希 —— 中文按字、英文按词内字符均天然覆盖，双语可用
- 模型：LogisticRegression（softmax 多分类，42 类），predict_proba 出置信度
- 用途：本机跑通全链路 + 级联 Tier1 占位。生产替换为 Qwen2.5-0.5B 判别式三头
  （训练方案见 docs/03；ONNX 导出权重经 intent/qwen_head.py 接入，接口不变）

注意：该模型在规则银标注上训练，评测须用独立金标（合成评测集 + 160 条基线金标）。
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

import joblib

from qp.common.taxonomy import l2_domain, all_l2_ids
from qp.intent.base import IntentModel

_CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")


def _prep(text: str) -> str:
    """去掉中文之间的空格：对齐真实 query 分布（合成模板带分词空格）。"""
    return _CJK_SPACE.sub("", text)


class LightweightIntentModel(IntentModel):
    name = "lightweight-char-ngram-lr"

    def __init__(self, model_path: Optional[str] = None, topk: int = 3):
        self.model = None
        self.classes: List[str] = []
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        self.topk = topk

    # ---------------- fit / save / load ----------------
    def fit(self, queries: List[str], labels: List[str], model_path: Optional[str] = None, **clf_kwargs):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline

        # SGD + log loss = 线性逻辑回归的随机梯度版本：
        # 10万样本 x 42类 秒~分钟级收敛，适合流程验证；效果增强后可无缝换
        # LogisticRegression(lbfgs) 或 0.6B 判别头（docs/03）
        params = dict(loss="log_loss", alpha=1e-5, max_iter=30, tol=1e-3, random_state=42)
        params.update(clf_kwargs)
        self.model = Pipeline([
            ("vec", HashingVectorizer(analyzer="char", ngram_range=(1, 4), n_features=2 ** 20,
                                      alternate_sign=False, lowercase=True)),
            ("clf", SGDClassifier(**params)),
        ])
        self.model.fit([_prep(q) for q in queries], labels)
        self.classes = list(self.model.classes_)
        if model_path:
            self.save(model_path)
        return self

    def save(self, model_path: str):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump({"pipeline": self.model, "classes": self.classes}, model_path)

    def load(self, model_path: str):
        blob = joblib.load(model_path)
        self.model = blob["pipeline"]
        self.classes = list(blob["classes"])

    # ---------------- predict ----------------
    def predict(self, query: str, topk: int = 3) -> List[dict]:
        if self.model is None:
            return []
        probs = self.model.predict_proba([_prep(query)])[0]
        order = probs.argsort()[::-1][:topk]
        out = []
        for i in order:
            cid = self.classes[i]
            p = float(probs[i])
            if p < 0.05 or len(out) >= topk:
                continue
            out.append({"intent_id": cid, "domain": l2_domain(cid), "confidence": round(p, 4),
                        "source": "model"})
        # 复合意图：跨域次高且概率差不大时并列
        if len(out) >= 2 and out[1]["domain"] != out[0]["domain"] and out[1]["confidence"] >= 0.25:
            return out[:2]
        return out[:1]
