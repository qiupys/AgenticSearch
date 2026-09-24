# -*- coding: utf-8 -*-
"""训练轻量意图模型（本机替代实现）：银标注 -> char n-gram LR -> data/models/lightweight_intent.joblib。"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qp.intent.lightweight import LightweightIntentModel


def main():
    src = os.path.join(ROOT, "data", "silver", "intent_silver.jsonl")
    dst = os.path.join(ROOT, "data", "models", "lightweight_intent.joblib")
    queries, labels = [], []
    with open(src, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            queries.append(rec["query"])
            labels.append(rec["silver_l2"])
    if not queries:
        print("no silver data; run scripts/build_silver.py first")
        return
    print(f"training on {len(queries)} samples, {len(set(labels))} classes")
    model = LightweightIntentModel()
    model.fit(queries, labels, model_path=dst)
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
