# -*- coding: utf-8 -*-
"""银标注构建：规则引擎对 10 万条线上 query 打标 -> data/silver/intent_silver.jsonl。

银标注 = 弱监督训练数据（Tier1 轻量模型训练用）。
高分样本（score>=2）置信较高，低分样本可作为难例池供后续人工/LLM teacher 清洗。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qp.rule_engine.fast_path import default_engine
from qp.common.taxonomy import l2_domain


def main():
    src = os.path.join(ROOT, "data", "raw", "queries.jsonl")
    dst = os.path.join(ROOT, "data", "silver", "intent_silver.jsonl")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    engine = default_engine()
    dist: Counter = Counter()
    n = high = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            q = rec["query"]
            r = engine.classify(q)
            n += 1
            if not r["hit"]:
                dist["<no_hit>"] += 1
                continue
            top = r["intents"][0]
            score = r["scores"].get(top["intent_id"], 0)
            if score >= 2:
                high += 1
            dist[top["intent_id"]] += 1
            fout.write(json.dumps({
                "id": rec["id"],
                "query": q,
                "silver_l2": top["intent_id"],
                "silver_l1": l2_domain(top["intent_id"]),
                "confidence": top["confidence"],
                "rule_score": score,
                "high_confidence": score >= 2,
                "multi_intents": [i["intent_id"] for i in r["intents"]],
                "lang": "zh",
            }, ensure_ascii=False) + "\n")
    print(f"total={n} high_confidence(score>=2)={high} ({high / n * 100:.1f}%)")
    for cid, c in dist.most_common():
        print(f"  {cid}: {c}")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
