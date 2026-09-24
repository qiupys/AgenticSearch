# -*- coding: utf-8 -*-
"""NER 弱标注数据集导出（服务器训练用）：data/raw/queries.jsonl -> data/ner/{train,val}.jsonl。

产出为 BiLSTM-CRF / BERT-CRF 训练数据（BIO），训练入口 ner/train.py 在服务器运行：
  python ner/train.py --data data/ner/train.jsonl --val data/ner/val.jsonl --epochs 20
"""
from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qp.ner.weak_label import weak_label_record, LABELS


def main(limit: int = 0, seed: int = 42, val_ratio: float = 0.02):
    src = os.path.join(ROOT, "data", "raw", "queries.jsonl")
    out_dir = os.path.join(ROOT, "data", "ner")
    os.makedirs(out_dir, exist_ok=True)

    records = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)["query"].strip()
            if not q:
                continue
            records.append(weak_label_record(q))
            if limit and len(records) >= limit:
                break

    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_ratio))
    val, train = records[:n_val], records[n_val:]

    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=1)

    n_ent = sum(len(r["entities"]) for r in records)
    by_type: dict = {}
    for r in records:
        for e in r["entities"]:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    print(f"train={len(train)} val={len(val)} entities={n_ent}")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"labels({len(LABELS)}) -> data/ner/labels.json")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(limit=limit)
