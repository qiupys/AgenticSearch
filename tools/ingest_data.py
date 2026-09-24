# -*- coding: utf-8 -*-
"""数据接入：
  1. 10 万条线上 query -> data/raw/queries.jsonl（带 id）
  2. 两份基线评测报告 HTML -> data/eval/baseline_gold.jsonl（160 条产品16类金标，中英各 80）
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAW_DIR = os.path.join(ROOT, "data", "raw")
EVAL_DIR = os.path.join(ROOT, "data", "eval")
QUERY_SRC = r"D:\UserIntension\data\搜索意图query.txt"
BASELINE_HTML = [
    r"C:\Users\viruser.v-desktop\Downloads\国内搜索引擎评测报告20260821.html",
    r"C:\Users\viruser.v-desktop\Downloads\海外搜索引擎评测报告20260821.html",
]


def ingest_queries():
    os.makedirs(RAW_DIR, exist_ok=True)
    dst = os.path.join(RAW_DIR, "queries.jsonl")
    n = 0
    with open(QUERY_SRC, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            q = line.strip()
            if not q:
                continue
            n += 1
            out.write(json.dumps({"id": f"q{n:06d}", "query": q}, ensure_ascii=False) + "\n")
    print(f"queries: {n} -> {dst}")
    return n


def ingest_baseline():
    """解析两份评测报告 HTML 内嵌 var DATA，抽取 160 条金标 query。"""
    os.makedirs(EVAL_DIR, exist_ok=True)
    dst = os.path.join(EVAL_DIR, "baseline_gold.jsonl")
    n = 0
    with open(dst, "w", encoding="utf-8") as out:
        for html_path, market in zip(BASELINE_HTML, ["domestic", "overseas"]):
            if not os.path.exists(html_path):
                print(f"[skip] not found: {html_path}")
                continue
            html = open(html_path, encoding="utf-8").read()
            m = re.search(r"var DATA = (\{[\s\S]*?\});", html)
            data = json.loads(m.group(1))
            for row in data["detail_rows"]:
                n += 1
                out.write(json.dumps({
                    "id": f"bl-{market}-{row['id']}",
                    "query": row["query"],
                    "gold_product_category": row["category"],
                    "market": market,
                    "lang": "en" if market == "overseas" else "zh",
                    "vs": row.get("vs"),
                    "reasons": row.get("reasons"),
                }, ensure_ascii=False) + "\n")
    print(f"baseline gold: {n} -> {dst}")
    return n


if __name__ == "__main__":
    ingest_queries()
    ingest_baseline()
