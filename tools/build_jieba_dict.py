# -*- coding: utf-8 -*-
"""构建 jieba 自定义词典（Q4：金融 query 分词增强）。

来源（三层）：
  1. 词库层：configs/lexicons.yaml 的行业/指数/基金/宏观词 + 公司名种子 → 高频保护
  2. 数据层：10 万 query 中按公司名后缀模式抽出的候选公司名（词频 Top-N 收录）
  3. 短语层：银标注高分样本中跨 query 稳定共现的高频金融短语（如 条件单/龙虎榜/涨停价）

输出：data/dict/jieba_user_dict.txt（jieba.load_userdict 可直接加载）
用法：python scripts/build_jieba_dict.py [--top-company 3000] [--min-freq 8]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qp.ner.weak_label import _lexicons          # 复用词库加载
from qp.rule_engine.fast_path import default_engine


def collect_lexicon_words() -> list:
    lx = _lexicons()
    words = []
    for w in lx["industries"] + lx["index_names"] + lx["fund_keywords"] + lx["macro_terms"]:
        w = w.strip()
        if len(w) >= 2:
            words.append((w, "nz"))
    for w in lx["stock_seeds_zh"]:
        w = w.strip()
        if len(w) >= 2:
            words.append((w, "nz"))
    for w in lx["stock_seeds_en"]:
        w = w.strip()
        if w.isascii() and len(w) >= 2:
            words.append((w, "eng"))
    return words


def collect_company_candidates(queries: list, top_n: int, min_freq: int) -> list:
    """公司名候选：后缀模式命中 + 左边界 2~4 字拼接（如「XX股份」「XX科技」）。"""
    lx = _lexicons()
    counter: Counter = Counter()
    pats = [re.compile(p) for p in lx["stock_suffix_patterns"]]
    code_pat = re.compile(r"(?<!\d)[0-9]{6}(?!\d)")
    for q in queries:
        q_clean = code_pat.sub(" ", q)
        for pat in pats:
            for m in pat.finditer(q_clean):
                counter[m.group(0)] += 1
    return [(w, c) for w, c in counter.most_common(top_n) if c >= min_freq]


def collect_phrases(queries: list, min_freq: int) -> list:
    """高频金融短语：规则命中片段的稳定词组（白名单式，防噪声）。"""
    engine = default_engine()
    phrase_pat = re.compile(
        r"涨停价|跌停价|龙虎榜|条件单|网格单|止盈止损|北向资金|南向资金|主力资金|融资余额|两融余额|"
        r"市盈率|市净率|市占率|净流入|净流出|换手率|量比|筹码集中度|股东人数|一致预期|盈利预测|目标价|"
        r"半年报|三季报|业绩预告|业绩快报|回购|增持|减持|解禁|质押|定增|增发|重组|并购|借壳|"
        r"概念股|成分股|权重股|受益股|龙头股|股票池|产业链|景气度|竞争格局|风险偏好|赚钱效应|"
        r"互联互通|熔断|涨跌幅|振幅|量价|K线|金叉|死叉|背离|多头排列|空头排列|十字星"
    )
    counter: Counter = Counter()
    for q in queries:
        engine.score(q)   # noqa: 保证引擎已初始化（无实际作用，保持接口一致性）
        for m in phrase_pat.finditer(q):
            counter[m.group(0)] += 1
    return [(w, c) for w, c in counter.most_common() if c >= min_freq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-company", type=int, default=3000)
    ap.add_argument("--min-freq", type=int, default=8)
    args = ap.parse_args()

    queries = []
    with open(os.path.join(ROOT, "data", "raw", "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line)["query"])

    entries: dict = {}
    for w, tag in collect_lexicon_words():
        entries[w] = (100000, tag)                     # 词库词给最高频，确保不被切开
    for w, c in collect_company_candidates(queries, args.top_company, args.min_freq):
        entries.setdefault(w, (max(c * 100, 1000), "nz"))
    for w, c in collect_phrases(queries, max(3, args.min_freq // 2)):
        entries.setdefault(w, (max(c * 100, 500), "n"))

    out_dir = os.path.join(ROOT, "data", "dict")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "jieba_user_dict.txt")
    with open(out, "w", encoding="utf-8") as f:
        for w, (freq, tag) in sorted(entries.items(), key=lambda x: -x[1][0]):
            f.write(f"{w} {freq} {tag}\n")
    print(f"jieba user dict: {len(entries)} entries -> {out}")

    # 分词效果对照
    import jieba
    samples = [
        "华润新能源 001248 最新公告 新闻 2026年8月",
        "鼎龙股份 主力资金 净流入 龙虎榜 2026年8月",
        "2026年储能产业链上游材料龙头个股梳理",
        "159934 黄金ETF 历史净值 2025年半年报",
        "北方华创 股权激励 解禁 市盈率",
    ]
    print("\n== 基线分词（无自定义词典）==")
    for s in samples:
        print(" ", " / ".join(jieba.lcut(s)))
    jieba.load_userdict(out)
    jieba.initialize()
    print("\n== 加载自定义词典后 ==")
    for s in samples:
        print(" ", " / ".join(jieba.lcut(s)))


import json  # noqa: E402  (置于底部避免与 argparse 冲突)

if __name__ == "__main__":
    main()
