# -*- coding: utf-8 -*-
"""评测指标体系。

对齐业务目标（QP大模型需求文档）与统一体系指标（intent_taxonomy.json）：
  - L1 top1 acc（8 域）/ L2 top1 acc（42 类）/ L2 top3 acc
  - 产品 16 类口径 acc（acceptable 集合命中即对）→ 对应「统计层面 85%」
  - 主意图 L1 命中率（召回 channel 选择正确）→ 对应「排序层面 95%」
  - 多标签 P/R/F1（micro/macro）
  - NER 实体级 P/R/F1（span+type exact match）→ 对应「标的/时间识别 99%」
  - 分维度切片：语言 / 长度桶 / 含代码 / 含时间词
  - 级联分层统计：Tier 流量占比 / 升级率
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from qp.common.taxonomy import acceptable_l1, acceptable_l2, l2_domain


# ---------------- 意图指标 ----------------

def intent_metrics(records: List[dict]) -> dict:
    """records: [{query, gold_l2 (或 None), gold_product_category, pred_l2, pred_l1, tier, escalated, lang}]"""
    n = len(records)
    m: Dict[str, object] = {"n": n}
    if n == 0:
        return m

    l1_hit = l2_hit = l2_top3_hit = prod_hit = 0
    prod_n = 0
    confusions: Counter = Counter()
    per_class: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    tiers: Counter = Counter()
    escalated = 0

    for r in records:
        gold_l2 = r.get("gold_l2")
        gold_cat = r.get("gold_product_category")
        pred_l2 = r.get("pred_l2")
        pred_l1 = r.get("pred_l1")
        pred_top3 = r.get("pred_l2_top3") or []
        tiers[r.get("tier", -1)] += 1
        if r.get("escalated"):
            escalated += 1

        if gold_l2:
            per_class[gold_l2]["fn" if pred_l2 != gold_l2 else "tp"] += 1
            if pred_l2 != gold_l2:
                per_class[pred_l2]["fp"] += 1
                confusions[(gold_l2, pred_l2)] += 1
            if pred_l2 == gold_l2:
                l2_hit += 1
            if gold_l2 in pred_top3:
                l2_top3_hit += 1
            if pred_l1 == l2_domain(gold_l2):
                l1_hit += 1

        if gold_cat:
            prod_n += 1
            ok_l2 = pred_l2 in acceptable_l2(gold_cat)
            ok_l1 = pred_l1 in acceptable_l1(gold_cat)
            if ok_l2:
                prod_hit += 1
            # 排序层面口径：召回 channel 由 L1 驱动
            r["rank_level_hit"] = ok_l1
            r["stats_level_hit"] = ok_l2

    m["L1_top1_acc"] = round(l1_hit / n, 4)
    m["L2_top1_acc"] = round(l2_hit / n, 4)
    m["L2_top3_acc"] = round(l2_top3_hit / n, 4)
    if prod_n:
        m["product16_acc"] = round(prod_hit / prod_n, 4)
    m["tier_dist"] = dict(tiers)
    m["escalate_rate"] = round(escalated / n, 4)
    m["rank_level_acc"] = round(sum(1 for r in records if r.get("rank_level_hit")) / prod_n, 4) if prod_n else None
    m["stats_level_acc"] = round(sum(1 for r in records if r.get("stats_level_hit")) / prod_n, 4) if prod_n else None

    # per-class P/R/F1
    pcs = {}
    for cid, c in per_class.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rc = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        if c["tp"] + c["fn"] > 0:
            pcs[cid] = {"P": round(p, 4), "R": round(rc, 4), "F1": round(f1, 4), "support": c["tp"] + c["fn"]}
    m["per_class"] = pcs
    tp_sum = sum(c["tp"] for c in per_class.values())
    fp_sum = sum(c["fp"] for c in per_class.values())
    fn_sum = sum(c["fn"] for c in per_class.values())
    micro_p = tp_sum / (tp_sum + fp_sum) if tp_sum + fp_sum else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if tp_sum + fn_sum else 0.0
    m["micro_P"] = round(micro_p, 4)
    m["micro_R"] = round(micro_r, 4)
    m["micro_F1"] = round(2 * micro_p * micro_r / (micro_p + micro_r), 4) if micro_p + micro_r else 0.0
    if pcs:
        m["macro_F1"] = round(sum(v["F1"] for v in pcs.values()) / len(pcs), 4)

    # top 混淆对
    m["top_confusions"] = [
        {"gold": g, "pred": p, "count": c}
        for (g, p), c in confusions.most_common(15)
    ]
    return m


def slice_metrics(records: List[dict]) -> dict:
    """分维度切片：语言 / 长度 / 含代码 / 含时间词。"""
    import re
    code_pat = re.compile(r"(?<!\d)[0-9]{6}(?!\d)")

    def acc(rs):
        rs = [r for r in rs if r.get("gold_l2")]
        if not rs:
            return None
        hit = sum(1 for r in rs if r.get("pred_l2") == r["gold_l2"])
        return {"n": len(rs), "L2_acc": round(hit / len(rs), 4)}

    out = {
        "lang_zh": acc([r for r in records if r.get("lang") == "zh"]),
        "lang_en": acc([r for r in records if r.get("lang") == "en"]),
        "len_lt10": acc([r for r in records if len(r.get("query", "")) < 10]),
        "len_10_19": acc([r for r in records if 10 <= len(r.get("query", "")) < 20]),
        "len_20_29": acc([r for r in records if 20 <= len(r.get("query", "")) < 30]),
        "len_30p": acc([r for r in records if len(r.get("query", "")) >= 30]),
        "has_code": acc([r for r in records if code_pat.search(r.get("query", ""))]),
        "no_code": acc([r for r in records if not code_pat.search(r.get("query", ""))]),
    }
    return {k: v for k, v in out.items() if v}


# ---------------- NER 指标 ----------------

def ner_metrics(records: List[dict]) -> dict:
    """records: [{query, gold_entities: [(text,type)], pred_entities: [(text,type)]}]
    实体级 exact match（span+type）。文本相同即可（弱标注与预测文本对齐）。
    """
    tp = fp = fn = 0
    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in records:
        gold = set(map(tuple, r.get("gold_entities", [])))
        pred = set(map(tuple, r.get("pred_entities", [])))
        for e in gold & pred:
            tp += 1; by_type[e[1]]["tp"] += 1
        for e in pred - gold:
            fp += 1; by_type[e[1]]["fp"] += 1
        for e in gold - pred:
            fn += 1; by_type[e[1]]["fn"] += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    out = {"n": len(records), "entity_P": round(p, 4), "entity_R": round(rc, 4),
           "entity_F1": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0}
    # 标的/时间识别准确率口径：STOCK/STOCK_CODE/TIME 三类合并
    tgt = {"STOCK", "STOCK_CODE", "TIME"}
    tp2 = fp2 = fn2 = 0
    for r in records:
        gold = {e for e in map(tuple, r.get("gold_entities", [])) if e[1] in tgt}
        pred = {e for e in map(tuple, r.get("pred_entities", [])) if e[1] in tgt}
        tp2 += len(gold & pred); fp2 += len(pred - gold); fn2 += len(gold - pred)
    p2 = tp2 / (tp2 + fp2) if tp2 + fp2 else 0.0
    r2 = tp2 / (tp2 + fn2) if tp2 + fn2 else 0.0
    out["target_time_P"] = round(p2, 4)
    out["target_time_R"] = round(r2, 4)
    out["target_time_F1"] = round(2 * p2 * r2 / (p2 + r2), 4) if p2 + r2 else 0.0
    out["by_type"] = {
        t: {"P": round(c["tp"] / (c["tp"] + c["fp"]), 4) if c["tp"] + c["fp"] else 0.0,
            "R": round(c["tp"] / (c["tp"] + c["fn"]), 4) if c["tp"] + c["fn"] else 0.0,
            "n_gold": c["tp"] + c["fn"]}
        for t, c in by_type.items()
    }
    return out


# ---------------- 时延指标 ----------------

def latency_metrics(latencies_ms: List[float]) -> dict:
    import numpy as np
    if not latencies_ms:
        return {}
    a = np.array(latencies_ms)
    return {"P50_ms": round(float(np.percentile(a, 50)), 2),
            "P90_ms": round(float(np.percentile(a, 90)), 2),
            "P99_ms": round(float(np.percentile(a, 99)), 2),
            "mean_ms": round(float(a.mean()), 2)}
