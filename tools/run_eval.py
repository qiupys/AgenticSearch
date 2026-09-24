# -*- coding: utf-8 -*-
"""端到端评测 runner：pipeline 全量跑评测集 → 指标 → 报告（markdown + json）。

评测集：
  1. data/eval/synth_gold.jsonl   —— 42 类覆盖的合成金标（中英双语）
  2. data/eval/baseline_gold.jsonl —— 160 条产品 16 类真实评测金标（来自 20260821 基线报告）

报告输出：data/eval/report/evaluation_report.md / .json
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation.metrics import intent_metrics, slice_metrics, ner_metrics, latency_metrics
from qp.pipeline.app import get_pipeline


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_synth(pipeline, records):
    out = []
    ner_records = []
    for rec in records:
        t0 = time.perf_counter()
        r = pipeline.process(rec["query"])
        ms = (time.perf_counter() - t0) * 1000
        top3 = [i.intent_id for i in r.intents[:3]]
        out.append({
            "query": rec["query"], "lang": rec.get("lang", r.lang),
            "gold_l2": rec["gold_l2"],
            "pred_l2": r.intents[0].intent_id if r.intents else "",
            "pred_l1": r.intents[0].domain if r.intents else "",
            "pred_l2_top3": top3,
            "tier": r.tier, "escalated": r.escalated,
            "latency_ms": ms,
        })
        # NER 金标：模板填充的 stock/time/index 即实体真值（仅在文中出现时）
        gold_ents = []
        if rec.get("stock_used") and rec["stock_used"] in rec["query"]:
            gold_ents.append([rec["stock_used"], "STOCK"])
        if rec.get("time_used") and rec["time_used"] in rec["query"]:
            gold_ents.append([rec["time_used"], "TIME"])
        elif rec.get("time_used_en") and rec["time_used_en"] in rec["query"]:
            gold_ents.append([rec["time_used_en"], "TIME"])
        if rec.get("index_used") and rec["index_used"] in rec["query"]:
            gold_ents.append([rec["index_used"], "INDEX"])
        if gold_ents:
            ner_records.append({
                "query": rec["query"],
                "gold_entities": gold_ents,
                "pred_entities": [[e.text, e.type] for e in r.entities
                                  if e.type in ("STOCK", "STOCK_CODE", "TIME", "INDEX", "FUND")],
            })
    return out, ner_records


def run_baseline(pipeline, records):
    out = []
    for rec in records:
        r = pipeline.process(rec["query"])
        top3 = [i.intent_id for i in r.intents[:3]]
        out.append({
            "query": rec["query"], "lang": rec.get("lang", r.lang),
            "gold_product_category": rec["gold_product_category"],
            "pred_l2": r.intents[0].intent_id if r.intents else "",
            "pred_l1": r.intents[0].domain if r.intents else "",
            "pred_l2_top3": top3,
            "tier": r.tier, "escalated": r.escalated,
            "market": rec.get("market"),
        })
    return out


def fmt_pct(x):
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "-"


def render_report(synth_m, base_m, synth_slice, ner_m, lat, targets) -> str:
    lines = []
    a = lines.append
    a("# QP Pipeline 离线评测报告")
    a("")
    a(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    a("- 评测对象：规则快通道(Tier0) + 轻量意图模型(Tier1) + 规则NER兜底 + 槽位 + 召回翻译 全链路")
    a("- 说明：0.6B 判别头权重未就位，Tier1 为轻量替代模型；NER 为规则兜底（模型服务器训练中）")
    a("")
    a("## 1. 合成金标集（42 类覆盖，中英双语）")
    a("")
    a(f"- 样本数：{synth_m.get('n')}")
    a(f"- L1 top1 准确率：**{fmt_pct(synth_m.get('L1_top1_acc'))}**")
    a(f"- L2 top1 准确率：**{fmt_pct(synth_m.get('L2_top1_acc'))}**")
    a(f"- L2 top3 准确率：{fmt_pct(synth_m.get('L2_top3_acc'))}")
    a(f"- micro P/R/F1：{fmt_pct(synth_m.get('micro_P'))} / {fmt_pct(synth_m.get('micro_R'))} / {fmt_pct(synth_m.get('micro_F1'))}")
    a(f"- macro F1：{fmt_pct(synth_m.get('macro_F1'))}")
    a(f"- Tier 流量分布：{synth_m.get('tier_dist')}  升级率：{fmt_pct(synth_m.get('escalate_rate'))}")
    a("")
    if synth_m.get("top_confusions"):
        a("### Top 混淆对")
        a("")
        a("| gold | pred | count |")
        a("|---|---|---|")
        for c in synth_m["top_confusions"]:
            a(f"| {c['gold']} | {c['pred']} | {c['count']} |")
        a("")
    if synth_slice:
        a("### 分维度切片（L2 acc）")
        a("")
        a("| 切片 | n | L2 acc |")
        a("|---|---|---|")
        for k, v in synth_slice.items():
            a(f"| {k} | {v['n']} | {fmt_pct(v['L2_acc'])} |")
        a("")
    a("## 2. 基线金标集（160 条真实评测 query，产品 16 类口径）")
    a("")
    if base_m:
        a(f"- 样本数：{base_m.get('n')}")
        a(f"- 统计层面准确率（16类口径，目标 {fmt_pct(targets.get('stats_level_acc'))}）：**{fmt_pct(base_m.get('stats_level_acc'))}**")
        a(f"- 排序层面准确率（L1 口径，目标 {fmt_pct(targets.get('rank_level_acc'))}）：**{fmt_pct(base_m.get('rank_level_acc'))}**")
        a(f"- Tier 流量分布：{base_m.get('tier_dist')}  升级率：{fmt_pct(base_m.get('escalate_rate'))}")
        a("")
        zh = [r for r in base_records_ if r.get("lang") == "zh"]
        en = [r for r in base_records_ if r.get("lang") == "en"]
        if zh:
            a(f"- 国内集（zh, n={len(zh)}）：统计口径 {fmt_pct(_prod_acc(zh))} / 排序口径 {fmt_pct(_rank_acc(zh))}")
        if en:
            a(f"- 海外集（en, n={len(en)}）：统计口径 {fmt_pct(_prod_acc(en))} / 排序口径 {fmt_pct(_rank_acc(en))}")
        a("")
    a("## 3. NER 实体识别（规则兜底，合成集真值）")
    a("")
    if ner_m:
        a(f"- 实体级 P / R / F1：{fmt_pct(ner_m.get('entity_P'))} / {fmt_pct(ner_m.get('entity_R'))} / {fmt_pct(ner_m.get('entity_F1'))}")
        a(f"- 标的+时间口径 P / R / F1（目标 {fmt_pct(targets.get('entity_time_acc'))}）："
          f"{fmt_pct(ner_m.get('target_time_P'))} / {fmt_pct(ner_m.get('target_time_R'))} / {fmt_pct(ner_m.get('target_time_F1'))}")
        if ner_m.get("by_type"):
            a("")
            a("| 类型 | P | R | n_gold |")
            a("|---|---|---|---|")
            for t, v in sorted(ner_m["by_type"].items()):
                a(f"| {t} | {fmt_pct(v['P'])} | {fmt_pct(v['R'])} | {v['n_gold']} |")
    a("")
    a("## 4. 时延")
    a("")
    for k, v in lat.items():
        a(f"- {k}: {v} ms")
    a("")
    a("## 5. 结论与差距")
    a("")
    a("- 合成集反映 42 类体系的覆盖性与规则/轻量模型基线水平；")
    a("- 基线 160 条金标反映真实分布上的当前水平与业务目标差距；")
    a("- NER 规则版为兜底基线，99% 目标由『词典+BiLSTM-CRF模型+实体链接』级联达成（服务器训练后回填）；")
    a("- 0.6B 判别头权重就位后替换 Tier1 轻量模型，预期显著抬升 L2 指标（训练方案见 docs/03）。")
    return "\n".join(lines)


base_records_ = []


def _prod_acc(rs):
    rs = [r for r in rs if r.get("gold_product_category")]
    if not rs:
        return None
    return sum(1 for r in rs if r.get("stats_level_hit")) / len(rs)


def _rank_acc(rs):
    rs = [r for r in rs if r.get("gold_product_category")]
    if not rs:
        return None
    return sum(1 for r in rs if r.get("rank_level_hit")) / len(rs)


def main():
    pipeline = get_pipeline()
    synth = load_jsonl(os.path.join(ROOT, "data", "eval", "synth_gold.jsonl"))
    baseline = load_jsonl(os.path.join(ROOT, "data", "eval", "baseline_gold.jsonl"))
    print(f"eval sets: synth={len(synth)} baseline={len(baseline)}")

    synth_out, ner_recs = run_synth(pipeline, synth) if synth else ([], [])
    base_out = run_baseline(pipeline, baseline) if baseline else []
    base_records_[:] = base_out

    synth_m = intent_metrics(synth_out)
    synth_slice = slice_metrics(synth_out)
    base_m = intent_metrics(base_out) if base_out else None
    ner_m = ner_metrics(ner_recs) if ner_recs else None
    lat = latency_metrics([r["latency_ms"] for r in synth_out])

    import yaml
    with open(os.path.join(ROOT, "configs", "service.yaml"), encoding="utf-8") as f:
        targets = yaml.safe_load(f)["eval_targets"]

    report_md = render_report(synth_m, base_m, synth_slice, ner_m, lat, targets)
    out_dir = os.path.join(ROOT, "data", "eval", "report")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(os.path.join(out_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump({"synth": synth_m, "synth_slice": synth_slice, "baseline": base_m,
                   "ner": ner_m, "latency": lat, "targets": targets}, f, ensure_ascii=False, indent=2)
    print(report_md)
    print(f"\nreport -> {out_dir}")


if __name__ == "__main__":
    main()
