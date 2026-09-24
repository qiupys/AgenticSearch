# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qp.ner.weak_label import find_entities, to_bio, weak_label_record
from qp.ner.predictor import predict


def test_time_entity():
    ents = find_entities("159934 黄金ETF 历史净值 2026年8月25日")
    times = [e for e in ents if e["type"] == "TIME"]
    assert any("2026" in e["text"] for e in times)


def test_stock_code():
    ents = find_entities("华虹宏力 688347 最新公告")
    assert any(e["type"] == "STOCK_CODE" and e["text"] == "688347" for e in ents)


def test_industry_fund():
    ents = find_entities("半导体板块 创新药ETF 行情")
    types = {e["type"] for e in ents}
    assert "INDUSTRY" in types
    assert "FUND" in types


def test_bio_consistency():
    text = "贵州茅台 2026年8月 股价"
    rec = weak_label_record(text)
    assert len(rec["tags"]) == len(text)
    # BIO 无跨实体 I- 起始
    for i, t in enumerate(rec["tags"]):
        if t.startswith("I-"):
            prev = rec["tags"][i - 1]
            assert prev[2:] == t[2:], f"illegal transition {prev}->{t}"


def test_predictor_rule_fallback():
    ents = predict("北方华创 002371 2026年 最新消息", weights_path=None)
    assert any(e["type"] == "STOCK_CODE" for e in ents)
    assert all(e["source"] == "rule" for e in ents)


def test_slots():
    from qp.slots.extractor import extract_slots
    text = "北方华创 002371 2026年8月 最新消息"
    ents = find_entities(text)
    slots, tinfo = extract_slots(text, ents, intent_ids=["INFO-07"])
    assert slots["object_type"] == "个股"
    assert tinfo["anchor"] in ("最新", "历史时点")
