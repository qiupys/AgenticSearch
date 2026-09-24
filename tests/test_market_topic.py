# -*- coding: utf-8 -*-
"""市场定向（两地上市消歧）+ 板块/概念/指数词表测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qp.pipeline.app import _detect_market                       # noqa: E402
from qp.ner.entity_linker import EntityLinker                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAZ = os.path.join(ROOT, "data", "entities", "gazetteer.jsonl")


def _ent(text, etype="STOCK_CODE"):
    return {"text": text, "type": etype, "start": 0, "end": len(text)}


def test_detect_market_by_words():
    assert _detect_market("阿里巴巴港股走势", []) == "HK"
    assert _detect_market("阿里巴巴美股行情", []) == "US"
    assert _detect_market("A股小米集团点评", []) == "A"
    assert _detect_market("港股 阿里巴巴 财报", []) == "HK"


def test_detect_market_by_code():
    assert _detect_market("600519 贵州茅台", [_ent("600519")]) == "A"
    assert _detect_market("09988 走势", [_ent("09988")]) == "HK"
    assert _detect_market("BAMA ticker", [_ent("BAMA")]) == "US"


def test_detect_market_none():
    assert _detect_market("今天天气如何", []) == ""
    assert _detect_market("美联储加息影响", []) == ""


def test_alibaba_single_entity_multi_market():
    lk = EntityLinker(GAZ)
    r = lk.link("阿里巴巴")
    assert r and r["type"] == "STOCK"
    meta = lk.meta_of(r["eid"])
    assert "HK" in (meta.get("markets") or []) and "US" in (meta.get("markets") or [])
    codes = meta.get("codes") or {}
    assert codes.get("US") == "BABA" and codes.get("HK", "").endswith("9988")


def test_topic_concept_linked():
    lk = EntityLinker(GAZ)
    for span, kw, mkt_ok in (("固态电池", "固态电池", False),
                             ("算力", "算力", False),
                             ("低空经济", "低空经济", False)):
        r = lk.link(span)
        assert r and r["type"] == "INDUSTRY", f"{span} -> {r}"
        assert kw in r["name"]
    # 指数别名
    for span in ("纳指", "道指", "恒指", "沪深三百"):
        r = lk.link(span)
        assert r and r["type"] == "INDEX", f"{span} -> {r}"
    # 行业别名挂到既有条目
    r = lk.link("有色")
    assert r and r["type"] == "INDUSTRY"


def test_indicator_alias_untouched():
    lk = EntityLinker(GAZ)
    r = lk.link("PE")
    assert r and "市盈率" in r["name"]
