# -*- coding: utf-8 -*-
"""证券主数据词表 + 指标词表构建后的链指效果测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qp.ner.entity_linker import EntityLinker  # noqa: E402

GAZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "entities", "gazetteer.jsonl")


def test_size_grew():
    lk = EntityLinker(GAZ)
    # A股 5000+ + 港股 + 美股 + 种子 + 指标
    assert lk.size() >= 6000, f"gazetteer too small: {lk.size()}"


def test_stock_link_by_abbr():
    lk = EntityLinker(GAZ)
    r = lk.link("浦发银行", etype="STOCK")
    assert r and r["type"] == "STOCK"
    assert "发展银行" in r["name"]      # canonical 为公司全称（SH 官方）


def test_stock_link_by_code():
    lk = EntityLinker(GAZ)
    for code, kw in (("600000", "发展银行"), ("600519", "茅台")):
        r = lk.link(code, etype="STOCK_CODE")
        assert r and r["type"] == "STOCK", f"code {code} not linked"
        assert kw in r["name"]


def test_dual_listing_single_entity():
    """阿里巴巴 HK+US 应链指到同一实体（eid 相同）。"""
    lk = EntityLinker(GAZ)
    r1 = lk.link("阿里巴巴")
    assert r1, "阿里巴巴 not linked"
    eids = {r1["eid"]}
    for c in ("09988", "BABA", "9988"):
        r = lk.link(c)
        if r:
            eids.add(r["eid"])
    assert len(eids) == 1, f"dual listing split across entities: {eids}"
    meta = None
    for e in lk._by_eid.values():
        if e["eid"] == r1["eid"]:
            meta = e.get("meta") or {}
    assert meta and len(meta.get("markets", [])) >= 2, f"expect multi-market: {meta}"


def test_indicator_link():
    lk = EntityLinker(GAZ)
    for span, expect in (("市盈率", "市盈率"), ("PE", "市盈率"), ("市净率", "市净率"),
                         ("净资产收益率", "ROE"), ("北向资金", "北向资金")):
        r = lk.link(span)
        assert r, f"{span} not linked"
        assert expect in r["name"], f"{span} -> {r['name']}, expect {expect}"


def test_tag_doc_code_and_name():
    lk = EntityLinker(GAZ)
    tags = lk.tag_doc("贵州茅台(600519)发布年报，市盈率回落")
    texts = {t["text"] for t in tags}
    types = {t["text"]: t["type"] for t in tags}
    assert any("茅台" in t for t in texts), texts
    assert any("市盈率" in t for t in texts), texts
    assert types.get("市盈率") == "METRIC"  # 指标已迁移为独立 METRIC 类
