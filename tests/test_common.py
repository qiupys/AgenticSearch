# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qp.common.normalize import normalize_query, cache_key, detect_lang
from qp.common.taxonomy import l2_domain, all_l2_ids, product_category_info, l2_to_product_categories
from qp.common.schema import QPOutput


def test_normalize():
    assert normalize_query("  Ａ股　涨跌  ") == "A股 涨跌"
    assert normalize_query("全角１２３") == "全角123"
    assert cache_key("  Hello， World! ") == "hello world"
    assert detect_lang("贵州茅台 股价") == "zh"
    assert detect_lang("What is the price of AAPL") == "en"
    assert detect_lang("A股 vs 美股") == "zh"


def test_taxonomy():
    ids = all_l2_ids()
    assert len(ids) == 42, f"expect 42 L2 intents, got {len(ids)}"
    assert l2_domain("INFO-07") == "INFO"
    assert l2_domain("BAD-99") is None
    info = product_category_info("查公告")
    assert info and info["primary_l2"] == "INFO-07"
    cats = l2_to_product_categories("INFO-07")
    assert "查公告" in cats


def test_schema():
    out = QPOutput(query="q", normalized="q")
    d = out.to_dict()
    assert d["version"] == "1.0" and d["tier"] == -1
    assert out.primary_intent() is None
