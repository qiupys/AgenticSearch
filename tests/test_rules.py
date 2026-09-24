# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qp.rule_engine.fast_path import RuleEngine


ENGINE = RuleEngine()


def test_info_07():
    r = ENGINE.classify("华润新能源 001248 最新公告 新闻 2026年8月")
    assert r["hit"] and r["intents"][0]["intent_id"] == "INFO-07"


def test_info_04():
    r = ENGINE.classify("鼎龙股份 最新财报 营收 毛利率 2026")
    assert r["hit"] and r["intents"][0]["intent_id"] == "INFO-04"


def test_analyze_06():
    r = ENGINE.classify("今天有色金属大跌原因")
    assert r["hit"] and r["intents"][0]["intent_id"] == "ANALYZE-06"


def test_select_01():
    r = ENGINE.classify("2026年风电行业龙头股名单")
    assert r["hit"] and r["intents"][0]["intent_id"].startswith("SELECT")


def test_predict_02():
    r = ENGINE.classify("说说九安医疗后面还有机会涨吗")
    assert r["hit"] and r["intents"][0]["intent_id"] == "PREDICT-02"


def test_service_01():
    r = ENGINE.classify("什么叫顶格申购")
    assert r["hit"] and r["intents"][0]["intent_id"] == "SERVICE-01"


def test_english():
    r = ENGINE.classify("What is the latest news on AMD stock?")
    assert r["hit"] and r["intents"][0]["intent_id"] == "INFO-07"
    r = ENGINE.classify("why did Tesla stock drop sharply after earnings")
    assert r["hit"] and r["intents"][0]["intent_id"] == "ANALYZE-06"


def test_no_hit():
    r = ENGINE.classify("zzzz qqqq")
    assert not r["hit"]
