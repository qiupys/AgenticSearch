# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qp.pipeline.app import get_pipeline


PIPE = None


def _pipe():
    global PIPE
    if PIPE is None:
        PIPE = get_pipeline()
    return PIPE


def test_pipeline_zh():
    out = _pipe().process("华润新能源 001248 最新公告 新闻 2026年8月")
    assert out.primary_intent() is not None
    assert out.primary_intent().intent_id == "INFO-07"
    # LLM 优先级联：配置可用时走 LLM（tier 2），LLM 不可用回落规则/轻量模型
    assert out.tier in (0, 1, 2)
    if out.tier == 2:
        assert out.intents[0].source == "llm"
    assert out.slots["object_type"] == "个股"
    assert out.entities and any(e.type == "STOCK_CODE" for e in out.entities)
    assert out.recall and out.recall[0].intent_id == "INFO-07"
    assert "total_ms" in out.latency_ms
    assert out.stats_category_16 is not None


def test_pipeline_en():
    out = _pipe().process("What is the latest news on AMD stock?")
    assert out.lang == "en"
    assert out.primary_intent().domain == "INFO"


def test_pipeline_recall_condition():
    out = _pipe().process("贵州茅台 2026年 业绩 点评 分析")
    rc = out.recall[0]
    assert rc.role  # 有召回角色
    assert rc.sentiment  # 有情绪策略
    assert isinstance(rc.time_window_days, (int, type(None)))


def test_pipeline_schema_roundtrip():
    out = _pipe().process("KDJ是什么意思")
    d = out.to_dict()
    assert d["intents"][0]["intent_id"] == "SERVICE-01"
    assert d["slots"]["composite"] == "单意图"


def test_cache():
    from qp.pipeline.app import process_query
    a = process_query("测试缓存 query 2026")
    b = process_query("测试缓存 query 2026")
    assert a.trace_id == b.trace_id  # 命中缓存返回同一对象
