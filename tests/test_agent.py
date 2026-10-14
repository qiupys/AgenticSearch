# -*- coding: utf-8 -*-
"""Agent 模块单测：planner 解析 / 引用提取 / JSON 容错 / 工具合并（不触网）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.agent.loop import (_parse_planner, _citations, _norm_url,
                               _history_text, _docs_for_synth, _qp_block,
                               _obs_from_results, _extractive_answer,
                               _web_authority)
from qp.common.llm_client import LLMClient
from types import SimpleNamespace as NS


# ---------------- planner 决策解析 ----------------

def test_parse_planner_valid():
    p = _parse_planner({"thought": "时效性强", "action": "search",
                        "args": {"query": "北方华创 订单"}})
    assert p == {"thought": "时效性强", "action": "search",
                 "args": {"query": "北方华创 订单"}}


def test_parse_planner_answer_no_args():
    p = _parse_planner({"thought": "结果足够", "action": "answer", "args": "bad"})
    assert p["action"] == "answer" and p["args"] == {}


def test_parse_planner_rejects_unknown_action():
    assert _parse_planner({"action": "delete_all"}) is None
    assert _parse_planner({"action": ""}) is None
    assert _parse_planner("not a dict") is None
    assert _parse_planner(None) is None


# ---------------- 引用提取 ----------------

def _docs():
    return [
        {"title": "A", "url": "http://a.com/1", "date": "2026-10-01"},
        {"title": "B", "url": "http://b.com/2", "date": None},
        {"title": "C", "url": "", "date": "2026-09-30"},
    ]


def test_citations_extracts_used_numbers():
    cits = _citations("订单增长 [1]，风险提示 [3]，越界引用 [9]", _docs())
    assert [c["n"] for c in cits] == [1, 3]
    assert cits[0]["title"] == "A" and cits[0]["url"] == "http://a.com/1"


def test_citations_empty_when_no_refs():
    assert _citations("无引用答案", _docs()) == []
    assert _citations("", _docs()) == []
    assert _citations(None, _docs()) == []


# ---------------- URL 归一化（sub_queries 去重基础） ----------------

def test_norm_url():
    assert _norm_url("https://www.a.com/x?utm=1") == _norm_url("http://a.com/x")
    assert _norm_url("") == ""


# ---------------- 历史文本 / synth 文档构造 ----------------

def test_history_text_empty():
    assert "第 1 轮" in _history_text([])


def test_history_text_rounds():
    steps = [{"round": 1, "action": "content_search", "args": {"query": "q"},
              "observation": {"n_new_docs": 3}},
             {"round": 2, "action": "web_search", "args": {},
              "observation": {"n_new_docs": 0, "note": "无结果"}}]
    txt = _history_text(steps)
    assert "第1轮" in txt and "第2轮" in txt
    assert "无结果" in txt


def test_docs_for_synth_numbered():
    docs = [{"title": f"t{i}", "url": "u", "date": "2026-10-01",
             "snippet": f"s{i}"} for i in range(3)]
    txt = _docs_for_synth(docs)
    assert "[1] t0" in txt and "[3] t2" in txt


# ---------------- JSON 容错（截断修复，CodeBuddy 思考模型产物） ----------------

def test_extract_json_plain():
    assert LLMClient._extract_json('{"a": 1}') == {"a": 1}
    assert LLMClient._extract_json('前置 {"a": [1, 2]} 后置') == {"a": [1, 2]}
    assert LLMClient._extract_json('```json\n{"a": "b"}\n```') == {"a": "b"}
    assert LLMClient._extract_json("") is None
    assert LLMClient._extract_json(None) is None


def test_extract_json_truncated_repair():
    # content 被截断（finish=length）：补引号/括号后可解析
    assert LLMClient._extract_json('{"answer": "订单增长 [1][2') == \
        {"answer": "订单增长 [1][2"}
    assert LLMClient._extract_json('{"next": "answer", "args": {"q": "北方华创') == \
        {"next": "answer", "args": {"q": "北方华创"}}
    # 截断在转义序列中间：仍应安全返回（不抛异常）
    r = LLMClient._extract_json('{"answer": "abc\\u4e')
    assert r is None or isinstance(r, dict)


# ---------------- 工具结果观察（不触网部分） ----------------

def test_obs_dedup_by_url():
    seen = set()
    docs = [{"title": "t", "url": "http://x.com/a", "snippet": "s",
             "date": "2026-10-01", "content_type": "news", "score": 1}]
    out = {"tool": "content_search", "queries": ["q"],
           "results": [(1.0, docs[0], {}, {}), (0.9, dict(docs[0], url="http://x.com/a?u=1"), {}, {})]}
    obs, collected, _ = _obs_from_results(out, seen)
    assert obs["n_new_docs"] == 1          # 同 URL（归一化后）去重
    assert len(collected) == 1
    assert "x.com/a" in seen               # seen 存归一化后的 URL


# ---------------- QP 理解注入（第 0 步 → Planner） ----------------

def test_qp_block_summary():
    qp = NS(intents=[NS(intent_id="INFO-07")],
            entities=[NS(text="北方华创", type="COMPANY")],
            recall=[NS(channel="news", role="主召回")],
            time=NS(anchor="最新", bucket="no_limit"))
    txt = _qp_block(qp)
    assert "INFO-07" in txt and "news(主召回)" in txt
    assert "北方华创/COMPANY" in txt and "最新/no_limit" in txt


def test_qp_block_tolerates_none_fields():
    txt = _qp_block(NS(intents=[], entities=[], recall=[], time=None))
    assert "意图 无" in txt and "时间 无" in txt


# ---------------- 检索诊断信号（observation 增强） ----------------

def test_obs_retrieval_signals():
    out = {"results": [], "diag": {"time_starved": True,
                                   "core_entity_coverage": 0,
                                   "channels": {"news": {"intent": "INFO-07"}}}}
    obs, _, _ = _obs_from_results(out, set())
    assert obs["retrieval_signals"]["time_starved"] is True
    assert obs["retrieval_signals"]["core_entity_coverage"] == 0
    assert obs["retrieval_signals"]["channels_hit"] == ["news"]
    assert "时效空窗" in obs["note"] and "核心实体零命中" in obs["note"]


def test_obs_no_signals_when_diag_clean():
    doc = {"title": "t", "url": "http://x.com/a", "snippet": "s",
           "date": None, "content_type": "news"}
    out = {"results": [(1.0, doc, {}, {})],
           "diag": {"time_starved": False}}
    obs, _, _ = _obs_from_results(out, set())
    assert "retrieval_signals" not in obs
    assert "note" not in obs


# ---------------- 摘录兜底（LLM 综合不可用时） ----------------

def test_extractive_answer_numbered():
    docs = [{"title": f"t{i}", "url": "u", "date": "2026-10-01",
             "snippet": f"要点{i}内容"} for i in range(2)]
    ans = _extractive_answer(docs)
    assert ans is not None and "[1]" in ans and "[2]" in ans and "要点0" in ans
    # 编号可被引用提取器消费
    assert [c["n"] for c in _citations(ans, docs)] == [1, 2]


def test_extractive_answer_empty():
    assert _extractive_answer([]) is None
    assert _extractive_answer([{"title": "t", "snippet": "", "url": "u"}]) is None


# ---------------- web 权威度分级 ----------------

def test_web_authority_tiers():
    assert _web_authority("http://www.csrc.gov.cn/abc") == 3       # 监管
    assert _web_authority("https://finance.sina.com.cn/x") == 2    # 专业财经
    assert _web_authority("https://eastmoney.com/a/1.html") == 2   # 行情站点
    assert _web_authority("https://tieba.baidu.com/p/123") == 0    # 论坛
    assert _web_authority("https://www.zhihu.com/question/1") == 0 # UGC
    assert _web_authority("https://unknown-site.example.com/x") == 1  # 默认综合
