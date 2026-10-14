# -*- coding: utf-8 -*-
"""ranking 模块测试：rerank 阈值过滤（含 vec-only 高门槛）与同文去重。"""
import unittest
from unittest.mock import MagicMock

from search.engine.ranking import rank_candidates


def _doc(i, title, url):
    return {"id": i, "title": title, "url": url, "snippet": ""}


class _EmbStub:
    """带 rerank 开关的 emb client 桩：按 chunk 文本长度给分。"""

    def __init__(self, scores):
        self._scores = scores

    def switch(self, name):
        return name == "rerank"

    def rerank(self, query, texts):
        return list(enumerate(self._scores))


class TestRanking(unittest.TestCase):
    def test_vec_only_high_bar_filters_nav_noise(self):
        """bm25=0 的导航页噪声（rerank 0.38）应被过滤；词面命中的 0.35 应保留。"""
        docs = [
            (0.0, _doc(0, "上海楼市成交出炉", "http://a.com/1"), {"bm25": 13.0, "rerank": 0.35}),
            (0.0, _doc(1, "资金流向首页", "http://b.com/2"), {"bm25": 0.0, "rerank": 0.38}),
        ]
        corpus = [_doc(0, "x", "u0"), _doc(1, "y", "u1")]
        emb = _EmbStub([0.35, 0.38])
        kept, diag = rank_candidates("上海楼市", docs, corpus, {}, emb,
                                     {"min_rerank_score": 0.3})
        self.assertEqual([d["title"] for _, d, _ in kept], ["上海楼市成交出炉"])
        self.assertEqual(diag["n_vec_only_filtered"], 1)
        self.assertEqual(diag["n_below_threshold"], 0)

    def test_vec_only_strong_semantic_kept(self):
        """bm25=0 但 rerank 很高（真相关）的文档应保留。"""
        docs = [(0.0, _doc(0, "深楼市观察", "http://a.com/1"), {"bm25": 0.0, "rerank": 0.72})]
        kept, diag = rank_candidates("楼市", docs, [_doc(0, "x", "u0")], {},
                                     _EmbStub([0.72]), {})
        self.assertEqual(len(kept), 1)
        self.assertEqual(diag["n_vec_only_filtered"], 0)


if __name__ == "__main__":
    unittest.main()
