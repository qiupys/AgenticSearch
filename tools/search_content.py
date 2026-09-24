# -*- coding: utf-8 -*-
"""兼容 shim：检索实现已迁移至 content_search 包。

模块划分（详见 docs/modules/）：
  content_search.corpus.py   语料/索引/chunk 基础设施
  content_search.recall.py   召回模块（多路召回 + RRF + 召回侧过滤）
  content_search.ranking.py  排序模块（rerank + 阈值 + 去重）

本文件仅为兼容层：serving / inline_crawl / 历史脚本继续
`from tools.search_content import search, load_all` 不需要改动。
新代码请直接 import content_search。
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from search.engine import (CONTENT_DIR, chunk_embedding_scores, load_all,  # noqa: F401
                            load_or_build_chunks, search, split_chunks)


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "鼎龙股份 最新公告 新闻 2026年8月"
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
    qp, intents, slots, results, diag = search(query, top)

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print(f"query: {query}")
    print(f"intents: {intents}  tier={qp.tier}")
    print(f"slots: object_type={slots['object_type']} time_anchor={slots['time_anchor']}")
    if qp.recall:
        rc = qp.recall[0]
        print(f"recall: role={rc.role} pref={rc.content_type_pref} "
              f"window={rc.time_window_days}d channel={rc.time_channel}")
    print("-" * 70)
    for s, d, sub, bc in results:
        print(f"[{s:.4f}] rerank={sub.get('rerank')} ({d['content_type']}) {d['title'][:60]}")
        if bc:
            print(f"          chunk: {bc['text'][:80]}")


if __name__ == "__main__":
    main()
