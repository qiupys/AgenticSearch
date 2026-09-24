# -*- coding: utf-8 -*-
"""存量语料日期回填：对缺 date 的 docs 抓正文提取（标题/URL 已尽力仍缺失的）。

用法：python scripts/backfill_dates.py [--proxy http://127.0.0.1:10808]
"""
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import requests

from tools.build_content_library import (
    CORPUS_PATH, extract_date, extract_date_from_html, extract_date_from_url, load_existing,
)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"


def backfill(proxy: str | None = None, timeout: int = 8, limit: int = 0):
    done, corpus, _ = load_existing()
    missing = [d for d in corpus if not d.get("date")]
    if limit:
        missing = missing[:limit]
    print(f"missing dates: {len(missing)}/{len(corpus)}")
    ok = 0
    for i, d in enumerate(missing, 1):
        # 1) URL 路径
        dt = extract_date_from_url(d.get("url", ""))
        # 2) 正文抓取
        if not dt:
            try:
                r = requests.get(d["url"], timeout=timeout,
                                 proxies={"http": proxy, "https": proxy} if proxy else None,
                                 headers={"User-Agent": _UA,
                                          "Accept-Language": "zh-CN,zh;q=0.9"})
                dt = extract_date_from_html(r.text)
                # 顺手补 body（供 chunk 召回）
                if r.ok and not d.get("body"):
                    body = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>",
                                  " ", r.text[:60000])
                    d["body"] = re.sub(r"\s+", " ", body).strip()[:6000] or None
                    if d["body"] is None:
                        d.pop("body", None)
            except Exception:
                dt = None
                time.sleep(0.2)
        if dt:
            d["date"] = dt
            ok += 1
        if i % 20 == 0:
            print(f"  ... {i}/{len(missing)} (+{ok})")
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    total = sum(1 for d in corpus if d.get("date"))
    print(f"backfilled: {ok}, coverage now: {total}/{len(corpus)} ({total * 100 // max(len(corpus), 1)}%)")


if __name__ == "__main__":
    proxy = None
    if "--proxy" in sys.argv:
        proxy = sys.argv[sys.argv.index("--proxy") + 1]
    backfill(proxy=proxy)
