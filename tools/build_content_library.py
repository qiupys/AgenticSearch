# -*- coding: utf-8 -*-
"""构建本地内容库（Q3 加强版）：多后端轮换爬虫 -> 治理 -> BM25+向量索引。

相对初版的增强（Q1：更强爬虫）：
  - 多后端轮换：duckduckgo / bing / brave / google / mojeek 自动降级切换
  - 指数退避：触发风控后 sleep 翻倍，连续失败则冷却切换后端
  - 断点续爬：--resume 读取已有 corpus.jsonl，跳过已完成 query、合并 URL 去重
  - 查询降级：长尾 query 失败后自动简化（去时间词/去低频修饰词）重试
  - 随机化节流：每请求随机 sleep，降低 bot 特征
  - 可选正文抓取：--fetch-body 拉取 top 结果正文（requests + 代理，超时跳过）

用法：
  python scripts/build_content_library.py --limit 100 --resume
  python scripts/build_content_library.py --limit 50 --fetch-body --max-body 60
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qp.ner.weak_label import _lexicons

CONTENT_DIR = os.path.join(ROOT, "data", "content")
CORPUS_PATH = os.path.join(CONTENT_DIR, "corpus.jsonl")

BACKENDS = ["duckduckgo", "bing", "brave", "google", "mojeek"]

# 内容类型分类：域名/站点 -> canonical 类型（configs/recall_linkage.json 的 13 类枚举）
DOMAIN_TYPE_RULES = [
    (r"video|douyin|bilibili|youtube|kuaishou", "video_short"),
    (r"zhihu|wenshu|guba|xueqiu|tieba|reddit", "qa"),
    (r"baike|wiki|encyclopedia", "encyclopedia"),
    (r"eastmoney|10jqka|sina|finance\.qq|cls\.cn|stcn|yicai|21jingji|caixin|reuters|bloomberg", "news"),
    (r"research|report|hibor|dfcfw", "report"),
    (r"gov\.cn|csrc|sse|szse|hkex|sec\.gov", "report"),
]
SNIPPET_TYPE_RULES = [
    (r"研报|研究报告|券商|评级|target price", "report"),
    (r"教程|指南|如何|怎么|tutorial|guide", "tutorial"),
    (r"视频|video", "video_short"),
]
# 公告频道：标题级判定（snippet 里出现「公告」的资讯页不算公告）
TITLE_ANNOUNCE_RULES = [
    r"公告|通告|通函|澄清|须予披露|披露易|announcement|8-k|filing|prospectus",
]
DEFAULT_TYPE = "article"


def classify_content_type(title: str, snippet: str, url: str) -> str:
    # 公告频道优先（标题级）：避免资讯标题带「公告」二字的误判需看标题主体
    if re.search("|".join(TITLE_ANNOUNCE_RULES), title, re.I):
        return "announcement"
    text = f"{title} {snippet}"
    for pat, t in DOMAIN_TYPE_RULES:
        if re.search(pat, url, re.I):
            return t
    for pat, t in SNIPPET_TYPE_RULES:
        if re.search(pat, text, re.I):
            return t
    return DEFAULT_TYPE


def tag_industry(text: str) -> list:
    lx = _lexicons()
    hits = [w for w in sorted(set(lx["industries"]), key=len, reverse=True) if w in text]
    return hits[:3]


def tag_sentiment(text: str) -> str:
    if re.search(r"利好|上涨|大涨|新高|rally|surge|beats?", text, re.I):
        return "positive"
    if re.search(r"利空|下跌|大跌|新低|暴跌|plunge|drop|misses?", text, re.I):
        return "negative"
    return "neutral"


# ----------------------------------------------------------------------
# 查询抽样（按意图域分层，优先高置信）

def sample_queries(limit: int) -> list:
    by_domain: dict = defaultdict(list)
    with open(os.path.join(ROOT, "data", "silver", "intent_silver.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_domain[r["silver_l1"]].append(r)
    domains = sorted(by_domain, key=lambda d: -len(by_domain[d]))
    quota = {d: max(2, int(len(by_domain[d]) / 100438 * limit)) for d in domains}
    picked = []
    for d in domains:
        rows = sorted(by_domain[d], key=lambda r: -int(r.get("rule_score", 0)))
        picked.extend(rows[:quota[d]])
    return [(r["query"], r["silver_l2"]) for r in picked[:limit]]


_EN_STOP = {
    "what", "is", "are", "was", "were", "the", "a", "an", "on", "of", "to",
    "for", "in", "at", "by", "with", "about", "how", "does", "do", "did",
    "can", "will", "would", "should", "it", "its", "this", "that", "and", "or",
}


def simplify_query(q: str) -> list:
    """查询降级候选：去时间词 -> 去英文停用词 -> 只留标的/主体 + 核心情词。"""
    cands = []
    q1 = re.sub(r"20\d{2}年?(\d{1,2}月)?(\d{1,2}日)?|最新|近期|最近|今天|今日", " ", q)
    q1 = re.sub(r"\s+", " ", q1).strip()
    if q1 and q1 != q:
        cands.append(q1)
    # 英文停用词过滤：避免 "what is the" 这类退化查询
    content = [t for t in q1.split() if t.lower() not in _EN_STOP]
    if content and " ".join(content) != q1:
        cands.append(" ".join(content))
    if len(content) > 3:
        cands.append(" ".join(content[:3]))
    return [c for c in cands if c]


# ----------------------------------------------------------------------
# 爬虫核心：多后端 + 退避 + 续爬

class ResilientSearcher:
    def __init__(self, proxy: str, max_results: int,
                 sleep_range: tuple = (1.5, 3.5), max_retry: int = 2,
                 max_backoff_s: int = 60, timeout: int = 20,
                 brave_key: str = "", deadline: float | None = None):
        self.proxy = proxy                     # 一律走代理；None 仅限显式 --no-proxy
        self.max_results = max_results
        self.sleep_range = sleep_range
        self.max_retry = max_retry
        self.max_backoff_s = max_backoff_s
        self.timeout = timeout
        self.brave_key = brave_key             # 配置了才启用 Brave 官方 API
        self.backend_idx = 0
        self.cooldown_until = 0.0
        self.stats: Counter = Counter()
        self.deadline = deadline               # 绝对时间戳：预算超时即不再发起引擎调用

    def _expired(self) -> bool:
        return self.deadline is not None and time.time() > self.deadline

    def _sleep(self):
        time.sleep(random.uniform(*self.sleep_range))

    def _backoff(self, hard_fail: bool):
        self.stats["backoff"] += 1
        wait = min(self.max_backoff_s, 4 * (2 ** min(self.stats["backoff"], 4)))
        if hard_fail:
            self.backend_idx = (self.backend_idx + 1) % len(BACKENDS)
            wait = max(wait, min(20, self.max_backoff_s))
        print(f"  [backoff] sleeping {wait}s, switch backend -> {BACKENDS[self.backend_idx]}")
        time.sleep(wait)

    def search_one(self, q: str, region: str, timelimit: str | None = None) -> list:
        from ddgs import DDGS
        for attempt in range(self.max_retry + 1):
            if self._expired():
                return []
            if time.time() < self.cooldown_until:
                time.sleep(self.cooldown_until - time.time())
            backend = BACKENDS[self.backend_idx]
            try:
                with DDGS(proxy=self.proxy or None, timeout=self.timeout) as dd:
                    kw = {"region": region, "max_results": self.max_results,
                          "backend": backend}
                    if timelimit:
                        kw["timelimit"] = timelimit   # d/w/m，部分后端支持
                    results = list(dd.text(q, **kw))
                self.stats["ok"] += 1
                self.stats["backend_" + backend] += 1
                self._sleep()
                return results or []
            except Exception as e:
                msg = str(e)
                self.stats["err"] += 1
                hard = ("load" in msg or "202" in msg or "blocked" in msg.lower()
                        or attempt == self.max_retry)
                if hard and attempt >= 1:
                    self._backoff(hard_fail=True)
                elif attempt < self.max_retry:
                    self._sleep()
                else:
                    self._backoff(hard_fail=False)
        # ddgs 全部后端失败 -> 独立引擎兜底（不经 ddgs 库）
        return self._direct_fallback(q)

    def _direct_fallback(self, q: str) -> list:
        """独立引擎链：ddg_html（首选，与 ddgs API 通道独立）-> bing_html -> brave_api。"""
        for fname in ("ddg_html_search", "bing_html_search"):
            if self._expired():
                return []
            try:
                import importlib
                engines = importlib.import_module("scripts.engines")
                r = getattr(engines, fname)(q, proxy=self.proxy,
                                            max_results=self.max_results,
                                            timeout=self.timeout)
                if r:
                    self.stats["ok"] += 1
                    self.stats["engine_" + fname.replace("_search", "")] += 1
                    print(f"  [direct] {fname} ok, {len(r)} results")
                    return r
            except Exception as e:
                self.stats["err_direct"] += 1
                print(f"  [direct] {fname} fail: {str(e)[:60]}")
        if self.brave_key:
            try:
                from tools.engines import brave_api_search
                r = brave_api_search(q, proxy=self.proxy, max_results=self.max_results,
                                     timeout=self.timeout, api_key=self.brave_key)
                if r:
                    self.stats["ok"] += 1
                    self.stats["engine_brave_api"] += 1
                    return r
            except Exception:
                self.stats["err_direct"] += 1
        return []

    def fetch_body(self, url: str) -> str:
        """可选正文抓取：requests + 代理，超时/失败跳过。"""
        try:
            import requests
            proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
            resp = requests.get(url, timeout=10, proxies=proxies,
                                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            html = resp.text
            html = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html)
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:2000]
        except Exception:
            self.stats["body_fail"] += 1
            return ""


# ----------------------------------------------------------------------
# 治理与索引

def load_existing() -> tuple:
    """断点续爬：返回 (done_queries, existing_corpus, seen_urls)。"""
    done, corpus, urls = set(), [], set()
    if os.path.exists(CORPUS_PATH) and os.path.getsize(CORPUS_PATH) > 0:
        with open(CORPUS_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                corpus.append(d)
                urls.add(d["url"].rstrip("/"))
                done.add(d["query"])
    return done, corpus, urls


# 发布日期抽取：标题/摘要里的显式日期 -> date 字段（时效过滤与新鲜度加权用）
_DOC_DATE = re.compile(r"(20\d{2})[-年/.](\d{1,2})(?:[-月/.](\d{1,2}))?")
# 网页 meta 日期（publishdate / datePublished / article:published_time 等）
_META_DATE = re.compile(
    r'"(?:publishdate|pubdate|publish_?time|datepublished|published_time|pubtime|publishdate)"'
    r'[^>"\']{0,30}?["\']?(\d{4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?)', re.I)


def _canon_date(s: str) -> str | None:
    m = re.match(r"(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?", s)
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    if not (1 <= mo <= 12):
        return None
    d = m.group(3)
    return f"{y}-{mo:02d}-{int(d):02d}" if d else f"{y}-{mo:02d}"


def _sane_date(s: str | None) -> str | None:
    """过滤来自未来的日期：页面上的到期日/报告期/JS 时间戳等不是发布时间。"""
    if not s:
        return None
    import datetime
    today = datetime.date.today().isoformat()
    # 月精度(YYYY-MM)与当月第一天比较：当月 1 日 <= 今天 才可信
    probe = s + "-01" if len(s) == 7 else s[:10]
    return s if probe <= today else None


def extract_date(text: str) -> str | None:
    m = _DOC_DATE.search(text or "")
    return _sane_date(_canon_date(m.group(0))) if m else None


_URL_DATE = re.compile(r"/(\d{4})[-/.]?(\d{2})[-/.]?(\d{2})/[a-z]*")


def extract_date_from_url(url: str) -> str | None:
    """新闻站 URL 普遍嵌日期（/2026-09-01/、/20260901/），低成本兜底。"""
    if not url:
        return None
    m = _URL_DATE.search(url)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2020 <= y <= 2035 and 1 <= mo <= 12 and 1 <= d <= 31:
            return _sane_date(f"{y}-{mo:02d}-{d:02d}")
    return None


def extract_date_from_html(html: str) -> str | None:
    """正文 HTML 抽日期：优先 meta 字段；退化全文扫描可见文本取最新可信日期
    （列表页每行带日期，最新日期 ≈ 页面/内容时效；未来日期剔除）。"""
    if not html:
        return None
    m = _META_DATE.search(html[:20000])
    if m:
        return _sane_date(_canon_date(m.group(1)))
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html[:60000])
    text = re.sub(r"<[^>]+>", " ", text)
    dates = [d for d in (_sane_date(_canon_date(x.group(0)))
                         for x in _DOC_DATE.finditer(text)) if d]
    return max(dates) if dates else None


def govern_row(r: dict, idx_num: int) -> dict:
    from qp.common.normalize import clean_title, fix_mojibake, is_junk_text
    r = {**r, "title": clean_title(r.get("title", ""))}  # 剥离引擎拼进标题的正文
    if r.get("body"):
        r["body"] = fix_mojibake(r["body"])
        if is_junk_text(r["body"]):   # WAF 挑战页/JS 脚本不入语料正文
            r["body"] = ""
    text = f"{r['title']} {r['snippet']} {r.get('body', '')}"
    # doc 侧实体链指：正文扫描词表，产出 linked_entities（与 query 侧同词表对齐）
    try:
        from qp.ner.entity_linker import default_linker
        linked = default_linker().tag_doc(text)
    except Exception:
        linked = []
    return {
        **r,
        "id": f"doc{idx_num:05d}",
        "content_type": classify_content_type(r["title"], r["snippet"], r["url"]),
        "linked_entities": linked,
        "date": extract_date(f"{r['title']} {r['snippet']}")
                or extract_date_from_url(r.get("url", "")) or r.get("date"),
        "industries": tag_industry(text),
        "sentiment": tag_sentiment(text),
        "lang": "en" if not re.search(r"[\u4e00-\u9fff]", text) else "zh",
    }


def build_index(corpus: list) -> str:
    import jieba
    from rank_bm25 import BM25Okapi

    dict_path = os.path.join(ROOT, "data", "dict", "jieba_user_dict.txt")
    if os.path.exists(dict_path):
        jieba.load_userdict(dict_path)
    jieba.initialize()

    def tok(text):
        return [t for t in jieba.lcut(text.lower()) if t.strip()]

    corpus_toks = [tok(f"{d['title']} {d['snippet']}") for d in corpus]
    bm25 = BM25Okapi(corpus_toks)
    idx_path = os.path.join(CONTENT_DIR, "index.pkl")
    with open(idx_path, "wb") as f:
        pickle.dump({"bm25": bm25,
                     "doc_ids": [d["id"] for d in corpus]}, f)
    print(f"index built: {len(corpus)} docs -> {idx_path}")
    return idx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="本次新增抽样 query 数")
    ap.add_argument("--max-results", type=int, default=5)
    ap.add_argument("--proxy", default="http://127.0.0.1:10808")
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--sleep-min", type=float, default=1.5)
    ap.add_argument("--sleep-max", type=float, default=3.5)
    ap.add_argument("--resume", action="store_true", help="跳过已爬 query，合并写入")
    ap.add_argument("--fetch-body", action="store_true", help="抓取结果页正文")
    ap.add_argument("--max-body", type=int, default=0, help="正文抓取条数上限（0=不限制）")
    args = ap.parse_args()

    os.makedirs(CONTENT_DIR, exist_ok=True)
    done, corpus, seen_urls = load_existing() if args.resume else (set(), [], set())
    if args.resume and corpus:
        print(f"resume: existing {len(corpus)} docs, {len(done)} queries done")

    queries = [(q, i) for q, i in sample_queries(args.limit + len(done)) if q not in done]
    print(f"to search: {len(queries)} queries (skipped {len(done)} done)")

    searcher = ResilientSearcher(None if args.no_proxy else args.proxy, args.max_results,
                                 (args.sleep_min, args.sleep_max))
    body_budget = args.max_body if args.max_body > 0 else float("inf")
    new_docs = 0

    with open(CORPUS_PATH, "a", encoding="utf-8") as fout:
        for i, (q, intent) in enumerate(queries):
            region = "wt-wt" if re.search(r"[A-Za-z]{3}", q) and not re.search(r"[\u4e00-\u9fff]", q) else "cn-zh"
            # 主查询 + 简化降级
            attempts = [(q, region)] + [(sq, region) for sq in simplify_query(q)]
            results = []
            for attempt_q, rg in attempts:
                results = searcher.search_one(attempt_q, rg)
                if results:
                    if attempt_q != q:
                        searcher.stats["simplified_ok"] += 1
                    break
            if not results:
                print(f"  [{i+1}/{len(queries)}] all attempts failed: {q[:36]}")
                continue
            for r in results:
                url = (r.get("href") or r.get("url") or "").rstrip("/")
                if not url or url in seen_urls:
                    continue
                row = {
                    "query": q, "query_intent": intent,
                    "title": (r.get("title") or "").strip(),
                    "url": url,
                    "snippet": (r.get("body") or r.get("description") or "").strip(),
                }
                if args.fetch_body and body_budget > 0:
                    row["body"] = searcher.fetch_body(url)
                    body_budget -= 1
                    searcher._sleep()
                text = f"{row['title']} {row['snippet']}"
                if len(text) < 20:
                    continue
                seen_urls.add(url)
                corpus.append(govern_row(row, len(corpus) + 1))
                new_docs += 1
                fout.write(json.dumps(corpus[-1], ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  progress {i+1}/{len(queries)}: +{new_docs} docs, stats={dict(searcher.stats)}")

    print(f"done: +{new_docs} docs, total {len(corpus)} docs, stats={dict(searcher.stats)}")
    if corpus:
        build_index(corpus)


if __name__ == "__main__":
    main()
