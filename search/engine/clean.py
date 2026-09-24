# -*- coding: utf-8 -*-
"""clean —— web 搜索/doc 文本清洗与质量门禁（可复用模块）。

解决两类污染：
  A. 模板污染：行情/频道页的滚动新闻条（「限制柴油出口的争论升温 (09-24 08:23)…」）、
     站点导航句（「新浪财经-美股频道为您提供…信息与服务」）、免责声明等模板文本
     会混入 snippet/body，导致 BM25 对无关 query 误命中（如柴油禁令 query 命中 BABA 行情页）。
  B. 低价值页面：行情页/频道页/列表页本身不是新闻内容，入库只增加噪声。

供三处调用：
  1. inline_crawl.ingest_hits  —— 入库前 gate（quote/模板页丢弃 + 文本清洗）
  2. content_search.corpus     —— chunk 切分前清洗 body
  3. serving 输出              —— 老库文档的 snippet/best_chunk 展示清洗（不动索引）
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 滚动新闻条 / 日期戳列表：轮播块的特征是密集的「(MM-DD HH:MM)」日期戳
_DATESEG = r"\d{2,4}[-/.]\d{1,2}(?:[-/.]\d{1,2})?"          # MM-DD 或 YYYY-MM-DD
_STAMP_RE = re.compile(rf"\({_DATESEG}(?:\s+\d{{1,2}}:\d{{2}})?\)")
_ROLLING_MIN_STAMPS = 3          # 无时分戳的日期括号需 >=3 个才判定
_ROLLING_MIN_TIMED = 2           # 带时分戳（盘中共振式轮播）>=2 即判定

# 站点模板句（整句删除；「为您提供」几乎只出现在站点模板文案中，可安全单条件匹配）
_TEMPLATE_SENT_RE = re.compile(
    r"[^。！？.!?]*(?:为您提供|等与[^。！？.!?]{2,20}相关的信息与服务|"
    r"实时交易数据|实时行情数据)[^。！？.!?]*[。！？.!?]?"
)
# 导航/推广噪声（片段删除；仅固定词组，不含可单独成词的「股票/基金/期货」等）
_NOISE_FRAG_RE = re.compile(
    r"(?:更多\s*$|更多\s*\(|^首页$|网站导航|登录|注册\s*$|下载(?:App|客户端)|"
    r"免责声明[：:]*|风险提示[：:]*|图片来源[：:]*|视觉中国|责任编辑[：:]*|"
    r"原标题[：:]*|点击查看|戳我|海量资讯|精准解读|尽在新浪|"
    r"APP\s*[|丨]\s*聚焦|聚焦\s*APP|以上内容仅供参考|不构成(?:任何)?投资建议|"
    r"相关的?信息与服务|股票股价|实时交易数据|实时行情数据|财务指标分析)"
)

# 行情/频道页特征（标题级判定）
_QUOTE_TITLE_STRONG_RE = re.compile(
    r"(?:股票股价|实时行情[数,，]|行情频道|行情中心|股价.{0,4}实时行情|"
    r"财务指标分析等与|研究报告.{0,4}评级.{0,4}财务指标)"
)
# 列表/聚合页特征
_LIST_TITLE_RE = re.compile(r"(?:新闻列表|最新消息汇总|消息面一览|个股直播|快讯汇总)")


def strip_boilerplate(text: str) -> str:
    """清洗模板/噪声文本，返回净化后的正文。

    顺序：删除滚动新闻条块 -> 删模板整句 -> 删噪声片段 -> 空白归一。
    不改变语义内容，可安全用于 snippet/body/chunk。
    """
    if not text:
        return text
    t = text
    # 1. 滚动新闻条：以「(日期[ 时:分])」日期戳密度判定；保留首个戳之前的实质正文
    #    （不做「标题前缀」吞并，避免正则把句中真实内容一并吃掉）
    stamps = list(_STAMP_RE.finditer(t))
    n_timed = sum(1 for m in stamps if re.search(r"\d{1,2}:\d{2}", m.group(0)))
    if len(stamps) >= _ROLLING_MIN_STAMPS or n_timed >= _ROLLING_MIN_TIMED:
        head = t[:stamps[0].start()].strip()
        # 头部是实质正文（够长或含完整句子）则保留，否则整体视为模板
        t = head if (len(head) >= 30 or "。" in head) else ""
    # 2. 模板整句
    t = _TEMPLATE_SENT_RE.sub(" ", t)
    # 3. 噪声片段
    t = _NOISE_FRAG_RE.sub(" ", t)
    # 4. 残余孤立日期戳
    t = _STAMP_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_quote_or_list_page(title: str, snippet: str = "") -> bool:
    """行情/频道/列表页判定：这类页面不是新闻正文，入库即噪声。"""
    title_l = (title or "").strip()
    if _QUOTE_TITLE_STRONG_RE.search(title_l):
        return True
    if _LIST_TITLE_RE.search(title_l):
        return True
    # snippet 大部分由模板句构成（清洗后所剩无几且原文本长）
    if len(snippet) >= 40:
        cleaned = strip_boilerplate(snippet)
        if len(cleaned) < 20:
            return True
    return False


def clean_doc_text(title: str, snippet: str = "", body: str = "") -> tuple[str, str, bool]:
    """doc 级清洗入口（入库/重建 chunk 共用）。

    返回 (title, text, ok)：text 为净化后的 snippet+body 拼接；
    ok=False 表示判定为行情/模板/垃圾页，调用方应跳过入库。
    """
    from qp.common.normalize import is_junk_text
    title_c = strip_boilerplate(title or "")
    text = strip_boilerplate(f"{snippet or ''} {body or ''}".strip())
    if not title_c or len(title_c) < 4:
        return title_c, text, False
    if is_junk_text(f"{title_c} {text}", min_len=60):
        return title_c, text, False
    if is_quote_or_list_page(title_c, snippet or ""):
        return title_c, text, False
    return title_c, text, True
