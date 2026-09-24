# -*- coding: utf-8 -*-
"""query 归一化：全半角 / 空白折叠 / 大小写 / 中英文语言判定。

归一化结果用于：① 缓存 key（配合 LRU） ② 意图模型输入。
NER 在原文上进行（保留位置信息），仅共享语言判定。
"""
from __future__ import annotations

import re
import unicodedata

# 全角 -> 半角
_FULL2HALF_TRANS = {i + 0xFEE0: i for i in range(0x21, 0x7F)}
_FULL2HALF_TRANS[0x3000] = 0x20  # 全角空格


def _fullwidth_to_halfwidth(text: str) -> str:
    return text.translate(_FULL2HALF_TRANS)


def normalize_query(query: str) -> str:
    """归一化：全角转半角 → Unicode NFC → 空白折叠 → 去首尾空白。
    不做小写化（NER/词典匹配需保留大小写语义），小写化由特征层自理。
    """
    if not query:
        return ""
    text = unicodedata.normalize("NFC", query)
    text = _fullwidth_to_halfwidth(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cache_key(query: str) -> str:
    """缓存 key：归一化 + 小写 + 去多余标点。"""
    text = normalize_query(query).lower()
    text = re.sub(r"[，。！？、；：‘’“”（）【】,.!?;:'\"()\[\]]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_lang(query: str) -> str:
    """极简语言判定：CJK 字符占比高 → zh；拉丁字符为主 → en。"""
    if not query:
        return "zh"
    cjk = len(_CJK_RE.findall(query))
    latin = len(_LATIN_RE.findall(query))
    if cjk >= 1 and cjk * 2 >= latin:
        return "zh"
    if latin > 0:
        return "en"
    return "zh"


def clean_title(title: str, max_len: int = 36) -> str:
    """清洗搜索引擎返回的标题：剥离拼接进来的正文片段。

    搜索引擎（DDG/Bing 等）常把首段正文拼在标题后，特征是
    「标题 + 空格 + 长正文」。规则（按序，命中即返回）：
      1) 长度 <= max_len 原样返回；
      2) 空格分界且前段以闭合括号/引号收尾（标题完整信号）→ 截到前段；
      3) 空格分界且前段 >= 14 字、后段 >= 20 字、分界前字符非数字（防误切「3 年」）→ 截到前段；
      4) 第一个句末标点（。！？!?…）位置 >= 14 → 截到标点后；
      5) 兜底硬截 max_len，去掉尾部残缺标点。
    """
    t = re.sub(r"\s+", " ", (title or "")).strip()
    if len(t) <= max_len:
        return t
    # 2) 前段以闭合括号/引号收尾
    for m in re.finditer(r" ", t):
        head = t[:m.start()]
        if len(head) >= 10 and head[-1] in "）)】》」』\"]":
            return head
    # 3) 空格分界 + 正文特征（后段长、前段够长、前后分界均非数字——防误切「3 年」）
    for m in re.finditer(r" ", t):
        head, tail = t[:m.start()], t[m.end():]
        if (len(head) >= 14 and len(tail) >= 15
                and not t[m.start() - 1].isdigit() and not tail[0].isdigit()):
            return head.rstrip("。，,、·")
    # 4) 句末标点（扫全部标点，取第一个位置足够靠后的）
    for m in re.finditer(r"[。！？!?…]", t):
        if m.end() >= 14:
            return t[:m.end()].rstrip("。")
    # 5) 硬截
    t = t[:max_len].rstrip()
    return t.rstrip("。，,、·…！？!?：:")


# ---------------- 入库文本质量（WAF 挑战页 / JS 脚本 / mojibake 检测） ----------------

_JUNK_SIGS = ("_waf_", "jsvmprt", "document.cookie", "window.location", "eval(",
              "function()", "function (", "Lblfpo", "cf_chl", "captcha",
              "challenge-platform", "悬崖 eztv", "setTimeout(", "localStorage.")

# mojibake 特征：UTF-8 字节被 Latin-1 展示的高频字符（正常中文文本不会出现）
_MOJIBAKE_MARKS = ("æ", "è", "å", "ç", "ã", "ï¼", "ä¸", "é", "å")
# 含 Latin-1 扩展字符的连续段（含前后普通字母/数字/空白，供整段 roundtrip）
_MOJI_RUN = re.compile(r"[\u0080-\u017F][\u0080-\u017F A-Za-z0-9,.:;!?()\-]*")


def fix_mojibake(text: str) -> str:
    """修复/剔除编码错乱段（GBK 或 UTF-8 字节被 Latin-1 展示，如 è¶ÔÍâ / çä¹¸ï¼）。

    策略：含 Latin-1 扩展字符的连续段，尝试 latin-1 -> utf-8 严格还原，
    还原成功且出现 CJK 则替换；失败（GBK 碎片、页脚样板）直接剔除——
    正常中文段不含此类字符，绝无误伤。
    """
    if not text:
        return text
    if not any(m in text for m in _MOJIBAKE_MARKS):
        return text

    def _fix_seg(m):
        seg = m.group(0)
        try:
            fixed = seg.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return " "
        return fixed if any("\u4e00" <= c <= "\u9fff" for c in fixed) else " "

    return re.sub(r"\s+", " ", _MOJI_RUN.sub(_fix_seg, text)).strip()


_DATE_PREFIX = re.compile(r"^(?:[A-Z][a-z]{2} \d{1,2}, \d{4}\s*[·|\-]\s*)+")
_AGO_PREFIX = re.compile(r"^(?:\d+\s*(?:天|小时|分钟|秒)前|\d+\s*days? ago)\s*[-·|]?\s*", re.I)


def deep_clean(text: str, min_cjk: int = 0) -> str:
    """语料字段的深度清洗（入库用，宁严勿滥）：

    1. mojibake 段修复/剔除（fix_mojibake）
    2. 剥搜索引擎拼接的日期前缀（"Apr 29, 2026 · " / "3天前 - "）
    3. 空白折叠
    4. 有效中文数 < min_cjk 视为无价值，返回空串
    """
    if not text:
        return ""
    t = fix_mojibake(text)
    t = _DATE_PREFIX.sub("", t.strip())
    t = _AGO_PREFIX.sub("", t.strip())
    t = re.sub(r"\s+", " ", t).strip()
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    if cjk < min_cjk:
        return ""
    return t


def is_junk_text(text: str, min_len: int = 120) -> bool:
    """判定正文是否为反爬脚本/挑战页等垃圾（title/snippet 不走此判定，正文专用）。

    策略：命中反爬签名直接判垃圾；否则长文本下若 CJK 占比过低且
    JS 符号密度高（;{} 组合），也判垃圾。
    """
    if not text or len(text) < min_len:
        return False
    if any(sig in text for sig in _JUNK_SIGS):
        return True
    n = len(text)
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if cjk / n < 0.15:
        js_marks = sum(text.count(c) for c in ";{}")
        if js_marks / n > 0.02:
            return True
    return False
