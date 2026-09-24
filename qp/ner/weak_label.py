# -*- coding: utf-8 -*-
"""NER 弱标注：正则 → BIO 序列标签，产出服务器训练数据。

实体类型（11 类）：
  STOCK / STOCK_CODE / COMPANY / INDEX / FUND / INDUSTRY / MACRO / ORG / PRODUCT / TIME / METRIC

弱标注策略（银标准）：
  1. TIME        —— 绝对/相对时间表达式正则（高精度，模式规整）
  2. STOCK_CODE  —— 6 位证券代码 + 港股 4~5 位.HK（前缀判交易所）
  3. COMPANY/ORG/PRODUCT/METRIC/FUND/INDEX —— gazetteer 词条（11 类扩充，长词优先）
  4. FUND        —— 基金关键词（ETF/联接基金/指数基金...）
  5. INDUSTRY    —— 行业种子词库（最长匹配）
  6. MACRO       —— 宏观主体/政策词
  7. STOCK       —— 知名公司种子词（中/英）+ 中文公司名后缀模式（如「XX股份/XX科技/XX集团」）

产出：JSONL，每行 {text, tokens(chars), tags(BIO)}。
注意：弱标注存在噪声（尤其 STOCK 后缀模式），训练前需抽样审计 + 服务器端可用
LLM teacher 清洗；精度优先，召回不足的部分由词典外实体链接补齐。
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")

ENTITY_TYPES = ["STOCK", "STOCK_CODE", "COMPANY", "INDEX", "FUND", "INDUSTRY",
                "MACRO", "ORG", "PRODUCT", "TIME", "METRIC"]
LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]


@lru_cache(maxsize=1)
def _lexicons() -> dict:
    with open(os.path.join(CONFIG_DIR, "lexicons.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _gazetteer_words() -> Dict[str, List[str]]:
    """从 gazetteer 读取 11 类扩充词条（name+别名，跳过纯数字/过短）。

    只取非 STOCK 类型（STOCK 主数据 1.5 万条由种子词+链接覆盖，全量进正则太重）。
    """
    gaz_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "entities", "gazetteer.jsonl")
    out: Dict[str, List[str]] = {}
    try:
        with open(gaz_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                t = e.get("type", "")
                if t not in ("COMPANY", "ORG", "PRODUCT", "METRIC", "FUND", "INDEX"):
                    continue
                words = out.setdefault(t, [])
                for w in [e.get("name", "")] + list(e.get("aliases", [])):
                    w = w.strip()
                    if len(w) >= 2 and not w.isdigit():
                        words.append(w)
    except OSError:
        pass
    return {k: sorted(set(v)) for k, v in out.items()}


def _build_matchers() -> List[Tuple[re.Pattern, str, int]]:
    """返回 [(compiled, entity_type, priority)]，priority 小者先占位（不可重叠）。"""
    lx = _lexicons()
    matchers: List[Tuple[re.Pattern, str, int]] = []

    # TIME（绝对时间优先级最高）
    for pat in lx["time_expressions"]["absolute"]:
        matchers.append((re.compile(pat), "TIME", 0))
    # 相对时间词（中文词不加 CJK 边界，ASCII 词加 alnum 边界）
    for key in ("relative_latest", "relative_past", "relative_future"):
        for w in lx["time_expressions"][key]:
            if w.isascii():
                pat = re.compile(rf"(?<![0-9A-Za-z]){re.escape(w)}(?![0-9A-Za-z])")
            else:
                pat = re.compile(re.escape(w))
            matchers.append((pat, "TIME", 1))

    # STOCK_CODE
    matchers.append((re.compile(lx["stock_code_pattern"]), "STOCK_CODE", 0))
    # 港股代码变体（0700.HK / 9988.HK），链接阶段经 gazetteer 别名挂到具体个股
    matchers.append((re.compile(r"(?<!\d)(\d{4,5})\.HK(?!\d)"), "STOCK_CODE", 0))

    # INDEX / FUND / MACRO / INDUSTRY / STOCK 种子词（词典词，长词优先 -> priority 相同按长度）
    def dict_matchers(words, etype, prio):
        """词典匹配：ASCII 词用 alnum 边界（防 AAPL 误中 CAPITAL）；
        中文词不加 CJK 边界（『半导体板块』需命中），嵌套冲突由最长优先+span占位解决。"""
        out = []
        for w in sorted(set(words), key=len, reverse=True):
            w = w.strip()
            if len(w) < 2:
                continue
            if w.isascii():
                pat = re.compile(rf"(?<![0-9A-Za-z]){re.escape(w)}(?![0-9A-Za-z])")
            else:
                pat = re.compile(re.escape(w))
            out.append((pat, etype, prio))
        return out

    matchers += dict_matchers(lx["index_names"], "INDEX", 1)
    matchers += dict_matchers(lx["fund_keywords"], "FUND", 1)
    matchers += dict_matchers(lx["macro_terms"], "MACRO", 2)
    matchers += dict_matchers(lx["industries"], "INDUSTRY", 3)
    matchers += dict_matchers(lx["stock_seeds_zh"] + [s.strip() for s in lx["stock_seeds_en"]], "STOCK", 1)

    # gazetteer 11 类扩充词条（priority 1，先于 yaml 的 MACRO/INDUSTRY 种子，
    # 保证「毛利率」压过「利率」、gazetteer 的 ORG「证监会」按 11 类规范生效）
    gaz = _gazetteer_words()
    for etype in ("COMPANY", "ORG", "PRODUCT", "METRIC", "FUND", "INDEX"):
        matchers += dict_matchers(gaz.get(etype, []), etype, 1)

    # STOCK 后缀模式（噪声较大，priority 最低，仅在无更强标注处生效）
    for pat in lx["stock_suffix_patterns"]:
        matchers.append((re.compile(pat), "STOCK", 4))

    return matchers


_MATCHERS: List[Tuple[re.Pattern, str, int]] = None


def _matchers() -> List[Tuple[re.Pattern, str, int]]:
    global _MATCHERS
    if _MATCHERS is None:
        _MATCHERS = _build_matchers()
    return _MATCHERS


# ----------------------------------------------------------------------
def find_entities(text: str) -> List[Dict]:
    """规则实体抽取（本机兜底推理 + 弱标注共用）。
    返回 [{text, type, start, end}]，按 (priority, -len) 占位，已占用 span 不再覆盖。
    """
    spans: List[Tuple[int, int, int, str]] = []   # (priority, start, end, type)
    for pat, etype, prio in _matchers():
        for m in pat.finditer(text):
            spans.append((prio, m.start(), m.end(), etype))
    # priority 升序、长度降序占位
    spans.sort(key=lambda x: (x[0], -(x[2] - x[1])))
    occupied = [False] * len(text)
    out: List[Dict] = []
    for prio, s, e, etype in spans:
        if any(occupied[s:e]):
            continue
        for i in range(s, e):
            occupied[i] = True
        out.append({"text": text[s:e], "type": etype, "start": s, "end": e})
    out.sort(key=lambda x: x["start"])
    return out


def to_bio(text: str, entities: List[Dict]) -> List[str]:
    """文本（按字符切分）→ BIO 标签序列。空格并入前字符的 O。"""
    tags = ["O"] * len(text)
    for ent in entities:
        first = True
        for i in range(ent["start"], ent["end"]):
            if text[i].isspace():
                continue
            tags[i] = f"B-{ent['type']}" if first else f"I-{ent['type']}"
            first = False
    return tags


def weak_label_record(text: str) -> Dict:
    ents = find_entities(text)
    tags = to_bio(text, ents)
    return {
        "text": text,
        "chars": list(text),
        "tags": tags,
        "entities": [{"text": e["text"], "type": e["type"], "start": e["start"], "end": e["end"]} for e in ents],
    }


def weak_label_file(src: str, dst: str, limit: int = 0) -> Dict[str, int]:
    """对 query 文件逐行弱标注，写 JSONL。返回统计。"""
    stats: Dict[str, int] = {"total": 0, "with_entity": 0}
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            q = line.strip()
            if not q:
                continue
            if limit and stats["total"] >= limit:
                break
            rec = weak_label_record(q)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["total"] += 1
            if rec["entities"]:
                stats["with_entity"] += 1
            for e in rec["entities"]:
                stats[e["type"]] = stats.get(e["type"], 0) + 1
    return stats
