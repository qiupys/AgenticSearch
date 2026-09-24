# -*- coding: utf-8 -*-
"""检索自学习：命中/未命中的实体反哺链指词表与 jieba 分词，让系统越用越好用。

反馈路径：
  1. query 侧：QP 识别出的核心实体（含 LLM 补识别的新实体）-> gazetteer 注册
     （pipeline 已做）+ jieba.add_word（保证下次查询「智谱」不被切碎，NER/BM25 都受益）
  2. 未命中 token：检索时发现语料全不含的词（missing_entities）-> jieba 用户词典，
     下次同样表达可整体成词
  3. doc 侧挖矿：爬取入库的新文档标题中，核心实体的扩展形式
     （小米 -> 小米集团－W(1810)）-> 注册为该实体 alias，提高 doc 链指命中率

线程安全；词典文件追加写 + 进程内去重。
"""
from __future__ import annotations

import json
import os
import re
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DICT_PATH = os.path.join(ROOT, "data", "dict", "jieba_user_dict.txt")

_lock = threading.Lock()
_known: set | None = None     # 进程内已学词集合（首次懒加载自词典文件）


def _load_known() -> set:
    global _known
    with _lock:
        if _known is None:
            words = set()
            if os.path.exists(DICT_PATH):
                with open(DICT_PATH, encoding="utf-8") as f:
                    for line in f:
                        w = line.split()[0].strip() if line.strip() else ""
                        if w:
                            words.add(w)
            _known = words
    return _known


def _add_words(words: list[str]):
    """jieba 运行时加词 + 用户词典文件追加（幂等）。"""
    if not words:
        return
    known = _load_known()
    fresh = []
    with _lock:
        import jieba
        for w in words:
            w = (w or "").strip()
            if len(w) < 2 or w in known or re.search(r"\s", w):
                continue
            jieba.add_word(w, freq=2_000_000)   # 高词频：保证整体切分优先
            known.add(w)
            fresh.append(w)
        if fresh:
            os.makedirs(os.path.dirname(DICT_PATH), exist_ok=True)
            with open(DICT_PATH, "a", encoding="utf-8") as f:
                for w in fresh:
                    f.write(f"{w} 2000000 nz\n")
    return fresh


def align_gazetteer_to_jieba() -> int:
    """服务启动时：链指词表全量同步进 jieba 用户词典（幂等）。返回新增词数。

    保证 NER/链指词表与 BM25 分词词典永远对齐 —— 新实体（query_llm 注册、
    doc 挖矿别名）只会进词表的路径也在这里兜底同步。
    """
    from qp.ner.entity_linker import default_linker
    words = []
    for e in default_linker()._by_eid.values():
        words.append(e["name"])
        words.extend(e.get("aliases", []))
    fresh = _add_words([w for w in words if w]) or []
    return len(fresh)


def learn_token(word: str):
    """非实体新词 -> 仅增强 jieba 分词（如缺失 token「克莱斯勒」）。"""
    _add_words([word])


def learn_entity(name: str, etype: str, source: str = "learned") -> dict:
    """实体注册（gazetteer）+ jieba 分词增强。幂等，返回 {eid, match}。"""
    from qp.ner.entity_linker import default_linker
    ref = default_linker().register(name, etype, source=source)
    _add_words([name, ref.get("name", "")])
    return ref


def learn_alias(eid: str, alias: str):
    """给已有实体补别名（同 eid），如 doc 标题里挖出的「小米集团－W(1810)」。"""
    if not alias or len(alias.strip()) < 2:
        return
    from qp.ner.entity_linker import default_linker
    changed = default_linker().add_alias(eid, alias.strip())
    if changed:
        _add_words([alias.strip()])


def mine_alias(core: str, titles: list[str]) -> str | None:
    """从 doc 标题挖实体的扩展形式：小米 -> 「小米集团－W(1810)」。

    规则：实体名开头 + 紧跟集团/公司/控股/科技等后缀或代码括号，总长 <= 16。
    """
    if not core:
        return None
    pat = re.compile(re.escape(core) + r"[0-9A-Za-z\u4e00-\u9fff－\-()（）\[\]]{1,12}")
    for t in titles:
        m = pat.search(t or "")
        if not m:
            continue
        cand = m.group(0)
        if cand == core or len(cand) > len(core) + 12:
            continue
        # 必须含后缀/代码特征，避免把标题后续普通词拼进来
        if not re.search(r"集团|股份|控股|科技|公司|－|-\s*[A-Z]|[（(]\d{4,6}[)）]", cand):
            continue
        return cand
    return None
