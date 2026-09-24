# -*- coding: utf-8 -*-
"""实体链指（Entity Linking）：query 侧 NER 实体与 doc 侧实体在统一词表层对齐。

组成：
  - 实体词表 gazetteer（data/entities/gazetteer.jsonl）：
    {eid, name, aliases, type, source}，eid = "E"+md5(canonical)[:8]，跨重建稳定。
    种子来自 configs/lexicons.yaml（指数/行业/股票/宏观），运行期增量注册
    （query 侧 LLM 新实体 source=query_llm；doc 侧挖矿 source=mined，后续升级）。
  - query 侧链指：NER span -> 词表（exact / alias / 归一化匹配），未命中注册为新实体。
  - doc 侧打标：正文扫描词表，产出 linked_entities 存入语料行，供检索加权与评测标注。

用法（线程安全）：
    from qp.ner.entity_linker import default_linker
    linker = default_linker()
    ref = linker.link("智谱", etype="COMPANY")   # -> {eid, name, type, match}
    linker.tag_doc(title + snippet)              # -> [{eid, text, type}, ...]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from functools import lru_cache

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAZ_PATH = os.path.join(ROOT, "data", "entities", "gazetteer.jsonl")
LEX_PATH = os.path.join(ROOT, "configs", "lexicons.yaml")

_SUFFIX = re.compile(r"(股份|集团|科技|控股|有限公司|公司)$")


def _norm(name: str) -> str:
    """链指归一化：小写、去公司后缀、去空白。"""
    n = _SUFFIX.sub("", name.strip().lower())
    return re.sub(r"\s+", "", n)


def eid_of(canonical: str) -> str:
    return "E" + hashlib.md5(canonical.strip().lower().encode("utf-8")).hexdigest()[:8]


class EntityLinker:
    def __init__(self, gaz_path: str = GAZ_PATH):
        self.gaz_path = gaz_path
        self._lock = threading.Lock()
        self._by_norm: dict[str, dict] = {}     # norm(name/alias) -> entry
        self._by_eid: dict[str, dict] = {}
        os.makedirs(os.path.dirname(gaz_path), exist_ok=True)
        self._load_or_seed()

    # ------------------------------------------------------------------
    def _load_or_seed(self):
        if os.path.exists(self.gaz_path) and os.path.getsize(self.gaz_path) > 0:
            with open(self.gaz_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._index(json.loads(line))
            return
        # 种子：lexicons.yaml 内置词表 + 媒体/平台名（消解「腾讯新闻≠腾讯」类歧义）
        with open(LEX_PATH, encoding="utf-8") as f:
            lx = yaml.safe_load(f)
        media = ["腾讯新闻", "腾讯网", "新浪财经", "新浪新闻", "东方财富网", "东方财富",
                 "天天基金网", "雪球", "澎湃新闻", "财新网", "证券之星", "同花顺",
                 "第一财经", "界面新闻", "华尔街见闻", "36氪", "格隆汇", "富途牛牛"]
        seeds = ([(n, "INDEX") for n in lx.get("index_names", [])]
                 + [(n, "INDUSTRY") for n in lx.get("industries", [])]
                 + [(n, "STOCK") for n in lx.get("stock_seeds_zh", [])]
                 + [(n, "STOCK") for n in lx.get("stock_seeds_en", [])]
                 + [(n, "MACRO") for n in lx.get("macro_terms", [])]
                 + [(n, "MEDIA") for n in media])
        with open(self.gaz_path, "w", encoding="utf-8") as f:
            for name, t in seeds:
                e = self._make(name, t, source="seed")
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                self._index(e)

    def _make(self, name: str, etype: str, source: str) -> dict:
        return {"eid": eid_of(name), "name": name, "aliases": [],
                "type": etype, "source": source}

    def _index(self, entry: dict):
        self._by_eid[entry["eid"]] = entry
        self._by_norm[_norm(entry["name"])] = entry
        for a in entry.get("aliases", []):
            self._by_norm[_norm(a)] = entry
        self._buckets_dirty = True   # tag_doc 首字桶索引失效

    # ------------------------------------------------------------------
    def link(self, span: str, etype: str = "") -> dict | None:
        """span -> 词表实体。命中返回 {eid, name, type, match}；未命中返回 None。"""
        n = _norm(span)
        if not n:
            return None
        e = self._by_norm.get(n)
        if e:
            return {"eid": e["eid"], "name": e["name"], "type": e["type"], "match": "exact"}
        # 别名级：span 是某实体名的子串（长度>=2），如「智谱」⊂「智谱AI」
        if len(n) >= 2:
            for norm, e2 in self._by_norm.items():
                if len(norm) >= 2 and (norm in n or n in norm):
                    return {"eid": e2["eid"], "name": e2["name"], "type": e2["type"],
                            "match": "partial"}
        return None

    def register(self, span: str, etype: str, source: str = "query_llm") -> dict:
        """注册（或复用）实体，返回 {eid, name, type, match}。幂等。"""
        n = _norm(span)
        with self._lock:
            hit = self.link(span, etype)
            if hit:
                return hit
            entry = self._make(span, etype, source)
            self._index(entry)
            with open(self.gaz_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return {"eid": entry["eid"], "name": entry["name"],
                    "type": entry["type"], "match": "new"}

    # ------------------------------------------------------------------
    def add_alias(self, eid: str, alias: str) -> bool:
        """给已有实体补别名（doc 侧挖矿）。别名 norm 未被占用时生效并重写词表文件。"""
        n = _norm(alias)
        if not n:
            return False
        with self._lock:
            e = self._by_eid.get(eid)
            if not e or n in self._by_norm:
                return False
            e.setdefault("aliases", [])
            if alias not in e["aliases"]:
                e["aliases"].append(alias)
                self._buckets_dirty = True
            self._by_norm[n] = e
            self._rewrite()
            return True

    def _rewrite(self):
        """词表文件全量重写（按 eid 去重，别名合并进条目）。"""
        with open(self.gaz_path, "w", encoding="utf-8") as f:
            for e in self._by_eid.values():
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def _rebuild_buckets(self):
        """tag_doc 扫描索引：norm 首字符 -> [(norm, entry)]。

        词表 1.5 万实体时，线性 find 扫描每条文本 O(15k)；按首字分桶后
        每个文本位置只探对应桶，命中探测数下降 ~2 个数量级。
        """
        buckets: dict[str, list] = {}
        for norm, e in self._by_norm.items():
            if len(norm) >= 2 and norm:
                buckets.setdefault(norm[0], []).append((norm, e))
        for v in buckets.values():
            v.sort(key=lambda x: -len(x[0]))   # 同首字内长名优先
        self._buckets = buckets
        self._buckets_dirty = False

    def tag_doc(self, text: str) -> list[dict]:
        """doc 侧打标：首字桶多模式匹配 + 最长优先占位去重叠。

        保证「智谱配售..._腾讯新闻」链接到 腾讯新闻(MEDIA) 而非 腾讯(COMPANY)，
        平台署名不会被误当成公司实体。
        """
        if getattr(self, "_buckets_dirty", True) or not hasattr(self, "_buckets"):
            self._rebuild_buckets()
        low = text.lower()
        out, seen = [], set()
        occupied = []  # (start, end) 已匹配片段
        hits = []
        for i, c in enumerate(low):
            lst = self._buckets.get(c)
            if not lst:
                continue
            for norm, e in lst:
                j = i + len(norm)
                if low[i:j] == norm:
                    hits.append((i, j, e))
        # 最长优先占位（与线性扫描版行为一致），同长按出现位置先后
        for i, j, e in sorted(hits, key=lambda x: (-(x[1] - x[0]), x[0])):
            if any(not (j <= s or i >= t) for s, t in occupied):
                continue
            occupied.append((i, j))
            if e["eid"] not in seen:
                seen.add(e["eid"])
                out.append({"eid": e["eid"], "text": e["name"], "type": e["type"]})
        return out

    def size(self) -> int:
        return len(self._by_eid)

    def meta_of(self, eid: str) -> dict:
        """实体 meta（markets/codes/full_name 等），无则空 dict。"""
        return (self._by_eid.get(eid) or {}).get("meta") or {}


@lru_cache(maxsize=1)
def default_linker() -> EntityLinker:
    return EntityLinker()
