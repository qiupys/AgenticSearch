# -*- coding: utf-8 -*-
"""意图体系加载与校验：8 域 × 42 二级 + 7 槽位 + 产品 16 类映射。"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")

_L2_DOMAIN_CACHE: Dict[str, str] = {}


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(content)


@lru_cache(maxsize=1)
def taxonomy() -> dict:
    return _load(os.path.join(CONFIG_DIR, "intent_taxonomy.json"))


@lru_cache(maxsize=1)
def category_mapping() -> dict:
    return _load(os.path.join(CONFIG_DIR, "category_mapping.json"))


def l2_domain(intent_id: str) -> Optional[str]:
    """INFO-07 -> INFO；非法 id 返回 None。"""
    if intent_id in _L2_DOMAIN_CACHE:
        return _L2_DOMAIN_CACHE[intent_id]
    dom = intent_id.split("-")[0] if "-" in intent_id else None
    if dom and dom in {d["id"] for d in taxonomy()["domains"]}:
        _L2_DOMAIN_CACHE[intent_id] = dom
        return dom
    return None


def all_l2_ids() -> List[str]:
    return [i["id"] for i in taxonomy()["intents"]]


def l2_name(intent_id: str) -> str:
    for i in taxonomy()["intents"]:
        if i["id"] == intent_id:
            return i["name"]
    return intent_id


def domain_name(domain_id: str) -> str:
    for d in taxonomy()["domains"]:
        if d["id"] == domain_id:
            return d["name"]
    return domain_id


# ---------- 产品 16 类映射 ----------

@lru_cache(maxsize=256)
def product_category_info(product_category: str) -> Optional[dict]:
    """产品 16 类 -> {primary_l2, acceptable_l2, acceptable_l1, aliases}"""
    for m in category_mapping()["mappings"]:
        if product_category in (m["product_category"], *m.get("aliases", [])):
            return m
    return None


def l2_to_product_categories(intent_id: str) -> List[str]:
    """反向映射：L2 -> 可对应的产品 16 类集合（用于输出 stats_category_16 与统计对齐）。"""
    hits = []
    for m in category_mapping()["mappings"]:
        if intent_id in m["acceptable_l2"]:
            hits.append(m["product_category"])
    return hits


def acceptable_l2(product_category: str) -> List[str]:
    info = product_category_info(product_category)
    return info["acceptable_l2"] if info else []


def acceptable_l1(product_category: str) -> List[str]:
    info = product_category_info(product_category)
    return info["acceptable_l1"] if info else []
