# -*- coding: utf-8 -*-
"""存量语料重治理：按最新规则重打 content_type + linked_entities，重建索引。

用法：python scripts/retag_corpus.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.build_content_library import (
    classify_content_type, extract_date, extract_date_from_url,
    load_existing, build_index, CORPUS_PATH,
)
from qp.ner.entity_linker import default_linker

done, corpus, _ = load_existing()
linker = default_linker()
changed_type = 0
for i, d in enumerate(corpus, start=1):
    d["id"] = f"doc{i:05d}"
    new_t = classify_content_type(d.get("title", ""), d.get("snippet", ""), d.get("url", ""))
    if new_t != d.get("content_type"):
        changed_type += 1
    d["content_type"] = new_t
    d["linked_entities"] = linker.tag_doc(
        f"{d.get('title', '')} {d.get('snippet', '')} {d.get('body', '')}")
    if not d.get("date"):
        d["date"] = extract_date(f"{d.get('title', '')} {d.get('snippet', '')}") \
            or extract_date_from_url(d.get("url", ""))

with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    for d in corpus:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
build_index(corpus)
from collections import Counter
print(f"docs={len(corpus)} type_changed={changed_type} gazetteer={linker.size()}")
print("types:", dict(Counter(d["content_type"] for d in corpus)))
tagged = sum(1 for d in corpus if d["linked_entities"])
print(f"tagged with linked_entities: {tagged}/{len(corpus)}")
