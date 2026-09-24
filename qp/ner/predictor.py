# -*- coding: utf-8 -*-
"""NER 推理器：有权重 → BiLSTM-CRF 模型；无权重 → 规则兜底（弱标注同一套匹配器）。

线上部署：模型权重（服务器训练产出）+ 词典/规则兜底级联 —— 与意图识别三级级联
同构：模型低置信 span 回退词典，保证 99% 目标由级联而非单模型承担。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from qp.ner.weak_label import find_entities

_MODEL = None                      # lazy singleton


def load_model(weights_path: str):
    """加载 BiLSTM-CRF 权重（需要 torch；仅服务器/有权重环境调用）。"""
    global _MODEL
    import torch
    from qp.ner.model import BiLSTMCRF
    from qp.ner.data import NERDataset

    ckpt = torch.load(weights_path, map_location="cpu")
    cfg = ckpt["config"]
    model = BiLSTMCRF(
        vocab_size=len(ckpt["char_vocab"]), num_tags=len(ckpt["label_map"]),
        embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _MODEL = {"model": model, "char_vocab": ckpt["char_vocab"],
              "id2label": {v: k for k, v in ckpt["label_map"].items()}}
    return _MODEL


def _model_predict(text: str) -> Optional[List[Dict]]:
    if _MODEL is None:
        return None
    import torch
    from qp.ner.data import char_ngram_hashes

    chars = list(text)
    char_ids = [_MODEL["char_vocab"].get(c, 1) for c in chars]
    ngrams = char_ngram_hashes(chars)
    t = torch.tensor([char_ids]); n = torch.tensor([ngrams]); m = torch.ones_like(t, dtype=torch.bool)
    paths = _MODEL["model"].decode(t, m, n)
    tags = [_MODEL["id2label"][i] for i in paths[0]]

    ents, cur, cur_type, start = [], [], None, 0
    for i, (ch, tag) in enumerate(zip(chars, tags)):
        if tag.startswith("B-"):
            if cur:
                ents.append({"text": "".join(cur), "type": cur_type, "start": start, "end": i, "source": "model"})
            cur, cur_type, start = [ch], tag[2:], i
        elif tag.startswith("I-") and cur_type == tag[2:] and cur:
            cur.append(ch)
        else:
            if cur:
                ents.append({"text": "".join(cur), "type": cur_type, "start": start, "end": i, "source": "model"})
            cur, cur_type = [], None
    if cur:
        ents.append({"text": "".join(cur), "type": cur_type, "start": start, "end": len(chars), "source": "model"})
    return ents


def predict(text: str, weights_path: Optional[str] = None, use_rule_fallback: bool = True) -> List[Dict]:
    """NER 推理总入口。

    - weights_path 给定且可加载 → 模型推理
    - 否则 use_rule_fallback=True → 规则兜底（词典+模式，与弱标注同源）
    - 模型输出为空且规则有命中时，取规则结果补齐（级联兜底）
    """
    if weights_path and os.path.exists(weights_path) and _MODEL is None:
        try:
            load_model(weights_path)
        except Exception as e:                      # torch 缺失 / 权重不兼容
            print(f"[ner] model load failed, fallback to rules: {e}")
    if _MODEL is not None:
        ents = _model_predict(text) or []
        if ents or not use_rule_fallback:
            return ents
        rule_ents = find_entities(text)
        return [dict(e, source="rule") for e in rule_ents]
    if use_rule_fallback:
        return [dict(e, source="rule") for e in find_entities(text)]
    return []
