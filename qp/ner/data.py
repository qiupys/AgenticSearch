# -*- coding: utf-8 -*-
"""NER 数据管道：词表构建 / char n-gram 哈希 / Dataset / Collate（PyTorch，服务器训练用）。"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"
NGRAM = 3


def build_vocab(records: List[dict], min_freq: int = 2) -> Dict[str, int]:
    counter = Counter(ch for r in records for ch in r["chars"])
    itos = [PAD, UNK] + [c for c, n in counter.most_common() if n >= min_freq]
    return {c: i for i, c in enumerate(itos)}


def build_label_map(labels: List[str]) -> Dict[str, int]:
    return {l: i for i, l in enumerate(labels)}


def char_ngram_hashes(chars: List[str]) -> List[int]:
    """字符 3-gram 哈希特征（fastText 风格），1-based（0 为 padding）。"""
    out = []
    text = "".join(chars)
    for i in range(len(text)):
        gram = text[max(0, i - NGRAM + 1): i + 1]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
        out.append(h % 49999 + 1)
    return out


class NERDataset(Dataset):
    def __init__(self, records: List[dict], char_vocab: Dict[str, int], label_map: Dict[str, int]):
        self.records = records
        self.char_vocab = char_vocab
        self.label_map = label_map

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        char_ids = [self.char_vocab.get(c, self.char_vocab[UNK]) for c in r["chars"]]
        tag_ids = [self.label_map.get(t, 0) for t in r["tags"]]
        ngrams = char_ngram_hashes(r["chars"])
        return {"chars": char_ids, "tags": tag_ids, "ngrams": ngrams}


def collate(batch: List[dict], pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maxlen = max(len(b["chars"]) for b in batch)
    chars = torch.full((len(batch), maxlen), pad_idx, dtype=torch.long)
    tags = torch.zeros((len(batch), maxlen), dtype=torch.long)
    ngrams = torch.zeros((len(batch), maxlen), dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
    for i, b in enumerate(batch):
        L = len(b["chars"])
        chars[i, :L] = torch.tensor(b["chars"], dtype=torch.long)
        tags[i, :L] = torch.tensor(b["tags"], dtype=torch.long)
        ngrams[i, :L] = torch.tensor(b["ngrams"], dtype=torch.long)
        mask[i, :L] = True
    return chars, tags, mask, ngrams


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split(records: List[dict], val_ratio: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * val_ratio))
    val = [records[i] for i in idx[:n_val]]
    train = [records[i] for i in idx[n_val:]]
    return train, val
