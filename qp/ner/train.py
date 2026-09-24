# -*- coding: utf-8 -*-
"""NER 训练入口（服务器/GPU 运行，本机仅写码不执行）。

用法（服务器）：
  python -m ner.train --train-data data/ner/train.jsonl \
                      --epochs 15 --batch-size 128 --lr 2e-3 \
                      --out data/models/ner_bilstm_crf.pt

数据：ner/weak_label.py 产出的弱标注 JSONL（银标准）。
上线前用人工标注金标评测集验收实体级 P/R/F1（目标 99% 由『词典+模型+实体链接』
级联达成，单模型弱标注直训是第一版基线）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qp.ner.data import NERDataset, build_label_map, build_vocab, collate, load_jsonl, split
from qp.ner.model import BiLSTMCRF
from qp.ner.weak_label import LABELS


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_entities_from_tags(chars: List[str], tags: List[str]):
    ents, cur, cur_type = [], [], None
    for ch, tag in zip(chars, tags):
        if tag.startswith("B-"):
            if cur:
                ents.append(("".join(cur), cur_type))
            cur, cur_type = [ch], tag[2:]
        elif tag.startswith("I-") and cur_type == tag[2:] and cur:
            cur.append(ch)
        else:
            if cur:
                ents.append(("".join(cur), cur_type))
            cur, cur_type = [], None
    if cur:
        ents.append(("".join(cur), cur_type))
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/ner/train.jsonl")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/models/ner_bilstm_crf.pt")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = load_jsonl(args.train_data)
    train_recs, val_recs = split(records, args.val_ratio, args.seed)
    print(f"train={len(train_recs)}  val={len(val_recs)}")

    char_vocab = build_vocab(train_recs)
    label_map = build_label_map(LABELS)
    id2label = {v: k for k, v in label_map.items()}

    train_ds = NERDataset(train_recs, char_vocab, label_map)
    val_ds = NERDataset(val_recs, char_vocab, label_map)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = BiLSTMCRF(
        vocab_size=len(char_vocab), num_tags=len(label_map),
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        num_layers=args.layers, dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1 = -1.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def entity_f1(loader, recs):
        """实体级 P/R/F1（exact span+type match，弱标注验证集口径）。"""
        model.eval()
        tp = fp = fn = 0
        i = 0
        with torch.no_grad():
            for chars, tags, mask, ngrams in loader:
                paths = model.decode(chars.to(device), mask.to(device), ngrams.to(device))
                for b, path in enumerate(paths):
                    rec = recs[i]; i += 1
                    pred_tags = [id2label[j] for j in path]
                    pred_ents = set(extract_entities_from_tags(rec["chars"], pred_tags))
                    gold_ents = {(e["text"], e["type"]) for e in rec["entities"]}
                    tp += len(pred_ents & gold_ents)
                    fp += len(pred_ents - gold_ents)
                    fn += len(gold_ents - pred_ents)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, nb = 0.0, 0
        t0 = time.time()
        for it, (chars, tags, mask, ngrams) in enumerate(train_loader, 1):
            chars, tags, mask, ngrams = (x.to(device) for x in (chars, tags, mask, ngrams))
            loss = model.neg_log_likelihood(chars, tags, mask, ngrams)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item(); nb += 1
            if it % args.log_every == 0:
                print(f"epoch {epoch} it {it}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        sched.step()
        p, r, f1 = entity_f1(val_loader, val_recs)
        print(f"[epoch {epoch}] loss={total_loss / nb:.4f} val_P={p:.4f} val_R={r:.4f} val_F1={f1:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "state_dict": model.state_dict(),
                "char_vocab": char_vocab,
                "label_map": label_map,
                "config": {"embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim,
                           "num_layers": args.layers, "dropout": args.dropout},
                "val_f1": f1,
            }, args.out)
            print(f"  saved best -> {args.out}")

    print(f"done. best val entity F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
