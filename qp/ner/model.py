# -*- coding: utf-8 -*-
"""BiLSTM-CRF 序列标注模型（PyTorch，服务器/集群训练）。

设计要点：
  - 字符级建模：中英文 query 通吃（英文按字符切分亦可，词级特征由 BiLSTM 自学）
  - 可选 char 3-gram 哈希特征增强（类比 fastText，缓解 OOV）
  - CRF 解码保证 BIO 合法转移
  - 双语：模型层直接支持（数据对齐是数据侧问题，见 docs/03）

依赖：torch >= 2.0（仅服务器训练需要；本机推理走规则兜底，无 torch 亦可运行）
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_char_ngram: bool = True,
        ngram_hash_size: int = 50000,
        ngram_embed_dim: int = 50,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx

        self.char_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.use_ngram = use_char_ngram
        if use_char_ngram:
            self.ngram_embed = nn.Embedding(ngram_hash_size, ngram_embed_dim, padding_idx=0)
            lstm_in = embed_dim + ngram_embed_dim
        else:
            lstm_in = embed_dim

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            lstm_in, hidden_dim // 2, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # CRF 转移矩阵；START/END 用 num_tags / num_tags+1
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    # ---------------- 特征 ----------------
    def _emit(self, chars: torch.Tensor, ngrams: torch.Tensor = None) -> torch.Tensor:
        x = self.char_embed(chars)
        if self.use_ngram and ngrams is not None:
            x = torch.cat([x, self.ngram_embed(ngrams)], dim=-1)
        x = self.dropout(x)
        h, _ = self.lstm(x)
        return self.hidden2tag(self.dropout(h))

    # ---------------- CRF ----------------
    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log partition function，向量化实现。emissions: (B, T, K)"""
        B, T, K = emissions.shape
        score = self.start_transitions + emissions[:, 0]              # (B, K)
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)   # (B, K, K)
            next_score = broadcast + emissions[:, t].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)                   # (B, K)
            mask_t = mask[:, t].unsqueeze(1).float()
            score = torch.where(mask_t.bool(), next_score, score)
        return torch.logsumexp(score + self.end_transitions, dim=1)          # (B,)

    def _gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, K = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, torch.arange(B), tags[:, 0]]
        for t in range(1, T):
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[:, t, torch.arange(B), tags[:, t]]
            score = score + trans * mask[:, t] + emit * mask[:, t]
        lengths = mask.sum(1)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def neg_log_likelihood(self, chars, tags, mask, ngrams=None) -> torch.Tensor:
        emissions = self._emit(chars, ngrams)
        ll = self._forward_alg(emissions, mask) - self._gold_score(emissions, tags, mask)
        return ll.mean()

    @torch.no_grad()
    def decode(self, chars: torch.Tensor, mask: torch.Tensor, ngrams: torch.Tensor = None) -> List[List[int]]:
        """Viterbi 解码。返回 batch 内每条序列的 tag id 列表。"""
        emissions = self._emit(chars, ngrams)
        B, T, K = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        history: List[torch.Tensor] = []
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_prev = broadcast.max(dim=1)         # (B, K)
            next_score = best_score + emissions[:, t]
            mask_t = mask[:, t].unsqueeze(1).bool()
            score = torch.where(mask_t, next_score, score)
            history.append(best_prev)
        score = score + self.end_transitions
        best_last = score.argmax(dim=1)

        lengths = mask.sum(1).tolist()
        paths: List[List[int]] = []
        for b in range(B):
            L = int(lengths[b])
            tag = int(best_last[b])
            path = [tag]
            for t in range(L - 2, -1, -1):
                tag = int(history[t][b, path[-1]])
                path.append(tag)
            paths.append(path[::-1])
        return paths
