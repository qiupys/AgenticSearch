# -*- coding: utf-8 -*-
"""corpus —— 语料/索引/chunk 基础设施（召回与排序模块的共享数据契约）。

职责边界：
  - 只负责「数据长什么样、从哪来」：corpus.jsonl / index.pkl / chunks.jsonl /
    chunk_embeddings.npz 的加载与增量维护
  - 不做任何打分、过滤、排序——那是召回/排序模块的事

数据契约（召回与排序两模块共同依赖）：
  corpus : list[dict]   每条至少含 id/title/snippet/url/content_type/sentiment/
                         linked_entities/date
  idx    : dict         {"bm25": BM25Okapi, "doc_ids": [...]}
  chunks : list[dict]   {chunk_id, doc_id, seq, text, linked_entities}
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # content_search/ -> 项目根
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONTENT_DIR = os.path.join(ROOT, "data", "content")
_CHUNKS_PATH = os.path.join(CONTENT_DIR, "chunks.jsonl")
_CHUNK_EMB_PATH = os.path.join(CONTENT_DIR, "chunk_embeddings.npz")
_CHUNK_LOCK = threading.Lock()


def load_all() -> tuple:
    """加载语料 + BM25 索引，并初始化 jieba 用户词典。

    自检：bm25 索引文档数与 corpus 不一致（多进程同时入库 / 重建竞态导致
    index.pkl 落后于 corpus.jsonl）时当场重建索引，避免召回层 shape 错乱。
    """
    import jieba
    with open(os.path.join(CONTENT_DIR, "corpus.jsonl"), encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(CONTENT_DIR, "index.pkl"), "rb") as f:
        idx = pickle.load(f)
    n_bm25 = len(getattr(idx.get("bm25"), "doc_freqs", ()) or ())
    if n_bm25 != len(corpus):
        from tools.build_content_library import build_index
        build_index(corpus)   # 自愈重建（~500 docs 秒级）
        with open(os.path.join(CONTENT_DIR, "index.pkl"), "rb") as f:
            idx = pickle.load(f)
    dict_path = os.path.join(ROOT, "data", "dict", "jieba_user_dict.txt")
    if os.path.exists(dict_path):
        jieba.load_userdict(dict_path)
    jieba.initialize()
    return corpus, idx


# ---------------- chunk 化存储 ----------------

def split_chunks(text: str, max_len: int = 280, overlap: int = 50) -> list:
    """句边界优先切块，带 overlap；超长句硬切。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            cuts = [text.rfind(p, start, end) for p in "。！？；\n"]
            m = max(cuts)
            if m > start + max_len // 2:
                end = m + 1
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return [c.strip() for c in chunks if c.strip()]


def load_or_build_chunks(corpus: list) -> list:
    """chunks.jsonl 与 corpus 的 doc-id 集一致则复用；只增文档时增量补新 doc
    （避免每次兜底入库后全量重切+全量 tag_doc 拖垮在线请求）；文档被删才全量重建。"""
    with _CHUNK_LOCK:
        doc_ids = [d["id"] for d in corpus]
        want = set(doc_ids)
        if os.path.exists(_CHUNKS_PATH):
            try:
                with open(_CHUNKS_PATH, encoding="utf-8") as f:
                    chunks = [json.loads(line) for line in f if line.strip()]
                have = {c["doc_id"] for c in chunks}
                if have == want:
                    return chunks
                if have < want:
                    # 增量：旧 chunk 原样复用，只为新 doc 切块打标
                    from qp.ner.entity_linker import default_linker
                    linker = default_linker()
                    have_doc = set(have)
                    for d in corpus:
                        if d["id"] in have_doc:
                            continue
                        body = f"{d.get('title', '')}。{d.get('snippet', '')}"
                        if d.get("body"):
                            body = f"{d.get('title', '')}。{d['body']}"
                        for seq, ctext in enumerate(split_chunks(body)):
                            chunks.append({"chunk_id": f"{d['id']}_c{seq:02d}",
                                           "doc_id": d["id"], "seq": seq,
                                           "text": ctext,
                                           "linked_entities": linker.tag_doc(ctext)})
                    with open(_CHUNKS_PATH, "w", encoding="utf-8") as f:
                        for c in chunks:
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    return chunks
                # have 与 want 有删减差异 -> 走全量重建
            except Exception:
                pass
        from qp.ner.entity_linker import default_linker
        linker = default_linker()
        chunks = []
        for d in corpus:
            body = f"{d.get('title', '')}。{d.get('snippet', '')}"
            if d.get("body"):
                body = f"{d.get('title', '')}。{d['body']}"
            for seq, ctext in enumerate(split_chunks(body)):
                chunks.append({"chunk_id": f"{d['id']}_c{seq:02d}", "doc_id": d["id"],
                               "seq": seq, "text": ctext,
                               "linked_entities": linker.tag_doc(ctext)})
        with open(_CHUNKS_PATH, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        return chunks


def load_chunk_emb_cache() -> tuple:
    if os.path.exists(_CHUNK_EMB_PATH):
        try:
            z = np.load(_CHUNK_EMB_PATH, allow_pickle=False)
            return {str(x): i for i, x in enumerate(z["ids"])}, z["matrix"]
        except Exception:
            pass
    return {}, None


def chunk_embedding_scores(query: str, chunks: list, client) -> np.ndarray | None:
    """query 与全部 chunk 的余弦相似度（缓存增量补齐）。失败返回 None。"""
    with _CHUNK_LOCK:
        cache, matrix = load_chunk_emb_cache()
        missing = [c for c in chunks if c["chunk_id"] not in cache]
        if missing:
            vecs = client.embed([c["text"][:1024] for c in missing])
            if vecs is None:
                return None
            mat = np.array(vecs, dtype=np.float32)
            matrix = mat if matrix is None else np.vstack([matrix, mat])
            for c, v in zip(missing, vecs):
                cache[c["chunk_id"]] = len(cache)
            np.savez(_CHUNK_EMB_PATH, ids=np.array(list(cache.keys())),
                     matrix=matrix.astype(np.float32))
        if matrix is None or not len(cache):
            return None
        qv = client.embed([query])
        if qv is None:
            return None
        q = np.array(qv[0], dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-9)
        mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        cscores = mn @ qn
        out = np.full(len(chunks), -1.0, dtype=np.float32)
        for i, c in enumerate(chunks):
            j = cache.get(c["chunk_id"])
            if j is not None:
                out[i] = cscores[j]
        return out
