# -*- coding: utf-8 -*-
"""LLM 客户端（阿里云百炼 compatible-mode，OpenAI 兼容协议 + 原生 rerank）。

设计约束：
  - requests.Session 连接池（并发下复用 TLS 连接，避免每次调用重新握手）
  - 不走本地代理（阿里云北京端点直连，trust_env=False）
  - 所有调用带超时；失败返回 None/[]，调用方自行降级，绝不阻塞主链路
  - chat 默认关闭思考模式（enable_thinking=false），保证 content 即最终答案
"""
from __future__ import annotations

import json
import os
import re
import threading
from functools import lru_cache

import requests as _requests
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.path.join(ROOT, "configs", "llm.yaml")

NATIVE_HOST = "https://dashscope.aliyuncs.com"
NATIVE_RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"

# 进程级连接池：所有 LLM/embed/rerank 调用共享（并发安全）
_SESSION = _requests.Session()
_SESSION.trust_env = False   # 维持「不走本地代理」的设计约束
_adapter = _requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=32, max_retries=1)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)


def load_config(config_path: str | None = None) -> dict:
    path = config_path or _CONFIG_PATH
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f).get("llm", {})


class LLMClient:
    def __init__(self, config_path: str | None = None):
        self.config_path = config_path or _CONFIG_PATH
        self._cfg_mtime = self._mtime()
        self.cfg = load_config(self.config_path)
        self.enabled = bool(self.cfg)
        if not self.cfg:
            return
        self._apply(self.cfg)

    def _mtime(self) -> float:
        try:
            return os.path.getmtime(self.config_path)
        except OSError:
            return 0.0

    def _apply(self, cfg: dict):
        self.base = cfg["api_base"].rstrip("/")
        self.key = cfg["api_key"]
        self.timeout = int(cfg.get("timeout_s", 30))
        models = cfg.get("models", {})
        self.chat_model = models.get("chat")
        self.embed_model = models.get("embedding")
        self.rerank_model = models.get("rerank")
        self.switches = cfg.get("switches", {})

    def _maybe_reload(self):
        """配置文件被外部修改（如开关面板）后自动重载，切换即时生效。"""
        mtime = self._mtime()
        if mtime != self._cfg_mtime:
            self._cfg_mtime = mtime
            cfg = load_config(self.config_path)
            if cfg:
                self.enabled = True
                self._apply(cfg)

    def switch(self, name: str) -> bool:
        self._maybe_reload()
        return bool(self.cfg) and bool(self.switches.get(name, False))

    def set_switch(self, name: str, value: bool) -> bool:
        """运行时切换开关并落盘 llm.yaml（下一次调用即生效）。"""
        cfg = load_config(self.config_path)
        if not cfg:
            return False
        cfg.setdefault("switches", {})[name] = bool(value)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"llm": cfg}, f, allow_unicode=True, sort_keys=False)
        self._cfg_mtime = 0.0     # 强制下次 switch() 重载
        self._maybe_reload()
        return True

    # ------------------------------------------------------------------
    def _post(self, url: str, payload: dict, timeout: int | None = None):
        try:
            r = _SESSION.post(url, json=payload, timeout=timeout or self.timeout,
                              headers={"Authorization": f"Bearer {self.key}",
                                       "Content-Type": "application/json"})
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str):
        """从回复中提取 JSON（容忍 ```json 围栏与前后杂文字）。"""
        if not text:
            return None
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"[\[{][\s\S]*[\]}]", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def chat(self, prompt: str, system: str | None = None,
             max_tokens: int = 1024, temperature: float = 0.1) -> str | None:
        """对话补全（关闭思考），返回最终 content 文本；失败返回 None。"""
        messages = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": prompt}]
        r = self._post(self.base + "/chat/completions", {
            "model": self.chat_model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "enable_thinking": False,
        })
        if not r or "choices" not in r:
            return None
        return (r["choices"][0].get("message") or {}).get("content")

    def chat_json(self, prompt: str, system: str | None = None,
                  max_tokens: int = 1024):
        """对话补全并解析 JSON；失败返回 None。"""
        return self._extract_json(self.chat(prompt, system, max_tokens))

    # ------------------------------------------------------------------
    def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]] | None:
        """批量向量化（自动分批，服务端对单批条数有上限）；失败返回 None。"""
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = [t[:1024] for t in texts[i:i + batch_size]]
            r = self._post(self.base + "/embeddings",
                           {"model": self.embed_model, "input": chunk})
            if not r or "data" not in r:
                return None
            data = sorted(r["data"], key=lambda x: x["index"])
            out.extend(d["embedding"] for d in data)
        return out

    # ------------------------------------------------------------------
    def rerank(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        """重排，返回 [(原索引, 相关性得分)]（按得分降序）；失败返回 []。

        端点：兼容模式无 /rerank（实测 404），走原生 dashscope host。
        """
        if not documents:
            return []
        url = NATIVE_HOST + NATIVE_RERANK_PATH
        r = self._post(url, {"model": self.rerank_model,
                             "input": {"query": query, "documents": documents}})
        if not r:
            return []
        try:
            results = r["output"]["results"]
            return [(int(x["index"]), float(x["relevance_score"])) for x in results]
        except (KeyError, TypeError, ValueError):
            return []


@lru_cache(maxsize=1)
def default_client() -> LLMClient:
    return LLMClient()
