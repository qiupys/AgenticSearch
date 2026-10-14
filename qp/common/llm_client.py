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


def load_config(config_path: str | None = None, section: str = "llm") -> dict:
    path = config_path or _CONFIG_PATH
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f).get(section, {}) or {}


class LLMClient:
    """LLM 客户端（支持多 profile：section="llm" 主链路 / "agent" 等）。

    不同 profile 可指向不同供应商（如 agent 路走 CodeBuddy TokenHub 免费模型）。
    """

    def __init__(self, config_path: str | None = None, section: str = "llm"):
        self.config_path = config_path or _CONFIG_PATH
        self.section = section
        self._cfg_mtime = self._mtime()
        self.cfg = load_config(self.config_path, section)
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
            cfg = load_config(self.config_path, self.section)
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
    def _repair_truncated(text: str):
        """截断 JSON 容错：字符串感知地补全未闭合的引号/括号再解析。

        思考模型常把 max_tokens 耗在 reasoning 上导致 content 被截断
        （finish=length），此时返回部分可用结果远好于整体丢弃。
        """
        stack: list = []
        in_str = esc = False
        for ch in text:
            if esc:
                esc = False
                continue
            if in_str:
                if ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]" and stack and stack[-1] + ch in ("{}", "[]"):
                stack.pop()
        repaired = text + ('"' if in_str else "") + \
            "".join("}" if c == "{" else "]" for c in reversed(stack))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # 最后兜底：直接提取 "answer"/"content" 等字符串字段的已完成部分
            m = re.search(r'"(answer|content|text)"\s*:\s*"((?:[^"\\]|\\.)*)', text)
            if m:
                try:
                    return {m.group(1): json.loads('"' + m.group(2) + '"')}
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _extract_json(text: str):
        """从回复中提取 JSON（容忍 ```json 围栏、前后杂文字与截断）。"""
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
                pass
        return LLMClient._repair_truncated(text)

    def chat(self, prompt: str, system: str | None = None,
             max_tokens: int = 1024, temperature: float = 0.1,
             timeout: float | None = None) -> str | None:
        """对话补全（关闭思考），返回最终 content 文本；失败返回 None。"""
        messages = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": prompt}]
        r = self._post(self.base + "/chat/completions", {
            "model": self.chat_model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "enable_thinking": False,
        }, timeout=timeout)
        if not r or "choices" not in r:
            return None
        return (r["choices"][0].get("message") or {}).get("content")

    def chat_json(self, prompt: str, system: str | None = None,
                  max_tokens: int = 1024, timeout: float | None = None):
        """对话补全并解析 JSON；失败返回 None。"""
        return self._extract_json(self.chat(prompt, system, max_tokens,
                                            timeout=timeout))

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


# ---------------------------------------------------------------------------
# CodeBuddy 上游（www.codebuddy.cn/v2/chat/completions，仅免费模型如 hy3）
#
# 协议要点（经 codebuddy-cli2api 抓包确认）：
#   - 强制 stream:true（非流式返回 400），SSE chunk 为标准 OpenAI delta 格式
#   - 认证：Authorization Bearer + X-API-Key 双头，附 CLI 指纹头
#   - ck_ 前缀 key 只在 .cn 域有效（.ai 域返回 401 not_found）
# ---------------------------------------------------------------------------

_CODEBUDDY_IDE_VERSION = "2.118.2"


class CodeBuddyClient:
    """CodeBuddy CLI 上游客户端（chat/chat_json 与 LLMClient 鸭子类型兼容）。"""

    def __init__(self, config_path: str | None = None, section: str = "agent"):
        cfg = load_config(config_path or _CONFIG_PATH, section)
        self.cfg = cfg
        self.enabled = bool(cfg.get("api_key")) and bool((cfg.get("models") or {}).get("chat"))
        if not self.enabled:
            return
        self.base = cfg["api_base"].rstrip("/")
        self.key = cfg["api_key"]
        self.chat_model = cfg["models"]["chat"]
        self.timeout = int(cfg.get("timeout_s", 45))
        self._session = _requests.Session()
        self._session.trust_env = False  # codebuddy.cn 国内直连，不走本地代理

    def _headers(self) -> dict:
        v = _CODEBUDDY_IDE_VERSION
        return {
            "Authorization": f"Bearer {self.key}",
            "X-API-Key": self.key,
            "X-Product": "SaaS",
            "X-IDE-Type": "CLI",
            "X-IDE-Name": "CLI",
            "X-IDE-Version": v,
            "x-codebuddy-request": "1",
            "X-Agent-Intent": "craft",
            "User-Agent": f"CLI/{v} CodeBuddy/{v} (Windows; x64)",
            "Content-Type": "application/json",
        }

    def chat(self, prompt: str, system: str | None = None,
             max_tokens: int = 1024, temperature: float = 0.1,
             timeout: float | None = None) -> str | None:
        """SSE 流式请求并聚合 content；失败重试 1 次，仍失败返回 None。

        timeout：本次调用超时上限（秒），缺省用配置值；首次已耗掉大部分
        时间时不再重试（快速失败，交给调用方兜底）。
        失败/截断主因（实测）：思考模型 reasoning 计入 max_tokens，可能耗尽
        额度导致 content 为空（finish=length）或答案中途截断——重试时放大
        max_tokens，且"非空但截断"同样触发重试以拿到完整答案。
        """
        import time as _t
        t0 = _t.perf_counter()
        limit = float(timeout) if timeout else float(self.timeout)
        for attempt in range(2):
            if attempt and _t.perf_counter() - t0 > limit * 0.9:
                return None   # 首次已耗掉大部分时间，重试也必超时
            mt = min(max_tokens * (2 if attempt else 1), 4000)
            out, finish = self._chat_once(prompt, system, mt, temperature, limit)
            if out is not None and (finish != "length" or attempt == 1):
                return out    # 正常完成；或重试后仍截断——返回部分内容由调用方兜底
            if attempt == 0:
                _t.sleep(0.8)   # 空内容/截断/偶发失败：短退避后加倍额度重试
        return None

    def _chat_once(self, prompt: str, system: str | None,
                   max_tokens: int, temperature: float,
                   timeout: float) -> tuple[str | None, str | None]:
        """单次 SSE 流式请求。返回 (content 或 None, finish_reason)。"""
        messages = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": prompt}]
        payload = {
            "model": self.chat_model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "enable_thinking": False,   # hy3 等思考模型：压低思考开销（实测 ~50%）
            "stream": True, "stream_options": {"include_usage": True},
        }
        content: list[str] = []
        finish: str | None = None
        try:
            r = self._session.post(self.base + "/v2/chat/completions",
                                   json=payload, headers=self._headers(),
                                   timeout=timeout, stream=True)
            if r.status_code != 200:
                return None, None
            for raw in r.iter_lines(decode_unicode=False):
                line = (raw or b"").decode("utf-8", "replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    if choices[0].get("finish_reason"):
                        finish = choices[0]["finish_reason"]
            r.close()
        except Exception:
            return None, None
        return ("".join(content) or None), finish

    def chat_json(self, prompt: str, system: str | None = None,
                  max_tokens: int = 1024, timeout: float | None = None):
        """对话补全并解析 JSON；失败返回 None。"""
        return LLMClient._extract_json(
            self.chat(prompt, system, max_tokens, timeout=timeout))


@lru_cache(maxsize=1)
def default_client() -> LLMClient:
    return LLMClient()
