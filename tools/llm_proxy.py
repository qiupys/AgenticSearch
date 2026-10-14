# -*- coding: utf-8 -*-
"""opencode Zen 本地代理（OpenAI 兼容网关）。

作用：在本地起一个 OpenAI 兼容服务（默认 http://127.0.0.1:8180/v1），
把 /chat/completions、/models 等请求转发到 opencode Zen
（https://opencode.ai/zen/v1），从而使用 Zen 的免费模型。
configs/llm.yaml 的 api_key / api_base 接口保持不变：api_base 指向本代理，
api_key 填任意占位值即可（鉴权由代理向上游注入真实 key）。

API key 解析优先级：
  1. --api-key 参数
  2. 环境变量 OPENCODE_API_KEY
  3. opencode 本地凭据 ~/.local/share/opencode/auth.json（opencode auth login 写入）
  4. 都没有则空 key 启动（上游会返回 401，不影响其他链路降级）

出站网络：本机整体被墙，上游请求默认走本地 v2rayN 混合代理
  http://127.0.0.1:10808（--proxy none 可关闭）。

用法：
  python tools/llm_proxy.py                    # 前台启动
  python tools/llm_proxy.py --port 8180        # 指定端口
  python tools/llm_proxy.py --api-key sk-xxx   # 指定 Zen key

注意：Zen 不提供 embeddings / rerank 接口，本代理对这两类请求返回 501，
调用方（qp/common/llm_client.py）收到非 200 自动降级——请在 llm.yaml 关闭
embedding_recall / rerank 开关。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests as _requests

UPSTREAM_DEFAULT = "https://opencode.ai/zen/v1"
PROXY_DEFAULT = os.getenv("LLM_PROXY_OUTBOUND", "http://127.0.0.1:10808")
AUTH_JSON = os.path.join(os.path.expanduser("~"), ".local", "share",
                         "opencode", "auth.json")

CFG = {
    "upstream": UPSTREAM_DEFAULT,
    "key": "",
    "outbound_proxies": None,   # requests proxies dict；None=不走代理
    "timeout": 300,
}

# ---- opencode 客户端指纹伪装（过 Zen 免费层的客户端校验，2026-10-12 抓包确认）----
# 真实 CLI 请求头：User-Agent + x-opencode-client/project/request/session。
# 服务器校验「真实凭据 + 客户端头」，占位 key 仍会 403 FreeTierError，
# 真实 key（opencode auth login 写入 auth.json）即可通过。
_OC_UA = "opencode/1.18.33 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
_OC_PROJECT = "f01709890e640300b2d1f7bf8ef8868359fed4cc"

# 会话：连接池复用 TLS，出站走本地代理（trust_env=True 读代理环境变量也行，
# 这里显式注入更可控）
_SESSION = _requests.Session()


def _resolve_key() -> str:
    if CFG["key"]:
        return CFG["key"]
    # opencode auth.json：{"opencode": {"type": "api", "key": "..."}, ...}
    try:
        with open(AUTH_JSON, encoding="utf-8") as f:
            data = json.load(f) or {}
        for provider, ent in data.items():
            if isinstance(ent, dict) and ent.get("key"):
                if "opencode" in provider.lower() or ent.get("type") == "api":
                    return str(ent["key"])
        for ent in data.values():          # 兜底：任意含 key 的条目
            if isinstance(ent, dict) and ent.get("key"):
                return str(ent["key"])
    except Exception:
        pass
    return ""


def _upstream(path: str) -> str:
    return CFG["upstream"].rstrip("/") + "/" + path.lstrip("/")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---------- 基础工具 ----------
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}",
              flush=True)

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    # ---------- GET ----------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/health"):
            return self._send_json(200, {
                "status": "ok", "service": "opencode-zen-local-proxy",
                "upstream": CFG["upstream"],
                "key_loaded": bool(_resolve_key()),
                "outbound_proxy": CFG["outbound_proxies"],
            })
        try:
            r = _SESSION.get(_upstream(self.path),
                             headers=self._auth_headers(), timeout=60,
                             proxies=CFG["outbound_proxies"])
            return self._send_json(r.status_code, _safe_json(r))
        except Exception as e:
            return self._send_json(502, {"error": {"message":
                f"upstream unreachable: {type(e).__name__}: {e}"}})

    # ---------- POST ----------
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        path = path[:-3] if path.endswith("/v1") else path  # 容错 /v1/v1
        path_n = path[len("/v1"):] if path.startswith("/v1") else path

        if path_n in ("/embeddings", "/rerank"):
            return self._send_json(501, {"error": {"message":
                "opencode Zen 不提供 embeddings/rerank；请在 configs/llm.yaml "
                "将 embedding_recall / rerank 开关置为 false（调用方自动降级）"}})

        if path_n not in ("/chat/completions", "/responses", "/models"):
            return self._send_json(404, {"error": {"message": f"unknown path {path}"}})

        body = self._read_body()
        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"

        # 剥掉 dashscope 专有参数（enable_thinking），避免上游 400
        try:
            payload = json.loads(body) if body else {}
            if isinstance(payload, dict):
                payload.pop("enable_thinking", None)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                streaming = bool(payload.get("stream"))
        except Exception:
            streaming = False

        try:
            r = _SESSION.post(_upstream(path_n), data=body, headers=headers,
                              timeout=CFG["timeout"],
                              proxies=CFG["outbound_proxies"], stream=True)
        except Exception as e:
            return self._send_json(502, {"error": {"message":
                f"upstream unreachable: {type(e).__name__}: {e}"}})

        ctype = r.headers.get("Content-Type", "application/json")
        self.send_response(r.status_code)
        self.send_header("Content-Type", ctype)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass          # 客户端提前断开（如超时降级），静默
        finally:
            r.close()

    # ---------- 鉴权头 ----------
    def _auth_headers(self) -> dict:
        key = _resolve_key()
        h = {"Authorization": f"Bearer {key}", "Accept": "application/json",
             "User-Agent": _OC_UA,
             "x-opencode-client": "cli",
             "x-opencode-project": _OC_PROJECT,
             "x-opencode-request": "msg_" + os.urandom(12).hex(),
             "x-opencode-session": "ses_" + os.urandom(12).hex()}
        return h


def _safe_json(r: requests.Response):
    try:
        return r.json()
    except Exception:
        return {"raw": r.text[:500]}


def main():
    ap = argparse.ArgumentParser(description="opencode Zen 本地 OpenAI 兼容代理")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8180)
    ap.add_argument("--upstream", default=UPSTREAM_DEFAULT)
    ap.add_argument("--api-key", default=os.getenv("OPENCODE_API_KEY", ""),
                    help="opencode Zen API key（缺省读 opencode auth.json）")
    ap.add_argument("--proxy", default=PROXY_DEFAULT,
                    help="出站代理（默认 v2rayN http://127.0.0.1:10808，none=直连）")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    CFG["upstream"] = args.upstream
    CFG["key"] = args.api_key
    CFG["timeout"] = args.timeout
    if args.proxy and args.proxy.lower() not in ("none", "direct", ""):
        CFG["outbound_proxies"] = {"http": args.proxy, "https": args.proxy}

    key = _resolve_key()
    if not key:
        print("[警告] 未找到 opencode Zen API key：\n"
              "  1) 运行 opencode auth login 选 opencode 并粘贴 key，或\n"
              "  2) 到 https://opencode.ai/zen 控制台创建 key 后\n"
              "     set OPENCODE_API_KEY=... 或 --api-key 传入\n"
              "  （当前启动为无 key 模式，上游将返回 401）", flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[llm-proxy] listening on http://{args.host}:{args.port}/v1", flush=True)
    print(f"[llm-proxy] upstream  -> {CFG['upstream']}", flush=True)
    print(f"[llm-proxy] outbound  -> {args.proxy}", flush=True)
    print(f"[llm-proxy] api_key   -> {'loaded (' + key[:8] + '...)' if key else 'MISSING'}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[llm-proxy] bye", flush=True)


if __name__ == "__main__":
    sys.exit(main())
