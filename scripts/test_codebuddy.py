# -*- coding: utf-8 -*-
"""直测 CodeBuddy 上游 /v2/chat/completions（ck_ key + CLI 指纹头 + SSE）。

key 从 configs/llm.yaml 的 agent 段读取（不入库），不在代码里硬编码。
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_key() -> str:
    try:
        import yaml
        with open(os.path.join(ROOT, "configs", "llm.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str((cfg.get("agent") or {}).get("api_key") or "")
    except Exception:
        return os.environ.get("CODEBUDDY_API_KEY", "")

KEY = _load_key()
URL = "https://www.codebuddy.cn/v2/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {KEY}",
    "X-API-Key": KEY,
    "X-Product": "SaaS",
    "X-IDE-Type": "CLI",
    "X-IDE-Name": "CLI",
    "X-IDE-Version": "2.118.2",
    "x-codebuddy-request": "1",
    "X-Agent-Intent": "craft",
    "User-Agent": "CLI/2.118.2 CodeBuddy/2.118.2 (Windows; x64)",
}

PROXY = "http://127.0.0.1:10808"
proxy_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
opener = urllib.request.build_opener(proxy_handler)


def ask(model: str = "hy3", use_proxy: bool = True) -> None:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个JSON决策器，只输出JSON。"},
            {"role": "user", "content": '用户问题：北方华创最新订单进展\n输出 {"next":"answer","sub_queries":[]} 格式'},
        ],
        "stream": True,
        "enable_thinking": False,   # 与 CodeBuddyClient 一致：压低思考开销
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers=HEADERS, method="POST")
    op = opener if use_proxy else urllib.request.build_opener()
    print(f"--- model={model} proxy={use_proxy} ---", flush=True)
    try:
        resp = op.open(req, timeout=60)
    except Exception as e:
        print(f"CONNECT FAIL: {type(e).__name__}: {e}", flush=True)
        return
    print(f"HTTP {resp.status}", flush=True)
    content, usage, finish = [], None, None
    try:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data_s = line[5:].strip()
            if data_s == "[DONE]":
                break
            try:
                chunk = json.loads(data_s)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
    except Exception as e:
        print(f"STREAM ERR: {type(e).__name__}: {e}", flush=True)
    text = "".join(content)
    print(f"finish={finish} usage={usage}", flush=True)
    print(f"text={text[:500]}", flush=True)


if __name__ == "__main__":
    use_proxy = "--noproxy" not in sys.argv
    ask(use_proxy=use_proxy)
