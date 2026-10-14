# -*- coding: utf-8 -*-
"""LLM 连通性自检：chat / embeddings / rerank 三接口各打一发，打印可用性。"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import yaml

with open(os.path.join(ROOT, "configs", "llm.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["llm"]

BASE = cfg["api_base"].rstrip("/")
KEY = cfg["api_key"]
MODELS = cfg["models"]


def post(path, payload, timeout=30):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")[:300]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


print("== chat =", MODELS["chat"])
s, r = post("/chat/completions", {
    "model": MODELS["chat"], "max_tokens": 512, "enable_thinking": False,
    "messages": [{"role": "user", "content": "只回复 JSON: {\"ok\": true}"}]})
print("  status:", s)
print("  resp:", json.dumps(r, ensure_ascii=False)[:300] if isinstance(r, dict)
      else str(r)[:300])

print("== embeddings =", MODELS["embedding"])
s, r = post("/embeddings", {"model": MODELS["embedding"], "input": ["智谱最新公告", "半导体设备"]})
if isinstance(r, dict) and "data" in r:
    print("  status:", s, "| dims:", len(r["data"][0]["embedding"]), "| n:", len(r["data"]))
else:
    print("  status:", s, "| resp:", str(r)[:300])

print("== rerank =", MODELS["rerank"], "at", cfg.get("rerank_path", "/rerank"))
s, r = post(cfg.get("rerank_path", "/rerank"), {
    "model": MODELS["rerank"], "query": "智谱最新公告",
    "documents": ["智谱拟配售新H股募资314亿港元", "白酒板块今日资金流向", "半导体设备国产替代进展"]})
print("  status:", s)
print("  resp:", json.dumps(r, ensure_ascii=False)[:400] if isinstance(r, dict) else str(r)[:400])
