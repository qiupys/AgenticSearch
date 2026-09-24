# -*- coding: utf-8 -*-
"""独立搜索引擎后端（不经 ddgs 库），作为 ddgs 多后端全部失败时的兜底。

- ddg_html  : DuckDuckGo HTML 版（html.duckduckgo.com），与 ddgs 内置的 API 通道
              完全独立，免 key，实测稳定返回中文财经结果（首选兜底）
- bing_html : Bing 网页版直接抓取（免 key；部分代理出口会被反爬干扰，作次选）
- brave_api : Brave Search 官方 API（需在 service.yaml 配 content.brave_api_key，可选）

返回结构对齐 ddgs：[{title, href, body}, ...]
"""
from __future__ import annotations

import re
import urllib.parse
from urllib.parse import unquote

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def _proxies(proxy: str | None):
    return {"http": proxy, "https": proxy} if proxy else None


def ddg_html_search(query: str, proxy: str | None = None, max_results: int = 8,
                    timeout: int = 10, timelimit: str | None = None) -> list:
    """DuckDuckGo HTML 版。timelimit 暂不支持（HTML 版无稳定时效参数）。"""
    import requests

    resp = requests.post("https://html.duckduckgo.com/html/",
                         data={"q": query, "kl": "cn-zh"},
                         headers={"User-Agent": _UA,
                                  "Accept-Language": "zh-CN,zh;q=0.9"},
                         timeout=timeout, proxies=_proxies(proxy))
    resp.raise_for_status()
    out = []
    for href, title in re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            resp.text, re.S)[:max_results * 2]:
        m = re.search(r"uddg=([^&]+)", href)
        url = unquote(m.group(1)) if m else href
        if not url.startswith("http"):
            continue
        snippet = ""
        a = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.S)
        if a:
            snippet = re.sub(r"<[^>]+>", "", a.group(1))
        out.append({"title": re.sub(r"<[^>]+>", "", title).strip(),
                    "href": url.strip(), "body": snippet.strip()})
        if len(out) >= max_results:
            break
    return out


def bing_html_search(query: str, proxy: str | None = None, max_results: int = 8,
                     timeout: int = 10, timelimit: str | None = None) -> list:
    """Bing 网页版抓取。timelimit: d/w/m（部分代理出口会被反爬干扰，结果需校验）。"""
    import requests

    params = {"q": query, "count": str(max_results), "mkt": "zh-CN"}
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(params)
    resp = requests.get(url, headers={"User-Agent": _UA,
                                      "Accept-Language": "zh-CN,zh;q=0.9"},
                        timeout=timeout, proxies=_proxies(proxy))
    resp.raise_for_status()
    out = []
    for m in re.finditer(r'<li class="b_algo"[^>]*>(.*?)</li>', resp.text, re.S):
        block = m.group(1)
        a = re.search(r'<h2[^>]*>\s*<a[^>]*href="(http[^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not a:
            continue
        cap = re.search(r'<p[^>]*>(.*?)</p>', block, re.S)
        out.append({"title": re.sub(r"<[^>]+>", "", a.group(2)).strip(),
                    "href": a.group(1).strip(),
                    "body": re.sub(r"<[^>]+>", "", cap.group(1)).strip() if cap else ""})
        if len(out) >= max_results:
            break
    return out


def brave_api_search(query: str, proxy: str | None = None, max_results: int = 8,
                     timeout: int = 10, api_key: str = "",
                     timelimit: str | None = None) -> list:
    """Brave Search 官方 API。未配置 key 时返回空（由调用方跳过）。"""
    if not api_key:
        return []
    import requests

    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query, "count": str(max_results)}
    if timelimit == "d":
        params["freshness"] = "pd"
    elif timelimit == "w":
        params["freshness"] = "pw"
    elif timelimit == "m":
        params["freshness"] = "pm"
    resp = requests.get("https://api.search.brave.com/res/v1/web/search",
                        headers=headers, params=params, timeout=timeout,
                        proxies=_proxies(proxy))
    resp.raise_for_status()
    web = (resp.json().get("web") or {}).get("results", [])
    return [{"title": r.get("title", ""), "href": r.get("url", ""),
             "body": r.get("description", "")} for r in web[:max_results]]
