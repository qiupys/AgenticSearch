# -*- coding: utf-8 -*-
"""构建证券主数据词表 + 金融指标词表，并合并进实体链指词表。

产出：
  1. data/entities/stock_master.jsonl   证券主数据（分市场）
     {"canonical", "abbr", "full_name", "markets": [...], "codes": {...}, "aliases": [...]}
     同一公司多地上市（阿里巴巴 HK+US）合并为一条，codes 按市场分列。
  2. data/entities/indicator_lexicon.jsonl  金融指标/术语词表（手工种子）
     {"name", "aliases", "category"}
  3. 合并进 data/entities/gazetteer.jsonl：
     - 股票 -> type=STOCK，canonical=公司全称（可得时）否则简称
     - 指标 -> type=MACRO（沿用现有 NER 标签集），category 存 extra 字段
     - 与既有条目按 norm 冲突合并（种子 eid 不裂变）

数据源：
  - A股：上交所/深交所/北交所官网名单（akshare，上交所含公司全称）
  - 港股/美股：东方财富 clist（多 host 重试；baostock 不覆盖这两个市场）
  - 指标：手工种子词表

用法：
  python scripts/build_stock_lexicon.py [--skip-fetch]   # --skip-fetch 只做合并/指标
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time

import requests

# 本机沙箱常有全局代理环境变量，境内源直连即可（北交所 bse.cn 走代理会被拒）
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENT_DIR = os.path.join(ROOT, "data", "entities")
GAZ_PATH = os.path.join(ENT_DIR, "gazetteer.jsonl")
MASTER_PATH = os.path.join(ENT_DIR, "stock_master.jsonl")
IND_PATH = os.path.join(ENT_DIR, "indicator_lexicon.jsonl")

# 与 ner/entity_linker.py 保持一致的归一化（除公司后缀外额外去市场标记后缀）
_SUFFIX = re.compile(r"(股份|集团|科技|控股|有限公司|公司)$")
_MKT_SUFFIX = re.compile(r"[-－][A-Z]{1,3}$", re.I)   # -W/-SW/-WR/-R 等港股市场标记
_ST_PREFIX = re.compile(r"^(\*?ST|\*?st)")
_FULLWIDTH = str.maketrans("０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def norm_name(s: str) -> str:
    """主数据归一化：全角->半角、小写、去空白、去 ST 前缀、去市场标记后缀、去公司后缀。"""
    if not s:
        return ""
    t = str(s).translate(_FULLWIDTH).strip().lower()
    t = re.sub(r"\s+", "", t)
    t = _MKT_SUFFIX.sub("", t)
    t = _ST_PREFIX.sub("", t)
    t = _SUFFIX.sub("", t)
    return t


def clean_abbr(s: str) -> str:
    """证券简称清洗：全角->半角、去内部空格（如「万  科Ａ」->「万科A」）。"""
    if not s:
        return ""
    return re.sub(r"\s+", "", str(s).translate(_FULLWIDTH).strip())


# ---------------------------------------------------------------------------
# 抓取
# ---------------------------------------------------------------------------
def fetch_a_shares() -> list[dict]:
    """A股：交易所官网名单。SH 含公司全称；SZ/BJ 仅简称。"""
    import akshare as ak
    rows: dict[str, dict] = {}

    def put(code, abbr, full, exch):
        code, abbr = clean_abbr(code), clean_abbr(abbr)
        if not re.fullmatch(r"\d{6}", code or ""):
            return
        rows[code] = {"code": code, "abbr": abbr,
                      "full_name": clean_abbr(full) if full else "",
                      "exchange": exch}

    for sym in ("主板A股", "科创板"):
        df = ak.stock_info_sh_name_code(symbol=sym)
        for _, r in df.iterrows():
            put(r["证券代码"], r["证券简称"], r.get("公司全称") or "", "SH")
    try:
        df = ak.stock_info_sz_name_code(symbol="A股列表")
        for _, r in df.iterrows():
            put(r["A股代码"], r["A股简称"], "", "SZ")
    except Exception as e:
        print(f"WARN SZ failed: {e}")
    try:
        df = ak.stock_info_bj_name_code()
        code_col = next(c for c in df.columns if "代码" in c)
        name_col = next(c for c in df.columns if "简称" in c)
        for _, r in df.iterrows():
            put(r[code_col], r[name_col], "", "BJ")
    except Exception as e:
        print(f"WARN BJ failed: {e}")
    out = sorted(rows.values(), key=lambda x: x["code"])
    print(f"A股: {len(out)} 条（SH {sum(1 for r in out if r['full_name'])} 条含全称）")
    return out


_EM_HOSTS = [f"{i}.push2.eastmoney.com" for i in
             (1, 5, 10, 20, 33, 48, 70, 82, 90)] + ["push2.eastmoney.com"]
_EM_H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
         "Referer": "https://quote.eastmoney.com/"}


def _em_clist(fs: str, expect_total: int = 30000, attempts: int = 24) -> list[dict]:
    """东财 clist（分页抓全量）。本机对该域名链路不稳定：每页多 host 轮询重试。

    服务端 pz 上限 100，按 pn 分页直到取满 total。po=0 升序（正股代码在前，
    窝轮/牛熊证等 9xxxx 大码在后，弱网下优先拿到正股）。
    """
    out: dict[str, dict] = {}
    hosts = _EM_HOSTS[:]
    random.shuffle(hosts)
    pn, total, fail = 1, None, 0
    while pn <= (total or expect_total // 100 + 1):
        host = hosts[(pn + fail) % len(hosts)]
        try:
            r = requests.get(f"https://{host}/api/qt/clist/get",
                             params={"pn": pn, "pz": 100, "po": 0, "fid": "f12",
                                     "fs": fs, "fields": "f12,f13,f14"},
                             headers=_EM_H, timeout=6,
                             proxies={"http": None, "https": None})
            data = (r.json() or {}).get("data") or {}
            diff = data.get("diff") or []
            if isinstance(diff, dict):
                diff = list(diff.values())
            total = int(data.get("total") or total or 0)
            if not diff:
                raise ValueError("empty page")
            for x in diff:
                if x.get("f12") and x.get("f14"):
                    out[str(x["f12"])] = {"code": str(x["f12"]), "abbr": str(x["f14"])}
            if len(out) >= total:
                break
            pn += 1
            fail = 0
            time.sleep(0.15)
        except Exception:
            fail += 1
            if fail >= attempts:
                break
            time.sleep(1.0)
    return list(out.values())


def fetch_hkex(xlsx_en: str, xlsx_cn: str) -> list[dict]:
    """港交所官方 List of Securities（股本证券全量，含官方中文名）。东财港股的替代主源。"""
    import openpyxl
    from zhconv import convert

    def _dl(url: str, path: str):
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60,
                         proxies={"http": None, "https": None})
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    try:
        _dl("https://www.hkex.com.hk/chi/services/trading/securities/securitieslists/ListOfSecurities_c.xlsx",
            xlsx_cn)
        _dl("https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx",
            xlsx_en)
    except Exception as e:
        if not (os.path.exists(xlsx_cn) and os.path.exists(xlsx_en)):
            print(f"WARN hkex download failed and no cache: {e}")
            return []

    names_en: dict[str, str] = {}
    wb = openpyxl.load_workbook(xlsx_en, read_only=True)
    ws = wb[wb.sheetnames[0]]
    ws.reset_dimensions()
    for r in ws.iter_rows(values_only=True):
        if r and r[0] is not None and r[1]:
            names_en[str(r[0]).strip().zfill(5)] = str(r[1]).strip()

    out: dict[str, dict] = {}
    wb = openpyxl.load_workbook(xlsx_cn, read_only=True)
    ws = wb[wb.sheetnames[0]]
    ws.reset_dimensions()
    for r in ws.iter_rows(values_only=True):
        # 列：股份代號/股份名稱/分類/次分類 —— 只取「股本」（正股），排除窝轮/牛熊证/债券/ETF
        if not r or r[0] is None or str(r[2] or "").strip() != "股本":
            continue
        code = str(r[0]).strip().zfill(5)
        name = convert(str(r[1] or "").strip(), "zh-cn")
        if not name:
            continue
        out[code] = {"code": code, "abbr": name,
                     "full_name": names_en.get(code, "")}
    rows = list(out.values())
    print(f"港股(HKEX官方股本): {len(rows)} 条")
    return rows


_US_NAME_TAIL = re.compile(
    r"\s*(Common Stock|Ordinary Shares|Class [A-Z]( Ordinary Shares| Common Stock)?"
    r"|ADS|American Depositary Shares|Depositary Shares|SPAC|Units?|Right[sW]?|Warrant[s]?)\s*$",
    re.I)


def fetch_us_nasdaq() -> list[dict]:
    """美股：NASDAQ 官方 screener 全量（NASDAQ+NYSE+AMEX，~7100 条）。

    返回 {code: ticker, abbr: 英文名（去 Common Stock 等尾部）, full_name: ""}；
    中文名靠东财累积（master 增量并集），英文名进 alias。
    """
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks",
                         params={"tableonly": "true", "limit": 25, "download": "true"},
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                  "Accept": "application/json"},
                         timeout=30, proxies={"http": None, "https": None})
        rows = ((r.json() or {}).get("data") or {}).get("rows") or []
    except Exception as e:
        print(f"WARN nasdaq fetch failed: {e}")
        return []
    out = []
    for x in rows:
        sym = str(x.get("symbol") or "").strip()
        name = str(x.get("name") or "").strip()
        if not sym or not name:
            continue
        name = _US_NAME_TAIL.sub("", name).strip(" -.")
        if len(name) < 2:
            name = sym
        out.append({"code": sym, "abbr": sym, "name_en": name})
    print(f"美股(NASDAQ官方): {len(out)} 条")
    return out


def fetch_hk_us() -> tuple[list[dict], list[dict]]:
    """港股/美股。港股走 HKEX 官方名单（全量+全称），失败回落东财；美股走东财分页。"""
    hk = fetch_hkex(os.path.join(ENT_DIR, "hkex_list.xlsx"),
                    os.path.join(ENT_DIR, "hkex_list_c.xlsx"))
    if not hk:
        raw = _em_clist("m:116", expect_total=4000)
        _WARRANT = re.compile(r"牛|熊|购|沽|界内|界外")
        # EM 港股代码可能不带前导零（如 700）；正股/ETF 代码 int<10000
        hk = [{"code": r["code"].zfill(5), "abbr": r["abbr"], "full_name": ""}
              for r in raw
              if r["code"].isdigit() and int(r["code"]) < 10000
              and not _WARRANT.search(r["abbr"])]
        print(f"港股(EM回落): {len(hk)} 条（原始 {len(raw)}）")
    us = fetch_us_nasdaq()
    if not us:   # NASDAQ 接口失败回落东财
        us = [{**r, "full_name": ""}
              for r in _em_clist("m:105,m:106,m:107", expect_total=20000)
              if 1 <= len(r["code"]) <= 6]
        print(f"美股(EM回落): {len(us)} 条")
    return hk, us


# ---------------------------------------------------------------------------
# 合并主数据
# ---------------------------------------------------------------------------
def build_master(a_rows, hk_rows, us_rows) -> list[dict]:
    """按归一化名称/代码/英文名合并跨市场同一公司 -> stock_master 行。

    跨市场合并键（优先级）：代码（EM 美股与 NASDAQ 同 ticker）-> 中文名 norm ->
    英文名 norm（HKEX 官方英文名 vs NASDAQ 英文名，如 ALIBABA GROUP ... 两边一致）。
    """
    groups: dict[str, dict] = {}
    code_index: dict[str, dict] = {}
    en_index: dict[str, dict] = {}

    def _lookup(code, abbr, en):
        g = code_index.get(code)
        if g:
            return g
        for key in (norm_name(abbr), norm_name(en or "")):
            if key and groups.get(key):
                return groups[key]
            if key and en_index.get(key):
                return en_index[key]
        return None

    def add(market: str, code: str, abbr: str, full: str = "",
            en_name: str | None = None, extra_alias: str | None = None):
        n = norm_name(abbr)
        if len(n) < 2:
            return
        g = _lookup(code, abbr, en_name)
        if g is None:
            g = {"abbr": abbr, "full_name": "", "markets": [], "codes": {},
                 "extras": set()}
            groups[n] = g
        code_index[code] = g
        if en_name:
            en_index.setdefault(norm_name(en_name), g)
        if market not in g["markets"]:
            g["markets"].append(market)
        g["codes"].setdefault(market, code)   # 同市场多代码（如港股正股/人民币柜台）保留首个
        if full and not g["full_name"]:
            g["full_name"] = full
        if extra_alias:
            g["extras"].add(extra_alias)

    for r in a_rows:
        add({"SH": "SH", "SZ": "SZ", "BJ": "BJ"}[r["exchange"]], r["code"],
            r["abbr"], r["full_name"])
    for r in hk_rows:
        # canonical 用中文名；HKEX 官方英文名作跨市场合并键 + alias
        add("HK", r["code"], r["abbr"], en_name=r.get("full_name"))
    for r in us_rows:
        add("US", r["code"], r["abbr"], extra_alias=r.get("name_en"),
            en_name=r.get("name_en"))

    # HKEX 官方英文名 -> 别名（英文 query 兜底）
    en_by_code = {r["code"]: r["full_name"] for r in hk_rows if r.get("full_name")}

    out = []
    for n, g in groups.items():
        markets = sorted(set(g["markets"]), key=lambda m: ["SH", "SZ", "BJ", "HK", "US"].index(m))
        aliases = set()
        abbr = clean_abbr(g["abbr"])
        raw = abbr
        aliases.add(abbr)
        stripped = _MKT_SUFFIX.sub("", abbr).strip("－- ")
        if stripped and stripped != abbr:
            aliases.add(stripped)
        nost = _ST_PREFIX.sub("", stripped or abbr).strip()
        if nost and len(nost) >= 2:
            aliases.add(nost)
        # 代码本身也作 alias（A股6位/港股5位/美股 ticker）
        for m, c in g["codes"].items():
            aliases.add(c)
            if c in en_by_code:
                aliases.add(en_by_code[c])
        aliases.update(g["extras"])
        full = clean_abbr(g["full_name"])
        # canonical 优先公司全称；无全称时剥掉 -W/-SW/－R 等市场标记后缀再作标准名
        canonical = full if len(full) >= 4 else (_MKT_SUFFIX.sub("", abbr).strip("－- ") or abbr)
        if full and full != abbr:
            aliases.add(abbr)
        out.append({"canonical": canonical, "full_name": full,
                    "abbr": abbr, "markets": markets, "codes": g["codes"],
                    "aliases": sorted(a for a in aliases if a)})
    out.sort(key=lambda x: x["canonical"])
    return out


# ---------------------------------------------------------------------------
# 指标词表（手工种子）
# ---------------------------------------------------------------------------
INDICATORS = [
    # (name, aliases, category)
    ("市盈率", ["PE", "pe", "P/E", "市盈率TTM", "滚动市盈率", "静态市盈率", "动态市盈率"], "估值"),
    ("市净率", ["PB", "pb", "P/B"], "估值"),
    ("市销率", ["PS", "P/S"], "估值"),
    ("市现率", ["PCF", "P/CF"], "估值"),
    ("PEG", ["peg比率", "PEG估值"], "估值"),
    ("股息率", ["分红率", "股息收益率"], "估值"),
    ("EV/EBITDA", ["企业价值倍数", "EV/EBIT"], "估值"),
    ("总市值", ["市值", "总市场价值"], "估值"),
    ("流通市值", ["流通盘市值"], "估值"),
    ("营业收入", ["营收", "主营业务收入", "总收入"], "财务"),
    ("净利润", ["归母净利润", "归属母公司净利润", "净利"], "财务"),
    ("扣非净利润", ["扣除非经常性损益", "扣非"], "财务"),
    ("毛利率", ["gross margin"], "财务"),
    ("净利率", ["销售净利率"], "财务"),
    ("ROE", ["净资产收益率", "roe"], "财务"),
    ("ROA", ["总资产收益率", "总资产回报率"], "财务"),
    ("EPS", ["每股收益", "每股盈利"], "财务"),
    ("资产负债率", ["负债率", "杠杆率"], "财务"),
    ("经营性现金流", ["经营现金流", "现金流净额"], "财务"),
    ("商誉", ["商誉减值"], "财务"),
    ("存货周转率", ["库存周转率"], "财务"),
    ("营收增速", ["收入增速", "营收同比增长", "营收环比"], "成长"),
    ("净利润增速", ["净利增速", "利润增速"], "成长"),
    ("CAGR", ["复合增长率", "年复合增速"], "成长"),
    ("同比", ["同比增长", "YoY", "yoy"], "成长"),
    ("环比", ["环比增长", "QoQ", "MoM"], "成长"),
    ("主力资金", ["主力净流入", "主力流出"], "资金"),
    ("北向资金", ["北上资金", "沪股通资金", "深股通资金"], "资金"),
    ("南向资金", ["南下资金", "港股通资金"], "资金"),
    ("融资融券", ["两融", "融资余额", "融券余额"], "资金"),
    ("换手率", ["turnover rate"], "交易"),
    ("量比", ["volume ratio"], "交易"),
    ("成交量", ["vol", "成交量放大"], "交易"),
    ("成交额", ["成交金额", "成交破万亿"], "交易"),
    ("振幅", ["日内振幅"], "交易"),
    ("涨停", ["封板", "打板"], "交易"),
    ("跌停", ["跌停板"], "交易"),
    ("龙虎榜", ["沪深龙虎榜"], "交易"),
    ("大宗交易", ["block trade"], "交易"),
    ("股权质押", ["质押率", "质押爆仓"], "交易"),
    ("解禁", ["限售解禁", "解禁潮"], "交易"),
    ("回购", ["股票回购", "回购注销"], "交易"),
    ("增持", ["股东增持"], "交易"),
    ("减持", ["股东减持", "减持套现"], "交易"),
    ("定增", ["定向增发", "非公开发行"], "交易"),
    ("配股", ["配售", "供股"], "交易"),
    ("分红", ["分红派息", "派息", "现金分红"], "交易"),
    ("除权除息", ["除权", "除息"], "交易"),
    ("MACD", ["macd金叉", "macd死叉"], "技术"),
    ("KDJ", ["kdj指标"], "技术"),
    ("RSI", ["相对强弱指标"], "技术"),
    ("布林线", ["BOLL", "布林带"], "技术"),
    ("均线", ["MA", "五日线", "十日线", "二十日线", "年线", "半年线"], "技术"),
    ("金叉", ["黄金交叉"], "技术"),
    ("死叉", ["死亡交叉"], "技术"),
    ("支撑位", ["支撑"], "技术"),
    ("压力位", ["阻力位", "压力"], "技术"),
    ("筹码分布", ["筹码集中度"], "技术"),
    ("牛市", ["多头市场", "大牛市"], "市场"),
    ("熊市", ["空头市场", "大熊市"], "市场"),
    ("震荡市", ["区间震荡", "结构性行情"], "市场"),
    ("板块轮动", ["高低切换"], "市场"),
    ("风格切换", ["大小盘切换"], "市场"),
    ("市场情绪", ["投资情绪", "避险情绪", "风险偏好"], "市场"),
    ("VIX", ["恐慌指数", "vix"], "市场"),
    ("期货", ["商品期货", "股指期货"], "衍生"),
    ("期权", ["stock option", "个股期权"], "衍生"),
    ("可转债", ["转债", "convertible bond"], "衍生"),
    ("杠杆", ["加杠杆", "去杠杆"], "衍生"),
    ("做空", ["沽空", "卖空", "做空机构"], "衍生"),
    ("熔断", [" Circuit Breaker", "二次熔断"], "衍生"),
    ("IPO", ["新股发行", "上市首发"], "市场"),
    ("打新", ["新股申购", "中签"], "市场"),
    ("注册制", ["全面注册制"], "市场"),
    ("财报", ["季报", "中报", "年报", "业绩报告", "业绩预告", "业绩快报"], "市场"),
]


def build_indicators() -> list[dict]:
    return [{"name": n, "aliases": sorted(set(a for a in als if a and a.strip() and a.strip() != n)),
             "category": cat}
            for n, als, cat in INDICATORS]


# ---------------------------------------------------------------------------
# 合并进 gazetteer
# ---------------------------------------------------------------------------
def _norm_linker(s: str) -> str:
    """与 ner/entity_linker._norm 完全一致（保持 eid/索引兼容）。"""
    n = _SUFFIX.sub("", str(s).strip().lower())
    return re.sub(r"\s+", "", n)


def eid_of(canonical: str) -> str:
    import hashlib
    return "E" + hashlib.md5(canonical.strip().lower().encode("utf-8")).hexdigest()[:8]


def merge_into_gazetteer(stock_master: list[dict], indicators: list[dict]) -> dict:
    # 读现有词表
    entries: dict[str, dict] = {}
    if os.path.exists(GAZ_PATH):
        with open(GAZ_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    entries[e["eid"]] = e
    # 清掉上次脚本写入的股票条目（canonical 可能修错），重新合并；种子/COMPANY 等保留
    stale = [eid for eid, e in entries.items() if e.get("source") == "akshare_master"]
    for eid in stale:
        del entries[eid]
    by_norm: dict[str, dict] = {}
    for e in entries.values():
        by_norm[_norm_linker(e["name"])] = e
        for a in e.get("aliases", []):
            by_norm.setdefault(_norm_linker(a), e)

    stats = {"stale_removed": len(stale), "stock_new": 0, "stock_merge": 0,
             "ind_new": 0, "ind_merge": 0, "alias_add": 0}

    def add_alias(e: dict, alias: str):
        n = _norm_linker(alias)
        if not n or n in by_norm:
            return
        if alias not in e.setdefault("aliases", []):
            e["aliases"].append(alias)
            by_norm[n] = e
            stats["alias_add"] += 1

    # 股票：锚点可命中 STOCK 或 COMPANY（query_llm 注册的公司），均合并
    for m in stock_master:
        anchor = None
        for cand in [m["canonical"], m["full_name"], m["abbr"]] + m["aliases"]:
            if not cand:
                continue
            e = by_norm.get(_norm_linker(cand))
            if e and e["type"] in ("STOCK", "COMPANY"):
                anchor = e
                break
        if anchor:
            if anchor["type"] == "COMPANY":
                anchor["type"] = "STOCK"   # 升级为主数据实体，保留 eid 稳定
                anchor["source"] = anchor.get("source", "") + "+master"
            stats["stock_merge"] += 1
        else:
            anchor = {"eid": eid_of(m["canonical"]), "name": m["canonical"],
                      "aliases": [], "type": "STOCK", "source": "akshare_master"}
            entries[anchor["eid"]] = anchor
            by_norm[_norm_linker(anchor["name"])] = anchor
            stats["stock_new"] += 1
        anchor.setdefault("meta", {})
        anchor["meta"]["markets"] = m["markets"]
        anchor["meta"]["codes"] = m["codes"]
        anchor["meta"]["full_name"] = m["full_name"]
        for a in m["aliases"]:
            add_alias(anchor, a)

    # 指标（并入 MACRO）
    for ind in indicators:
        e = by_norm.get(_norm_linker(ind["name"]))
        if e and e["type"] == "MACRO":
            stats["ind_merge"] += 1
        else:
            e = {"eid": eid_of(ind["name"]), "name": ind["name"], "aliases": [],
                 "type": "MACRO", "source": "indicator_seed"}
            entries[e["eid"]] = e
            by_norm[_norm_linker(e["name"])] = e
            stats["ind_new"] += 1
        e.setdefault("meta", {})["category"] = ind["category"]
        for a in ind["aliases"]:
            add_alias(e, a)

    with open(GAZ_PATH, "w", encoding="utf-8") as f:
        for e in sorted(entries.values(), key=lambda x: x["eid"]):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    stats["total"] = len(entries)
    return stats


def sync_jieba():
    """词表全量同步进 jieba 用户词典。

    词典文件分两区：手工区（人工维护，原样保留）+ 自动区（marker 之后，由
    gazetteer 全量重建，保证清掉历史残留如窝轮名）。
    """
    words = set()
    with open(GAZ_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e["name"] and len(e["name"]) >= 2:
                words.add(e["name"])
            for a in e.get("aliases", []):
                if a and len(a) >= 2:
                    words.add(a)
    path = os.path.join(ROOT, "data", "dict", "jieba_user_dict.txt")
    marker = "# --- auto: gazetteer sync (do not edit below) ---"
    manual: list[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        head = content.split(marker)[0]
        manual = [l for l in head.splitlines() if l.strip()]
    existing = {l.split()[0] for l in manual if l.split()}
    fresh = sorted(w for w in words if w not in existing)
    with open(path, "w", encoding="utf-8") as f:
        for l in manual:
            f.write(l + "\n")
        f.write(marker + "\n")
        for w in fresh:
            f.write(f"{w} 2000000 nz\n")
    return len(fresh)


def main():
    skip_fetch = "--skip-fetch" in sys.argv
    master_path = MASTER_PATH

    if skip_fetch and os.path.exists(master_path):
        master = [json.loads(l) for l in open(master_path, encoding="utf-8") if l.strip()]
        print(f"[skip-fetch] master from disk: {len(master)}")
    else:
        t0 = time.time()
        a_rows = fetch_a_shares()
        hk_rows, us_rows = fetch_hk_us()
        # 增量累积：网络抖动下单轮抓不全，把上一轮 master 的 HK/US 条目并回来
        if os.path.exists(master_path):
            for l in open(master_path, encoding="utf-8"):
                if not l.strip():
                    continue
                m = json.loads(l)
                for mkt, code in (m.get("codes") or {}).items():
                    if mkt == "HK":
                        hk_rows.append({"code": code, "abbr": m.get("abbr") or m["canonical"]})
                    elif mkt == "US":
                        us_rows.append({"code": code, "abbr": m.get("abbr") or m["canonical"]})
        print(f"fetch done in {time.time()-t0:.0f}s")
        master = build_master(a_rows, hk_rows, us_rows)
        with open(master_path, "w", encoding="utf-8") as f:
            for m in master:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        n_hk = sum(1 for m in master if "HK" in m["markets"])
        n_us = sum(1 for m in master if "US" in m["markets"])
        print(f"stock_master: {len(master)} 条（HK {n_hk} / US {n_us}）-> {master_path}")

    indicators = build_indicators()
    with open(IND_PATH, "w", encoding="utf-8") as f:
        for i in indicators:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    print(f"indicator_lexicon: {len(indicators)} 条 -> {IND_PATH}")

    stats = merge_into_gazetteer(master, indicators)
    print(f"gazetteer merge: {stats}")
    n = sync_jieba()
    print(f"jieba user dict: +{n} words")


if __name__ == "__main__":
    main()
