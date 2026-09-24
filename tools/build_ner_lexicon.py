# -*- coding: utf-8 -*-
"""按 NER 11 类体系补齐词表缺口（在 stock/topic 词表之上）。

类别盘点与本次动作：
  STOCK/STOCK_CODE  已有主数据 1.5 万；本脚本补代码带后缀变体（0700.HK / 600519.SH）
  INDEX             已有 ~70；补板块指数 ~30（中证机器人/中证白酒...）
  INDUSTRY          已有 ~235；不动
  FUND              空 -> 知名 ETF 种子 ~80（名称+代码）
  MACRO             已有 114；不动
  ORG               15(query_llm) -> 机构种子 ~35（监管/国际组织/交易所/评级）
  COMPANY           种子弱 -> 知名未上市/品牌 ~40（华为、字节跳动、智谱AI...）
  PRODUCT           空 -> 产品/业务/技术种子 ~80（换电模式、iPhone、鸿蒙、智驾方案...）
  METRIC            指标从 MACRO 迁移为独立 METRIC + 补行情指标
  TIME              规则覆盖（time_expressions），无需词表

幂等可重跑：norm 占用的种子自动跳过（如「中金公司」已链指 STOCK 则不建 ORG）。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.build_stock_lexicon import (  # noqa: E402
    ENT_DIR, GAZ_PATH, _norm_linker, eid_of, sync_jieba)

NER_PATH = os.path.join(ENT_DIR, "ner_seed_lexicon.jsonl")

# ---------------------------------------------------------------------------
COMPANIES = [
    # 未上市公司/品牌（上市公司由主数据覆盖，norm 占用自动跳过）
    ("华为", ["华为技术", "华为公司"]),
    ("字节跳动", ["字节", "ByteDance", "抖音集团"]),
    ("抖音", []),
    ("小红书", ["Xiaohongshu", "REDnote"]),
    ("大疆", ["DJI", "大疆创新"]),
    ("智谱AI", ["智谱", "Zhipu", "智谱清言"]),
    ("月之暗面", ["Kimi", "Moonshot"]),
    ("MiniMax", ["海螺AI"]),
    ("DeepSeek", ["深度求索"]),
    ("阶跃星辰", ["StepFun"]),
    ("百川智能", ["百川", "Baichuan"]),
    ("零一万物", ["01.AI", "零一"]),
    ("宇树科技", ["宇树", "Unitree"]),
    ("米哈游", ["原神", "miHoYo"]),
    ("老干妈", []),
    ("娃哈哈", []),
    ("Shein", ["希音"]),
    ("得物", ["毒App"]),
    ("菜鸟网络", ["菜鸟"]),
    ("蚂蚁集团", ["蚂蚁金服", "Ant Group"]),
    ("饿了么", []),
    ("联想集团", []),          # 已上市会自动合并到 STOCK
    ("荣耀", ["荣耀终端", "Honor"]),
    ("传音控股", []),
    ("汇川技术", []),
    ("威马汽车", []),
    ("高合汽车", []),
    ("蔚来资本", []),
    ("平安集团", ["中国平安集团"]),
    ("泰康保险", ["泰康"]),
    ("复星集团", ["复星"]),
    ("海航集团", ["海航"]),
    ("宝能集团", ["宝能"]),
    ("恒大集团", ["恒大"]),
    ("碧桂园服务", []),
    ("万达集团", ["万达"]),
    ("苏宁", ["苏宁易购", "苏宁控股"]),
    ("国美", ["国美零售"]),
    ("中植集团", ["中植系"]),
    ("华兴资本", []),
]

ORGS = [
    ("证监会", ["中国证监会", "CSRC"]),
    ("金融监管总局", ["国家金融监督管理总局", "银保监会"]),
    ("工信部", ["工业和信息化部"]),
    ("商务部", ["中国商务部"]),
    ("住建部", ["住房和城乡建设部"]),
    ("农业农村部", []),
    ("国家能源局", ["能源局"]),
    ("国家统计局", ["统计局"]),
    ("国家发改委", ["发改委"]),
    ("世界银行", ["World Bank"]),
    ("国际货币基金组织", ["IMF"]),
    ("世贸组织", ["WTO", "世界贸易组织"]),
    ("欧佩克", ["OPEC", "石油输出国组织"]),
    ("亚洲开发银行", ["亚开行"]),
    ("亚洲基础设施投资银行", ["亚投行", "AIIB"]),
    ("上交所", ["上海证券交易所", "SSE"]),
    ("深交所", ["深圳证券交易所", "SZSE"]),
    ("北交所", ["北京证券交易所"]),       # 北交所 norm 撞 INDUSTRY/地域时自动跳过
    ("港交所", ["香港交易所", "HKEX"]),
    ("纽交所", ["纽约证券交易所", "NYSE"]),
    ("中金所", ["中国金融期货交易所"]),
    ("上期所", ["上海期货交易所"]),
    ("大商所", ["大连商品交易所"]),
    ("郑商所", ["郑州商品交易所"]),
    ("广期所", ["广州期货交易所"]),
    ("中国结算", ["中证登", "中国证券登记结算"]),
    ("中基协", ["中国证券投资基金业协会"]),
    ("证券业协会", ["中国证券业协会"]),
    ("惠誉", ["惠誉评级", "Fitch"]),
    ("穆迪", ["Moody's", "穆迪评级"]),
    ("Gartner", ["高德纳"]),
    ("IDC", ["国际数据公司"]),
    ("世界经贸组织", []),
    ("金融稳定理事会", ["FSB"]),
    ("清算所", []),
]

PRODUCTS = [
    # 业务/模式
    ("换电模式", ["换电站", "电池换电"]),
    ("智驾方案", ["智能驾驶方案", "高阶智驾"]),
    ("价格战", ["降价潮"]),
    ("以旧换新", []),
    ("降价促销", []),
    ("快充", ["超充", "高压快充"]),
    ("增程式", ["增程电动", "EREV"]),
    ("一体化压铸", []),
    ("电池租赁", []),
    ("直营模式", []),
    ("会员制", ["付费会员"]),
    ("直播带货", ["直播电商"]),
    ("社区团购", []),
    ("即时零售", ["外卖大战"]),
    ("无人配送", ["无人快递"]),
    ("低空物流", []),
    ("固态硬盘", ["SSD"]),
    ("数据中心租赁", ["IDC租赁"]),
    ("算力中心", ["超算中心"]),
    # 硬件/终端产品
    ("iPhone", ["苹果手机", "iPhone 17"]),
    ("Mate 60", ["Mate60"]),
    ("Mate 70", ["Mate70"]),
    ("问界M9", ["问界 M9"]),
    ("问界M8", []),
    ("Model Y", ["特斯拉Model Y"]),
    ("Model 3", []),
    ("Cybertruck", []),
    ("SU7", ["小米SU7"]),
    ("YU7", ["小米YU7"]),
    ("仰望U8", ["仰望"]),
    ("理想MEGA", []),
    ("小鹏X9", []),
    ("蔚来ET5", []),
    ("Mate XT", ["三折叠"]),
    ("Vision Pro", ["vision pro"]),
    ("Quest", []),
    ("Switch", []),
    ("RTX 5090", ["RTX5090", "5090显卡"]),
    ("H20", ["英伟达H20"]),
    ("B200", ["英伟达B200"]),
    ("A100", []),
    ("H100", []),
    ("昇腾", ["昇腾芯片", "Atlas"]),
    ("鲲鹏", []),
    ("麒麟芯片", ["麒麟9000", "Kirin"]),
    ("天玑芯片", ["天玑"]),
    ("骁龙", ["Snapdragon"]),
    ("M4芯片", ["M4"]),
    ("C919", ["大飞机C919"]),
    ("AG600", []),
    ("运20", []),
    ("福建舰", []),
    ("山东舰", []),
    ("北斗三号", []),
    ("天宫空间站", ["中国空间站"]),
    ("嫦娥六号", ["嫦娥"]),
    # 软件/模型/系统
    ("ChatGPT", ["GPT-4", "GPT-5", "OpenAI模型"]),
    ("Sora", []),
    ("Gemini", ["Gemini模型"]),
    ("Claude", ["Claude模型"]),
    ("Copilot", ["GitHub Copilot"]),
    ("文心一言", ["文心大模型", "ERNIE"]),
    ("通义千问", ["Qwen", "通义"]),
    ("混元", ["腾讯混元"]),
    ("豆包", ["豆包大模型"]),
    ("DeepSeek-R1", ["DeepSeek模型"]),
    ("星火大模型", ["讯飞星火"]),
    ("HarmonyOS", ["鸿蒙系统", "鸿蒙OS", "纯血鸿蒙"]),
    ("OpenHarmony", []),
    ("欧拉", ["openEuler"]),
    ("龙芯", ["龙芯中科CPU"]),
    ("Windows", []),
    ("安卓", ["Android"]),
    ("微信小店", []),
    ("微信豆包", []),
    ("钉钉", []),
    ("飞书", []),
    ("企业微信", []),
    ("WPS", ["金山办公"]),
    ("同花顺App", []),
    ("雪球App", []),
]

FUNDS = [
    ("华夏机器人ETF", "562500"), ("机器人ETF", "562500"),
    ("沪深300ETF", "510300"), ("中证500ETF", "510500"),
    ("中证1000ETF", "512100"), ("上证50ETF", "510050"),
    ("创业板ETF", "159915"), ("科创50ETF", "588000"),
    ("科创创业50ETF", "159780"), ("双创50ETF", "159780"),
    ("红利ETF", "510880"), ("红利低波ETF", "512890"),
    ("央企共赢ETF", "517090"), ("央企创新ETF", "159974"),
    ("券商ETF", "512000"), ("证券保险ETF", "512070"),
    ("银行ETF", "512800"), ("金融ETF", "510230"),
    ("芯片ETF", "512760"), ("半导体ETF", "512480"),
    ("半导体设备ETF", "159516"), ("芯片龙头ETF", "159801"),
    ("人工智能ETF", "515980"), ("AI ETF", "515980"),
    ("云计算ETF", "516510"), ("大数据ETF", "515400"),
    ("软件ETF", "515230"), ("计算机ETF", "512720"),
    ("通信ETF", "515880"), ("5G ETF", "515050"),
    ("军工ETF", "512660"), ("国防ETF", "512670"),
    ("医药ETF", "512010"), ("医疗ETF", "512170"),
    ("创新药ETF", "159992"), ("生物医药ETF", "512290"),
    ("疫苗ETF", "159643"), ("中药ETF", "560080"),
    ("酒ETF", "512690"), ("消费ETF", "159928"),
    ("食品饮料ETF", "515170"), ("家电ETF", "159996"),
    ("新能源汽车ETF", "515030"), ("新能源车ETF", "515030"),
    ("光伏ETF", "515790"), ("风电ETF", "516440"),
    ("锂电池ETF", "159840"), ("碳中和ETF", "159790"),
    ("绿色电力ETF", "159625"), ("电力ETF", "159611"),
    ("煤炭ETF", "515220"), ("钢铁ETF", "515210"),
    ("有色金属ETF", "512400"), ("稀土ETF", "516150"),
    ("黄金ETF", "518880"), ("黄金股ETF", "517400"),
    ("豆粕ETF", "159985"), ("能源化工ETF", "159981"),
    ("房地产ETF", "512200"), ("基建ETF", "516950"),
    ("化工ETF", "516020"), ("养殖ETF", "159865"),
    ("畜牧养殖ETF", "159867"), ("游戏ETF", "516010"),
    ("影视ETF", "516620"), ("传媒ETF", "512980"),
    ("旅游ETF", "159766"), ("农业ETF", "159825"),
    ("恒生科技ETF", "513130"), ("恒生互联网ETF", "513330"),
    ("中概互联ETF", "513050"), ("香港证券ETF", "513090"),
    ("恒生医疗ETF", "513060"), ("港股通医药ETF", "513200"),
    ("纳指ETF", "513100"), ("标普500ETF", "513500"),
    ("日经ETF", "513520"), ("德国ETF", "159561"),
    ("法国CAC40ETF", "513080"), ("沙特ETF", "520830"),
    ("东南亚科技ETF", "513730"), ("美国50ETF", "513850"),
    ("国债ETF", "511010"), ("货币ETF", "511990"),
    ("城投债ETF", "511220"), ("可转债ETF", "511380"),
]

SECTOR_INDICES = [
    ("中证机器人", ["中证机器人指数", "机器人指数"]),
    ("中证军工", ["中证军工指数"]),
    ("中证白酒", ["中证白酒指数"]),
    ("中证医疗", ["中证医疗指数"]),
    ("中证中药", []),
    ("中证畜牧养殖", ["畜牧养殖指数"]),
    ("中证银行", ["中证银行指数"]),
    ("中证证券保险", ["证券保险指数"]),
    ("中证煤炭", []),
    ("中证钢铁", []),
    ("中证有色金属", []),
    ("中证稀有金属", []),
    ("中证影视", []),
    ("中证游戏", ["中证动漫游戏"]),
    ("中证体育", []),
    ("中证人工智能", ["人工智能主题指数", "CS人工智能"]),
    ("中证新能源汽车", ["新能源汽车指数"]),
    ("中证光伏产业", ["光伏产业指数"]),
    ("中证全指半导体", ["全指半导体"]),
    ("国证芯片", ["芯片指数"]),
    ("中华半导体芯片", ["中华半导体"]),
    ("中证消费电子", []),
    ("中证家电", []),
    ("中证化工", []),
    ("中证有色金属矿业", []),
    ("中证港股通科技", ["港股通科技"]),
    ("中证红利质量", []),
    ("中证央企红利", []),
    ("科创芯片", ["科创芯片指数"]),
    ("科创AI", ["科创人工智能指数"]),
]

METRIC_EXTRA = [
    ("股价", ["股票价格", "现价", "最新价"]),
    ("涨跌幅", ["涨幅", "跌幅", "涨跌幅度"]),
    ("开盘价", []),
    ("收盘价", []),
    ("最高价", []),
    ("最低价", []),
    ("总资产", ["资产总计"]),
    ("总负债", ["负债合计"]),
    ("股东权益", ["净资产"]),
    ("资产负债表", []),
    ("利润表", ["损益表"]),
    ("现金流量表", ["现金流量"]),
    ("市占率", ["市场份额", "市场占有率"]),
    ("产能利用率", ["稼动率"]),
    ("订单", ["在手订单", "新增订单"]),
    ("出货量", ["销量", "交付量"]),
]


def main():
    seeds = []
    for name, als in COMPANIES:
        seeds.append({"name": name, "type": "COMPANY", "category": "公司/品牌",
                      "aliases": sorted({a for a in als if a and a != name})})
    for name, als in ORGS:
        seeds.append({"name": name, "type": "ORG", "category": "机构/组织",
                      "aliases": sorted({a for a in als if a and a != name})})
    for name, als in PRODUCTS:
        seeds.append({"name": name, "type": "PRODUCT", "category": "产品/技术",
                      "aliases": sorted({a for a in als if a and a != name})})
    seen_codes = set()
    for name, code in FUNDS:
        if code in seen_codes:
            continue
        seen_codes.add(code)
        seeds.append({"name": name, "type": "FUND", "category": "基金/ETF",
                      "aliases": [code]})
    for name, als in SECTOR_INDICES:
        seeds.append({"name": name, "type": "INDEX", "category": "板块指数",
                      "aliases": sorted({a for a in als if a and a != name})})
    for name, als in METRIC_EXTRA:
        seeds.append({"name": name, "type": "METRIC", "category": "财务/行情指标",
                      "aliases": sorted({a for a in als if a and a != name})})
    with open(NER_PATH, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"ner_seed_lexicon: {len(seeds)} 条 -> {NER_PATH}")
    print("  by type:", dict(Counter(s["type"] for s in seeds)))

    # ---- 合并进 gazetteer ----
    entries: dict = {}
    with open(GAZ_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                entries[e["eid"]] = e
    by_norm = {}
    for e in entries.values():
        by_norm.setdefault(_norm_linker(e["name"]), e)
        for a in e.get("aliases", []):
            by_norm.setdefault(_norm_linker(a), e)

    n_new, n_merge, n_skip, n_alias = 0, 0, 0, 0

    def add_alias(e, alias):
        nonlocal n_alias
        n = _norm_linker(alias)
        if not n or n in by_norm:
            return False
        if alias not in e.setdefault("aliases", []):
            e["aliases"].append(alias)
            by_norm[n] = e
            n_alias += 1
        return True

    for s in seeds:
        anchor = by_norm.get(_norm_linker(s["name"]))
        if anchor and anchor["type"] == s["type"]:
            n_merge += 1
        elif anchor:
            # norm 被其他类型占用（如「北交所」已是 INDUSTRY、「中金公司」是 STOCK）
            # -> 种子别名并入该条目，不新建，避免同 norm 两实体
            n_skip += 1
            anchor.setdefault("meta", {}).setdefault("also", s["type"])
            for a in s["aliases"]:
                add_alias(anchor, a)
            continue
        else:
            anchor = {"eid": eid_of(s["name"]), "name": s["name"], "aliases": [],
                      "type": s["type"], "source": "ner_seed",
                      "meta": {"category": s["category"]}}
            entries[anchor["eid"]] = anchor
            by_norm[_norm_linker(anchor["name"])] = anchor
            n_new += 1
        for a in s["aliases"]:
            add_alias(anchor, a)

    # ---- METRIC 迁移：indicator_seed 的 MACRO -> METRIC ----
    n_mig = 0
    for e in entries.values():
        if e.get("source") == "indicator_seed" and e.get("type") == "MACRO":
            e["type"] = "METRIC"
            e.setdefault("meta", {})["category"] = e.get("meta", {}).get("category") or "财务/行情指标"
            n_mig += 1

    # ---- ORG 迁移：11 类规范下「证监会」归 ORG（原为 MACRO seed 词条）----
    n_org = 0
    for e in entries.values():
        if e.get("name") == "证监会" and e.get("type") == "MACRO":
            e["type"] = "ORG"
            m = e.get("meta") or {}
            m.pop("also", None)
            m["category"] = "机构/组织"
            e["meta"] = m
            n_org += 1

    # ---- STOCK 代码后缀变体（0700.HK / 600519.SH / 000001.SZ）----
    n_codevar = 0
    for e in entries.values():
        if e.get("type") != "STOCK":
            continue
        codes = (e.get("meta") or {}).get("codes") or {}
        for mkt, code in codes.items():
            if not code:
                continue
            if mkt == "HK" and code.isdigit():
                variants = [f"{code}.HK", f"{code.lstrip('0')}.HK"]
            elif mkt == "SH" and code.isdigit():
                variants = [f"{code}.SH"]
            elif mkt == "SZ" and code.isdigit():
                variants = [f"{code}.SZ"]
            elif mkt == "BJ" and code.isdigit():
                variants = [f"{code}.BJ"]
            else:
                continue
            for v in variants:
                if add_alias(e, v):
                    n_codevar += 1

    with open(GAZ_PATH, "w", encoding="utf-8") as f:
        for e in sorted(entries.values(), key=lambda x: x["eid"]):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"gazetteer: total {len(entries)} | new {n_new} merge {n_merge} "
          f"skip(占用) {n_skip} | metric迁移 {n_mig} | org迁移 {n_org} "
          f"| 代码变体 +{n_codevar} | alias +{n_alias}")
    n = sync_jieba()
    print(f"jieba user dict: +{n} words")


if __name__ == "__main__":
    main()
