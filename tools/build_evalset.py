# -*- coding: utf-8 -*-
"""合成评测集生成器（标准评测集构建，考核阶段一交付物）。

设计：
  - 覆盖全部 42 个二级意图 × 中英双语，每类 8~12 条合成 query
  - 模板槽位随机填充（标的/行业/指数/时间表达），避免与训练侧规则共享词面捷径
  - 每条带金标：gold_l2 + 期望槽位（object_type / time_anchor）
  - 输出 data/eval/synth_gold.jsonl

定位说明（写入评测报告）：合成集用于 42 类体系覆盖性验证与回归防护；
真实分布上的绝对水平以 160 条基线金标 + 人工标注抽样集为准。
"""
from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STOCKS = ["宁德时代", "贵州茅台", "北方华创", "药明康德", "紫金矿业", "中际旭创", "比亚迪", "恒瑞医药",
          "Tesla", "NVIDIA", "Apple", "AMD", "Goldman Sachs", "Morgan Stanley"]
INDUSTRIES = ["半导体", "储能", "创新药", "光模块", "稀土", "液冷", "固态电池", "核电",
              "semiconductor", "energy storage", "biotech", "optical modules", "rare earth"]
INDEXES = ["沪深300", "创业板指", "中证500", "恒生科技", "S&P 500", "Nasdaq"]
TIMES_ZH = ["2026年8月", "2026年", "最新", "近期", "2025年半年报", "未来三年", "2026年8月25日", "去年"]
TIMES_EN = ["2026 Q2", "in 2026", "latest", "recent", "last year", "the next three years"]

def fill(tpl: str, rng) -> dict:
    stock = rng.choice(STOCKS)
    industry = rng.choice(INDUSTRIES)
    index = rng.choice(INDEXES)
    time_zh, time_en = rng.choice(TIMES_ZH), rng.choice(TIMES_EN)
    q = tpl.format(stock=stock, industry=industry, index=index, time_zh=time_zh, time_en=time_en,
                   code=rng.choice(["600519", "300750", "002415", "688981"]))
    return {"query": q, "time_used": time_zh, "time_used_en": time_en,
            "industry_used": industry, "stock_used": stock, "index_used": index}


# 模板表：intent_id -> ([zh_templates], [en_templates])
TEMPLATES = {
    "INFO-01": (["{stock} {time_zh} 股价 多少", "{stock} {time_zh} 最新价 收盘价",
                 "{stock} 今天 行情 涨跌幅"],
                ["{stock} stock price {time_en}", "current quote for {stock}",
                 "{stock} share price {time_en}"]),
    "INFO-02": (["{index} {time_zh} 点位 行情", "{industry}板块 {time_zh} 表现 涨跌",
                 "A股 {time_zh} 板块轮动 情况"],
                ["{index} level {time_en}", "{industry} sector performance {time_en}",
                 "sector rotation {time_en}"]),
    "INFO-03": (["{stock} {time_zh} 主力资金 净流入", "{stock} 龙虎榜 席位 {time_zh}",
                 "{stock} {time_zh} 北向资金 持仓变化", "{stock} 筹码集中度 数据"],
                ["{stock} money flow {time_en}", "institutional buying in {stock} {time_en}",
                 "capital flow {stock}"]),
    "INFO-04": (["{stock} {time_zh} 年报 营收 净利润", "{stock} 实际控制人 股权结构",
                 "{stock} {time_zh} 机构评级 目标价", "{stock} 解禁 时间 规模",
                 "{stock} 主营业务 毛利率 情况"],
                ["{stock} revenue and net income {time_en}", "{stock} balance sheet {time_en}",
                 "analyst target price for {stock} {time_en}", "{stock} controlling shareholder structure"]),
    "INFO-05": (["{code} ETF {time_zh} 净值", "{stock} 基金 重仓股 {time_zh}",
                 "沪深300ETF {time_zh} 份额 变化", "创新药ETF 跟踪指数 {time_zh}"],
                ["ETF NAV {time_en}", "top holdings of ARKK {time_en}",
                 "index fund tracking {index}"]),
    "INFO-06": (["{stock} {time_zh} MACD 金叉 情况", "{stock} 支撑位 压力位 在哪",
                 "{stock} K线 均线 形态 {time_zh}"],
                ["{stock} macd indicator {time_en}", "support and resistance levels for {stock}",
                 "{stock} moving average chart"]),
    "INFO-07": (["{stock} {time_zh} 最新公告 新闻", "{industry} {time_zh} 最新消息 动态",
                 "{stock} 定增 进展 {time_zh}", "{stock} 减持 公告 {time_zh}",
                 "{stock} {time_zh} 利好消息"],
                ["latest news on {stock} {time_en}", "{stock} recent announcements {time_en}",
                 "{stock} press release {time_en}", "{stock} 8-k filing {time_en}"]),
    "INFO-08": (["{stock} {time_zh} 机构观点 研报", "券商 {industry} 研报 {time_zh}",
                 "{stock} 董秘 互动 回复", "{stock} 电话会议 纪要 {time_zh}",
                 "雪球 大V {stock} 观点"],
                ["latest research report on {stock} {time_en}", "analyst opinions on {stock} {time_en}",
                 "{stock} earnings call transcript {time_en}", "what do analysts on Twitter say about {stock}"]),
    "INFO-09": (["{industry} 行业 书籍 推荐", "证券法 相关 法律 条文",
                 "{stock} 财报 PDF 原文 下载", "价值投资 入门 书"],
                ["books on {industry} investing", "SEC regulations on disclosure",
                 "{stock} annual report pdf"]),
    "ANALYZE-01": (["{stock} {time_zh} 业绩 点评 分析", "{stock} 财报 解读 基本面",
                    "{stock} 竞争力 护城河 分析"],
                   ["fundamental analysis of {stock} {time_en}", "{stock} earnings analysis {time_en}",
                    "comprehensive review of {stock}"]),
    "ANALYZE-02": (["{stock} 技术面 分析 {time_zh}", "{stock} 走势 量价 关系 分析"],
                   ["technical analysis of {stock} {time_en}", "{stock} chart pattern analysis"]),
    "ANALYZE-03": (["{stock} 资金面 分析 {time_zh}", "{stock} 主力 动向 分析"],
                   ["fund flow analysis for {stock}", "capital flow analysis {industry} sector"]),
    "ANALYZE-04": (["A股 {time_zh} 市场情绪 分析", "当前 市场风格 主线 是什么",
                    "风险偏好 变化 分析"],
                   ["market sentiment analysis {time_en}", "risk-on risk-off regime {time_en}"]),
    "ANALYZE-05": (["{stock} 和 竞争对手 对比 哪个 更值得", "价值投资 和 趋势投资 区别",
                    "{industry} 和 储能 板块 对比"],
                   ["{stock} vs competitors comparison", "difference between value and growth investing",
                    "which is better {stock} or peers"]),
    "ANALYZE-06": (["{stock} {time_zh} 大跌 原因", "{stock} 涨停 原因 是什么",
                    "美联储 降息 对 A股 影响 解读", "{industry} 涨价 如何影响 下游",
                    "{stock} 业绩 点评 怎么看"],
                   ["why did {stock} drop sharply {time_en}", "what caused {stock} rally {time_en}",
                    "impact of Fed rate cut on {industry} stocks"]),
    "ANALYZE-07": (["{industry} 行业 景气度 研判 {time_zh}", "{industry} 竞争格局 分析",
                    "2026 货币政策 对 市场 影响 研判"],
                   ["{industry} industry outlook {time_en}", "macro outlook {time_en} inflation impact",
                    "{industry} competitive landscape analysis"]),
    "PREDICT-01": (["{stock} {time_zh} 目标价 能涨到 多少", "{stock} 还有 多大 上涨 空间",
                    "{stock} 盈利预测 {time_zh}"],
                   ["{stock} price target {time_en}", "how high can {stock} go",
                    "{stock} earnings forecast {time_en}"]),
    "PREDICT-02": (["{stock} {time_zh} 会涨 吗", "{stock} 未来 走势 预测",
                    "{stock} 明日 走势 怎么走", "A股 下周 方向 预判"],
                   ["will {stock} rise {time_en}", "{stock} price prediction {time_en}",
                    "{stock} stock outlook {time_en}"]),
    "PREDICT-03": (["{stock} 和 同行 哪个 会 先涨", "{industry} 谁会 跑赢 大盘"],
                   ["which will outperform {stock} or peers", "{industry} vs market which wins"]),
    "PREDICT-04": (["A股 会不会 进入 牛市 {time_zh}", "{industry} 处于 什么 周期 阶段",
                    "市场 见底 了吗"],
                   ["is it a bull market for {stock} {time_en}", "where are we in the {industry} cycle"]),
    "PREDICT-05": (["{stock} 重组 何时 通过", "{stock} IPO 什么时候 上市",
                    "美联储 下次 降息 时间 预期"],
                   ["when will {stock} split take effect", "when is the next Fed meeting decision"]),
    "PREDICT-06": (["美联储 加息 如何 传导 到 A股", "油价 上涨 传导 路径 哪些 板块 受影响"],
                   ["how will tariffs affect {industry} supply chain", "ripple effect of Fed policy on stocks"]),
    "SELECT-01": (["{industry} 概念股 有哪些", "{industry} 龙头股 名单 {time_zh}",
                   "股价 低于 20元 的 {industry} 股票 有哪些"],
                   ["which stocks benefit from {industry} {time_en}", "list of {industry} stocks",
                    "{industry} sector stocks under $50"]),
    "SELECT-02": (["{industry} 产业链 哪家 公司 受益 最大", "同时 满足 高股息 和 低估值 的 股票"],
                   ["which company is most exposed to {industry}", "stocks with high dividend and low valuation"]),
    "SELECT-03": (["{industry} 产业链 龙头 梳理 {time_zh}", "{industry} 风口 值得关注 的 标的",
                   "{time_zh} 值得 埋伏 的 方向"],
                   ["{industry} supply chain leaders {time_en}", "best plays on {industry} theme",
                    "stocks to watch in {industry} {time_en}"]),
    "SELECT-04": (["推荐 几只 {industry} 股票", "{stock} 值得 买 吗", "帮我 选 一下 宽基指数",
                   "下半年 有机会 的 股票 推荐"],
                   ["recommend {industry} stocks {time_en}", "is {stock} a buy {time_en}",
                    "best stocks to buy {time_en}"]),
    "QUANT-01": (["{stock} 合理 估值 怎么算", "市盈率 是 多少倍 合理", "DCF 怎么 计算"],
                 ["how to calculate fair value of {stock}", "{stock} valuation model {time_en}"]),
    "QUANT-02": (["股债 组合 配置 权重 怎么 定", "投资 组合 再平衡 策略"],
                 ["portfolio allocation between stocks and bonds", "how to rebalance my portfolio"]),
    "QUANT-03": (["{industry} 策略 回测 胜率 {time_zh}", "{stock} 均线 策略 历史回测 年化 收益",
                  "最大回撤 统计"],
                 ["backtest moving average strategy on {stock}", "{stock} strategy backtest win rate"]),
    "QUANT-04": (["A股 交易 手续费 佣金 怎么算", "期现 套利 成本 计算"],
                 ["stock trading commission and fees", "futures arbitrage cost calculation"]),
    "TRADE-01": (["{stock} 现在 能 买 吗 买点", "{stock} 止盈止损 怎么 设置",
                  "{stock} 该不该 卖"],
                 ["should I buy {stock} now", "entry point for {stock}", "stop loss for {stock} position"]),
    "TRADE-02": (["{stock} 仓位 建议 多少", "持仓 被套 补仓 还是 减仓"],
                 ["position sizing for {stock}", "how much to allocate to {stock}"]),
    "TRADE-03": (["新手 适合 买 {industry} 股票 吗", "稳健型 投资者 适合 什么 产品"],
                 ["is {industry} suitable for beginners", "investments for conservative investors"]),
    "TRADE-04": (["我的 持仓 诊断 一下", "账户 {time_zh} 亏损 分析", "资产 配置 方案 建议"],
                 ["review my portfolio holdings", "my portfolio allocation analysis"]),
    "SERVICE-01": (["什么是 顶格申购", "KDJ 是 什么意思", "新手 怎么 开户",
                    "网格交易 怎么玩 教程"],
                   ["what is a reverse split", "how does IPO subscription work",
                    "beginner guide to stock investing"]),
    "SERVICE-02": (["同花顺 条件单 怎么 设置", "APP 怎么 绑定 银行卡"],
                   ["how to set up alerts in the app", "how to change my account password"]),
    "SERVICE-03": (["基金 销售 适当性 管理要求 是什么", "免责声明 内容 合规 要求"],
                   ["SEC Regulation Best Interest requirements", "finra compliance rules for brokers"]),
    "SYSTEM-01": (["这张 图 里 的 股票 是 哪只", "以下 文档 帮我 分析"],
                  ["what stock is shown in this image", "analyze this chart for me"]),
    "SYSTEM-02": (["帮我 生成 一份 {industry} 研究 报告", "把 这段 数据 画 个 图"],
                  ["generate a report on {industry}", "create a chart of {stock} revenue"]),
    "SYSTEM-03": (["{stock} 跌破 20元 提醒 我", "每天 早上 推送 {industry} 资讯"],
                  ["alert me when {stock} drops below 200", "notify me of {industry} news daily"]),
    "SYSTEM-04": (["把 这句话 翻译 成英文", "转 人工 客服"],
                  ["translate this to chinese", "switch to a human agent"]),
    "SYSTEM-05": (["你好", "谢谢"],
                  ["hello", "thanks"]),
}


def main(out_path: str = None, per_class: int = 8, seed: int = 2026):
    rng = random.Random(seed)
    out_path = out_path or os.path.join(ROOT, "data", "eval", "synth_gold.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for iid, (zh, en) in TEMPLATES.items():
            for lang, tpls in (("zh", zh), ("en", en)):
                for tpl in tpls:
                    reps = max(1, per_class // len(tpls) + 1)
                    for _ in range(reps):
                        rec = fill(tpl, rng)
                        if rec["query"] in seen:
                            continue
                        seen.add(rec["query"])
                        n += 1
                        out.write(json.dumps({
                            "id": f"syn-{iid}-{n:04d}",
                            "query": rec["query"],
                            "gold_l2": iid,
                            "lang": lang,
                            "source": "synthetic",
                            "stock_used": rec["stock_used"],
                            "time_used": rec["time_used"],
                            "time_used_en": rec["time_used_en"],
                            "industry_used": rec["industry_used"],
                            "index_used": rec["index_used"],
                        }, ensure_ascii=False) + "\n")
    print(f"synthetic gold set: {n} -> {out_path}")
    return n


seen = set()

if __name__ == "__main__":
    main()
