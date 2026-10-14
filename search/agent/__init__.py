# -*- coding: utf-8 -*-
"""Agentic 检索（v2.1）：ReAct 循环 + 答案收束 + 复合 query 分解。

把「固定流水线 + 规则阈值兜底」升级为 LLM 决策循环：
  Planner 每轮输出 {thought, action, args} 或收束 {action: answer}：
    content_search  库内多通道召回（支持 sub_queries 并行拆查，复合 query 分解）
    web_search      web 实时兜底 + 后台自愈入库
    answer          收束：综合答案 + 引用
约束：≤3 轮 / 20s 预算；observation（结果+质量信号）每轮回传 Planner。
LLM 不可用/连续解析失败时降级为单发检索（degraded=true），不阻塞主链路。
"""
from .loop import agent_search  # noqa: F401
