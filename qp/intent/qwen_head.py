# -*- coding: utf-8 -*-
"""Qwen2.5-0.5B 判别式三头模型的线上推理适配器（占位实现）。

生产形态（见 docs/03 训练技术方案）：
  - 服务器端训练：Qwen2.5-0.5B 编码器 + L1(8)/L2(42)/槽位 三个线性头
  - 导出 ONNX（INT8 PTQ）后由本适配器加载，单次前向、无自回归解码
  - 输出经温度缩放校准（验证集拟合 T，ECE<0.05）

本文件实现接口与后处理（软解码 score = p_L1(域) × p_L2(类)，top-k 跨域），
权重未就位时 predict 返回空、级联自动跳过 —— 保证骨架现在即可运行。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from qp.common.taxonomy import l2_domain
from qp.intent.base import IntentModel


class QwenHeadIntentModel(IntentModel):
    name = "qwen-0.5b-judge-head"

    def __init__(self, onnx_path: Optional[str] = None,
                 calib_path: Optional[str] = None,
                 l1_threshold: float = 0.75,
                 topk: int = 3):
        self.session = None
        self.temperature = 1.0
        self.l1_threshold = l1_threshold
        self.topk = topk
        self.l1_ids: List[str] = []
        self.l2_ids: List[str] = []
        if onnx_path and os.path.exists(onnx_path):
            self.load(onnx_path, calib_path)

    # ---------------- 加载 ----------------
    def load(self, onnx_path: str, calib_path: Optional[str] = None):
        import onnxruntime as ort          # 仅服务器/有权重环境安装

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.l1_ids = json.loads(meta.get("l1_ids", "[]"))
        self.l2_ids = json.loads(meta.get("l2_ids", "[]"))
        if calib_path and os.path.exists(calib_path):
            with open(calib_path, encoding="utf-8") as f:
                self.temperature = json.load(f).get("temperature", 1.0)

    def is_ready(self) -> bool:
        return self.session is not None

    # ---------------- 推理 ----------------
    def predict(self, query: str, topk: int = 3) -> List[dict]:
        """权重就位后：tokenize（需 transformers）→ session.run → 软解码。
        当前返回空列表，级联将跳过本模型。"""
        if not self.is_ready():
            return []
        # ---- 生产路径（服务器权重导出后启用）----
        # from transformers import AutoTokenizer
        # tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
        # ids = tok(query, return_tensors="np").input_ids
        # l1_logits, l2_logits, slot_logits = self.session.run(None, {"input_ids": ids})
        # l1_p = softmax(l1_logits[0] / self.temperature)
        # l2_p = sigmoid(l2_logits[0])                    # BCE 多标签头
        # score[j] = l1_p[domain(l2_ids[j])] * l2_p[j]    # 软解码，跨域 top-k
        # ...
        raise NotImplementedError("等待服务器导出 ONNX 权重后启用（见 docs/03）")
