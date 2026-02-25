"""
损失函数模块
这个文件目前给分类任务返回 `CrossEntropyLoss`
在初始化阶段调用后，如果扩展成别的任务，可以在这里再加点别的
"""

from typing import Dict

import torch.nn as nn


def build_criterion(cfg: Dict) -> nn.Module:
    # 从 config 读取 label smoothing 等训练参数，返回可直接调用的损失函数
    # 训练阶段在 `train_one_epoch.py` 执行，验证阶段在 `validate.py` 处理
    train_cfg = cfg.get("train", {})
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
