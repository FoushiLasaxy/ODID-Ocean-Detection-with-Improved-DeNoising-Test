"""
随机种子
在 `train.py` 执行时调用
"""

import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    # 看资料说`benchmark=True` 能让 cudnn 自选更快算法，速度更好。不过复现性就可能没那么行了
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
