"""
一些验证小工具
其中：
- `AverageMeter`: 做平均值统计
- `accuracy`: 算 top-k 准确率
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch


@dataclass
class AverageMeter:
    name: str
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1):
        # 做连续累加后，得到平均值
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    # 通用 top-k 准确率计算函数
    # 输入是模型输出 logits 和标签，返回百分比（例如 87.5 表示 87.5%）
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: List[torch.Tensor] = []
        for k in topk:
            kk = min(k, output.size(1))
            correct_k = correct[:kk].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
