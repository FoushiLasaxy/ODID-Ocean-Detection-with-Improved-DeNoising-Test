"""
验证逻辑部分
这个文件被 `train.py`（每轮训练后验证一次）和 `val.py`（单独验证脚本）调用
职责是只做前向推理和指标统计，不做参数更新
"""

import time
from typing import Dict

import torch

from src.utils.meters import AverageMeter, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device, logger=None, prefix: str = "Val") -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc1")
    acc5_meter = AverageMeter("acc5")
    batch_time = AverageMeter("batch_time")

    end = time.time()
    for images, targets in loader:
        # 验证阶段和训练一样要搬运数据
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        num_classes = outputs.shape[1]
        # 如果类别数不足 5 时，top5 没意义，这里自动降级为 top1
        topk = (1, 5) if num_classes >= 5 else (1,)
        accs = accuracy(outputs, targets, topk=topk)
        acc1 = accs[0]
        acc5 = accs[1] if len(accs) > 1 else accs[0]

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(acc1.item(), images.size(0))
        acc5_meter.update(acc5.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    if logger is not None:
        # train.py/val.py 传了 logger 时会写log文件；不传时也能单独复用这个函数
        logger.info(
            "%s: loss=%.4f acc1=%.2f acc5=%.2f batch_time=%.3fs",
            prefix,
            loss_meter.avg,
            acc1_meter.avg,
            acc5_meter.avg,
            batch_time.avg,
        )

    return {
        # 返回汇总结果给上层脚本做打印、保存、比较 best checkpoint
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg,
        "val_batch_time": batch_time.avg,
    }
