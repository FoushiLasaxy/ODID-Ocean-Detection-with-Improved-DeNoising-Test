"""
单轮训练逻辑（train one epoch）
这个文件被 `train.py` 在 epoch 循环里调用：
1. 前向 + 反向
2. optimizer 更新
3. 训练指标统计（loss/acc）
4. 打印过程日志
"""

import time
from typing import Dict, Optional

import torch

from src.utils.meters import AverageMeter, accuracy


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch: int,
    logger,
    log_interval: int = 20,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp_enabled: bool = False,
    grad_clip_norm: float = 0.0,
) -> Dict[str, float]:
    # 返回这一轮训练的汇总指标，供 `train.py` 记录到 csv 并打印
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc1")
    batch_time = AverageMeter("batch_time")

    end = time.time()
    for step, (images, targets) in enumerate(loader, start=1):
        # 数据先搬到 device，再进行前向/反向
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            # autocast 只在 amp_enabled=True 时生效（CUDA）
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None and amp_enabled:
            # AMP 路径：混合精度训练，配合 GradScaler 更稳
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通 FP32 训练路径
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if step % max(1, log_interval) == 0 or step == len(loader):
            # 这里打印的是“当前 epoch 的滚动平均值”，不是单步 step 瞬时值
            logger.info(
                "Epoch [%d] Step [%d/%d] loss=%.4f acc1=%.2f lr=%.6g batch_time=%.3fs",
                epoch,
                step,
                len(loader),
                loss_meter.avg,
                acc_meter.avg,
                optimizer.param_groups[0]["lr"],
                batch_time.avg,
            )

    return {
        # train.py 会把这些字段和验证指标合并后写入 `metrics.csv`
        "train_loss": loss_meter.avg,
        "train_acc1": acc_meter.avg,
        "train_batch_time": batch_time.avg,
    }
