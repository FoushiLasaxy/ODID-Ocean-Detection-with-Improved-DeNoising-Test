"""
训练入口脚本
- `src/data/*`   准备数据集和 DataLoader
- `src/models/*` 创建模型
- `src/losses/*` 创建损失函数
- `src/engine/*` 执行训练/验证
- `src/utils/*`  log、checkpoint、随机种子等工具

最终结果会保存到 `outputs/<exp_name>/`，、metrics.csv、best.pt、last.pt
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.data import build_dataloaders
from src.engine import train_one_epoch, validate
from src.losses import build_criterion
from src.models import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import MetricsLogger, create_logger, save_json
from src.utils.seed import seed_everything


def parse_args():
    # 这里把常用训练参数开放成命令行参数，方便临时改实验设置，不用每次都去改 yaml 配置文件
    parser = argparse.ArgumentParser(description="OceanClsProject image classification training")
    parser.add_argument("--config", type=str, default="configs/configs/default.yaml")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="cifar10/cifar100/fashionmnist/fake/imagefolder")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fake-data", action="store_true", help="Shortcut: use torchvision FakeData for smoke test")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    # 从 yaml 读取基础配置
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    # 把命令行参数覆盖到 config 上
    cfg = copy.deepcopy(cfg)
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.model is not None:
        cfg.setdefault("model", {})["name"] = args.model
    if args.dataset is not None:
        cfg["data"]["name"] = args.dataset
    if args.fake_data:
        cfg["data"]["name"] = "fake"
    if args.img_size is not None:
        cfg["data"]["img_size"] = args.img_size
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.exp_name is not None:
        cfg["output"]["exp_name"] = args.exp_name
    if args.output_root is not None:
        cfg["output"]["root_dir"] = args.output_root
    if args.num_classes is not None:
        cfg["data"]["num_classes"] = args.num_classes
    if args.amp:
        cfg["train"]["amp"] = True
    if args.no_amp:
        cfg["train"]["amp"] = False
    return cfg


def resolve_device(device_name: str) -> torch.device:
    #自动选择 CUDA 去算,要是真遇到wsl兼容问题的时候还可以用cpu先凑合用着
    #不过用vsc+wsl之后这种兼容性问题就再也没出现过
    #不敢想纯cpu跑会有多慢
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    # 优化器构建统一放在这里，如果后续要整点扩展也能更方便点
    optim_cfg = cfg.get("optimizer", {})
    train_cfg = cfg["train"]
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    name = str(optim_cfg.get("name", "adamw")).lower()

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            weight_decay=wd,
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {optim_cfg.get('name')}")


def build_scheduler(cfg: Dict[str, Any], optimizer):
    # 额外任务3：动态的学习率
    sch_cfg = cfg.get("scheduler", {})
    name = str(sch_cfg.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get("step_size", 10)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )
    if name == "cosine":
        epochs = int(cfg["train"]["epochs"])
        min_lr = float(sch_cfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)
    raise ValueError(f"Unsupported scheduler: {sch_cfg.get('name')}")


def maybe_warmup_lr(cfg: Dict[str, Any], optimizer, epoch: int):
    # 这里把 scheduler.warmup_epochs 设为 0，自动跳过
    sch_cfg = cfg.get("scheduler", {})
    warmup_epochs = int(sch_cfg.get("warmup_epochs", 0))
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    base_lr = float(cfg["train"]["lr"])
    scale = epoch / max(1, warmup_epochs)
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def main():
    # main顺序：
    # 1) 读配置 + 覆盖参数
    # 2) 准备输出目录和log
    # 3) 构建数据、模型、loss、optimizer、scheduler
    # 4) epoch 循环：训练 -> 验证 -> 保存指标/权重
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    seed_everything(int(cfg.get("seed", 42)))

    device = resolve_device(str(cfg.get("device", "auto")).lower())
    output_cfg = cfg["output"]
    exp_dir = Path(output_cfg.get("root_dir", "./outputs")) / output_cfg.get("exp_name", "exp01")
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(exp_dir / output_cfg.get("log_name", "train.log"))
    metrics_logger = MetricsLogger(exp_dir / output_cfg.get("metrics_name", "metrics.csv"))
    # 能跑的话就把参数存一下
    save_json(cfg, exp_dir / "config_used.json")

    logger.info("Experiment directory: %s", exp_dir)
    logger.info("Device: %s", device)
    logger.info("Config: %s", json.dumps(cfg, ensure_ascii=False))

    train_loader, val_loader, meta = build_dataloaders(cfg)
    # num_classes 优先从数据集自动推断
    num_classes = int(meta["num_classes"])
    logger.info(
        "Dataset ready: name=%s train_size=%d val_size=%d num_classes=%d",
        cfg["data"]["name"],
        meta["train_size"],
        meta["val_size"],
        num_classes,
    )

    model = build_model(cfg, num_classes=num_classes).to(device)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    amp_enabled = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    # GradScaler 只有 CUDA + AMP 时才真正会启用
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled) if device.type == "cuda" else None

    start_epoch = 1
    best_acc1 = -1.0
    if args.resume:
        # 恢复训练：模型、优化器、scheduler、scaler
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc1 = float(ckpt.get("best_acc1", -1.0))
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    total_start = time.time()
    epochs = int(cfg["train"]["epochs"])
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 0.0))

    for epoch in range(start_epoch, epochs + 1):
        # 每轮开始前可选 warmup，再进入训练 + 验证
        epoch_start = time.time()
        maybe_warmup_lr(cfg, optimizer, epoch)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            log_interval=int(cfg["train"].get("log_interval", 20)),
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = validate(model, val_loader, criterion, device, logger=logger, prefix=f"Val Epoch [{epoch}]")

        if scheduler is not None:
            # 当前实现是按 epoch step
            scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            # 这一行会被写到 metrics.csv，方便后面绘图
            "epoch": epoch,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
            **train_metrics,
            **val_metrics,
        }
        metrics_logger.log(row)

        is_best = val_metrics["val_acc1"] > best_acc1
        if is_best:
            best_acc1 = val_metrics["val_acc1"]

        state = {
            # checkpoint 
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None and amp_enabled else None,
            "best_acc1": best_acc1,
            "config": cfg,
        }

        last_path = exp_dir / output_cfg.get("save_last_name", "last.pt")
        best_path = exp_dir / output_cfg.get("save_best_name", "best.pt")
        save_checkpoint(state, last_path)
        if is_best:
            # 自动保存当前最优模型（参考 val_acc1）
            save_checkpoint(state, best_path)
            logger.info("New best model saved: epoch=%d val_acc1=%.2f path=%s", epoch, best_acc1, best_path)

        logger.info(
            "Epoch [%d/%d] done: train_loss=%.4f train_acc1=%.2f val_loss=%.4f val_acc1=%.2f val_acc5=%.2f time=%.2fs",
            epoch,
            epochs,
            train_metrics["train_loss"],
            train_metrics["train_acc1"],
            val_metrics["val_loss"],
            val_metrics["val_acc1"],
            val_metrics["val_acc5"],
            epoch_time,
        )

    total_time = time.time() - total_start
    logger.info("Training finished in %.2fs. Best val_acc1=%.2f", total_time, best_acc1)


if __name__ == "__main__":
    main()
