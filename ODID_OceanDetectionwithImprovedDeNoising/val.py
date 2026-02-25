"""
独立验证脚本
单独加载 checkpoint 
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.data import build_dataloaders
from src.engine import validate
from src.losses import build_criterion
from src.models import build_model
from src.utils.logger import create_logger


def parse_args():
    # 这里保留了和 train.py 类似的覆盖参数，方便在不同数据集/模型配置之间快速切换
    parser = argparse.ArgumentParser(description="Validation / inference on validation set")
    parser.add_argument("--config", type=str, default="configs/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--fake-data", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str):
    # 和 train.py 一样，支持 auto 自动选设备
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def main():
    # 主流程：
    # 1) 读配置并做命令行覆盖
    # 2) 构建验证数据、模型、loss
    # 3) 加载 checkpoint
    # 4) 调用 `src/engine/validate.py` 输出结果
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.dataset is not None:
        cfg["data"]["name"] = args.dataset
    if args.model is not None:
        cfg.setdefault("model", {})["name"] = args.model
    if args.fake_data:
        cfg["data"]["name"] = "fake"
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.img_size is not None:
        cfg["data"]["img_size"] = args.img_size
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    if args.num_classes is not None:
        cfg["data"]["num_classes"] = args.num_classes

    device = resolve_device(args.device)
    out_dir = Path(cfg["output"].get("root_dir", "./outputs")) / cfg["output"].get("exp_name", "exp01")
    logger = create_logger(out_dir / "val.log", name="ocean_cls_val")

    train_loader, val_loader, meta = build_dataloaders(cfg)
    # 这里只需要 val_loader，保留 train_loader 是因为要接口统一返回值
    _ = train_loader
    model = build_model(cfg, num_classes=int(meta["num_classes"])).to(device)
    criterion = build_criterion(cfg)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    # validate 会同时打印日志并返回一个 dict，这里直接 print 方便命令行查看
    metrics = validate(model, val_loader, criterion, device, logger=logger, prefix="Validation")
    print(metrics)


if __name__ == "__main__":
    main()
