"""
数据集与 DataLoader 构建模块
这个文件会被 `train.py` 和 `val.py` 调用（通过 `src.data.build_dataloaders`）
作用是把 config 里的数据集配置落地成：
1. train dataset ， val dataset
2. train dataloader  ，val dataloader
3. 其余的元信息（类别数、样本数等）
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .transforms import build_eval_transform, build_train_transform


def _infer_num_classes(dataset_obj) -> int:
    # 尽量从 torchvision 数据集对象里自动推断类别数
    # 这样训练脚本减少手工填参工作量
    if hasattr(dataset_obj, "classes"):
        return len(dataset_obj.classes)
    if hasattr(dataset_obj, "targets"):
        targets = dataset_obj.targets
        if isinstance(targets, torch.Tensor):
            return int(targets.max().item() + 1)
        return len(set(targets))
    raise ValueError("Unable to infer num_classes from dataset object.")


def _build_dataset_pair(cfg: Dict) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    # 按 config 里的 data.name 构建训练集和验证集
    data_cfg = cfg["data"]
    name = data_cfg["name"].lower()
    root = Path(data_cfg.get("data_dir", "./data"))
    img_size = int(data_cfg.get("img_size", 224))
    normalize = bool(data_cfg.get("normalize", True))
    train_tfm = build_train_transform(img_size, name, normalize=normalize)
    eval_tfm = build_eval_transform(img_size, name, normalize=normalize)
    download = bool(data_cfg.get("download", True))

    if name == "cifar10":
        # torchvision 自带数据集，适合做图像分类训练流程测试
        train_set = datasets.CIFAR10(root=str(root), train=True, transform=train_tfm, download=download)
        val_set = datasets.CIFAR10(root=str(root), train=False, transform=eval_tfm, download=download)
    elif name == "cifar100":
        train_set = datasets.CIFAR100(root=str(root), train=True, transform=train_tfm, download=download)
        val_set = datasets.CIFAR100(root=str(root), train=False, transform=eval_tfm, download=download)
    elif name == "fashionmnist":
        train_set = datasets.FashionMNIST(root=str(root), train=True, transform=train_tfm, download=download)
        val_set = datasets.FashionMNIST(root=str(root), train=False, transform=eval_tfm, download=download)
    elif name == "imagefolder":
        # 自定义数据集目录时，按 ImageFolder 规范组织数据：
        # train/class_x/*.jpg, val/class_x/*.jpg
        train_dir = data_cfg.get("imagefolder_train_dir")
        val_dir = data_cfg.get("imagefolder_val_dir")
        if not train_dir or not val_dir:
            raise ValueError("imagefolder dataset requires data.imagefolder_train_dir and data.imagefolder_val_dir")
        train_set = datasets.ImageFolder(root=str(train_dir), transform=train_tfm)
        val_set = datasets.ImageFolder(root=str(val_dir), transform=eval_tfm)
    elif name == "fake":
        # 用FakeData 跑 smoke test ，测试能不能跑
        num_classes = int(data_cfg.get("num_classes", 10))
        train_size = int(data_cfg.get("fake_train_size", 512))
        val_size = int(data_cfg.get("fake_val_size", 128))
        image_size = (3, img_size, img_size)
        train_set = datasets.FakeData(
            size=train_size,
            image_size=image_size,
            num_classes=num_classes,
            transform=train_tfm,
        )
        val_set = datasets.FakeData(
            size=val_size,
            image_size=image_size,
            num_classes=num_classes,
            transform=eval_tfm,
        )
    else:
        raise ValueError(f"Unsupported dataset: {data_cfg['name']}")

    return train_set, val_set


def build_dataloaders(cfg: Dict):
    # 对外主入口：train.py / val.py 就是拿这个函数来准备数据
    # 返回 dataloader + meta，方便后面建模型时自动拿 num_classes
    train_set, val_set = _build_dataset_pair(cfg)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    meta = {
        # meta 给 `train.py` 用来打印日志、构建模型
        "num_classes": _infer_num_classes(train_set) if data_cfg["name"].lower() != "fake" else int(data_cfg["num_classes"]),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "class_names": getattr(train_set, "classes", None),
    }
    return train_loader, val_loader, meta
