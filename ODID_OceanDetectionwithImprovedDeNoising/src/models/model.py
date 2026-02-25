"""
模型构建模块（nn.Module）
"""

from typing import Dict

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # 特征提取 -> 分类头
        x = self.features(x)
        return self.classifier(x)


def _replace_resnet_head(model: nn.Module, num_classes: int, dropout: float = 0.0) -> nn.Module:
    # 把 torchvision 的 resnet 分类头替换成当前任务类别数
    # 这样就能把通用 backbone 用在自定义分类任务上
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def build_model(cfg: Dict, num_classes: int) -> nn.Module:
    # 对外统一入口：训练脚本和验证脚本都通过它建模型
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "resnet18")).lower()
    pretrained = bool(model_cfg.get("pretrained", False))
    dropout = float(model_cfg.get("dropout", 0.0))

    if model_name == "resnet18":
        # 常见分类 baseline
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model = _replace_resnet_head(model, num_classes=num_classes, dropout=dropout)
    elif model_name == "mobilenet_v3_small":
        # 如果还是跑的不理想，可以用下面跑
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        layers = list(model.classifier.children())
        layers[-1] = nn.Linear(in_features, num_classes)
        if dropout > 0 and hasattr(layers[0], "p"):
            layers[0].p = dropout
        model.classifier = nn.Sequential(*layers)
    elif model_name == "simple_cnn":
        # 自定义小网络
        in_channels = 1 if cfg["data"]["name"].lower() == "fashionmnist" else 3
        model = SimpleCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model: {model_cfg.get('name')}")

    return model
