"""图像预处理与增强（transforms）构建模块

这个文件主要被 `src/data/dataset.py` 调用，用来给训练集和验证集生成不同的
transform 流程训练集会带一点基础增强，让验证集则尽量稳定
"""

from typing import Tuple

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def _stats_for_dataset(dataset_name: str, normalize: bool) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    # 这里统一决定是否做归一化，以及用哪一套 mean/std
    # 结果会被下面的 train/eval transform 拼进去
    if not normalize:
        return (), ()
    if dataset_name.lower() == "fashionmnist":
        return MNIST_MEAN, MNIST_STD
    return IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(img_size: int, dataset_name: str, normalize: bool = True):
    # 给训练集用的 transforms：会做一点随机增强，减少模型的过拟合情况
    # 这个函数通常由 `src/data/dataset.py` 在构建 train dataset 时调用
    mean, std = _stats_for_dataset(dataset_name, normalize)
    tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
    ]
    if mean and std:
        tfms.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tfms)


def build_eval_transform(img_size: int, dataset_name: str, normalize: bool = True):
    # 给验证/推理阶段用的 transforms：尽量稳定，不加随机扰动
    # 这样每轮验证指标更可比
    mean, std = _stats_for_dataset(dataset_name, normalize)
    tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if mean and std:
        tfms.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tfms)
