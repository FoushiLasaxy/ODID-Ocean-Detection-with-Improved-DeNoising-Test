"""
checkpoint 
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(state: Dict[str, Any], path: Path):
    # 统一封装保存逻辑
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    # train.py/val.py 直接用 torch.load 也能跑，这里保留成公共接口，方便后续统一处理
    return torch.load(path, map_location=map_location)
