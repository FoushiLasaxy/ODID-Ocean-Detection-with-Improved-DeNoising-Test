"""
日志保存
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


def create_logger(log_file: Path, name: str = "ocean_cls") -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        # 避免重复添加 handler（不然同一条日志会打印多次）
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class MetricsLogger:
    def __init__(self, csv_path: Path):
        # 这个类专门管每轮指标 csv，train.py 每个 epoch 会调一次 log(row)
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: Optional[Iterable[str]] = None

    def log(self, row: Dict):
        # 以追加方式写入，第一次会自动写表头
        # 最终结果文件通常是 `outputs/<exp>/metrics.csv`
        row = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items()}
        is_new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def save_json(data: Dict, path: Path):
    # 保存一份“本次实际使用的配置”，避免后面忘了参数怎么设的
    # 不然又出问题
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
