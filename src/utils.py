import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

# Required for CuBLAS deterministic mode on CUDA >= 10.2.
# Must be set before any CuBLAS workspace is allocated; module-level guarantees that.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_config_used(config: dict, model_log_dir: Path) -> None:
    model_log_dir.mkdir(parents=True, exist_ok=True)
    with open(model_log_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_environment(model_log_dir: Path) -> None:
    model_log_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    import sys
    lines.append(f"Python: {sys.version}")

    lines.append(f"PyTorch: {torch.__version__}")

    try:
        import torchvision
        lines.append(f"torchvision: {torchvision.__version__}")
    except ImportError:
        lines.append("torchvision: not installed")

    lines.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            lines.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    for pkg in ("ultralytics", "timm", "sklearn", "statsmodels"):
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            lines.append(f"{pkg}: {mod.__version__}")
        except ImportError:
            lines.append(f"{pkg}: not installed")

    with open(model_log_dir / "environment.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
