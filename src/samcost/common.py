"""Utilidades comunes: rutas, semillas, carga de SAM y entorno."""
import json
import os
import platform
import random
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CKPT = Path(os.environ.get("SAM_CKPT", ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"))
SAM_MD5 = "01ec64d29a2fca3f0661936605ae66f8"  # md5 del checkpoint oficial sam_vit_b_01ec64.pth


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def set_threads(n: int | None = None) -> int:
    n = n or int(os.environ.get("SAMCOST_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(n)
    return n


def load_sam(model_type: str = "vit_b", checkpoint: Path | None = None, device: str = "cpu"):
    from segment_anything import sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint or CKPT))
    sam.eval().to(device)
    for p in sam.parameters():
        p.requires_grad_(False)
    return sam


def env_info() -> dict:
    import segment_anything  # noqa: F401

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
    }
    info["packages"] = {}
    for package in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "pillow", "segment-anything"):
        try:
            info["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info["packages"][package] = None
    try:
        direct = metadata.distribution("segment-anything").read_text("direct_url.json")
        info["sam_source"] = json.loads(direct) if direct else None
    except metadata.PackageNotFoundError:
        info["sam_source"] = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    temporary.replace(path)
