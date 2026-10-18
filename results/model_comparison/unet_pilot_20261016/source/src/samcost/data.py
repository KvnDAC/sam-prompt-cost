"""Carga de Kvasir-SEG y preprocesamiento identico al de SAM (ResizeLongestSide 1024 + normalizacion + padding)."""
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

from .common import DATA

KVASIR = DATA / "Kvasir-SEG"


def read_split(name: str) -> list[str]:
    return [l.strip() for l in open(DATA / "splits" / f"{name}.txt") if l.strip()]


def load_image(iid: str) -> np.ndarray:
    return np.array(Image.open(KVASIR / "images" / f"{iid}.jpg").convert("RGB"))


def load_mask(iid: str) -> np.ndarray:
    """Mascara GT binaria. Las mascaras oficiales son JPEG: se umbraliza en 127."""
    return np.array(Image.open(KVASIR / "masks" / f"{iid}.jpg").convert("L")) > 127


def instances(gt: np.ndarray, rel: float = 0.01, min_px: int = 64) -> list[np.ndarray]:
    """Componentes conexas de la GT usadas para generar prompts oraculo.
    Se ignoran motas de compresion JPEG (< rel * componente mayor o < min_px)."""
    lab, n = ndi.label(gt)
    if n == 0:
        return []
    sizes = ndi.sum(gt, lab, range(1, n + 1))
    thr = max(rel * sizes.max(), min_px)
    return [lab == (j + 1) for j in range(n) if sizes[j] >= thr]


class SamPreprocessor:
    def __init__(self, sam):
        from segment_anything.utils.transforms import ResizeLongestSide

        self.sam = sam
        self.tf = ResizeLongestSide(sam.image_encoder.img_size)

    def __call__(self, image: np.ndarray):
        x = self.tf.apply_image(image)
        input_size = x.shape[:2]
        t = torch.as_tensor(x, device=self.sam.device).permute(2, 0, 1).contiguous()[None].float()
        t = self.sam.preprocess(t)  # normaliza y rellena a 1024x1024
        return t, image.shape[:2], input_size

    def coords(self, xy: np.ndarray, original_size) -> np.ndarray:
        return self.tf.apply_coords(xy, original_size)

    def boxes(self, b: np.ndarray, original_size) -> np.ndarray:
        return self.tf.apply_boxes(b, original_size)


def gt_lowres(gt: np.ndarray, input_size, size: int = 256) -> np.ndarray:
    """GT llevada al marco de 256x256 de las mascaras de baja resolucion de SAM
    (redimensionado al lado largo y relleno con ceros), para entrenar en ese espacio."""
    h, w = input_size
    hs, ws = int(round(h * size / 1024)), int(round(w * size / 1024))
    m = Image.fromarray(gt.astype(np.uint8) * 255).resize((ws, hs), Image.BILINEAR)
    out = np.zeros((size, size), np.float32)
    out[:hs, :ws] = np.asarray(m, np.float32) / 255.0
    return out
