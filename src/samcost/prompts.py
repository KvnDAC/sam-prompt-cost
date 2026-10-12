"""Prompts oráculo y simulación de clics (protocolo estándar de segmentación interactiva)."""
import numpy as np
from scipy import ndimage as ndi


def interior_point(region: np.ndarray):
    """Punto más interior de una región (máximo de la transformada de distancia al borde). Devuelve (x, y)."""
    d = ndi.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return np.array([x, y], float), float(d[y, x])


def bbox(region: np.ndarray):
    ys, xs = np.where(region)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)


def next_click(pred: np.ndarray, gt: np.ndarray):
    """Siguiente clic: en la región de error más grande (por profundidad interior), positivo si es falso
    negativo, negativo si es falso positivo (Xu et al. 2016; Sofiiuk et al. 2022)."""
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(~gt, pred)
    best = None
    for region, label in ((fn, 1), (fp, 0)):
        if region.any():
            p, depth = interior_point(region)
            if best is None or depth > best[2]:
                best = (p, label, depth)
    return best  # None si la predicción es perfecta
