"""Métricas por imagen y estadística (bootstrap, Wilcoxon)."""
import numpy as np
from scipy import stats


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(pred, gt).sum() / s)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    u = np.logical_or(pred, gt).sum()
    return 1.0 if u == 0 else float(np.logical_and(pred, gt).sum() / u)


def bootstrap_ci(x, n=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_test(a, b):
    """Wilcoxon de rangos con signo (pareado, dos colas) + diferencia media con IC bootstrap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    p = float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    m, lo, hi = bootstrap_ci(d)
    return {"mean_diff": m, "ci95": [lo, hi], "wilcoxon_p": p}


def fmt(m, lo, hi, nd=3):
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
