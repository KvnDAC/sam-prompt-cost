"""Atributos por imagen para análisis por subgrupo: área del pólipo, nº de pólipos, recuadro negro del marcador,
marcador verde visible. Heurísticas simples y documentadas (se validan a ojo en la Fase C).

Salida: results/image_attributes_<split>.csv
"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy import ndimage as ndi

from samcost.common import RESULTS
from samcost.data import instances, load_image, load_mask, read_split


def rect_region(mask, min_frac=0.015):
    """¿Hay una región conexa casi rectangular (área/caja > 0,9) de al menos min_frac del área?"""
    lab, n = ndi.label(mask)
    if n == 0:
        return False
    for sl, j in zip(ndi.find_objects(lab), range(1, n + 1)):
        reg = lab[sl] == j
        if reg.sum() >= min_frac * mask.size and reg.mean() > 0.9:
            return True
    return False


def main(split="test"):
    rows = []
    for iid in read_split(split):
        img = load_image(iid).astype(int)
        gt = load_mask(iid)
        h, w = gt.shape
        q = img[h // 2:, : w // 2]  # cuadrante inferior izquierdo (donde está el marcador)
        black = (q.max(2) < 12)
        green = (q[..., 1] > 110) & (q[..., 1] - q[..., 0] > 50) & (q[..., 1] - q[..., 2] > 10)
        rows.append({"id": iid, "gt_area": float(gt.mean()), "n_inst": len(instances(gt)),
                     "black_box": int(rect_region(black)), "green_marker": int(rect_region(green, 0.01)),
                     "width": w, "height": h})
    with open(RESULTS / f"image_attributes_{split}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    a = np.array([r["gt_area"] for r in rows])
    print(f"{split}: n={len(rows)} recuadro negro={sum(r['black_box'] for r in rows)} marcador verde={sum(r['green_marker'] for r in rows)} "
          f"multi-pólipo={sum(r['n_inst'] > 1 for r in rows)} área<2%={int((a < 0.02).sum())} área mediana={np.median(a):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
