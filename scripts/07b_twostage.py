"""Dos etapas sin humano: la máscara de los tokens aprendidos (el «qué») se convierte en una caja (el «dónde»)
y se vuelve a consultar a SAM congelado con esa caja. ¿Mejora sobre los tokens solos?

Salida: results/twostage_summary.json y results/learned/<tag>/twostage_test_per_image.csv
"""
import csv, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from scipy import ndimage as ndi

from samcost.common import RESULTS, load_sam, save_json, set_threads
from samcost.data import load_mask
from samcost.decode import CachedSAM
from samcost.metrics import dice
from samcost.prompts import bbox


def main():
    set_threads()
    cs = CachedSAM(load_sam())
    out = {}
    for d in sorted(glob.glob(str(RESULTS / "learned" / "lp_k*_s*"))):
        d = Path(d)
        if not (d / "test_lowres.npz").exists():
            continue
        z = np.load(d / "test_lowres.npz")
        rows = []
        for iid, lr in zip(z["ids"], z["lowres"]):
            emb, osz, isz = cs.emb(str(iid))
            gt = load_mask(str(iid))
            m = (cs.full(torch.from_numpy(lr.astype(np.float32))[None, None], osz, isz)[0, 0] > 0).numpy()
            lab, n = ndi.label(m)
            two = np.zeros_like(gt)
            if n:
                sizes = ndi.sum(m, lab, range(1, n + 1))
                for j in range(n):
                    if sizes[j] >= max(0.01 * sizes.max(), 64):
                        with torch.no_grad():
                            lrb, _ = cs.decode(emb, osz, box=bbox(lab == (j + 1)), multimask=False)
                        two |= (cs.full(lrb, osz, isz)[0, 0] > 0).numpy()
            rows.append({"id": str(iid), "tokens_dice": dice(m, gt), "twostage_dice": dice(two, gt)})
        with open(d / "twostage_test_per_image.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        out[d.name] = {"tokens_dice": float(np.mean([r["tokens_dice"] for r in rows])),
                       "twostage_dice": float(np.mean([r["twostage_dice"] for r in rows]))}
        print(d.name, out[d.name], flush=True)
    save_json(out, RESULTS / "twostage_summary.json")


if __name__ == "__main__":
    main()
