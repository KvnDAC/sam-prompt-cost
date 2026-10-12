"""Fase A — Campo receptivo efectivo (ERF, Luo et al. 2016) del image encoder de SAM ViT-B.

Para un token de consulta, se toma la proyección de su vector de salida en una dirección aleatoria fija y se
calcula |∂salida/∂píxel|, promediado sobre canales RGB e imágenes reales del test. Se mide en varias
profundidades y se compara con el campo receptivo teórico:
  bloques 0–1 (ventanas 14×14 SIN desplazamiento) → solo la ventana propia (224×224 px)
  desde el bloque 2 (atención global) → toda la imagen.
Se usan dos tokens vecinos en diagonal, (27,27) y (28,28), que caen en ventanas distintas.

Salida: results/erf.json, cache/outputs/erf_maps.npz
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from samcost.common import CACHE, RESULTS, load_sam, save_json, set_threads
from samcost.data import SamPreprocessor, load_image, read_split

WIN = 14  # tamaño de ventana en tokens; rejilla con padding de 70×70 → ventanas alineadas a múltiplos de 14


def home_window(t):
    a = (t // WIN) * WIN
    return a * 16, (a + WIN) * 16  # píxeles [ini, fin) en el marco 1024


def mass_stats(g, ty, tx):
    tot = g.sum()
    y0, y1 = home_window(ty); x0, x1 = home_window(tx)
    inwin = g[y0:y1, x0:x1].sum() / tot
    cy, cx = ty * 16 + 8, tx * 16 + 8
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
    order = np.argsort(r)
    cum = np.cumsum(g.ravel()[order]) / tot
    r50 = float(r[order][np.searchsorted(cum, 0.5)])
    r90 = float(r[order][np.searchsorted(cum, 0.9)])
    return {"mass_in_home_window": float(inwin), "radius50_px": r50, "radius90_px": r90}


def main():
    set_threads()
    torch.manual_seed(0)
    sam = load_sam()
    enc = sam.image_encoder
    pp = SamPreprocessor(sam)
    ids = read_split("test")[:6]
    depths_a = [0, 1, 2, 5, 11, "neck"]
    depths_b = [0, 1]
    tok_a, tok_b = (28, 28), (27, 27)
    v768 = torch.nn.functional.normalize(torch.randn(768), dim=0)
    v256 = torch.nn.functional.normalize(torch.randn(256), dim=0)
    acc = {}
    t0 = time.time()
    for n, iid in enumerate(ids):
        x, _, _ = pp(load_image(iid))
        x = x.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            h = enc.patch_embed(x) + enc.pos_embed
            outs = {}
            for i, blk in enumerate(enc.blocks):
                h = checkpoint(blk, h, use_reentrant=False)
                outs[i] = h
            outs["neck"] = enc.neck(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        jobs = [(d, tok_a) for d in depths_a] + [(d, tok_b) for d in depths_b]
        for d, (ty, tx) in jobs:
            o = outs[d]
            v = v256 if d == "neck" else v768
            s = (o[0, ty, tx, :].float() * v).sum()
            g, = torch.autograd.grad(s, x, retain_graph=True)
            gm = g[0].abs().sum(0).float().numpy()
            key = f"{d}@{ty},{tx}"
            acc[key] = acc.get(key, 0) + gm / gm.sum()
        print(f"[erf] {n + 1}/{len(ids)} {(time.time() - t0) / 60:.1f} min", flush=True)
    res = {"images": ids, "tokens": {"A": tok_a, "B": tok_b}, "window_tokens": WIN, "stats": {}}
    for key, g in acc.items():
        g = g / len(ids)
        d, t = key.split("@")
        ty, tx = map(int, t.split(","))
        res["stats"][key] = mass_stats(g, ty, tx)
        acc[key] = g.astype(np.float32)
    save_json(res, RESULTS / "erf.json")
    (CACHE / "outputs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / "outputs" / "erf_maps.npz", **{k.replace("@", "_at_").replace(",", "_"): v for k, v in acc.items()})
    for k, v in res["stats"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
