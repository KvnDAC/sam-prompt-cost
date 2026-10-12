"""Fase A — Características intermedias (PCA por capa) y atención del decoder sin prompt vs. con caja.

Salida: cache/outputs/features_pca.npz, cache/outputs/decoder_attention.npz, results/features.json
"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from samcost.attention import capture_token_to_image
from samcost.common import CACHE, RESULTS, load_sam, save_json, set_threads
from samcost.data import SamPreprocessor, instances, load_image, load_mask, read_split
from samcost.decode import CachedSAM
from samcost.prompts import bbox

LAYERS = [2, 5, 8, 11]


def pick_images():
    rows = list(csv.DictReader(open(RESULTS / "baseline_test_per_image.csv")))
    rows.sort(key=lambda r: float(r["gt_area"]))
    multi = [r for r in rows if int(float(r["n_inst"])) > 1]
    picks = [rows[2]["id"], rows[len(rows) // 2]["id"], rows[-3]["id"]]
    if multi:
        picks.append(multi[0]["id"])
    return picks


def pca_rgb(feats_list):
    """PCA conjunta (mismas componentes para todas las imágenes de una capa) → 3 canales en [0,1]."""
    X = np.concatenate([f.reshape(-1, f.shape[-1]) for f in feats_list]).astype(np.float64)
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:3].T
    ev = (S[:3] ** 2) / (S ** 2).sum()
    outs = []
    for f in feats_list:
        y = (f.reshape(-1, f.shape[-1]) - mu) @ P
        lo, hi = np.percentile(y, 1, 0), np.percentile(y, 99, 0)
        outs.append(np.clip((y - lo) / (hi - lo + 1e-9), 0, 1).reshape(f.shape[0], f.shape[1], 3))
    return outs, ev


def main():
    set_threads()
    sam = load_sam()
    pp = SamPreprocessor(sam)
    cs = CachedSAM(sam)
    ids = pick_images()
    feats = {l: [] for l in LAYERS + ["neck"]}
    hooks = [sam.image_encoder.blocks[l].register_forward_hook(
        lambda m, i, o, l=l: feats[l].append(o[0].float().numpy())) for l in LAYERS]
    info = {"ids": ids, "input_size": {}}
    with torch.no_grad():
        for iid in ids:
            x, osz, isz = pp(load_image(iid))
            e = sam.image_encoder(x)
            feats["neck"].append(e[0].permute(1, 2, 0).numpy())
            info["input_size"][iid] = list(isz)
    for h in hooks:
        h.remove()
    pca, ev = {}, {}
    for l, fl in feats.items():
        outs, ev[str(l)] = pca_rgb(fl)
        for iid, o in zip(ids, outs):
            pca[f"{l}_{iid}"] = o.astype(np.float16)
    info["pca_explained_var_top3"] = {k: [float(a) for a in v] for k, v in ev.items()}
    # Separabilidad lineal simple por capa: distancia entre medias pólipo/fondo en unidades de la desviación
    sep = {}
    for l, fl in feats.items():
        vals = []
        for iid, f in zip(ids, fl):
            g = load_mask(iid)
            h, w = info["input_size"][iid]
            gt = torch.nn.functional.interpolate(torch.from_numpy(g[None, None].astype(np.float32)),
                                                 size=(int(round(h / 16)), int(round(w / 16))), mode="area")[0, 0].numpy()
            ff = f[: gt.shape[0], : gt.shape[1]].reshape(-1, f.shape[-1])
            lab = gt.ravel()
            if (lab > 0.5).sum() < 3 or (lab < 0.5).sum() < 3:
                continue
            mp, mb = ff[lab > 0.5].mean(0), ff[lab < 0.5].mean(0)
            d = mp - mb
            proj = ff @ d / (np.linalg.norm(d) + 1e-9)
            vals.append(float((proj[lab > 0.5].mean() - proj[lab < 0.5].mean()) / (proj.std() + 1e-9)))
        sep[str(l)] = float(np.mean(vals)) if vals else None
    info["mean_direction_separation"] = sep
    # Atención token→imagen del decoder
    att = {}
    with torch.no_grad():
        for iid in ids:
            emb, osz, isz = cs.emb(iid)
            gt = load_mask(iid)
            with capture_token_to_image(sam) as st:
                cs.decode_all4(emb, torch.zeros(1, 0, 256))
            for k, a in st.items():
                att[f"noprompt_{k}_{iid}"] = a[0].mean(0).numpy().astype(np.float16)  # (tokens, 4096)
            inst = instances(gt)
            if inst:
                with capture_token_to_image(sam) as st:
                    cs.decode(emb, osz, box=bbox(inst[0]), multimask=False)
                for k, a in st.items():
                    att[f"box_{k}_{iid}"] = a[0].mean(0).numpy().astype(np.float16)
    np.savez_compressed(CACHE / "outputs" / "features_pca.npz", **pca)
    np.savez_compressed(CACHE / "outputs" / "decoder_attention.npz", **att)
    save_json(info, RESULTS / "features.json")
    print(info)


if __name__ == "__main__":
    main()
