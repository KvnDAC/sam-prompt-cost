"""Figuras de imagen para el documento y el repositorio (ERF, PCA, atención del decoder, ejemplos sin prompt,
comparación cualitativa). Cada figura se genera solo si existen sus insumos.

Uso: python scripts/09_figures.py [--only erf,pca,att,noprompt,qual,calib]
"""
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi

from samcost.common import CACHE, FIGURES, RESULTS, load_sam, set_threads
from samcost.data import instances, load_image, load_mask, read_split
from samcost.decode import CachedSAM
from samcost.prompts import bbox, interior_point

INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.titlecolor": INK})


def contour(ax, m, color=BLUE, lw=1.2):
    ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def frame_crop(img_1024_shape, isz):
    return slice(0, isz[0]), slice(0, isz[1])


def fig_erf():
    p = CACHE / "outputs" / "erf_maps.npz"
    if not p.exists():
        return
    z = np.load(p)
    E = json.load(open(RESULTS / "erf.json"))
    keys = [("0_at_28_28", "bloque 0 (ventana)"), ("1_at_28_28", "bloque 1 (ventana)"), ("1_at_27_27", "bloque 1 · token vecino"),
            ("2_at_28_28", "bloque 2 (1.º global)"), ("5_at_28_28", "bloque 5"), ("11_at_28_28", "bloque 11"), ("neck_at_28_28", "salida (neck)")]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.6))
    axes = axes.ravel()
    axes[-1].axis("off")
    axes[-1].text(0.0, 0.5, "Líneas azules: rejilla de ventanas\n(14×14 tokens = 224×224 px).\nPunto naranja: token consultado.\nEscala logarítmica de |∂salida/∂píxel|,\npromedio de 6 imágenes de test.",
                  fontsize=8, color=INK2, va="center")
    for ax, (k, title) in zip(axes, keys):
        g = z[k]
        ax.imshow(g + 1e-9, cmap="Greys", norm=LogNorm(vmin=max(g.max() * 1e-4, 1e-12), vmax=g.max()))
        for v in range(0, 1025, 224):
            ax.axhline(v, color=BLUE, lw=0.4, alpha=0.6); ax.axvline(v, color=BLUE, lw=0.4, alpha=0.6)
        ty, tx = (27, 27) if "27_27" in k else (28, 28)
        ax.plot(tx * 16 + 8, ty * 16 + 8, "o", color=ORANGE, ms=3)
        st = E["stats"][k.replace("_at_", "@").replace("_", ",")]
        ax.set_title(f"{title}\n{100 * st['mass_in_home_window']:.0f} % en su ventana", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIGURES / "erf.png", dpi=170); plt.close(fig)


def fig_pca_att():
    pz, az = CACHE / "outputs" / "features_pca.npz", CACHE / "outputs" / "decoder_attention.npz"
    if not pz.exists():
        return
    P, A = np.load(pz), np.load(az)
    info = json.load(open(RESULTS / "features.json"))
    ids = info["ids"]
    # (1) PCA por capa
    cols = ["imagen + GT", "bloque 2", "bloque 5", "bloque 8", "bloque 11", "embedding (neck)"]
    fig, axes = plt.subplots(len(ids), len(cols), figsize=(1.9 * len(cols), 1.85 * len(ids)))
    for r, iid in enumerate(ids):
        img, gt = load_image(iid), load_mask(iid)
        h, w = info["input_size"][iid]
        th, tw = int(round(h / 16)), int(round(w / 16))
        axes[r, 0].imshow(img); contour(axes[r, 0], gt)
        for c, l in enumerate([2, 5, 8, 11, "neck"]):
            axes[r, c + 1].imshow(P[f"{l}_{iid}"].astype(np.float32)[:th, :tw], interpolation="nearest")
        for c in range(len(cols)):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "pca_por_capa.png", dpi=160); plt.close(fig)
    # (2) atención del token de máscara (capa final token→imagen): sin prompt vs. caja
    cols = ["imagen + GT", "sin prompt", "caja oráculo"]
    fig, axes = plt.subplots(len(ids), 3, figsize=(6.6, 1.95 * len(ids)))
    for r, iid in enumerate(ids):
        img, gt = load_image(iid), load_mask(iid)
        h, w = info["input_size"][iid]
        th, tw = int(round(h / 16)), int(round(w / 16))
        axes[r, 0].imshow(img); contour(axes[r, 0], gt)
        grey = img.mean(2)
        for c, mode in enumerate(["noprompt", "box"]):
            key = f"{mode}_final_{iid}"
            ax = axes[r, 1 + c]
            if key in A.files:
                att = A[key].astype(np.float32)[1].reshape(64, 64)[:th, :tw]
                att = att / att.sum()
                gt_t = torch.nn.functional.interpolate(torch.from_numpy(gt[None, None].astype(np.float32)), size=(th, tw), mode="area")[0, 0].numpy()
                mass = float((att * gt_t).sum())
                ax.imshow(grey, cmap="Greys_r", alpha=0.35)
                ax.imshow(np.sqrt(att / att.max()), extent=(0, img.shape[1], img.shape[0], 0), cmap="magma", alpha=0.85, interpolation="nearest")
                contour(ax, gt, color="#7fd4ff", lw=1.0)
                ax.set_title(f"{100 * mass:.0f} % de la atención en el pólipo", fontsize=7.5)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, t in enumerate(cols):
        axes[0, c].set_title(t + ("" if c == 0 else "\n" + axes[0, c].get_title()), fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "atencion_decoder.png", dpi=160); plt.close(fig)


def fig_noprompt():
    p = CACHE / "outputs" / "noprompt_lowres_test.npz"
    if not p.exists():
        return
    z = np.load(p)
    ids = list(z["ids"])
    rows = {r["id"]: r for r in csv.DictReader(open(RESULTS / "baseline_test_per_image.csv"))}
    meta = json.load(open(CACHE / "meta.json"))
    sel = [ids[i] for i in np.linspace(0, len(ids) - 1, 4).astype(int)]
    fig, axes = plt.subplots(len(sel), 5, figsize=(9, 1.9 * len(sel)))
    sam = None
    for r, iid in enumerate(sel):
        img, gt = load_image(iid), load_mask(iid)
        lr = torch.from_numpy(z["lowres"][ids.index(iid)].astype(np.float32))[None]
        isz, osz = tuple(meta[iid]["input_size"]), img.shape[:2]
        full = torch.nn.functional.interpolate(lr, (1024, 1024), mode="bilinear", align_corners=False)[..., :isz[0], :isz[1]]
        full = torch.nn.functional.interpolate(full, osz, mode="bilinear", align_corners=False)[0] > 0
        axes[r, 0].imshow(img); contour(axes[r, 0], gt)
        for k in range(4):
            ax = axes[r, k + 1]; ax.imshow(img); ax.imshow(np.ma.masked_where(~full[k].numpy(), full[k].numpy()), cmap="autumn", alpha=0.55)
            contour(ax, gt)
            ax.set_title(f"token {k}: Dice {float(rows[iid][f'np_dice_t{k}']):.2f} · IoU pred {float(rows[iid][f'np_pred_iou_t{k}']):.2f}", fontsize=7)
        axes[r, 0].set_title("imagen + GT" if r == 0 else "", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIGURES / "sin_prompt_ejemplos.png", dpi=160); plt.close(fig)


def fig_qual():
    lp = sorted((RESULTS / "learned").glob("lp_k4_meanpoint_s0/test_lowres.npz")) if (RESULTS / "learned").exists() else []
    if not lp:
        return
    set_threads()
    sam = load_sam(); cs = CachedSAM(sam)
    z = np.load(lp[0]); lids = list(z["ids"])
    lpr = {r["id"]: float(r["dice"]) for r in csv.DictReader(open(lp[0].parent / "test_per_image.csv"))}
    # 4 casos: los cuartiles del Dice de los tokens aprendidos
    order = sorted(lpr, key=lpr.get)
    sel = [order[int(q * (len(order) - 1))] for q in (0.05, 0.35, 0.65, 0.95)]
    cols = ["imagen + GT", "sin prompt (token 0)", "tokens aprendidos k=4", "caja oráculo"]
    fig, axes = plt.subplots(len(sel), len(cols), figsize=(2.3 * len(cols), 2.0 * len(sel)))
    from samcost.metrics import dice
    for r, iid in enumerate(sel):
        emb, osz, isz = cs.emb(iid); img, gt = load_image(iid), load_mask(iid)
        with torch.no_grad():
            lr0, _ = cs.decode_all4(emb, torch.zeros(1, 0, 256))
            m0 = (cs.full(lr0[:, :1], osz, isz)[0, 0] > 0).numpy()
            ml = (cs.full(torch.from_numpy(z["lowres"][lids.index(iid)].astype(np.float32))[None, None], osz, isz)[0, 0] > 0).numpy()
            mb = np.zeros_like(gt)
            for reg in instances(gt):
                lrb, _ = cs.decode(emb, osz, box=bbox(reg), multimask=False)
                mb |= (cs.full(lrb, osz, isz)[0, 0] > 0).numpy()
        axes[r, 0].imshow(img); contour(axes[r, 0], gt)
        for c, m in enumerate([m0, ml, mb]):
            ax = axes[r, c + 1]; ax.imshow(img); ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.55); contour(ax, gt)
            ax.set_title(f"Dice {dice(m, gt):.2f}", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    for c, t in enumerate(cols):
        axes[0, c].set_title(t + ("" if c == 0 else "\n" + axes[0, c].get_title()), fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "cualitativo_tokens.png", dpi=160); plt.close(fig)


def fig_calib():
    ca = RESULTS / "calib_amg16_test.npz"
    if not ca.exists():
        return
    z = np.load(ca)
    p, t = z["pred"].ravel(), z["true"].ravel()
    rng = np.random.default_rng(0); i = rng.choice(len(p), min(6000, len(p)), replace=False)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.scatter(p[i], t[i], s=3, color="#a3a29d", alpha=0.4, linewidths=0)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.axvline(0.88, color=BLUE, lw=1); ax.text(0.885, 0.95, "umbral AMG 0,88", color=BLUE, fontsize=7)
    ax.set_xlabel("IoU predicho por la cabeza IoU"); ax.set_ylabel("IoU real con la GT del pólipo")
    ax.set_xlim(0.3, 1.05); ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIGURES / "calibracion_amg.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="erf,pca,att,noprompt,qual,calib")
    a = ap.parse_args(); FIGURES.mkdir(exist_ok=True)
    only = set(a.only.split(","))
    if "erf" in only: fig_erf()
    if "pca" in only or "att" in only: fig_pca_att()
    if "noprompt" in only: fig_noprompt()
    if "qual" in only: fig_qual()
    if "calib" in only: fig_calib()
    print("figuras:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
