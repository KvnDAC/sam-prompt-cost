"""¿Está el pólipo en el embedding? Probes lineales y auto-prompts (el «dónde» sin humano).

(a) probe lineal sobre el embedding final (256-d por token, 64×64) → máscara (umbral elegido en dev)
(b) auto-prompt: caja / punto derivados de la máscara del probe → SAM congelado (como self-prompting [12])
(c) probe por capa sobre las salidas de los 12 bloques (promediadas a 16×16): AUROC / AP en test

Salida: results/probe_test_per_image.csv, results/probe_layers.json, results/probe_summary.json
"""
import csv, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from samcost.common import CACHE, RESULTS, load_sam, save_json, set_seed, set_threads
from samcost.data import load_mask, read_split
from samcost.decode import CachedSAM
from samcost.metrics import dice, iou
from samcost.prompts import bbox, interior_point


def token_labels(gt, isz, n=64):
    h, w = isz
    hh, ww = int(round(h * n / 1024)), int(round(w * n / 1024))
    lab = F.interpolate(torch.from_numpy(gt[None, None].astype(np.float32)), size=(hh, ww), mode="area")[0, 0].numpy()
    return lab, hh, ww


def collect(cs, ids, per_img=None):
    X, Y = [], []
    rng = np.random.default_rng(0)
    for iid in ids:
        e = np.load(CACHE / "emb" / f"{iid}.npy").astype(np.float32)  # 256×64×64
        lab, hh, ww = token_labels(load_mask(iid), tuple(cs.meta[iid]["input_size"]))
        f = e[:, :hh, :ww].reshape(256, -1).T
        y = lab.ravel()
        if per_img and len(y) > per_img:
            sel = rng.choice(len(y), per_img, replace=False); f, y = f[sel], y[sel]
        X.append(f); Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


def train_probe(X, Y, epochs=4, seed=0):
    set_seed(seed)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y).float()
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    lin = torch.nn.Linear(256, 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 8192):
            i = perm[s:s + 8192]
            loss = F.binary_cross_entropy_with_logits(lin((Xt[i] - mu) / sd)[:, 0], Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
    return lin, mu, sd


@torch.no_grad()
def probe_logits(cs, lin, mu, sd, iid):
    emb, osz, isz = cs.emb(iid)
    z = lin(((emb[0].permute(1, 2, 0) - mu) / sd))[..., 0]  # 64×64
    lr = F.interpolate(z[None, None], size=(256, 256), mode="bilinear", align_corners=False)
    return lr, emb, osz, isz


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "layers", "all"], default="all")
    a = ap.parse_args()
    set_threads()
    sam = load_sam()
    cs = CachedSAM(sam)
    tr, dev, te = read_split("train400"), read_split("dev"), read_split("test")
    t0 = time.time()
    if a.part == "layers":
        out = json.load(open(RESULTS / "probe_summary.json"))
        layer_probe(cs, tr, te, out)
        return
    X, Y = collect(cs, tr, per_img=1500)
    out = {"n_tokens_train": int(len(Y))}
    lin, mu, sd = train_probe(X, Y)
    # umbral en dev
    thr_grid = np.linspace(-3, 3, 25)
    dscore = np.zeros(len(thr_grid))
    for iid in dev:
        lr, emb, osz, isz = probe_logits(cs, lin, mu, sd, iid)
        full = cs.full(lr, osz, isz)[0, 0].numpy()
        gt = load_mask(iid)
        dscore += [dice(full > t, gt) for t in thr_grid]
    thr = float(thr_grid[np.argmax(dscore)])
    out["probe_threshold_dev"] = thr
    rows = []
    for iid in te:
        lr, emb, osz, isz = probe_logits(cs, lin, mu, sd, iid)
        full = cs.full(lr, osz, isz)[0, 0].numpy()
        gt = load_mask(iid)
        pm = full > thr
        r = {"id": iid, "probe_dice": dice(pm, gt), "probe_iou": iou(pm, gt)}
        # auto-prompts a partir de la máscara del probe (componentes ≥ 1 % del área de la mayor)
        lab, n = ndi.label(pm)
        if n:
            sizes = ndi.sum(pm, lab, range(1, n + 1))
            comps = [lab == (j + 1) for j in range(n) if sizes[j] >= 0.01 * sizes.max() and sizes[j] >= 64]
            big = lab == (int(np.argmax(sizes)) + 1)
            bm, pt_m, bpm = np.zeros_like(gt), np.zeros_like(gt), np.zeros_like(gt)
            for c in comps:
                lrb, _ = cs.decode(emb, osz, box=bbox(c), multimask=False)
                bm |= (cs.full(lrb, osz, isz)[0, 0] > 0).numpy()
            p, _ = interior_point(big)
            lrp, ipp = cs.decode(emb, osz, points=p[None], labels=[1], multimask=True)
            pt_m = (cs.full(lrp, osz, isz)[0, int(torch.argmax(ipp[0]))] > 0).numpy()
            lrbp, _ = cs.decode(emb, osz, points=p[None], labels=[1], box=bbox(big), multimask=False)
            bpm = (cs.full(lrbp, osz, isz)[0, 0] > 0).numpy()
            r.update(selfbox_dice=dice(bm, gt), selfpoint_dice=dice(pt_m, gt), selfboxpoint_dice=dice(bpm, gt),
                     selfbox_iou=iou(bm, gt), probe_n_comp=len(comps))
        else:
            r.update(selfbox_dice=0.0, selfpoint_dice=0.0, selfboxpoint_dice=0.0, selfbox_iou=0.0, probe_n_comp=0)
        rows.append(r)
    with open(RESULTS / "probe_test_per_image.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    for k in ["probe_dice", "selfbox_dice", "selfpoint_dice", "selfboxpoint_dice"]:
        out[k] = float(np.mean([r[k] for r in rows]))
    print(json.dumps(out), f"{(time.time() - t0) / 60:.1f} min", flush=True)
    save_json(out, RESULTS / "probe_summary.json")
    if a.part == "all":
        layer_probe(cs, tr, te, out)


def layer_probe(cs, tr, te, out):
    """(c) probe por capa sobre las salidas de bloque promediadas a 16×16; se lee un bloque a la vez con mmap."""
    def labels(ids):
        L = []
        for iid in ids:
            lab, hh, ww = token_labels(load_mask(iid), tuple(cs.meta[iid]["input_size"]), n=16)
            L.append((lab.ravel(), hh, ww))
        return L
    Ltr, Lte = labels(tr), labels(te)
    def feats(ids, L, l):
        return np.concatenate([np.load(CACHE / "blocks16" / f"{iid}.npy", mmap_mode="r")[l, :, :hh, :ww].astype(np.float32).reshape(768, -1).T
                               for iid, (_, hh, ww) in zip(ids, L)])
    ytr = (np.concatenate([y for y, _, _ in Ltr]) >= 0.5).astype(int)
    yte = (np.concatenate([y for y, _, _ in Lte]) >= 0.5).astype(int)
    layers = {}
    for l in range(12):
        Xtr, Xte = feats(tr, Ltr, l), feats(te, Lte, l)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(Xtr), ytr)
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        layers[l] = {"auroc": float(roc_auc_score(yte, p)), "ap": float(average_precision_score(yte, p))}
        print("block", l, layers[l], flush=True)
        del Xtr, Xte
    out["layers"] = layers
    out["layer_cells"] = {"train": int(len(ytr)), "test": int(len(yte)), "test_pos_rate": float(yte.mean())}
    save_json(out, RESULTS / "probe_summary.json")


if __name__ == "__main__":
    main()
