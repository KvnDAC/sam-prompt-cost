"""Línea base: ¿cuánto cuesta el prompt? SAM ViT-B zero-shot sobre Kvasir-SEG.

Modos (todos sobre los embeddings en caché, SAM congelado):
  noprompt  decoder con 0 tokens de prompt (solo los tokens de salida); se evalúan las 4 salidas
  point     1 clic oráculo por pólipo (punto más interior) — salida multimáscara, elige la cabeza IoU / oráculo
  box       1 caja oráculo por pólipo (caja ajustada de la GT) — salida única
  clicks    hasta 20 clics simulados (protocolo estándar) con la máscara previa como prompt denso
  amg       "segment everything" con rejilla 16×16 y el embudo de selección (más lento)

Uso: python scripts/03_prompt_cost.py --split test --modes noprompt,point,box,clicks
     python scripts/03_prompt_cost.py --split test --modes amg
"""
import argparse, csv, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from samcost.common import CACHE, RESULTS, load_sam, set_threads
from samcost.data import instances, load_mask, read_split
from samcost.decode import CachedSAM
from samcost.metrics import dice, iou
from samcost.prompts import bbox, interior_point, next_click


def border_frac(m):
    """Fracción de la máscara que cae en el marco exterior (5 % del lado): mide si se pega al viñeteado."""
    h, w = m.shape
    b = np.zeros_like(m); bh, bw = max(1, h // 20), max(1, w // 20)
    b[:bh] = b[-bh:] = True; b[:, :bw] = b[:, -bw:] = True
    return float(np.logical_and(m, b).sum() / max(1, m.sum()))


def run_basic(cs, iid, modes, rows, clicks_iou, calib, store):
    emb, osz, isz = cs.emb(iid)
    gt = load_mask(iid)
    inst = instances(gt)
    r = {"id": iid, "n_inst": len(inst), "gt_area": float(gt.mean())}
    if "noprompt" in modes:
        lr, ip = cs.decode_all4(emb, torch.zeros(1, 0, 256))
        full = cs.full(lr, osz, isz)[0] > 0
        ds = [dice(full[k].numpy(), gt) for k in range(4)]
        ious = [iou(full[k].numpy(), gt) for k in range(4)]
        ip = ip[0].numpy()
        sel = 1 + int(np.argmax(ip[1:]))
        r.update({f"np_dice_t{k}": ds[k] for k in range(4)})
        r.update({f"np_pred_iou_t{k}": float(ip[k]) for k in range(4)})
        r.update({f"np_area_t{k}": float(full[k].numpy().mean()) for k in range(4)})
        r.update({f"np_border_t{k}": border_frac(full[k].numpy()) for k in range(4)})
        r["np_dice_single"] = ds[0]                # modo salida única (token 0)
        r["np_dice_multi_ioupick"] = ds[sel]       # modo multimáscara, elige la cabeza IoU
        r["np_dice_oracle4"] = max(ds)
        r["np_maxlogit_t0"] = float(lr[0, 0].max())
        calib["noprompt"].append(np.stack([ip, np.array(ious)], 1))
        store["np_lowres"].append(lr[0].half().numpy())
    if "point" in modes:
        pm_sel, pm_orc, pm_single = np.zeros_like(gt), np.zeros_like(gt), np.zeros_like(gt)
        for reg in inst:
            p, _ = interior_point(reg)
            lr, ip = cs.decode(emb, osz, points=p[None], labels=[1], multimask=True)
            full = (cs.full(lr, osz, isz)[0] > 0).numpy()
            ip = ip[0].numpy()
            pm_sel |= full[int(np.argmax(ip))]
            d_reg = [dice(full[k], reg) for k in range(3)]
            pm_orc |= full[int(np.argmax(d_reg))]
            calib["point"].append(np.stack([ip, np.array([iou(full[k], reg) for k in range(3)])], 1))
            lr1, _ = cs.decode(emb, osz, points=p[None], labels=[1], multimask=False)
            pm_single |= (cs.full(lr1, osz, isz)[0, 0] > 0).numpy()
        r["pt_dice_ioupick"], r["pt_dice_oracle3"], r["pt_dice_single"] = dice(pm_sel, gt), dice(pm_orc, gt), dice(pm_single, gt)
        r["pt_iou_ioupick"] = iou(pm_sel, gt)
    if "box" in modes:
        bm = np.zeros_like(gt)
        for reg in inst:
            lr, ip = cs.decode(emb, osz, box=bbox(reg), multimask=False)
            full = (cs.full(lr, osz, isz)[0, 0] > 0).numpy()
            bm |= full
            calib["box"].append(np.array([[float(ip[0, 0]), iou(full, reg)]]))
        r["box_dice"], r["box_iou"] = dice(bm, gt), iou(bm, gt)
    if "clicks" in modes:
        p, _ = interior_point(inst[0] if inst else gt)
        P, L = [p], [1]
        lr, ip = cs.decode(emb, osz, points=np.array(P), labels=L, multimask=True)
        k = int(torch.argmax(ip[0]))
        prev = lr[:, k:k + 1]
        pred = (cs.full(prev, osz, isz)[0, 0] > 0).numpy()
        seq = [iou(pred, gt)]
        for _ in range(19):
            nc = next_click(pred, gt)
            if nc is None:
                seq.append(seq[-1]); continue
            P.append(nc[0]); L.append(nc[1])
            lr, ip = cs.decode(emb, osz, points=np.array(P), labels=L, mask_input=prev, multimask=False)
            prev = lr
            pred = (cs.full(prev, osz, isz)[0, 0] > 0).numpy()
            seq.append(iou(pred, gt))
        seq = np.array(seq)
        clicks_iou.append(seq)
        for thr in (0.80, 0.85, 0.90):
            hit = np.nonzero(seq >= thr)[0]
            r[f"noc{int(thr * 100)}"] = int(hit[0] + 1) if len(hit) else 20
            r[f"fail{int(thr * 100)}"] = int(len(hit) == 0)
        r["click_iou_1"], r["click_iou_3"], r["click_iou_5"] = seq[0], seq[2], seq[4]
    rows.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--modes", default="noprompt,point,box,clicks")
    ap.add_argument("--pps", type=int, default=16, help="puntos por lado en AMG")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    set_threads()
    modes = set(a.modes.split(","))
    cs = CachedSAM(load_sam())
    ids = read_split(a.split)
    if a.limit:
        ids = ids[:a.limit]
    cs.require_ids(ids)
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    if "amg" in modes:
        from samcost.amg import amg_funnel
        rows, calib_p, calib_t, keep_masks = [], [], [], {}
        for n, iid in enumerate(ids):
            emb, osz, isz = cs.emb(iid)
            gt = load_mask(iid)
            out, fm, best = amg_funnel(cs, emb, osz, isz, gt, points_per_side=a.pps)
            calib_p.append(out.pop("_pred_iou_all")); calib_t.append(out.pop("_true_iou_all"))
            out["id"] = iid
            rows.append(out)
            if n % 10 == 0:
                print(f"[amg] {n + 1}/{len(ids)} {(time.time() - t0) / 60:.1f} min", flush=True)
        tag = f"amg{a.pps}_{a.split}"
        np.savez_compressed(RESULTS / f"calib_{tag}.npz", pred=np.stack(calib_p), true=np.stack(calib_t))
        with open(RESULTS / f"{tag}_per_image.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["id"] + [k for k in rows[0] if k != "id"]); wr.writeheader(); wr.writerows(rows)
        print(f"AMG listo en {(time.time() - t0) / 60:.1f} min")
        return
    rows, clicks_iou = [], []
    calib = {"noprompt": [], "point": [], "box": []}
    store = {"np_lowres": []}
    for n, iid in enumerate(ids):
        run_basic(cs, iid, modes, rows, clicks_iou, calib, store)
        if n % 10 == 0:
            print(f"[{a.split}] {n + 1}/{len(ids)} {(time.time() - t0) / 60:.1f} min", flush=True)
    tag = f"baseline_{a.split}"
    with open(RESULTS / f"{tag}_per_image.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    if clicks_iou:
        np.save(RESULTS / f"clicks_iou_{a.split}.npy", np.stack(clicks_iou))
    np.savez_compressed(RESULTS / f"calib_{tag}.npz", **{k: np.concatenate(v) for k, v in calib.items() if v})
    if store["np_lowres"]:
        (CACHE / "outputs").mkdir(exist_ok=True)
        np.savez_compressed(CACHE / "outputs" / f"noprompt_lowres_{a.split}.npz", ids=np.array(ids), lowres=np.stack(store["np_lowres"]))
    print(f"Listo en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
