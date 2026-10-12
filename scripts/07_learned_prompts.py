"""Fase B (piloto): tokens de prompt aprendidos con SAM congelado.

Entrena k tokens (k×256 parámetros) con BCE + Dice en el marco 256×256 de baja resolución, elige la época por
Dice en dev (resolución original) y reporta en test. Guarda tokens, curva y métricas por imagen.

Uso: python scripts/07_learned_prompts.py --k 4 --seed 0 --epochs 8
"""
import argparse, csv, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from samcost.common import CACHE, RESULTS, load_sam, save_json, set_seed, set_threads
from samcost.data import gt_lowres, load_mask, read_split
from samcost.decode import CachedSAM
from samcost.learned import LearnedPrompt, seg_loss, valid_mask
from samcost.metrics import dice, iou


def load_set(cs, ids):
    E = torch.empty(len(ids), 256, 64, 64, dtype=torch.float16)
    T = np.zeros((len(ids), 256, 256), np.float32)
    V = np.zeros((len(ids), 256, 256), np.float32)
    for n, iid in enumerate(ids):
        E[n] = torch.from_numpy(np.load(CACHE / "emb" / f"{iid}.npy"))
        isz = tuple(cs.meta[iid]["input_size"])
        T[n] = gt_lowres(load_mask(iid), isz)
        V[n] = valid_mask(isz)
    return E, torch.from_numpy(T), torch.from_numpy(V)


@torch.no_grad()
def evaluate(model, cs, ids, keep_masks=False):
    rows, masks = [], {}
    for iid in ids:
        emb, osz, isz = cs.emb(iid)
        lr, ip = model(emb)
        pred = (cs.full(lr[:, None], osz, isz)[0, 0] > 0).numpy()
        gt = load_mask(iid)
        rows.append({"id": iid, "dice": dice(pred, gt), "iou": iou(pred, gt), "pred_iou": float(ip[0]),
                     "area": float(pred.mean())})
        if keep_masks:
            masks[iid] = lr[0].half().numpy()
    return rows, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--learn_dense", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--train_split", default="train400")
    ap.add_argument("--init", choices=["random", "meanpoint"], default="random",
                    help="random: N(0, σ) (D12, pre-registrado); meanpoint: media del embedding de un clic en el centro de los pólipos de train + ruido")
    ap.add_argument("--eval_limit", type=int, default=0, help="solo para pruebas rápidas")
    a = ap.parse_args()
    set_threads()
    set_seed(a.seed)
    sam = load_sam()
    cs = CachedSAM(sam)
    tr = [i for i in read_split(a.train_split) if i in cs.meta]
    if a.train_limit and len(tr) > a.train_limit:
        rng = np.random.default_rng(1234)  # subconjunto fijo, independiente de la semilla del modelo
        tr = sorted(rng.choice(tr, a.train_limit, replace=False).tolist())
    dev, te = [i for i in read_split("dev") if i in cs.meta], read_split("test")
    if a.eval_limit:
        dev, te = dev[:a.eval_limit], te[:a.eval_limit]
    E, T, V = load_set(cs, tr)
    init_tokens = None
    if a.init == "meanpoint":
        from samcost.data import instances
        from samcost.prompts import interior_point
        embs = []
        with torch.no_grad():
            for iid in tr[:200]:
                gt = load_mask(iid); inst = instances(gt)
                if not inst:
                    continue
                p, _ = interior_point(max(inst, key=lambda m: m.sum()))
                osz = tuple(cs.meta[iid]["original_size"])
                c = torch.as_tensor(cs.tf.apply_coords(p[None], osz), dtype=torch.float)[None]
                sp, _ = sam.prompt_encoder(points=(c, torch.ones(1, 1, dtype=torch.int)), boxes=None, masks=None)
                embs.append(sp[0, 0])
        mean_pt = torch.stack(embs).mean(0)
        init_tokens = mean_pt[None].repeat(a.k, 1) + 0.05 * torch.randn(a.k, 256)
    model = LearnedPrompt(sam, k=a.k, learn_dense=a.learn_dense, init_tokens=init_tokens)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * int(np.ceil(len(tr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    tag = a.tag or f"lp_k{a.k}{'_dense' if a.learn_dense else ''}{'' if a.init == 'random' else '_' + a.init}_s{a.seed}"
    out_dir = RESULTS / "learned" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(a.seed)
    hist, best = [], -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in ("tokens", "dense")}
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g)
        losses = []
        for s in range(0, len(tr), a.bs):
            idx = perm[s:s + a.bs]
            logits, _ = model(E[idx].float())
            loss = seg_loss(logits, T[idx], V[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            losses.append(float(loss.detach()))
        model.eval()
        drows, _ = evaluate(model, cs, dev)
        dd = float(np.mean([r["dice"] for r in drows])) if drows else float("nan")
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "dev_dice": dd, "min": (time.time() - t0) / 60})
        print(json.dumps(hist[-1]), flush=True)
        if dd > best or (np.isnan(dd) and ep == a.epochs - 1):
            best, best_state = dd, {k: v.detach().clone() for k, v in model.state_dict().items() if k in ("tokens", "dense")}
    model.load_state_dict(best_state, strict=False)
    trows, tmasks = evaluate(model, cs, te, keep_masks=True)
    with open(out_dir / "test_per_image.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(trows[0].keys())); wr.writeheader(); wr.writerows(trows)
    np.save(out_dir / "tokens.npy", model.tokens.detach().numpy())
    np.savez_compressed(out_dir / "test_lowres.npz", ids=np.array(list(tmasks)), lowres=np.stack(list(tmasks.values())))
    summary = {"tag": tag, "k": a.k, "seed": a.seed, "init": a.init, "learn_dense": a.learn_dense, "n_params": n_params,
               "n_train": len(tr), "epochs": a.epochs, "lr": a.lr, "bs": a.bs, "best_dev_dice": best,
               "best_epoch": int(np.argmax([h["dev_dice"] for h in hist]) + 1),
               "test_dice": float(np.mean([r["dice"] for r in trows])), "test_iou": float(np.mean([r["iou"] for r in trows])),
               "history": hist, "minutes": (time.time() - t0) / 60}
    save_json(summary, out_dir / "summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}))


if __name__ == "__main__":
    main()
