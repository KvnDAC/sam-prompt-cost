"""AMG ("segment everything") sobre embeddings en caché, replicando la lógica de SamAutomaticMaskGenerator
(una sola capa de recorte, sin eliminación de regiones pequeñas): rejilla de puntos → 3 máscaras por punto →
filtro por IoU predicho (0.88) → filtro por estabilidad (0.95, offset 1.0) → NMS de cajas (0.7).
Además registra el "embudo": la mejor propuesta posible (oráculo) en cada etapa."""
import numpy as np
import torch
from segment_anything.utils.amg import batched_mask_to_box, build_point_grid, calculate_stability_score
from torchvision.ops.boxes import batched_nms

from .metrics import dice, iou


@torch.no_grad()
def amg_funnel(cs, emb, osz, isz, gt, points_per_side=16, batch=4, pred_iou_thresh=0.88,
               stability_thresh=0.95, stability_offset=1.0, nms_thresh=0.7):
    sam = cs.sam
    h, w = osz
    pts = build_point_grid(points_per_side) * np.array([w, h])[None]
    gt_t = torch.from_numpy(gt)
    rec = {"iou": [], "dice": [], "pred_iou": [], "stab": [], "area": [], "lowres": [], "boxes": [], "pt": []}
    for s in range(0, len(pts), batch):
        p = pts[s:s + batch]
        c = torch.as_tensor(cs.tf.apply_coords(p, osz), dtype=torch.float)[:, None, :]
        l = torch.ones(len(p), 1, dtype=torch.int)
        sp, de = sam.prompt_encoder(points=(c, l), boxes=None, masks=None)
        lr, ip = sam.mask_decoder(image_embeddings=emb, image_pe=cs.pe, sparse_prompt_embeddings=sp,
                                  dense_prompt_embeddings=de, multimask_output=True)
        logits = cs.full(lr, osz, isz).flatten(0, 1)  # (b*3, H, W)
        ip = ip.flatten()
        st = calculate_stability_score(logits, 0.0, stability_offset)
        m = logits > 0
        inter = (m & gt_t).flatten(1).sum(1).float()
        uni = (m | gt_t).flatten(1).sum(1).float()
        msum = m.flatten(1).sum(1).float()
        rec["iou"].append(torch.where(uni > 0, inter / uni.clamp(min=1), torch.ones_like(uni)))
        rec["dice"].append(torch.where(msum + gt_t.sum() > 0, 2 * inter / (msum + gt_t.sum()).clamp(min=1), torch.ones_like(msum)))
        rec["pred_iou"].append(ip); rec["stab"].append(st); rec["area"].append(msum / (h * w))
        rec["boxes"].append(batched_mask_to_box(m).float())
        rec["lowres"].append(lr.flatten(0, 1).half())
        rec["pt"].append(torch.as_tensor(np.repeat(p, 3, axis=0)))
    R = {k: torch.cat(v) for k, v in rec.items()}
    out = {"n_raw": int(len(R["iou"]))}
    j = int(torch.argmax(R["iou"]))
    out.update(best_raw_iou=float(R["iou"][j]), best_raw_dice=float(R["dice"][j]),
               best_raw_pred_iou=float(R["pred_iou"][j]), best_raw_stab=float(R["stab"][j]),
               best_raw_area=float(R["area"][j]))
    k1 = R["pred_iou"] > pred_iou_thresh
    k2 = k1 & (R["stab"] >= stability_thresh)
    out["n_iou_filter"], out["n_stab_filter"] = int(k1.sum()), int(k2.sum())
    out["best_after_iou_filter_dice"] = float(R["dice"][k1].max()) if k1.any() else 0.0
    out["best_after_filters_dice"] = float(R["dice"][k2].max()) if k2.any() else 0.0
    idx = torch.nonzero(k2).flatten()
    if len(idx):
        keep = idx[batched_nms(R["boxes"][idx], R["pred_iou"][idx], torch.zeros(len(idx)), nms_thresh)]
    else:
        keep = idx
    out["n_final"] = int(len(keep))
    final_masks = None
    if len(keep):
        d, pi, st, ar = R["dice"][keep], R["pred_iou"][keep], R["stab"][keep], R["area"][keep]
        out["best_final_dice"] = float(d.max())
        out["top1_prediou_dice"] = float(d[torch.argmax(pi)])
        out["top1_stability_dice"] = float(d[torch.argmax(st)])
        out["largest_dice"] = float(d[torch.argmax(ar)])
        b = R["boxes"][keep]
        ctr = torch.stack([(b[:, 0] + b[:, 2]) / 2 / w - 0.5, (b[:, 1] + b[:, 3]) / 2 / h - 0.5], 1).norm(dim=1)
        out["central_dice"] = float(d[torch.argmin(ctr)])
        out["top1_prediou_value"] = float(pi.max())
        out["final_mean_area"] = float(ar.mean())
        final_masks = cs.full(R["lowres"][keep].float()[:, None], osz, isz)[:, 0] > 0
    else:
        for k in ["best_final_dice", "top1_prediou_dice", "top1_stability_dice", "largest_dice", "central_dice"]:
            out[k] = 0.0
    # calibración: IoU predicho vs IoU real de todas las propuestas crudas
    out["_pred_iou_all"] = R["pred_iou"].numpy().astype(np.float32)
    out["_true_iou_all"] = R["iou"].numpy().astype(np.float32)
    best_raw_mask = cs.full(R["lowres"][j:j + 1].float()[:, None], osz, isz)[0, 0] > 0
    return out, final_masks, best_raw_mask
