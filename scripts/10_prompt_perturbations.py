"""Spatial prompt ablation. Default: dev; fixed protocol, image-level uncertainty."""
import argparse
import csv
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from samcost.common import RESULTS, env_info, load_sam, save_json, set_threads
from samcost.data import instances, load_mask, read_split
from samcost.decode import CachedSAM
from samcost.metrics import bootstrap_ci, dice, iou
from samcost.prompts import bbox, interior_point


def shifted_point(point, depth, severity, direction, osz):
    h, w = osz
    return np.clip(point + depth * severity * np.asarray(direction), [0, 0], [w - 1, h - 1])


def expanded_box(box, scale, osz):
    h, w = osz
    center = (box[:2] + box[2:]) / 2
    half = (box[2:] - box[:2]) * scale / 2
    return np.concatenate([np.clip(center - half, [0, 0], [w - 1, h - 1]),
                           np.clip(center + half, [0, 0], [w - 1, h - 1])])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output_dir", type=Path, default=RESULTS / "phase_b" / "perturbations")
    a = ap.parse_args()
    set_threads()
    cs = CachedSAM(load_sam())
    ids = read_split(a.split)
    if a.limit:
        ids = ids[:a.limit]
    cs.require_ids(ids)
    out = a.output_dir / a.split
    if out.exists():
        raise FileExistsError(f"Salida existente: {out}. Use otro --output_dir.")
    out.mkdir(parents=True)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    conditions = [("noprompt", 0, (0, 0)), ("fullbox", 0, (0, 0)),
                  ("point", 0, (0, 0)), ("box", 1, (0, 0))]
    conditions += [("point", s, d) for s in (0.5, 1.0, 2.0, 4.0) for d in directions]
    conditions += [("box", s, (0, 0)) for s in (0.75, 1.25, 1.5, 2.0)]
    save_json({"split": a.split, "ids": ids, "env": env_info(),
               "point_offset_unit": "interior distance-transform radius",
               "box_severity": "scale around GT box center, clipped to image",
               "conditions": conditions, "multimask": False,
               "cache_meta": {i: cs.meta[i] for i in ids}}, out / "manifest.json")
    rows = []
    with torch.no_grad():
        for n, iid in enumerate(ids):
            emb, osz, isz = cs.emb(iid)
            gt = load_mask(iid)
            regs = instances(gt)
            if not regs:
                raise ValueError(f"GT sin componentes elegibles: {iid}")
            for mode, severity, direction in conditions:
                pred = np.zeros_like(gt)
                t0 = time.perf_counter()
                if mode in ("noprompt", "fullbox"):
                    box = np.array([0, 0, osz[1] - 1, osz[0] - 1]) if mode == "fullbox" else None
                    lr, _ = cs.decode(emb, osz, box=box, multimask=False)
                    pred = (cs.full(lr, osz, isz)[0, 0] > 0).numpy()
                else:
                    for reg in regs:
                        if mode == "point":
                            point, depth = interior_point(reg)
                            point = shifted_point(point, depth, severity, direction, osz)
                            lr, _ = cs.decode(emb, osz, points=point[None], labels=[1], multimask=False)
                        else:
                            lr, _ = cs.decode(emb, osz, box=expanded_box(bbox(reg), severity, osz), multimask=False)
                        pred |= (cs.full(lr, osz, isz)[0, 0] > 0).numpy()
                rows.append({"id": iid, "condition": f"{mode}_{severity:g}",
                             "dx": direction[0], "dy": direction[1], "dice": dice(pred, gt),
                             "iou": iou(pred, gt), "empty": int(not pred.any()),
                             "seconds": time.perf_counter() - t0})
            print(f"[perturbations] {n + 1}/{len(ids)}", flush=True)
    with open(out / "per_image_direction.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for condition in sorted({r["condition"] for r in rows}):
        image_values = []
        for iid in ids:
            image_values.append(np.mean([r["dice"] for r in rows if r["id"] == iid and r["condition"] == condition]))
        mean, lo, hi = bootstrap_ci(image_values)
        summary[condition] = {"dice": mean, "ci95": [lo, hi], "n_images": len(ids)}
    save_json(summary, out / "summary.json")


if __name__ == "__main__":
    main()
