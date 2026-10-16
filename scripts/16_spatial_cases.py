"""Reproduce and visualize dev cases selected by GT-box minus GT-point Dice."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from samcost.common import load_sam, save_json, set_threads
from samcost.data import instances, load_image, load_mask, read_split
from samcost.decode import CachedSAM
from samcost.metrics import dice
from samcost.prompts import bbox, interior_point


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--n_cases", type=int, default=3)
    a = ap.parse_args()
    if not 1 <= a.n_cases <= 10:
        ap.error("--n_cases must be between 1 and 10")
    source = a.experiment / "perturbations/dev/per_image_direction.csv"
    with source.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = read_split("dev")
    scores = {}
    for condition in ("noprompt_0", "point_0", "box_1"):
        selected = [r for r in rows if r["condition"] == condition]
        if len(selected) != len(ids) or {r["id"] for r in selected} != set(ids):
            raise ValueError(f"Invalid image coverage: {condition}")
        for r in selected:
            scores[r["id"], condition] = float(r["dice"])
    chosen = sorted(ids, key=lambda i: scores[i, "box_1"] - scores[i, "point_0"], reverse=True)[:a.n_cases]
    out = a.experiment / "spatial_cases"
    if out.exists():
        raise FileExistsError(out)
    set_threads()
    cs = CachedSAM(load_sam())
    cs.require_ids(chosen)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13, 3.7 * len(chosen)), squeeze=False)
    case_rows, masks = [], {}
    with torch.no_grad():
        for index, iid in enumerate(chosen):
            emb, osz, isz = cs.emb(iid)
            gt, image = load_mask(iid), load_image(iid)
            regs = instances(gt)
            prompts = [(interior_point(reg)[0], bbox(reg)) for reg in regs]
            predictions = {}
            for condition in ("noprompt_0", "point_0", "box_1"):
                pred = np.zeros_like(gt)
                if condition == "noprompt_0":
                    low, _ = cs.decode(emb, osz, multimask=False)
                    pred = (cs.full(low, osz, isz)[0, 0] > 0).numpy()
                else:
                    for point, box in prompts:
                        kwargs = {"points": point[None], "labels": [1]} if condition == "point_0" else {"box": box}
                        low, _ = cs.decode(emb, osz, multimask=False, **kwargs)
                        pred |= (cs.full(low, osz, isz)[0, 0] > 0).numpy()
                metric = dice(pred, gt)
                if abs(metric - scores[iid, condition]) > 1e-8:
                    raise ValueError(f"Reproduction mismatch: {iid}/{condition}")
                predictions[condition] = pred
                masks[f"{iid}_{condition}"] = pred
                case_rows.append({"id": iid, "condition": condition, "dice": metric,
                                  "pred_area": float(pred.mean()), "gt_area": float(gt.mean())})
            for col, (condition, label) in enumerate(((None, "Imagen y GT"), ("noprompt_0", "Sin prompt"),
                                                     ("point_0", "Punto GT"), ("box_1", "Caja GT"))):
                ax = axes[index, col]
                ax.imshow(image)
                ax.contour(gt, levels=[0.5], colors=["cyan"], linewidths=0.8)
                if condition:
                    pred = predictions[condition]
                    if pred.any() and not pred.all():
                        ax.contour(pred, levels=[0.5], colors=["magenta"], linewidths=0.8)
                    label += f" · Dice {scores[iid, condition]:.3f}"
                if condition == "point_0":
                    for point, _ in prompts:
                        ax.plot(*point, marker="*", color="yellow", markersize=8)
                if condition == "box_1":
                    for _, box in prompts:
                        ax.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1],
                                               fill=False, edgecolor="yellow", linestyle="--", linewidth=0.8))
                ax.set_title(label, fontsize=9)
                ax.axis("off")
            print(f"case {index + 1}/{len(chosen)} reproduced", flush=True)
    out.mkdir()
    save_json({"split": "dev", "ids": chosen, "selection": "largest GT-box minus GT-point Dice; post-hoc extremes",
               "warning": "Illustrations are not representative or causal evidence.", "reproduced_tolerance": 1e-8,
               "input_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "cache_meta": {i: cs.meta[i] for i in chosen}}, out / "manifest.json")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(case_rows[0]))
        w.writeheader()
        w.writerows(case_rows)
    np.savez_compressed(out / "predictions.npz", **masks)
    fig.suptitle("Casos extremos de dev · GT cian, predicción magenta, prompt amarillo", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94, hspace=0.2)
    fig.savefig(out / "predictions.png", dpi=150)
    plt.close(fig)
    print(out / "predictions.png")


if __name__ == "__main__":
    main()
