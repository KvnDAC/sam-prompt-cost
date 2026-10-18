"""Plot completed technical pilots and verify the selected U-Net checkpoint."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from samcost.data import load_image, load_mask
from samcost.unet import UNet, prepare_arrays, restore_logits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot_root", type=Path, default=ROOT / "runs/model_comparison")
    ap.add_argument("--output_dir", type=Path, required=True)
    a = ap.parse_args()
    tags = ("unet_pilot_20261016", "unet_pilot20_20261016", "unet_groupnorm_pilot_20261016")
    summaries = [json.loads((a.pilot_root / tag / "summary.json").read_text(encoding="utf-8")) for tag in tags]
    labels = ("Sin normalización · 3 épocas", "Sin normalización · 20 épocas", "GroupNorm · 20 épocas")
    manifests = [json.loads((a.pilot_root / tag / "manifest.json").read_text(encoding="utf-8")) for tag in tags]
    if any(m["ids"] != manifests[0]["ids"] for m in manifests[1:]):
        raise ValueError("Pilot IDs differ; do not plot as the same sample")
    torch.set_num_threads(1)
    run = a.pilot_root / tags[-1]
    saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model = UNet(saved["base_channels"], saved.get("normalization", "none"))
    model.load_state_dict(saved["model"])
    model.eval()
    spec = importlib.util.spec_from_file_location("unet_trainer", ROOT / "scripts/18_unet_baseline.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    ids = manifests[-1]["ids"]["dev"]
    rows = trainer.evaluate(model, ids, saved["image_size"], "cpu")
    if abs(np.mean([r["dice"] for r in rows]) - summaries[-1]["best_dev_dice"]) > 1e-8:
        raise ValueError("Checkpoint reproduction mismatch")
    with (run / "dev_per_image.csv").open(encoding="utf-8") as f:
        reference = {r["id"]: float(r["dice"]) for r in csv.DictReader(f)}
    if set(reference) != set(ids) or any(abs(r["dice"] - reference[r["id"]]) > 1e-8 for r in rows):
        raise ValueError("Per-image predictions mismatch")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for label, summary in zip(labels, summaries):
        h = summary["history"]
        axes[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label=label)
        axes[1].plot([0] + [r["epoch"] for r in h], [summary["initial_dev_dice"]] + [r["dev_dice"] for r in h], label=label,
                     marker="." if len(h) <= 3 else None, zorder=3 if len(h) <= 3 else 2)
    axes[0].set(xlabel="Época", ylabel="BCE+Dice train")
    axes[1].set(xlabel="Época", ylabel="Dice dev", ylim=(-0.03, 1.03))
    axes[1].legend(fontsize=7)
    fig.suptitle("Pilotos técnicos U-Net · mismos 32 train / 16 dev · una semilla")
    fig.tight_layout()
    fig.savefig(a.output_dir / "pilot_curves.png", dpi=160)
    plt.close(fig)
    chosen = ids[:4]  # fixed order, not selected for high Dice
    fig, axes = plt.subplots(len(chosen), 2, figsize=(7, 3 * len(chosen)), squeeze=False)
    with torch.inference_mode():
        for pair, iid in zip(axes, chosen):
            image, gt = load_image(iid), load_mask(iid)
            x, _, _, input_size = prepare_arrays(image, gt, saved["image_size"])
            logits = model(x[None])[0]
            pred = (restore_logits(logits, input_size, gt.shape) > 0).numpy()
            for ax in pair:
                ax.imshow(image)
                ax.contour(gt, levels=[0.5], colors=["cyan"], linewidths=0.9)
                ax.axis("off")
            pair[0].set_title("Imagen y GT", fontsize=9)
            if pred.any() and not pred.all():
                pair[1].contour(pred, levels=[0.5], colors=["magenta"], linewidths=0.9)
            pair[1].set_title(f"U-Net GN · Dice {reference[iid]:.3f}", fontsize=9)
    fig.suptitle("Primeras cuatro imágenes dev del piloto · GT cian / predicción magenta")
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, hspace=0.2)
    fig.savefig(a.output_dir / "pilot_predictions.png", dpi=140)
    plt.close(fig)
    print("Selected checkpoint and every per-image Dice reproduced; figures saved", flush=True)


if __name__ == "__main__":
    main()
