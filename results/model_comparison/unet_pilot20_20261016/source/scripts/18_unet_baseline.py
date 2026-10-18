"""Train an independent U-Net baseline on RGB images; development only, no test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from samcost.common import save_json, set_seed, set_threads
from samcost.data import KVASIR, load_image, load_mask, read_split
from samcost.learned import seg_loss
from samcost.metrics import dice, iou
from samcost.unet import UNet, prepare_arrays, restore_logits


class Images(Dataset):
    def __init__(self, ids, size, augment):
        self.ids, self.size, self.augment = ids, size, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        iid = self.ids[index]
        image, mask = load_image(iid), load_mask(iid)
        if self.augment:
            if torch.rand(()) < 0.5:
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            if torch.rand(()) < 0.5:
                image, mask = image[::-1].copy(), mask[::-1].copy()
        x, y, valid, _ = prepare_arrays(image, mask, self.size)
        return x, y, valid


@torch.inference_mode()
def evaluate(model, ids, size, device):
    model.eval()
    rows = []
    for iid in ids:
        mask = load_mask(iid)
        image, _, _, input_size = prepare_arrays(load_image(iid), mask, size)
        # Mask is used for metrics and frame geometry only, never given to the network.
        logits = model(image[None].to(device))[0].cpu()
        pred = (restore_logits(logits, input_size, mask.shape) > 0).numpy()
        rows.append({"id": iid, "dice": dice(pred, mask), "iou": iou(pred, mask),
                     "pred_area": float(pred.mean())})
    return rows


def write_rows(rows, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset(ids, limit):
    return sorted(np.random.default_rng(1234).choice(ids, limit, replace=False).tolist()) if 0 < limit < len(ids) else ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--dev_limit", type=int, default=0)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = ap.parse_args()
    if min(a.epochs, a.base_channels, a.batch_size, a.threads) < 1 or a.image_size < 16 or a.image_size % 16 or a.lr <= 0 or min(a.train_limit, a.dev_limit) < 0:
        ap.error("Positive settings required; image_size must be a multiple of 16; limits >=0")
    if a.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA requested but unavailable")
    out = a.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    tr = subset(read_split("train400"), a.train_limit)
    dev = read_split("dev")[:a.dev_limit] if a.dev_limit else read_split("dev")
    if not tr or not dev or set(tr) & set(dev) or len(set(tr)) != len(tr) or len(set(dev)) != len(dev):
        raise ValueError("Nonempty disjoint unique train/dev IDs required")
    for iid in tr + dev:
        if not all((KVASIR / kind / f"{iid}.jpg").is_file() for kind in ("images", "masks")):
            raise FileNotFoundError(iid)
    set_threads(a.threads)
    set_seed(a.seed)
    model = UNet(a.base_channels).to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    out.mkdir(parents=True)
    params = sum(p.numel() for p in model.parameters())
    save_json({"model": "UNet_padded", "pretraining": "none", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
               "ids": {"train": tr, "dev": dev, "test": []}, "test_evaluated": False,
               "pilot": len(tr) != 400 or len(dev) != 100, "n_parameters": params,
               "input": "RGB [0,1], longest side resize, top-left zero padding; nearest GT; valid-only BCE+Dice",
               "augmentation": "independent horizontal and vertical flips, probability 0.5 each, train only",
               "selection": "maximum mean dev Dice including initialization; fixed logit threshold 0",
               "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": a.device, "threads": a.threads},
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), ROOT / "src/samcost/unet.py", ROOT / "src/samcost/data.py", ROOT / "src/samcost/learned.py")},
               "data_sha256": {iid: {kind: hashlib.sha256((KVASIR / kind / f"{iid}.jpg").read_bytes()).hexdigest() for kind in ("images", "masks")} for iid in tr + dev}}, out / "manifest.json")
    rows = evaluate(model, dev, a.image_size, a.device)
    write_rows(rows, out / "initial_dev_per_image.csv")
    initial = best = float(np.mean([r["dice"] for r in rows]))
    best_epoch = 0
    checkpoint = out / "model.pt"

    def save_checkpoint():
        torch.save({"model": model.state_dict(), "base_channels": a.base_channels,
                    "image_size": a.image_size, "best_epoch": best_epoch, "best_dev_dice": best}, checkpoint)

    save_checkpoint()
    loader = DataLoader(Images(tr, a.image_size, True), batch_size=a.batch_size, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(a.seed))
    history, started = [], time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        total, count = 0., 0
        for image, target, valid in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(image.to(a.device))
            loss = seg_loss(logits, target.to(a.device), valid.to(a.device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(image)
            count += len(image)
        rows = evaluate(model, dev, a.image_size, a.device)
        score = float(np.mean([r["dice"] for r in rows]))
        if score > best:
            best, best_epoch = score, epoch
            save_checkpoint()
        history.append({"epoch": epoch, "train_loss": total / count, "dev_dice": score,
                        "elapsed_seconds": time.perf_counter() - started})
        save_json({"history": history, "best_epoch": best_epoch, "best_dev_dice": best}, out / "progress.json")
        print(json.dumps(history[-1]), flush=True)
    saved = torch.load(checkpoint, map_location=a.device, weights_only=True)
    model.load_state_dict(saved["model"])
    selected = evaluate(model, dev, a.image_size, a.device)
    if abs(np.mean([r["dice"] for r in selected]) - best) > 1e-8:
        raise ValueError("Reloaded checkpoint does not reproduce chosen dev score")
    write_rows(selected, out / "dev_per_image.csv")
    save_json({"model": "UNet_padded", "base_channels": a.base_channels, "image_size": a.image_size,
               "n_parameters": params, "seed": a.seed, "n_train": len(tr), "n_dev": len(dev),
               "pilot": len(tr) != 400 or len(dev) != 100, "epochs": a.epochs,
               "initial_dev_dice": initial, "best_dev_dice": best, "best_epoch": best_epoch,
               "test_evaluated": False, "checkpoint_reproduced": True, "history": history,
               "note": "Development only. A limited run is a technical pilot, not a comparative benchmark. No optimizer state saved for exact resume."}, out / "summary.json")
    print(out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
