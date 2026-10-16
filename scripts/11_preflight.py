"""Read-only check of local experiment inputs; does not download or run SAM."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="train400,dev,test")
    a = ap.parse_args()
    checkpoint = Path(os.environ.get("SAM_CKPT", ROOT / "checkpoints/sam_vit_b_01ec64.pth"))
    report = {"packages": {m: importlib.util.find_spec(m) is not None for m in
                           ("torch", "torchvision", "numpy", "scipy", "sklearn", "segment_anything")},
              "checkpoint_exists": checkpoint.is_file(), "splits": {}}
    if checkpoint.is_file():
        md5 = hashlib.md5()
        with checkpoint.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)
        report["checkpoint_md5_valid"] = md5.hexdigest() == "01ec64d29a2fca3f0661936605ae66f8"
    mp = ROOT / "cache/meta.json"
    meta = json.loads(mp.read_text()) if mp.is_file() else {}
    for split in a.splits.split(","):
        ids = (ROOT / "data/splits" / f"{split}.txt").read_text().split()
        report["splits"][split] = {"n": len(ids),
            "missing_images": sum(not (ROOT / "data/Kvasir-SEG/images" / f"{i}.jpg").is_file() for i in ids),
            "missing_masks": sum(not (ROOT / "data/Kvasir-SEG/masks" / f"{i}.jpg").is_file() for i in ids),
            "missing_cache": sum(i not in meta or not (ROOT / "cache/emb" / f"{i}.npy").is_file() for i in ids)}
    print(json.dumps(report, indent=2))
    ready = (all(report["packages"].values()) and report.get("checkpoint_md5_valid", False)
             and all(not any(v[k] for k in ("missing_images", "missing_masks", "missing_cache"))
                     for v in report["splits"].values()))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
