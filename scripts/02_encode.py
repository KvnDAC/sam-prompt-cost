"""Calcula una sola vez los embeddings del image encoder (congelado) y los guarda en caché.

Guarda por imagen:
  cache/emb/<id>.npy        embedding final del neck, 256×64×64, float16
  cache/blocks16/<id>.npy   salida de cada bloque ViT promediada a 16×16: 12×768×16×16, float16 (probes por capa)
  cache/meta.json           tamaño original, tamaño de entrada y precisión usada

Uso: python scripts/02_encode.py --split test --precision fp32
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import torch.nn.functional as F

from samcost.common import CACHE, CKPT, load_sam, set_threads
from samcost.data import KVASIR, SamPreprocessor, load_image, read_split


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_matches(meta, signature, emb_path, blocks_path, need_blocks):
    return (all(meta.get(k) == v for k, v in signature.items())
            and emb_path.is_file() and (not need_blocks or blocks_path.is_file()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--no_blocks", action="store_true")
    ap.add_argument("--storage", choices=["fp16", "fp32"], default="fp16",
                    help="fp16 reproduce E1; fp32 conserva precisión para la nueva fase")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="subconjunto explícito para comprobación técnica")
    ap.add_argument("--sample_seed", type=int, help="muestreo fijo del subconjunto, como --train_limit")
    a = ap.parse_args()
    set_threads()
    dev = os.environ.get("SAMCOST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    sam = load_sam(device=dev)
    pp = SamPreprocessor(sam)
    (CACHE / "emb").mkdir(parents=True, exist_ok=True)
    (CACHE / "blocks16").mkdir(parents=True, exist_ok=True)
    meta_p = CACHE / "meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {}
    model_hash = sha256_file(CKPT)
    encoder_hash = sha256_file(Path(__import__(sam.image_encoder.__class__.__module__, fromlist=["x"]).__file__))
    feats = {}
    hooks = []
    if not a.no_blocks:
        for i, b in enumerate(sam.image_encoder.blocks):
            hooks.append(b.register_forward_hook(
                lambda m, inp, out, i=i: feats.__setitem__(i, F.avg_pool2d(out.permute(0, 3, 1, 2).float(), 4)[0])))
    ids = read_split(a.split)
    if a.limit:
        ids = (sorted(np.random.default_rng(a.sample_seed).choice(ids, min(a.limit, len(ids)), replace=False).tolist())
               if a.sample_seed is not None else ids[:a.limit])
    t0 = time.time()
    for n, iid in enumerate(ids):
        signature = {"precision": a.precision, "storage": a.storage,
                     "checkpoint_sha256": model_hash, "encoder_source_sha256": encoder_hash,
                     "image_sha256": sha256_file(KVASIR / "images" / f"{iid}.jpg"),
                     "torch_version": str(torch.__version__), "device": dev}
        if not a.force and cache_matches(meta.get(iid, {}), signature,
                CACHE / "emb" / f"{iid}.npy", CACHE / "blocks16" / f"{iid}.npy", not a.no_blocks):
            continue
        im = load_image(iid)
        x, osz, isz = pp(im)
        x = x.to(dev)
        with torch.no_grad():
            if a.precision == "bf16":
                with torch.autocast(dev, dtype=torch.bfloat16):
                    e = sam.image_encoder(x)
            else:
                e = sam.image_encoder(x)
        dtype = np.float32 if a.storage == "fp32" else np.float16
        np.save(CACHE / "emb" / f"{iid}.npy", e[0].float().cpu().numpy().astype(dtype))
        if feats:
            np.save(CACHE / "blocks16" / f"{iid}.npy", torch.stack([feats[i] for i in range(12)]).cpu().numpy().astype(np.float16))
        meta[iid] = {"original_size": list(osz), "input_size": list(isz), "split": a.split, **signature}
        if n % 10 == 0 or n == len(ids) - 1:
            json.dump(meta, open(meta_p, "w"))
            el = time.time() - t0
            print(f"[{a.split}] {n + 1}/{len(ids)}  {el / 60:.1f} min", flush=True)
    json.dump(meta, open(meta_p, "w"))


if __name__ == "__main__":
    main()
