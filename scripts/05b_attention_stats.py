"""¿Hacia dónde mira el token de máscara? Masa de atención token→imagen (capa final del decoder) dentro del pólipo.

Para cada imagen de test y cada modo (sin prompt, caja oráculo y, si existen, tokens aprendidos):
  mass_in_gt = Σ atención sobre tokens del pólipo (ponderada por la fracción de pólipo en cada token)
  lift       = mass_in_gt / área relativa del pólipo   (1 = no prefiere el pólipo)
  entropy    = entropía normalizada de la atención (1 = uniforme)
  top1_mass  = fracción de la atención que recibe el token más atendido
Salida: results/attention_stats.csv, results/attention_summary.json
"""
import csv, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import torch.nn.functional as F

from samcost.attention import capture_token_to_image
from samcost.common import RESULTS, load_sam, save_json, set_threads
from samcost.data import instances, load_mask, read_split
from samcost.decode import CachedSAM
from samcost.prompts import bbox


def stats_of(att, lab, valid):
    a = att.reshape(64, 64)
    a = a / a.sum()
    area = (lab * valid).sum() / valid.sum()
    valid_mass = float((a * valid).sum())
    m = float((a * lab).sum())
    p = a[valid > 0].ravel(); p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return {"mass_in_gt": m, "mass_in_valid": valid_mass,
            "mass_in_padding": 1.0 - valid_mass,
            "lift": m / max(valid_mass * area, 1e-12),
            "entropy": ent, "top1_mass": float(a.max())}


def main():
    set_threads()
    sam = load_sam(); cs = CachedSAM(sam)
    ids = read_split("test")
    tok_files = sorted(glob.glob(str(RESULTS / "learned" / "lp_k4_meanpoint_s*" / "tokens.npy")))
    learned = torch.from_numpy(np.load(tok_files[0])).float() if tok_files else None
    rows = []
    with torch.no_grad():
        for iid in ids:
            emb, osz, isz = cs.emb(iid)
            gt = load_mask(iid)
            h, w = isz
            hh, ww = int(round(h / 16)), int(round(w / 16))
            lab = np.zeros((64, 64), np.float32)
            lab[:hh, :ww] = F.interpolate(torch.from_numpy(gt[None, None].astype(np.float32)), size=(hh, ww), mode="area")[0, 0].numpy()
            valid = np.zeros((64, 64), np.float32); valid[:hh, :ww] = 1
            r = {"id": iid}
            with capture_token_to_image(sam) as st:
                cs.decode_all4(emb, torch.zeros(1, 0, 256))
            for k, v in stats_of(st["final"][0].mean(0)[1].numpy(), lab, valid).items():
                r[f"noprompt_{k}"] = v
            inst = instances(gt)
            if inst:
                big = max(inst, key=lambda m: m.sum())
                with capture_token_to_image(sam) as st:
                    cs.decode(emb, osz, box=bbox(big), multimask=False)
                for k, v in stats_of(st["final"][0].mean(0)[1].numpy(), lab, valid).items():
                    r[f"box_{k}"] = v
            if learned is not None:
                with capture_token_to_image(sam) as st:
                    cs.decode_all4(emb, learned)
                for k, v in stats_of(st["final"][0].mean(0)[1].numpy(), lab, valid).items():
                    r[f"learned_{k}"] = v
            rows.append(r)
    keys = sorted({k for r in rows for k in r})
    with open(RESULTS / "attention_stats.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["id"] + [k for k in keys if k != "id"]); wr.writeheader(); wr.writerows(rows)
    summ = {k: float(np.median([r[k] for r in rows if k in r])) for k in keys if k != "id"}
    summ["learned_tokens_file"] = tok_files[0] if tok_files else None
    save_json(summ, RESULTS / "attention_summary.json")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
