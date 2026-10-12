"""Dev-only intervention: keep decoder image fixed, shuffle prompt generator context."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from samcost.common import load_sam, save_json, set_threads
from samcost.data import load_mask, read_split
from samcost.decode import CachedSAM
from samcost.learned import ConditionalPrompt
from samcost.metrics import bootstrap_ci, dice, paired_test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    a = ap.parse_args()
    root = a.experiment
    out = root / "conditioning_diagnostic"
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads((root / "plan.json").read_text())
    ids = read_split("dev")
    order = np.random.default_rng(0).permutation(len(ids))
    permutation = np.empty(len(ids), dtype=int)
    permutation[order] = np.roll(order, 1)
    assert np.all(permutation != np.arange(len(ids)))
    set_threads()
    sam = load_sam()
    cs = CachedSAM(sam)
    cs.require_ids(ids)
    for seed in plan["seeds"]:
        if not (root / "runs" / f"conditional_k4_s{seed}" / "summary.json").is_file():
            raise FileNotFoundError(f"Conditional seed {seed} not completed")
    out.mkdir()
    save_json({"split": "dev", "test_evaluated": False, "permutation_seed": 0,
               "context_by_target": {iid: ids[int(j)] for iid, j in zip(ids, permutation)},
               "intervention": "Only prompt generator context changes; decoder image and weights fixed."}, out / "manifest.json")
    matrix_normal, matrix_shuffled = [], []
    with torch.no_grad():
        for seed in plan["seeds"]:
            directory = root / "runs" / f"conditional_k4_s{seed}"
            summary = json.loads((directory / "summary.json").read_text())
            model = ConditionalPrompt(sam, k=4, init_tokens=torch.zeros(4, 256))
            model.load_state_dict(torch.load(directory / "prompt_state.pt", weights_only=True), strict=False)
            model.eval()
            rows = []
            for iid, j in zip(ids, permutation):
                emb, osz, isz = cs.emb(iid)
                context, _, _ = cs.emb(ids[int(j)])
                normal, _ = model(emb)
                sparse = model.prompt_embeddings(context)
                shuffled, _ = cs.decode_all4(emb, sparse)
                gt = load_mask(iid)
                pred_normal = (cs.full(normal[:, None], osz, isz)[0, 0] > 0).numpy()
                pred_shuffled = (cs.full(shuffled[:, :1], osz, isz)[0, 0] > 0).numpy()
                rows.append({"id": iid, "context_id": ids[int(j)], "seed": seed,
                             "normal_dice": dice(pred_normal, gt), "shuffled_dice": dice(pred_shuffled, gt)})
            normal_vec = np.array([r["normal_dice"] for r in rows])
            shuffled_vec = np.array([r["shuffled_dice"] for r in rows])
            if abs(normal_vec.mean() - summary["best_dev_dice"]) > 1e-8:
                raise ValueError("Reloaded checkpoint no longer reproduces selected dev score")
            matrix_normal.append(normal_vec)
            matrix_shuffled.append(shuffled_vec)
            with (out / f"seed{seed}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    normal_mean = np.stack(matrix_normal).mean(0)
    shuffled_mean = np.stack(matrix_shuffled).mean(0)
    save_json({"n_images": len(ids), "seeds": plan["seeds"], "normal": bootstrap_ci(normal_mean),
               "shuffled": bootstrap_ci(shuffled_mean),
               "normal_minus_shuffled": paired_test(normal_mean, shuffled_mean),
               "interpretation": "Exploratory sensitivity to context, not a held-out generalization result or a parameter-matched baseline."}, out / "summary.json")
    print(out / "summary.json")


if __name__ == "__main__":
    main()
