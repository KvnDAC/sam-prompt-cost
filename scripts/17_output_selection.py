"""Exploratory dev control: mask generation versus selection, preserve E1 outputs."""
import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from samcost.common import load_sam, save_json, set_threads
from samcost.data import read_split
from samcost.decode import CachedSAM
from samcost.metrics import bootstrap_ci, paired_test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    a = ap.parse_args()
    source = ROOT / "scripts/03_prompt_cost.py"
    spec = importlib.util.spec_from_file_location("baseline", source)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    ids = read_split("dev")
    input_csv = a.experiment / "perturbations/dev/per_image_direction.csv"
    with input_csv.open(encoding="utf-8") as f:
        originals = {(r["id"], r["condition"]): float(r["dice"]) for r in csv.DictReader(f)
                     if r["condition"] in ("noprompt_0", "point_0")}
    if len(originals) != 2 * len(ids):
        raise ValueError("Invalid spatial ablation coverage")
    out = a.experiment / "output_selection"
    if out.exists():
        raise FileExistsError(out)
    set_threads()
    cs = CachedSAM(load_sam())
    cs.require_ids(ids)
    rows, clicks = [], []
    calibration = {"noprompt": [], "point": [], "box": []}
    storage = {"np_lowres": []}
    with torch.no_grad():
        for n, iid in enumerate(ids):
            baseline.run_basic(cs, iid, {"noprompt", "point"}, rows, clicks, calibration, storage)
            storage["np_lowres"].clear()
            for key, condition in (("np_dice_single", "noprompt_0"), ("pt_dice_single", "point_0")):
                if abs(rows[-1][key] - originals[iid, condition]) > 1e-8:
                    raise ValueError(f"Reproduction mismatch: {iid}/{key}: {rows[-1][key]:.12g} vs {originals[iid, condition]:.12g}")
            if (n + 1) % 10 == 0:
                print(f"selection {n + 1}/{len(ids)}", flush=True)
    out.mkdir()
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    vectors = {key: np.array([r[key] for r in rows]) for key in
               ("np_dice_single", "np_dice_multi_ioupick", "np_dice_oracle4",
                "pt_dice_single", "pt_dice_ioupick", "pt_dice_oracle3")}
    summary = {"split": "dev", "n": len(ids), "test_evaluated": False, "exploratory": True,
               "metrics": {key: bootstrap_ci(value) for key, value in vectors.items()},
               "paired_tests": {f"{left} vs {right}": paired_test(vectors[left], vectors[right])
                                for left, right in (("np_dice_oracle4", "np_dice_single"),
                                                    ("pt_dice_oracle3", "pt_dice_single"),
                                                    ("pt_dice_ioupick", "pt_dice_single"),
                                                    ("pt_dice_oracle3", "pt_dice_ioupick"))},
               "oracle_warning": "Oracle3 chooses best Dice against each GT component before union; oracle4 uses entire GT. Neither is deployable.",
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (source, Path(__file__))},
               "cache_meta": {i: cs.meta[i] for i in ids},
               "reproduction_tolerance": 1e-8}
    save_json(summary, out / "summary.json")
    lines = ["# Generación y selección de salidas en dev", "",
             "Control exploratorio añadido después de inspeccionar la ablación espacial. SAM congelado, mismos 100 IDs y caché FP32. Sin test.", "",
             "Las 200 predicciones únicas sin prompt/con punto reproducen los scores de la ablación a tolerancia 1e−8.", "",
             "| Salida | Dice [IC 95 %] |", "|---|---:|"]
    for key, (mean, low, high) in summary["metrics"].items():
        lines.append(f"| {key} | {mean:.4f} [{low:.4f}, {high:.4f}] |")
    lines += ["", "El oráculo usa GT para elegir y mide disponibilidad de candidatos útiles; no es una mitigación automática.",
              "Con punto, se elige por componente y después se unen las máscaras. Sin prompt, las cuatro salidas se comparan contra la GT completa.",
              "Los intervalos remuestrean imágenes. Las pruebas pareadas son exploratorias y no corrigen multiplicidad.",
              "Una mejora del oráculo frente a la salida única muestra que esa salida puede ocultar candidatos útiles; no prueba por sí sola la causa del fallo ni garantiza que puedan seleccionarse sin GT."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "report.md")


if __name__ == "__main__":
    main()
