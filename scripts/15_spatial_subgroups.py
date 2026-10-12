"""Exploratory GT-defined subgroups of the completed dev spatial ablation."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from samcost.common import save_json
from samcost.data import instances, load_image, load_mask, read_split
from samcost.metrics import bootstrap_ci


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    a = ap.parse_args()
    root = a.experiment
    ids = read_split("dev")
    with (root / "perturbations/dev/per_image_direction.csv").open(encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    conditions = ["noprompt_0", "fullbox_0", "point_0", "point_2", "box_1", "box_2"]
    values = {}
    for condition in conditions:
        selected = [r for r in raw if r["condition"] == condition]
        expected = 4 if condition == "point_2" else 1
        if len(selected) != len(ids) * expected or {r["id"] for r in selected} != set(ids):
            raise ValueError(f"Incomplete condition: {condition}")
        for iid in ids:
            image_rows = [r for r in selected if r["id"] == iid]
            if len(image_rows) != expected:
                raise ValueError(f"Incorrect direction count: {condition}/{iid}")
            values[iid, condition] = float(np.mean([float(r["dice"]) for r in image_rows]))
    attributes = []
    for iid in ids:
        gt = load_mask(iid)
        area = float(gt.mean())
        attributes.append({"id": iid, "area_fraction": area,
                           "area_bin": "<5%" if area < 0.05 else "5-15%" if area < 0.15 else ">=15%",
                           "multiple_components": int(len(instances(gt)) > 1),
                           "touches_edge": int(gt[0].any() or gt[-1].any() or gt[:, 0].any() or gt[:, -1].any())})
    out = root / "spatial_subgroups"
    out.mkdir(exist_ok=True)
    with (out / "attributes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(attributes[0]))
        w.writeheader()
        w.writerows(attributes)
    groups = [("all", attributes)]
    for key in ("area_bin", "multiple_components", "touches_edge"):
        for value in sorted({r[key] for r in attributes}, key=str):
            groups.append((f"{key}={value}", [r for r in attributes if r[key] == value]))
    summaries = []
    for name, group in groups:
        for condition in conditions:
            vector = [values[r["id"], condition] for r in group]
            mean, low, high = bootstrap_ci(vector)
            summaries.append({"group": name, "condition": condition, "n": len(group),
                              "dice": mean, "ci95": [low, high]})
    save_json({"split": "dev", "exploratory": True, "groups": summaries,
               "definitions": {"area_bins": [0.05, 0.15],
                               "components": "same GT connected-component filtering used by oracle prompts",
                               "edge": "any positive GT pixel on the image boundary"},
               "warning": "Post-hoc descriptive groups; small groups and correlated attributes limit interpretation."},
              out / "summary.json")
    lines = ["# Subgrupos de la ablación espacial (dev)", "",
             "Análisis descriptivo posterior a observar la ablación. No establece causas ni generalización.", "",
             "| Grupo | n | Sin prompt | Punto GT | Punto desplazado 2 radios | Caja GT | Caja ×2 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, group in groups:
        cells = []
        for condition in ("noprompt_0", "point_0", "point_2", "box_1", "box_2"):
            row = next(r for r in summaries if r["group"] == name and r["condition"] == condition)
            cells.append(f"{row['dice']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]")
        lines.append(f"| {name} | {len(group)} | " + " | ".join(cells) + " |")
    lines += ["", "Las direcciones se promedian por imagen antes del bootstrap. Los grupos se solapan; los IC no corrigen selección posterior ni multiplicidad.",
              "En imágenes con varias componentes, el oráculo recibe un prompt por componente y une las máscaras: también recibe más información.",
              "El panel muestra seis casos con mayor diferencia caja GT − punto GT; es una selección ilustrativa, no una muestra representativa."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = sorted(ids, key=lambda i: values[i, "box_1"] - values[i, "point_0"], reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    for ax, iid in zip(axes.flat, chosen):
        ax.imshow(load_image(iid))
        ax.contour(load_mask(iid), levels=[0.5], colors=["cyan"], linewidths=1)
        ax.set_title(f"Punto {values[iid, 'point_0']:.2f} · caja {values[iid, 'box_1']:.2f}\n{iid}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Contorno GT: casos con mayor brecha entre caja y punto en dev")
    fig.tight_layout()
    fig.subplots_adjust(top=0.91, hspace=0.28)
    fig.savefig(out / "cases.png", dpi=140)
    plt.close(fig)
    print(out / "report.md")


if __name__ == "__main__":
    main()
