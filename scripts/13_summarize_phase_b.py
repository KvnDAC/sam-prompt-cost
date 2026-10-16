"""Summarize complete development runs; preserve E1 and label selection on dev."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from samcost.common import save_json
from samcost.data import read_split
from samcost.metrics import bootstrap_ci, paired_test


def read_values(path, key, ids):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = {r["id"]: float(r[key]) for r in rows}
    if len(rows) != len(ids) or set(values) != set(ids):
        raise ValueError(f"Incomplete or duplicated dev IDs in {path}")
    vector = np.array([values[i] for i in ids])
    if not np.isfinite(vector).all() or ((vector < 0) | (vector > 1)).any():
        raise ValueError(f"Invalid metric in {path}")
    return vector


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    a = ap.parse_args()
    root = a.experiment
    plan = json.loads((root / "plan.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    ids = read_split("dev")
    vectors, table = {}, []

    def add(name, matrix, trained):
        mat = np.atleast_2d(matrix)
        vec = mat.mean(0)
        mean, low, high = bootstrap_ci(vec)
        seed_means = mat.mean(1)
        vectors[name] = vec
        table.append({"method": name, "dice_mean": mean, "ci95_low": low, "ci95_high": high,
                      "n_images": len(ids), "n_seeds": len(mat), "seed_means": seed_means.tolist(),
                      "seed_std": float(seed_means.std(ddof=1)) if len(mat) > 1 else None,
                      "trained_parameters": trained})

    with (root / "perturbations/dev/per_image_direction.csv").open(encoding="utf-8") as f:
        perturb_rows = list(csv.DictReader(f))
    for condition, name in (("noprompt_0", "Sin prompt"), ("point_0", "Punto GT"), ("box_1", "Caja GT")):
        rows = [r for r in perturb_rows if r["condition"] == condition]
        values = {r["id"]: float(r["dice"]) for r in rows}
        if len(rows) != len(ids) or set(values) != set(ids):
            raise ValueError(f"Incomplete perturbation baseline {condition}")
        add(name, [values[i] for i in ids], 0)
    runs = {}
    for model, label in (("static", "Tokens estáticos"), ("conditional", "Tokens condicionados")):
        finals, initial, summaries = [], [], []
        for seed in plan["seeds"]:
            directory = root / "runs" / f"{model}_k4_s{seed}"
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["ids"]["test"] or not manifest["args"]["dev_only"]:
                raise ValueError(f"Run unexpectedly evaluated test: {directory}")
            summaries.append(json.loads((directory / "summary.json").read_text()))
            finals.append(read_values(directory / "dev_per_image.csv", "dice", ids))
            initial.append(read_values(directory / "initial_dev_per_image.csv", "dice", ids))
        add(label, finals, summaries[0]["n_params"])
        add(label + " · inicialización", initial, 0)
        runs[model] = summaries
    for key, name in (("probe_dice", "Probe lineal"), ("selfbox_dice", "Probe → caja → SAM"),
                      ("selfboxpoint_dice", "Probe → caja + punto → SAM")):
        mat = [read_values(root / "probes" / f"seed{s}" / "probe_dev_per_image.csv", key, ids) for s in plan["seeds"]]
        add(name, mat, 257)
    gap = vectors["Caja GT"].mean() - vectors["Sin prompt"].mean()
    for row in table:
        row["gap_recovered"] = float((row["dice_mean"] - vectors["Sin prompt"].mean()) / gap)
    tests = {}
    for left, right in (("Tokens condicionados", "Tokens estáticos"),
                        ("Tokens condicionados", "Tokens condicionados · inicialización"),
                        ("Tokens estáticos", "Tokens estáticos · inicialización"),
                        ("Probe → caja + punto → SAM", "Probe lineal"),
                        ("Tokens condicionados", "Probe → caja + punto → SAM")):
        tests[f"{left} vs {right}"] = paired_test(vectors[left], vectors[right])
    diagnostic_path = root / "conditioning_diagnostic/summary.json"
    if not diagnostic_path.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts/14_conditioning_diagnostic.py"),
                        "--experiment", str(root.resolve())], cwd=ROOT, check=True)
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    selection_path = root / "output_selection/summary.json"
    output_selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    summary = {"split": "dev", "n_images": len(ids), "test_evaluated": False,
               "selection_warning": "Epochs, LR and probe threshold selected on dev; scores are development results.",
               "plan": plan, "selection": selection, "table": table, "runs": runs, "paired_tests_exploratory": tests,
               "conditioning_diagnostic": diagnostic,
               "output_selection_metrics_exploratory": output_selection["metrics"] if output_selection else None}
    save_json(summary, root / "summary.json")
    with (root / "table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Fase B: resultados de desarrollo", "",
             "400 imágenes train, 100 dev; SAM ViT-B congelado, encoder y caché FP32. No se evaluó test.", "",
             "Estos resultados sirven para selección de configuración; no estiman generalización en un conjunto independiente.", "",
             f"LR condicionado seleccionado: {selection['chosen']['lr']:g}. Tres semillas en las comparaciones entrenadas.", "",
             "| Método | Dice dev [IC 95 % por imagen] | Desv. entre semillas | Brecha recuperada |",
             "|---|---:|---:|---:|"]
    for row in sorted(table, key=lambda r: r["dice_mean"], reverse=True):
        sd = "—" if row["seed_std"] is None else f"{row['seed_std']:.4f}"
        lines.append(f"| {row['method']} | {row['dice_mean']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {sd} | {100*row['gap_recovered']:.1f} % |")
    lines += ["", "Los IC remuestrean imágenes tras promediar semillas; la dispersión entre semillas se informa por separado.",
              "Punto/caja GT son referencias oráculo. Las inicializaciones no entrenadas sí usan localización GT de train para construir meanpoint.",
              "Las pruebas pareadas son exploratorias y no tienen ajuste de multiplicidad.", "",
              "## Épocas seleccionadas", ""]
    for model, summaries in runs.items():
        lines.append(f"- {model}: " + ", ".join(f"semilla {s['seed']}: época {s['best_epoch']}" for s in summaries))
    lines += ["", "Si se selecciona época 0, las actualizaciones no mejoraron la referencia inicial en dev.", "",
              "Los resultados E1 se conservan. El siguiente paso es congelar la configuración y evaluar los checkpoints en test."]
    difference = diagnostic["normal_minus_shuffled"]
    lines += ["", "## Control de contexto del generador", "",
              f"Prompts de la imagen correcta: Dice {diagnostic['normal'][0]:.4f}; contexto permutado: {diagnostic['shuffled'][0]:.4f}.",
              f"Diferencia pareada normal − permutado: {difference['mean_diff']:.4f}, IC 95 % [{difference['ci95'][0]:.4f}; {difference['ci95'][1]:.4f}].",
              "Decoder, imagen objetivo y pesos permanecen fijos. Los checkpoints recargados reprodujeron los scores elegidos en dev.",
              "Este control exploratorio mide sensibilidad al contexto; no sustituye un baseline de igual capacidad ni validación externa.",
              "Detalles en [conditioning_diagnostic/summary.json](conditioning_diagnostic/summary.json).", "",
              "## Caracterización espacial y selección", "",
              "La ablación espacial completa, los subgrupos exploratorios y los casos reproducidos están en [spatial_ablation.md](spatial_ablation.md)."]
    if output_selection:
        m = output_selection["metrics"]
        lines += [f"Sin prompt, el mejor de cuatro según GT alcanza Dice {m['np_dice_oracle4'][0]:.4f}; seleccionar entre esas cuatro salidas no recupera por sí solo la referencia de caja GT.",
                  f"Con punto, la salida única da {m['pt_dice_single'][0]:.4f}, la elección por IoU predicho {m['pt_dice_ioupick'][0]:.4f} y el mejor de tres según GT {m['pt_dice_oracle3'][0]:.4f}.",
                  "El oráculo de selección usa GT y no es una mitigación desplegable. Control posterior a inspeccionar la ablación, detallado en [output_selection/report.md](output_selection/report.md)."]
    lines += ["", "No se midió tiempo humano de anotación; las referencias GT caracterizan el valor y precisión de la información espacial."]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(root, table)
    print(root / "report.md")


def figures(root, table):
    perturbation_figure(root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(table, key=lambda r: r["dice_mean"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in enumerate(ordered):
        ax.plot([row["ci95_low"], row["ci95_high"]], [i, i], color="#2166ac")
        ax.plot(row["dice_mean"], i, "o", color="#2166ac")
    ax.set_yticks(range(len(ordered)), [r["method"] for r in ordered])
    ax.set(xlim=(0, 1), xlabel="Dice en dev · IC por imagen", title="Fase B: selección y comparación en dev")
    fig.tight_layout()
    fig.savefig(root / "comparison_dev.png", dpi=180)
    plt.close(fig)


def perturbation_figure(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    perturb = json.loads((root / "perturbations/dev/summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, mode, xlabel in ((axes[0], "point", "Desplazamiento / radio interior"),
                            (axes[1], "box", "Escala de caja respecto de GT")):
        keys = sorted([k for k in perturb if k.startswith(mode + "_")], key=lambda k: float(k.split("_")[1]))
        x = [float(k.split("_")[1]) for k in keys]
        y = [perturb[k]["dice"] for k in keys]
        low = [perturb[k]["ci95"][0] for k in keys]
        high = [perturb[k]["ci95"][1] for k in keys]
        ax.plot(x, y, "o-", color="#2166ac")
        ax.fill_between(x, low, high, color="#2166ac", alpha=0.2)
        ax.set(xlabel=xlabel, ylabel="Dice dev", ylim=(0, 1))
        for condition, label in (("noprompt_0", "Sin prompt"), ("fullbox_0", "Caja completa")):
            ax.axhline(perturb[condition]["dice"], linestyle="--", label=label, alpha=0.7)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "perturbations.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
