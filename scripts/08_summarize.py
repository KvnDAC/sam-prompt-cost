"""Agrega resultados: tabla principal con IC bootstrap, pruebas pareadas, embudo AMG, calibración, NoC,
probes, tokens aprendidos y figuras para el repositorio. Robusto a resultados faltantes.

Salida: results/summary_main.json, results/main_table.csv, figures/*.png
"""
import csv, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from scipy import stats

from samcost.common import FIGURES, RESULTS, save_json
from samcost.metrics import bootstrap_ci, paired_test

INK, INK2, GRID, BLUE, GREY = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#a3a29d"


def rcsv(p):
    p = RESULTS / p
    return list(csv.DictReader(open(p))) if p.exists() else []


def col(rows, k, ids=None):
    d = {r["id"]: float(r[k]) for r in rows if r.get(k) not in (None, "")}
    if ids is None:
        return d
    return np.array([d[i] for i in ids])


def main():
    S = {}
    base = rcsv("baseline_test_per_image.csv")
    amg = rcsv("amg16_test_per_image.csv")
    probe = rcsv("probe_test_per_image.csv")
    ids = [r["id"] for r in base]
    S["n_test"] = len(ids)
    methods = {}  # nombre -> (vector dice, prompts humanos, parámetros entrenados)

    def add(name, vec, human, trained, note=""):
        m, lo, hi = bootstrap_ci(vec)
        methods[name] = {"dice": vec, "human": human, "trained": trained, "note": note, "mean": m, "ci": [lo, hi]}

    if base:
        add("Sin prompt · salida única (token 0)", col(base, "np_dice_single", ids), "0", "0")
        add("Sin prompt · multimáscara, elige cabeza IoU", col(base, "np_dice_multi_ioupick", ids), "0", "0")
        add("Sin prompt · mejor de 4 salidas (oráculo)", col(base, "np_dice_oracle4", ids), "0 (oráculo)", "0")
        add("1 clic por pólipo · elige cabeza IoU", col(base, "pt_dice_ioupick", ids), "1 clic/pólipo", "0")
        add("1 clic por pólipo · salida única (token 0)", col(base, "pt_dice_single", ids), "1 clic/pólipo", "0")
        add("1 clic por pólipo · mejor de 3 (oráculo)", col(base, "pt_dice_oracle3", ids), "1 clic/pólipo", "0")
        add("Caja por pólipo", col(base, "box_dice", ids), "1 caja/pólipo", "0")
        # clics: Dice no disponible; se reporta IoU
        S["noprompt_tokens"] = {f"t{k}": {"dice": float(np.mean(col(base, f'np_dice_t{k}', ids))),
                                          "area": float(np.mean(col(base, f'np_area_t{k}', ids))),
                                          "border_frac": float(np.mean(col(base, f'np_border_t{k}', ids))),
                                          "pred_iou": float(np.mean(col(base, f'np_pred_iou_t{k}', ids))),
                                          "empty_rate": float(np.mean(col(base, f'np_area_t{k}', ids) < 1e-4))}
                                for k in range(4)}
        S["gt_area_median"] = float(np.median(col(base, "gt_area", ids)))
        # NoC
        if "noc85" in base[0]:
            S["clicks"] = {f"noc{t}": float(np.mean(col(base, f"noc{t}", ids))) for t in (80, 85, 90)}
            S["clicks"].update({f"fail{t}": float(np.mean(col(base, f"fail{t}", ids))) for t in (80, 85, 90)})
            ci = np.load(RESULTS / "clicks_iou_test.npy")
            S["clicks"]["mean_iou_by_click"] = ci.mean(0).tolist()
            S["clicks"]["box_iou_mean"] = float(np.mean(col(base, "box_iou", ids)))
    if amg:
        a_ids = [r["id"] for r in amg]
        assert a_ids == ids or set(a_ids) == set(ids)
        for k, name in [("best_raw_dice", "AMG · mejor propuesta cruda (oráculo)"),
                        ("best_after_filters_dice", "AMG · mejor tras filtros de calidad (oráculo)"),
                        ("best_final_dice", "AMG · mejor tras NMS (oráculo)"),
                        ("top1_prediou_dice", "AMG · elige mayor IoU predicho"),
                        ("largest_dice", "AMG · elige la mayor"), ("central_dice", "AMG · elige la más central")]:
            add(name, col(amg, k, ids), "0" if "oráculo" not in name else "0 (oráculo)", "0")
        S["amg"] = {k: float(np.mean(col(amg, k, ids))) for k in ["n_raw", "n_iou_filter", "n_stab_filter", "n_final",
                                                                   "best_raw_pred_iou", "best_raw_stab", "best_raw_area"]}
        S["amg"]["frac_images_no_mask"] = float(np.mean(col(amg, "n_final", ids) == 0))
        br = col(amg, "best_raw_dice", ids); bf = col(amg, "best_after_filters_dice", ids)
        bn = col(amg, "best_final_dice", ids); t1 = col(amg, "top1_prediou_dice", ids)
        S["amg"]["frac_best_raw_ge_0.8"] = float(np.mean(br >= 0.8))
        S["amg"]["frac_lost_in_filters(>0.1)"] = float(np.mean(br - bf > 0.1))
        S["amg"]["frac_lost_in_selection(>0.1)"] = float(np.mean(bn - t1 > 0.1))
        S["amg"]["best_raw_pred_iou_lt_0.88"] = float(np.mean(col(amg, "best_raw_pred_iou", ids) <= 0.88))
        S["amg"]["best_raw_stab_lt_0.95"] = float(np.mean(col(amg, "best_raw_stab", ids) < 0.95))
    if probe:
        add("Probe lineal sobre el embedding", col(probe, "probe_dice", ids), "0", "257")
        add("Auto-prompt: caja desde el probe → SAM", col(probe, "selfbox_dice", ids), "0", "257")
        add("Auto-prompt: punto desde el probe → SAM", col(probe, "selfpoint_dice", ids), "0", "257")
        add("Auto-prompt: caja + punto desde el probe → SAM", col(probe, "selfboxpoint_dice", ids), "0", "257")
    # tokens aprendidos (media por imagen sobre semillas + dispersión entre semillas)
    lp = {}
    for d in sorted(glob.glob(str(RESULTS / "learned" / "*"))):
        sj = Path(d) / "summary.json"
        if not sj.exists():
            continue
        s = json.load(open(sj))
        rows = list(csv.DictReader(open(Path(d) / "test_per_image.csv")))
        cfg = f"k{s['k']}{'_dense' if s['learn_dense'] else ''}_{s.get('init', 'random')}"
        lp.setdefault(cfg, []).append((s, col(rows, "dice", ids)))
    S["learned"] = {}
    for cfg, runs in lp.items():
        mat = np.stack([v for _, v in runs])
        per_seed = [float(v.mean()) for v in mat]
        init_lbl = "init. aleatoria, pre-registrada" if cfg.endswith("_random") else "init. clic medio"
        add(f"Tokens aprendidos k={runs[0][0]['k']} · {init_lbl} ({len(runs)} semilla{'s' if len(runs) > 1 else ''})", mat.mean(0), "0", str(runs[0][0]["n_params"]))
        S["learned"][cfg] = {"seeds": [s["seed"] for s, _ in runs], "test_dice_per_seed": per_seed,
                             "mean": float(np.mean(per_seed)), "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                             "best_epoch": [s["best_epoch"] for s, _ in runs], "dev_dice": [s["best_dev_dice"] for s, _ in runs],
                             "minutes": [s["minutes"] for s, _ in runs], "n_params": runs[0][0]["n_params"],
                             "history": [s["history"] for s, _ in runs]}
    # dos etapas (tokens k=4 → caja → SAM), media sobre semillas
    ts = sorted(glob.glob(str(RESULTS / "learned" / "lp_k4_meanpoint_s*" / "twostage_test_per_image.csv")))
    if ts:
        mat = np.stack([col(list(csv.DictReader(open(f))), "twostage_dice", ids) for f in ts])
        add(f"Dos etapas: tokens k=4 → caja → SAM ({len(ts)} semillas)", mat.mean(0), "0", "1024")
        S["twostage_per_seed"] = [float(v.mean()) for v in mat]
    # subgrupos
    attr = rcsv("image_attributes_test.csv")
    if attr:
        A = {r["id"]: r for r in attr}
        groups = {"multi_polipo": lambda a: int(a["n_inst"]) > 1, "area_lt_2pct": lambda a: float(a["gt_area"]) < 0.02,
                  "area_gt_15pct": lambda a: float(a["gt_area"]) > 0.15, "marcador_verde": lambda a: a["green_marker"] == "1",
                  "recuadro_negro": lambda a: a["black_box"] == "1"}
        sub = {}
        for g, f in groups.items():
            mask = np.array([f(A[i]) for i in ids])
            sub[g] = {"n": int(mask.sum())}
            for n_, m_ in methods.items():
                sub[g][n_] = float(m_["dice"][mask].mean()) if mask.any() else None
        S["subgroups"] = sub
    # tabla principal
    table = []
    for name, m in methods.items():
        table.append({"metodo": name, "dice_media": round(m["mean"], 4), "ic95_lo": round(m["ci"][0], 4),
                      "ic95_hi": round(m["ci"][1], 4), "prompts_humanos": m["human"], "parametros_entrenados": m["trained"]})
    if table:
        with open(RESULTS / "main_table.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(table[0].keys())); wr.writeheader(); wr.writerows(table)
    S["methods"] = {n: {"mean": m["mean"], "ci": m["ci"], "human": m["human"], "trained": m["trained"]} for n, m in methods.items()}
    # fracción de la brecha recuperada (respecto de token 0 sin prompt y de la caja)
    if base:
        d0 = methods["Sin prompt · salida única (token 0)"]["mean"]; db = methods["Caja por pólipo"]["mean"]
        S["gap_noprompt_to_box"] = db - d0
        S["gap_recovered"] = {n: (m["mean"] - d0) / (db - d0) for n, m in methods.items()}
    # pruebas pareadas clave
    tests = {}
    def pt(a, b):
        if a in methods and b in methods:
            tests[f"{a}  vs  {b}"] = paired_test(methods[a]["dice"], methods[b]["dice"])
    names = list(methods)
    lpn = [n for n in names if n.startswith("Tokens aprendidos")]
    for n in lpn:
        pt(n, "Probe lineal sobre el embedding")
        pt(n, "Auto-prompt: caja desde el probe → SAM")
        pt(n, "Caja por pólipo")
        pt(n, "Sin prompt · salida única (token 0)")
    if len(lpn) >= 2:
        pt(lpn[0], lpn[1])
    k4m = [n for n in lpn if "k=4" in n and "clic medio" in n]
    two = [n for n in names if n.startswith("Dos etapas")]
    if k4m and two:
        pt(two[0], k4m[0])
        pt(two[0], "Caja por pólipo")
        pt(two[0], "Auto-prompt: caja desde el probe → SAM")
    pt("Auto-prompt: caja + punto desde el probe → SAM", "Probe lineal sobre el embedding")
    pt("1 clic por pólipo · salida única (token 0)", "Probe lineal sobre el embedding")
    pt("Auto-prompt: caja desde el probe → SAM", "Probe lineal sobre el embedding")
    pt("AMG · mejor tras NMS (oráculo)", "AMG · elige mayor IoU predicho")
    pt("1 clic por pólipo · mejor de 3 (oráculo)", "1 clic por pólipo · elige cabeza IoU")
    pt("1 clic por pólipo · salida única (token 0)", "1 clic por pólipo · elige cabeza IoU")
    pt("Caja por pólipo", "1 clic por pólipo · elige cabeza IoU")
    S["paired_tests"] = tests
    # calibración de la cabeza IoU
    cal = {}
    cb = RESULTS / "calib_baseline_test.npz"
    if cb.exists():
        z = np.load(cb)
        for k in z.files:
            arr = z[k]
            if arr.ndim == 3:  # (n, salidas, 2)
                p, t = arr[..., 0].ravel(), arr[..., 1].ravel()
            else:
                p, t = arr[:, 0], arr[:, 1]
            cal[k] = {"spearman": float(stats.spearmanr(p, t).correlation), "mean_pred": float(p.mean()),
                      "mean_true": float(t.mean()), "overconfidence": float((p - t).mean()), "n": int(len(p))}
    ca = RESULTS / "calib_amg16_test.npz"
    if ca.exists():
        z = np.load(ca)
        p, t = z["pred"].ravel(), z["true"].ravel()
        per_img = [stats.spearmanr(a, b).correlation for a, b in zip(z["pred"], z["true"])]
        cal["amg_raw"] = {"spearman_pooled": float(stats.spearmanr(p, t).correlation),
                          "spearman_within_image_median": float(np.nanmedian(per_img)),
                          "mean_pred": float(p.mean()), "mean_true": float(t.mean()), "n": int(len(p))}
    S["calibration"] = cal
    for f in ["teardown.json", "erf.json", "features.json", "probe_summary.json", "attention_summary.json", "twostage_summary.json"]:
        if (RESULTS / f).exists():
            S[f.replace(".json", "")] = json.load(open(RESULTS / f))
    save_json(S, RESULTS / "summary_main.json")
    make_figures(S, methods, base, amg)
    for r in table:
        print(f"{r['metodo'][:60]:60s} {r['dice_media']:.3f} [{r['ic95_lo']:.3f}, {r['ic95_hi']:.3f}]")


def make_figures(S, methods, base, amg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    if "teardown" in S:
        blk = S["teardown"]["flops"]["blocks"]
        fig, ax = plt.subplots(figsize=(7, 3))
        x = [b["block"] for b in blk]
        mlp = [b["mlp_GFLOPs"] for b in blk]; qkv = [b["qkv_proj_GFLOPs"] for b in blk]; att = [b["attn_matmul_GFLOPs"] for b in blk]
        ax.bar(x, mlp, color=GREY, label="MLP", edgecolor="white", linewidth=1)
        ax.bar(x, qkv, bottom=mlp, color="#c9c8c3", label="qkv + proyección", edgecolor="white", linewidth=1)
        ax.bar(x, att, bottom=np.add(mlp, qkv), color=BLUE, label="matmul de atención", edgecolor="white", linewidth=1)
        ax.set_xticks(x); ax.set_xlabel("bloque del encoder (2, 5, 8, 11 = atención global)"); ax.set_ylabel("GFLOPs")
        ax.set_title("Los 4 bloques globales cuestan 1,7× un bloque con ventana", loc="left", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.22))
        fig.tight_layout(); fig.savefig(FIGURES / "teardown_flops_por_bloque.png", dpi=180); plt.close(fig)
    if "clicks" in S:
        ci = np.array(S["clicks"]["mean_iou_by_click"])
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.plot(np.arange(1, len(ci) + 1), ci, color=BLUE, lw=2, marker="o", ms=4)
        ax.axhline(S["clicks"]["box_iou_mean"], color=INK2, lw=1, ls="--")
        ax.text(len(ci), S["clicks"]["box_iou_mean"] + 0.01, "caja oráculo", ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_xlabel("número de clics simulados"); ax.set_ylabel("IoU medio (test)"); ax.set_ylim(0, 1)
        ax.set_title("IoU medio vs. clics", loc="left", color=INK, fontsize=11)
        fig.tight_layout(); fig.savefig(FIGURES / "iou_vs_clics.png", dpi=180); plt.close(fig)
    if methods:
        order = sorted(methods.items(), key=lambda kv: kv[1]["mean"])
        fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(order) + 1))
        for i, (n, m) in enumerate(order):
            c = BLUE if n.startswith("Tokens") else GREY
            ax.plot([m["ci"][0], m["ci"][1]], [i, i], color=c, lw=2)
            ax.plot(m["mean"], i, "o", color=c, ms=5)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([n for n, _ in order], fontsize=7)
        ax.set_xlim(0, 1); ax.set_xlabel("Dice medio en test (IC 95 % bootstrap)")
        fig.tight_layout(); fig.savefig(FIGURES / "comparacion_metodos.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
