"""Genera los cuadernos del repositorio (Colab para reproducir + cuaderno de resultados)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

# 1) Reproducir en Colab (GPU)
nb = nbf.v4.new_notebook()
nb.cells = [
    md("# E1 · El costo del prompt en SAM — reproducir en Colab\n\n"
       "Ejecuta el pipeline completo del Entregable 1. Con GPU (T4) el paso más caro —calcular los embeddings del encoder— "
       "toma minutos en vez de horas; el resto corre sobre la caché.\n\n"
       "**Antes de empezar:** Entorno de ejecución → Cambiar tipo → GPU."),
    code("!git clone https://github.com/<usuario>/sam-prompt-cost.git\n%cd sam-prompt-cost\n# Colab ya trae torch/numpy/scipy/sklearn; solo falta el código oficial de SAM (commit fijado)\n!pip -q install git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"),
    md("## 1. Pesos oficiales de SAM ViT-B\nSe verifica el MD5 (`01ec64d2…`), cuyo prefijo da nombre al archivo oficial."),
    code("!mkdir -p checkpoints\n!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth\n!md5sum checkpoints/sam_vit_b_01ec64.pth"),
    md("## 2. Kvasir-SEG oficial y particiones\nTérminos: solo investigación y educación; citar Jha et al. (MMM 2020). "
       "Si el enlace cambia, descargar desde https://datasets.simula.no/kvasir-seg/."),
    code("!wget -q https://datasets.simula.no/downloads/kvasir-seg.zip -O kvasir-seg.zip\n!python scripts/00_prepare_kvasir.py --official_zip kvasir-seg.zip"),
    md("## 3. Fase A — teardown"),
    code("!python scripts/01_teardown.py\n!python scripts/04_receptive_field.py"),
    md("## 4. Embeddings (una pasada del encoder por imagen)"),
    code("!python scripts/02_encode.py --split test --precision fp32\n!python scripts/02_encode.py --split dev --precision fp32\n!python scripts/02_encode.py --split train400 --precision bf16"),
    md("## 5. Línea base: el costo del prompt"),
    code("!python scripts/03_prompt_cost.py --split test --modes noprompt,point,box,clicks\n!python scripts/03_prompt_cost.py --split test --modes amg --pps 16\n!python scripts/05_features.py\n!python scripts/05b_attention_stats.py"),
    md("## 6. Probe lineal, auto-prompts e intervención (tokens aprendidos)"),
    code("!python scripts/06_probe.py\nfor k in (4, 1):\n    for s in (0, 1, 2):\n        !python scripts/07_learned_prompts.py --k {k} --seed {s} --init meanpoint --epochs 6 --lr 5e-3 --bs 8"),
    md("## 7. Resumen y figuras"),
    code("!python scripts/07b_twostage.py\n!python scripts/08b_subgroups.py test\n!python scripts/05b_attention_stats.py\n!python scripts/08_summarize.py\n!python scripts/09_figures.py"),
]
nbf.write(nb, NB / "01_reproducir_E1_colab.ipynb")

# 2) Resultados (se ejecuta sobre results/ y figures/ ya calculados)
nb = nbf.v4.new_notebook()
nb.cells = [
    md("# E1 · Resultados\n\nCarga las métricas guardadas en `results/` y las figuras de `figures/`. No recalcula nada."),
    code("import json, pandas as pd\nfrom pathlib import Path\nfrom IPython.display import Image, display\nR = Path('../results'); F = Path('../figures')\nS = json.load(open(R / 'summary_main.json'))\npd.set_option('display.max_colwidth', 80)"),
    md("## Tabla principal (test, n = 120; IC 95 % bootstrap)"),
    code("pd.read_csv(R / 'main_table.csv').sort_values('dice_media')"),
    md("## Teardown"),
    code("t = S['teardown']\nprint('parámetros:', t['params'])\nprint('GFLOPs encoder: %.1f | global/ventana: %.2f' % (t['flops']['encoder_GFLOPs'], t['flops']['global_over_window_ratio']))\npd.DataFrame(t['flops']['blocks'])"),
    code("display(Image(F / 'teardown_flops_por_bloque.png', width=650))\ndisplay(Image(F / 'erf.png', width=650))\ndisplay(Image(F / 'pca_por_capa.png', width=650))"),
    md("## Sin prompt, AMG y calibración"),
    code("print(json.dumps(S.get('noprompt_tokens'), indent=1))\nprint(json.dumps(S.get('amg'), indent=1))\nprint(json.dumps(S.get('calibration'), indent=1))"),
    code("for f in ['sin_prompt_ejemplos.png', 'atencion_decoder.png', 'calibracion_amg.png', 'iou_vs_clics.png']:\n    if (F / f).exists():\n        display(Image(F / f, width=650))"),
    md("## Intervención: tokens aprendidos"),
    code("print(json.dumps({k: {kk: v[kk] for kk in ['test_dice_per_seed', 'mean', 'std', 'n_params', 'best_epoch']} for k, v in S.get('learned', {}).items()}, indent=1))\nfor f in ['cualitativo_tokens.png', 'comparacion_metodos.png']:\n    if (F / f).exists():\n        display(Image(F / f, width=650))"),
    md("## Pruebas pareadas (Wilcoxon) y fracción de la brecha recuperada"),
    code("pd.DataFrame(S['paired_tests']).T"),
    code("pd.Series(S['gap_recovered']).sort_values()"),
]
nbf.write(nb, NB / "02_resultados_E1.ipynb")
print("ok")
