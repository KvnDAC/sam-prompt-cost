# El costo del prompt en SAM

Proyecto de investigación de INF659 Deep Learning (Maestría en Informática, 2026-2).
**Pregunta:** ¿qué parte del desempeño de Segment Anything (SAM) en segmentación de pólipos depende de la información que aporta el prompt humano, en qué componente se origina la caída cuando el prompt se elimina, y cuánto de esa brecha recupera un prompt aprendido sin humano?

Este repositorio corresponde al **Entregable 1** (propuesta, teardown, intervención preliminar, cuaderno de decisiones y related work inicial). El cuaderno de decisiones está en [`docs/design_log.md`](docs/design_log.md).

## Resultados principales del E1 (test oficial, 120 imágenes)

| Modo | Humano en test | Dice [IC 95 %] |
| --- | --- | --- |
| Sin prompt, salida única (token 0) | — | 0,012 [0,001–0,028] |
| AMG 16×16, mayor IoU predicho | — | 0,058 [0,036–0,082] |
| AMG 16×16, mejor propuesta (oráculo) | — | 0,895 [0,874–0,914] |
| Tokens aprendidos k = 4, SAM congelado (3 semillas) | — | 0,318 [0,253–0,383] |
| Probe lineal → caja → SAM | — | 0,622 [0,570–0,674] |
| Probe lineal → caja + punto → SAM | — | 0,655 [0,596–0,712] |
| 1 clic por pólipo, salida única | 1 clic | 0,696 [0,638–0,753] |
| Caja por pólipo | 1 caja | 0,886 [0,852–0,914] |

Resumen completo en `results/summary_main.json` y `results/main_table.csv`; el documento del E1 explica cada cifra.

## Estructura

```
data/            particiones (splits/) y json oficial de cajas; las imágenes NO se versionan
scripts/         pipeline numerado (00 → 09), cada script escribe en results/ o figures/
src/samcost/     código común: datos, decoder sobre caché, AMG, prompts, métricas, tokens aprendidos
results/         métricas por imagen (csv), resúmenes (json)
figures/         figuras del documento
docs/            cuaderno de decisiones
notebooks/       cuaderno para reproducir en Colab y cuaderno de resultados
```

## Reproducir

```bash
pip install -r requirements.txt

# 1) Pesos oficiales de SAM ViT-B (md5 01ec64d29a2fca3f0661936605ae66f8)
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth

# 2) Kvasir-SEG oficial (https://datasets.simula.no/kvasir-seg/) y particiones
python scripts/00_prepare_kvasir.py --official_zip kvasir-seg.zip

# 3) Pipeline
python scripts/01_teardown.py                      # parámetros, FLOPs, latencia, memoria
python scripts/02_encode.py --split test --precision fp32
python scripts/02_encode.py --split dev --precision fp32
python scripts/02_encode.py --split train400 --precision bf16
python scripts/03_prompt_cost.py --split test --modes noprompt,point,box,clicks
python scripts/03_prompt_cost.py --split test --modes amg --pps 16
python scripts/04_receptive_field.py
python scripts/05_features.py
python scripts/06_probe.py
for k in 4 1; do for s in 0 1 2; do
  python scripts/07_learned_prompts.py --k $k --seed $s --init meanpoint --epochs 6 --lr 5e-3 --bs 8
done; done
python scripts/07b_twostage.py                   # tokens -> caja -> SAM (dos etapas)
python scripts/05b_attention_stats.py            # hacia dónde mira el token de máscara
python scripts/08b_subgroups.py test             # atributos por imagen para subgrupos
python scripts/08_summarize.py                   # tabla principal, pruebas, results/summary_main.json
python scripts/09_figures.py
```

`scripts/02_encode.py` usa GPU automáticamente si está disponible (`SAMCOST_DEVICE=cpu` para forzar CPU). Todo lo demás corre sobre los embeddings en caché y es rápido en CPU.

## Semillas y entorno

- Particiones: `data/splits/test.txt` = `val.txt` oficial de MediaEval 2020 (120); `train.txt`/`dev.txt` = 780/100 del `train.txt` oficial con semilla 0; `train400.txt` = subconjunto fijo (semilla 1234) usado en el piloto.
- Tokens aprendidos: semillas 0, 1, 2 (`--seed`), `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Entorno del E1: Python 3.13.16, PyTorch 2.14.1 (CPU), 2 vCPU Intel Xeon con AMX, 7 GB RAM, sin GPU.

## Datos y licencias

- **Kvasir-SEG** (Jha et al., MMM 2020): uso solo para investigación y educación; citar el artículo. Ni las imágenes ni `kavsir_bboxes.json` se redistribuyen aquí; el script los toma del zip oficial.
  Para el E1 el entorno de trabajo no podía acceder al servidor oficial; se usó un espejo público en GitHub y se restauraron y verificaron los ids oficiales (ver `docs/design_log.md`, D4). Para el trabajo final, regenerar con `--official_zip`.
- Las figuras de `figures/` y del cuaderno de resultados muestran imágenes de Kvasir-SEG con fines académicos, bajo los mismos términos.
- **SAM** (Kirillov et al., ICCV 2023): código y pesos Apache-2.0.
- Partición oficial: <https://github.com/DebeshJha/2020-MediaEval-Medico-polyp-segmentation>.

## Uso de asistentes de IA

El código, la ejecución de los experimentos del E1 y el borrador de la documentación se hicieron con asistencia de Claude (Anthropic). Cada decisión asistida está marcada con **[IA]** en el cuaderno de decisiones; el equipo es responsable de revisarlas y defenderlas.

## Continuación: Fase B

Protocolo y comandos en [docs/fase_b.md](docs/fase_b.md). Los resultados E1 usan encoder FP32 en test/dev, caché FP16 y decoder FP32. Las nuevas ejecuciones permiten `--storage fp32` y validan la procedencia de la caché. No mezclar resultados de ambas precisiones. Las corridas existentes se protegen contra sobrescritura.
