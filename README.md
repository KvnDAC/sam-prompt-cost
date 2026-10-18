# El costo del prompt en SAM

El borrador científico completo en inglés está en [WAIMLAP_DRAFT.md](WAIMLAP_DRAFT.md). La versión condensada en la plantilla oficial está disponible como [Word](WAIMLAP_DRAFT.docx) y [PDF de dos páginas](WAIMLAP_DRAFT.pdf); conserva campos pendientes para autoría y afiliaciones y no constituye un envío.

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

# 3) Pipeline. Cada corrida escribe resultados y figuras en su propia carpeta;
#    results/ y figures/ (artefactos versionados del E1) no se sobrescriben.
export SAMCOST_RUN_DIR=runs/$(date +%Y%m%d_%H%M%S)
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
python scripts/08_summarize.py                   # tabla principal, pruebas, $SAMCOST_RUN_DIR/summary_main.json
python scripts/09_figures.py
```

`scripts/02_encode.py` usa GPU automáticamente si está disponible (`SAMCOST_DEVICE=cpu` para forzar CPU). Todo lo demás corre sobre los embeddings en caché y es rápido en CPU.

**Carpeta por corrida.** Con `SAMCOST_RUN_DIR` definida, todos los scripts escriben en esa carpeta (las figuras, en su subcarpeta `figures/`). Para comparar con el E1, contrastar `$SAMCOST_RUN_DIR/summary_main.json` con `results/summary_main.json`. `runs/` está en `.gitignore`: versionar una corrida es una decisión explícita. En PowerShell: `$env:SAMCOST_RUN_DIR = "runs/$(Get-Date -Format yyyyMMdd_HHmmss)"`. Sin la variable, los scripts escriben en `results/` y `figures/`, y `07_learned_prompts.py` se detiene si la carpeta de la corrida ya existe, en lugar de sobrescribirla. La caché de embeddings (`cache/`) sí es compartida entre corridas.

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

La ampliación con FCN-Lite, DS-U-Net y U-Net compacta sobre train400/dev100 completó nueve corridas; una continuación controlada añadió seis corridas U-Net (128/256, tres semillas) y la suite actual suma **31 tests y 2 subtests aprobados**. U-Net 256 alcanza Dice 0,8267 en el endpoint fijo de 45 épocas; SAM automático da 0,8179 y su diferencia media frente a U-Net tiene IC pareado [−0,0472; 0,0281]. No se demuestra superioridad ni equivalencia. [Resultados y decisión científica](docs/resultados_experimentacion_20261017.md), [informe de resolución](results/model_comparison/unet_resolution_20261017/report.md), [protocolo inicial](docs/protocolo_comparacion_ligera.md) y [protocolo de continuación](docs/protocolo_resolucion_unet_20261017.md).

La curva nueva de **eficiencia de etiquetas** compara subconjuntos anidados de 25/100/400 máscaras, tres semillas y el mismo límite de actualizaciones. Dice de `probe→caja+punto→SAM`: 0,7821/0,8061/0,8190; U-Net desde cero: 0,5103/0,6198/0,6964. Las diferencias pareadas son +0,2718, +0,1863 y +0,1226, respectivamente, con IC que excluyen cero. Este resultado demuestra una ventaja bajo cómputo fijo en `dev100`; no contradice que la U-Net prolongada alcance 0,8267 con una trayectoria de entrenamiento distinta. [Protocolo](docs/protocolo_eficiencia_etiquetas_20261017.md), [informe](results/model_comparison/label_efficiency_20261017/report.md) y [curva](results/model_comparison/label_efficiency_20261017/label_efficiency.png).

La **validación externa congelada** evaluó los modelos de Kvasir directamente en los 612 frames y 29 secuencias de CVC-ClinicDB. Dice por frame: probe 0,4243; probe→caja+punto→SAM 0,4587; U-Net 256 madura 0,3405. SAM automático supera a U-Net por +0,1182, IC 95 % agrupado por secuencia [0,0672; 0,1684], pero la fuerte caída absoluta descarta interpretar la transferencia como calidad clínica suficiente. [Protocolo externo](docs/protocolo_validacion_externa_20261017.md) e [informe externo](results/model_comparison/external_clinicdb_20261017/report.md).

El **control negativo externo** usa las 1.009 imágenes `cecum` de HyperKvasir, sin ajuste en destino. La fracción falsa media es 0,0581 para el probe, 0,0521 para SAM automático y 0,1733 para U-Net; la tasa de área falsa ≥1 % es 87,94 %, 79,42 % y 99,31 %, respectivamente. SAM automático reduce el área falsa frente a U-Net en −0,1212 [−0,1267; −0,1159], pero genera alguna máscara en todas las imágenes. Es un control por etiqueta de clase anatómica, no especificidad clínica por paciente ni una máscara negativa revisada píxel a píxel. [Protocolo](docs/protocolo_especificidad_negativos_20261017.md) e [informe](results/model_comparison/negative_specificity_20261017/report.md).

La evaluación científica del alcance, el inventario de experimentos y el plan para comparar modelos están en [docs/decision_investigacion_waimlap.md](docs/decision_investigacion_waimlap.md). El [borrador de abstract WAIMLAp](docs/waimlap_2026/README.md) presenta el núcleo terminado como estudio exploratorio; todavía requiere completar autoría y verificar la versión de envío.

La [nota detallada sobre la utilidad de las comparaciones y el estado del arte](docs/nota_comparacion_y_estado_arte_20261017.md) explica qué variable aísla cada control, qué ideas ya fueron publicadas y cómo posicionar el aporte como diagnóstico de calidad/costo. También fija la prioridad experimental: eficiencia de etiquetas, evaluación externa y negativos antes de ampliar el ranking de modelos.

El documento de la rama `cambios-analisis` está en [docs/related_work.md](docs/related_work.md); sus afirmaciones contrastadas y correcciones propuestas, en [docs/revision_related_work.md](docs/revision_related_work.md). Las ejecuciones retomadas y el nuevo piloto U-Net se registran en [docs/experimentos_modelos.md](docs/experimentos_modelos.md).

La propuesta con Polyp-PVT y SAM 2.1, junto con antecedentes de self-prompting, está en [docs/estado_arte_y_comparacion_modelos.md](docs/estado_arte_y_comparacion_modelos.md). U-Net y dos controles RGB ya completaron train400/dev100; Polyp-PVT y SAM 2.1 siguen pendientes. El [informe de calidad/costo completo](results/model_comparison/comparison_train400_20261017/report_with_runtime.md) compara los prototipos en CPU; la continuación midió 13,30 ms para U-Net 128 y 27,08 ms para U-Net 256, frente a 8,37 s para SAM automático, con distinta resolución y preentrenamiento. El [abstract revisado con la continuación](docs/waimlap_2026/revision_20261017_resolution/Abstract_TheInformationalCostOfSpatialPromptsInSAM.pdf) ocupa dos páginas, conserva la plantilla y mantiene autoría pendiente; la versión previa también se conserva.

Protocolo y comandos en [docs/fase_b.md](docs/fase_b.md). Los resultados E1 usan encoder FP32 en test/dev, caché FP16 y decoder FP32. Las nuevas ejecuciones permiten `--storage fp32` y validan la procedencia de la caché. No mezclar resultados de ambas precisiones. Las corridas existentes se protegen contra sobrescritura.

El experimento de desarrollo `results/phase_b/full_dev_20261012/` está **completo en esta copia local**: 400 train/100 dev, tres semillas y 16 etapas terminadas. Dice medio: tokens estáticos 0,3633; condicionados 0,5013; probe 0,7489; probe → caja+punto → SAM 0,8179. El control de contexto permutado reduce el condicionado a 0,4953: la ventaja frente al estático no puede atribuirse íntegramente al contexto por imagen. Tabla, intervalos y límites en el [informe de desarrollo](results/phase_b/full_dev_20261012/report.md). No se realizó una nueva evaluación de test; la comparación completa con otras familias sigue pendiente.
