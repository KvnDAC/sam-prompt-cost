# Inicio de comparación de modelos — 16 de octubre de 2026

**Actualización del 17 de octubre:** completadas tres semillas sobre 400 train/100 dev para FCN-Lite, DS-U-Net y U-Net compacta (Dice 0,6002 / 0,6548 / 0,7814), control de prior espacial y suite completa de 21 tests. Ver [resultados actuales](resultados_experimentacion_20261017.md). Las referencias a «benchmark U-Net pendiente» más abajo describen la etapa anterior y sus pilotos de 32/16; no son el estado actual de esta comparación ligera.

Este registro distingue pruebas de ejecución de una comparación científica terminada. Los experimentos nuevos de esta sesión no evalúan test.

**Estado al cierre:** Fase B de SAM completa (16 etapas); tres pilotos U-Net completos. La comparación completa entre familias permanece pendiente. Las secciones de inicio y decisiones previas conservan la cronología; los resultados finales se detallan al final.

## Trabajo iniciado

- Se descargaron las referencias de `origin` y se leyó `origin/cambios-analisis`, commit `ca03c31380de04d35ad5b2f3bd35bcfc71f0c328`, sin cambiar de rama ni mezclar commits.
- Se copió exactamente su `docs/related_work.md` al workspace para revisión. SHA-256 de la copia: `f9e08091e319d1d640baaec9120b8141e9aaf65302f3341297dffd3d85ed3589`. Las observaciones se guardan por separado.
- Se comprobó el preflight train400/dev: paquetes, imágenes, máscaras, caché y MD5 del checkpoint disponibles.
- Se archivó la corrida parcial `conditional_k4_s0`, incluido su checkpoint, log y estado previo del pipeline, en `results/phase_b/full_dev_20261012/interrumpidas/conditional_k4_s0_20261016_e79f4dda/`.
- Se relanzó el protocolo original de Fase B con cuatro hilos. Conserva las tres semillas estáticas completas y reinicia únicamente la etapa parcial desde cero. Los siete hashes del plan siguen coincidiendo. La variable `SAMCOST_RUN_DIR` se dejó sin definir en ese proceso para preservar sus rutas originales.

## Baseline nuevo: U-Net

Implementación en `src/samcost/unet.py` y `scripts/18_unet_baseline.py`. Cuatro etapas de reducción/ampliación con skips, convoluciones con padding y salida binaria de logits. Es una variante U-Net declarada, no una réplica exacta de la receta de 2015. Ancho configurable; sin preentrenamiento ni batch norm.

La red recibe RGB [0,1], redimensionado conservando proporción y rellenado hasta el tamaño elegido. La GT se redimensiona con vecino más cercano. BCE+Dice ignora el padding. La evaluación elimina ese padding antes de volver a resolución original. Aumentos de train: flips horizontales y verticales independientes. Se guarda inicialización, checkpoint elegido en dev, métricas por imagen, versiones, hashes e IDs; se verifica el score al recargar.

Dos pruebas nuevas aprobadas:

1. Una máscara rectangular recupera exactamente su geometría después del redimensionado y recorte, incluso si el padding contiene logits positivos extremos.
2. La salida mantiene el tamaño esperado y la pérdida propaga gradientes hasta la primera convolución y la salida.

### Piloto corto terminado

32 imágenes de train400 seleccionadas con semilla fija 1234, primeras 16 de dev, semilla de modelo 0, lado 256, ancho 16, lote 4, Adam LR 0,001, tres épocas y un hilo CPU. **1.941.105 parámetros entrenables**. Se ejecutó mientras el pipeline SAM usaba CPU, por lo que sus tiempos no son un benchmark de latencia.

Pérdida train: 1,4730 → 1,4187 → 1,3945. Dice dev: 0,0000 en las tres épocas y en la inicialización. Se seleccionó época 0 y se verificó su checkpoint. Se conserva este ensayo negativo en `runs/model_comparison/unet_pilot_20261016/`.

Esto comprueba la ejecución y detecta una receta todavía insuficiente; no demuestra que U-Net rinda peor que SAM. No tiene los 400/100 IDs de la comparación ni un entrenamiento convergido.

### Ampliación del piloto, fijada antes de ejecutarla

Repetir la misma configuración e IDs, desde la misma semilla, con **20 épocas** en una carpeta nueva `runs/model_comparison/unet_pilot20_20261016/`. Es un diagnóstico de optimización del baseline; no cambia el plan de SAM ni utiliza test. Conservar el piloto corto y reportar ambos aunque no mejoren. La receta para un benchmark final aún debe elegirse en dev con presupuesto de cómputo adecuado y tres semillas.

```powershell
.\.venv\Scripts\python.exe scripts/18_unet_baseline.py --output_dir runs/model_comparison/unet_pilot20_20261016 --train_limit 32 --dev_limit 16 --epochs 20 --base_channels 16 --image_size 256 --batch_size 4 --threads 1 --seed 0
```

## Comparación completa todavía pendiente

### Resultado del piloto ampliado y siguiente control

El piloto sin normalización de 20 épocas terminó con Dice dev 0 en todas las épocas, mientras la pérdida train bajó de 1,4730 a 1,0885. Se conservaron sus métricas y snapshots exactos del código en `results/model_comparison/unet_pilot20_20261016/`. El modelo aprende una región sintética visible en una prueba separada; eso no garantiza que aprenda pólipos con esta receta.

Antes de ejecutarlo, se fija un siguiente piloto: mismo train32/dev16, semilla 0, tamaño 256, ancho 16, lote 4, LR 0,001 y 20 épocas, **añadiendo GroupNorm**. Sigue siendo una variante U-Net, sin preentrenamiento. La única intervención arquitectónica prevista es normalización; se registran también probabilidad media, pérdida dev y vacío, y se guardan pesos finales además del checkpoint seleccionado. Las recetas anteriores no se modifican. Este cambio se decide en desarrollo tras observar un fallo, por lo que no es un contraste confirmatorio.

```powershell
.\.venv\Scripts\python.exe scripts/18_unet_baseline.py --output_dir runs/model_comparison/unet_groupnorm_pilot_20261016 --train_limit 32 --dev_limit 16 --epochs 20 --base_channels 16 --normalization group --image_size 256 --batch_size 4 --threads 1 --seed 0
```

Cuatro pruebas U-Net cubren geometría, gradientes, aprendizaje sintético y logits/gradientes finitos con GroupNorm. La comprobación de aprendizaje sintético exige Dice >0,95 en su tarea artificial y no usa imágenes de test.

### Piloto GroupNorm terminado

Se completaron sus 20 épocas. Dice dev inicial 0,2406, mejor Dice **0,5082 en época 12**, y última época 0,4227. La pérdida train bajó de 1,4235 a 0,8967. Tiene **1.944.049 parámetros**. El checkpoint seleccionado se recargó y reprodujo los 16 Dice individuales a tolerancia 1e−8. Se conservaron también el endpoint de época 20 y sus métricas.

En este piloto, GroupNorm resolvió la salida vacía que presentaba la receta sin normalización. No demuestra una mejora general sobre U-Net estándar ni permite compararlo con SAM entrenado sobre 400 imágenes y evaluado en 100. Hubo selección de receta después de resultados negativos y selección de época en las 16 imágenes dev.

Los [gráficos de pérdida y Dice](../results/model_comparison/figures/pilot_curves.png) muestran las tres recetas. Las primeras tres épocas sin normalización se superponen porque se repitió la misma semilla/configuración. El [panel cualitativo](../results/model_comparison/figures/pilot_predictions.png) usa las primeras cuatro imágenes dev, sin seleccionarlas por score; muestra segmentación parcial y falsos positivos que todavía deben mejorarse.

Métricas y snapshots de código: `results/model_comparison/unet_pilot_20261016/`, `unet_pilot20_20261016/` y `unet_groupnorm_pilot_20261016/`. Los checkpoints completos están en las carpetas correspondientes bajo `runs/model_comparison/`.

Un futuro ensayo de partición completa puede usar `18_unet_baseline.py` con límites cero, `--base_channels 16 --normalization group`, una receta elegida en dev y tres semillas. No se lanzó todavía ese benchmark ni se eligió una receta final de entrenamiento completo.

Ejemplo preparado para desarrollo en los 400 train/100 dev (carpeta nueva, semilla 0):

```powershell
.\.venv\Scripts\python.exe scripts/18_unet_baseline.py --output_dir runs/model_comparison/unet_gn16_train400_s0 --epochs 30 --base_channels 16 --normalization group --image_size 256 --batch_size 4 --threads 4 --device cpu --seed 0
```

El script admite `--device cuda` cuando exista GPU. Treinta épocas es una receta candidata a validar, no una garantía de convergencia. Las evaluaciones de latencia deben usar el mismo hardware/dispositivo para todos los métodos, aunque el entrenamiento se haga en GPU. Elegida la receta en dev, repetir con semillas 1 y 2; no mezclar los pilotos con esas corridas completas.

Para una tabla comparativa: train400 y dev100 completos, tres semillas, selección de receta en dev y evaluación con configuración congelada. Los pilotos de 32/16 imágenes no deben agregarse a esa tabla. SAM 2.1, MedSAM y Polyp-PVT todavía no tienen una ejecución propia en este workspace.

Las carpetas `runs/` están ignoradas por Git. Se conservaron copias ligeras de manifiestos, curvas y métricas de los pilotos bajo `results/model_comparison/`; los pesos completos permanecen en `runs/`. No se debe afirmar que subir un commit sube automáticamente esos checkpoints.

## Cierre de Fase B: 400 train / 100 dev

El pipeline original terminó con código 0 y las 16 etapas completas. Los tres condicionados eligieron la época 6, última del presupuesto: no se afirma convergencia. Se conservó la corrida parcial archivada y no se modificó ninguno de los siete archivos con hash en el plan.

| Método | Dice medio dev | DE entre semillas |
| --- | ---: | ---: |
| Sin prompt, salida única | 0,0089 | — |
| Tokens estáticos k=4 | 0,3633 | 0,0125 |
| Tokens condicionados k=4 | 0,5013 | 0,0089 |
| Probe lineal | 0,7489 | 0,0007 |
| Probe → caja → SAM | 0,7738 | 0,0025 |
| Probe → caja+punto → SAM | 0,8179 | 0,0053 |
| Caja GT (referencia privilegiada) | 0,9352 | — |

Los métodos entrenados usan semillas 0/1/2. El [informe](../results/phase_b/full_dev_20261012/report.md), [resumen JSON](../results/phase_b/full_dev_20261012/summary.json) y [figura](../results/phase_b/full_dev_20261012/comparison_dev.png) contienen intervalos por imagen y comparaciones exploratorias. Condicionado menos estático: +0,1380, IC pareado 95 % [0,0895; 0,1898]. Probe → caja+punto menos probe: +0,0690 [0,0471; 0,0919]. Son scores de desarrollo con selección en dev, no resultados nuevos de test ni comparación directa con la tabla E1.

### Qué aporta realmente el contexto

El control fijado en D17 mantiene imagen objetivo, decoder y pesos, y permuta únicamente la imagen cuyo embedding recibe el generador. La misma permutación sin puntos fijos se aplica en las tres semillas. Dice normal 0,501337; permutado 0,495295; diferencia **0,006042**, IC 95 % [0,000840; 0,011245], p exploratorio de Wilcoxon 0,0191, sin ajuste por comparaciones múltiples.

La sensibilidad al emparejamiento del contexto es pequeña en este control. No basta observar condicionado > estático para atribuir toda la mejora al contexto: cambian también capacidad y optimización. Este control tampoco descompone causalmente esas contribuciones. Conviene añadir un generador de igual capacidad con contexto constante antes de sostener una explicación mecanística fuerte.

### Verificación y siguiente experimento

Se verificaron siete hashes congelados, seis CSV finales con 100 IDs únicos cada uno y la reproducción de los **300 Dice individuales** del condicionado bajo contexto correcto al recargar los checkpoints del diagnóstico. Sus permutaciones tienen 100 contextos únicos sin coincidencias consigo mismos. La copia del related work mantiene su SHA-256 original. Los cuatro tests nuevos U-Net pasaron; la prueba previa de temporales de Windows mantiene la limitación de permisos documentada, por lo que no se declara toda la suite aprobada.

El siguiente trabajo es ejecutar U-Net con train400/dev100 y tres semillas tras fijar la receta en desarrollo, seguido de SAM 2.1 con modos de prompt compatibles y los mismos IDs. Medir además latencia, memoria y parámetros entrenables con hardware común. Los pilotos U-Net de 32/16 no se incluyen en la tabla anterior. Después se necesita evaluación externa: el test histórico ya se inspeccionó en E1. No se han entrenado aquí SAM 2.1, MedSAM ni Polyp-PVT.
