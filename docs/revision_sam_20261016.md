# Revisión de SAM y del estado de Git — 16 de octubre de 2026

**Verificación posterior del 17:** se resolvió el test de rutas/temporales y pasaron las 21 pruebas de la suite ampliada. Hay nueve corridas RGB completas sobre train400/dev100; [resultados actuales](resultados_experimentacion_20261017.md). Las limitaciones y estados de Git descritos abajo corresponden a la inspección anterior.

**Actualización posterior:** el entrenamiento se retomó y completó las 16 etapas. Las observaciones de interrupción y las instrucciones Git de la sección 6 son históricas. Tras descargar referencias, `origin/fix/pruebasExperimentos` quedó en `51185c6` y se leyó `origin/cambios-analisis` (`ca03c31`); no hubo switch, merge ni commit local. Consultar el cierre y los pilotos U-Net en [experimentos_modelos.md](experimentos_modelos.md). No archivar ni reiniciar las corridas completas.

Se contrastaron ramas, commits, cambios locales, scripts, documentación, logs, manifiestos y métricas del experimento. Esta revisión no cambió de rama, no hizo commits, no descartó archivos y no reinició entrenamientos. Las correcciones realizadas se limitan a documentación.

## 1. Qué ocurrió con Git

`git switch main` se canceló porque dos archivos locales contienen avances del experimento que no están en el commit:

- `results/phase_b/full_dev_20261012/runs/static_k4_s1/progress.json`.
- `results/phase_b/full_dev_20261012/status.json`.

Al fallar el cambio de rama, se mantuvo activa `fix/pruebasExperimentos`. Después, `git pull origin main` integró `origin/main` **en esa rama activa**, mediante fast-forward. El comando no cambió automáticamente a `main`.

| Referencia inspeccionada | Commit | Interpretación |
|---|---|---|
| `HEAD`, rama `fix/pruebasExperimentos` | `dc88ba5` | Rama que sigue activa |
| `origin/main` | `dc88ba5` | Coincide con HEAD al revisar las referencias descargadas |
| `main` local | `ab30c46` | Sigue atrás; no se actualizó al fallar el switch |
| `origin/fix/pruebasExperimentos` | `eb3e7e3` | Su referencia remota quedó en el commit previo de experimentos |

El indicador `ahead 3` se refiere a la comparación con `origin/fix/pruebasExperimentos`; no significa que esos tres commits falten en `origin/main`. En esta inspección HEAD y `origin/main` son idénticos.

Los archivos locales que bloquearon el switch contienen avances reales: `static_k4_s1` pasó de tres a seis épocas y el estado registró su finalización, la de `static_k4_s2` y el inicio del condicionado. También existen resúmenes, métricas y checkpoints locales todavía sin versionar.

## 2. Cambios que llegaron con el pull

El commit `51185c6`, incorporado mediante el merge `dc88ba5`, añadió `SAMCOST_RUN_DIR`. Cuando se define esa variable, las rutas por defecto de resultados y figuras se redirigen a una carpeta por corrida. Sin la variable se conservan `results/` y `figures/`.

Se actualizaron README, Colab, su generador, `.gitignore`, la documentación y una prueba de rutas. Esto resuelve el conflicto entre reproducir E1 y la protección de carpetas ya existentes. La caché `cache/` sigue compartida, incluidos sus artefactos auxiliares; separar resultados no aísla esa caché ni vuelve seguras ejecuciones simultáneas con configuraciones diferentes. Los argumentos explícitos `--output_root`/`--output_dir` siguen determinando sus salidas correspondientes.

Se verificó que los siete hashes de código de `full_dev_20261012/plan.json` coinciden con los archivos de esta copia original. `common.py` cambió y no forma parte de esos siete hashes; la variable `SAMCOST_RUN_DIR` no estaba definida durante esta revisión. La coincidencia del plan no debe interpretarse como un hash exhaustivo de todas las dependencias del proyecto.

## 3. Qué quedó efectivamente terminado

| Trabajo | Evidencia local y estado |
|---|---|
| Preparación de Kvasir-SEG y SAM ViT-B | Datos y checkpoint disponibles, con verificaciones registradas en los manifiestos |
| Embeddings FP32 | 400/400 train400 y 100/100 dev |
| Ablación espacial | 100 imágenes, 2.400 evaluaciones y 12 condiciones |
| Control de selección de máscaras | 100 imágenes; reproducción de 200 salidas únicas |
| Subgrupos y casos ilustrativos | Informes, métricas y figuras disponibles |
| Barrido condicionado | Tres LR, semilla 0, tres épocas; seleccionado `1e-4` |
| Estático, semillas 0/1/2 | Las tres corridas terminaron seis épocas y guardaron resumen y métricas dev |
| Condicionado, semilla 0 | Parcial: tres de seis épocas, sin `summary.json` ni evaluación dev final del checkpoint seleccionado |
| Condicionado, semillas 1/2 | Sin corridas locales |
| Probes FP32 de la comparación | Sin carpeta `probes/` |
| Contexto permutado y resumen final | Sin ejecutar; faltan `conditioning_diagnostic/`, `report.md`, `table.csv` y `summary.json` globales |

No se encontraron procesos `python`/`pythonw` activos al comprobar la ejecución. Los últimos logs del condicionado contienen épocas 1, 2 y 3. El estado persistido sigue diciendo `running`, pero **la ejecución está interrumpida**; ese campo no es una comprobación de actividad del proceso. No se puede determinar la causa de la interrupción solamente a partir de esos archivos.

La documentación que inspeccionó únicamente el commit describía `static_k4_s1` como interrumpida. Esa afirmación no describe esta copia local: sí terminó y su resultado debe conservarse. Se corrigió la distinción entre la fotografía versionada y los archivos locales en `fase_b.md`.

## 4. Resultados de SAM que sí están respaldados

### Dependencia del prompt en las 100 imágenes dev

| Condición | Dice medio |
|---|---:|
| Sin prompt, token 0 | 0,0089 |
| Caja de imagen completa | 0,0333 |
| Punto GT interior | 0,6831 |
| Punto desplazado dos radios interiores | 0,1529 |
| Caja GT precisa | 0,9352 |
| Caja ampliada 2× | 0,3216 |

El resultado respalda la importancia de ubicación y extensión precisas del prompt. Puntos y cajas son referencias oráculo obtenidas de GT; no representan localización automática ni tiempo humano de anotación.

En el control de selección, sin prompt el mejor de cuatro según GT alcanza 0,1824. Elegir mejor entre esas cuatro candidatas no recupera la referencia de caja. Con punto, la selección por IoU predicho da 0,5401 frente a 0,8600 del mejor de tres según GT; hay candidatos útiles que la selección no identifica. Estos controles son exploratorios y los oráculos no son métodos desplegables sin GT.

### Tokens estáticos: tres corridas completas

Se verificó que cada CSV final contiene los 100 IDs dev únicos y que su media coincide con el resumen. Cada manifiesto usa 400 train, 100 dev y cero imágenes de test.

| Semilla | Dice inicial | Mejor Dice dev | Época seleccionada |
|---|---:|---:|---:|
| 0 | 0,3505 | 0,3513 | 6 |
| 1 | 0,3486 | 0,3625 | 2 |
| 2 | 0,3478 | 0,3762 | 3 |

La media final es **0,3633**, con desviación entre semillas **0,0125**. Se recalculó un IC 95 % por bootstrap de imágenes, después de promediar semillas por imagen: **[0,2913; 0,4375]**. La media de las inicializaciones es 0,3490; la diferencia final−inicial es +0,0143, con IC pareado **[−0,0383; 0,0702]**. Este intervalo no establece una ganancia media concluyente. Se usaron 10.000 remuestreos y semilla 0; el intervalo no incorpora toda la incertidumbre de reentrenamiento y las épocas se seleccionaron en dev.

### Condicionado: búsqueda terminada, comparación final incompleta

El barrido de tres épocas seleccionó LR `1e-4`, con mejor Dice dev 0,3987. La corrida posterior de seis épocas quedó en la época 3 con Dice 0,4766 registrado en `progress.json`. **0,4766 es una observación intermedia, no un resultado final ni una comparación de tres semillas.** No se verificó en esta revisión recargando ese checkpoint y no se usó para afirmar superioridad de la mitigación.

SAM permanece congelado. El generador condicionado tiene 331.008 parámetros entrenables frente a 1.024 del estático; una diferencia futura de score no aislará automáticamente el efecto del condicionamiento.

## 5. Verificación técnica del código incorporado

Se ejecutó la suite actual de nueve pruebas en `.venv`: **ocho pasaron y una terminó con error de permisos del entorno**. Las ocho verifican decoder por lotes, gradientes con SAM congelado, dependencia de imagen, padding, lift de atención, caché, coordenadas y alineación de IDs.

La nueva prueba `test_run_dir_redirects_results_and_figures` falló al acceder al directorio creado con `tempfile.TemporaryDirectory()` bajo `AppData/Local/Temp`, con `WinError 5`. No se presenta la suite como nueve pruebas aprobadas.

Se comprobó por separado la misma resolución de rutas en subprocesos, usando una carpeta desechable dentro del workspace. Las tres condiciones pasaron: variable ausente, ruta absoluta y ruta relativa a la raíz. También se verificó que se crea la carpeta de salida. Esto respalda el funcionamiento de la redirección, pero no elimina el error de permisos de la prueba automática en este entorno.

Las pruebas con decoder oficial usan su arquitectura y entradas sintéticas; no sustituyen la inferencia real de los experimentos ni prueban generalización clínica. No se reentrenaron modelos durante esta revisión.

## 6. Cómo cambiar a main conservando lo local

Después de revisar estos archivos, los siguientes comandos guardan también los no versionados. Como la inspección no encontró entrenamientos activos, no hay un proceso identificado que siga escribiéndolos en este momento:

```powershell
git stash push -u -m "Respaldo de resultados SAM antes de cambiar a main"
git switch main
git pull --ff-only origin main
git stash apply 'stash@{0}'
git status --short --branch
```

`apply` recupera el trabajo sobre `main` y conserva el respaldo en el stash. Si se produjera un conflicto, revisar antes de eliminar el respaldo. Estos comandos no se ejecutaron durante la revisión. `stash -u` incluye archivos no versionados, pero no los ignorados, como datos, `.venv` y caché; esos archivos permanecen en el workspace.

No utilizar `git restore` sobre los dos JSON para desbloquear el switch: descartaría el estado local posterior al commit. Agregar un patrón a `.gitignore` tampoco deja de rastrear archivos que ya están versionados.

## 7. Qué sigue en experimentación

Para retomar este experimento hay que conservar las tres corridas estáticas y archivar solamente la carpeta parcial `runs/conditional_k4_s0`. Después se puede relanzar el pipeline con el mismo plan. No basta con relanzarlo sin archivar: el trainer rechaza una carpeta existente.

La etapa parcial se repite desde cero; el checkpoint del prompt no contiene los estados completos de optimizador y scheduler para continuar exactamente en la época 4. La revisión no movió carpetas ni reinició ese trabajo.

Los pasos pendientes son terminar tres semillas condicionadas, calcular probes y auto-prompts sobre la misma caché, ejecutar el control de contexto y generar la comparación final. Las decisiones usan dev: el test histórico E1 ya fue inspeccionado y una evaluación externa sigue pendiente. Los resultados dev FP32 no se deben comparar directamente con los de test E1 como si fueran mejoras del modelo.

Evidencia complementaria: [trabajo realizado](trabajo_realizado_y_pruebas.md), [protocolo y reanudación](fase_b.md), [ablación espacial](../results/phase_b/full_dev_20261012/spatial_ablation.md) y [selección de máscaras](../results/phase_b/full_dev_20261012/output_selection/report.md).
