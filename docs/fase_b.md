# Fase B: localización y prompts condicionados

Protocolo preparado el 12 de octubre de 2026, antes de ejecutar estos nuevos experimentos. Los resultados de E1 ya se inspeccionaron: esta fase es exploratoria y necesita evaluación externa para confirmar generalización.

En el workspace local se instaló SAM y scikit-learn en `.venv` con acceso a los paquetes existentes del sistema. Ejecutar los comandos con `.\.venv\Scripts\python.exe` en lugar de `python`; no hace falta activar el entorno. Las versiones y el commit oficial están en `docs/environment_phase_b.json`.

## Orden de ejecución

1. Comprobar insumos con `python scripts/11_preflight.py`. El código 1 indica insumos faltantes; no descarga archivos. Obtener imágenes/máscaras oficiales mediante `00_prepare_kvasir.py` y el checkpoint según README.
2. Reproducir E1 con caché FP16 e inicialización meanpoint, usando una carpeta de resultados nueva. No sobrescribir los CSV históricos.
3. Para la nueva fase, calcular **train400, dev y test con encoder FP32 y caché FP32**. Esto evita confundir la comparación principal con precisión diferente entre particiones. Los comandos de ejemplo son para una terminal; ejecutar por separado en PowerShell:

```text
python scripts/02_encode.py --split train400 --precision fp32 --storage fp32
python scripts/02_encode.py --split dev --precision fp32 --storage fp32
python scripts/02_encode.py --split test --precision fp32 --storage fp32
python scripts/10_prompt_perturbations.py --split dev
```

La nueva caché reemplaza embeddings locales incompatibles. Los resultados históricos no se regeneran automáticamente. Para comparar precisiones, guardar los artefactos y manifiestos de cada evaluación antes de cambiar la caché; no comparar ejecuciones de distintas rutas como si fueran el mismo experimento.

## Experimento espacial

Hipótesis: desplazar puntos y perturbar cajas degrada la segmentación; una caja de imagen completa no recupera lo mismo que una caja de lesión.

Se usa salida única en todas las condiciones, una consulta por componente elegible y unión de máscaras. Los puntos se desplazan en cuatro direcciones cardinales, por 0,5/1/2/4 radios interiores de la lesión, con recorte a límites de imagen. Las cajas se escalan alrededor de su centro por 0,75/1/1,25/1,5/2. Se incluyen sin prompt y caja de imagen completa. Todos los prompts espaciales de esta ablación son oráculo y usan GT; no son mitigaciones automáticas.

El CSV conserva imagen y dirección. Los IC de Dice se calculan remuestreando imágenes después de promediar direcciones dentro de cada imagen, para no contar direcciones como observaciones independientes. Se registran también IoU, máscaras estrictamente vacías y tiempo conjunto de generación de prompts, decoder, postproceso y métricas; no es latencia aislada del decoder. El primer ensayo puede limitarse a cinco imágenes con `--limit 5 --output_dir results/phase_b/smoke_perturbations`; no usar este ensayo como resultado final.

## Generador condicionado

Arquitectura: embedding SAM → promedio espacial 16×16 → atención de cuatro consultas aprendibles a características + codificación posicional SAM → actualización residual de cuatro prompts → decoder congelado. Se conserva la secuencia de tokens de salida y el embedding no_mask original. La agregación 16×16 es una decisión de costo; podría afectar lesiones pequeñas y debe declararse.

La red atiende también a posiciones de padding; no usa máscara explícita de padding en esta primera versión. Analizar su impacto antes de atribuir errores exclusivamente a la lesión.

Comparación principal: estático k=4 y condicionado k=4; mismas imágenes, inicialización meanpoint, pérdida BCE+Dice, seis épocas, lote 8, semillas 0/1/2. El generador tiene 331.008 parámetros frente a 1.024 estáticos: no atribuir toda diferencia exclusivamente al condicionamiento. La receta original proponía reutilizar LR 0,005; el ensayo técnico posterior detectó pérdida casi total de segmentación incluso con LR 0,001. La nueva arquitectura necesita ajustar LR en dev antes de comparar desempeño. El LR 0,00001 del ejemplo siguiente es un punto de partida para diagnóstico, no una configuración validada en dev completo.

Ejemplo de desarrollo, **sin evaluar test**:

```text
python scripts/07_learned_prompts.py --model static --k 4 --init meanpoint --seed 0 --epochs 6 --lr 5e-3 --bs 8 --output_root results/phase_b/dev_runs --dev_only --include_initial
python scripts/07_learned_prompts.py --model conditional --k 4 --init meanpoint --seed 0 --epochs 6 --lr 1e-5 --bs 8 --output_root results/phase_b/dev_runs --dev_only --include_initial
```

Repetir con semillas 1 y 2. Cada ejecución conserva configuración, entorno, IDs, metadatos de caché, historia de dev y parámetros entrenables. Una carpeta existente causa error para proteger resultados.

Una vez fijada la configuración en dev, ejecutar corridas en `results/phase_b/final_runs` sin `--dev_only`; entrenan de nuevo y eligen época en dev antes de evaluar test. `--include_initial` permite seleccionar época 0 si ninguna actualización mejora dev; sin esa opción se conserva el criterio histórico E1 de seleccionar entre épocas entrenadas. El resumen E1 no incorpora automáticamente estas carpetas: conservar la comparación de Fase B por separado. Hay resultados de comprobación técnica, pero no una evaluación completa de la nueva mitigación.

## Controles pendientes

- Comparación completa con inicialización meanpoint sin entrenar. El trainer ya guarda su evaluación inicial en dev.
- Probe puro, probe→caja y probe→caja+punto recalculados sobre la misma caché de la fase.
- Comparación de precisión FP16/FP32, especialmente en salida sin prompt.
- Comparación de calidad de localización, métricas por imagen, dispersión entre semillas y latencia.
- Evaluación externa y revisión de agrupaciones por paciente/video.

La ruta actual de entrenamiento/decoder es CPU, aunque el cálculo de embeddings puede usar GPU. Para usar GPU en entrenamiento hay que trasladar consistentemente modelo, embeddings, prompts y etiquetas; cambiar SAMCOST_DEVICE solo afecta 02_encode.py.

## Comprobaciones ejecutadas

- Dataset descargado del servidor oficial: 1.000 parejas, dimensiones correctas; área GT de las 120 imágenes de test idéntica a los registros E1. Zip SHA256: `03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7`.
- Pesos oficiales descargados y MD5 verificado: `01ec64d29a2fca3f0661936605ae66f8`.
- Ocho pruebas automáticas aprobadas; incluyen equivalencia del decoder por lotes con el oficial, gradientes a través del decoder congelado y rechazo de métricas con IDs incompletos o duplicados. Ejecutar `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- El ensayo inicial calculó embeddings FP32 para cuatro imágenes seleccionadas de train400 y dos de dev, sin bloques intermedios. Para completar caché se ejecutan los comandos sin `--limit`. `--no_blocks` es válido para estas comparaciones; los bloques intermedios solo son necesarios para otros análisis del encoder.
- Ablación real de dos imágenes dev: 48 filas y 12 condiciones en `results/phase_b/smoke_perturbations/dev/`.
- Tres corridas técnicas, cuatro imágenes train y dos dev, dos épocas, semilla 0: estático LR 0,001; condicionado LR 0,001; condicionado LR 0,00001 con selección que incluye época 0. Archivos en `results/phase_b/smoke_training/`. El primero condicionado cae de Dice inicial 0,253 a menos de 0,001; con LR menor las épocas entrenadas quedan alrededor de 0,243–0,245 y se conserva época 0. No hay evidencia de mejora del condicionado en este ensayo mínimo.
- Los tres checkpoints se recargaron y reprodujeron exactamente su Dice seleccionado en dev; guardan solo los parámetros entrenables. Ninguna corrida técnica evalúa test.

Estos resultados validan la ejecución y revelan sensibilidad a optimización. No estiman generalización ni justifican elegir el mejor método con solo dos imágenes de dev. El siguiente paso experimental pendiente es completar embeddings y ajustar la optimización en train/dev completos.

## Ejecución completa iniciada el 12 de octubre

`scripts/12_run_phase_b.py --output_root results/phase_b/full_dev_20261012 --threads 4` automatiza el protocolo D16: embeddings FP32 train400/dev sin bloques, ablación dev, barrido de LR condicionado, seis corridas con tres semillas y tres probes. No evalúa test. El estado puede consultarse con `.\.venv\Scripts\python.exe scripts/phase_b_status.py --experiment results/phase_b/full_dev_20261012`.

La ablación ya terminó: 100 imágenes dev, 2.400 evaluaciones y 12 condiciones. Dice sin prompt 0,0089; caja de imagen completa 0,0333; punto GT 0,6831; caja GT 0,9352. Desplazar el punto dos radios interiores reduce Dice a 0,1529 y duplicar la caja lo reduce a 0,3216. Resultados y curvas: `results/phase_b/full_dev_20261012/spatial_ablation.md`. Los entrenamientos completos siguen pendientes mientras se calcula train400.

`scripts/15_spatial_subgroups.py --experiment results/phase_b/full_dev_20261012` produce subgrupos descriptivos por área, múltiples componentes y contacto con borde, además de casos ilustrativos. Se añadió después de inspeccionar la ablación: es exploratorio. El fallo sin prompt aparece en todos los grupos de área; no se puede atribuir exclusivamente a lesiones pequeñas. La comparación oráculo con varias componentes usa más prompts.

Después de entrenar, ejecutar `scripts/14_conditioning_diagnostic.py --experiment results/phase_b/full_dev_20261012` para el control D17: decoder e imagen objetivo fijos, contexto del generador permutado. Antes de interpretarlo, el script verifica que los checkpoints recargados reproducen su Dice dev seleccionado.

El resumen automatizado `13_summarize_phase_b.py` ya ejecuta este control si falta su resultado y lo incorpora al informe. También incorpora el control exploratorio de selección ejecutado mediante `17_output_selection.py`: con punto, salida única 0,6831; elección por IoU predicho 0,5401; mejor candidato según GT 0,8600. Sin prompt, mejor de cuatro según GT 0,1824. Los casos reproducidos con contornos de predicción están en `results/phase_b/full_dev_20261012/spatial_cases/predictions.png`. Estas extensiones no alteran la configuración de entrenamiento fijada en D16.

## Estado del experimento versionado (revisión del 16 de octubre)

El commit `eb3e7e3` (12 de octubre, 18:29) guardó `results/phase_b/full_dev_20261012/` **con la ejecución en curso**. `status.json` registra `state: running` y `current_stage: static_k4_s1`. **El experimento está incompleto.**

| Etapa | Estado en el repositorio |
|---|---|
| `encode_dev`, `encode_train400` | Completas (exit 0) |
| `perturbations_dev` y controles derivados (subgrupos, casos, selección de salidas) | Completos |
| Barrido de LR del condicionado (`search/`, 3 corridas) | Completo; elegido LR 1e-4 (`selection.json`) |
| `static_k4_s0` | Completa |
| `static_k4_s1` | **Interrumpida**: `progress.json` con 3 de 6 épocas, sin `summary.json` ni `dev_per_image.csv`. No es un resultado; no citarla |
| `static_k4_s2`, `conditional_k4_s0–s2`, `probe_s0–s2` | Sin ejecutar |
| `13_summarize_phase_b.py` (incluye el control D17 de `14_conditioning_diagnostic.py`) | Sin ejecutar: no existen `report.md`, `table.csv` ni `summary.json` del experimento |

Por tanto, **todavía no hay comparación estático vs. condicionado vs. probe en dev**. Lo disponible son lecturas parciales de una semilla, sin IC:

- Estático, semilla 0: la inicialización meanpoint sin entrenar ya da Dice dev 0,3505, y la mejor época (6) llega a 0,3513.
- Barrido del condicionado, semilla 0: 0,3607 con LR 1e-5, 0,3602 con 3e-5 y 0,3987 con 1e-4.

### Cómo reanudar

1. **Confirmar con quien lanzó la corrida** si el proceso sigue activo o si hay resultados locales más nuevos sin subir. El commit es una foto a mitad de camino.
2. **Reanudar solo en la copia de trabajo original.** `12_run_phase_b.py` compara `plan.json`, que incluye el SHA-256 de siete archivos fuente calculado sobre los bytes en disco, con claves de ruta en formato Windows (`scripts\02_encode.py`). Verificado el 16 de octubre:
   - **Clon nuevo en Windows** (`core.autocrlf=true`): `scripts/10_prompt_perturbations.py` no coincide. El hash del plan corresponde a la versión con fin de línea LF, y el clon la tiene con CRLF.
   - **Linux/Colab**: no coinciden los otros seis archivos (sus hashes corresponden a CRLF), ni las claves con `\`.
   - **Cualquier edición** de `02_encode.py`, `06_probe.py`, `07_learned_prompts.py`, `10_prompt_perturbations.py`, `learned.py`, `decode.py` o `data.py` también bloquea la reanudación. Es intencional: evita mezclar versiones de código dentro de un experimento.
3. **Mover, no borrar**, la corrida parcial (las corridas fallidas se conservan, D15). Por ejemplo, `runs/static_k4_s1` → `interrumpidas/static_k4_s1_20261012`, dentro de la carpeta del experimento. Si sigue en `runs/`, `07_learned_prompts.py` lanza `FileExistsError` y la etapa queda como fallida.
4. **Relanzar** `.\.venv\Scripts\python.exe scripts\12_run_phase_b.py --output_root results\phase_b\full_dev_20261012 --threads 4`. Las etapas `complete` se saltan y `static_k4_s1` se repite desde cero.
5. **Antes de reanudar, decidir sobre la grilla de LR.** El valor elegido (1e-4) es el extremo superior de {1e-5, 3e-5, 1e-4}, y su mejor época fue la última del barrido. Ampliar la grilla cambia `plan.json` y exige una carpeta de experimento nueva.

Si no se puede reanudar (otra máquina, Colab o grilla ampliada), crear una carpeta nueva con `--output_root` y repetir el protocolo completo. Duraciones registradas con 4 hilos de CPU:

- `encode_train400`: 68,5 min. En la misma máquina se omite si la caché es válida, porque la firma incluye precisión, hashes, versión de torch y dispositivo.
- Cada corrida de 6 épocas: unos 7 min.
- Cada corrida del barrido: unos 4 min.
- Ablación espacial: 4,3 min.
