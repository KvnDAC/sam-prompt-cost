# Qué se hizo y qué se probó en el proyecto SAM

**Ampliación del 17 de octubre:** completadas nueve corridas RGB con 400 train/100 dev y 21 tests aprobados. Nuevos controles y lectura científica en [resultados_experimentacion_20261017.md](resultados_experimentacion_20261017.md); los cortes que siguen son históricos.

**Cierre posterior de la misma jornada:** Fase B completó sus 16 etapas, tres semillas y control de contexto. También terminaron tres pilotos U-Net. Resultados actuales y límites en [experimentos_modelos.md](experimentos_modelos.md); el [informe SAM](../results/phase_b/full_dev_20261012/report.md) contiene la tabla de desarrollo. Las referencias a interrupción o entrenamiento pendiente que siguen son cortes históricos, anteriores a este cierre.

**Actualización del 16 de octubre de 2026:** el texto y el corte original de este documento describen la ejecución del 12 de octubre. En la revisión actual se encontraron las tres semillas estáticas completas y el condicionado semilla 0 parcial en época 3/6, sin procesos Python activos. La ejecución ya no está en marcha. Estado contrastado y explicación de Git en [revision_sam_20261016.md](revision_sam_20261016.md).

Este documento resume el trabajo de revisión, corrección y experimentación realizado durante la continuación del proyecto. Separa resultados históricos inspeccionados, pruebas técnicas, experimentos nuevos terminados y ejecuciones pendientes. El corte y las referencias a ejecución en curso que siguen corresponden al 12 de octubre; la actualización enlazada arriba describe la interrupción comprobada el 16.

## 1. Objetivo y términos

La pregunta del proyecto es: **¿dónde y por qué pierde calidad el mask decoder de SAM sin prompts espaciales, y qué mitigación puede recuperar esa calidad?**

Se trabaja con SAM ViT-B y Kvasir-SEG, segmentando las máscaras de pólipos. En esta fase, los prompts son puntos, cajas o vectores aprendidos. No se implementó un modelo de lenguaje ni un sistema de prompts de texto.

- **GT:** máscara de referencia del dataset.
- **Prompt oráculo:** punto o caja calculados usando GT; aportan información privilegiada y sirven como referencia de diagnóstico.
- **Sin prompt espacial:** cero tokens de puntos/cajas y sin máscara de entrada; el decoder conserva sus tokens de salida, posiciones y embedding `no_mask`.
- **Sin intervención humana en inferencia:** el método genera sus propios prompts. Puede haber usado máscaras para entrenarse.
- **Dice:** solapamiento entre predicción y GT, de 0 a 1; mayor es mejor.

El costo medido hasta ahora es la dependencia de información espacial y de su precisión. **No se midió tiempo humano de anotación.**

## 2. Revisión inicial del proyecto

Se revisaron README, scripts, módulos, notebooks, particiones, documentos de decisiones, tablas y figuras existentes. Se verificaron las particiones train/dev/test de 780/100/120 imágenes, su separación y que train400 es un subconjunto de train.

Se recalcularon estadísticas de los resultados E1 guardados, incluyendo sin prompt, punto, caja, probe y selección de propuestas. Esto fue una comprobación de consistencia de los artefactos históricos; no una nueva ejecución completa de E1 desde imágenes.

La revisión detectó problemas de reproducción y del relato científico: comandos sin la inicialización usada realmente, almacenamiento FP16 descrito simplemente como FP32, muestras faltantes que podían descartarse silenciosamente y confusión entre máscaras vacías y casi vacías. La revisión original está en [analisis_experto.md](analisis_experto.md); describe el estado previo a las correcciones.

## 3. Cambios implementados

| Cambio | Propósito y alcance |
|---|---|
| Comandos de README, Colab y generador de notebooks con `meanpoint` | Reproducir la inicialización de los tokens reportados; añadir etapas omitidas del cuaderno. |
| Caché con almacenamiento FP16 o FP32 explícito | Separar precisión del encoder y precisión del archivo almacenado. |
| Validación de caché y hashes | Registrar imágenes, pesos, código del encoder, precisión y entorno; recalcular entradas incompatibles. |
| Comprobación de IDs completos | Detener una evaluación si faltan embeddings/metadatos, en lugar de reducir la muestra silenciosamente. |
| Separación de vacío y casi vacío | Diferenciar área exactamente cero de área menor que `0,0001`. |
| Corrección del lift de atención | Normalizar sobre la región válida y medir por separado la masa sobre padding. No se regeneraron las estadísticas históricas E1. |
| Compatibilidad Windows | Retirar una importación de `resource` que no se usaba. |
| Generador de prompts condicionado a la imagen | Añadir una alternativa a los tokens estáticos, manteniendo SAM congelado. |
| Selección opcional de época 0 | Conservar la inicialización si el entrenamiento no mejora dev; se aplica a ambos métodos de la nueva fase. |
| Manifiestos, checkpoints y métricas por imagen | Registrar configuración, versiones, IDs, hashes, curvas y resultados del checkpoint seleccionado. |
| Protección de carpetas y escritura JSON atómica | Evitar sobrescribir corridas y reducir el riesgo de leer archivos parcialmente escritos. |
| Pipeline secuencial y monitor de estado | Automatizar etapas, conservar logs y detenerse ante errores. |

Los resultados históricos E1 se conservaron. La nueva fase se guarda en `results/phase_b/`.

## 4. Datos, pesos y entorno preparados

Se descargaron el dataset y el checkpoint desde los servidores oficiales. Se comprobó que las 1.000 parejas imagen/máscara tienen dimensiones consistentes con los metadatos oficiales. La fracción de área GT de los 120 IDs de test coincide con los CSV históricos; esta comprobación no equivale a reproducir sus predicciones ni a probar identidad byte a byte con el espejo anterior.

El MD5 del checkpoint coincide con el oficial: `01ec64d29a2fca3f0661936605ae66f8`. Los hashes y las versiones quedan registrados en los manifiestos.

Se preparó `.venv` con acceso a paquetes existentes del sistema e instalación de SAM y scikit-learn. La ejecución usa Python 3.13.5, Torch 2.7.1 CPU y el código oficial de SAM fijado al commit `dca509fe793f601edb92606367a655c15ac00fdf`. Detalles en [environment_phase_b.json](environment_phase_b.json).

**Se completaron 400 embeddings de train400 y 100 de dev, con encoder y almacenamiento FP32.** Se omitieron los bloques intermedios del encoder para ajustarse al espacio disponible; no son necesarios para esta comparación. El entrenamiento y la inferencia de esta fase usan CPU, con cuatro hilos.

## 5. Pruebas técnicas originales: ocho aprobadas el 12 de octubre

El 12 de octubre se ejecutó la suite entonces disponible. El 16 de octubre la suite tiene una prueba adicional de rutas, cuyo error de permisos y comprobación separada se documentan en la revisión actual. Comando:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

| Prueba | Qué verifica |
|---|---|
| Decoder por lotes frente al oficial | Dos imágenes independientes, con cero y cuatro tokens, producen salidas equivalentes dentro de la tolerancia numérica. |
| Gradientes a través del decoder oficial congelado | La pérdida llega al generador condicionado mientras SAM permanece sin gradientes. |
| Dependencia de imagen del generador | Imágenes distintas generan prompts distintos; una actualización cambia parámetros entrenables y conserva los de SAM. |
| Pérdida y padding | Cambiar logits fuera de la región válida no cambia la pérdida. |
| Atención uniforme y lift | El lift es 1 con atención uniforme, incluso cuando existe padding. |
| Incompatibilidad de caché | Un cambio de almacenamiento o bloques requeridos ausentes impide reutilizar esa entrada. |
| Coordenadas perturbadas | La perturbación cero conserva las coordenadas y los desplazamientos se recortan a los límites de la imagen. |
| Alineación de métricas por ID | Ordenar distinto las filas no cambia la comparación; IDs incompletos o duplicados se rechazan. |

Son pruebas de implementación con entradas sintéticas; las que usan el decoder oficial instancian su arquitectura con pesos aleatorios. No prueban calidad de segmentación ni sustituyen experimentos con el checkpoint preentrenado. Esos experimentos reales se describen a continuación.

También se verificó sintaxis de los scripts y `git diff --check`.

## 6. Ensayos pequeños de ejecución

Antes del experimento completo se ejecutó una ablación de dos imágenes dev: 48 evaluaciones y 12 condiciones. También se entrenaron tres ensayos con cuatro imágenes train, dos dev y dos épocas:

| Ensayo | Resultado observado en dev |
|---|---|
| Tokens estáticos, LR `0,001` | Checkpoint entrenado seleccionado con Dice cercano a `0,250`. |
| Tokens condicionados, LR `0,001` | Caída desde aproximadamente `0,253` inicial a menos de `0,001`. |
| Tokens condicionados, LR `0,00001`, época 0 elegible | Las épocas entrenadas quedaron en `0,243–0,245`; se conservó la inicialización de `0,253`. |

Los tres checkpoints se recargaron y reprodujeron su Dice seleccionado. Se conservaron los ensayos negativos en [smoke_training](../results/phase_b/smoke_training/). Estos ensayos motivaron el barrido de LR; no permiten elegir un método con solo dos imágenes dev.

## 7. Experimento espacial completo: terminado

Se evaluaron las **100 imágenes dev, con 2.400 evaluaciones y 12 condiciones**. Se mantuvieron encoder, decoder y caché. La salida fue el token 0, con una consulta por componente GT elegible y unión de predicciones.

Los puntos se desplazaron en cuatro direcciones por 0,5/1/2/4 radios interiores de la lesión. El radio se obtiene de la transformada de distancia de GT. Las cajas se escalaron alrededor de su centro; multiplicar la escala afecta ancho y alto. Todos los prompts se recortaron a los límites de imagen.

| Condición | Dice medio dev | IC 95 % |
|---|---:|---:|
| Sin prompt | 0,0089 | [0,0007; 0,0227] |
| Caja de imagen completa | 0,0333 | [0,0183; 0,0508] |
| Punto interior GT | 0,6831 | [0,6155; 0,7463] |
| Punto desplazado dos radios | 0,1529 | [0,1309; 0,1756] |
| Caja GT precisa | 0,9352 | [0,9279; 0,9420] |
| Caja ampliada 1,25× | 0,8971 | [0,8757; 0,9152] |
| Caja ampliada 2× | 0,3216 | [0,2559; 0,3917] |

Las cuatro direcciones se promedian dentro de cada imagen antes del bootstrap de 10.000 remuestreos. No se cuentan las direcciones como imágenes independientes. Sin prompt, el 46 % de las máscaras fueron estrictamente vacías en dev.

La caja genérica aporta poco frente a la caja precisa. Desplazar el punto o ampliar mucho la caja degrada el solapamiento: una mitigación debe controlar ubicación y extensión. La pequeña mejora aparente con medio radio no queda establecida, porque el intervalo de la diferencia frente al punto interior cruza cero.

Las doce condiciones, curvas y diferencias pareadas están en [spatial_ablation.md](../results/phase_b/full_dev_20261012/spatial_ablation.md). El tiempo guardado incluye generación de prompts, decoder, postproceso y métricas; no es latencia aislada del decoder.

## 8. Generación frente a selección de máscaras: control terminado

Se evaluaron las mismas 100 imágenes y se verificó que las 200 predicciones únicas sin prompt/con punto reproducen los scores de la ablación a tolerancia `1e-8`.

| Condición y forma de elegir | Dice dev |
|---|---:|
| Sin prompt, salida única | 0,0089 |
| Sin prompt, multimáscara + IoU predicho | 0,1504 |
| Sin prompt, mejor de cuatro según GT | 0,1824 |
| Punto GT, salida única | 0,6831 |
| Punto GT, multimáscara + IoU predicho | 0,5401 |
| Punto GT, mejor de tres según GT | 0,8600 |

**Sin prompt, elegir mejor entre esas cuatro máscaras no basta para recuperar la referencia de caja GT.** Con un punto hay candidatos mejores que los que elige la cabeza IoU: la selección deja una brecha importante.

El mejor candidato según GT es un oráculo y no puede utilizarse sin conocer la respuesta. Este resultado tampoco se extiende automáticamente a AMG, que consulta muchos prompts espaciales. El control se añadió después de inspeccionar la ablación y es exploratorio. Un intento con un hilo se detuvo por una discrepancia de reproducción; con los mismos cuatro hilos de la ablación se reprodujeron las 200 salidas.

Detalles e intervalos: [output_selection/report.md](../results/phase_b/full_dev_20261012/output_selection/report.md).

## 9. Dónde falla: subgrupos y casos revisados

Se describieron subgrupos por área GT (<5 %, 5–15 % y ≥15 %), múltiples componentes y contacto con el borde. El fallo sin prompt aparece en los tres grupos de área; la caja GT mantiene Dice alrededor de 0,93–0,94. Con caja ampliada 2×, el grupo de lesiones grandes cae a aproximadamente 0,105.

Estos grupos se estudiaron después de ver los resultados. Tamaño, posición y contexto pueden estar relacionados; las cinco imágenes con múltiples componentes ofrecen evidencia limitada. Una componente conexa no equivale necesariamente a una lesión clínica diferente.

Se seleccionaron tres casos extremos por su diferencia caja−punto y se reprodujeron sus nueve predicciones. En esos ejemplos, el punto interior cae sobre una zona brillante y la salida única segmenta una pequeña región local; la caja recupera mucho más de la lesión. Esta observación ilustra ambigüedad del clic, pero no estima la frecuencia del patrón ni demuestra que los reflejos causen todos los fallos.

- [Tabla de subgrupos](../results/phase_b/full_dev_20261012/spatial_subgroups/report.md).
- [Predicciones, GT y prompts superpuestos](../results/phase_b/full_dev_20261012/spatial_cases/predictions.png).

## 10. Mitigación implementada y entrenamiento

Se implementaron cuatro prompts condicionados al embedding de la imagen: agregación espacial a 16×16, atención de consultas aprendidas a características y posiciones SAM, y actualización residual de los prompts. SAM permanece congelado. El generador tiene **331.008 parámetros entrenables**, frente a **1.024** de los cuatro tokens estáticos.

Se fijó la comparación en train400/dev, con inicialización `meanpoint`, pérdida BCE+Dice sobre región válida, lote 8, seis épocas y semillas 0/1/2. `meanpoint` usa posiciones GT de train para construir la inicialización; incluso la referencia sin entrenamiento contiene información de train. La época 0 puede ser seleccionada para ambos métodos si ninguna actualización mejora dev.

El barrido condicionado ya terminó, con semilla 0 y tres épocas por LR:

| LR | Mejor Dice dev | Época seleccionada |
|---|---:|---:|
| 0,00001 | 0,3607 | 1 |
| 0,00003 | 0,3602 | 3 |
| **0,0001** | **0,3987** | **3** |

Se eligió LR `0,0001` por el mayor Dice dev. En esa corrida, la inicialización tenía Dice `0,3505`. Es un resultado de búsqueda, no la comparación final con tres semillas. La primera corrida estática de seis épocas obtuvo `0,3513` frente a `0,3505` inicial: la ganancia en esa semilla fue pequeña.

El pipeline del 12 de octubre se programó para continuar con las corridas estáticas y condicionadas, tres probes lineales y sus variantes caja/caja+punto, todo sobre la misma caché FP32. Se interrumpió antes de completar ese recorrido. El resumen está preparado para ejecutar el control de contexto: conservar imagen objetivo, decoder y pesos, pero permutar la imagen que recibe el generador. Este control se fijó antes de los entrenamientos completos; permite evaluar sensibilidad al contexto con los mismos pesos. Aún debe ejecutarse antes de interpretarlo.

Configuración y elección de LR: [plan.json](../results/phase_b/full_dev_20261012/plan.json) y [selection.json](../results/phase_b/full_dev_20261012/selection.json). El estado actualizado y las corridas completas al corte aparecen al final de este documento.

## 11. Qué podemos concluir y qué falta

Los experimentos terminados apoyan que SAM necesita información espacial útil para esta tarea y que la elección de candidatos también limita el desempeño con un punto. El mismo encoder/decoder funciona mucho mejor con caja precisa; eso no demuestra que todo el fallo proceda exclusivamente del decoder ni que el encoder sea perfecto.

Falta terminar la comparación de tres semillas, repetir el probe y sus auto-prompts sobre FP32, ejecutar el control de contexto y evaluar los checkpoints con configuración congelada. La arquitectura condicionada tiene más parámetros y no utiliza una máscara explícita de padding en su atención: son limitaciones para atribuir una mejora exclusivamente al condicionamiento.

Los nuevos experimentos usan dev para elegir LR, época y umbral del probe. **Son resultados de desarrollo, no estimaciones independientes de generalización.** No se hizo nueva evaluación de predicciones en test durante esta fase; sus resultados históricos E1 sí se habían inspeccionado al diseñar el trabajo. Una evaluación externa sigue siendo necesaria. Tampoco se completó una comparación FP16/FP32 ni un estudio de tiempos de anotadores.

## 12. Cómo consultar el estado y los artefactos

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe scripts/phase_b_status.py --experiment results/phase_b/full_dev_20261012
```

No hace falta iniciar otra copia mientras el pipeline actual siga activo. `status.json` registra etapas y errores; `logs/` contiene sus registros. Cada corrida terminada conserva `summary.json`, checkpoint, curvas y métricas por imagen. Cuando termine la etapa de resumen se generarán `report.md`, `summary.json`, `table.csv` y la figura comparativa en la carpeta del experimento.

Las decisiones y sus cambios quedan en [design_log.md](design_log.md), y el protocolo detallado en [fase_b.md](fase_b.md).

## 13. Estado histórico observado el 12 de octubre

<!-- ESTADO_ACTUAL_INICIO -->
Corte: **2026-10-12 18:32:53, hora de Lima (UTC-5)**.

- Estado del pipeline: `running`.
- Etapa actual: `static_k4_s2`.
- Embeddings completados: **400/400 train400 y 100/100 dev**.
- Barrido de tres LR: terminado; elegido **0,0001**.
- Corridas finales terminadas: **2/6**.
- Coincidencia de hashes del entrenamiento con el plan: verificada.

| Modelo | Semilla | Dice inicial dev | Mejor Dice dev | Epoca elegida |
|---|---:|---:|---:|---:|
| static | 0 | 0.3505 | 0.3513 | 6 |
| static | 1 | 0.3486 | 0.3625 | 2 |

Probe semilla 0: `pendiente`.

Probe semilla 1: `pendiente`.

Probe semilla 2: `pendiente`.

Resumen final y control de contexto: `pendiente`.

Esta tabla incluye solamente corridas con resumen final guardado; no publica scores de epocas de corridas incompletas. El monitor puede mostrar una etapa posterior cuando se lea este documento.
<!-- ESTADO_ACTUAL_FIN -->
