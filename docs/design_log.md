# Cuaderno de decisiones (design log)

Proyecto: **El costo del prompt en SAM** — INF659 Deep Learning, 2026-2.
Convención: cada entrada tiene fecha y hora (America/Lima), la decisión, la justificación, las alternativas descartadas y si hubo **asistencia de IA**. Las predicciones (P-…) se escriben **antes** de correr el experimento correspondiente y no se editan después; el contraste se agrega en una entrada posterior.

> **Uso de IA (declaración general).** Para el E1 se usó Claude (Anthropic, modelo Claude Opus 5.5) como asistente de código y de análisis: buscó y verificó datos y referencias, escribió el código del repositorio, ejecutó los experimentos en un entorno solo-CPU y redactó un primer borrador de este cuaderno y del documento E1. Cada entrada lo marca con **[IA]**. Todas las decisiones deben ser revisadas, aceptadas o corregidas por el equipo, que es responsable de defenderlas en la sustentación oral.

---

## 2026-10-11 19:30 · D1 · Modelo base: SAM v1 con encoder ViT-B **[IA]**
- **Decisión:** usar Segment Anything (Kirillov et al., ICCV 2023), variante ViT-B (93,7 M parámetros), con el código oficial `facebookresearch/segment-anything` (commit `dca509f`).
- **Por qué:** (i) el mask decoder y el prompt encoder son idénticos en ViT-B/L/H, de modo que el análisis del núcleo que nos interesa (decoder sin prompts) se transfiere; (ii) ViT-B es la única variante que permite iterar en CPU; (iii) es la variante más estudiada en la literatura médica (MedSAM, SAMed, SAM-Med2D parten de ViT-B), lo que facilita posicionarnos.
- **Alternativas descartadas:** ViT-H (636 M) inviable sin GPU; SAM 2 (Hiera) se deja como extensión de la Fase B (¿cambia el costo del prompt con otro encoder?); SAM 3 cambia la tarea (prompts de concepto/texto) y se discute en related work.

## 2026-10-11 19:35 · D2 · Procedencia de los pesos **[IA]**
- Los pesos se descargaron desde un espejo en GitHub (`ultralytics/assets`, `sam_b.pt`) porque el entorno de trabajo solo permite GitHub y PyPI. **Verificación:** MD5 = `01ec64d29a2fca3f0661936605ae66f8` y tamaño 375 042 383 bytes; el sufijo del nombre oficial `sam_vit_b_01ec64.pth` corresponde a los primeros 6 hex de ese MD5 ⇒ archivo idéntico byte a byte al oficial de Meta (licencia Apache-2.0).

## 2026-10-11 19:45 · D3 · Dataset de reto: Kvasir-SEG **[IA]**
- **Decisión:** Kvasir-SEG (Jha et al., MMM 2020): 1000 imágenes de colonoscopía con máscara de pólipo, resoluciones 332–1920 px.
- **Por qué:** (i) dominio muy distinto a SA-1B (reflejos especulares, textura de mucosa, viñeteado negro, marcador verde del endoscopio); (ii) una lesión principal por imagen en ~95 % de los casos, lo que define sin ambigüedad «el objeto que el prompt debería señalar»; (iii) es el benchmark más usado para evaluar SAM en pólipos, lo que permite comparar con trabajos previos.
- **Alternativas:** ISIC (piel), BUSI (ultrasonido) y datos satelitales quedan como candidatos para la Fase C (segundo desplazamiento de dominio).
- **Licencia:** uso restringido a investigación y educación; se cita el artículo original.

## 2026-10-11 19:50 · D4 · Obtención y verificación de los datos **[IA]**
- El servidor oficial (datasets.simula.no) está bloqueado en el entorno de trabajo. Se usó un espejo público en GitHub (`TimKong21/Polyp-Segmentation-Tool`, commit `f7734b0`) que contiene las 1000 parejas con nombres cambiados (`colonoscopy_XXXX.jpg`).
- **Hallazgo:** el `renamed_bboxes.json` del espejo es **inconsistente con sus propias imágenes** (solo 65/1000 tamaños coinciden): las imágenes se renombraron en orden alfabético de los ids oficiales y el json en el orden de inserción del json oficial. Se descartó ese json.
- **Restauración:** asignación por orden alfabético verificada contra `kavsir_bboxes.json` oficial: **1000/1000 tamaños exactos**; IoU entre la caja de la máscara (sin motas JPEG) y la caja oficial: mediana 0,983, percentil 1 = 0,898.
- **Acción para el equipo:** re-ejecutar `scripts/00_prepare_kvasir.py --official_zip Kvasir-SEG.zip` con la descarga oficial; los ids y particiones resultantes deben ser idénticos.

## 2026-10-11 19:50 · D5 · Particiones **[IA]**
- **test** = `val.txt` oficial de MediaEval 2020 publicado por los autores del dataset (120 imágenes). **train/dev** = `train.txt` oficial (880) partido 780/100 con semilla 0. El test no se usa para ninguna decisión (umbrales, épocas, hiperparámetros se eligen en dev).
- **Alternativa:** la partición de PraNet (900/100) es más común en la literatura de pólipos, pero no encontramos su lista de ids oficial en una fuente verificable; se adoptará si se consigue, para comparar con PraNet/Polyp-PVT.

## 2026-10-11 19:50 · D6 · Preprocesamiento de la GT y prompts oráculo **[IA]**
- Máscaras oficiales en JPEG ⇒ binarizar con umbral 127. Para generar prompts se usan las componentes conexas con área ≥ max(1 % de la mayor, 64 px) (las motas de 1 px son artefactos de compresión: 207 componentes secundarias, mediana 1 px). Las métricas usan la GT binarizada completa (sin limpiar).
- Prompts por instancia: un prompt por componente y unión de las máscaras predichas.

## 2026-10-11 19:50 · D7 · Protocolo de evaluación **[IA]**
- Métricas por imagen a resolución original: Dice e IoU. Agregado: media con IC 95 % por bootstrap (10 000 remuestreos por imagen). Comparaciones pareadas: Wilcoxon de rangos con signo. Métodos entrenados: ≥ 3 semillas, se reporta media ± desviación entre semillas.
- «Costo del prompt» se mide de tres formas: (a) brecha de Dice entre no dar prompt y dar una caja; (b) número de clics simulados para llegar a IoU 0,85/0,90 (NoC, protocolo estándar de segmentación interactiva); (c) fracción de la brecha recuperada por un método sin humano: R = (Dice_método − Dice_sin prompt) / (Dice_caja − Dice_sin prompt).

## 2026-10-11 19:50 · D8 · Cómputo **[IA]**
- Entorno del E1: 2 vCPU Intel Xeon (con AMX-BF16), 7 GB RAM, sin GPU; PyTorch 2.14.1. Como el encoder se mantiene congelado en todos los experimentos del E1, los embeddings (256×64×64) se calculan una sola vez y se guardan en fp16; el decoder se ejecuta sobre la caché.

---

## Predicciones registradas ANTES de correr (2026-10-11 19:55) **[IA, a validar por el equipo]**

### Fase A — teardown
- **P-A1 (FLOPs por bloque):** los 4 bloques de atención global (2, 5, 8, 11) costarán ~1,7× los bloques con ventana (atención sobre 4096 tokens vs ventanas de 14×14), y el encoder concentrará >99 % de los FLOPs de una inferencia con un prompt. (Estimación a mano: ≈ 32 GMAC por bloque con ventana, ≈ 55 GMAC por bloque global.)
- **P-A2 (latencia):** en esta CPU, encoder fp32 ≈ 6–12 s/imagen; bf16 2–4× más rápido; decoder ≈ 20–60 ms por prompt. Consecuencia esperada: el costo del prompt es humano, no computacional.
- **P-A3 (campo receptivo):** los bloques 0–1 tienen campo receptivo **limitado a la ventana fija de 14×14 tokens (224 px)** que contiene al token — con borde duro alineado a la rejilla, no centrado en el token, porque SAM no desplaza las ventanas (a diferencia de Swin). Desde el bloque 2 (primer global) el campo receptivo teórico es toda la imagen, pero el efectivo seguirá concentrado en la ventana propia (>50 % de la masa del gradiente en el bloque 2) y se irá ensanchando con la profundidad.
- **P-A4 (características):** en las primeras capas las componentes principales mostrarán estructura posicional y de color; en el embedding final el pólipo aparecerá como región diferenciada en las 3 primeras componentes en la mayoría de imágenes.

### Línea base — costo del prompt (test, n = 120)
- **P-B1:** sin prompt (decoder con 0 tokens de prompt): Dice medio 0,15–0,30; las máscaras cubrirán regiones grandes (mucosa, fondo o el marco negro), no el pólipo.
- **P-B2:** AMG (modo «segment everything»): la mejor propuesta según oráculo tendrá Dice 0,75–0,85 (el decoder *sí* propone el pólipo), pero elegir la propuesta con mayor IoU predicho dará Dice 0,20–0,40 ⇒ el fallo dominante será de **selección**, no de propuesta.
- **P-B3:** un clic oráculo en el interior: Dice 0,60–0,75 con la máscara que elige la cabeza IoU; 0,75–0,85 con la mejor de las 3 (oráculo). Caja oráculo: Dice 0,85–0,90.
- **P-B4:** NoC@IoU 0,85 ≈ 4–7 clics de media (tope 20).
- **P-B5:** la cabeza IoU estará mal calibrada fuera de dominio (sobreestima) y su correlación con el IoU real será débil (Spearman < 0,4) sin prompt.
- **P-B6:** un *probe* lineal sobre el embedding congelado alcanzará Dice 0,60–0,75 ⇒ la información del pólipo está en el embedding; la caída sin prompt se origina en la consulta (decoder/prompt), no en el encoder.

### Intervención preliminar — tokens de prompt aprendidos (SAM congelado)
- **P-C1:** k ∈ {1, 4} tokens aprendidos (256·k parámetros) alcanzarán Dice 0,70–0,80 y recuperarán 60–80 % de la brecha sin prompt → caja.
- **P-C2:** k = 4 superará a k = 1 por ≤ 0,02 de Dice; desviación entre semillas ≤ 0,015.
- **P-C3:** superarán al probe lineal por ≥ 0,05 de Dice (el decoder aporta bordes y contexto).
- **P-C4:** los errores se concentrarán en imágenes con varios pólipos, pólipos pequeños y reflejos especulares, porque un token estático aporta el «qué» (clase) pero no el «dónde» (ubicación por imagen).

---

## 2026-10-11 20:20 · D9 · Mediciones del teardown ya realizadas (antes de las predicciones de ERF) **[IA]**
- Parámetros: encoder 89,67 M (95,7 %), decoder 4,06 M (4,3 %), prompt encoder 6 220. FLOPs (convención 2 FLOPs por MAC, solo matmul/conv): encoder 972 GFLOPs; decoder 3,6 GFLOPs por prompt. Bloques globales/ventana = 1,70 (P-A1 se cumple). Latencia (2 hilos): encoder 10,6 s fp32 y 5,1 s bf16; decoder 54 ms por prompt (P-A2 se cumple).
- Hallazgo no previsto: los bloques con ventana proyectan qkv/proj sobre 4 900 tokens (64→70 de padding para encajar ventanas de 14), 19,6 % más que los 4 096 reales.
- Hallazgo no previsto: el decoder en lote es **más lento por prompt** (100 ms con 64 prompts) que en lotes de 4 (64 ms), porque `repeat_interleave` replica el embedding de la imagen por cada prompt.

## 2026-10-11 20:20 · D10 · AMG con rejilla 16×16 **[IA]**
- La rejilla oficial 32×32 cuesta ~110 s por imagen en esta CPU. Se usa 16×16 (256 puntos, 64 px de separación en el marco 1024) con los mismos umbrales. Riesgo: perder pólipos muy pequeños; mitigación: repetir con 32×32 en GPU en la Fase B.

## 2026-10-11 20:20 · D11 · Precisión de los embeddings y subconjunto de entrenamiento **[IA]**
- test y dev en fp32; entrenamiento en bf16 (coseno 0,99996; Dice de acuerdo con caja 0,999 en 6 imágenes). Sin prompt, la salida es inestable entre precisiones (logits cercanos a 0): por eso todo lo que se evalúa usa fp32.
- Por costo, el piloto entrena con 400 de las 780 imágenes de train (subconjunto fijo, semilla 1234, `data/splits/train400.txt`). Plan: train completo en GPU.

## 2026-10-11 20:20 · D12 · Configuración del piloto de tokens aprendidos (fijada ANTES de entrenar, sin barrido) **[IA]**
- SAM congelado; k ∈ {1, 4} tokens en la posición de los prompts dispersos; denso = `no_mask_embed` original; salida = token 0 (modo salida única). Inicialización N(0, σ) con σ = desviación de un prompt de punto real (0,47).
- Pérdida BCE + Dice suave en el marco 256×256 (solo región válida). AdamW, lr 5·10⁻³ con OneCycle, lote 8, 6 épocas; época elegida por Dice en dev; 3 semillas (0, 1, 2). El lr se fijó a priori; un barrido queda para la Fase B.
- Probe lineal: regresión logística sobre el embedding final (1 500 tokens por imagen de train400), umbral elegido en dev. Auto-prompt: caja y punto derivados de la máscara del probe.

---

## 2026-10-11 20:40 · Contraste predicción vs. resultado — Fase A **[IA]**
| Predicción | Resultado | Veredicto |
| --- | --- | --- |
| P-A1: bloques globales ≈ 1,7× los de ventana; encoder > 99 % de FLOPs | 110,3 vs 64,9 GFLOPs (1,70×); encoder = 99,6 % con un prompt | Se cumple |
| P-A2: encoder 6–12 s fp32, bf16 2–4× más rápido, decoder 20–60 ms | 10,6 s; bf16 5,1 s (2,1×); decoder 52–57 ms | Se cumple |
| P-A3: bloques 0–1 confinados a la ventana fija; >50 % de la masa en la ventana propia en el bloque 2; ensanchamiento con la profundidad | 100 % en la ventana en bloques 0–1 (tokens vecinos con campos disjuntos); 66 % en el bloque 2; 48 %, 36 %, 29 % después | Se cumple |
| P-A4: posición/color en capas tempranas; pólipo separado en el embedding final | El recuadro negro, el viñeteado y los reflejos dominan las componentes en todas las capas; la separación pólipo/fondo crece con la profundidad (1,67 → 2,34 desviaciones) | Se cumple parcialmente: el pólipo NO domina las 3 primeras componentes; lo dominan artefactos |
- **Sorpresa útil:** en los bloques 8 y 11 una sola dirección explica ~59 % de la varianza; y el gradiente profundo dibuja el borde circular del endoscopio. Pregunta abierta para la Fase B: ¿esa dirección dominante perjudica la consulta sin prompt?

## 2026-10-11 20:40 · Contraste predicción vs. resultado — línea base (test, n = 120) **[IA]**
| Predicción | Resultado | Veredicto |
| --- | --- | --- |
| P-B1: sin prompt Dice 0,15–0,30 | Token 0: 0,012 (máscara vacía en 71 % de las imágenes); multimáscara + cabeza IoU: 0,126; mejor de 4 (oráculo): 0,157 | Peor de lo previsto: sin prompt el decoder casi no produce el pólipo |
| P-B3: 1 clic 0,60–0,75 (cabeza IoU) y 0,75–0,85 (oráculo); caja 0,85–0,90 | 0,571; 0,840; caja 0,886 | Clic con cabeza IoU por debajo; oráculo y caja dentro del rango |
| P-B4: NoC@0,85 ≈ 4–7 clics | 3,48 clics (NoC@0,90 = 5,32; 2,5 % no llega a 0,90 en 20) | Mejor de lo previsto |
| P-B5: cabeza IoU sobreconfiada, Spearman < 0,4 sin prompt | Sin prompt: IoU predicho medio 0,46 vs real 0,04, Spearman 0,44; con 1 clic: 0,86 vs 0,44, Spearman 0,23; con caja: 0,95 vs 0,83, Spearman 0,68 | Sobreconfianza confirmada; la peor ordenación ocurre con 1 clic |
- **Hallazgo clave no previsto:** incluso CON un clic humano, la cabeza IoU elige mal entre las 3 máscaras y se pierden 0,27 de Dice (0,571 vs 0,840; Wilcoxon p < 10⁻⁶). El fallo de selección no es exclusivo del modo sin prompt.
- **Mecanismo (atención del decoder):** sin prompt, el token de máscara pone en el pólipo una mediana de 4,9 % de su atención final (lift 0,42 respecto del área: lo evita); con caja, 59 % (lift 5,4).

## 2026-10-11 21:20 · Contraste predicción vs. resultado — AMG (P-B2) **[IA]**
| Predicción | Resultado | Veredicto |
| --- | --- | --- |
| P-B2: mejor propuesta (oráculo) Dice 0,75–0,85; elegir por IoU predicho 0,20–0,40; fallo dominante = selección | Mejor propuesta cruda 0,895 (85 % de imágenes con alguna propuesta ≥ 0,8); tras filtros 0,499; tras NMS 0,493; elegir por IoU predicho 0,058 | Se cumple y se refina: la propuesta es aún mejor de lo previsto, la selección aún peor, y aparece una segunda causa no prevista |
- **Segunda causa no prevista:** el filtro de estabilidad (≥ 0,95) quita 0,29 de Dice; la mejor propuesta tiene estabilidad mediana 0,79. Interpretación: los bordes suaves de la mucosa dejan logits cercanos a 0 y la máscara cambia mucho con ±1 de umbral.
- **Calibración:** en las 92 160 propuestas, Spearman(IoU predicho, IoU real con el pólipo) = 0,03. La cabeza IoU estima la calidad de la máscara respecto del objeto que señala el punto, no si ese objeto es la lesión: el *qué* no está en ningún componente de SAM.

## 2026-10-11 21:20 · D13 · Datos de terceros fuera del repositorio **[IA]**
- `kavsir_bboxes.json` es parte del dataset: no se versiona; el script lo lee del zip oficial. Las listas de ids de la partición sí se versionan (son públicas en el repositorio de los autores).
- Atributos para subgrupos (recuadro negro del marcador, marcador verde, nº de pólipos, área) con heurísticas simples en `scripts/08b_subgroups.py`; en test: 55/120 con recuadro negro, 13 con marcador verde visible, 7 con varios pólipos, 5 con pólipo < 2 % del área. Las heurísticas deben validarse a ojo antes de la Fase C.

## 2026-10-11 21:35 · Matiz sobre P-B3 (selección con un clic) **[IA]**
- Con 1 clic, la salida única (token 0, `multimask_output=False`) da Dice 0,696 [0,638–0,753], mejor que multimáscara + cabeza IoU (0,571; Wilcoxon p = 7·10⁻⁶) y peor que el oráculo de 3 (0,840). La práctica recomendada por SAM para un clic (multimáscara + cabeza IoU) es la peor de las tres aquí. Se reporta en la tabla de la sección 4.

## 2026-10-11 22:15 · Resultado negativo del piloto pre-registrado y corrección (D14) **[IA]**
- **Lo que pasó:** con la configuración fijada en D12 (tokens inicializados al azar, lr 5·10⁻³), k = 4, semilla 0, el Dice en dev quedó en 0,003 → 0,0001 durante 5 épocas mientras la pérdida bajaba de 1,67 a 1,33: el modelo colapsó a predecir máscaras vacías.
- **Diagnóstico:** desde una inicialización aleatoria la salida del token 0 es vacía (igual que sin prompt, 71 % de máscaras vacías); la BCE sobre un fondo mayoritario empuja todos los logits a negativo y el Dice suave casi no da gradiente cuando la máscara está vacía. Un lr 6× mayor (3·10⁻²) tampoco resolvió (dev 0,03 en una prueba corta).
- **Corrección (una sola desviación):** inicializar los k tokens con la media del embedding de un clic en el punto más interior de los pólipos de 200 imágenes de train, más ruido N(0; 0,05) para romper la simetría (`--init meanpoint`). Todo lo demás queda como en D12. En una prueba corta (120 imágenes, 3 épocas) el Dice en dev pasó de ~0 a 0,40.
- **Implicación para la pregunta:** el token aprendido necesita partir de algo con forma de prompt real; el espacio de prompts de SAM no es fácil de recorrer desde cero. Se reporta la corrida fallida en el documento.
- Las 6 corridas siguientes (k ∈ {4, 1} × semillas 0, 1, 2) usan `--init meanpoint`.

## 2026-10-11 23:10 · Contraste predicción vs. resultado — intervención (test, n = 120) **[IA]**
| Predicción | Resultado | Veredicto |
| --- | --- | --- |
| P-C1: k ∈ {1, 4} tokens con Dice 0,70–0,80 (60–80 % de la brecha) | k = 4 (init. clic medio): 0,318 ± 0,014 (35 %); k = 1: 0,279 ± 0,002 (30 %); init. aleatoria pre-registrada: 0,008 | Refutada |
| P-C2: k = 4 supera a k = 1 por ≤ 0,02; desviación ≤ 0,015 | +0,039 (IC 0,010–0,069; p = 6·10⁻⁴); desviaciones 0,014 y 0,002 | Parcial |
| P-C3: tokens ≥ probe lineal + 0,05 | Probe 0,614; probe → caja → SAM 0,622; tokens k = 4 −0,296 frente al probe (p < 10⁻¹⁰) | Refutada |
| P-C4: errores en varios pólipos, pequeños y con reflejos | Varios pólipos 0,11 (n = 7); < 5 % del área 0,18 vs 0,47 en > 15 %; recuadro negro 0,24 vs 0,39; reflejos no medidos | Parcial |
- **Lectura:** con SAM congelado, un prompt estático (el *qué*) no sustituye a uno condicionado a la imagen (el *dónde*): el probe → caja recupera 70 % de la brecha y los tokens 35 %. Los fallos de los tokens son sistemáticos (correlación por imagen entre semillas 0,84–0,94; 52 % de imágenes con Dice < 0,1). La atención del token de máscara sobre el pólipo sube de 4,9 % (sin prompt) a 23 % (tokens), lejos del 59 % con caja.
- **Dos etapas** (máscara de tokens → caja → SAM): +0,009 (p = 0,03); la caja hereda el error de ubicación.
- **Siguiente paso (Fase B):** prompts condicionados a la imagen (B2) y cabeza de selección reentrenada (B3), con barrido de lr y 5 semillas en GPU.

## 2026-10-12 14:09 · D15 · Revisión y preparación de Fase B **[IA: Codex]**
- **Autorización del equipo:** continuar con correcciones de reproducibilidad, ablación espacial y generador de prompts condicionado a la imagen.
- **Correcciones:** comandos meanpoint en README/Colab/generador de notebooks; fases omitidas del cuaderno; separación de vacío estricto y casi vacío; lift de atención condicionado a región válida; importación Windows; validación de splits completos y protección de corridas existentes.
- **Precisión:** E1 usa encoder FP32 en dev/test pero almacena embeddings FP16. La fase nueva puede conservar FP32; se registra precisión, almacenamiento, hash de imagen/pesos/código de encoder, dispositivo y versión torch. No mezclar ambas rutas en una comparación.
- **Datos:** se descargaron pesos y dataset desde servidores oficiales. MD5 de pesos coincide. Las 1.000 parejas cumplen dimensiones oficiales y la fracción de área GT de los 120 IDs test coincide exactamente con los CSV E1. Esto verifica ese atributo, no identidad byte a byte del espejo anterior.
- **Intervención:** cuatro consultas con atención a embeddings agregados a 16×16 y posiciones SAM, actualización residual de prompts, decoder congelado. 331.008 parámetros entrenables frente a 1.024 de tokens estáticos. Solo se implementó la ruta CPU de entrenamiento.
- **Predicción exploratoria:** el condicionamiento por imagen puede mejorar localización frente a tokens estáticos. No se establece como resultado; existen antecedentes de self-prompting y se necesitan controles de capacidad y ajuste en dev.
- **Verificación técnica:** siete pruebas aprobadas, incluidas equivalencia por lotes con decoder oficial y gradientes sin modificar SAM. Ablación de dos imágenes dev: 48 evaluaciones y 12 condiciones. Se ejecutaron tres entrenamientos pequeños con cuatro imágenes train, dos dev y dos épocas, sin evaluar test.
- **Resultado del ensayo:** estático LR 0,001 tiene dev alrededor de 0,25; condicionado LR 0,001 cae desde 0,253 inicial a menos de 0,001. Con LR 0,00001 conserva segmentaciones, pero no mejora la inicialización en esas dos imágenes. No atribuir esta caída únicamente a un mecanismo ni extrapolar a todo Kvasir.
- **Cambio posterior al ensayo:** añadir `--include_initial` para permitir selección de época 0 en dev, igual para ambos métodos en Fase B; conservarlo desactivado por defecto para reproducción del criterio E1. La receta de LR del condicionado debe ajustarse en dev completo. Guardar evaluación inicial, curvas, configuración, versiones, fuentes y checkpoints. Las corridas fallidas se conservan.
- **Pendiente:** completar caché de train400/dev, ajustar optimización, ejecutar tres semillas y comparar con probe/caja+punto sobre la misma precisión; luego evaluación externa. Protocolo y detalle en `docs/fase_b.md`. Los resultados históricos E1 no se sobrescribieron.

## 2026-10-12 16:44 · D16 · Experimento completo de desarrollo **[IA: Codex]**
- **Autorización:** el equipo pidió continuar con la experimentación. Se inicia train400/dev completos, sin evaluación de test.
- **Protocolo fijado antes de entrenar:** 400 imágenes train, 100 dev; k=4; inicialización meanpoint; BCE+Dice; lote 8; época 0 elegible; tres semillas 0/1/2 para estático y condicionado. Barrido condicionado con semilla 0, tres épocas y LR {0,00001; 0,00003; 0,0001}; elegir mayor Dice dev, empates a favor de LR menor. Comparación posterior de seis épocas; estático LR 0,005. También se repiten probe, caja y caja+punto con tres semillas.
- **Ablación espacial:** las 100 imágenes dev, con condiciones fijadas en D15. Promedios por imagen antes de bootstrap de direcciones o semillas. La elección de época, LR y umbral de probe usa dev: reportar los scores como desarrollo, no generalización independiente.
- **Recursos:** CPU sin CUDA, cuatro hilos y aproximadamente 3,8 GB libres al inicio. Embeddings finales FP32, sin guardar los bloques intermedios porque no son necesarios en esta comparación y excederían el espacio disponible. Los resultados históricos se conservan.
- **Trazabilidad:** ejecución secuencial, comandos y registros por etapa, plan con hashes del código, checkpoints y métricas dev de la época elegida. El estado y los resultados se escriben bajo `results/phase_b/full_dev_20261012/`; se detiene ante errores y no reutiliza corridas incompletas como resultados finales.
- **Estado de esta entrada:** la codificación de dev está en ejecución; todavía no hay resultados del experimento completo. Las predicciones y la lectura de los ensayos técnicos previos no se modifican.

## 2026-10-12 17:08 · D17 · Control de condicionamiento antes de entrenar **[IA: Codex]**
- **Control adicional de desarrollo:** después de completar los tres checkpoints condicionados, mantener imagen/decoder fijos y permutar la imagen que recibe el generador de prompts. Permutación sin puntos fijos, semilla 0, común a las tres semillas de entrenamiento; evaluación sobre los 100 IDs de dev.
- **Motivación:** el condicionado tiene más parámetros que el estático. Una diferencia de score entre ambos no aísla el aporte del contexto por imagen. Esta intervención evalúa sensibilidad al contexto con exactamente los mismos pesos; no reemplaza un baseline de igual capacidad ni constituye validación externa.
- **Momento:** se fija antes del entrenamiento completo de los modelos; la caché de train400 aún está en cálculo. La ablación espacial de dev sí se terminó y sus resultados se conservan en `results/phase_b/full_dev_20261012/spatial_ablation.md`.
- **Implementación:** `scripts/14_conditioning_diagnostic.py`; se verificará que cada checkpoint recargado reproduce su score dev antes de interpretar la permutación. No se consulta test.

## 2026-10-12 17:27 · D18 · Ablación completa y controles exploratorios **[IA: Codex]**
- **Resultado espacial (100 dev, FP32):** 2.400 evaluaciones, 12 condiciones. Dice sin prompt 0,0089; caja completa 0,0333; punto GT 0,6831; caja GT 0,9352; desplazamiento de dos radios 0,1529; caja ampliada 2× 0,3216. Informe `results/phase_b/full_dev_20261012/spatial_ablation.md`.
- **Subgrupos posteriores a los resultados:** áreas <5 %, 5–15 % y ≥15 %, múltiples componentes y contacto con borde. El fallo sin prompt aparece en todos los grupos de área. Son asociaciones descriptivas; no constituyen hipótesis confirmatorias. Se usan los mismos filtros de componentes que el oráculo. `scripts/15_spatial_subgroups.py`.
- **Casos ilustrativos:** tres extremos elegidos por diferencia caja−punto; nueve predicciones reproducidas a tolerancia 1e−8. Los puntos interiores sobre zonas brillantes producen máscaras locales pequeñas; las cajas recuperan la lesión. No se estima frecuencia ni causalidad de este patrón. `scripts/16_spatial_cases.py`.
- **Selección de salidas, control posterior:** las 200 salidas únicas sin prompt/con punto reproducen exactamente la ablación. Sin prompt: IoU-pick 0,1504; mejor de cuatro según GT 0,1824. Con punto: IoU-pick 0,5401; mejor de tres según GT 0,8600. Por tanto, la selección entre estas cuatro candidatas no resuelve por sí sola el fallo sin prompt. Los oráculos no son mitigaciones desplegables. `scripts/17_output_selection.py`.
- **Reproducción numérica:** un intento del control con un hilo se detuvo por discrepancia; con los mismos cuatro hilos de la ablación reprodujo las 200 salidas. No se mezclan los resultados del intento detenido.
- **Automatización:** el resumen final ejecutará el control de contexto D17 después de las corridas entrenadas y lo integrará junto con los controles terminados. La configuración y los hashes del entrenamiento fijados en D16 siguen intactos. La caché train400 y los entrenamientos completos aún están en ejecución; no hay resultados de esa comparación y no se evalúa test.
- **Alcance del costo:** no se midieron tiempos humanos de anotación; se caracteriza valor y precisión de información espacial. Las referencias de punto/caja GT usan anotaciones oráculo.

## 2026-10-16 18:31 · D19 · Carpeta por corrida y estado del experimento de desarrollo **[IA: Claude]**
- **Problema:** desde D15, `07_learned_prompts.py` se niega a escribir en una carpeta de corrida existente. Como `results/learned/<tag>` del E1 está versionado, el bucle de reproducción del README fallaba en su primera iteración en un clon limpio.
- **Decisión:** la variable `SAMCOST_RUN_DIR` redirige **todo el pipeline** a una carpeta propia por corrida (`runs/<fecha>`, ignorada por git), con resultados y figuras. README y Colab la definen antes del primer script. Sin la variable, el comportamiento no cambia: `results/` y `figures/` siguen siendo los artefactos del E1.
- **Por qué a nivel de pipeline y no una subcarpeta por ejecución de `07`:** `07b_twostage.py`, `05b_attention_stats.py`, `08_summarize.py` y `09_figures.py` leen `results/learned/<tag>` con glob. Si hubiera varias ejecuciones por tag, habría que elegir cuál usar, con riesgo de promediar semillas de corridas o versiones de código distintas. Con una carpeta por corrida completa, cada resumen se calcula solo con artefactos de esa corrida, y además no se sobrescriben los CSV y figuras que generan los demás scripts.
- **Alternativas descartadas:** un flag `--overwrite` en `07` (sí sobrescribe); sufijos automáticos por tag (los glob mezclarían corridas).
- **Sin cambios en los siete archivos con hash en `results/phase_b/full_dev_20261012/plan.json`:** modificarlos impediría reanudar ese experimento. Solo cambió `src/samcost/common.py`, que no está en el plan; con la variable sin definir, las rutas son idénticas a las anteriores.
- **Estado del experimento de desarrollo:** se versionó a mitad de ejecución. `static_k4_s1` quedó interrumpida y faltan 7 corridas y el resumen. Además, el plan solo se puede reanudar en la copia de trabajo original (hashes sensibles a fin de línea y rutas con `\`). Detalle y pasos en `docs/fase_b.md`, sección «Estado del experimento versionado». No se modificaron sus resultados.
- **Verificación:** se comprobó la resolución de rutas sin variable, con ruta absoluta y con ruta relativa (relativa a la raíz del repositorio). Se añadió una prueba unitaria. Las pruebas no se ejecutaron en esta máquina porque no tiene torch instalado.

## 2026-10-16 · D20 · Revisión de la copia original y cambios locales **[IA: Codex]**
- **Git:** el switch a `main` falló por cambios locales en dos JSON del experimento. El pull posterior actualizó `fix/pruebasExperimentos`; HEAD y `origin/main` coinciden en `dc88ba5`, pero `main` local sigue en `ab30c46`. No se cambió de rama, hizo stash/commit ni descartó archivos durante esta revisión.
- **Corrección de alcance de D19:** el commit era una fotografía parcial. En esta copia original sí terminaron las tres semillas estáticas: Dice dev 0,3513; 0,3625; 0,3762. Los CSV contienen los 100 IDs dev y coinciden con los resúmenes. Media 0,3633, desviación entre semillas 0,0125; IC por imagen [0,2913; 0,4375]. Frente a la inicialización, diferencia media +0,0143, IC pareado [−0,0383; 0,0702], exploratorio.
- **Interrupción observada:** `conditional_k4_s0` conserva tres de seis épocas y score intermedio 0,4766, sin resumen final. No se encontraron procesos Python activos. `running` es un estado persistido antiguo, no prueba de actividad. Faltan condicionado completo, probes, control de contexto y resumen. No se determinó la causa de la interrupción ni se reinició el experimento.
- **Reanudación documentada:** conservar las corridas estáticas completas y archivar únicamente la etapa que siga parcial. El trainer no guarda estados completos de optimizador/scheduler para continuar exactamente desde la época 4. Los siete hashes del plan coinciden en esta copia; no se modificó su código ni sus artefactos científicos.
- **Pruebas actuales:** ocho de nueve pruebas pasan; la prueba nueva de rutas termina con error de permisos de temporales de Windows (`WinError 5`). La redirección se comprobó por separado mediante subprocesos y carpetas del workspace: rutas por defecto, absoluta y relativa correctas. No se afirma que la suite completa esté aprobada.
- **Documentación:** [revisión de SAM y Git](revision_sam_20261016.md); se corrigieron `fase_b.md` y la nota de vigencia del resumen previo. La fecha de esta entrada corresponde al contexto del cliente (16 de octubre).

## 2026-10-16 · D21 · Lectura de cambios-analisis y ejecución retomada **[IA: Codex]**
- **Autorización:** el equipo compartió la captura con la rama de trabajos relacionados y pidió comenzar la siguiente etapa. Se descargaron referencias remotas sin cambiar de rama ni hacer merge.
- **Fuente:** se incorporó una copia exacta de `docs/related_work.md` de `origin/cambios-analisis`, commit `ca03c31`. La evaluación de afirmaciones queda separada en `docs/revision_related_work.md`; el archivo original no se reescribió. La entrada D20 de esa rama remota no se mezcló con la D20 local de revisión.
- **Contraste:** se revisaron secciones de método/tablas de Zhou, AutoSAM, Self-Prompting, HYDRA, MedSAM3 y Medical SAM3, además de otros resúmenes. Se identificó un error al describir Medical SAM3 como evaluación con cajas: su sección 4.2 declara texto sin prompts espaciales. Se matiza la equivalencia de AutoSAM con el generador local y la atribución causal del filtro de estabilidad a bordes difusos.
- **SAM:** preflight train400/dev correcto y hashes del plan coincidentes. Se archivaron la corrida parcial, checkpoint, log y estado anterior en `interrumpidas/conditional_k4_s0_20261016_e79f4dda/`, dentro del experimento. Se reinició esa etapa desde cero, conservando los estáticos completos. Sus primeras tres épocas reprodujeron exactamente pérdida, Dice y tasas de vacío de la ejecución anterior. Semilla condicionada 0 terminó seis épocas, con Dice dev seleccionado 0,50685; el resto del pipeline sigue pendiente al escribir esta entrada.
- **Baseline externo:** se implementó una variante U-Net independiente sobre RGB y sin preentrenamiento. Pilotos de 32 train/16 dev, tres y veinte épocas sin normalización, mantuvieron Dice dev 0; se conservan resultados y snapshots. Se fijó después un piloto de 20 épocas con GroupNorm, mismos IDs, semilla, LR y tamaño; es ajuste exploratorio de optimización, no benchmark final.
- **Verificación:** cuatro pruebas nuevas U-Net aprobadas: geometría/recorte, gradientes, aprendizaje de una región sintética y variante GroupNorm. Se añadió guardado del endpoint entrenado y diagnósticos de probabilidad, pérdida dev y vacío. La prueba antigua de temporales de Windows conserva la limitación de permisos documentada en D20.
- **Alcance:** sin nueva evaluación de test, cambios al plan SAM ni afirmaciones de superioridad frente a otras familias. Registro de comandos, pilotos y límites en `docs/experimentos_modelos.md`. No se han ejecutado todavía SAM 2.1, MedSAM ni Polyp-PVT.

## 2026-10-16 · D22 · Cierre de desarrollo y límites del contexto **[IA: Codex]**
- **Ejecución:** las 16 etapas de Fase B terminaron con código 0. Los siete hashes originales siguen coincidiendo. Se preservó la corrida parcial archivada y se conservaron las tres semillas estáticas completas.
- **Resultados (100 dev):** Dice medio estático 0,3633; condicionado 0,5013; probe 0,7489; probe → caja 0,7738; probe → caja+punto 0,8179. La diferencia condicionado−estático es +0,1380, IC pareado exploratorio [0,0895; 0,1898]. Los tres condicionados eligieron época 6, última del presupuesto; no se afirma convergencia.
- **Control D17:** contexto correcto 0,501337 frente a permutado 0,495295; diferencia 0,006042, IC [0,000840; 0,011245], p exploratorio 0,0191 sin ajuste por multiplicidad. La sensibilidad pequeña al emparejamiento del contexto impide atribuir toda la mejora sobre el estático al contexto por imagen. Hace falta un control de igual capacidad con contexto constante para profundizar esa explicación.
- **Trazabilidad:** seis CSV con 100 IDs únicos; los 300 Dice individuales condicionados se reprodujeron al recargar checkpoints para el diagnóstico. Las tres permutaciones conservan 100 contextos únicos sin puntos fijos. Informe, intervalos y gráficos en `results/phase_b/full_dev_20261012/`.
- **U-Net:** el piloto GroupNorm completó 20 épocas, con mejor Dice 0,5082 en época 12 y final 0,4227; 32 train/16 dev, una semilla. Sus 16 Dice seleccionados se reprodujeron al recargar. Los dos pilotos sin normalización mantuvieron Dice 0 y se preservaron. No comparar estos pilotos con SAM de 400/100 ni afirmar superioridad entre familias. Copias ligeras y gráficos en `results/model_comparison/`; pesos en `runs/`, ignorado por Git.
- **Pendientes:** entrenamiento completo U-Net con receta fijada y tres semillas; SAM 2.1 bajo los mismos IDs; costo computacional con hardware común y evaluación externa. No hubo nueva evaluación de test, commit, merge ni push. Los cuatro nuevos tests U-Net pasan; sigue documentada la limitación de permisos del test previo de temporales de Windows.

## 2026-10-16 · D23 · Alcance científico y borrador WAIMLAp **[IA: Codex]**
- **Solicitud:** analizar si la idea merece continuar, explicar experimentos, decidir cómo comparar modelos y preparar el trabajo para la plantilla de resumen extendido. El equipo pidió revisar todas las ramas también para localizar autoría.
- **Ramas:** fetch sin cambios de rama ni merge. Main remoto y fix remoto tienen árboles idénticos; cambios-analisis añade bibliografía, no experimentos de otras familias. Los textos versionados no contienen una lista formal de autores; nombres/correos de commits no se convierten en autoría sin confirmación.
- **Decisión:** conservar el costo informativo de prompts espaciales, diagnóstico de candidatas/selección y mitigación automática como núcleo. El mejor pipeline actual es probe→caja+punto, no el generador. No afirmar que se identificó un defecto intrínseco del decoder ni que se midió tiempo humano.
- **Comparaciones:** U-Net completo primero, SAM 2.1 congelado para transferencia del diagnóstico, Polyp-PVT para una ampliación con especialista. Control de contexto constante de igual capacidad si se quiere sostener el generador como aporte principal. Datos externos y cómputo común tienen prioridad sobre añadir variantes sin hipótesis.
- **Convocatoria:** plantilla local y página oficial de pósteres contrastadas: admite trabajo en curso, abstract inglés de dos páginas máximo y póster separado. Fecha publicada de envío: 16 de octubre de 2026. No se realizó envío.
- **Documentación:** `docs/decision_investigacion_waimlap.md` reúne inventario, límites y protocolo de cierre; `docs/waimlap_2026/` contiene un borrador en inglés y Word basado en la plantilla, conservando cabecera e imágenes. El abstract usa solo resultados dev de Fase B; no mezcla test histórico ni pilotos U-Net y marca extensiones pendientes. Sin nuevos entrenamientos ni evaluación de test en esta tarea.
- **Verificación del documento:** PDF exportado con Word local, dos páginas A4 confirmadas también con PDFium; páginas renderizadas e inspeccionadas. Cabecera, pie y medios del Word original preservados. Tabla contrastada con el resumen de Fase B, enlaces locales comprobados y `git diff --check` correcto. Autoría pendiente, sin póster preparado ni envío realizado.

## 2026-10-16 · D24 · Comparación ligera y controles espaciales **[IA: Codex]**
- **Solicitud:** mejorar experimentación y añadir al menos un par de modelos ligeros. Se implementaron FCN-Lite y DS-U-Net, además de una nueva U-Net compacta. Son variantes declaradas, no reproducciones exactas de artículos ni especialistas de estado del arte.
- **Plan previo al entrenamiento:** train400/dev100 completos, semillas 0/1/2, entrada RGB 128, ancho 8, GroupNorm, sin preentrenamiento, Adam LR 0,001, lote 8, hasta 30 épocas, mínimo 15 y paciencia 8; umbral 0 y selección dev incluyendo inicialización. Plan, hashes y datos se congelan antes de correr. El encoder SAM mantiene su receta anterior y entrada 1024: no es un contraste de arquitectura aislado. `docs/protocolo_comparacion_ligera.md` detalla alcance y comandos.
- **Reanudación:** checkpoint de época con pesos, Adam, RNG y orden de muestras. Un test reproduce exactamente la actualización después de restaurar estado. Cada checkpoint elegido debe reproducir los 100 Dice individuales. Los pesos completos quedan en `runs/`; exportación ligera al finalizar nueve corridas.
- **Pruebas:** cuatro tests nuevos de arquitecturas/reanudación/emparejamiento y dos del prior pasan. La prueba previa de rutas se corrigió para evitar el ACL privado 0700 de `tempfile` en Windows; carpeta ordinaria UUID dentro del workspace, validación del destino antes de limpiar. La suite de 17 existente tras incorporar los cuatro tests pasó; los dos tests del prior pasaron después por separado.
- **Datos:** no hay duplicados exactos train400/dev100 ni candidatos de alta similitud en el screening DCT fijado (63 bits, Hamming ≤4). Esto no prueba independencia por paciente/video. No se alteran particiones ni se evalúa test.
- **Control previo y resultado:** se fijó mediana de cajas/puntos normalizados de GT train400, una consulta por imagen dev sin GT para generar prompts. Dice punto fijo 0,36335; caja fija 0,29690; caja+punto fijo 0,35818. Probe→caja+punto supera a caja+punto fijos por 0,45974, IC pareado exploratorio [0,38087; 0,53697]. No equivale a control entrenado del generador de igual capacidad ni a ahorro humano medido.
- **Estado al escribir:** FCN-Lite terminó tres semillas (Dice 0,61146; 0,58833; 0,60086); DS-U-Net está entrenando y U-Net compacta sigue en cola. Se prepara informe común, subgrupos con cortes previos, panel de primeros cuatro IDs y reestimación de latencia una vez acabados los entrenamientos. No se afirma todavía que la comparación de nueve corridas esté completa.

## 2026-10-17 · D25 · Cierre de nueve corridas y costo completo **[IA: Codex]**
- **Estado:** nueve entrenamientos completos, cada uno con 400 train/100 dev y tres semillas por arquitectura. Suite ampliada de 21 tests aprobada. La fecha de cierre corresponde al nuevo contexto del cliente del 17; las carpetas de entrenamiento conservan el 16, fecha de inicio y plan.
- **Calidad:** Dice medio FCN-Lite 0,60022 (18.355 parámetros), DS-U-Net 0,65483 (18.665), U-Net compacta 0,78144 (487.289). Fuente `results/model_comparison/light_train400_20261016/`. Los nueve checkpoints recargados reprodujeron sus 100 Dice individuales a 1e−8. Los hashes de entrenamiento y los siete originales SAM permanecen intactos.
- **Comparación con SAM:** auto caja+punto 0,81792. Diferencia media pareada sobre U-Net +0,03648, IC 95 % [−0,00554; 0,07712]. No se demuestra una mejora positiva de esa media ni equivalencia. Los p de rangos del JSON no prueban la misma hipótesis sobre la media. Todo es exploratorio y seleccionado en dev.
- **Convergencia:** épocas elegidas FCN 30/29/28, DS-U-Net 30/30/30, U-Net 30/30/28. Muchas elecciones están al borde del presupuesto y las curvas siguen mejorando; no se afirma techo del baseline ni superioridad global de SAM.
- **Cómputo medido después de entrenar y ejecutar tests:** CPU/cuatro hilos/FP32, batch1, cinco calentamientos y veinte imágenes cronometradas. Mediana RGB FCN 6,86 ms; DS-U-Net 14,08 ms; U-Net 12,11 ms. SAM+probe→caja+punto completo 8.369,38 ms (p95 8.905,77), 93.735.729 parámetros totales únicos. Encoder mediano 8.243,85 ms. Se incluyen preprocesado, red completa y restauración; se excluyen disco y métricas. Los inputs son 128 frente a 1024 y el preentrenamiento difiere: son costos de sistemas prototipo, no aislamiento de arquitectura ni benchmark universal.
- **Fidelidad del perfil SAM:** las veinte predicciones desde caché y veinticinco ejecuciones desde RGB reprodujeron los scores previos a 1e−8; SHA del checkpoint cargado coincide con caché. El runtime representa semilla 0; los scores de calidad, tres semillas. Las medianas de componentes no necesariamente suman la mediana total.
- **Controles e interpretación:** prior espacial fijo train→dev da 0,35818 frente a 0,81792 localizado por imagen. Los subgrupos con cortes previos muestran medias SAM mayores en área <5 % y 5–15 %, y U-Net mayor en ≥15 %; son descripciones con grupos solapados, no mecanismos ni interacción clínica demostrada. Se inspeccionaron curvas, comparación de calidad y panel fijo de primeras cuatro imágenes.
- **Artefactos:** `docs/resultados_experimentacion_20261017.md`; informes y gráficos en `results/model_comparison/comparison_train400_20261017/`, incluyendo calidad/costo. Checkpoints y Adam/RNG en `runs/`, ignorado por Git. Se conserva el informe previo de calidad separado del perfil runtime y se prepara una revisión del abstract en carpeta nueva, sin sobrescribir el borrador inicial.
- **Decisión siguiente:** madurar resolución/convergencia, estudiar presupuestos de máscaras anidados y evaluación externa antes de ampliar el ranking. SAM 2.1/especialista y control del generador de igual capacidad siguen pendientes. Sin nueva evaluación de test, cambio de rama, commit, push ni envío al evento.
- **Cierre documental:** revisión del abstract en `docs/waimlap_2026/revision_20261017/`, dos páginas A4 verificadas con Word y PDFium e inspección visual, cabecera/logos preservados; borrador anterior intacto y autoría aún pendiente. Se verificaron 1.000 hashes de datos, ambos planes de fuentes congeladas, 900 filas seleccionadas con IDs únicos, enlaces locales y `git diff --check`.

## 2026-10-17 · D26 · Continuación U-Net a 128/256 **[IA: Codex]**
- **Plan previo:** seis brazos, resolución 128/256 × semillas 0/1/2. Todos parten del endpoint de época 30 de su semilla con pesos, Adam y RNG idénticos, y realizan 15 épocas adicionales sin parada temprana. Endpoint de 750 actualizaciones como medida principal; selección dev reiniciada simétricamente como secundaria. Plan y hashes se escribieron antes de entrenar en `runs/model_comparison/unet_resolution_20261017/`.
- **Resultado principal:** endpoint Dice 128 = 0,79548 y 256 = 0,82673; diferencia pareada 256−128 = +0,03125, IC 95 % [0,01215; 0,05128]. El cambio inmediato de los mismos pesos a 256 baja 0,07531 frente a 128; tras entrenamiento, el brazo 256 mejora +0,12345 sobre su inicio. Es readaptación desde un checkpoint entrenado a 128, no dos recetas desde cero.
- **Selección y SAM:** U-Net seleccionada 128 = 0,82142; 256 = 0,82721. SAM automático = 0,81792. SAM−U-Net 256 endpoint = −0,00881, IC [−0,04717; 0,02815]; SAM−U-Net 256 seleccionada = −0,00929 [−0,04808; 0,02793]. Los intervalos contienen cero: no prueban superioridad ni equivalencia. Dos semillas 256 eligen la última época; no se afirma convergencia.
- **Costo y control:** mediana local del pipeline U-Net 13,30 ms a 128 y 27,08 ms a 256; promedio de entrenamiento+dev por corrida 136,8 s y 484,3 s, respectivamente. Mismo protocolo CPU/cuatro hilos, pero no es una medición energética ni benchmark universal.
- **Verificación:** las seis corridas terminaron con código 0, se reprodujeron los checkpoints seleccionados por imagen y se exportaron endpoints separados. La suite de 22 tests pasó antes de ejecutar. El test nuevo comprueba pesos, Adam, RNG, selección por resolución e inicialización idempotente. No se evaluó test, no se cambió de rama y no hubo commit/push.
- **Decisión:** la aparente ventaja puntual de SAM bajo el presupuesto U-Net de 30 épocas no se sostiene al ampliar el baseline. El próximo experimento prioritario es eficiencia de etiquetas y luego evaluación externa; agregar más modelos queda subordinado a esas preguntas.

## 2026-10-17 · D27 · Utilidad de comparadores y revisión del estado del arte **[IA: Codex]**
- **Pregunta:** explicar para qué sirve comparar SAM con segmentadores directos y comprobar si la idea ya fue publicada.
- **Revisión dirigida:** se contrastaron fuentes primarias de `Can SAM Segment Polyps?`, AutoSAM, Self-Prompting, Semantic AutoSAM, PP-SAM, HSP-SAM, IP-SAM, Segment Any Tissue, HYDRA, el estudio TMLR de configuraciones de prompts y el trabajo MICCAI 2026 de incertidumbre por perturbación. No se presenta como revisión sistemática ni la ausencia de una coincidencia exacta como prueba de novedad.
- **Conclusión bibliográfica:** evaluación sin prompts, sensibilidad espacial, prompts automáticos, probe lineal→punto/caja, prompts abstractos y selección de candidatas tienen antecedentes. En particular, Self-Prompting es el antecedente directo del pipeline probe→caja+punto local, y AutoSAM/HSP-SAM/IP-SAM impiden reclamar novedad por eliminar el prompt humano.
- **Posicionamiento:** el diferencial candidato es la combinación de entrada realmente vacía, candidatos frente a selección, degradación espacial, prior fijo, permutación de contexto, baseline directo fortalecido y costo completo bajo IDs comunes. Se presenta como estudio controlado de diagnóstico/utilidad/costo, no arquitectura nueva ni SOTA.
- **Valor de U-Net:** controla si SAM agrega valor después de entrenar un localizador con las mismas máscaras. La igualdad no resuelta de Dice y la gran diferencia de latencia trasladan la pregunta a eficiencia de etiquetas y transferencia externa, donde el preentrenamiento de SAM podría justificar su costo.
- **Siguiente orden:** subconjuntos anidados 25/100/400; evaluación externa congelada; negativos y especificidad; selector solo donde exista headroom; después un self-prompting moderno/SAM2 y un especialista. Documento: `docs/nota_comparacion_y_estado_arte_20261017.md`.

## 2026-10-17 · D28 · Eficiencia de etiquetas bajo cómputo fijo **[IA: Codex]**
- **Pregunta y plan previo:** se compararon probe SAM, probe→caja+punto→SAM y U-Net RGB desde cero con 25/100/400 máscaras. Los subconjuntos son prefijos anidados de una permutación por semilla; tres semillas, `dev100` fijo y test vacío. El plan, IDs y hashes de 1.000 archivos se escribieron antes de entrenar.
- **Control de cómputo:** 300 actualizaciones Adam para el probe y 1500 para U-Net 256, independientemente del número de imágenes. El endpoint fijo es primario; selección U-Net cada 100 pasos es secundaria. Así se estudia eficiencia conjunta de etiquetas y preentrenamiento bajo cómputo constante, no convergencia universal.
- **Resultados:** Dice SAM automático 0,7821/0,8061/0,8190 y U-Net endpoint 0,5103/0,6198/0,6964 para 25/100/400. Diferencias pareadas +0,2718 [0,2185; 0,3243], +0,1863 [0,1333; 0,2390] y +0,1226 [0,0757; 0,1692]. El decoder añade 0,0665/0,0645/0,0702 sobre el probe; los tres IC excluyen cero.
- **Lectura:** SAM muestra mejor eficiencia con pocas etiquetas y la brecha cae al aumentar supervisión. U-Net mejora más con nuevas máscaras. La U-Net continuada previa (0,8267) recibió otra trayectoria de entrenamiento; por ello este estudio no sostiene superioridad final de SAM ni contradice la comparación de convergencia.
- **Ejecución y verificación:** nueve probes y nueve U-Net completos, 100 filas por corrida, reanudación por checkpoints y `test_evaluated=false`. Las tres pruebas nuevas y la suite restringida a `tests/` pasan: 25 tests y dos subtests. Una colección global inicial encontró una carpeta temporal inaccesible creada por pytest; no fue un fallo de código. Sin cambio de rama, commit, push ni nueva evaluación de test.
- **Artefactos:** `docs/protocolo_eficiencia_etiquetas_20261017.md`, `scripts/28_label_efficiency.py`, ejecutor de celda `scripts/29_run_label_efficiency_cell.py`, `results/model_comparison/label_efficiency_20261017/` y checkpoints en `runs/model_comparison/label_efficiency_20261017/`.

## 2026-10-17 · D29 · Transferencia externa congelada a CVC-ClinicDB **[IA: Codex]**
- **Datos y procedencia:** 612 frames/29 secuencias CVC-ClinicDB, archivo público SHA-256 `79b8105a…43f453`, uso restringido a investigación/educación según el README y la página oficial. La página oficial no mostró un archivo descargable; se documentó la copia pública utilizada. Datos y RAR quedan ignorados por Git.
- **Preflight:** 612 pares 1–612, resolución 384×288, ninguna máscara vacía, imágenes únicas y cero duplicados exactos por píxel con train400/dev100/test120 de Kvasir. Se congelaron hashes de 1224 archivos, SAM, tres probes, tres U-Net y fuentes antes de predecir.
- **Protocolo:** sin entrenamiento, calibración ni selección en destino. Probe/umbral de Fase B; auto-prompt caja+punto sin GT; endpoints U-Net 256 de la continuación. Media por frame y macro-secuencia; IC 95 % mediante 10.000 bootstraps de las 29 secuencias completas.
- **Resultado:** Dice probe 0,4243 [0,3624; 0,4892], SAM automático 0,4587 [0,3793; 0,5405] y U-Net 0,3405 [0,2841; 0,4002]. SAM−U-Net +0,1182 [0,0672; 0,1684]; SAM automático−probe +0,0344 [0,0110; 0,0576].
- **Interpretación:** el preentrenamiento y refinamiento de SAM muestran mayor robustez relativa al cambio de dominio, pero todos sufren una caída absoluta grande y no alcanzan calidad clínica defendible. U-Net presenta DE entre semillas 0,0729 en destino. CVC-ClinicDB carece de negativos, por lo que especificidad sigue pendiente.
- **Verificación:** 612 filas ordenadas y únicas, 29 grupos completos, métricas no constantes por método/semilla, `external_tuning=false`; 28 tests y dos subtests pasan. Ejecución completa en 89,3 min CPU/cuatro hilos. Sin nueva consulta al test Kvasir, commit, push ni ajuste post hoc.
- **Artefactos:** `docs/protocolo_validacion_externa_20261017.md`, `scripts/30_external_clinicdb.py`, `tests/test_external_clinicdb.py` y `results/model_comparison/external_clinicdb_20261017/`.

## 2026-10-18 · D30 · Control negativo externo en HyperKvasir **[IA: Codex]**
- **Pregunta y plan previo:** medir falsas activaciones donde no existe una máscara de pólipo positiva. Se fijaron antes de inferir 1.009 imágenes `cecum` de HyperKvasir, tres métodos y semillas congelados, área falsa, componente mayor y tasas ≥0,5/1/5 %. Dice/IoU se excluyeron porque colapsan el tamaño del error cuando la referencia es vacía.
- **Datos:** descarga selectiva desde OSF, 234.727.385 bytes, 1.009 hashes SHA-256 verificados, 1.009 imágenes decodificables y sin duplicados exactos de píxeles. La etiqueta anatómica sirve como control negativo de nivel imagen; no equivale a revisión negativa píxel a píxel y no incluye IDs públicos de paciente/video.
- **Resultados:** área falsa media probe 0,0581 [0,0552; 0,0611], SAM automático 0,0521 [0,0488; 0,0554] y U-Net 0,1733 [0,1690; 0,1776]. Tasa de área ≥1 %: 0,8794/0,7942/0,9931. SAM−U-Net en área = −0,1212 [−0,1267; −0,1159] y en tasa ≥1 % = −0,1989 [−0,2217; −0,1767].
- **Interpretación:** SAM automático es más conservador que U-Net y reduce parte del ruido del probe, pero no rechaza ningún frame por completo: probe y SAM generan máscara en 100 %; U-Net, 99,83 %. El estudio demuestra falsas activaciones técnicas bajo transferencia, no especificidad clínica por examen o paciente.
- **Verificación:** 1.009 filas únicas y completas; valores finitos en [0,1], variación no degenerada por método/semilla, `external_tuning=false`; suite completa: 31 tests y 2 subtests aprobados. La advertencia de caché pytest se debe al ACL local y no es un fallo de pruebas.
- **Artefactos:** `docs/protocolo_especificidad_negativos_20261017.md`, `scripts/31_download_hyperkvasir_cecum.py`, `scripts/32_negative_specificity.py`, `tests/test_negative_specificity.py` y `results/model_comparison/negative_specificity_20261017/`.

## 2026-10-18 · D31 · Manuscrito maestro y abstract WAIMLAp integrado **[IA: Codex]**
- **Solicitud:** documentar en inglés todo el proyecto, su contribución y resultados, añadir figuras y producir un borrador basado en la plantilla `WAIMLAp_2026_Extended_Abstract_Template (1).docx` antes de crear una rama nueva.
- **Manuscrito maestro:** `WAIMLAP_DRAFT.md` contiene pregunta, alcance, datasets, modelos, estadística, seis bloques de resultados, contribución, límites, próximo experimento, mapa reproducible y campos pendientes. Separa resultados de desarrollo, test histórico y transferencias externas.
- **Figura:** `scripts/33_make_paper_figures.py` lee cinco resúmenes congelados y genera un panel reproducible de dependencia del prompt, eficiencia de etiquetas, CVC-ClinicDB y control negativo. Procedencia en `results/paper_20261018/figure_provenance.json`.
- **Abstract:** nueva revisión en `docs/waimlap_2026/revision_20261018_complete/`, con la figura integrada. Word y PDF conservan la plantilla, tienen dos páginas verificadas y fueron inspeccionados visualmente. Copias de acceso directo: `WAIMLAP_DRAFT.docx` y `WAIMLAP_DRAFT.pdf`.
- **Posicionamiento:** el aporte se describe como diagnóstico controlado e integrado, no como nueva arquitectura, primer self-prompting ni estado del arte. El siguiente experimento propuesto es un gate calibrado de presencia/abstención sobre negativos agrupados por video/paciente.
- **Pendiente humano:** nombres, orden, afiliaciones y correo de correspondencia. El material es borrador interno; no se realizó envío, commit ni push.
