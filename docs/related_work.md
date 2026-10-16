# Trabajos relacionados y posicionamiento

Proyecto: **El costo del prompt en SAM** — INF659 Deep Learning, 2026-2. Artículo objetivo: WAIMLAp 2026 (PUCP, 18–20 de noviembre; artículo final de 4–8 páginas, 30 de noviembre).

Documento preparado el 16 de octubre de 2026 **[IA: Claude]** en respuesta a dos pedidos de los docentes:

1. ¿Hay trabajos que ya hayan hecho esto mismo? ¿Dónde está el diferencial?
2. Comparar SAM con otros modelos.

**Alcance de la búsqueda:** unas 25 consultas web y lectura de resúmenes, tablas o páginas de cada trabajo. **No es exhaustiva**. La columna «Revisado» indica qué se leyó; nadie del equipo ha leído aún los artículos completos. Los pendientes de lectura están en la sección 8.

---

## 1. Respuesta corta

- **Lo que ya está publicado:** que SAM sin prompt falla en pólipos y en imágenes médicas, y que se pueden generar prompts automáticamente para SAM. Ninguna de las dos cosas es un diferencial.
- **Dónde está el diferencial:** en el **diagnóstico**, es decir, en qué etapa exacta se pierde el desempeño (propuesta, filtros, selección o localización) y por qué. Lo cuantifican nuestros oráculos, las ablaciones y la atención del decoder.
- **El trabajo más cercano** es HYDRA (julio de 2026). Muestra, en imágenes naturales y con segmentadores tipo DETR (incluido SAM 3), que el cuello de botella es la selección y no la generación de máscaras. Hay que citarlo y diferenciarse explícitamente.
- **Comparar con otros modelos sí aporta**, pero como prueba de que el diagnóstico generaliza (¿es un problema de SAM v1 o de la familia?) y no como tabla de ranking.

## 2. Trabajos relacionados por grupo

### 2.1 Evaluaciones de SAM en pólipos y medicina

| Trabajo | Qué hace | Relación con nosotros | Revisado |
|---|---|---|---|
| Zhou et al., *Can SAM Segment Polyps?*, arXiv 2023 | SAM-H/L en modo *everything* sobre Kvasir, CVC-ClinicDB, CVC-300, ColonDB y ETIS. En Kvasir obtiene mDice 0,778 (SAM-H) y 0,782 (SAM-L), frente a 0,926 de HSNet | **De todas las máscaras elige la de mayor S-α calculada contra la GT, o sea, su cifra «zero-shot» incluye un oráculo de selección.** Nosotros medimos cuánto pesa ese oráculo (0,895 vs. 0,058). Atribuyen el fallo a los bordes difusos y al preentrenamiento natural, sin descomponerlo | Resumen y método |
| Mazurowski et al., *SAM for Medical Image Analysis: an Experimental Study*, MedIA 2023 | 11 conjuntos médicos; un prompt y prompts iterativos; reporta el «oráculo» de las 3 máscaras | Ya reporta la brecha entre el oráculo y la elección de SAM, pero no la analiza en AMG ni con la cabeza IoU fuera de dominio | Resumen |
| Huang et al., *Segment Anything Model for Medical Images?*, MedIA 2024 | COSMOS 1050K (53 conjuntos, 18 modalidades); modos *everything*, punto y caja | Benchmark amplio de modos de prompt; no diagnostica el fallo sin prompt | Resumen |
| Cheng et al., *SAM on Medical Images: A Comprehensive Study on Three Prompt Modes*, arXiv 2023 | Modos *everything*, punto y caja en varios conjuntos médicos | Igual que el anterior | Título y resumen |
| Mansoori et al., *Polyp SAM 2*, arXiv 2024 | SAM vs. SAM 2 *zero-shot* con puntos y cajas en Kvasir, CVC-ClinicDB y CVC-300 | La caja supera al punto y SAM 2 supera a SAM. Útil como referencia para comparar con SAM 2 | Resumen |

### 2.2 Prompts automáticos con SAM congelado o casi congelado

| Trabajo | Qué hace | Relación con nosotros | Revisado |
|---|---|---|---|
| Shaharabany et al., **AutoSAM**, BMVC 2023 | Reemplaza el prompt encoder por una red condicionada a la imagen y la entrena con gradientes de SAM congelado; incluye pólipos (SUN-SEG) | **Equivale a nuestro generador condicionado (Fase B).** Como contribución ya no es nueva; sirve como baseline o antecedente | Resumen |
| Wu, Zhang y Elbatel, **Self-Prompting Large Vision Models**, taller MICCAI DART 2023 | Clasificador lineal por píxel sobre el embedding de SAM que genera caja y punto para el propio SAM | **Equivale a nuestro «probe → caja/punto → SAM».** Hay que citarlo al presentar ese baseline | Resumen |
| Gaillochet et al., *Automating MedSAM by Learning Prompts*, taller MICCAI MedAGI 2024 | Módulo ligero que aprende el embedding del prompt a partir del embedding de imagen, con MedSAM congelado y supervisión débil con 10 muestras (RM y ultrasonido) | Cercano a los tokens aprendidos, pero condicionado a la imagen | Resumen |
| Zhang et al., *Hierarchical Self-Prompting SAM*, arXiv 2025 | Prompts «abstractos» aprendidos por auto-prompting; pólipos y piel | Otro método de prompt automático; no diagnostica | Resumen |
| *MaskSAM*, arXiv 2024 | Generador de prompts más tokens clasificadores aprendibles; volumétrico (AMOS, ACDC, Synapse) | Antecedente de tokens aprendibles dentro del decoder | Resumen |

### 2.3 Prompts de clase y tokens aprendibles

| Trabajo | Qué hace | Relación con nosotros | Revisado |
|---|---|---|---|
| *SAM-Path*, LNCS 2023 | Prompts de clase entrenables para segmentación semántica en patología, sin prompt humano | Análogo a nuestros tokens estáticos, aunque ellos también ajustan el modelo | Resumen |
| Yue et al., *SurgicalSAM*, AAAI 2024 | Prompt encoder de clase basado en prototipos que interactúan con el embedding de la imagen, más ajuste del decoder | Prompt de clase condicionado a la imagen; ajusta el decoder | Resumen |

### 2.4 Personalización y un solo ejemplo

| Trabajo | Qué hace | Relación con nosotros | Revisado |
|---|---|---|---|
| Zhang et al., **PerSAM**, ICLR 2024 | A partir de un solo ejemplo anotado, construye un prior de ubicación y guía la atención del decoder; sin entrenamiento | Obtiene el «dónde» automáticamente a partir de un ejemplo | Resumen |
| Mao et al., **OP-SAM**, ICCV 2025 | Pólipos con un solo ejemplo: DINOv2 y SAM 2 congelados, prior por correlación e iteración de prompts. En Kvasir reporta IoU 76,9 % | Competidor directo en «sin humano por imagen» para pólipos. Su criterio de parada usa el IoU predicho por SAM 2, cuya calibración en colonoscopía nadie ha validado; nuestro análisis de calibración es pertinente aquí | Reseña y resumen |

### 2.5 Selección, calibración e incertidumbre

| Trabajo | Qué hace | Relación con nosotros | Revisado |
|---|---|---|---|
| De la Jara, Rodriguez-Opazo y Ranasinghe, **HYDRA**: *Queries Knew More Than We Thought*, arXiv julio 2026 | Con un oráculo muestra que las máscaras útiles ya existen entre las propuestas, pero la selección las suprime. Lo prueba en Mask2Former, MaskDINO, OneFormer y SAM 3 (ADE20k/COCO y 8 dominios). Propone un selector entrenado sobre salidas en caché: +7,41 mIoU y +9,4 en SAM 3 | **Es el más cercano a nuestro hallazgo central.** Diferencias: no trabaja en dominio médico, no estudia la cabeza IoU ni el AMG de SAM v1 ni el filtro de estabilidad, y no analiza la ausencia de prompt. Si proponemos un selector, hay que posicionarlo frente a este | Resumen |
| *SAMRefiner*, arXiv 2025 | Refinamiento de máscaras con SAM. Señala que la elección por IoU predicho queda bajo el límite que da elegir con la GT, porque la cabeza IoU no se entrena para el objeto de interés | Apoya nuestra lectura de la cabeza IoU | Fragmento |
| Deng et al., *SAM-U*, arXiv 2023 | Incertidumbre a partir de varias cajas perturbadas en imágenes médicas | Antecedente de fiabilidad de SAM en medicina | Resumen |

### 2.6 Prompts de concepto (SAM 3)

| Trabajo | Qué hace | Relación con nosotros | Revisado |
|---|---|---|---|
| Carion et al., **SAM 3: Segment Anything with Concepts**, ICLR 2026 | Segmentación promptable por concepto (frase o ejemplo visual) | Un prompt de texto aporta el «qué» sin el «dónde», como nuestros tokens estáticos. Es el puente natural para la comparación | Título y página |
| *MedSAM3*, arXiv 2511.19046 | Adapta SAM 3 a medicina con conceptos clínicos y un agente | **Reporta que SAM 3 original, solo con texto, da Dice 0 en Kvasir-SEG, y 0,767 con texto + caja.** Lo atribuye a una desalineación entre concepto y región | Tabla de resultados |
| *Medical SAM3*, arXiv 2601.10880 | *Fine-tuning* completo de SAM 3 en 33 conjuntos médicos | SAM 3 original solo compite cuando recibe cajas derivadas de la GT. Reporta colapso en CVC-ClinicDB y ETIS; el valor de Kvasir hay que leerlo en su tabla (sección 8) | Resumen y tabla (parcial) |

### 2.7 Robustez al prompt en pólipos

| Trabajo | Qué hace | Relación con nosotros | Revisado |
|---|---|---|---|
| Rahman et al., **PP-SAM**, CVPRW 2024 | Ajuste de SAM con cajas perturbadas para pólipos; en Kvasir evalúa perturbaciones de 50 y 100 px | **Antecedente directo de nuestra ablación espacial** (caja ampliada, punto desplazado; D18). Citarlo al presentarla | Resumen |

## 3. Lo que ya está hecho (no es diferencial)

- **SAM sin prompt o en modo *everything* rinde mal en pólipos** (Zhou 2023; Huang 2024; Cheng 2023).
- **La caja rinde mejor que el punto, y SAM 2 mejor que SAM** (Polyp SAM 2; Mazurowski 2023).
- **Generar prompts automáticos con SAM congelado**: con una red condicionada a la imagen (AutoSAM), con un clasificador lineal sobre el embedding (Self-Prompting), con prompts de clase (SAM-Path, SurgicalSAM) o con un solo ejemplo (PerSAM, OP-SAM).
- **El cuello de botella es la selección, no la propuesta**, en segmentadores tipo DETR sobre imágenes naturales (HYDRA).
- **SAM 3 solo con texto falla en pólipos** (MedSAM3, Medical SAM3).

## 4. Lo que no encontramos: diferencial candidato

Cifras del E1 (test, n = 120). Donde se indica, se replican en el dev de la Fase B (n = 100, D18).

1. **El embudo del AMG en dominio médico, cuantificado.** La mejor propuesta cruda (oráculo) logra 0,895. Después de cada etapa queda:

   | Etapa | Dice |
   |---|---|
   | Filtro de IoU predicho | 0,792 |
   | Filtro de estabilidad | 0,500 |
   | NMS | 0,493 |
   | Elección por IoU predicho | 0,058 |

   **El filtro de estabilidad por sí solo quita 0,29.** La mejor propuesta tiene estabilidad mediana 0,79, por los bordes difusos de la mucosa. No lo encontramos reportado.
2. **La cabeza IoU no sirve como selector de la lesión.** En las 92 160 propuestas del AMG, Spearman(IoU predicho, IoU con el pólipo) = 0,03.
   - **Matiz necesario** (`docs/analisis_experto.md`): esa correlación global mezcla relevancia semántica y calidad geométrica. Prueba que la cabeza no sirve para elegir la lesión, no que esté descalibrada respecto de cualquier objeto. Para separar ambos efectos falta el análisis dentro de un mismo punto, sobre todo en el interior del pólipo.
   - **Con un clic humano, la práctica recomendada por SAM (multimáscara + cabeza IoU) es la peor de tres opciones**: test 0,571, frente a 0,696 con salida única y 0,840 con el oráculo de 3. **Se replica en dev**: 0,540, 0,683 y 0,860.
3. **Sensibilidad cuantificada al «dónde»** (dev, D18):
   - punto desplazado 2 radios: 0,153;
   - caja ampliada 2×: 0,322;
   - caja de imagen completa: 0,033, frente a 0,935 con la caja GT.

   PP-SAM trata la robustez a cajas perturbadas como objetivo de entrenamiento; nosotros la usamos como instrumento de diagnóstico.
4. **Evidencia mecanística.** Masa de atención del token de máscara sobre el pólipo: 4,9 % sin prompt, 23 % con tokens aprendidos y 59 % con caja. Es descriptiva, no causal.
5. **«Qué» vs. «dónde» con predicciones pre-registradas y resultados negativos reportados.**
   - El token estático inicializado al azar colapsa (0,008).
   - **Hallazgo nuevo de la Fase B:** en dev, la inicialización con el clic medio *sin entrenar* ya da 0,3505, y el entrenamiento llega a 0,3513 (semilla 0). El 0,318 del E1 se explica casi por completo por la inicialización. El artículo debe describirlo como «prompt de clic medio», no como aprendizaje.

## 5. Enfoque recomendado para el artículo

**Pregunta:** ¿dónde se pierde el desempeño de SAM cuando se quita el prompt humano en un dominio fuera de distribución, y qué parte de esa pérdida se recupera atacando cada etapa?

**Título de trabajo:** *«SAM propone pero no elige: descomponiendo el costo del prompt en segmentación de pólipos»*.

Contribuciones defendibles:

- **C1 (diagnóstico).** Descomponer el costo del prompt en propuesta, filtros, selección y localización, con oráculos, ablación espacial, predicciones pre-registradas, IC por bootstrap y pruebas pareadas.
- **C2 (mecanismo).** Mostrar que la cabeza IoU no sirve para elegir la lesión, que el filtro de estabilidad castiga los bordes difusos, cuánta precisión del «dónde» hace falta y cómo cambia la atención del decoder.
- **C3 (generalidad y mitigación).** Medir lo mismo en otros modelos de la familia (sección 6) y aplicar una mitigación dirigida al cuello de botella identificado (sección 7).

**Qué no reclamar:**

- un método nuevo de auto-prompting (AutoSAM, Self-Prompting);
- que el encoder «no falla», en sentido causal;
- que la atención explica el mecanismo completo;
- que un prompt estático es *incapaz* de localizar. Solo se probó una familia limitada de configuraciones.

## 6. Comparación con otros modelos (plan)

**Criterio:** pocos modelos, elegidos para responder una pregunta. A cada uno se le aplica el mismo protocolo:

- sin prompt: salida única, multimáscara + IoU y oráculo;
- un punto y una caja GT;
- embudo del AMG, donde exista;
- calibración de la cabeza IoU, global y dentro de un mismo punto;
- probe lineal;
- fracción de la brecha recuperada, R.

| Modelo | Pregunta que responde | Costo / viabilidad | Riesgos |
|---|---|---|---|
| SAM ViT-B (base) | Línea base | Hecho (E1) | — |
| **MedSAM** (ViT-B ajustado con imágenes médicas) | ¿El ajuste médico arregla la selección y el «qué», o solo mejora con caja? | Bajo: misma arquitectura, entra en el pipeline con caché. Hay que recalcular embeddings (unos 70 min por 400 imágenes en 4 hilos de CPU) | **Verificar si Kvasir-SEG estuvo en su entrenamiento (fuga de datos).** Se entrenó solo con cajas y salida única, así que su cabeza IoU y su multimáscara pueden no ser comparables |
| **SAM 2.1** (encoder Hiera) | ¿El fallo de selección depende del encoder o del diseño del decoder? | Medio: las variantes *tiny* y *small* son viables en CPU; tiene generador automático propio | Requiere adaptar la caché; el curso lista SAM 2 explícitamente |
| **SAM 3 con el texto «polyp»** | ¿Un prompt de concepto (el «qué» sin el «dónde») sustituye al clic? Es el análogo con lenguaje de nuestros tokens estáticos | Alto: necesita GPU (Colab) y aceptar la licencia en Hugging Face | MedSAM3 ya reporta Dice ≈ 0. El aporte sería diagnosticar *por qué*, no reportar la cifra. Revisar su cabeza de presencia, que separa reconocimiento de localización |
| Especialistas (PraNet, Polyp-PVT) | Techo de referencia | Ninguno: se citan las cifras publicadas | **Usan otra partición** (normalmente 900/100 de PraNet; nosotros usamos MediaEval 2020, D5): no comparar directamente |
| (Opcional) probe lineal sobre DINOv2/v3 | ¿El «qué» está mejor representado en otro encoder? | Bajo con caché; el curso lista DINOv2 | Amplía el alcance; solo si sobra tiempo |

**Qué evitar:** una tabla de 6 a 8 modelos sin pregunta detrás. Consume el tiempo que queda y no aporta contribución.

## 7. Mitigación recomendada para la Fase B

Como el cuello de botella medido es la selección, la mitigación más defendible apunta ahí:

1. **Ablación sin costo:** relajar o quitar el filtro de estabilidad y medir de nuevo el embudo, ajustando los umbrales en dev.
2. **Selector ligero sobre las propuestas del AMG**, entrenado solo con train y elegido en dev.
   - **Entradas:** IoU predicho, estabilidad, área, posición y embedding promediado dentro de la máscara.
   - **Reporte:** la calidad del conjunto de propuestas y la calidad de la selección, por separado.
   - **Posicionamiento:** frente a HYDRA (dominio natural, segmentadores tipo DETR) y frente a AutoSAM y Self-Prompting, que generan prompts en vez de elegir propuestas.
3. **El generador condicionado (D15–D16) se mantiene como baseline**, citando AutoSAM y *Automating MedSAM*, no como contribución. Su único resultado hasta ahora es el barrido con una semilla (0,399 en dev), y el experimento está incompleto (ver `docs/fase_b.md`).

## 8. Pendientes y riesgos

**Lectura completa** (repartir en el equipo):

- AutoSAM y Self-Prompting: qué conjuntos usan y si incluyen Kvasir-SEG.
- HYDRA: método del selector, para la comparación de la sección 7.
- OP-SAM: protocolo, y cómo usa el IoU predicho como criterio de parada.
- MedSAM3 y Medical SAM3: las cifras de SAM 3 solo con texto en Kvasir. Los resúmenes difieren y hay que leer las tablas.
- MedSAM: lista de conjuntos de entrenamiento (posible fuga de Kvasir-SEG).

**Búsqueda pendiente:**

- quién cita a AutoSAM, Self-Prompting y HYDRA (Google Scholar);
- trabajos sobre el AMG de SAM 2 en pólipos;
- análisis de la cabeza IoU fuera de dominio.

**Riesgos:**

- **Partición distinta a la de la literatura de pólipos**: hay que declararlo en cada comparación.
- **El test ya se inspeccionó en detalle**: las decisiones nuevas son exploratorias y la confirmación necesita dev o un conjunto externo.
- **Tiempo**:
  - entrega del trabajo: 16 de noviembre;
  - WAIMLAp: 18–20 de noviembre;
  - artículo final: 30 de noviembre.

  Conviene priorizar C1–C2 y una sola mitigación antes que ampliar la comparación.

## Referencias

- Zhou et al. *Can SAM Segment Polyps?* arXiv:2304.07583. <https://arxiv.org/abs/2304.07583>
- Mazurowski et al. *Segment Anything Model for Medical Image Analysis: an Experimental Study.* arXiv:2304.10517. <https://arxiv.org/abs/2304.10517>
- Huang et al. *Segment Anything Model for Medical Images?* arXiv:2304.14660. <https://arxiv.org/abs/2304.14660>
- Cheng et al. *SAM on Medical Images: A Comprehensive Study on Three Prompt Modes.* arXiv:2305.00035. <https://arxiv.org/abs/2305.00035>
- Mansoori et al. *Polyp SAM 2: Advancing Zero-shot Polyp Segmentation in Colorectal Cancer Detection.* arXiv:2408.05892. <https://arxiv.org/abs/2408.05892>
- Shaharabany et al. *AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder.* BMVC 2023. arXiv:2306.06370. <https://arxiv.org/abs/2306.06370>
- Wu, Zhang, Elbatel. *Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation.* MICCAI DART 2023. arXiv:2308.07624. <https://arxiv.org/abs/2308.07624>
- Gaillochet, Desrosiers, Lombaert. *Automating MedSAM by Learning Prompts with Weak Few-Shot Supervision.* MICCAI MedAGI 2024. arXiv:2409.20293. <https://arxiv.org/abs/2409.20293>
- Zhang et al. *Hierarchical Self-Prompting SAM: A Prompt-Free Medical Image Segmentation Framework.* arXiv:2506.02854. <https://arxiv.org/abs/2506.02854>
- *MaskSAM: Towards Auto-prompt SAM with Mask Classification for Medical Image Segmentation.* arXiv:2403.14103. <https://arxiv.org/abs/2403.14103>
- *SAM-Path: A Segment Anything Model for Semantic Segmentation in Digital Pathology.* arXiv:2307.09570. <https://arxiv.org/abs/2307.09570>
- Yue et al. *SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation.* AAAI 2024. arXiv:2308.08746. <https://arxiv.org/abs/2308.08746>
- Zhang et al. *Personalize Segment Anything Model with One Shot (PerSAM).* ICLR 2024. arXiv:2305.03048. <https://arxiv.org/abs/2305.03048>
- Mao et al. *One Polyp Identifies All: One-Shot Polyp Segmentation with SAM via Cascaded Priors and Iterative Prompt Evolution (OP-SAM).* ICCV 2025. arXiv:2507.16337. <https://arxiv.org/abs/2507.16337>
- De la Jara, Rodriguez-Opazo, Ranasinghe. *Queries Knew More Than We Thought: Uncovering Latent Knowledge in Segmentation Models (HYDRA).* arXiv:2609.20283. <https://arxiv.org/abs/2609.20283>
- *SAMRefiner: Taming Segment Anything Model for Universal Mask Refinement.* arXiv:2502.06756. <https://arxiv.org/abs/2502.06756>
- Deng et al. *SAM-U: Multi-box Prompts Triggered Uncertainty Estimation for Reliable SAM in Medical Image.* arXiv:2307.04973. <https://arxiv.org/abs/2307.04973>
- Carion et al. *SAM 3: Segment Anything with Concepts.* ICLR 2026. <https://proceedings.iclr.cc/paper_files/paper/2026/file/e0982cbc81401df3430ee1ff780dc7a2-Paper-Conference.pdf>
- *MedSAM3: Delving into Segment Anything with Medical Concepts.* arXiv:2511.19046. <https://arxiv.org/abs/2511.19046>
- Jiang et al. *Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation.* arXiv:2601.10880. <https://arxiv.org/abs/2601.10880>
- Rahman et al. *PP-SAM: Perturbed Prompts for Robust Adaptation of Segment Anything Model for Polyp Segmentation.* CVPRW 2024. arXiv:2405.16740. <https://arxiv.org/abs/2405.16740>
- Kirillov et al. *Segment Anything.* ICCV 2023. arXiv:2304.02643. <https://arxiv.org/abs/2304.02643>
