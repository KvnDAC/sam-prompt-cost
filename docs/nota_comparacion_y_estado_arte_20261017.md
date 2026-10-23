# Para qué sirve la comparación y cómo posicionar la investigación

Revisión dirigida al 17 de octubre de 2026. Este documento distingue resultados propios, antecedentes publicados y decisiones para mejorar el proyecto. No es una revisión sistemática exhaustiva; no encontrar una combinación idéntica no demuestra por sí solo novedad.

## Respuesta central

Comparar SAM con U-Net y otros modelos automáticos no busca decidir qué arquitectura es universalmente mejor. Sirve para responder una pregunta necesaria:

> Si para eliminar el prompt humano debemos entrenar un localizador con máscaras de Kvasir-SEG, ¿qué aporta conservar SAM frente a entrenar directamente un segmentador especializado con las mismas máscaras?

Sin ese control, un Dice alto de `probe → caja+punto → SAM` podría atribuirse a SAM aunque la mayor parte de la tarea ya la resolviera el probe. La comparación directa muestra si la representación y el decoder preentrenados aportan calidad, eficiencia con pocas etiquetas, robustez o transferencia suficientes para justificar su costo.

Con 400 imágenes de entrenamiento y 100 de desarrollo, el endpoint U-Net a 256 obtiene Dice 0,8267 y el pipeline automático de SAM 0,8179. SAM−U-Net es −0,0088, IC 95 % [−0,0472; 0,0281]. El intervalo no demuestra superioridad ni equivalencia. En esta CPU, las medianas son 27,08 ms para U-Net 256 y 8,37 s para SAM completo. La conclusión práctica es que SAM todavía no justifica su mayor costo por calidad media en este escenario de desarrollo. La hipótesis que queda abierta es que su preentrenamiento pueda resultar ventajoso con menos etiquetas o al transferir a otros centros/datasets.

## Qué identifica cada comparación

| Comparación | Variable que intenta aislar | Resultado propio | Conclusión permitida | Lo que no demuestra |
|---|---|---:|---|---|
| Entrada espacial vacía frente a punto/caja GT | Valor de aportar localización y extensión correctas | 0,0089 frente a 0,6831/0,9352 Dice | SAM depende fuertemente de información espacial en Kvasir dev | Que el decoder tenga un defecto intrínseco; sin prompt también cambia la especificación del objeto |
| Selección realizable frente al mejor candidato según GT | Calidad de candidatas frente a calidad del selector | Sin prompt: 0,1504 frente a 0,1824; con punto GT: 0,5401 frente a 0,8600 | Con entrada vacía faltan candidatas útiles; con punto existe además una brecha grande de selección | Que un selector pueda recuperar la caja GT cuando la candidata no existe |
| Prompts GT perturbados | Robustez ante error de localización | Punto desplazado dos radios: 0,1529; caja 2×: 0,3216 | La precisión espacial importa y el tipo de error cambia el rendimiento | Tiempo o dificultad real de una anotación humana |
| Prior fijo train→dev frente a localización por imagen | Tener cualquier prompt frente a encontrar el objeto concreto | Caja+punto fijo 0,3582; automático por imagen 0,8179 | La ganancia procede en gran parte de localización específica por imagen | Que el generador usado sea óptimo o arquitectónicamente nuevo |
| Tokens estáticos frente a condicionados y contexto permutado | Contenido específico de imagen frente a capacidad/prior aprendido | 0,3633 frente a 0,5013; permutar contexto reduce solo 0,0060 | No puede atribuirse toda la mejora a contexto correctamente emparejado | Un mecanismo causal completo del generador |
| Probe frente a probe→prompts→SAM | Valor marginal del decoder de SAM sobre el localizador | 0,7489 frente a 0,8179; +0,0690 [0,0471; 0,0919] | SAM refina de forma útil la salida del probe en este protocolo | Que el pipeline sea mejor que un segmentador directo entrenado |
| SAM automático frente a U-Net | Valor del sistema foundation frente a una solución supervisada directa | 0,8179 frente a 0,8267; diferencia compatible con cero | U-Net es un control competitivo y evita atribuir calidad automáticamente a SAM | Comparación pura de arquitecturas: cambian preentrenamiento, resolución y receta |
| U-Net 128 frente a 256 con estado inicial común | Sensibilidad del baseline a presupuesto/resolución | Endpoint 256−128: +0,0313 [0,0122; 0,0513] | El baseline anterior estaba limitado y la resolución ayuda tras readaptación | Dos entrenamientos desde cero o convergencia definitiva |
| Latencia completa desde RGB | Costo práctico del sistema desplegable | 27,08 ms frente a 8,37 s en CPU local | El encoder SAM domina el prototipo local | Ventaja universal en GPU, energía o tiempo humano |

Estas comparaciones convierten una tabla de Dice en un diagnóstico. Además producen resultados negativos útiles: el generador condicionado no demostró depender fuertemente del contexto correcto y SAM no demostró una ventaja media sobre U-Net al fortalecer el baseline.

## Qué ya se publicó

### Evaluar SAM sin prompts o en pólipos

- [Can SAM Segment Polyps?](https://arxiv.org/html/2304.07583v1) evalúa SAM en cinco datasets de pólipos. Su condición llamada *unprompted* genera varias máscaras y selecciona con la GT la de mayor S-measure; es una selección oráculo, no un sistema automático desplegable. El proyecto actual aporta la separación explícita entre candidato elegido por score, mejor candidato y entrada realmente vacía.
- [SAM on Medical Images](https://arxiv.org/abs/2305.00035) compara puntos, cajas y máscaras en múltiples modalidades, y ya reporta que las cajas y su perturbación cambian mucho el rendimiento. Por eso, “la caja rinde mejor que el punto” o “SAM es sensible al prompt” no son novedades suficientes.
- [PP-SAM, CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Rahman_PP-SAM_Perturbed_Prompts_for_Robust_Adaption_of_Segment_Anything_Model_CVPRW_2024_paper.html) entrena con cajas perturbadas para hacer SAM más robusto en pólipos. Nuestra curva de degradación es diagnóstica; no debe presentarse como la primera investigación de perturbaciones.
- [When Does Prompt-Perturbation Uncertainty Catch Interactive-Segmentation Failures?, MICCAI 2026](https://papers.miccai.org/miccai-2026-sat/UNSURE2026_002.html) incluye Kvasir-SEG, SAM/MedSAM, clics y cajas no oráculo. Estudia si la variación ante prompts puede detectar errores y encuentra que varias conclusiones desaparecen con pruebas pareadas. Refuerza la necesidad de incertidumbre pareada, semillas y cautela causal.

### Generar prompts automáticamente

- [AutoSAM, BMVC 2023](https://arxiv.org/html/2306.06370v1) reemplaza el condicionamiento manual por un encoder HarDNet entrenado mediante un SAM congelado. Evalúa Kvasir-SEG y otros datasets de pólipos. Es un antecedente directo del generador condicionado, aunque no es la misma arquitectura.
- [Self-Prompting Large Vision Models, 2023](https://arxiv.org/html/2308.07624v1) entrena un clasificador lineal por píxel sobre embeddings SAM, obtiene una máscara gruesa y deriva un punto interior y una caja. Es el antecedente más directo de `probe → caja+punto → SAM` y también usa Kvasir-SEG. El método local debe describirse como una implementación relacionada/adaptada, no como una idea arquitectónica original.
- [Semantic AutoSAM, EMBC 2024](https://pubmed.ncbi.nlm.nih.gov/40040072/) predice embeddings de prompt desde features mediante atención cruzada y elimina el prompt manual en CT y ultrasonido. Es cercano a los tokens condicionados locales.
- [Hierarchical Self-Prompting SAM, 2025](https://arxiv.org/html/2506.02854v1) aprende prompts abstractos, evalúa pólipos y reporta transferencia entre Kvasir, ClinicDB, CVC-300, ColonDB y ETIS. Reduce el espacio para reclamar novedad por “prompts aprendidos sin humano”.
- [IP-SAM, 2026](https://arxiv.org/abs/2603.27250) formula explícitamente el desajuste entre entrenamiento condicionado y despliegue sin prompts; genera prompts intrínsecos para el prompt encoder congelado de SAM2 y también evalúa segmentación de pólipos.
- [Segment Any Tissue](https://pubmed.ncbi.nlm.nih.gov/40120286/) genera puntos automáticos desde una referencia anotada, sin entrenamiento adicional, y evalúa seis tareas médicas. Muestra que también existe una línea de prompting automático *one-shot*.

### Selección de candidatas y costo

- SAM fue diseñado para producir varias máscaras ante prompts ambiguos y ordenarlas con IoU predicho; esta separación forma parte del [artículo original de SAM](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf).
- [HYDRA, 2026](https://arxiv.org/abs/2609.20283) muestra en modelos DETR congelados y SAM 3 que pueden existir buenas candidatas que la regla de selección no expone, y entrena un selector ligero sobre salidas almacenadas. Es un antecedente importante del diagnóstico de selección, aunque no estudia SAM v1 sin prompt ni Kvasir-SEG.
- [Mastering SAM Prompts, TMLR 2025](https://mlanthology.org/tmlr/2025/price2025tmlr-mastering/) compara miles de configuraciones de puntos, cajas y máscaras en imágenes científicas. Confirma que comparar configuraciones de prompts es una línea activa, no una novedad aislada.

## Veredicto de novedad

No se encontró, en esta búsqueda dirigida, un trabajo que reúna exactamente en Kvasir-SEG los siguientes elementos bajo un protocolo común:

1. Entrada espacial realmente vacía.
2. Separación de disponibilidad de candidatas y selección.
3. Curvas de degradación de punto/caja.
4. Prior espacial fijo frente a localización por imagen.
5. Prueba de permutación del contexto del generador.
6. Mitigación automática y baseline RGB fortalecido con los mismos IDs.
7. Comparación de calidad y latencia completa desde RGB.

Esa combinación es el diferencial candidato: **un estudio controlado de diagnóstico, utilidad y costo**, no un nuevo método de auto-prompting. La afirmación debe ser “hasta donde cubre nuestra revisión dirigida” y necesita una búsqueda bibliográfica sistemática antes de una publicación formal.

### Afirmaciones defendibles

- La ausencia de información espacial cambia tanto la localización disponible como la especificación del objetivo.
- Bajo entrada vacía, mejorar solo la selección no recupera la calidad de una caja correcta porque el conjunto evaluado carece de candidatas suficientemente buenas.
- Bajo un punto correcto, la selección por IoU predicho deja una brecha grande frente al mejor candidato.
- La localización automática específica por imagen aporta mucho más que un prior espacial promedio.
- Un segmentador directo compacto es un control competitivo, por lo que la utilidad de SAM debe evaluarse mediante etiquetas, transferencia y costo, no solo Dice en un dev.

### Afirmaciones que deben evitarse

- “Primer método sin prompts para pólipos”.
- “Primer generador automático de prompts para SAM”.
- “El probe→caja+punto es una arquitectura nueva”.
- “Se demostró que el decoder está defectuoso”.
- “SAM es peor/mejor que U-Net” o “son equivalentes” usando un IC que contiene cero.
- “Se midió el costo humano del prompt”: se midió pérdida de calidad y costo computacional, no tiempo de anotadores.
- “Estado del arte” sin evaluación externa, especialistas modernos y protocolo comparable.

## Cómo mejorarlo: orden experimental recomendado

### 1. Experimento principal siguiente: eficiencia de etiquetas

Usar subconjuntos anidados, fijados antes de correr, de 25, 100 y 400 máscaras. Para cada presupuesto:

- U-Net 256 con receta razonable y tres semillas.
- Probe lineal solo.
- Probe→caja+punto→SAM.
- Como reproducción relacionada, regresión logística+morfología del Self-Prompting original o una declaración precisa de las diferencias con el probe local.

La pregunta primaria sería si el preentrenamiento de SAM mejora la calidad cuando las etiquetas son escasas. Reportar Dice/IoU, diferencia pareada, tiempo de entrenamiento, latencia y memoria. Este experimento es más informativo que añadir varias arquitecturas con 400 etiquetas, y conecta directamente con la motivación de foundation models.

No reutilizar dev para escoger una receta distinta en cada punto sin contabilizarlo. Fijar los subconjuntos por inclusión (`train25 ⊂ train100 ⊂ train400`), mantener los 100 dev y no leer nuevamente el test histórico.

### 2. Generalización externa congelada

Después de fijar método, umbral y checkpoint sin mirar el destino, entrenar en Kvasir y evaluar directamente en CVC-ClinicDB, CVC-ColonDB, CVC-300 y ETIS. No mezclar cifras publicadas: ejecutar los modelos propios con el mismo preprocesado y métricas. Registrar si los pesos externos usados por cualquier comparador ya vieron esos datasets.

La contribución mejora mucho si se responde: ¿SAM mantiene calidad mejor que U-Net cuando cambia el centro, la resolución o la apariencia? Esa es una justificación más fuerte para pagar el costo de un foundation model.

### 3. Añadir negativos y especificidad

Kvasir-SEG es un conjunto positivo: cada imagen contiene un pólipo anotado. Un sistema automático también debe decidir cuándo no hay pólipo. Incorporar frames negativos, idealmente agrupados por video/paciente, y reportar tasa de falsos positivos por imagen o frame. Esto puede cambiar la comparación: una caja/punto siempre generados fuerzan a SAM a segmentar algo.

### 4. Mejorar el diagnóstico de selección

Entrenar solo con train un selector ligero de candidatas usando IoU predicho, estabilidad, área, posición y consistencia ante pequeñas perturbaciones. Calibrar su opción de conservar la predicción base en dev. Compararlo con IoU predicho y oráculo.

Este experimento debe concentrarse donde ya existe headroom, por ejemplo con punto o propuestas automáticas. Los resultados actuales muestran que un selector no resuelve por sí solo la entrada vacía.

### 5. Un comparador moderno, elegido por hipótesis

Si hay GPU, la ampliación más relevante es un self-prompting moderno como HSP-SAM o IP-SAM/SAM2, además de un especialista de pólipos como Polyp-PVT. No conviene sumar modelos solo para hacer una tabla grande:

- HSP-SAM/IP-SAM prueba si prompts abstractos o intrínsecos superan la localización explícita.
- SAM2 prueba transferencia del diagnóstico entre generaciones.
- Polyp-PVT prueba si un especialista moderno domina tanto a SAM como a la U-Net compacta.

Con CPU, es preferible terminar eficiencia de etiquetas y evaluación externa antes de intentar reproducciones costosas o incompletas.

### 6. Formalizar mejor el “costo del prompt”

Usar dos ejes separados:

- **Costo informativo empírico:** pérdida de Dice/IoU al retirar, degradar o reducir prompts; número de clics hasta un umbral; AUC de calidad frente a perturbación.
- **Costo operacional:** latencia, memoria, parámetros, tiempo de entrenamiento y, si se realiza un estudio con personas, segundos de anotación y correcciones.

No llamar “bits de información” a punto/caja sin una codificación formal. No equiparar clics simulados con tiempo humano.

## Relato recomendado para el artículo

La historia más fuerte es:

1. SAM colapsa con entrada espacial vacía, pero “sin prompt” debe definirse con precisión.
2. El motivo depende del régimen: sin prompt faltan candidatas; con un punto, también falla la selección.
3. La localización automática recupera gran parte de la calidad, mientras un prior espacial fijo no.
4. Esa mitigación ya pertenece a una familia publicada de self-prompting y se presenta como baseline relacionado.
5. Al fortalecer U-Net, SAM no muestra ventaja media en dev y cuesta mucho más en la CPU observada.
6. La pregunta científica decisiva pasa a ser cuándo el preentrenamiento de SAM compensa ese costo: pocas etiquetas y transferencia externa.

Con este posicionamiento, el proyecto deja de depender de una reclamación frágil de arquitectura nueva y se convierte en una evaluación reproducible que puede orientar cuándo usar SAM y cuándo usar un segmentador directo.
