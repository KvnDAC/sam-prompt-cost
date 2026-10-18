# Revisión del documento de la rama cambios-analisis

Fecha: 16 de octubre de 2026. Documento revisado: `docs/related_work.md`, commit remoto `ca03c31380de04d35ad5b2f3bd35bcfc71f0c328`. Se copió el archivo original sin cambiar sus bytes ni hacer merge; las observaciones se registran aquí. No se evalúa al asistente que lo redactó, sino la precisión y el soporte de sus afirmaciones.

## Dictamen

El documento es una base útil: identifica antecedentes directos, reconoce lecturas pendientes, distingue oráculos de métodos desplegables y propone un foco diagnóstico. **Todavía necesita correcciones antes de convertirse en el relato del artículo.** La ausencia de un trabajo en una búsqueda dirigida no acredita novedad y algunas conclusiones exceden la evidencia disponible.

## Aciertos comprobados

- [Can SAM Segment Polyps?, sección II-B2](https://arxiv.org/html/2304.07583v1) selecciona la máscara con mayor S-measure respecto de GT. Es correcto señalar que su resultado incluye selección oráculo; no es la calidad de un selector automático realizable.
- [Self-Prompting, secciones 3–4](https://arxiv.org/html/2308.07624v1) usa un clasificador lineal sobre embeddings SAM para producir prompts. Es un antecedente directo del probe→prompts de este proyecto.
- [PP-SAM](https://arxiv.org/abs/2405.16740) estudia adaptación a prompts de caja perturbados en pólipos. Debe citarse al discutir robustez espacial.
- [HYDRA](https://arxiv.org/html/2609.20283v1) estudia selección de candidatos congelados. Su tabla de SAM 3 abarca ADE-847, VOC, Cityscapes, CamVid, LoveDA, PC-59, ImageNet-S50 y COCO; es pertinente como antecedente de selección, no como un resultado propio en Kvasir.

## Correcciones prioritarias

| Afirmación o propuesta del documento | Corrección y consecuencia |
|---|---|
| Tratar «everything» y decoder sin prompts como la misma condición | AMG consulta puntos internos. Separar ausencia de intervención humana, ausencia de prompts espaciales y ausencia de supervisión específica. Los resultados de un generador automático no caracterizan por sí solos el decoder con cero tokens. |
| Titular de forma general que SAM propone bien pero no elige | Eso puede describir el AMG evaluado. En entrada vacía, el oráculo de cuatro salidas solo llega a 0,1824 en dev: también falta una candidata útil. El cuello de botella depende de la condición. |
| Presentar HYDRA como el único trabajo «más cercano» | Para un selector de candidatos es un antecedente importante; para el generador condicionado lo son AutoSAM, Automating MedSAM y Semantic AutoSAM. La cercanía depende de qué contribución se pretenda defender. |
| AutoSAM «equivale» a nuestro generador | Coinciden en aprender condicionamiento con SAM congelado, pero no son implementaciones equivalentes. AutoSAM usa un encoder adicional sobre RGB y prompts densos; aquí usamos cuatro vectores residuales desde features SAM. Citar antecedente y comparar diferencias antes de afirmar identidad o novedad. |
| Self-Prompting «equivale» al probe local | Comparten la idea. Difieren entrenamiento, partición, número de ejemplos y posproceso; el paper utiliza regresión logística y operaciones morfológicas, mientras la implementación local usa una capa lineal optimizada con Adam y filtrado de componentes. Etiquetarlo como baseline relacionado, no reproducción exacta. |
| El filtro de estabilidad pierde 0,29 «por los bordes difusos» | La pérdida entre oráculos de conjuntos sucesivos está medida; esa explicación causal no. Registrar hipótesis y medir relación con contraste/bordes, tamaño y margen de logits antes de atribuir la causa. |
| Llamar mecanística a la atención observada | Es evidencia descriptiva. No se intervino causalmente sobre la atención y deben declararse normalización, padding y semillas utilizadas. |
| La inicialización dev explica el resultado test E1 | En dev, los estáticos completos promedian 0,3633 frente a 0,3490 inicial, con diferencia pareada cuyo IC cruza cero. Eso no prueba la causa del 0,318 en otra partición/caché del E1. Comparar inicialización y entrenamiento sobre la misma ruta antes de atribuirlo. |
| Especialistas solo citados porque usan otra partición | No se pueden copiar sus cifras a una tabla propia. Sí se pueden reentrenar con nuestros mismos IDs y usarlos como controles automáticos. La partición diferente es un problema de protocolo, no una prohibición de comparación. |
| Repetir punto, multimáscara, oráculo y AMG idénticos para todos los modelos | Comprobar qué admite cada API y qué salidas se entrenaron. Registrar «no aplicable» si una condición no es comparable; no convertir un modo no supervisado de MedSAM en evaluación de su uso previsto. |
| Atribuir un patrón compartido por SAM v1 y v2 al diseño del decoder | Dos modelos completos cambian encoder, decoder y otros componentes. La comparación muestra transferencia del fenómeno; no identifica causalmente qué componente lo produce. |
| Suponer que la arquitectura común del decoder transfiere conclusiones entre ViT-B/L/H | Compartir estructura no garantiza compartir comportamiento: pesos, encoder y distribución de features cambian. La compatibilidad de una API no sustituye una evaluación experimental en cada checkpoint. |
| Confirmar decisiones nuevas otra vez en el mismo dev | Dev ya se utiliza para ajuste y selección. La confirmación requiere datos no usados para decidir, idealmente evaluación externa; el test E1 también fue inspeccionado. |

### Error concreto en la descripción de Medical SAM3

El texto revisado dice que SAM 3 original solo compite cuando recibe cajas GT. Sin embargo, **Medical SAM3, sección 4.2**, declara entrenamiento y evaluación solo con texto, sin puntos ni cajas. Su tabla 1 incluye Kvasir-SEG en el corpus de entrenamiento; la tabla 3 no informa una métrica específica para Kvasir. No se puede atribuirle una evaluación Kvasir con caja GT. [Fuente primaria](https://arxiv.org/html/2601.10880v1).

Ese artículo es distinto de **MedSAM3**, cuyo protocolo sí compara texto y texto con referencia espacial. Su tabla 2 reporta SAM 3 T = 0 y T+I = 0,7671 en Kvasir. Son resultados bajo el protocolo de esos autores, no una garantía para nuestros IDs o configuración. [MedSAM3, secciones 4.2–4.3](https://arxiv.org/html/2511.19046v1).

### Otros antecedentes que añadir

- [AutoSAM, sección de arquitectura](https://arxiv.org/html/2306.06370v1): el encoder adicional es HarDNet y produce condicionamiento denso. También evalúa Kvasir y SUN-SEG; no limitar su relación con pólipos únicamente a video. La descripción de cantidades por dataset en el texto presenta una asignación que requiere contrastar con sus datos/código, antes de adoptar sus splits literalmente.
- [Semantic AutoSAM](https://pubmed.ncbi.nlm.nih.gov/40040072/): atención cruzada para predecir prompts desde features; cercano al generador local.
- [Automating MedSAM](https://arxiv.org/abs/2409.20293): prompts aprendidos desde embeddings con supervisión débil y pocos ejemplos; coincide con una familia de ideas ya existente.
- [Stable-SAM](https://arxiv.org/abs/2311.15776): ya estudia inestabilidad ante cajas imprecisas y puntos insuficientes. Por ello, sensibilidad espacial no es por sí sola una contribución nueva.
- [IP-SAM, preprint 2026](https://arxiv.org/abs/2603.27250v2): otro antecedente de condicionamiento intrínseco y ausencia de prompts externos. No equivale a mantener todo SAM congelado.

## Cómo combinar ambas propuestas del equipo

El documento de la rama prioriza MedSAM/SAM 2.1/SAM 3 para transferir el diagnóstico. La [propuesta de comparación](estado_arte_y_comparacion_modelos.md) añade U-Net y Polyp-PVT para evaluar utilidad práctica frente a segmentadores automáticos. Son preguntas complementarias.

Alcance inicial propuesto: cerrar SAM ViT-B, añadir U-Net como control supervisado y SAM 2.1 para diagnóstico. Polyp-PVT y MedSAM pueden seguir según recursos; SAM 3 con texto es una extensión, que requiere protocolo distinto y cómputo validado. Esto conserva el foco del proyecto y permite una comparación externa concreta.

Los umbrales numéricos P-D1–P-D4 del documento son borradores, no resultados ni predicciones ya confirmadas. No se cambió la grilla del experimento SAM que se ha retomado. Las fechas de eventos/entrega citadas en ese documento no se verificaron aquí y deben contrastarse con la convocatoria oficial antes de planificar con ellas.

## Fuentes y alcance de esta revisión

Se contrastaron secciones de método/tablas de Zhou, AutoSAM, Self-Prompting, HYDRA, MedSAM3 y Medical SAM3, y resúmenes de PP-SAM, Stable-SAM y Automating MedSAM. Esto no certifica todas las referencias ni representa una lectura exhaustiva de los artículos y sus suplementos. Las diferencias entre protocolos deben registrarse antes de reutilizar scores publicados.

El archivo original se preserva en [related_work.md](related_work.md). Las acciones y ensayos que ya se iniciaron están en [experimentos_modelos.md](experimentos_modelos.md).
