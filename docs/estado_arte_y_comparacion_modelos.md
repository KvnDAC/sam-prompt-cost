# Estado del arte dirigido y propuesta de comparación en Kvasir-SEG

Fecha de consulta: **16 de octubre de 2026**. Este documento propone experimentos; no presenta nuevos modelos como ya evaluados. Se revisaron fuentes primarias y repositorios de autores. Es una selección dirigida a la pregunta del proyecto, no una revisión sistemática exhaustiva ni un ranking de los mejores modelos de 2026.

**Estado posterior, 17 de octubre:** tres redes RGB compactas ya completaron train400/dev100 y tres semillas. [Resultados y límites](resultados_experimentacion_20261017.md). U-Net dejó de tener solo pilotos; SAM 2.1/MedSAM/Polyp-PVT siguen pendientes. Las secciones de diagnóstico inicial siguientes conservan la situación del 16.

**Revisión actualizada tras la continuación U-Net:** seis corridas adicionales 128/256 finalizaron. U-Net 256 obtiene 0,8267 en el endpoint fijo frente a 0,8179 del pipeline SAM automático; la diferencia pareada es compatible con cero. La utilidad de la comparación, los antecedentes publicados hasta 2026, el diferencial defendible y el nuevo orden experimental están en [nota_comparacion_y_estado_arte_20261017.md](nota_comparacion_y_estado_arte_20261017.md). Esta nota sustituye las recomendaciones operativas antiguas de este documento; el resto se conserva como registro de la planificación inicial.

**Actualización posterior de la misma jornada:** se incorporó el documento de la rama `cambios-analisis`, se retomó Fase B y se implementó un baseline U-Net con pilotos técnicos. La descripción del estado inicial que sigue es anterior a esas acciones; resultados y alcance actuales en [experimentos_modelos.md](experimentos_modelos.md), y revisión de la bibliografía del equipo en [revision_related_work.md](revision_related_work.md).

## 1. Qué se ha probado realmente

La evidencia del repositorio corresponde a **SAM v1 ViT-B** con distintas formas de condicionamiento: sin prompt espacial, puntos/cajas GT, clics, AMG, tokens estáticos y generador condicionado a la imagen. También hay un probe lineal entrenado sobre embeddings congelados, con variantes que generan caja y punto automáticamente.

El probe es otro modelo entrenable pequeño, pero no constituye una comparación con otra familia de segmentadores completos. No se encontraron implementaciones y resultados propios de U-Net, PraNet, Polyp-PVT, MedSAM, SAM 2.1 o SAM 3. El generador condicionado sigue siendo una intervención sobre SAM ViT-B y su comparación final está incompleta. El estado verificado está en [revision_sam_20261016.md](revision_sam_20261016.md).

El design log menciona alternativas y el README anuncia related work inicial, pero no se encontró un documento dedicado de estado del arte en los archivos disponibles de este checkout. Eso no descarta bibliografía en un entregable externo al repositorio.

## 2. Recomendación central

Conservar el objetivo de **caracterizar el costo informativo del prompt y evaluar una mitigación automática**, añadiendo dos preguntas:

1. ¿La sensibilidad al prompt se reproduce en otra arquitectura SAM?
2. ¿La mitigación automática ofrece calidad y costo de cómputo útiles frente a segmentadores supervisados que no necesitan prompts en inferencia?

Mi selección mínima es **SAM ViT-B + U-Net + Polyp-PVT + SAM 2.1 Hiera-Tiny**, junto con los controles internos ya presentes: probe, probe→caja+punto y tokens estáticos/condicionados. PraNet es una alternativa a Polyp-PVT si su integración resultara más viable; no hace falta incorporar ambos en la primera ronda. Esta selección es una decisión metodológica para cubrir familias, no una afirmación de superioridad de alguno en nuestra partición.

## 3. Mapa de modelos y antecedentes

| Modelo o trabajo | Familia y relación con la pregunta | Uso propuesto y estado de evidencia |
|---|---|---|
| [U-Net, 2015](https://arxiv.org/abs/1505.04597) | CNN encoder–decoder para segmentación biomédica; no recibe prompts espaciales por imagen | Baseline supervisado simple. Implementar una configuración explícita y entrenar con nuestros IDs; no copiar resultados de otros datasets. |
| [PraNet, MICCAI 2020](https://github.com/DengPingFan/PraNet) | Segmentación especializada en pólipos con atención inversa | Referencia histórica del dominio y alternativa práctica para la comparación. El repositorio incluye código, entrenamiento y pesos enlazados. |
| [Polyp-PVT, AIR 2023](https://github.com/DengPingFan/Polyp-PVT) | Segmentador especializado basado en Pyramid Vision Transformer | Baseline principal del dominio. El repositorio ofrece instrucciones de entrenamiento, inicialización y evaluación. No se verificó su ejecución en nuestro entorno. |
| [SAM 2 / SAM 2.1, 2024](https://github.com/facebookresearch/sam2) | Segmentación con prompts en imágenes y videos; arquitectura distinta de SAM v1 | Evaluar imágenes independientes con SAM 2.1 Hiera-Tiny. El código oficial ofrece predictor de imágenes y checkpoints. Requiere un adaptador propio; no reutilizar directamente la caché ViT-B. |
| [MedSAM](https://github.com/bowang-lab/MedSAM) | Adaptación médica de SAM; la interfaz documentada usa cajas para señalar el objetivo | Extensión para estudiar adaptación de dominio con una misma caja. No asumir que sea automático: para ese escenario necesita un generador de cajas. Auditar datos del checkpoint antes de llamarlo evaluación sin exposición al test. |
| [Polyp-SAM, 2023](https://arxiv.org/abs/2305.00293) | Adaptación de SAM a segmentación de pólipos; estudia estrategias de ajuste | Antecedente temático. El [repositorio de los autores](https://github.com/ricklisz/Polyp-SAM) contiene scripts y un enlace a pesos; no se validó su reproducción ni su protocolo completo. |
| [AutoSAM, 2023](https://arxiv.org/abs/2306.06370) | Aprende un encoder que obtiene el condicionamiento de la propia imagen, propagando gradientes a través de SAM congelado | Antecedente directo de la mitigación. Debe compararse conceptualmente con nuestro generador. No se verificó aquí una instalación reproducible. |
| [Semantic AutoSAM, EMBC 2024](https://pubmed.ncbi.nlm.nih.gov/40040072/) | Predice embeddings de prompt desde características de imagen mediante un módulo ligero de atención cruzada | Antecedente especialmente cercano al generador implementado. Sus experimentos descritos son CT y ultrasonido; sus scores no son resultados en Kvasir-SEG. No se identificó un paquete oficial reproducible durante esta búsqueda. |
| [SAM-Mamba, WACV 2025](https://arxiv.org/abs/2412.08482) | Adaptación de representaciones SAM mediante un módulo Mamba orientado a pólipos | Incluir en la revisión reciente. En el [repositorio consultado](https://github.com/TapasKumarDutta1/SAM_Mamba_2025) el listado raíz muestra assets, licencia y README; no se validó una implementación entrenable. No hacerlo dependencia de la primera comparación. |
| [IP-SAM, preprint 2026, v2](https://arxiv.org/abs/2603.27250v2) | Condicionamiento por prompts intrínsecos en SAM2; incluye transferencia a segmentación de pólipos | Antecedente reciente del despliegue sin prompts externos. La adaptación incluye módulos y entrenamiento distintos de nuestro SAM totalmente congelado; no es un equivalente directo de cuatro tokens ligeros. Código y reproducción pendientes de verificar. |
| [SAM 3](https://github.com/facebookresearch/sam3) | Segmentación por conceptos mediante lenguaje y ejemplos, además de interacción visual | Extensión si se decide estudiar el valor de un prompt semántico fijo como «polyp». Ese texto aporta información: etiquetar la condición como «sin prompt espacial, con texto», no como ausencia total de prompts. |

### Implicación para la novedad

Generar prompts desde imágenes y usar atención cruzada **ya tiene antecedentes**. La cercanía con AutoSAM y Semantic AutoSAM obliga a describir nuestra implementación como una variante experimental hasta establecer diferencias verificables. No se ha demostrado novedad arquitectónica ni superioridad frente a esos trabajos.

Un aporte defendible del proyecto podría ser la combinación de ablación espacial, separación entre generación y selección, análisis de fallos, y una mitigación de bajo costo evaluada con controles en Kvasir-SEG. La transferencia de esas conclusiones a SAM 2.1 aportaría evidencia adicional. Son objetivos de investigación, no resultados ya alcanzados.

Para afirmar reproducción de un trabajo previo hay que contrastar arquitectura, entrenamiento, datos y evaluación con su publicación. Si se implementa una simplificación sin esos detalles, se debe llamar «baseline inspirado en…» y explicar las diferencias.

## 4. Dos comparaciones complementarias

### A. Diagnóstico del costo del prompt

Comparar SAM ViT-B y SAM 2.1 Tiny, con pesos congelados y las mismas imágenes. MedSAM puede añadirse en las condiciones que admite su implementación.

| Condición | Información proporcionada | Pregunta |
|---|---|---|
| Cero puntos/cajas/máscaras de entrada | Ninguna localización externa | ¿Qué produce el modelo sin condicionamiento espacial explícito? |
| Punto interior GT por componente elegible | Localización oráculo | ¿Cuánto recupera un punto correcto? |
| Caja GT por componente elegible | Localización y extensión oráculo | ¿Cuánto recupera información espacial más completa? |
| Perturbaciones espaciales actuales | Prompts GT degradados de forma controlada | ¿Qué tan estable es cada modelo ante errores? |
| Salida elegida por calidad frente a mejor candidata según GT | Selección realizable frente a oráculo | ¿Faltan candidatas útiles o falla la elección? |

La condición de entrada vacía exige revisar la API y el decoder de cada arquitectura. Conservar sus convenciones de objeto ausente y salida; si no admite una condición comparable, documentar «no aplicable». No inventar una caja completa ni un punto central para llamarlo «sin prompt». AMG usa puntos internos y constituye una condición automática distinta.

No asumir que los tokens de salida o los generadores implementados para SAM v1 se transfieren sin cambios a SAM 2.1. MedSAM con caja aporta una comparación de adaptación médica, pero no una tabla completa de puntos y entrada vacía si esas rutas no están soportadas.

### B. Segmentación automática de pólipos

En esta tabla, **ningún método puede usar GT de la imagen evaluada para producir o elegir su máscara**:

| Método | Supervisión específica de Kvasir | Inferencia automática |
|---|---|---|
| U-Net | Máscaras de train400 | Predicción directa |
| Polyp-PVT | Máscaras de los mismos IDs train400 | Predicción directa |
| Probe lineal | Máscaras de train400 sobre embeddings SAM congelados | Predicción directa |
| Probe→caja+punto→SAM ViT-B | Misma supervisión del probe | Prompts calculados de la predicción del probe |
| Tokens estáticos→SAM ViT-B | Máscaras de train400; incluye inicialización meanpoint | Vectores aprendidos fijos |
| Generador condicionado→SAM ViT-B | Máscaras de train400 | Vectores producidos por la imagen |
| SAM 2.1 + auto-prompts, extensión | Declarar el origen y entrenamiento del generador | Prompts automáticos, sin GT de evaluación |

Para aislar el efecto del segmentador se pueden reutilizar exactamente las mismas cajas/puntos predichos por un generador fijo en SAM v1 y SAM 2.1. En ese caso se debe declarar que el generador usa embeddings de SAM v1. Entrenar un generador nativo para cada arquitectura sería otra comparación del pipeline completo, con costo y supervisión propios.

Puntos/cajas GT y selección oráculo se presentan en la tabla de diagnóstico, identificados como referencias privilegiadas. No compiten como métodos automáticos en esta tabla.

## 5. Protocolo para que «el mismo dataset» sea una comparación justa

1. **Mismos IDs:** primera ronda con train400, dev100 y test120 existentes. Los otros 380 IDs train quedan fuera de esa ronda. Un estudio posterior con train780 exige repetir los métodos comparados con ese presupuesto de etiquetas.
2. **Datos de entrenamiento verificables:** registrar preentrenamiento genérico, adaptación médica y uso de otros datasets de pólipos por separado. Usar inicialización genérica cuando sea posible y entrenar los baselines con nuestra partición. Un checkpoint especializado publicado no es automáticamente válido para nuestro test.
3. **Sin mezcla de particiones:** el protocolo público de PraNet usa Kvasir 900/100 y también datos de ClinicDB, según su repositorio. Nuestro protocolo no coincide. Esos checkpoints podrían haber visto IDs de nuestro test; las cifras del paper no son comparables directamente con las nuestras. [Fuente primaria del protocolo](https://github.com/DengPingFan/PraNet).
4. **Selección en dev:** LR, épocas, umbrales, posproceso y prompts textuales, si se incorporan, se eligen en dev. El test histórico E1 ya se inspeccionó: las nuevas conclusiones son exploratorias y conviene añadir una evaluación externa no usada para decisiones.
5. **Métrica común:** GT JPEG binarizada a 127, predicción llevada a resolución original, mismo cálculo de Dice/IoU por imagen y media de imágenes. Respetar la normalización y entrada necesarias para cada modelo. No mezclar métricas de mapas normalizados o diferentes umbrales como si fueran equivalentes.
6. **Entrenamiento adecuado:** tres semillas y un presupuesto de búsqueda comparable. No imponer seis épocas a una U-Net desde cero solo porque los tokens actuales entrenan seis: dar a cada arquitectura una receta y criterio de parada razonables elegidos en dev. Registrar todo el cómputo invertido.
7. **Incertidumbre:** IC 95 % por bootstrap de imágenes, dispersión entre semillas y diferencias pareadas con objetivos principales definidos antes de las nuevas ejecuciones. Direcciones, componentes y máscaras candidatas no son muestras independientes adicionales.
8. **Costo real:** parámetros totales y entrenables, memoria máxima, tiempo de entrenamiento y latencia por imagen incluyendo encoder y generador. Misma máquina, precisión declarada, batch 1, calentamiento y repeticiones. No equiparar tiempo de decoder con latencia total ni tiempo computacional con tiempo humano de anotación.

Antes de una evaluación externa, auditar que sus imágenes no se usaron para adaptar los checkpoints. Revisar también duplicados y agrupaciones por paciente/video cuando haya metadatos; compartir un nombre de dataset no garantiza independencia clínica.

## 6. Qué hacer primero con los recursos actuales

La máquina observada es CPU. La implementación actual aprovecha embeddings congelados; entrenar U-Net y Polyp-PVT requiere procesar imágenes durante el entrenamiento y no reutiliza directamente esa caché. No se presupuestaron tiempos ni VRAM sin medirlos.

Orden propuesto:

1. **Cerrar Fase B actual:** conservar los estáticos completos, recuperar la etapa condicionada interrumpida según la guía y terminar probe/control de contexto. Esto proporciona el baseline automático que ya tiene trabajo invertido.
2. **Preparar el comparador común:** exportar por imagen predicción/métricas, IDs, checkpoint, semilla, supervisión y tiempos. Separar el código de cada modelo del cálculo de métricas.
3. **Piloto pequeño de U-Net:** verificar carga de datos, gradientes y reconstrucción de máscaras; un piloto no cuenta como benchmark terminado. Con GPU disponible, entrenar las tres semillas en el presupuesto común.
4. **SAM 2.1 Tiny:** empezar por inferencia congelada y prompts controlados. Crear un entorno separado; el repositorio oficial recomienda WSL/Ubuntu en Windows. Medir la viabilidad antes de ampliar a variantes grandes. [Instrucciones oficiales](https://github.com/facebookresearch/sam2).
5. **Polyp-PVT:** incorporar su arquitectura y entrenarla con los IDs propios, conservando por separado pesos de inicialización y pesos ajustados. Su entorno y dependencias deben validarse antes de prometer reproducción.
6. **Comparación con self-prompting:** convertir la lectura de AutoSAM/Semantic AutoSAM en una lista de diferencias de arquitectura y supervisión. Incluir un baseline reproducible si se consigue código o detalles suficientes. Para publicar una propuesta de generador, esta comparación es más relevante que añadir muchas variantes genéricas.
7. **Extensiones según resultados:** MedSAM con cajas, SAM-Mamba/IP-SAM como reproducciones verificadas, o SAM 3 con texto fijo. Cambiar la pregunta a lenguaje requiere explicitar la información semántica aportada por ese prompt.

Los modelos nuevos requieren una carpeta y un protocolo nuevos; no modificar el `plan.json` de `full_dev_20261012` ni sobrescribir sus resultados. No se instalaron dependencias, descargaron pesos ni ejecutaron estos nuevos entrenamientos como parte de esta revisión bibliográfica.

## 7. Qué llevar a la reunión con el investigador

Propuesta de alcance: **caracterización de dependencia espacial en dos arquitecturas SAM, comparada con segmentación automática supervisada en Kvasir-SEG, y evaluación de una mitigación ligera sin humano en inferencia**.

Entregables concretos: matriz de antecedentes con fuentes y condiciones; tabla automática con partición común; curvas de sensibilidad a prompts para SAM v1/v2.1; análisis por tamaño y errores de selección; tabla calidad–costo y una evaluación externa si los datos y el tiempo lo permiten.

La respuesta actual a «¿ya se compararon distintos modelos?» es: **se compararon formas de usar SAM ViT-B y un probe; la comparación con otras familias todavía debe realizarse**. La recomendación del investigador añade una evaluación que falta y ayuda a situar la mitigación frente a alternativas reales.
