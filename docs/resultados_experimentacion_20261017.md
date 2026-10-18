# Ampliación experimental completada: controles ligeros y prior espacial

Inicio: 16 de octubre; cierre de la ampliación de resolución: 17 de octubre de 2026, fecha del cliente. Las nueve corridas iniciales y las seis continuaciones U-Net terminaron con código 0. Se ejecutó la suite completa antes de la continuación: **22 pruebas aprobadas**. No hubo nueva evaluación de test, cambio de rama, commit ni push.

## Qué se añadió

Dos comparadores nuevos, **FCN-Lite** y **DS-U-Net**, y una comparación completa de **U-Net compacta**. Son variantes locales declaradas; DS-U-Net pertenece a la familia U-shaped y no constituye una familia independiente de U-Net. No se presentan como reproducciones exactas de FCN-8s/MobileNet/U-Net originales ni como especialistas modernos de pólipos.

Las tres redes usan los mismos 400 train/100 dev, tres semillas, RGB 128, ancho inicial 8, GroupNorm, sin preentrenamiento y una receta común fijada antes de entrenar. Se conservaron todas las corridas, checkpoints, métricas por imagen y fuentes. Protocolo en [protocolo_comparacion_ligera.md](protocolo_comparacion_ligera.md).

Se añadieron también auditoría de similitud, reanudación con estado completo, control espacial fijo aprendido solo de train, comparación emparejada por ID, subgrupos, curvas y panel cualitativo con una regla de selección previa. El costo se compara como latencia local del pipeline completo, separado de tiempo humano.

## Calidad de desarrollo

| Método | Dice [IC 95 % por imagen] | DE semillas | Parámetros entrenables |
| --- | --- | --- | ---: |
| FCN-Lite | 0,6002 [0,5549; 0,6429] | 0,0116 | 18.355 |
| DS-U-Net | 0,6548 [0,6122; 0,6946] | 0,0150 | 18.665 |
| U-Net compacta | 0,7814 [0,7447; 0,8140] | 0,0109 | 487.289 |
| Probe SAM | 0,7489 [0,7067; 0,7879] | 0,0007 | 257 |
| Probe→caja+punto→SAM | 0,8179 [0,7707; 0,8600] | 0,0053 | 257 + SAM congelado |

Los intervalos de esta tabla usan el cálculo común de 10.000 remuestreos de imágenes, semilla 0, después de promediar las tres semillas de entrenamiento. El informe RGB inicial usa 3.000 remuestreos, semilla 1234: sus límites pueden diferir ligeramente por Monte Carlo, sin cambiar las medias. La DE entre semillas se muestra por separado.

**La ventaja media de SAM automático sobre U-Net es +0,03648, IC pareado [−0,00554; 0,07712].** El intervalo incluye cero: este análisis no demuestra una mejora positiva de la media y tampoco demuestra equivalencia. El p de Wilcoxon disponible en el JSON responde a una hipótesis de rangos/distribución distinta del IC bootstrap de la media; no debe usarse para declarar significativa esa misma diferencia media. Son análisis exploratorios sin ajuste por selección ni multiplicidad.

La U-Net compacta alcanza una calidad cercana al pipeline SAM en este presupuesto local. Es una razón para conservar la comparación directa y medir costo, no para afirmar una superioridad universal de alguno. Los 257 parámetros del probe excluyen los millones de parámetros congelados de SAM: no representan tamaño ni costo total del sistema.

## Continuación U-Net: más entrenamiento y resolución

Se fijó un [protocolo nuevo](protocolo_resolucion_unet_20261017.md) antes de continuar. Para cada semilla, dos brazos parten del mismo estado de la época 30 —pesos, Adam y RNG— y realizan exactamente 15 épocas adicionales a 128 o 256 píxeles. El endpoint fijo es el análisis principal; la selección por Dice dev es secundaria. La selección se reinicia simétricamente en ambos brazos porque un score histórico medido a 128 no es comparable para elegir pesos evaluados a 256.

| U-Net | Dice dev [IC 95 % por imagen] | DE semillas |
| --- | --- | ---: |
| Mejor anterior, 30 épocas/128 | 0,7814 [0,7447; 0,8140] | 0,0109 |
| Inicial compartido, evaluado a 128 | 0,7786 [0,7419; 0,8115] | 0,0133 |
| Endpoint 45 épocas/128 | 0,7955 [0,7577; 0,8292] | 0,0276 |
| Seleccionado 30–45/128 | 0,8214 [0,7870; 0,8517] | 0,0222 |
| Inicial compartido, evaluado a 256 | 0,7033 [0,6672; 0,7369] | 0,0100 |
| Endpoint 45 épocas/256 | **0,8267 [0,7944; 0,8546]** | 0,0106 |
| Seleccionado 30–45/256 | **0,8272 [0,7955; 0,8547]** | 0,0104 |

Entrenar 15 épocas más mejora el endpoint 128 sobre su inicio en +0,0169 [0,0017; 0,0321]. En 256, la mejora desde el mismo estado evaluado a esa resolución es +0,1235 [0,0991; 0,1494]: el modelo inicialmente pierde calidad al cambiar de escala y luego se readapta. Con igual número de actualizaciones adicionales, el endpoint 256 supera al endpoint 128 en +0,0313 [0,0122; 0,0513]. Esto demuestra sensibilidad favorable a resolución dentro de esta continuación, no el efecto de entrenar ambas recetas desde cero.

SAM automático menos U-Net 256 da −0,0088 [−0,0472; 0,0281] para el endpoint fijo y −0,0093 [−0,0481; 0,0279] para el seleccionado. Ambos intervalos contienen cero: **en estas 100 imágenes dev no se demuestra una diferencia media entre SAM automático y la U-Net continuada**. Tampoco se demuestra equivalencia. Dos de tres corridas 256 seleccionan la última época, de modo que aún no se establece convergencia.

Resultados completos, scores por imagen y curvas: [informe de continuación](../results/model_comparison/unet_resolution_20261017/report.md), [figura](../results/model_comparison/unet_resolution_20261017/curves.png), [JSON de la continuación](../results/model_comparison/unet_resolution_20261017/summary.json) y [contrastes reproducibles con SAM](../results/model_comparison/unet_resolution_20261017/comparison_with_sam.json).

## Eficiencia de etiquetas: 25/100/400 máscaras

Se fijaron tres subconjuntos anidados por semilla y un número constante de actualizaciones para separar cantidad de etiquetas de cantidad de cómputo: 300 pasos para el probe y 1500 para U-Net RGB 256 entrenada desde cero. Cada celda presupuesto–semilla usa los mismos IDs en los métodos; `dev100` permanece fijo y no se consultó test. El endpoint fijo es primario y el checkpoint U-Net seleccionado en desarrollo es secundario.

| Método | 25 máscaras | 100 máscaras | 400 máscaras |
| --- | ---: | ---: | ---: |
| Probe SAM | 0,7156 | 0,7416 | 0,7488 |
| Probe→caja+punto→SAM | **0,7821** | **0,8061** | **0,8190** |
| U-Net endpoint fijo | 0,5103 | 0,6198 | 0,6964 |
| U-Net seleccionada en dev | 0,5254 | 0,6338 | 0,7029 |

SAM automático menos U-Net endpoint es +0,2718 [0,2185; 0,3243] con 25 máscaras, +0,1863 [0,1333; 0,2390] con 100 y +0,1226 [0,0757; 0,1692] con 400. La brecha disminuye al añadir etiquetas, mientras U-Net mejora +0,1096 de 25→100 y +0,0766 de 100→400. El probe satura antes: su incremento 100→400 es +0,0071 [−0,0024; 0,0159].

El decoder condicionado añade aproximadamente 0,065–0,070 Dice sobre el probe en los tres presupuestos, con intervalos pareados que excluyen cero. Esto apoya dos contribuciones distintas del preentrenamiento: representación espacial útil con pocas máscaras y refinamiento adicional una vez convertido el mapa del probe en prompts.

La conclusión está limitada a **desarrollo y cómputo fijo**. No es un contraste puro de arquitecturas ni prueba que SAM sea mejor cuando cada modelo se entrena hasta convergencia. De hecho, la continuación U-Net 128→256 alcanza 0,8267 con otra trayectoria y más aprendizaje acumulado. Ambos resultados juntos muestran el intercambio relevante: SAM arranca con mejor eficiencia de etiquetas, mientras una red directa mucho más barata puede cerrar la brecha con suficientes datos y entrenamiento.

Artefactos: [protocolo](protocolo_eficiencia_etiquetas_20261017.md), [informe](../results/model_comparison/label_efficiency_20261017/report.md), [curva](../results/model_comparison/label_efficiency_20261017/label_efficiency.png) y [resumen JSON](../results/model_comparison/label_efficiency_20261017/summary.json).

## Validación externa congelada en CVC-ClinicDB

Los modelos y umbrales seleccionados únicamente en Kvasir-SEG se evaluaron sin adaptación en los 612 frames de CVC-ClinicDB. El conjunto representa 29 secuencias; por ello los IC remuestrean secuencias completas y no frames independientes. La auditoría encontró los 612 pares esperados, ninguna máscara vacía, 612 imágenes únicas y cero coincidencias exactas de píxeles con las 620 imágenes Kvasir utilizadas.

| Método congelado | Dice por frame | IC 95 % agrupado por secuencia | Macro-secuencia | DE semillas |
| --- | ---: | ---: | ---: | ---: |
| Probe SAM | 0,4243 | [0,3624; 0,4892] | 0,4097 | 0,0077 |
| Probe→caja+punto→SAM | **0,4587** | [0,3793; 0,5405] | **0,4472** | 0,0142 |
| U-Net 256 endpoint | 0,3405 | [0,2841; 0,4002] | 0,3242 | 0,0729 |

SAM automático supera a U-Net por +0,1182 [0,0672; 0,1684] y al probe por +0,0344 [0,0110; 0,0576]. Los contrastes son pareados por frame y sus IC se agrupan por secuencia. La ventaja relativa externa apoya que las representaciones preentrenadas y el refinamiento condicionado transfieren mejor que el control RGB local.

La calidad absoluta cae con fuerza frente a Kvasir dev: SAM automático baja de 0,8179 a 0,4587 y U-Net de 0,8267 a 0,3405. No se debe describir el resultado como generalización clínica satisfactoria. La gran DE externa de U-Net procede de medias por semilla 0,3057/0,4243/0,2915 pese a rendimientos Kvasir cercanos, señal compatible con soluciones dependientes de inicialización que explotan regularidades del dominio fuente. Es una interpretación, no una identificación causal del atajo.

CVC-ClinicDB sólo contiene frames con pólipos; esta evaluación no mide especificidad ni falsos positivos en negativos. Esa limitación se estudia en el control HyperKvasir siguiente.

Artefactos: [protocolo](protocolo_validacion_externa_20261017.md), [informe](../results/model_comparison/external_clinicdb_20261017/report.md), [resumen JSON](../results/model_comparison/external_clinicdb_20261017/summary.json) y métricas por frame en `results/model_comparison/external_clinicdb_20261017/per_frame.csv`.

## Control negativo externo en HyperKvasir

Se congelaron los mismos modelos y umbrales de Kvasir-SEG y se aplicaron, sin ajuste externo, a las **1.009 imágenes** de la clase anatómica `cecum` de HyperKvasir. La descarga selectiva desde OSF verificó el SHA-256 publicado de cada archivo; la auditoría encontró 1.009 imágenes decodificables y 1.009 contenidos de píxeles únicos. Esta clase funciona como control negativo de nivel imagen para pólipo, pero no proporciona máscaras negativas revisadas píxel a píxel ni IDs públicos de paciente/video.

| Método congelado | Área falsa media [IC 95 %] | P95 área | Tasa área ≥1 % | P95 componente mayor | Salida vacía |
| --- | ---: | ---: | ---: | ---: | ---: |
| Probe SAM | 0,0581 [0,0552; 0,0611] | 0,1552 | 0,8794 | 0,1196 | 0,0000 |
| Probe→caja+punto→SAM | **0,0521 [0,0488; 0,0554]** | 0,1631 | **0,7942** | 0,1628 | 0,0000 |
| U-Net 256 endpoint | 0,1733 [0,1690; 0,1776] | 0,2922 | 0,9931 | 0,2814 | 0,0017 |

SAM automático reduce el área falsa media frente a U-Net en **−0,1212 [−0,1267; −0,1159]** y la tasa de área falsa ≥1 % en −0,1989 [−0,2217; −0,1767]. También reduce ambas medidas frente al probe: −0,0061 [−0,0076; −0,0045] de área y −0,0852 [−0,1008; −0,0704] de tasa ≥1 %. Por tanto, el refinamiento condicionado elimina parte del ruido del localizador y resulta mucho más conservador que U-Net bajo cambio de dominio.

El resultado revela una limitación fuerte: probe y SAM automático producen alguna región en el 100 % de las imágenes; U-Net lo hace en 99,83 %. La métrica binaria de máscara vacía oculta la diferencia de magnitud, por eso el área y el componente mayor fueron preespecificados como resultados principales. Este control no autoriza afirmar especificidad clínica: los IC remuestrean imágenes y no pueden corregir dependencia entre imágenes del mismo paciente.

Artefactos: [protocolo](protocolo_especificidad_negativos_20261017.md), [informe](../results/model_comparison/negative_specificity_20261017/report.md), [resumen JSON](../results/model_comparison/negative_specificity_20261017/summary.json) y las 1.009 filas en `results/model_comparison/negative_specificity_20261017/per_image.csv`.

## Qué revela el control espacial fijo

Se obtiene una caja y un punto típicos a partir de medianas de coordenadas normalizadas de las 400 GT train. Se aplica ese mismo prior a cada imagen dev usando solo sus dimensiones: no se consulta GT dev para producir prompts. Punto fijo: Dice 0,3633; caja fija: 0,2969; caja+punto fijos: **0,3582**.

El pipeline probe→caja+punto supera a caja+punto fijos en **0,45974**, IC pareado [0,38087; 0,53697]. Ambos hacen una consulta de caja+punto por imagen y usan el mismo SAM congelado. Esto respalda el valor de localizar desde la imagen frente a usar un prior típico, bajo el protocolo observado. No mide ahorro humano ni aísla capacidad/optimización del generador de cuatro tokens.

El promedio de tokens estáticos (0,36331) y punto fijo (0,36335) es casi idéntico. La diferencia media tiene IC [−0,05109; 0,05359]: no es una prueba de equivalencia ni demuestra que produzcan las mismas máscaras. Es un control útil para evitar atribuir a entrenamiento una recuperación que un prior espacial sencillo también consigue en promedio.

Fuente: [informe del prior](../results/model_comparison/fixed_prior_20261016/report.md).

## Dónde aparecen diferencias

Se reutilizaron los cortes definidos en Fase B; no se buscaron nuevos umbrales según estos scores.

| Área GT relativa | n | U-Net compacta | SAM automático |
| --- | ---: | ---: | ---: |
| <5 % | 25 | 0,7145 | 0,7991 |
| 5–15 % | 38 | 0,8152 | 0,8776 |
| ≥15 % | 37 | 0,7920 | 0,7693 |

Estas medias descriptivas muestran heterogeneidad: la media SAM es mayor en los dos primeros grupos; la media U-Net es mayor en el grupo de mayor área. No se establece una interacción causal ni una regla para elegir modelo por paciente. Los grupos de borde/componentes se solapan y el de múltiples componentes tiene solo cinco imágenes.

El [panel cualitativo](../results/model_comparison/comparison_train400_20261017/qualitative_first4.png) usa los primeros cuatro IDs dev y semilla 0. Muestra regiones adicionales y cobertura parcial en varias predicciones; algunos ejemplos U-Net se ajustan mejor que los comparadores de menor capacidad, con errores todavía visibles. Son ilustraciones, no una estimación de frecuencia ni prueba de una causa visual.

## Qué mejoró en reproducibilidad

- Plan y hashes de código/datos fijados antes de entrenar; rechazo de reanudación si cambian. No se modificó ninguno de los siete archivos congelados de Fase B.
- Checkpoints con pesos, Adam y RNG Python/NumPy/Torch/orden de muestras. La prueba de restauración reproduce exactamente la siguiente actualización.
- Los nueve checkpoints seleccionados reprodujeron sus 100 Dice individuales al recargar: **900 comprobaciones por imagen**, con tolerancia 1e−8.
- El comparador empareja por ID y rechaza duplicados, faltantes o imágenes extra; no depende del orden de filas CSV.
- Auditoría de archivos/píxeles y screening perceptual train400/dev100: sin duplicados exactos entre particiones ni candidatos DCT con distancia ≤4. No prueba independencia por paciente/video y no cambia las particiones.
- Se resolvió el fallo previo de temporales de Windows y la suite completa de 22 tests pasó. El test nuevo comprueba la herencia idéntica de pesos/Adam/RNG y la selección específica por resolución. Los pilotos negativos anteriores se conservan.

## Cómputo y artefactos

Las latencias RGB se reestimaron después de finalizar el entrenamiento y los tests. CPU/cuatro hilos, batch 1, cinco calentamientos y veinte repeticiones, imágenes decodificadas en RAM, preprocesado experimental+red+restauración+umbral. La comparación SAM incluye su encoder desde RGB, probe, componentes y decoder; no se equipara el tiempo de decoder sobre caché con el del sistema completo.

| Sistema | Mediana del pipeline completo |
| --- | ---: |
| FCN-Lite | 6,86 ms |
| DS-U-Net | 14,08 ms |
| U-Net compacta | 12,11 ms |
| U-Net continuada, RGB 128 | 13,30 ms |
| U-Net continuada, RGB 256 | 27,08 ms |
| SAM + probe→caja+punto | 8.369,38 ms (8,37 s) |

SAM: p95 8,91 s, semilla 0, **93.735.729 parámetros totales únicos** contando probe, frente a 487.289 de U-Net compacta. El encoder SAM tiene mediana 8.243,85 ms; domina el costo. Las medianas de componentes no tienen por qué sumar la mediana total. Los RGB resumen mediciones separadas de tres semillas; no se midió una distribución sobre todas las máquinas/semillas posibles.

Esto favorece considerar un segmentador directo compacto para el escenario automático de este dataset, mientras se amplían entrenamiento y evaluación independiente. No prueba que U-Net sea universalmente preferible ni que el factor de latencia se transfiera a GPU/otras implementaciones. DS-U-Net tiene menos parámetros que U-Net, pero no fue más rápida en esta implementación CPU; los parámetros no sustituyen el perfil medido.

Se comprobaron las veinte predicciones del pipeline de tiempo usando caché y las veinticinco ejecuciones desde RGB, reproduciendo los scores de referencia a tolerancia 1e−8. Se verificó también que el SHA-256 del checkpoint SAM cargado coincide con la procedencia de la caché.

Calidad, diferencias pareadas y subgrupos: [informe común](../results/model_comparison/comparison_train400_20261017/report.md). Costo completo integrado: [informe de calidad y costo](../results/model_comparison/comparison_train400_20261017/report_with_runtime.md), [figura](../results/model_comparison/comparison_train400_20261017/quality_vs_latency.png) y [JSON con mediciones](../results/model_comparison/comparison_train400_20261017/summary_with_runtime.json). Se conservó por separado el informe de calidad y el [perfil SAM](../results/model_comparison/sam_full_runtime_20261017/summary.json).

Los pesos/estados completos están en `runs/model_comparison/light_train400_20261016/`, ignorado por Git. Las copias ligeras de nueve corridas están en `results/model_comparison/light_train400_20261016/`. Subir los resultados ligeros no sube automáticamente los pesos.

## Límites y siguiente decisión científica

SAM usa preentrenamiento y entrada 1024; las redes RGB parten de cero y usan 128 o continúan a 256. Se igualan los IDs y presupuesto de máscaras locales, no todo el conocimiento previo ni la resolución. La comparación actual es un control práctico bajo restricciones de CPU.

Las épocas elegidas inicialmente fueron FCN-Lite 30/29/28, DS-U-Net 30/30/30 y U-Net 30/30/28. En la continuación, U-Net 256 elige 44/45/45 en total. **No se estableció convergencia ni techo de desempeño**. La nueva evidencia elimina la aparente ventaja media puntual de SAM que quedaba con 30 épocas, pero no demuestra superioridad ni equivalencia entre sistemas.

La prioridad siguiente es:

1. Estudiar eficiencia de etiquetas con subconjuntos anidados train25/train100/train400 para probe/SAM y U-Net. Eso conecta la mitigación con el presupuesto de supervisión; cantidad de máscaras sigue sin equivaler a tiempo humano medido.
2. Evaluar recetas congeladas en un dataset externo sin exposición/adaptación previa verificable. El test histórico ya se inspeccionó y dev guió ajustes. No hay evidencia clínica de generalización ni protocolo de negativos sin pólipos.
3. Añadir SAM 2.1 para transferencia del diagnóstico y un especialista reentrenado con IDs propios para el artículo ampliado. El control entrenado del generador de igual capacidad sigue pendiente si se quiere defender su mecanismo.

La contribución más defendible continúa siendo **diagnosticar la dependencia de información espacial, distinguir candidatas/selección y evaluar calidad/costo de sustituir prompts humanos en inferencia**. Generar prompts automáticamente ya tiene antecedentes; estos resultados no acreditan novedad arquitectónica ni estado del arte.

La [revisión nueva del abstract](waimlap_2026/revision_20261017_resolution/Abstract_TheInformationalCostOfSpatialPromptsInSAM.pdf) incorpora la continuación en dos páginas verificadas y conserva cabeceras, logos y pie de la plantilla. Se mantiene también la revisión anterior. La autoría sigue pendiente de confirmación; no es un envío realizado.
