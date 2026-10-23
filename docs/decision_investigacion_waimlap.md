# Decisión de investigación y cierre para WAIMLAp 2026

Revisión del 16 de octubre de 2026. Este documento integra el estado local terminado, las ramas disponibles y la plantilla proporcionada. No ejecuta una nueva comparación de modelos ni presenta resultados pendientes como terminados.

**Actualización del 17:** terminó la comparación ligera de FCN-Lite, DS-U-Net y U-Net compacta, con tres semillas y train400/dev100. Se añadió un prior espacial de train y pasaron 21 tests. [Nuevo análisis científico](resultados_experimentacion_20261017.md): la diferencia media SAM automático menos U-Net es +0,0365, IC pareado [−0,0055; 0,0771]. Las propuestas y referencias a entrenamiento pendiente del cuerpo original describen el corte del 16; quedan pendientes convergencia/resolución, datos externos y comparación SAM 2.1/especialista.

## Dictamen

**La idea merece continuar como estudio controlado de dependencia del condicionamiento espacial y mitigación automática en segmentación de pólipos.** Ya hay evidencia suficiente para redactar un resumen extendido de trabajo en curso con resultados reales. El proyecto ampliado de comparación entre familias todavía no está completo.

La pregunta recomendada es: **¿qué calidad pierde SAM al retirar o degradar la localización del objetivo, qué parte del fallo corresponde a candidatas y selección, y cuánto recupera una localización automática aprendida con el mismo presupuesto de etiquetas?**

Probar más modelos sirve si cada uno responde una pregunta concreta. Una tabla extensa de Dice sobre Kvasir no establece por sí sola una contribución. Tampoco basta mostrar que SAM, diseñado para recibir prompts, pierde calidad cuando se elimina ese condicionamiento. El valor está en los controles que distinguen situaciones, los límites de las explicaciones y la mitigación evaluada.

Título propuesto: **The Informational Cost of Spatial Prompts in SAM: A Controlled Study on Polyp Segmentation**. No anuncia una nueva arquitectura, superioridad general ni medición de tiempo humano.

## Plantilla y convocatoria verificadas

La plantilla local exige inglés, un máximo de dos páginas incluyendo referencias, las secciones Introduction, Methodology, Results and Discussion y Conclusions, PDF final y conservar la cabecera WAIMLAp/PUCP. El borrador Word parte de ese archivo; el original no se modifica.

La [convocatoria oficial de pósteres](https://waimlap.org/registration/) admite trabajos en curso y resultados preliminares. Para esta modalidad solicita también póster A0 vertical en inglés, ambos PDF en un ZIP de hasta 10 MB. La fecha publicada para envío es **16 de octubre de 2026**, coincidente con la fecha del cliente. No confundir esta modalidad con la convocatoria previa de artículos completos. No se verificó una hora de cierre y no se envió nada.

## Revisión de todas las ramas disponibles

Se actualizaron referencias con `git fetch origin`, sin cambiar de rama, hacer merge ni sobrescribir cambios locales. Se inspeccionaron los árboles versionados y los textos de README, documentos y celdas Markdown de notebooks en las referencias disponibles.

| Referencia | Commit | Qué aporta |
| --- | --- | --- |
| `main` local | `ab30c46` | Fotografía E1, 91 archivos versionados |
| `fix/pruebasExperimentos` local, HEAD | `dc88ba5` | Base actual, más los cambios y resultados locales sin commit |
| `origin/main` | `dc88ba5` | Base versionada con Fase B parcial y ajuste de rutas |
| `origin/fix/pruebasExperimentos` | `51185c6` | Su árbol es idéntico al de `origin/main`; distinto historial |
| `origin/cambios-analisis` | `ca03c31` | Añade related work y notas en README/design log respecto a main |

`origin/HEAD` es un alias de `origin/main`. No se encontraron otras ramas en las referencias descargadas. Los resultados completos recientes y U-Net están en el workspace; no aparecen automáticamente en los commits remotos. No se encontró otra familia ya evaluada en las ramas.

**Autoría:** el historial identifica a Alessandro Santé (`ssante@pucp.edu.pe`) y al usuario de commits ChrisRosales (`a20201294@pucp.edu.pe`), además del commit inicial firmado como Equipo INF659. VinKzzlo utiliza el mismo correo que Alessandro. Son metadatos de contribuciones, no una declaración de autores del artículo: no permiten completar el nombre legal de ChrisRosales, identificar a todos los integrantes, fijar orden de autores ni confirmar afiliación y autor de correspondencia. No se encontró una lista formal en los documentos o notebooks de ninguna rama. Por eso el borrador conserva campos de autoría pendientes; no se inventan integrantes a partir del propietario del repositorio o correos de commits.

## Qué experimentos existen y qué significan

### E1 histórico: 120 imágenes de test oficial

El [README](../README.md) y [resumen E1](../results/summary_main.json) conservan la evaluación inicial: sin prompt/token 0, Dice 0,0122; AMG con mayor IoU predicho, 0,058; mejor propuesta AMG cruda según GT, 0,895; tokens estáticos k=4, 0,3178; probe→caja+punto, 0,655; punto GT, 0,696; caja GT, 0,886. Son referencias históricas con otra ruta de precisión/caché; no se mezclan con Fase B para calcular mejoras.

También hay simulación iterativa de clics, arquitectura/cómputo teórico, campo receptivo, PCA de features, atención y calibración/filtrado AMG. NoC@IoU 0,85 es 3,48 clics y NoC@0,90 es 5,32, con límite de 20 y fallos que deben informarse. Es interacción simulada usando GT, no observación de anotadores. Los mapas de atención y PCA aportan descripción, no una causa identificada del fallo. El lift de atención requiere revisar su normalización y padding antes de usarlo como argumento cuantitativo; ver [auditoría técnica](analisis_experto.md).

AMG consulta una rejilla de puntos: puede ser automático sin interacción humana y seguir usando prompts espaciales. En ese conjunto grande de propuestas existe una buena candidata oráculo, pero los filtros y la selección automática pierden calidad. Esa conclusión no se traslada a la entrada realmente vacía del decoder.

### Fase B terminada: 400 train y 100 dev

SAM ViT-B congelado, encoder y caché FP32. Las comparaciones aprendidas tienen tres semillas; época, LR y umbrales se seleccionan con dev. Las 16 etapas terminaron. Los siete hashes congelados del plan coinciden. No hubo nueva evaluación de test.

| Método | Dice dev [IC 95 % por imagen] | Lectura |
| --- | --- | --- |
| Entrada sin prompt, salida única | 0,0089 [0,0007; 0,0227] | Fallo severo en esta condición de ablación |
| Punto GT | 0,6831 [0,6155; 0,7463] | Una localización interior aporta información útil |
| Caja GT | 0,9352 [0,9279; 0,9420] | Referencia con extensión espacial privilegiada |
| Tokens estáticos k=4 | 0,3633 [0,2913; 0,4375] | Recuperación parcial; efecto pequeño sobre inicialización |
| Tokens condicionados k=4 | 0,5013 [0,4260; 0,5762] | Mejora frente al estático, con explicaciones alternativas |
| Probe lineal | 0,7489 [0,7067; 0,7879] | Features congeladas permiten extraer señal de segmentación |
| Probe→caja→SAM | 0,7738 [0,7271; 0,8170] | Mitigación automática usando localización aprendida |
| Probe→caja+punto→SAM | 0,8179 [0,7707; 0,8600] | Mejor pipeline automático entre los evaluados aquí |

Los IC remuestrean imágenes después de promediar semillas; no son IC de independencia por paciente. La diferencia pareada caja+punto menos probe es +0,0690 [0,0471; 0,0919]. El probe tiene 257 parámetros entrenables; utiliza todo el encoder SAM, por lo que eso no equivale a un pipeline total de 257 parámetros. El estático tiene 1.024 y el condicionado 331.008 parámetros entrenables. Todos usan supervisión de máscaras en train; meanpoint también utiliza GT de train. No son métodos sin supervisión.

La recuperación normalizada `(Dice_auto − Dice_vacío)/(Dice_caja_GT − Dice_vacío)` da 87,3 % para caja+punto. Es recuperación respecto de esas dos referencias en esta partición; no significa ahorro del 87,3 % del tiempo humano, exactitud clínica ni porcentaje de una cota superior universal.

Fuentes: [informe](../results/phase_b/full_dev_20261012/report.md), [métricas](../results/phase_b/full_dev_20261012/summary.json), [registro de ejecución](experimentos_modelos.md).

### Intervenciones que permiten interpretar el resultado

| Experimento terminado | Evidencia | Alcance de la conclusión |
| --- | --- | --- |
| Perturbaciones espaciales, 2.400 evaluaciones y 12 condiciones | Punto desplazado dos radios: 0,1529; caja al doble: 0,3216; caja de imagen completa: 0,0333 | Cambiar la información espacial cambia fuertemente la salida con pesos fijos; no demuestra cómo falla cada componente interno |
| Selección entre salidas del decoder sin prompt | Selección IoU 0,1504; mejor de cuatro según GT 0,1824 | Estas cuatro candidatas son insuficientes para recuperar el desempeño de caja GT; no es solo elegir mal dentro de ese conjunto |
| Selección con punto GT | Salida única 0,6831; selección IoU 0,5401; mejor de tres según GT 0,8600 | Hay potencial de selección en este conjunto condicionado; un oráculo no es un método desplegable |
| Contexto permutado del generador | Correcto 0,501337; permutado 0,495295; delta 0,006042 [0,000840; 0,011245] | Sensibilidad pequeña al emparejamiento de imagen; la ventaja de 0,138 sobre estático no se puede atribuir íntegramente al contexto |
| Subgrupos por área, borde y componentes | Sin prompt sigue siendo pobre en todos los grupos de área | Asociación exploratoria; grupos pequeños y solapados, sin explicación causal ni frecuencia clínica |

Los tres condicionados eligieron la última época disponible, 6: falta demostrar convergencia. El estático apenas mejora sobre meanpoint inicial: delta +0,0143, IC [−0,0383; 0,0702]. Las comparaciones son exploratorias y no se corrigieron por multiplicidad. La permutación no prueba que el contexto nunca sirva: el generador puede estar infraentrenado o aprovechar poco sus entradas.

### U-Net: otros modelos iniciados, benchmark todavía pendiente

Existe una variante U-Net RGB desde cero, con ancho 16 y entrada 256, entrenada en pilotos de 32 train/16 dev, semilla 0. Sin normalización, tres y veinte épocas mantuvieron Dice 0; se conservaron ambos resultados. Con GroupNorm, mejor Dice 0,5082 en época 12 y final 0,4227 en época 20. Se reprodujeron los 16 Dice individuales al recargar el checkpoint. Los cuatro tests nuevos cubren geometría, gradientes, aprendizaje sintético y GroupNorm; una prueba previa de temporales de Windows sigue limitada por permisos.

Esto valida parte de la implementación y detecta problemas de optimización. No constituye evidencia de que U-Net gane o pierda frente a SAM sobre 400/100. SAM 2.1, Polyp-PVT y MedSAM no tienen ejecuciones propias completas aquí.

## Cómo conservar y fortalecer la idea inicial

Definir tres dimensiones separadas: **valor de información espacial** (diferencias de Dice/IoU), **interacción simulada** (clics necesarios, con regla y censura explícitas) y **cómputo** (latencia, memoria, parámetros y entrenamiento). El costo humano real solo puede añadirse si se recoge con usuarios y un protocolo de anotación. No llamar «costo en bits» a perturbaciones geométricas: no se estimó información de Shannon.

La caja GT comunica qué región interesa y su extensión. Retirarla cambia la especificación del objetivo de un modelo genérico; un fallo sin prompt no demuestra por sí solo incapacidad del decoder ni ausencia de señal en el encoder. El probe muestra accesibilidad de señal con supervisión, pero no identifica una única causa interna. La formulación defendible es dependencia de condicionamiento en el sistema congelado y sus rutas de generación/selección.

El aporte recomendado combina **ablación controlada, diagnóstico de candidatas frente a selección y mitigación mediante prompts automáticos**. Las ideas de self-prompting ya tienen antecedentes: [Self-Prompting](https://arxiv.org/abs/2308.07624) y [AutoSAM](https://arxiv.org/abs/2306.06370). [Can SAM Segment Polyps?](https://arxiv.org/abs/2304.07583) también estudia el dominio. No afirmar primera solución automática ni generador novedoso sin una comparación adicional. Las correcciones detalladas están en [revisión bibliográfica](revision_related_work.md).

El [dataset](https://datasets.simula.no/kvasir-seg/) contiene imágenes de pólipos con máscaras. Este protocolo no incorpora un conjunto de colonoscopías negativas sin pólipos; no permite evaluar la especificidad de un sistema de detección en un flujo clínico completo. Los falsos positivos dentro de una imagen positiva no sustituyen esa evaluación. Es un estudio de segmentación, no una validación clínica de cribado.

No añadir NLP solo para ampliar el alcance. Actualmente se estudian puntos, cajas y embeddings visuales. Un modelo condicionado por texto abriría otra pregunta sobre información semántica y tendría que declararse como tal.

## Comparación de modelos recomendada

Mantener dos tablas distintas: diagnóstico con referencias GT y segmentación totalmente automática sin GT de evaluación. Cada fila nueva debe responder una pregunta, no repetir variantes sin propósito.

| Modelo/control | Pregunta que responde | Prioridad y estado |
| --- | --- | --- |
| SAM v1 ViT-B y controles actuales | ¿Cuánto importa localización y dónde se pierde calidad? | Núcleo terminado en dev |
| U-Net GroupNorm declarada | ¿Qué consigue un segmentador directo sencillo con las mismas máscaras? | Primero: entrenamiento completo y tres semillas pendientes |
| SAM 2.1 Hiera-Tiny congelado | ¿La dependencia espacial se reproduce en otra arquitectura SAM? | Después: inferencia controlada; no exige replicar ya todos los entrenamientos |
| Polyp-PVT | ¿Es competitiva la mitigación frente a un especialista del dominio? | Extensión prioritaria para un artículo ampliado; reentrenar en IDs propios |
| Generador de igual capacidad con contexto constante | ¿La ventaja del condicionado depende del contexto o de capacidad/optimización? | Prioritario si se quiere defender el generador como contribución principal |
| MedSAM con cajas y datos auditados | ¿Qué cambia al adaptar el dominio manteniendo la información espacial? | Opcional; cajas GT no lo convierten en baseline automático |
| SAM 3 con texto fijo | ¿Qué sustituye un concepto semántico frente a un prompt espacial? | Fuera del núcleo de este abstract |

SAM 2.1 tiene [API oficial de imagen](https://github.com/facebookresearch/sam2); la entrada vacía necesita comprobar semántica y compatibilidad. Si no es comparable, marcar no aplicable en vez de fabricar un punto central. Para contrastar segmentadores con localización automática, reutilizar las mismas cajas/puntos exportados del probe; declarar que el generador usa features SAM v1. Eso aísla mejor el cambio del segmentador, aunque el pipeline completo incluye ambos costes.

[Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) aporta un control especializado. No usar sus pesos finales de pólipos sin auditar entrenamiento: el protocolo público difiere del nuestro y puede haber incluido IDs de nuestro test. Inicialización genérica y reentrenamiento propio permiten una comparación más defendible. U-Net desde cero frente a SAM preentrenado es una comparación práctica bajo igual presupuesto de etiquetas locales, no igualdad del conocimiento previo.

## Protocolo de cierre del proyecto ampliado

1. **Fijar alcance y contrastes antes de ejecutar.** Priorizar el incremento caja+punto sobre probe, sensibilidad espacial entre SAM v1/v2 y desempeño automático frente a U-Net. Usar el generador como resultado secundario salvo que se complete el control de capacidad. Guardar un plan nuevo; no alterar el plan ya ejecutado.
2. **Mismos IDs y métricas.** Train400/dev100 para desarrollo; umbral GT, resolución original, media por imagen, definición de componentes y posproceso comunes. Mantener cajas GT fuera de la clasificación de métodos automáticos. Auditar duplicados; usar agrupación por paciente/video si existen metadatos.
3. **Entrenamiento suficiente por familia.** Elegir receta U-Net con dev, documentar búsqueda y curvas, y ejecutar tres semillas. No imponer seis épocas a todos ni llamar negativo a un baseline sin convergencia. Si se usa train780, repetir todos los métodos comparados con ese presupuesto; no mezclarlo con train400.
4. **Evaluación congelada e independiente.** El test120 conserva valor como benchmark histórico después de congelar recetas, pero ya se observó durante E1 y no es confirmación nueva de las hipótesis actuales. Priorizar otro conjunto, por ejemplo CVC-ClinicDB/ColonDB, tras revisar procedencia y posible exposición de los checkpoints. Fijar ID y análisis antes de mirar resultados; no reajustar en ese conjunto. Si no se consigue evaluación independiente, declarar estudio exploratorio en Kvasir sin prometer generalización clínica.
5. **Calidad y cómputo.** Dice/IoU por imagen, diferencias pareadas, dispersión de semillas y CI. Medir latencia total con batch 1, calentamiento, mismo hardware y precisión; incluir encoder y generador, además de tiempo del decoder separado. La caché acelera entrenamiento pero no elimina el costo del encoder en una imagen nueva. Reportar memoria y cómputo de entrenamiento con límites del dispositivo empleado.
6. **Cerrar trazabilidad.** Checkpoints, manifiestos, versiones, hashes, listas de IDs y scripts de inferencia; figuras representativas con regla de selección declarada. Los resultados negativos se conservan. Verificar que documentos no anuncien como hecho lo que sigue pendiente.

Para el **resumen de póster actual**, basta presentar con honestidad el núcleo SAM terminado como estudio en curso. Para la **comparación ampliada completa**, faltan U-Net completo, SAM 2.1, cómputo común y evaluación independiente; Polyp-PVT fortalece después la comparación con especialistas. Tener más modelos sin estos controles no cierra el proyecto.

## Qué entra al abstract preparado

Se usa únicamente Fase B dev, con una tabla compacta, los resultados de generación/selección y el control de contexto. Se excluyen las cifras E1 y pilotos U-Net para evitar mezclar particiones y recetas. U-Net/SAM 2.1 y validación externa aparecen como trabajo pendiente, no como resultados. No se afirma superioridad al estado del arte.

El borrador en inglés, su Word basado en la plantilla y el registro de verificación están en [waimlap_2026](waimlap_2026/README.md). La autoría queda pendiente de una fuente formal; los metadatos encontrados arriba no la sustituyen.
