# Revisión técnica y científica del proyecto

Fecha: 12 de octubre de 2026. Alcance: README, cuaderno de decisiones, dependencias, todos los módulos y scripts, los dos notebooks, particiones y resultados guardados. Se inspeccionaron visualmente las figuras de salidas sin prompt y de comparación cualitativa. Esta revisión no reentrena SAM ni certifica la procedencia de los experimentos originales.

## Dictamen

El proyecto está bien orientado para un entregable preliminar: mide la dependencia de prompts, distingue generación de propuestas de filtrado y selección, y prueba una intervención pequeña manteniendo SAM congelado. Su aporte más defendible es la caracterización experimental de las fallas en Kvasir-SEG. La mitigación con tokens estáticos recupera parte de la brecha, pero la mitigación automática más fuerte ya presente es el probe que genera caja y punto.

Todavía no permite concluir causalmente que el encoder no falla, que la atención explica el mecanismo completo, o que un prompt estático es incapaz de localizar. Hay problemas concretos de reproducción que deben resolverse antes de presentar el repositorio como pipeline reproducible.

## Evidencia comprobada

Se verificó la sintaxis mediante AST de todos los archivos Python. Train/dev/test tienen 780/100/120 IDs, son disjuntos y train400 es un subconjunto de train. Todos los CSV por imagen revisados contienen exactamente los 120 IDs de test, sin duplicados, y valores Dice dentro de [0, 1].

Se recalcularon desde los CSV las medias e intervalos bootstrap de las líneas base sin prompt, caja y clic único; coinciden con el resumen guardado. Se recalculó la comparación pareada de probe frente a caja automática. La matriz de clics tiene forma 120×20; las propuestas de calibración AMG tienen forma 120×768, consistente con 256 puntos y tres máscaras por punto. Los NoC@0,85 y @0,90 se reproducen a partir de la matriz.

No se ejecutó la inferencia completa: en este checkout faltan las imágenes, los pesos, los embeddings y la instalación de segment-anything. Por tanto, la consistencia interna de los resultados está comprobada parcialmente; su reproducción desde imágenes sigue pendiente.

| Condición | Dice medio | Interpretación |
|---|---:|---|
| Sin prompt, token 0 | 0,0122 | Casi ningún solapamiento con el pólipo |
| Sin prompt, multimáscara + IoU | 0,1259 | La elección del modo de salida cambia mucho el resultado |
| Sin prompt, mejor de cuatro, oráculo | 0,1573 | Elegir mejor entre estas salidas no basta |
| Un punto por componente, token 0 | 0,6961 | Gran recuperación con información espacial |
| Un punto por componente, selección IoU | 0,5708 | Selección inferior a la salida única |
| Un punto por componente, oráculo de tres | 0,8403 | Buenas alternativas que la selección no identifica |
| Caja GT por componente | 0,8859 | Referencia con información privilegiada |
| AMG, mejor propuesta cruda | 0,8946 | Hay buenas propuestas al consultar muchos puntos |
| AMG, mayor IoU predicho tras filtros/NMS | 0,0578 | La calidad genérica no selecciona necesariamente el pólipo |
| Probe lineal | 0,6142 | Hay información utilizable por un clasificador supervisado |
| Probe → caja → SAM | 0,6224 | Ganancia media pequeña sobre el probe |
| Probe → caja + punto → SAM | 0,6545 | Mejor mitigación automática evaluada |
| Tokens estáticos k=4, init. clic medio | 0,3178 | Recuperación parcial, media de tres semillas |
| Tokens k=4 → caja → SAM | 0,3265 | Refinamiento limitado por la localización inicial |

## Qué significa realmente «sin prompts»

Conviene separar tres condiciones en el título, las tablas y la discusión:

1. **Decoder sin prompts espaciales:** cero embeddings dispersos, sin máscara de entrada; conserva los tokens de salida, la codificación posicional y no_mask_embed.
2. **Sin intervención humana en inferencia:** AMG y auto-prompts sí introducen puntos, cajas o vectores; los generan automáticamente.
3. **Sin supervisión específica de Kvasir:** SAM original cumple esta condición; el probe y los tokens aprendidos no, porque se entrenan con máscaras de Kvasir.

AMG no es una ejecución del decoder sin prompts: consulta una rejilla de puntos. El oráculo usa GT para elegir, por lo que es una herramienta de diagnóstico y no un método desplegable sin conocimiento de la respuesta.

SAM fue diseñado para segmentación condicionada a prompts; evaluar el caso vacío es una ablación de ese condicionamiento. La pregunta interesante es qué información falta y cuánto puede reemplazarse automáticamente. El código oficial distingue token 0 para salida única y tokens 1–3 para multimáscara; la implementación local respeta esta distinción.

Fuente: [artículo de SAM](https://arxiv.org/abs/2304.02643) y [mask decoder oficial en el commit utilizado](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/mask_decoder.py).

## Dónde falla: conclusiones sólidas y límites

**Sin prompts espaciales hay un problema de propuesta respecto del objetivo.** El oráculo de cuatro salidas solo llega a 0,1573: la baja calidad no puede atribuirse únicamente a elegir el token equivocado.

**Con AMG aparecen pérdidas distintas y medibles.** El embudo recalculado es 0,8946 → 0,7921 después del filtro IoU → 0,4995 después de estabilidad → 0,4931 después de NMS → 0,0578 al seleccionar por IoU predicho. Las caídas medias son aproximadamente 0,1025, 0,2927, 0,0064 y 0,4353. Corresponden a diferencias entre oráculos de conjuntos sucesivos y selección final, no a porcentajes causales independientes.

**El probe aporta evidencia contra la ausencia total de información útil en el encoder.** Su Dice de 0,6142 y los resultados por capa apoyan que parte de la información se puede extraer con supervisión. No prueban que el encoder sea óptimo ni que todo el error proceda exclusivamente del decoder. La gran mejora con caja, manteniendo el mismo encoder, refuerza la dependencia del condicionamiento espacial.

**La cabeza IoU no es un detector semántico de pólipos.** En AMG muchos puntos consultan otros objetos o regiones. Comparar cada máscara exclusivamente con la GT del pólipo mezcla relevancia semántica y calidad geométrica. Una máscara de mucosa puede tener buena calidad respecto de su objetivo y baja IoU con el pólipo. Por ello, la correlación global de 0,03 demuestra mala utilidad como selector de lesiones, pero no basta para demostrar descalibración intrínseca respecto de cualquier objeto. La comparación dentro de un mismo punto, especialmente en el interior del pólipo, es más informativa para separar ambos efectos.

**La atención es evidencia descriptiva.** El cambio de masa sobre GT —aproximadamente 4,9 % sin prompt, 22,9 % con tokens y 59,4 % con caja— es compatible con un cambio de localización. No constituye por sí solo una explicación causal. Además, la condición aprendida se calcula únicamente con la primera semilla encontrada, y la caja de atención usa la mayor componente mientras la evaluación de cajas une todas las componentes.

**Los subgrupos necesitan prudencia.** Hay solo siete imágenes con múltiples componentes y cinco con área inferior al 2 %. Las componentes conexas de una máscara no equivalen necesariamente a lesiones clínicas diferentes. Las heurísticas de marcador/recuadro no han sido validadas manualmente y pueden estar confundidas con tamaño o posición del pólipo. Los reflejos no están medidos. Las visualizaciones PCA usan cuatro imágenes seleccionadas; no sostienen una conclusión general sobre artefactos causantes de fallas.

## Problemas concretos que deben corregirse

### Prioridad alta: reproducción

- **README y Colab no reproducen el resultado de tokens reportado.** Ambos omiten `--init meanpoint`; el valor por defecto es `random`. La corrección debe hacerse también en `scripts/make_notebooks.py`, para que regenerar los notebooks no restaure el error. Las ejecuciones exitosas necesitan `--init meanpoint --epochs 6 --lr 5e-3 --bs 8`.
- **Colab omite dos etapas y atributos por subgrupo.** Añadir `07b_twostage.py` y `08b_subgroups.py test`. Ejecutar las estadísticas de atención después de entrenar los tokens. Como se clona un repositorio con resultados existentes, resumir archivos viejos puede ocultar que estas fases no se ejecutaron.
- **FP32 no describe toda la ruta de evaluación.** `02_encode.py` siempre guarda embeddings en float16, incluso si el encoder corre en FP32. `CachedSAM.emb` los convierte de nuevo a float32, sin recuperar la precisión perdida. Debe declararse «encoder FP32, caché FP16, decoder FP32», o almacenar también embeddings FP32 y medir acuerdo de máscaras, especialmente sin prompts y cerca del umbral cero.
- **La caché puede reutilizar artefactos de otra configuración.** El script salta imágenes si ya existen archivo y metadatos; no valida precisión solicitada, hash de imagen/pesos, commit ni disponibilidad de bloques. Cambiar `--precision` no obliga a recalcular. La caché necesita un manifiesto y validación explícita.
- **Evaluación parcial silenciosa.** Algunos scripts eliminan IDs sin metadatos. La evaluación final debe abortar si faltan imágenes y registrar los IDs efectivamente evaluados.
- **Portabilidad Windows.** `01_teardown.py` importa `resource`, no disponible en Windows, aunque no se usa. Los bucles shell del README son de Bash y Colab/Linux. Documentar el entorno compatible o ajustar la reproducción para PowerShell.
- **Dependencias y trazabilidad.** El entorno local observado es Python 3.13.5, torch 2.7.1+cpu, NumPy 1.26.4 y SciPy 1.16.0; el entorno declarado del E1 es distinto. Esto no invalida los CSV, pero no se verificó que el requirements fijado pueda instalarse. Colab tampoco fija todo el entorno y usa una URL de repositorio de ejemplo. Guardar un manifiesto verificable por corrida, comandos, hashes, versiones y registros. El checkout contiene un solo commit; el design log por sí solo no acredita temporalmente el prerregistro.

### Prioridad alta: exactitud del relato

- **«71 % vacías» es inexacto.** `empty_rate` usa área < 0,0001. Hay 85/120 máscaras casi vacías (70,8 %), pero 70/120 estrictamente vacías (58,3 %). Reportar ambas definiciones.
- **La tabla principal del README omite el mejor automático.** Caja + punto desde el probe alcanza 0,6545 y recupera aproximadamente 73,5 % de la brecha respecto de token 0 y caja GT; caja sola recupera aproximadamente 69,8 % y tokens k=4, 35,0 %. Declarar también la dependencia de R respecto de la línea base: usar el modo multimáscara sin prompt cambia la brecha.
- **La superioridad de caja automática frente a probe es débil en media.** Diferencia +0,00824; IC bootstrap [-0,01525, 0,02854]; Wilcoxon p=0,03955. No hay contradicción: Wilcoxon y el IC de la media evalúan propiedades distintas. No presentar una mejora media concluyente. Caja + punto tiene diferencia +0,04033 e IC [0,01335, 0,06752] en los resultados guardados.
- **El fracaso de tokens estáticos no demuestra imposibilidad.** Se probó una familia limitada, seis épocas y una receta principal. La explicación del colapso por BCE/Dice y la interpretación «qué sin dónde» son hipótesis plausibles; faltan curvas de gradientes, prevalencia de salida, comparación de pérdidas e inicialización sin entrenar. Un token fijo puede consultar información dependiente de la imagen dentro del decoder.

### Prioridad media: medición y estadística

- **El lift de atención mezcla marcos.** La masa se normaliza sobre los 4096 tokens, pero el área GT se divide entre tokens válidos. Con atención uniforme sobre toda la rejilla, el lift sería `valid.sum()/4096`, no 1. Renormalizar atención sobre región válida antes de comparar, o usar área relativa de toda la rejilla; registrar por separado masa en padding. `09_figures.py` ya renormaliza los mapas recortados, por lo que figura y resumen no usan exactamente la misma definición.
- **Los IC aprendidos condicionan a tres semillas concretas.** Se promedian las semillas por imagen y se hace bootstrap de imágenes. Es válido para esa media, pero no incorpora toda la incertidumbre de reentrenamiento. Reportar dispersión por semilla y, si corresponde, remuestreo de imágenes y semillas. No afirmar que segmentar máscaras por semilla y promediar resultados sea un ensemble de máscaras.
- **Control de multiplicidad.** Hay varias pruebas pareadas y subgrupos; distinguir hipótesis confirmatorias de exploratorias y ajustar comparaciones cuando proceda. No tratar miles de propuestas o tokens como observaciones independientes; la unidad de evaluación principal es la imagen. Investigar agrupación por paciente/video y duplicados antes de asumir independencia clínica.
- **NoC y costo humano.** NoC@0,85=3,48 incluye una imagen que no alcanza el umbral; @0,90=5,32 incluye tres. Registrar censura a 20 y porcentaje de fallos. Punto interior y caja GT son prompts oráculo; no representan tiempo ni precisión de un usuario real. El primer clic interactivo consulta la primera componente, que no está ordenada por tamaño, mientras el modo punto consulta cada componente.
- **Medición de 64 puntos.** En teardown, `n_points=64` crea forma (64,1,2): son 64 consultas de un punto sobre una imagen, no una consulta con 64 clics. Renombrar la métrica. Los >99 % de FLOPs del encoder aplican a una consulta, no necesariamente a AMG, que ejecuta cientos de consultas.
- **AMG propio frente al oficial.** La lógica central de puntos/filtros/NMS coincide, pero cambian rejilla y lote. La implementación oficial también tiene comprobación de bordes de recorte; con una sola imagen completa su efecto suele ser nulo. Validar equivalencia con el generador oficial a igual configuración antes de atribuir diferencias al dominio. Los valores por defecto oficiales se eligieron para ViT-H, mientras el proyecto usa ViT-B.

Fuente para la última observación: [generador automático oficial](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/automatic_mask_generator.py).

## Mitigación y experimentos recomendados

La mitigación principal defendible hoy es **localización supervisada ligera sobre embeddings congelados → caja + punto automáticos → SAM congelado**. No requiere humano en test, sí máscaras de entrenamiento. Mantener probe puro como control permite medir cuánto aporta realmente el decoder.

Para extender el proyecto, priorizar dos preguntas y conservar los tokens estáticos como ablación:

1. **¿Condicionar el prompt a la imagen mejora la localización?** Comparar la inicialización media sin entrenar, tokens estáticos entrenados, probe puro, caja y caja + punto, y una pequeña red que genere prompts desde embeddings. Igualar train400 y usar dev para todas las decisiones. Registrar también número de componentes/cajas y latencia.
2. **¿Seleccionar propuestas por relevancia a la lesión recupera la pérdida de AMG?** Construir un selector supervisado sobre propuestas con características del embedding y de la máscara; usar calidad predicha y estabilidad como señales complementarias. Variar umbrales en dev y comparar candidatos crudos frente a filtrados. Entrenar solo con train y reportar tanto calidad del conjunto como calidad de selección.

Añadir ablaciones espaciales: punto correcto, desplazado y aleatorio; caja GT, perturbada y de imagen completa. Evaluar acuerdo FP32/FP16 por condición. Estas intervenciones separan mejor localización, selección y precisión numérica que una interpretación basada únicamente en atención.

El test actual ya se ha examinado en detalle. Nuevas decisiones inspiradas por sus errores deben declararse exploratorias; reservar un segundo dataset o evaluación externa para confirmar generalización. Mantener fijo el protocolo y no seleccionar hiperparámetros por test.

## Formulación científica sugerida

«En SAM ViT-B sobre Kvasir-SEG, eliminar prompts espaciales reduce fuertemente la segmentación de pólipos, incluso con selección oráculo entre las cuatro salidas del decoder. Al introducir puntos automáticos mediante AMG aparecen buenas propuestas, pero los filtros de calidad y la selección genérica pierden gran parte del desempeño. Un probe supervisado sobre embeddings congelados proporciona localización útil; convertido en caja y punto automáticos alcanza Dice 0,655, frente a 0,318 de tokens estáticos aprendidos. Los resultados apoyan la importancia del condicionamiento espacial y de la selección orientada a la lesión; la explicación causal y la transferencia a otros dominios requieren experimentos adicionales».
