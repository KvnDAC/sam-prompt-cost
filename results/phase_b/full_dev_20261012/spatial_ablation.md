# Ablación espacial: resultados completos en dev

SAM ViT-B original congelado, encoder FP32 y caché FP32. Evaluación de las 100 imágenes de dev a resolución original. Se usa salida única, una consulta por componente GT elegible y unión de predicciones. Los prompts son oráculo: este experimento caracteriza dependencia espacial, no es una mitigación automática.

| Condición | Dice medio | IC 95 % por imagen |
|---|---:|---:|
| Sin prompt | 0,0089 | [0,0007; 0,0227] |
| Caja de imagen completa | 0,0333 | [0,0183; 0,0508] |
| Punto más interior | 0,6831 | [0,6155; 0,7463] |
| Punto desplazado 0,5 radios | 0,7095 | [0,6687; 0,7509] |
| Punto desplazado 1 radio | 0,6255 | [0,5805; 0,6707] |
| Punto desplazado 2 radios | 0,1529 | [0,1309; 0,1756] |
| Punto desplazado 4 radios | 0,0969 | [0,0786; 0,1156] |
| Caja reducida a 0,75× | 0,7243 | [0,6905; 0,7568] |
| Caja ajustada GT | 0,9352 | [0,9279; 0,9420] |
| Caja ampliada a 1,25× | 0,8971 | [0,8757; 0,9152] |
| Caja ampliada a 1,5× | 0,6547 | [0,5814; 0,7247] |
| Caja ampliada a 2× | 0,3216 | [0,2559; 0,3917] |

Se ejecutaron 2.400 evaluaciones. Para los puntos desplazados se promedian las cuatro direcciones dentro de cada imagen antes de calcular medias e intervalos. Radio significa distancia interior máxima de la máscara GT; los puntos y las cajas se recortan a límites de imagen. Escalar una caja multiplica ancho y alto, no solamente área. Los intervalos usan 10.000 remuestreos por imagen.

## Lectura de los resultados

El mismo encoder y decoder alcanzan un Dice alto con información precisa de ubicación y extensión. Introducir una caja genérica que abarca la imagen no recupera ese desempeño. Esto apoya la importancia del contenido espacial del prompt, además de su mera presencia.

La respuesta es sensible a errores de localización: desplazar el punto dos radios reduce Dice en 0,5302 respecto del punto interior, con IC pareado [-0,5997; -0,4589]. Ampliar la caja al doble reduce Dice en 0,6135 respecto de la caja GT, con IC [-0,6809; -0,5433]. Una mitigación que genere cajas necesita controlar tanto ubicación como extensión.

La curva de puntos no es estrictamente decreciente cerca del centro. La diferencia de medio radio respecto del punto interior es +0,0264, pero su IC pareado [-0,0362; 0,0937] cruza cero. No se establece una mejora ni se propone cambiar la política de puntos a partir de este resultado.

Sin prompt hay 46 % de máscaras estrictamente vacías en dev. Esto no se compara directamente con el porcentaje de test del E1: cambian la partición y la precisión de almacenamiento. El valor 0,9352 de caja GT en dev tampoco significa que se haya mejorado el modelo respecto de la tabla de test anterior.

Los puntos desplazados pueden salir de la lesión y las cajas ampliadas pueden incluir otros objetos; estas intervenciones simulan información espacial incorrecta. Los resultados no demuestran que el decoder falle con prompts correctos, ni excluyen otros mecanismos del fallo sin prompt. Falta evaluar la mitigación aprendida y confirmar generalización en un conjunto independiente.

## Casos reproducidos y subgrupos exploratorios

El análisis por área, múltiples componentes y contacto con borde está en [spatial_subgroups/report.md](spatial_subgroups/report.md). Sin prompt, Dice es cercano a cero en los tres grupos de área. Una caja GT mantiene Dice entre 0,9305 y 0,9423, mientras una caja ampliada 2× cae a 0,1049 en lesiones que ocupan al menos 15 % de la imagen. Estos grupos se examinaron después de ver la ablación y sus atributos están relacionados; no son contrastes confirmatorios.

Se reprodujeron las nueve salidas sin prompt, punto y caja de los tres casos con mayor diferencia caja−punto. Cada Dice coincide con el CSV original a tolerancia 1e−8. En el [panel de predicciones](spatial_cases/predictions.png), los puntos interiores están sobre regiones brillantes y las salidas únicas cubren pequeñas zonas locales: Dice 0,0237; 0,0109; 0,0121. Las cajas de esas mismas lesiones dan 0,9776; 0,9600; 0,9577. La inspección sugiere ambigüedad de qué región debe abarcar el objeto aun con un punto positivo interior. No mide la frecuencia de este patrón, no compara las salidas multimáscara y no demuestra que los reflejos causen todos los fallos. Son extremos elegidos después de conocer los scores.

El costo humano no se midió: no hay tiempos de anotación ni comparación entre anotadores. Por ahora se caracteriza dependencia de información espacial y sensibilidad a su precisión, con puntos/cajas GT como referencias oráculo.

Un [control posterior de selección de salidas](output_selection/report.md) reprodujo las 200 predicciones únicas sin prompt/con punto. Con un punto, la salida única da 0,6831, la elección por IoU predicho 0,5401 y el mejor candidato según GT 0,8600. Sin prompt, incluso el mejor de las cuatro salidas según GT da solo 0,1824. Dentro de esas cuatro candidatas, la selección por sí sola no explica todo el fallo sin prompt.

## Artefactos

- [Curvas con intervalos](perturbations.png).
- [Resultados por imagen y dirección](perturbations/dev/per_image_direction.csv).
- [Resumen de condiciones](perturbations/dev/summary.json).
- [Comparaciones pareadas exploratorias](perturbations/dev/paired_tests_exploratory.json).
- [Manifiesto de ejecución](perturbations/dev/manifest.json).

Las comparaciones pareadas son exploratorias, sin ajuste de multiplicidad. El campo seconds del CSV incluye generación de prompts, decoder, postproceso y cálculo de métricas; no representa latencia aislada del decoder.
