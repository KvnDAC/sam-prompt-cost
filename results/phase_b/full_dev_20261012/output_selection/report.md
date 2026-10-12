# Generación y selección de salidas en dev

Control exploratorio añadido después de inspeccionar la ablación espacial. SAM congelado, mismos 100 IDs y caché FP32. Sin test.

Las 200 predicciones únicas sin prompt/con punto reproducen los scores de la ablación a tolerancia 1e−8.

| Salida | Dice [IC 95 %] |
|---|---:|
| Sin prompt · salida única | 0.0089 [0.0007, 0.0227] |
| Sin prompt · multimáscara + IoU predicho | 0.1504 [0.1157, 0.1876] |
| Sin prompt · mejor de cuatro según GT | 0.1824 [0.1440, 0.2240] |
| Punto GT · salida única | 0.6831 [0.6155, 0.7463] |
| Punto GT · multimáscara + IoU predicho | 0.5401 [0.4678, 0.6114] |
| Punto GT · mejor de tres según GT | 0.8600 [0.8277, 0.8895] |

El oráculo usa GT para elegir y mide disponibilidad de candidatos útiles; no es una mitigación automática.
Con punto, se elige por componente y después se unen las máscaras. Sin prompt, las cuatro salidas se comparan contra la GT completa.
Los intervalos remuestrean imágenes. Las pruebas pareadas son exploratorias y no corrigen multiplicidad.
Una mejora del oráculo frente a la salida única muestra que esa salida puede ocultar candidatos útiles; no prueba por sí sola la causa del fallo ni garantiza que puedan seleccionarse sin GT.

Con un punto, elegir por IoU predicho es peor que usar la salida única: diferencia −0,1430, IC pareado [−0,1955; −0,0938]. El mejor candidato según GT supera al elegido por IoU predicho en 0,3199 [0,2590; 0,3832]. En este protocolo existe una limitación de selección además de la ambigüedad ilustrada en los casos.

Sin prompt, ni siquiera escoger según GT entre las cuatro salidas alcanza un Dice alto (0,1824). Por tanto, mejorar exclusivamente el selector de esas cuatro candidatas no puede recuperar la referencia de caja GT (0,9352) en este conjunto. Esto no se aplica automáticamente a AMG, que genera muchas más propuestas mediante prompts espaciales.

Se ejecutó con cuatro hilos, igual que la ablación. Un intento previo con un hilo se detuvo por una discrepancia de reproducción y no se incorporó a la tabla.
