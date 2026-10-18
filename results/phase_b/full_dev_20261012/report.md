# Fase B: resultados de desarrollo

400 imágenes train, 100 dev; SAM ViT-B congelado, encoder y caché FP32. No se evaluó test.

Estos resultados sirven para selección de configuración; no estiman generalización en un conjunto independiente.

LR condicionado seleccionado: 0.0001. Tres semillas en las comparaciones entrenadas.

| Método | Dice dev [IC 95 % por imagen] | Desv. entre semillas | Brecha recuperada |
|---|---:|---:|---:|
| Caja GT | 0.9352 [0.9279, 0.9420] | — | 100.0 % |
| Probe → caja + punto → SAM | 0.8179 [0.7707, 0.8600] | 0.0053 | 87.3 % |
| Probe → caja → SAM | 0.7738 [0.7271, 0.8170] | 0.0025 | 82.6 % |
| Probe lineal | 0.7489 [0.7067, 0.7879] | 0.0007 | 79.9 % |
| Punto GT | 0.6831 [0.6155, 0.7463] | — | 72.8 % |
| Tokens condicionados | 0.5013 [0.4260, 0.5762] | 0.0089 | 53.2 % |
| Tokens estáticos | 0.3633 [0.2913, 0.4375] | 0.0125 | 38.3 % |
| Tokens condicionados · inicialización | 0.3490 [0.2811, 0.4179] | 0.0013 | 36.7 % |
| Tokens estáticos · inicialización | 0.3490 [0.2811, 0.4178] | 0.0014 | 36.7 % |
| Sin prompt | 0.0089 [0.0007, 0.0227] | — | 0.0 % |

Los IC remuestrean imágenes tras promediar semillas; la dispersión entre semillas se informa por separado.
Punto/caja GT son referencias oráculo. Las inicializaciones no entrenadas sí usan localización GT de train para construir meanpoint.
Las pruebas pareadas son exploratorias y no tienen ajuste de multiplicidad.

## Épocas seleccionadas

- static: semilla 0: época 6, semilla 1: época 2, semilla 2: época 3
- conditional: semilla 0: época 6, semilla 1: época 6, semilla 2: época 6

Si se selecciona época 0, las actualizaciones no mejoraron la referencia inicial en dev.

Los resultados E1 se conservan. El siguiente paso es congelar la configuración y evaluar los checkpoints en test.

## Control de contexto del generador

Prompts de la imagen correcta: Dice 0.5013; contexto permutado: 0.4953.
Diferencia pareada normal − permutado: 0.0060, IC 95 % [0.0008; 0.0112].
Decoder, imagen objetivo y pesos permanecen fijos. Los checkpoints recargados reprodujeron los scores elegidos en dev.
Este control exploratorio mide sensibilidad al contexto; no sustituye un baseline de igual capacidad ni validación externa.
Detalles en [conditioning_diagnostic/summary.json](conditioning_diagnostic/summary.json).

## Caracterización espacial y selección

La ablación espacial completa, los subgrupos exploratorios y los casos reproducidos están en [spatial_ablation.md](spatial_ablation.md).
Sin prompt, el mejor de cuatro según GT alcanza Dice 0.1824; seleccionar entre esas cuatro salidas no recupera por sí solo la referencia de caja GT.
Con punto, la salida única da 0.6831, la elección por IoU predicho 0.5401 y el mejor de tres según GT 0.8600.
El oráculo de selección usa GT y no es una mitigación desplegable. Control posterior a inspeccionar la ablación, detallado en [output_selection/report.md](output_selection/report.md).

No se midió tiempo humano de anotación; las referencias GT caracterizan el valor y precisión de la información espacial.
