# Control de prior espacial fijo

400 máscaras train para estimar el prior; 100 imágenes dev; SAM congelado, FP32, una consulta por imagen. No se evaluó test.

| Condición automática | Dice [IC 95 % imagen] |
|---|---:|
| fixed_point | 0.3633 [0.2937, 0.4340] |
| fixed_box | 0.2969 [0.2293, 0.3657] |
| fixed_boxpoint | 0.3582 [0.2909, 0.4260] |

Probe→caja+punto menos prior caja+punto: **+0.4597**, IC pareado [0.3809; 0.5370].

Ambos usan una caja y un punto positivos por imagen y el mismo SAM; el prior es fijo, mientras el probe localiza desde la imagen. Solo el entrenamiento del prior usa GT: no se usan máscaras dev para generar prompts.
La comparación es exploratoria y usa el pipeline probe ya seleccionado en dev. No mide tiempo humano ni sustituye un control entrenado del generador con igual capacidad.
La mediana de coordenadas se fija con las GT train completas, binarizadas a 127; no se ajustaron sus coordenadas después de ver dev.
