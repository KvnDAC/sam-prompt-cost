# Validación externa congelada: CVC-ClinicDB

Modelos y umbrales fijados en Kvasir-SEG; 612 frames/29 secuencias; sin adaptación externa.

| Método | Dice por frame | IC 95 % agrupado por secuencia | Macro-secuencia | DE semillas |
|---|---:|---:|---:|---:|
| probe | 0.4243 | [0.3624, 0.4892] | 0.4097 | 0.0077 |
| sam_auto | 0.4587 | [0.3793, 0.5405] | 0.4472 | 0.0142 |
| unet | 0.3405 | [0.2841, 0.4002] | 0.3242 | 0.0729 |

## Diferencias pareadas

| Contraste | Diferencia Dice | IC 95 % por secuencia |
|---|---:|---:|
| sam_auto_minus_unet | 0.1182 | [0.0672, 0.1684] |
| sam_auto_minus_probe | 0.0344 | [0.0110, 0.0576] |

Los IC remuestrean las 29 secuencias completas. CVC-ClinicDB no contiene negativos y no mide especificidad.
