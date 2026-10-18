# U-Net: entrenamiento adicional y resolución

400 train / 100 dev; tres semillas; sin nueva evaluación de test.
Cada par parte del mismo estado de la época 30 (pesos, Adam y RNG).
15 épocas adicionales por brazo, 50 pasos por época, LR 0.001, batch 8.
La selección se reinicia en ambos brazos: estado inicial más las 15 épocas nuevas.
El endpoint usa exactamente 750 pasos adicionales; selected elige por Dice dev.

| Método | Dice medio | IC 95 % imágenes |
|---|---:|---:|
| 128_initial | 0.7786 | [0.7419, 0.8115] |
| 128_endpoint | 0.7955 | [0.7577, 0.8292] |
| 128_selected | 0.8214 | [0.7870, 0.8517] |
| 256_initial | 0.7033 | [0.6672, 0.7369] |
| 256_endpoint | 0.8267 | [0.7944, 0.8546] |
| 256_selected | 0.8272 | [0.7955, 0.8547] |
| 128_previous_selected | 0.7814 | [0.7447, 0.8140] |
| sam_auto | 0.8179 | [0.7707, 0.8600] |

## Diferencias pareadas exploratorias

| Contraste | Diferencia media | IC 95 % |
|---|---:|---:|
| 256_initial minus 128_initial | -0.0753 | [-0.1016, -0.0502] |
| 256_endpoint minus 128_endpoint | 0.0313 | [0.0122, 0.0513] |
| 256_selected minus 128_selected | 0.0058 | [-0.0086, 0.0202] |
| 128_endpoint minus 128_initial | 0.0169 | [0.0017, 0.0321] |
| 128_selected minus 128_previous_selected | 0.0400 | [0.0258, 0.0556] |
| sam_auto minus 128_selected | -0.0035 | [-0.0424, 0.0329] |
| 256_endpoint minus 256_initial | 0.1235 | [0.0991, 0.1494] |
| 256_selected minus 128_previous_selected | 0.0458 | [0.0276, 0.0652] |
| sam_auto minus 256_selected | -0.0093 | [-0.0481, 0.0279] |

Los IC remuestrean imágenes tras promediar semillas. No corrigen selección, multiplicidad ni dependencia entre imágenes.
La comparación es de desarrollo y fue motivada por resultados anteriores. No demuestra generalización.
Llegar a la época 45 no prueba convergencia. Inspeccionar curvas y épocas seleccionadas.
SAM usa preentrenamiento genérico y entrada 1024; U-Net se entrena desde cero.
La intervención de resolución cambia también la geometría efectiva de la red y su coste de cómputo.
