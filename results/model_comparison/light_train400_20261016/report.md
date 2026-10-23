# Comparación ligera: resultados de desarrollo

400 train / 100 dev; tres semillas; RGB 128, sin preentrenamiento. No se evaluó test.

| Modelo | Dice [IC 95 % imagen] | DE semillas | Parámetros | Latencia mediana (ms) |
|---|---:|---:|---:|---:|
| fcn_lite | 0.6002 [0.5562, 0.6429] | 0.0116 | 18355 | 5.94 |
| ds_unet | 0.6548 [0.6119, 0.6940] | 0.0150 | 18665 | 14.87 |
| unet | 0.7814 [0.7439, 0.8143] | 0.0109 | 487289 | 13.88 |

Los IC remuestrean imágenes después de promediar semillas. No ajustan selección ni multiplicidad.
Latencia CPU batch1 incluye preprocesado RGB, red completa y restauración; excluye lectura de disco.
Variantes locales compactas: no son reproducciones exactas de U-Net/FCN publicados ni modelos de estado del arte.
No comparar esta latencia con tiempos de decoder SAM sobre caché. SAM incorpora otro preentrenamiento y resolución.
El tope de épocas es un presupuesto, no prueba de convergencia. Curvas y épocas elegidas deben acompañar la tabla.
