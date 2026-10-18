# Calidad y costo completo: comparación de desarrollo

400 train / 100 dev; tres semillas para calidad, selección en dev. Sin nueva evaluación de test.

| Método | Dice [IC 95 % imagen] | Entrada | Parámetros totales | Entrenables | Latencia mediana (ms) |
|---|---:|---:|---:|---:|---:|
| Probe → caja + punto → SAM | 0.8179 [0.7707, 0.8600] | 1024 | 93735729 | 257 | 8369.38 |
| unet | 0.7814 [0.7447, 0.8140] | 128 | 487289 | 487289 | 12.11 |
| ds_unet | 0.6548 [0.6122, 0.6946] | 128 | 18665 | 18665 | 14.08 |
| fcn_lite | 0.6002 [0.5549, 0.6429] | 128 | 18355 | 18355 | 6.86 |

SAM completo: mediana 8369.38 ms, p95 8905.77 ms; semilla 0.
CPU/cuatro hilos/FP32/batch1. Mismos veinte RGB decodificados, cinco calentamientos y veinte repeticiones.
Incluye preprocesado, red completa y restauración/umbral; SAM incluye encoder, probe, componentes, prompt encoder y decoder. Excluye disco, carga de pesos y métricas.
Las predicciones del pipeline SAM desde caché y RGB reprodujeron los scores previos en las imágenes comprobadas a tolerancia 1e−8.

## Interpretación y límites

- Las redes compactas se entrenaron desde cero a resolución 128; SAM usa preentrenamiento genérico y resolución 1024. Es una comparación de sistemas con el mismo presupuesto local de etiquetas, no de arquitecturas aisladas.
- Latencia local de implementaciones prototipo, no benchmark universal ni medición de tiempo humano. Los tiempos RGB resumen tres semillas; el de SAM representa semilla 0.
- En RGB se usa el preprocesador experimental completo, incluida creación de tensores auxiliares sin GT; no es una implementación optimizada para producción.
- Los parámetros entrenables de SAM son solo 257, pero su encoder y decoder congelados sí cuentan en tamaño y cómputo total.
- La selección de épocas y otros ajustes usa dev. Los IC remuestrean imágenes tras promediar semillas; no son validación independiente por paciente.
- Los contrastes, controles de prior y subgrupos están en `report.md`/`summary.json`; no deben interpretarse como pruebas clínicas confirmatorias.
- Pendientes: mayor resolución/presupuesto de controles RGB, especialista/SAM 2.1, control de capacidad del generador y evaluación externa.

## Diferencia media frente a U-Net

SAM automático menos U-Net: 0.03648, IC pareado 95 % [-0.00554; 0.07712].
El IC incluye cero: este análisis no demuestra una mejora positiva de la media ni equivalencia. El p de Wilcoxon del JSON se refiere a rangos/distribución, no a esa misma hipótesis sobre la media.
El cómputo local mucho mayor de SAM y esta incertidumbre justifican conservar U-Net como comparador principal y madurar entrenamiento/evaluación independiente.
