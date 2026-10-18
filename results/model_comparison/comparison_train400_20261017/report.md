# Comparación ampliada de desarrollo

400 train / 100 dev, mismos IDs. Tres semillas por método aprendido; sin nueva evaluación de test.
Las recetas, resolución y preentrenamiento difieren. La tabla no aísla causalmente arquitectura ni demuestra superioridad al estado del arte.

| Método | Dice [IC 95 % imagen] | DE semillas | Entrada | Parámetros entrenables | Latencia RGB (ms) |
|---|---:|---:|---:|---:|---:|
| Probe → caja + punto → SAM | 0.8179 [0.7707, 0.8600] | 0.0053 | 1024 | 257 | pendiente SAM completo |
| unet | 0.7814 [0.7447, 0.8140] | 0.0109 | 128 | 487289 | 12.11 |
| Probe → caja → SAM | 0.7738 [0.7271, 0.8170] | 0.0025 | 1024 | 257 | pendiente SAM completo |
| Probe lineal | 0.7489 [0.7067, 0.7879] | 0.0007 | 1024 | 257 | pendiente SAM completo |
| ds_unet | 0.6548 [0.6122, 0.6946] | 0.0150 | 128 | 18665 | 14.08 |
| fcn_lite | 0.6002 [0.5549, 0.6429] | 0.0116 | 128 | 18355 | 6.86 |
| Tokens condicionados | 0.5013 [0.4260, 0.5762] | 0.0089 | 1024 | 331008 | pendiente SAM completo |
| Tokens estáticos | 0.3633 [0.2913, 0.4375] | 0.0125 | 1024 | 1024 | pendiente SAM completo |

Los parámetros entrenables de los métodos SAM no incluyen el encoder/decoder congelados; no representan tamaño total.
Latencias RGB reestimadas después de terminar el entrenamiento, CPU/cuatro hilos/batch1, preprocesado+red+restauración+umbral; sin lectura de disco.

## Selección y presupuesto

- fcn_lite: semilla 0: época 30 de 30, semilla 1: época 29 de 30, semilla 2: época 28 de 30.
- ds_unet: semilla 0: época 30 de 30, semilla 1: época 30 de 30, semilla 2: época 30 de 30.
- unet: semilla 0: época 30 de 30, semilla 1: época 30 de 30, semilla 2: época 28 de 30.

Elegir la última época disponible no acredita convergencia; la receta y resolución son controles de presupuesto limitado.

## Control de prior espacial fijo

- fixed_point: Dice 0.3633 [0.2937, 0.4340].
- fixed_box: Dice 0.2969 [0.2293, 0.3657].
- fixed_boxpoint: Dice 0.3582 [0.2909, 0.4260].

El prior usa solo GT train; los prompts no conocen localización ni número de componentes de dev.
Contrastes pareados exploratorios y sus intervalos en `summary.json`; no se ajustó selección/multiplicidad.

## Subgrupos descriptivos

Se reutilizan los bins de área, contacto con borde y componentes de Fase B; no se optimizan cortes nuevos.
Resultados por grupo y tamaño muestral en `summary.json`. Son grupos solapados, algunos pequeños; no causas ni independencia clínica.

## Límites que permanecen

- Dev guio selección; test histórico ya inspeccionado. Falta evaluación independiente.
- Entrada 128 y entrenamiento desde cero limitan los controles RGB; SAM usa 1024 y preentrenamiento genérico.
- El presupuesto común no garantiza convergencia ni recetas óptimas; inspeccionar curvas y épocas elegidas.
- Faltan SAM 2.1/especialista, control entrenado del generador de igual capacidad y costo completo SAM.
