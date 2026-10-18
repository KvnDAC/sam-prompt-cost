# Protocolo congelado: eficiencia de etiquetas

## Pregunta

¿El preentrenamiento de SAM reduce la cantidad de máscaras manuales necesarias para adaptar un segmentador a Kvasir-SEG?

La comparación separa tres sistemas:

1. **Probe:** clasificador lineal sobre los embeddings congelados de SAM. Mide cuánta información espacial ya está presente en el encoder.
2. **Probe → caja+punto → SAM:** convierte la máscara del probe en prompts automáticos y ejecuta el prompt encoder y mask decoder congelados. Mide el valor añadido por SAM después de localizar automáticamente el pólipo.
3. **U-Net RGB:** control compacto sin preentrenamiento, entrenado desde cero. Mide cuánto puede aprender un segmentador supervisado directo con las mismas máscaras.

## Diseño

- Dataset: Kvasir-SEG.
- Entrenamiento: partición fija `train400`.
- Desarrollo: partición fija `dev100`.
- Test: no se consulta.
- Presupuestos de etiquetas: 25, 100 y 400 imágenes.
- Semillas: 0, 1 y 2.
- Para cada semilla se genera una permutación reproducible de `train400`; sus primeros 25, 100 y 400 elementos forman subconjuntos anidados.
- Los tres métodos usan los mismos IDs dentro de cada celda presupuesto–semilla.
- En 25 y 100, la variación entre semillas incluye elección del subconjunto e inicialización. En 400, los IDs son los mismos y sólo cambia la optimización.

## Presupuesto de optimización

Para no regalar más cómputo a los presupuestos con más imágenes, el endpoint usa un número fijo de actualizaciones:

- Probe: 300 actualizaciones Adam, batch de 8192 tokens, LR 0.01. Se extraen hasta 1500 tokens por imagen y se muestrea con reemplazo durante la optimización.
- U-Net: 1500 actualizaciones Adam, batch de 8 imágenes, LR 0.001, entrada 256 × 256, ancho base 8, GroupNorm, pérdida BCE + Dice y flips horizontal/vertical. Las imágenes se muestrean con reemplazo.

El umbral del probe se elige en `dev100` sobre la grilla fija `[-3, 3]` con 25 valores. Para U-Net, el **endpoint de 1500 pasos** es el resultado principal. El mejor checkpoint según Dice de desarrollo, evaluado cada 100 pasos e incluyendo el paso 0, es secundario y se identifica como seleccionado en desarrollo.

## Métricas y contrastes

- Dice e IoU por imagen.
- Media de las tres semillas por imagen antes de construir intervalos.
- IC 95 % por bootstrap de las 100 imágenes de desarrollo, 10 000 remuestreos.
- Curvas Dice frente al número de máscaras.
- Diferencias pareadas exploratorias:
  - `probe→caja+punto→SAM − probe`, para estimar el aporte del decoder condicionado;
  - `probe→caja+punto→SAM − U-Net`, para comparar adaptación sobre SAM con aprendizaje RGB desde cero;
  - incrementos 25→100 y 100→400 dentro de cada método.

Los intervalos no corrigen multiplicidad, selección ni dependencia entre imágenes. El experimento estima comportamiento en desarrollo y no autoriza afirmaciones de generalización independiente.

## Interpretación prevista

- Una ventaja de SAM con 25 etiquetas que disminuya al llegar a 400 apoyaría la hipótesis de eficiencia de datos por preentrenamiento.
- Una mejora de `probe→caja+punto→SAM` sobre el probe aislaría la utilidad del mask decoder una vez recuperado el componente espacial del prompt.
- Si U-Net iguala o supera al sistema automático con pocas etiquetas, el argumento de eficiencia de SAM no estaría respaldado en este protocolo.
- Las curvas no miden tiempo humano de anotación. Cada imagen presupuestada supone una máscara completa disponible.

## Alcance

Es un estudio controlado de mecanismos y eficiencia de etiquetas. La U-Net es una variante local compacta, no una reproducción exacta ni un modelo de estado del arte. SAM usa preentrenamiento genérico y resolución interna diferente. No se reutiliza el test ni se presenta el mejor resultado de desarrollo como rendimiento final.
