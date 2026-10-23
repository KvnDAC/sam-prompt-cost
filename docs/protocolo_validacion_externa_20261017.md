# Protocolo congelado: transferencia Kvasir-SEG → CVC-ClinicDB

## Pregunta

¿La mitigación automática basada en embeddings de SAM conserva su calidad fuera de Kvasir-SEG, y cómo se compara con una U-Net madura entrenada únicamente en Kvasir?

## Datos y separación

- Fuente de entrenamiento y selección: Kvasir-SEG `train400/dev100`.
- Destino externo: CVC-ClinicDB completo, 612 frames de 29 secuencias, 384 × 288, con máscaras binarias.
- No se entrena, calibra, selecciona ni modifica ningún parámetro con CVC-ClinicDB.
- Los modelos, umbrales y posprocesamiento quedan congelados antes de producir la primera predicción externa.
- Se auditan 612 pares, numeración 1–612, dimensiones coincidentes, máscaras no vacías, hashes y duplicados exactos respecto de Kvasir.

## Sistemas congelados

1. **Probe SAM:** probe lineal de Fase B, tres semillas, entrenado en `train400`; umbral elegido previamente en `dev100`.
2. **Probe → caja+punto → SAM:** misma máscara del probe, componente mayor, punto interior y caja, con SAM ViT-B congelado. No usa GT externa para construir prompts.
3. **U-Net 256 endpoint:** tres endpoints de la continuación de resolución, partiendo de la época 30 a 128 y continuados 15 épocas a 256. Se usa el endpoint fijo, que fue el estimando principal previo.

No se incluyen cajas ni puntos derivados de GT, selección oráculo de candidatas ni checkpoints publicados que pudieran haber visto CVC-ClinicDB.

## Métricas

- Dice e IoU por frame, calculados después de restaurar la resolución 384 × 288.
- Para cada método se promedian primero las tres semillas por frame.
- Se reporta media estándar por frame para comparabilidad con trabajos previos.
- El IC 95 % usa bootstrap por las 29 secuencias completas (10 000 réplicas); los frames de una secuencia nunca se separan dentro de una réplica.
- Se reporta además la media macro de las 29 medias por secuencia.
- Contrastes pareados principales: SAM automático menos U-Net endpoint y SAM automático menos probe.

## Regla de interpretación

- El resultado externo es confirmatorio para transferencia entre estos dos datasets, pero no prueba generalización clínica universal.
- CVC-ClinicDB contiene únicamente frames con pólipos; no mide especificidad en negativos.
- Los frames consecutivos están correlacionados. El bootstrap agrupado reduce la falsa precisión, pero 29 secuencias siguen siendo una muestra limitada.
- No se cambiará el método después de observar resultados. Cualquier análisis nuevo se marcará como exploratorio y requerirá otro conjunto.

## Procedencia

La página oficial de Grand Challenge describe el dataset, la estructura y la restricción a investigación/educación, pero actualmente no expone un archivo en su apartado de descarga. La copia utilizada procede del enlace público `CVC-ClinicDB.rar` señalado por un repositorio académico de segmentación; se conserva el README original y se registra SHA-256 del archivo. No se redistribuyen las imágenes en Git.
