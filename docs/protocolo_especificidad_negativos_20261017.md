# Protocolo congelado: control negativo externo en HyperKvasir

## Pregunta

Cuando no hay un pólipo etiquetado en la imagen, ¿cuánto tejido marca de forma espuria el sistema automático construido sobre SAM y cómo se compara con el *probe* lineal y U-Net?

## Datos

- Fuente externa: HyperKvasir, clase `cecum` dentro de `lower-gi-tract/anatomical-landmarks`.
- Se usarán las 1,009 imágenes publicadas en esa carpeta, sin selección posterior a observar predicciones.
- La clase es un **control negativo de nivel imagen para pólipo**. HyperKvasir no proporciona máscaras de pólipo para estas imágenes ni identificadores de paciente/video. Por ello, el experimento estima falsas activaciones en este conjunto y no especificidad clínica por paciente.
- Se verificará el SHA-256 local de cada archivo contra el hash publicado por OSF y se descartarán archivos corruptos, duplicados exactos o no decodificables antes de inferencia.

## Modelos congelados

Se reutilizan sin ajuste externo los tres *seeds* de:

1. *Probe* lineal sobre el *embedding* de SAM ViT-B, con su umbral aprendido en `train400/dev` de Kvasir-SEG.
2. `SAM auto`: mayor componente del *probe* convertido en caja + punto para el *mask decoder* de SAM. Si el *probe* está vacío, la salida es vacía.
3. U-Net ligera a 256 px, punto final de 45 épocas del experimento de resolución.

No se calibra umbral, arquitectura ni pesos en HyperKvasir.

## Métricas preespecificadas

Para cada imagen, método y semilla se calcula:

- fracción del área de imagen predicha como pólipo;
- fracción ocupada por el componente conexo más grande;
- indicador de máscara no vacía;
- indicadores de área espuria mayor o igual a 0.5 %, 1 % y 5 %;
- indicador de componente mayor o igual a 1 %.

El resumen principal promedia primero las tres semillas por imagen. Se informan media e IC 95 % *bootstrap* por imagen (10,000 réplicas), mediana y percentil 95. La comparación principal es `SAM auto - U-Net` en área falsa media y tasa de área falsa >=1 %. Valores negativos favorecen a SAM auto.

## Alcance y criterio de interpretación

Dice/IoU no se usarán: con una máscara de referencia vacía colapsan el error espacial a una decisión binaria. Una tasa de máscara no vacía puede ser dominada por pocos píxeles, por eso el área y el mayor componente son los resultados principales. Los IC por imagen no corrigen correlación entre imágenes del mismo paciente porque HyperKvasir no publica ese vínculo; el resultado se presentará como control técnico externo, no como validación clínica.
