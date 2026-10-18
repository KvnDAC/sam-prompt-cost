# Continuación controlada de U-Net: 128 frente a 256

Fijado el 17 de octubre de 2026 antes de ejecutar esta continuación. Es un seguimiento motivado por las curvas de desarrollo de la comparación anterior; no es una evaluación confirmatoria independiente.

Se mantienen las 400 imágenes de entrenamiento y las 100 de desarrollo, las tres semillas (0, 1, 2), la arquitectura U-Net compacta de anchura 8 con GroupNorm, Adam a 0.001, batch 8, BCE+Dice y las inversiones horizontales/verticales anteriores. No se evalúa nuevamente test. Los resultados anteriores se conservan en sus carpetas originales.

Cada semilla tiene dos brazos: lado largo 128 o 256 píxeles, con la misma convención de padding y restauración al tamaño original. Ambos parten del **mismo checkpoint del último paso de la época 30**, incluyendo pesos, momentos de Adam y estados RNG de Python, NumPy, PyTorch y del DataLoader. Cada brazo realiza exactamente 15 épocas adicionales: 750 actualizaciones, hasta 45 épocas totales. No se interrumpe por desarrollo. Es un presupuesto explícito para comprobar sensibilidad al entrenamiento y a la resolución; no asegura convergencia. Los brazos reciben el mismo orden de imágenes y las mismas decisiones aleatorias de augmentación, aunque la representación y los gradientes cambian con la resolución.

La medida principal es Dice por imagen en el **endpoint fijo** de la época 45. También se conserva el estado inicial a cada resolución: comparar esos estados aísla el cambio de resolución en inferencia sin entrenamiento adicional. Como análisis secundario se selecciona el checkpoint con mayor Dice medio dev entre el estado inicial y las 15 épocas nuevas, por separado en cada brazo. Se reinicia la selección de manera simétrica: reutilizar el mejor score histórico a 128 para seleccionar a 256 mezclaría evaluaciones distintas. El mejor checkpoint histórico de 30 épocas aparece como referencia separada, no como candidato de selección de esta etapa.

Se publican medias por semilla, desviación entre semillas, scores por imagen y curvas. Los IC del 95 % usan 10 000 remuestreos pareados de imágenes después de promediar semillas, semilla bootstrap 0. Contrastes: resolución a pesos iniciales fijos; 256−128 en endpoint y seleccionados; endpoint−inicial de cada brazo; seleccionados frente al mejor histórico y frente al pipeline SAM automático ya ejecutado. Son exploratorios: los IC no corrigen reutilización de desarrollo, selección, multiplicidad ni dependencia por paciente/video. No se usarán para afirmar equivalencia si contienen cero.

El script `scripts/26_unet_resolution_continuation.py` comprueba IDs, hashes del código y de los 1 000 archivos train/dev contra el plan anterior, y registra los hashes de los tres checkpoints padre. El plan nuevo incluye el hash de su propio código. Se escribe antes de entrenar y se rechazan cambios al reanudar. Las fuentes anteriores se mantienen intactas. Cada época guarda modelo, optimizador y RNG de manera atómica. Se comprueba que el checkpoint seleccionado reproduce los 100 scores individuales, y se exporta además el endpoint fijo. La prueba de inicialización verifica que ambos brazos heredan los mismos pesos/Adam/RNG y recalculan la selección a su resolución.

Ejecución en CPU, cuatro hilos, sin entrenamientos concurrentes:

```powershell
.\.venv\Scripts\python.exe scripts/26_unet_resolution_continuation.py --plan_only
.\.venv\Scripts\python.exe scripts/26_unet_resolution_continuation.py
```

Pesos y estados: `runs/model_comparison/unet_resolution_20261017/`. Artefactos revisables: `results/model_comparison/unet_resolution_20261017/`. La comparación de calidad con SAM mantiene explícitas las diferencias de preentrenamiento y entrada (SAM ViT-B a 1024). Más resolución implica más cómputo; igualdad de pasos no significa igualdad de tiempo.
