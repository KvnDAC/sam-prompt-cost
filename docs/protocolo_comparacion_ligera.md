# Comparación ligera: protocolo fijado antes del entrenamiento

Fecha del cliente: 16 de octubre de 2026. Ampliación autorizada por el equipo: mejorar experimentación e incorporar dos comparadores ligeros. No se modifica el plan ni los siete archivos congelados de Fase B.

**Cierre del 17 de octubre:** nueve corridas completas, 21 tests aprobados y perfil SAM desde RGB terminado. [Resultados e interpretación](resultados_experimentacion_20261017.md). Se conservan abajo las decisiones fijadas antes de ejecutar; las referencias futuras a las mediciones describen ese momento previo.

## Preguntas

1. ¿Qué desempeño logran tres segmentadores RGB compactos con los mismos 400 train y 100 dev usados en Fase B?
2. ¿Cómo cambian calidad, parámetros y latencia al sustituir convoluciones estándar y decoder por variantes ligeras?
3. ¿Qué mejora adicional obtiene la mitigación SAM frente al probe y a estos controles, declarando las diferencias de preentrenamiento, entrada y receta?

La comparación es exploratoria y bajo presupuesto común. No constituye búsqueda de estado del arte ni prueba causal que aísle una única operación: las variantes cambian más componentes. El control de contexto constante del generador SAM sigue siendo un experimento separado, pendiente.

## Modelos

| Nombre local | Diseño | Interpretación |
| --- | --- | --- |
| `unet` | U-Net de cuatro etapas, ancho inicial 8, GroupNorm, convoluciones estándar y skips concatenados | Control directo compacto; diferente del piloto anterior de ancho 16/entrada 256 |
| `ds_unet` | Tres reducciones, convoluciones depthwise/pointwise y decoder bilineal con skips | Control de eficiencia inspirado en convoluciones separables; no es MobileNet publicado |
| `fcn_lite` | CNN stride 8, fusión aditiva de logits a strides 2/4/8 y salida bilineal | Control multiescala; no es reproducción de FCN-8s/VGG |

Son implementaciones locales declaradas, sin preentrenamiento. Antecedentes: [U-Net](https://arxiv.org/abs/1505.04597), [FCN](https://arxiv.org/abs/1411.4038) y [convoluciones separables en MobileNets](https://arxiv.org/abs/1704.04861). Añaden dos arquitecturas de control, no dos modelos clínicamente validados ni una comparación con especialistas modernos.

## Receta común fijada

- Train400/dev100 íntegros, IDs fijos existentes; tres semillas 0/1/2. Sin evaluación de test.
- RGB [0,1], lado mayor 128, proporción conservada y padding superior izquierdo. GT nearest-neighbor, máscara válida para BCE+Dice; predicción restaurada a resolución original antes de Dice/IoU.
- Flips H/V independientes p=0,5 dentro del rectángulo válido, solo en train. Imágenes preprocesadas en RAM para reducir I/O repetido.
- Adam, LR constante 0,001, lote 8, CPU con cuatro hilos. No barrido de hiperparámetros en esta ronda; mismo presupuesto inicial por arquitectura.
- Máximo 30 épocas, mínimo 15 antes de parar, paciencia 8 épocas sin aumento del Dice dev. Selección del máximo Dice medio, incluyendo inicialización; empate conserva época anterior. Umbral fijo logit >0. No TTA ni ajuste posterior de umbral.
- Tres semillas para cada arquitectura; se conservan resultados negativos y épocas de borde del presupuesto. Alcanzar el tope no demuestra convergencia; revisar curvas antes de proponer una ronda nueva.
- Sin igualar el conocimiento previo de SAM: este usa preentrenamiento y entrada 1024. Los controles RGB usan 128 por viabilidad CPU. Es una comparación práctica bajo igual presupuesto de etiquetas locales, con resolución declarada, no un contraste de arquitecturas aislado.

## Trazabilidad y reanudación

`scripts/20_compare_light_models.py` guarda plan inmutable con hashes de código y datos antes del entrenamiento. Si plan, datos o código cambian, exige carpeta nueva. Snapshots de código y checkpoints completos quedan en `runs/model_comparison/light_train400_20261016/`. Cada época guarda pesos actuales, Adam, RNG de Python/NumPy/Torch y generador de orden de muestras, además de historial y estado de parada. La reanudación es desde el último límite de época guardado, sin mezclar corridas ni reiniciar semillas completas.

La escritura de checkpoints/JSON usa temporal y reemplazo. Cada checkpoint seleccionado se recarga y debe reproducir los 100 Dice individuales a tolerancia 1e−8 antes de cerrar la corrida. Los artefactos ligeros finales se exportan a `results/model_comparison/light_train400_20261016/`; pesos y estado de optimizador permanecen ignorados por Git en `runs/`.

## Métricas y costo

Dice/IoU por imagen, media de tres semillas e IC 95 % mediante 3.000 bootstrap de imágenes después de promediar semillas. DE entre semillas por separado. Diferencias pareadas DS-U-Net menos U-Net y FCN-Lite menos U-Net, con emparejamiento por ID y sin asumir independencia de semillas/imágenes repetidas. Son contrastes exploratorios sin ajuste por multiplicidad ni por selección en dev.

Latencia: CPU, batch 1, cinco calentamientos y veinte repeticiones; incluye preprocesado RGB, red completa, restauración y umbral, con imágenes decodificadas en RAM. Excluye lectura de disco y cálculo de métricas. Se reportan mediana/p95, parámetros, tiempo de entrenamiento+dev y máximo **muestreado** de RSS del proceso; este último incluye datos en RAM y no equivale a memoria exclusiva del modelo. No comparar estos tiempos con SAM decoder-only sobre caché. Falta medir el pipeline SAM completo con el mismo alcance.

## Verificación previa

Cuatro tests nuevos aprobados: ambos modelos aprenden una región sintética visible (Dice >0,95) y admiten dimensiones impares; restaurar estado reproduce actualización Adam/RNG/orden; emparejamiento por ID rechaza faltantes/duplicados; particiones únicas y disjuntas obligatorias. Estas pruebas no garantizan aprendizaje de pólipos ni validación clínica. Los cuatro tests U-Net previos siguen siendo controles separados.

## Ejecución

```powershell
.\.venv\Scripts\python.exe scripts/20_compare_light_models.py --run_root runs/model_comparison/light_train400_20261016 --output results/model_comparison/light_train400_20261016
```

El mismo comando reanuda solamente si coincide el plan. Cambiar recetas requiere nueva carpeta y documentar que la decisión usó resultados previos. El estado activo está en `runs/model_comparison/light_train400_20261016/status.json`; el informe final solo existe cuando terminaron las nueve corridas.

## Auditoría de datos y control espacial adicional

`21_audit_split_similarity.py` compara hashes de archivo, de píxeles RGB decodificados y un screening DCT perceptual de 63 bits (distancia Hamming ≤4) entre train400/dev100. El resultado en `results/model_comparison/split_audit_20261016/summary.json` no contiene duplicados exactos entre particiones ni candidatos del screening. No demuestra independencia por paciente/video ni excluye variantes de la misma lesión. No cambió las particiones ni evaluó test.

Antes de ejecutarlo se fija otro control, `22_fixed_spatial_prior.py`: obtener de las 400 GT de entrenamiento la mediana coordenada a coordenada de cajas y puntos interiores normalizados. Aplicar ese mismo prior a cada imagen dev, escalado solo con sus dimensiones; una consulta por imagen, sin usar GT dev para generar prompts. Evaluar punto fijo, caja fija y caja+punto fijos, salida única y umbral 0. Contraste exploratorio: pipeline probe→caja+punto (media de tres semillas previas) menos caja+punto fijos, emparejado por ID. Es un control supervisado de prior espacial, no una evaluación sin supervisión ni el control del generador de igual capacidad.

Dos tests verifican que el prior normalizado solo consulta los IDs suministrados de train y rechaza máscaras vacías. El problema del test previo de rutas se resolvió usando una carpeta ordinaria y única del workspace: `tempfile` con modo privado 0700 causaba un ACL incompatible con los subprocesos restringidos de Windows. La limpieza valida que la ruta esté dentro de `runs/`. Las 17 pruebas pasaron después de ese cambio; los dos controles nuevos se verifican por separado al añadirlos.

`23_compare_completed_models.py` exige nueve corridas terminadas, comprueba los hashes de entrenamiento y vuelve a medir latencia RGB después de finalizar los procesos de entrenamiento. Combina los scores con los artefactos SAM existentes, empareja por ID y añade subgrupos descriptivos usando los cortes de área/borde/componentes ya definidos en Fase B, sin optimizar cortes nuevos. Los primeros cuatro IDs dev y semilla 0 forman el panel cualitativo, sin elegir casos por score. Este análisis se prepara después de inspeccionar FCN-Lite; no se presenta como contraste confirmatorio nuevo.

## Medición completa de SAM, preparada antes de ejecutarla

`24_time_full_sam_pipeline.py` mide el pipeline automático seleccionado de semilla 0, incluyendo **encoder desde RGB**, probe, componentes, prompts, decoder y restauración. CPU, cuatro hilos, FP32, batch 1, cinco calentamientos y veinte repeticiones sobre los mismos primeros veinte IDs dev que los modelos RGB, con imágenes decodificadas en RAM. Se excluyen carga de pesos, disco, lectura de GT y métricas del intervalo cronometrado. Se verifican por separado las veinte predicciones usando caché contra los CSV previos y las veinticinco ejecuciones desde RGB contra esa misma referencia, a tolerancia 1e−8 en Dice. Si difieren se detiene en vez de reportar costo de otro comportamiento.

Se reportan mediana, p95 y desglose preprocesado/encoder/resto, parámetros totales únicos SAM+probe y entrenables. Es latencia local de una semilla y veinte imágenes, no benchmark universal ni tiempo humano. Se ejecutará después de los entrenamientos y las mediciones RGB para evitar carga concurrente de esta sesión. Las mediciones de latencia originales dentro del trainer se conservan, pero el informe común usa las mediciones posteriores.
