# Borrador de resumen extendido WAIMLAp 2026

**Revisión del 22 de octubre (vigente):** [texto](revision_20261022_review/abstract_draft.md). Corrige sobre la del 18:

- La ventaja de SAM con 400 máscaras en eficiencia de etiquetas se atribuye al presupuesto fijo de 1.500 actualizaciones. Con las mismas 400 máscaras, la receta larga de U-Net llega a 0,8267, y la figura 1B ahora lo muestra.
- "U-Net matches" pasa a "comparable (difference unresolved)", coherente con el IC.
- La ausencia de máscaras vacías en SAM se explica por diseño: el probe se activa siempre. El decoder agranda el componente que recibe ([control post hoc](../../results/model_comparison/negative_postprocessing_20261022/summary.json)).
- Referencias completas, con Mazurowski, PP-SAM y Stable-SAM como trabajos previos.
- El generador ya no duplica cabecera ni pie en la página 2, y `--flow` evita el salto forzado antes de Results.

El Word y el PDF de esta revisión aún no se han generado. Requieren `python-docx` y `pywin32`; después hay que comprobar que siguen siendo dos páginas:

```powershell
python docs/waimlap_2026/generar_borrador.py --source docs/waimlap_2026/revision_20261022_review/abstract_draft.md --output_dir docs/waimlap_2026/revision_20261022_review --flow --raster_header docs/waimlap_2026/revision_20261018_complete/header_strip.png --pdf
```

La entrega oficial (consultada el 22 de octubre en <https://waimlap.org/registration/>) vence el **23 de octubre de 2026**. Se envía un único ZIP de hasta 10 MB con `Abstract_TitleOfWork.pdf` y `Poster_TitleOfWork.pdf` (A0 vertical, plantilla oficial).

**Borrador integrado final (18 de octubre):** reúne dependencia del prompt, diagnóstico de candidatas/selección, U-Net madura, eficiencia de etiquetas, transferencia CVC-ClinicDB, negativos HyperKvasir, latencia y una figura de cuatro paneles. [Texto de dos páginas](revision_20261018_complete/abstract_draft.md), [Word basado en la plantilla](revision_20261018_complete/Abstract_TheInformationalCostOfSpatialPromptsInSAM.docx) y [PDF verificado de dos páginas](revision_20261018_complete/Abstract_TheInformationalCostOfSpatialPromptsInSAM.pdf). El manuscrito maestro detallado está en [`WAIMLAP_DRAFT.md`](../../WAIMLAP_DRAFT.md). Autoría y afiliaciones continúan pendientes; no se ha enviado.

**Revisión con control negativo (18 de octubre):** incorpora las 1.009 imágenes `cecum` de HyperKvasir y el hallazgo de que SAM automático produce menos área falsa que U-Net, aunque ambos generan máscaras en más del 99,8 % de las imágenes. [Texto](revision_20261018_specificity/abstract_draft.md), [Word](revision_20261018_specificity/Abstract_TheInformationalCostOfSpatialPromptsInSAM.docx) y [PDF verificado de dos páginas](revision_20261018_specificity/Abstract_TheInformationalCostOfSpatialPromptsInSAM.pdf). La autoría continúa pendiente y el documento no ha sido enviado.

**Revisión con validación externa:** incorpora la curva 25/100/400 y la transferencia congelada a los 612 frames/29 secuencias de CVC-ClinicDB. [Texto](revision_20261017_external/abstract_draft.md), [Word](revision_20261017_external/Abstract_TheInformationalCostOfSpatialPromptsInSAM.docx) y [PDF de dos páginas](revision_20261017_external/Abstract_TheInformationalCostOfSpatialPromptsInSAM.pdf). La autoría continúa pendiente y el documento no ha sido enviado. En esta revisión el encabezado se estabiliza como una franja rasterizada desde la propia plantilla porque Word desplazaba sus objetos flotantes entre páginas; la adaptación está registrada en `verification.json` y fue inspeccionada visualmente.

Este material es un **borrador para revisión**, no un envío realizado. Usa resultados de desarrollo completados; la comparación entre familias está pendiente.

**Revisión del 17 de octubre:** incorporó nueve corridas RGB completas, el prior espacial, el IC pareado SAM−U-Net que incluye cero y el costo completo desde RGB. [PDF actualizado](revision_20261017/Abstract_TheInformationalCostOfSpatialPromptsInSAM.pdf), [Word actualizado](revision_20261017/Abstract_TheInformationalCostOfSpatialPromptsInSAM.docx) y [texto](revision_20261017/abstract_draft.md). Conserva aparte los borradores iniciales del 16 enlazados abajo. La comparación ligera está completa; convergencia/resolución, SAM 2.1/especialista y validación externa siguen pendientes. La autoría aún debe confirmarse.

- [Texto editable en inglés](abstract_draft.md).
- [Word basado en la plantilla oficial suministrada](Abstract_TheInformationalCostOfSpatialPromptsInSAM.docx).
- [PDF de revisión, dos páginas](Abstract_TheInformationalCostOfSpatialPromptsInSAM.pdf).
- [Decisión científica y revisión de todas las ramas](../decision_investigacion_waimlap.md).
- [Registro técnico de generación](verification.json).

La plantilla original, en la raíz del repositorio, queda intacta. Se conservan exactamente los bytes de la cabecera, el pie y las imágenes del paquete Word. El contenido reemplaza instrucciones y ejemplos; se mantienen las cuatro secciones solicitadas y referencias numéricas consistentes.

## Qué queda antes de enviar

La autoría, su orden, afiliaciones y correspondencia requieren una fuente formal: se buscaron en todas las ramas, pero solo se encontraron metadatos de commits. Esos metadatos no autorizan inferir una lista de autores. Los campos están marcados en el documento y deben completarse antes de usarlo como versión de envío.

El borrador se exportó con Word y se verificaron dos páginas A4 mediante un segundo lector PDF. Las dos páginas se renderizaron y revisaron visualmente. Cabecera y logos del Word conservan los bytes del original. El recuento queda en `verification.json`; al cambiar autores o texto hay que volver a comprobarlo. No afirmar que el PDF está listo para enviar si conserva campos pendientes.

La [convocatoria oficial de pósteres](https://waimlap.org/registration/) pide abstract y póster PDF, ambos en un ZIP de hasta 10 MB, con fecha publicada del 16 de octubre de 2026. El póster A0 no se ha preparado en esta tarea. No confundir la entrega del abstract con un proyecto ampliado ya terminado.

## Fuentes y decisiones del borrador

Las cifras de la tabla salen de [summary.json](../../results/phase_b/full_dev_20261012/summary.json). El control de contexto usa [conditioning_diagnostic/summary.json](../../results/phase_b/full_dev_20261012/conditioning_diagnostic/summary.json); las candidatas se documentan en [output_selection/report.md](../../results/phase_b/full_dev_20261012/output_selection/report.md).

No se incluyen cifras históricas de test ni los pilotos U-Net para evitar mezclarlas con dev100. Las cajas/puntos GT se identifican como oráculos. Se explicita supervisión, meanpoint, selección en dev, presupuesto limitado de entrenamiento, ausencia de evaluación independiente nueva y de medición de tiempo humano. Los demás modelos se presentan como trabajo pendiente.

Bibliografía primaria comprobada:

1. [SAM, Kirillov et al., ICCV 2023](https://arxiv.org/abs/2304.02643).
2. [Can SAM Segment Polyps?, Zhou et al., 2023](https://arxiv.org/abs/2304.07583).
3. [Self-Prompting, Wu, Zhang y Elbatel, 2023](https://arxiv.org/abs/2308.07624).
4. [AutoSAM, Shaharabany, Dahan, Giryes y Wolf, 2023](https://arxiv.org/abs/2306.06370).
5. [Kvasir-SEG, Jha et al., MMM 2020](https://datasets.simula.no/kvasir-seg/).

## Regenerar

Desde la raíz, usando el entorno existente con `python-docx`:

```powershell
.\.venv\Scripts\python.exe docs/waimlap_2026/generar_borrador.py
```

En Windows con Word disponible, añadir `--pdf` intenta exportar y comprobar páginas mediante una instancia oculta propia. Puede requerir que Word ya tenga completada su configuración inicial y acceso a automatización local. El script no envía archivos, no cambia documentos abiertos del usuario y no instala dependencias. Para otros entornos, abrir el Word generado y exportarlo manualmente a PDF conservando el formato.

Para regenerar la revisión del 17 sin sobrescribir la inicial:

```powershell
.\.venv\Scripts\python.exe docs/waimlap_2026/generar_borrador.py --source docs/waimlap_2026/revision_20261017/abstract_draft.md --output_dir docs/waimlap_2026/revision_20261017 --pdf
```

Para la revisión externa, que usa el encabezado estabilizado:

```powershell
.\.venv\Scripts\python.exe docs/waimlap_2026/generar_borrador.py --source docs/waimlap_2026/revision_20261017_external/abstract_draft.md --output_dir docs/waimlap_2026/revision_20261017_external --page_break --raster_header docs/waimlap_2026/revision_20261017_external/header_strip.png --pdf
```
