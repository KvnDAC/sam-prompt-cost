# Figuras del póster WAIMLAp 2026

El póster A0 (841 × 1189 mm, dos columnas) se edita en Canva. Cada figura se genera al tamaño exacto de su hueco: 96 px de Canva equivalen a una pulgada, y los PNG se exportan a 300 dpi.

| Figura | Hueco en Canva | Fuente | Cómo se genera |
|---|---|---|---|
| Diagrama de metodología con ejemplo real | 1355 × 1360 px | [`method_diagram.svg`](method_diagram.svg), dibujado a mano, y las teselas reales de `scripts/37_make_poster_examples.py` | `python scripts/37_make_poster_examples.py` y luego `python scripts/36_render_poster_diagram.py` → `method_diagram.png` |
| 1 · Dependencia del prompt | 660 × 405 px | Fase B + prior fijo | `python scripts/35_make_poster_figures.py` |
| 2 · Candidatas vs. selección | 660 × 405 px | `output_selection/summary.json` | ídem |
| 3 · Eficiencia de etiquetas | 660 × 405 px | eficiencia de etiquetas + U-Net de resolución | ídem |
| 4 · CVC-ClinicDB y negativos HyperKvasir | 660 × 405 px | protocolos externos congelados | ídem |

Las figuras de resultados se guardan en [`results/poster_20261023/`](../../../results/poster_20261023/), en PNG con fondo transparente y en PDF vectorial. La lista de JSON de origen está en `figure_provenance.json`. Ninguna figura vuelve a ejecutar modelos: solo leen los resúmenes congelados, así que las cifras coinciden con las del abstract.

El ejemplo real es el caso de dev `cju40sdwukv3k0755y99ug1k8`, procesado por el mismo SAM congelado sin prompt, con punto GT y con caja GT (Dice 0.000 / 0.012 / 0.958). Las imágenes originales de Kvasir-SEG no están en el repo, así que el fotograma se recupera de la figura publicada `results/phase_b/full_dev_20261012/spatial_cases/predictions.png`:

- la máscara GT es el contorno cian rellenado;
- el fotograma limpio se obtiene borrando ese contorno;
- las salidas de SAM vienen de `predictions.npz`;
- la caja oráculo es el rectángulo que encierra la GT.

El caso se eligió a posteriori en la Fase B (mayor diferencia de Dice entre caja y punto GT), por eso en el póster se rotula como "selected case": ilustra el problema, no es representativo. La procedencia está en `examples_provenance.json`.

El diagrama se rasteriza con Edge o Chrome en modo headless para que el texto use Arial, igual que el póster. El SVG es la fuente editable; si se cambia, hay que volver a ejecutar el script de renderizado.

Paleta, tomada de la plantilla: morado `#4B2A82` (SAM congelado / Auto SAM), texto `#28282D`, lavanda `#E6E1F0`, teal `#238B8E` (probe), naranja `#E76F51` (U-Net) y gris `#7A7A7A` (oráculos).
