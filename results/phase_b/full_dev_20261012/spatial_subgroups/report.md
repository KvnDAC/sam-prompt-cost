# Subgrupos de la ablación espacial (dev)

Análisis descriptivo posterior a observar la ablación. No establece causas ni generalización.

| Grupo | n | Sin prompt | Punto GT | Punto desplazado 2 radios | Caja GT | Caja ×2 |
|---|---:|---:|---:|---:|---:|---:|
| all | 100 | 0.009 [0.001, 0.023] | 0.683 [0.615, 0.746] | 0.153 [0.131, 0.176] | 0.935 [0.928, 0.942] | 0.322 [0.256, 0.392] |
| area_bin=5-15% | 38 | 0.000 [0.000, 0.001] | 0.721 [0.616, 0.816] | 0.140 [0.105, 0.176] | 0.942 [0.934, 0.950] | 0.304 [0.205, 0.405] |
| area_bin=<5% | 25 | 0.002 [0.000, 0.006] | 0.576 [0.436, 0.718] | 0.112 [0.077, 0.156] | 0.931 [0.924, 0.939] | 0.669 [0.558, 0.766] |
| area_bin=>=15% | 37 | 0.022 [0.001, 0.058] | 0.716 [0.607, 0.811] | 0.194 [0.159, 0.230] | 0.931 [0.914, 0.945] | 0.105 [0.043, 0.177] |
| multiple_components=0 | 95 | 0.009 [0.001, 0.024] | 0.683 [0.613, 0.749] | 0.150 [0.127, 0.174] | 0.936 [0.928, 0.943] | 0.320 [0.250, 0.390] |
| multiple_components=1 | 5 | 0.009 [0.000, 0.026] | 0.693 [0.537, 0.820] | 0.208 [0.151, 0.269] | 0.927 [0.909, 0.952] | 0.359 [0.241, 0.559] |
| touches_edge=0 | 79 | 0.008 [0.000, 0.024] | 0.657 [0.580, 0.732] | 0.130 [0.110, 0.153] | 0.933 [0.926, 0.941] | 0.370 [0.294, 0.447] |
| touches_edge=1 | 21 | 0.011 [0.000, 0.030] | 0.782 [0.648, 0.887] | 0.238 [0.183, 0.292] | 0.942 [0.923, 0.958] | 0.140 [0.048, 0.257] |

Las direcciones se promedian por imagen antes del bootstrap. Los grupos se solapan; los IC no corrigen selección posterior ni multiplicidad.
En imágenes con varias componentes, el oráculo recibe un prompt por componente y une las máscaras: también recibe más información.
El panel muestra seis casos con mayor diferencia caja GT − punto GT; es una selección ilustrativa, no una muestra representativa.
