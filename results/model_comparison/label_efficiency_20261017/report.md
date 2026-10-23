# Eficiencia de etiquetas: resultados de desarrollo

Subconjuntos anidados de 25, 100 y 400 máscaras; tres semillas; test no evaluado.

| Método | Máscaras | Dice | IC 95 % por imagen | DE semillas |
|---|---:|---:|---:|---:|
| probe | 25 | 0.7156 | [0.6735, 0.7544] | 0.0070 |
| sam_auto | 25 | 0.7821 | [0.7321, 0.8281] | 0.0081 |
| unet_endpoint | 25 | 0.5103 | [0.4635, 0.5552] | 0.0159 |
| unet_selected | 25 | 0.5254 | [0.4786, 0.5704] | 0.0200 |
| probe | 100 | 0.7416 | [0.6994, 0.7802] | 0.0048 |
| sam_auto | 100 | 0.8061 | [0.7616, 0.8460] | 0.0090 |
| unet_endpoint | 100 | 0.6198 | [0.5713, 0.6671] | 0.0238 |
| unet_selected | 100 | 0.6338 | [0.5895, 0.6772] | 0.0197 |
| probe | 400 | 0.7488 | [0.7065, 0.7879] | 0.0004 |
| sam_auto | 400 | 0.8190 | [0.7719, 0.8609] | 0.0009 |
| unet_endpoint | 400 | 0.6964 | [0.6579, 0.7341] | 0.0108 |
| unet_selected | 400 | 0.7029 | [0.6664, 0.7385] | 0.0209 |

## Diferencias pareadas exploratorias

| Contraste | Diferencia Dice | IC 95 % |
|---|---:|---:|
| sam_auto_minus_probe_n25 | 0.0665 | [0.0435, 0.0889] |
| sam_auto_minus_unet_endpoint_n25 | 0.2718 | [0.2185, 0.3243] |
| sam_auto_minus_probe_n100 | 0.0645 | [0.0451, 0.0840] |
| sam_auto_minus_unet_endpoint_n100 | 0.1863 | [0.1333, 0.2390] |
| sam_auto_minus_probe_n400 | 0.0702 | [0.0475, 0.0938] |
| sam_auto_minus_unet_endpoint_n400 | 0.1226 | [0.0757, 0.1692] |
| probe_n100_minus_n25 | 0.0261 | [0.0155, 0.0374] |
| probe_n400_minus_n100 | 0.0071 | [-0.0024, 0.0159] |
| sam_auto_n100_minus_n25 | 0.0240 | [0.0032, 0.0469] |
| sam_auto_n400_minus_n100 | 0.0129 | [-0.0046, 0.0314] |
| unet_endpoint_n100_minus_n25 | 0.1096 | [0.0811, 0.1392] |
| unet_endpoint_n400_minus_n100 | 0.0766 | [0.0480, 0.1057] |

El endpoint fijo es primario; `unet_selected` es secundario porque usa selección en desarrollo.
Los IC remuestrean imágenes después de promediar semillas. No se evaluó test.
