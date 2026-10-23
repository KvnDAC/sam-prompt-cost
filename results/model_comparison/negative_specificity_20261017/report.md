# Control negativo externo: HyperKvasir cecum

1.009 imágenes; modelos/umbrales congelados en Kvasir-SEG; sin ajuste externo.

| Método | Área falsa media (IC 95 %) | P95 área | Tasa >=1 % | Componente mayor P95 | Vacío |
|---|---:|---:|---:|---:|---:|
| probe | 0.0581 [0.0552, 0.0611] | 0.1552 | 0.8794 | 0.1196 | 0.0000 |
| sam_auto | 0.0521 [0.0488, 0.0554] | 0.1631 | 0.7942 | 0.1628 | 0.0000 |
| unet | 0.1733 [0.1690, 0.1776] | 0.2922 | 0.9931 | 0.2814 | 0.0017 |

## Diferencias pareadas (SAM auto menos comparador)

| Comparación | Área falsa media | IC 95 % | Diferencia tasa >=1 % | IC 95 % |
|---|---:|---:|---:|---:|
| sam_auto_minus_unet | -0.1212 | [-0.1267, -0.1159] | -0.1989 | [-0.2217, -0.1767] |
| sam_auto_minus_probe | -0.0061 | [-0.0076, -0.0045] | -0.0852 | [-0.1008, -0.0704] |

La etiqueta es de clase anatómica, no una máscara negativa revisada píxel a píxel. Los IC remuestrean imágenes porque no hay IDs públicos de paciente/video; no representan especificidad clínica por paciente.
