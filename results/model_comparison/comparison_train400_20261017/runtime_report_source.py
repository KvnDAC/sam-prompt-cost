"""Attach measured full-SAM runtime to completed quality/latency comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison',type=Path,required=True)
    ap.add_argument('--sam_runtime',type=Path,required=True)
    a=ap.parse_args()
    target=a.comparison/'summary.json'
    results=json.loads(target.read_text(encoding='utf-8'))
    runtime=json.loads((a.sam_runtime/'summary.json').read_text(encoding='utf-8'))
    if not runtime['cached_and_fresh_predictions_reproduced'] or runtime['test_evaluated']:
        raise ValueError('Require reproduced development-only runtime')
    name='Probe → caja + punto → SAM'
    table=[dict(row) for row in results['table']]
    for row in table:
        if row['method']==name:
            row.update(latency_median_ms=runtime['median_ms'],latency_p95_ms=runtime['p95_ms'],
                       latency_seed=0,parameters_total=runtime['parameters_total_unique'])
    final={**results,'table':table,'sam_runtime':runtime,
           'runtime_report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'runtime_note':'Same CPU/4 threads/FP32, batch1, 5 warmup/20 timed, same 20 decoded RGB dev images. Prototype implementations, different input resolutions; SAM latency seed0 vs three separately timed RGB seeds.'}
    # Retain the quality-only report and JSON; create a distinct final artifact.
    (a.comparison/'summary_with_runtime.json').write_text(json.dumps(final,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Calidad y costo completo: comparación de desarrollo','',
           '400 train / 100 dev; tres semillas para calidad, selección en dev. Sin nueva evaluación de test.',
           '','| Método | Dice [IC 95 % imagen] | Entrada | Parámetros totales | Entrenables | Latencia mediana (ms) |',
           '|---|---:|---:|---:|---:|---:|']
    timed=[row for row in table if row['latency_median_ms'] is not None]
    for row in sorted(timed,key=lambda r:r['dice'][0],reverse=True):
        m,l,h=row['dice'];lines.append(f"| {row['method']} | {m:.4f} [{l:.4f}, {h:.4f}] | {row['input_longest_side']} | {row['parameters_total']} | {row['parameters_trainable']} | {row['latency_median_ms']:.2f} |")
    lines+=['',f"SAM completo: mediana {runtime['median_ms']:.2f} ms, p95 {runtime['p95_ms']:.2f} ms; semilla 0.",
            'CPU/cuatro hilos/FP32/batch1. Mismos veinte RGB decodificados, cinco calentamientos y veinte repeticiones.',
            'Incluye preprocesado, red completa y restauración/umbral; SAM incluye encoder, probe, componentes, prompt encoder y decoder. Excluye disco, carga de pesos y métricas.',
            'Las predicciones del pipeline SAM desde caché y RGB reprodujeron los scores previos en las imágenes comprobadas a tolerancia 1e−8.',
            '','## Interpretación y límites','',
            '- Las redes compactas se entrenaron desde cero a resolución 128; SAM usa preentrenamiento genérico y resolución 1024. Es una comparación de sistemas con el mismo presupuesto local de etiquetas, no de arquitecturas aisladas.',
            '- Latencia local de implementaciones prototipo, no benchmark universal ni medición de tiempo humano. Los tiempos RGB resumen tres semillas; el de SAM representa semilla 0.',
            '- En RGB se usa el preprocesador experimental completo, incluida creación de tensores auxiliares sin GT; no es una implementación optimizada para producción.',
            '- Los parámetros entrenables de SAM son solo 257, pero su encoder y decoder congelados sí cuentan en tamaño y cómputo total.',
            '- La selección de épocas y otros ajustes usa dev. Los IC remuestrean imágenes tras promediar semillas; no son validación independiente por paciente.',
            '- Los contrastes, controles de prior y subgrupos están en `report.md`/`summary.json`; no deben interpretarse como pruebas clínicas confirmatorias.',
            '- Pendientes: mayor resolución/presupuesto de controles RGB, especialista/SAM 2.1, control de capacidad del generador y evaluación externa.']
    contrast=next(v for k,v in results['paired_exploratory'].items() if k.endswith('minus unet'))
    lines += ['', '## Diferencia media frente a U-Net', '',
              f"SAM automático menos U-Net: {contrast['mean_diff']:.5f}, IC pareado 95 % [{contrast['ci95'][0]:.5f}; {contrast['ci95'][1]:.5f}].",
              'El IC incluye cero: este análisis no demuestra una mejora positiva de la media ni equivalencia. El p de Wilcoxon del JSON se refiere a rangos/distribución, no a esa misma hipótesis sobre la media.',
              'El cómputo local mucho mayor de SAM y esta incertidumbre justifican conservar U-Net como comparador principal y madurar entrenamiento/evaluación independiente.']
    (a.comparison/'report_with_runtime.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (a.comparison/'runtime_report_source.py').write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for row in timed:
        m,l,h=row['dice'];x=row['latency_median_ms']
        ax.errorbar([x],[m],yerr=np.array([[m-l],[h-m]]),fmt='o',capsize=3)
        label=row['method'] if row['family']=='RGB compact' else 'SAM + automatic box/point'
        ax.annotate(label,(x,m),xytext=(5,8),textcoords='offset points',fontsize=8)
    ax.set_xscale('log');ax.margins(x=.35,y=.25)
    ax.set_xlabel('Full prototype pipeline median latency (ms, CPU batch1; log scale)')
    ax.set_ylabel('Development Dice; image bootstrap CI')
    ax.set_title('Same train/dev IDs; different pretraining and input size',fontsize=10)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(a.comparison/'quality_vs_latency.png',dpi=160)
    print(a.comparison/'report_with_runtime.md')


if __name__=='__main__':main()
