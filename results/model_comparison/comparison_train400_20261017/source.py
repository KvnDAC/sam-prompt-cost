"""Compare completed RGB/SAM development artifacts; rerun RGB timing after training."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from samcost.common import set_threads
from samcost.data import load_image,load_mask,read_split
from samcost.light_baselines import build_model
from samcost.metrics import bootstrap_ci,paired_test
from samcost.unet import prepare_arrays,restore_logits


def read_scores(path,ids,column='dice'):
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    byid={r['id']:float(r[column]) for r in rows}
    if len(byid)!=len(rows) or set(byid)!=set(ids):raise ValueError(f'IDs differ: {path}')
    return np.array([byid[i] for i in ids])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run_root',type=Path,required=True);ap.add_argument('--rgb_results',type=Path,required=True)
    ap.add_argument('--prior_results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    state=json.loads((a.run_root/'status.json').read_text())
    if state['state']!='complete' or state['completed_runs']!=9:raise ValueError('Nine RGB runs must be complete')
    plan=json.loads((a.run_root/'plan.json').read_text());ids=read_split('dev')
    if ids!=plan['ids']['dev']:raise ValueError('Dev split changed')
    for path,digest in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Training source changed')
    set_threads(plan['args']['threads'])
    spec=importlib.util.spec_from_file_location('light_comparison',ROOT/'scripts/20_compare_light_models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    vectors,table,timings,selected_models,epoch_details={},[],{},{},{}
    for name in plan['models']:
        scores=[];times=[];iou_scores=[];epoch_details[name]=[]
        for seed in plan['args']['seeds']:
            run=a.run_root/name/f'seed{seed}'
            summary=json.loads((run/'summary.json').read_text())
            scores.append(read_scores(run/'dev_per_image.csv',ids))
            iou_scores.append(read_scores(run/'dev_per_image.csv',ids,'iou'))
            epoch_details[name].append({'seed':seed,'selected_epoch':summary['best_epoch'],
                                        'epochs_run':summary['epochs_run'],
                                        'selected_last_available':summary['best_epoch']==summary['epochs_run']})
            model=build_model(name,plan['args']['width'])
            model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True));model.eval()
            # Execute after training/test processes finish; these replace no prior measurements.
            times.append(module.latency(model,ids,plan['args']['size']))
            if seed==0:selected_models[name]=model
        matrix=np.stack(scores);mean=matrix.mean(0);vectors[name]=mean;timings[name]=times
        table.append({'method':name,'family':'RGB compact','dice':bootstrap_ci(mean),
                      'iou':bootstrap_ci(np.mean(iou_scores,axis=0)),
                      'seed_std':float(matrix.mean(1).std(ddof=1)),
                      'parameters_total':summary['parameters'],'parameters_trainable':summary['parameters'],
                      'input_longest_side':plan['args']['size'],'pretraining':'none',
                      'latency_median_ms':float(np.median([t['median_ms'] for t in times]))})
        print('TIMED',name,flush=True)
    sam=ROOT/'results/phase_b/full_dev_20261012';sam_summary=json.loads((sam/'summary.json').read_text())
    columns={'Probe lineal':'probe_dice','Probe → caja → SAM':'selfbox_dice',
             'Probe → caja + punto → SAM':'selfboxpoint_dice'}
    for label,column in columns.items():
        values=np.array([read_scores(sam/f'probes/seed{s}/probe_dev_per_image.csv',ids,column) for s in [0,1,2]])
        mean=values.mean(0);vectors[label]=mean
        table.append({'method':label,'family':'SAM-based','dice':bootstrap_ci(mean),
                      'seed_std':float(values.mean(1).std(ddof=1)),
                      'parameters_total':None,'parameters_trainable':257,'input_longest_side':1024,
                      'pretraining':'SAM ViT-B generic','latency_median_ms':None})
    for label,key in [('Tokens estáticos','static'),('Tokens condicionados','conditional')]:
        values=np.array([read_scores(sam/f'runs/{key}_k4_s{s}/dev_per_image.csv',ids) for s in [0,1,2]])
        mean=values.mean(0);vectors[label]=mean
        source=next(r for r in sam_summary['table'] if r['method']==label)
        table.append({'method':label,'family':'SAM-based','dice':bootstrap_ci(mean),
                      'seed_std':float(values.mean(1).std(ddof=1)),
                      'parameters_total':None,'parameters_trainable':source['trained_parameters'],
                      'input_longest_side':1024,'pretraining':'SAM ViT-B generic','latency_median_ms':None})
    prior=list(csv.DictReader((a.prior_results/'dev_per_image.csv').open(encoding='utf-8')))
    for mode in ['fixed_point','fixed_box','fixed_boxpoint']:
        byid={r['id']:float(r['dice']) for r in prior if r['mode']==mode}
        if set(byid)!=set(ids):raise ValueError('Prior IDs differ')
        vectors[mode]=np.array([byid[i] for i in ids])
    auto='Probe → caja + punto → SAM'
    comparisons={f'{auto} minus {name}':paired_test(vectors[auto],vectors[name]) for name in plan['models']}
    comparisons['image-specific boxpoint minus fixed boxpoint']=paired_test(vectors[auto],vectors['fixed_boxpoint'])
    comparisons['static tokens minus fixed point']=paired_test(vectors['Tokens estáticos'],vectors['fixed_point'])
    attribute_rows=list(csv.DictReader((sam/'spatial_subgroups/attributes.csv').open(encoding='utf-8')))
    attributes={row['id']:row for row in attribute_rows}
    if set(attributes)!=set(ids):raise ValueError('Subgroup attribute IDs differ')
    subgroup_results={}
    for field in ['area_bin','multiple_components','touches_edge']:
        if field not in attribute_rows[0]:continue
        for value in sorted({attributes[i][field] for i in ids}):
            indices=[j for j,i in enumerate(ids) if attributes[i][field]==value]
            subgroup_results[f'{field}={value}']={'n':len(indices),
                'methods':{name:bootstrap_ci(scores[indices]) for name,scores in vectors.items()}}
    a.output.mkdir(parents=True)
    result={'test_evaluated':False,'n_train':400,'n_dev':100,'seeds':[0,1,2],
            'table':table,'paired_exploratory':comparisons,'isolated_rgb_latency':timings,
            'epoch_selection':epoch_details,
            'subgroups_descriptive':subgroup_results,
            'sam_runtime':'Not measured here; encoder cost cannot be inferred from cached decoder timings.',
            'differences':'Same label IDs, different input sizes/pretraining/training recipes. Local compact variants, no SOTA claim.',
            'selection':'All values development-selected, no independent new test.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    lines=['# Comparación ampliada de desarrollo','',
           '400 train / 100 dev, mismos IDs. Tres semillas por método aprendido; sin nueva evaluación de test.',
           'Las recetas, resolución y preentrenamiento difieren. La tabla no aísla causalmente arquitectura ni demuestra superioridad al estado del arte.',
           '','| Método | Dice [IC 95 % imagen] | DE semillas | Entrada | Parámetros entrenables | Latencia RGB (ms) |',
           '|---|---:|---:|---:|---:|---:|']
    for row in sorted(table,key=lambda r:r['dice'][0],reverse=True):
        m,l,h=row['dice'];lat='pendiente SAM completo' if row['latency_median_ms'] is None else f"{row['latency_median_ms']:.2f}"
        lines.append(f"| {row['method']} | {m:.4f} [{l:.4f}, {h:.4f}] | {row['seed_std']:.4f} | {row['input_longest_side']} | {row['parameters_trainable']} | {lat} |")
    lines+=['','Los parámetros entrenables de los métodos SAM no incluyen el encoder/decoder congelados; no representan tamaño total.',
            'Latencias RGB reestimadas después de terminar el entrenamiento, CPU/cuatro hilos/batch1, preprocesado+red+restauración+umbral; sin lectura de disco.',
            '','## Selección y presupuesto','']
    for name,details in epoch_details.items():
        lines.append('- '+name+': '+', '.join(f"semilla {d['seed']}: época {d['selected_epoch']} de {d['epochs_run']}" for d in details)+'.')
    lines+=['','Elegir la última época disponible no acredita convergencia; la receta y resolución son controles de presupuesto limitado.',
            '','## Control de prior espacial fijo','']
    for mode in ['fixed_point','fixed_box','fixed_boxpoint']:
        m,l,h=bootstrap_ci(vectors[mode]);lines.append(f'- {mode}: Dice {m:.4f} [{l:.4f}, {h:.4f}].')
    lines+=['','El prior usa solo GT train; los prompts no conocen localización ni número de componentes de dev.',
            'Contrastes pareados exploratorios y sus intervalos en `summary.json`; no se ajustó selección/multiplicidad.',
            '','## Subgrupos descriptivos','',
            'Se reutilizan los bins de área, contacto con borde y componentes de Fase B; no se optimizan cortes nuevos.',
            'Resultados por grupo y tamaño muestral en `summary.json`. Son grupos solapados, algunos pequeños; no causas ni independencia clínica.',
            '','## Límites que permanecen','',
            '- Dev guio selección; test histórico ya inspeccionado. Falta evaluación independiente.',
            '- Entrada 128 y entrenamiento desde cero limitan los controles RGB; SAM usa 1024 y preentrenamiento genérico.',
            '- El presupuesto común no garantiza convergencia ni recetas óptimas; inspeccionar curvas y épocas elegidas.',
            '- Faltan SAM 2.1/especialista, control entrenado del generador de igual capacidad y costo completo SAM.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,5,figsize=(12,9))
    with torch.inference_mode():
        for idx,iid in enumerate(ids[:4]):
            image,gt=load_image(iid),load_mask(iid)
            axes[idx,0].imshow(image);axes[idx,1].imshow(gt,cmap='gray',vmin=0,vmax=1)
            x,_,_,shape=prepare_arrays(image,gt,plan['args']['size'])
            for col,name in enumerate(plan['models'],2):
                logits=selected_models[name](x[None])[0]
                pred=(restore_logits(logits,shape,gt.shape)>0).numpy()
                axes[idx,col].imshow(image);axes[idx,col].contour(pred,levels=[.5],colors=['magenta'],linewidths=.8)
                axes[idx,col].contour(gt,levels=[.5],colors=['cyan'],linewidths=.8)
            for col in range(5):axes[idx,col].axis('off')
        for ax,label in zip(axes[0],['RGB','GT',*plan['models']]):ax.set_title(label)
    fig.suptitle('First four dev IDs, seed 0; cyan GT / magenta prediction. Illustrative, no score selection.',fontsize=10)
    fig.tight_layout();fig.savefig(a.output/'qualitative_first4.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    ordered=sorted(table,key=lambda r:r['dice'][0]);y=np.arange(len(ordered))
    m=np.array([r['dice'][0] for r in ordered]);lo=np.array([r['dice'][1] for r in ordered]);hi=np.array([r['dice'][2] for r in ordered])
    ax.errorbar(m,y,xerr=np.stack([m-lo,hi-m]),fmt='o',capsize=3)
    ax.set_yticks(y,[r['method'] for r in ordered]);ax.set_xlim(0,1);ax.set_xlabel('Development Dice; bootstrap images after averaging seeds')
    ax.set_title('Same label IDs; different pretraining/input sizes. Exploratory comparison.')
    fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=160);plt.close(fig)
    print(a.output/'report.md',flush=True)


if __name__=='__main__':main()
