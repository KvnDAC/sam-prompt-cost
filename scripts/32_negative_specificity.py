"""Frozen negative-frame evaluation on HyperKvasir cecum images."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('OPENCV_LOG_LEVEL','ERROR')
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from samcost.common import CKPT,load_sam,set_threads
from samcost.data import SamPreprocessor
from samcost.decode import CachedSAM
from samcost.light_baselines import build_model
from samcost.prompts import bbox,interior_point
from samcost.unet import prepare_arrays,restore_logits

DATA=ROOT/'data/hyperkvasir_cecum';MANIFEST=DATA/'manifest.json'
RUN=ROOT/'runs/model_comparison/negative_specificity_20261017'
OUT=ROOT/'results/model_comparison/negative_specificity_20261017'
PROTOCOL=ROOT/'docs/protocolo_especificidad_negativos_20261017.md'
PROBES=[ROOT/f'results/phase_b/full_dev_20261012/probes/seed{s}/probe_state.pt' for s in range(3)]
UNETS=[ROOT/f'runs/model_comparison/unet_resolution_20261017/size256/seed{s}/resume.pt' for s in range(3)]
SOURCES=[Path(__file__),PROTOCOL,*[ROOT/'src/samcost'/x for x in
 ['common.py','data.py','decode.py','light_baselines.py','prompts.py','unet.py']]]
AREA_THRESHOLDS=(.005,.01,.05)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save_json(value,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=float),encoding='utf-8');tmp.replace(path)
def load_rgb(path):
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f'Cannot decode {path}')
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
def component_metrics(mask):
    mask=np.asarray(mask,bool);area=float(mask.mean())
    labels,n=ndi.label(mask)
    largest=0.0 if n==0 else float(np.max(ndi.sum(mask,labels,range(1,n+1)))/mask.size)
    return area,largest
def bootstrap_mean(x,n=10000,seed=1710):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);means=np.empty(n)
    for i in range(n):means[i]=x[rng.integers(0,len(x),len(x))].mean()
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}
def manifest_items():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'));items=m['files']
    if m.get('class')!='cecum' or len(items)!=1009:raise ValueError('Expected frozen 1009-image cecum manifest')
    return items,m
def audit(items):
    seen={};shapes={};bad=[]
    for item in items:
        p=DATA/item['name']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:bad.append(item['name']);continue
        image=load_rgb(p);shapes[str(tuple(image.shape))]=shapes.get(str(tuple(image.shape)),0)+1
        pix=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        if pix in seen:raise ValueError(f'Exact pixel duplicate: {seen[pix]}, {item["name"]}')
        seen[pix]=item['name']
    if bad:raise ValueError(f'{len(bad)} missing/corrupt files')
    return {'files':len(items),'unique_exact_pixels':len(seen),'shape_counts':shapes,'corrupt':bad}
def make_plan():
    items,manifest=manifest_items()
    for p in [CKPT,*PROBES,*UNETS,*SOURCES]:
        if not p.is_file():raise FileNotFoundError(p)
    return {'dataset':'HyperKvasir','class':'cecum','n_images':len(items),'filenames':[x['name'] for x in items],
      'manifest_sha256':sha(MANIFEST),'manifest_total_bytes':manifest['total_bytes'],'audit':audit(items),
      'model_sha256':{'sam':sha(CKPT),'probes':[sha(p) for p in PROBES],'unet_endpoints':[sha(p) for p in UNETS]},
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
      'frozen_from_kvasir':{'probe_thresholds':[float(torch.load(p,weights_only=False,map_location='cpu')['threshold']) for p in PROBES],
        'unet':'size256 epoch15 endpoint (45 total epochs)'},
      'metrics':{'area_thresholds':list(AREA_THRESHOLDS),'largest_component_threshold':.01,'bootstrap_images':10000},
      'external_tuning':False,'gt_prompts':False,'pixel_masks_available':False,'patient_groups_available':False}
def load_models():
    probes=[]
    for p in PROBES:
        state=torch.load(p,weights_only=False,map_location='cpu');model=torch.nn.Linear(256,1);model.load_state_dict(state['linear']);model.eval()
        probes.append((model,state['mu'],state['sd'],float(state['threshold'])))
    unets=[]
    for p in UNETS:
        state=torch.load(p,weights_only=False,map_location='cpu');model=build_model('unet',8);model.load_state_dict(state['model']);model.eval();unets.append(model)
    return probes,unets
@torch.inference_mode()
def evaluate_image(name,image,sam,helper,pre,probes,unets):
    x,osz,isz=pre(image);emb=sam.image_encoder(x);row={'image':name,'height':image.shape[0],'width':image.shape[1]}
    for seed,(model,mu,sd,threshold) in enumerate(probes):
        z=model((emb[0].permute(1,2,0)-mu)/sd)[...,0]
        low=F.interpolate(z[None,None],size=(256,256),mode='bilinear',align_corners=False)
        probe=(sam.postprocess_masks(low,isz,osz)[0,0]>threshold).numpy()
        area,largest=component_metrics(probe);row[f'probe_area_s{seed}']=area;row[f'probe_largest_s{seed}']=largest
        labels,n=ndi.label(probe)
        if n:
            sizes=ndi.sum(probe,labels,range(1,n+1));big=labels==(int(np.argmax(sizes))+1);point,_=interior_point(big)
            logits,_=helper.decode(emb,osz,points=point[None],labels=[1],box=bbox(big),multimask=False)
            pred=(sam.postprocess_masks(logits,isz,osz)[0,0]>0).numpy()
        else:pred=np.zeros(probe.shape,bool)
        area,largest=component_metrics(pred);row[f'sam_auto_area_s{seed}']=area;row[f'sam_auto_largest_s{seed}']=largest
    empty=np.zeros(image.shape[:2],bool);ux,_,_,usize=prepare_arrays(image,empty,256)
    for seed,model in enumerate(unets):
        pred=(restore_logits(model(ux[None])[0],usize,image.shape[:2])>0).numpy()
        area,largest=component_metrics(pred);row[f'unet_area_s{seed}']=area;row[f'unet_largest_s{seed}']=largest
    return row
def summarize_method(rows,method):
    area=np.array([[r[f'{method}_area_s{s}'] for s in range(3)] for r in rows],float)
    largest=np.array([[r[f'{method}_largest_s{s}'] for s in range(3)] for r in rows],float)
    av=area.mean(1);lv=largest.mean(1)
    out={'method':method,'area_fraction':{**bootstrap_mean(av),'median':float(np.median(av)),'p95':float(np.quantile(av,.95))},
      'largest_component_fraction':{**bootstrap_mean(lv),'median':float(np.median(lv)),'p95':float(np.quantile(lv,.95))},
      'any_mask_rate':bootstrap_mean((area>0).mean(1)),'empty_mask_specificity':bootstrap_mean((area==0).mean(1)),
      'largest_ge_1pct_rate':bootstrap_mean((largest>=.01).mean(1)),
      'seed_area_means':area.mean(0).tolist(),'seed_any_mask_rates':(area>0).mean(0).tolist()}
    out['area_threshold_rates']={str(t):bootstrap_mean((area>=t).mean(1)) for t in AREA_THRESHOLDS}
    return out,av,area,largest
def summarize(rows):
    table=[];vectors={};areas={}
    for method in ['probe','sam_auto','unet']:
        out,av,area,_=summarize_method(rows,method);table.append(out);vectors[method]=av;areas[method]=area
    contrast={}
    for other in ['unet','probe']:
        key=f'sam_auto_minus_{other}'
        contrast[key]={'mean_area_fraction':bootstrap_mean(vectors['sam_auto']-vectors[other]),
          'rate_area_ge_1pct':bootstrap_mean((areas['sam_auto']>=.01).mean(1)-(areas[other]>=.01).mean(1))}
    result={'dataset':'HyperKvasir/cecum','n_images':len(rows),'external_tuning':False,
      'unit_of_resampling':'image (patient/video IDs unavailable)','table':table,'paired_differences':contrast}
    save_json(result,OUT/'summary.json')
    lines=['# Control negativo externo: HyperKvasir cecum','',
      f'{len(rows):,} imágenes; modelos/umbrales congelados en Kvasir-SEG; sin ajuste externo.'.replace(',','.'),'',
      '| Método | Área falsa media (IC 95 %) | P95 área | Tasa >=1 % | Componente mayor P95 | Vacío |','|---|---:|---:|---:|---:|---:|']
    for r in table:
        a=r['area_fraction'];lines.append(f"| {r['method']} | {a['mean']:.4f} [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}] | {a['p95']:.4f} | {r['area_threshold_rates']['0.01']['mean']:.4f} | {r['largest_component_fraction']['p95']:.4f} | {r['empty_mask_specificity']['mean']:.4f} |")
    lines+=['','## Diferencias pareadas (SAM auto menos comparador)','',
      '| Comparación | Área falsa media | IC 95 % | Diferencia tasa >=1 % | IC 95 % |','|---|---:|---:|---:|---:|']
    for name,r in contrast.items():
        a=r['mean_area_fraction'];q=r['rate_area_ge_1pct'];lines.append(f"| {name} | {a['mean']:.4f} | [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}] | {q['mean']:.4f} | [{q['ci95'][0]:.4f}, {q['ci95'][1]:.4f}] |")
    lines+=['','La etiqueta es de clase anatómica, no una máscara negativa revisada píxel a píxel. Los IC remuestrean imágenes porque no hay IDs públicos de paciente/video; no representan especificidad clínica por paciente.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan-only',action='store_true');a=ap.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);plan=make_plan()
    if (RUN/'plan.json').exists() and json.loads((RUN/'plan.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen plan/code/data changed')
    save_json(plan,RUN/'plan.json');save_json(plan,OUT/'plan.json')
    if a.plan_only:print('PLAN FROZEN',RUN/'plan.json');return
    items,_=manifest_items();set_threads(4);probes,unets=load_models();sam=load_sam();pre=SamPreprocessor(sam);helper=CachedSAM(sam)
    progress=RUN/'rows.jsonl';rows=[]
    if progress.exists():rows=[json.loads(x) for x in progress.read_text(encoding='utf-8').splitlines() if x.strip()]
    done={r['image'] for r in rows};started=time.perf_counter()
    with progress.open('a',encoding='utf-8') as f:
        for item in items:
            if item['name'] in done:continue
            row=evaluate_image(item['name'],load_rgb(DATA/item['name']),sam,helper,pre,probes,unets)
            f.write(json.dumps(row)+'\n');f.flush();rows.append(row)
            save_json({'state':'running','completed':len(rows),'total':len(items),'last_image':item['name'],'external_tuning':False},OUT/'status.json')
            print(json.dumps({'completed':len(rows),'total':len(items),'elapsed_min':(time.perf_counter()-started)/60}),flush=True)
    order={x['name']:i for i,x in enumerate(items)};rows.sort(key=lambda r:order[r['image']])
    if len(rows)!=len(items) or {r['image'] for r in rows}!={x['name'] for x in items}:raise ValueError('Incomplete or duplicate rows')
    with (OUT/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summarize(rows);save_json({'state':'complete','completed':len(rows),'external_tuning':False},OUT/'status.json')
    print('COMPLETE',OUT/'report.md',flush=True)
if __name__=='__main__':main()
