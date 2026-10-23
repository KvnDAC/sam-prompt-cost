"""Frozen Kvasir->CVC-ClinicDB evaluation with sequence-clustered uncertainty."""
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
from samcost.data import SamPreprocessor,load_image,read_split
from samcost.decode import CachedSAM
from samcost.light_baselines import build_model
from samcost.metrics import dice,iou
from samcost.prompts import bbox,interior_point
from samcost.unet import prepare_arrays,restore_logits

DATA=ROOT/'data/CVC-ClinicDB'; ARCHIVE=ROOT/'data/CVC-ClinicDB.rar'
RUN=ROOT/'runs/model_comparison/external_clinicdb_20261017'
OUT=ROOT/'results/model_comparison/external_clinicdb_20261017'
PROTOCOL=ROOT/'docs/protocolo_validacion_externa_20261017.md'
PROBES=[ROOT/f'results/phase_b/full_dev_20261012/probes/seed{s}/probe_state.pt' for s in range(3)]
UNETS=[ROOT/f'runs/model_comparison/unet_resolution_20261017/size256/seed{s}/resume.pt' for s in range(3)]
SOURCES=[Path(__file__),PROTOCOL,*[ROOT/'src/samcost'/x for x in
 ['common.py','data.py','decode.py','light_baselines.py','metrics.py','prompts.py','unet.py']]]
BOUNDS=(25,50,67,78,103,126,151,177,199,205,227,252,277,297,317,342,363,383,408,428,447,466,478,503,528,546,571,591,612)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sequence(frame):
    if not 1<=int(frame)<=612:raise ValueError(frame)
    return int(np.searchsorted(BOUNDS,int(frame),side='left')+1)
def load_pair(frame):
    image=cv2.imread(str(DATA/'Original'/f'{frame}.tif'),cv2.IMREAD_COLOR)
    mask=cv2.imread(str(DATA/'Ground Truth'/f'{frame}.tif'),cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2]!=mask.shape:raise ValueError(f'Invalid pair {frame}')
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),mask>127
def save_json(value,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=float),encoding='utf-8');tmp.replace(path)
def audit():
    expected={str(i) for i in range(1,613)}
    images={p.stem:p for p in (DATA/'Original').glob('*.tif')};masks={p.stem:p for p in (DATA/'Ground Truth').glob('*.tif')}
    if set(images)!=expected or set(masks)!=expected:raise ValueError('Require complete 1..612 pairs')
    hashes={};empty=[];shapes=set()
    for frame in range(1,613):
        image,mask=load_pair(frame);shapes.add(mask.shape);empty += [frame] if not mask.any() else []
        key=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        if key in hashes:raise ValueError(f'Duplicate external pixels: {hashes[key]}, {frame}')
        hashes[key]=frame
    kvasir=set()
    for iid in read_split('train400')+read_split('dev')+read_split('test'):
        image=load_image(iid);kvasir.add(hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest())
    return {'pairs':612,'sequences':29,'shapes':[list(x) for x in sorted(shapes)],'empty_masks':empty,
            'unique_external_pixels':len(hashes),'exact_pixel_overlap_with_kvasir_620':len(set(hashes)&kvasir)}
def make_plan():
    for p in [ARCHIVE,CKPT,*PROBES,*UNETS,*SOURCES]:
        if not p.is_file():raise FileNotFoundError(p)
    return {'dataset':'CVC-ClinicDB','frames':list(range(1,613)),'sequence_bounds':list(BOUNDS),
      'audit':audit(),'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,
      'data_sha256':{str(i):{'image':sha(DATA/'Original'/f'{i}.tif'),'mask':sha(DATA/'Ground Truth'/f'{i}.tif')} for i in range(1,613)},
      'model_sha256':{'sam':sha(CKPT),'probes':[sha(p) for p in PROBES],'unet_endpoints':[sha(p) for p in UNETS]},
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
      'frozen_from_kvasir':{'probe_thresholds':[float(torch.load(p,weights_only=False,map_location='cpu')['threshold']) for p in PROBES],
                            'unet':'size256 epoch15 endpoint (45 total epochs)'},
      'metrics':'frame Dice/IoU; seed-average; sequence-cluster bootstrap 10000',
      'external_tuning':False,'gt_prompts':False,'negative_frames':False}
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
def evaluate_frame(frame,image,gt,sam,helper,pre,probes,unets):
    x,osz,isz=pre(image);emb=sam.image_encoder(x);row={'frame':frame,'sequence':sequence(frame),'gt_area':float(gt.mean())}
    for seed,(model,mu,sd,threshold) in enumerate(probes):
        z=model((emb[0].permute(1,2,0)-mu)/sd)[...,0]
        low=F.interpolate(z[None,None],size=(256,256),mode='bilinear',align_corners=False)
        pm=(sam.postprocess_masks(low,isz,osz)[0,0]>threshold).numpy()
        row[f'probe_dice_s{seed}']=dice(pm,gt);row[f'probe_iou_s{seed}']=iou(pm,gt)
        labels,n=ndi.label(pm)
        if n:
            sizes=ndi.sum(pm,labels,range(1,n+1));big=labels==(int(np.argmax(sizes))+1);point,_=interior_point(big)
            logits,_=helper.decode(emb,osz,points=point[None],labels=[1],box=bbox(big),multimask=False)
            pred=(sam.postprocess_masks(logits,isz,osz)[0,0]>0).numpy()
        else:pred=np.zeros_like(gt)
        row[f'sam_auto_dice_s{seed}']=dice(pred,gt);row[f'sam_auto_iou_s{seed}']=iou(pred,gt)
    ux,_,_,usize=prepare_arrays(image,gt,256)
    for seed,model in enumerate(unets):
        pred=(restore_logits(model(ux[None])[0],usize,gt.shape)>0).numpy()
        row[f'unet_dice_s{seed}']=dice(pred,gt);row[f'unet_iou_s{seed}']=iou(pred,gt)
    return row
def cluster_interval(values,sequences,n=10000,seed=1701):
    values=np.asarray(values,float);sequences=np.asarray(sequences,int);groups=[values[sequences==s] for s in range(1,30)]
    rng=np.random.default_rng(seed);means=[]
    for _ in range(n):means.append(float(np.concatenate([groups[i] for i in rng.integers(0,29,29)]).mean()))
    return {'frame_mean':float(values.mean()),'sequence_macro_mean':float(np.mean([g.mean() for g in groups])),
            'cluster_ci95':np.quantile(means,[.025,.975]).tolist()}
def summarize(rows):
    sequences=[r['sequence'] for r in rows];vectors={};table=[]
    for method in ['probe','sam_auto','unet']:
        vector=np.array([np.mean([float(r[f'{method}_dice_s{s}']) for s in range(3)]) for r in rows]);vectors[method]=vector
        seeds=[float(np.mean([float(r[f'{method}_dice_s{s}']) for r in rows])) for s in range(3)]
        table.append({'method':method,**cluster_interval(vector,sequences),'seed_means':seeds,'seed_std':float(np.std(seeds,ddof=1))})
    contrasts={'sam_auto_minus_unet':cluster_interval(vectors['sam_auto']-vectors['unet'],sequences),
               'sam_auto_minus_probe':cluster_interval(vectors['sam_auto']-vectors['probe'],sequences)}
    result={'dataset':'CVC-ClinicDB','n_frames':612,'n_sequences':29,'external_tuning':False,'table':table,'paired_differences':contrasts}
    save_json(result,OUT/'summary.json')
    lines=['# Validación externa congelada: CVC-ClinicDB','',
      'Modelos y umbrales fijados en Kvasir-SEG; 612 frames/29 secuencias; sin adaptación externa.','',
      '| Método | Dice por frame | IC 95 % agrupado por secuencia | Macro-secuencia | DE semillas |','|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['method']} | {r['frame_mean']:.4f} | [{r['cluster_ci95'][0]:.4f}, {r['cluster_ci95'][1]:.4f}] | {r['sequence_macro_mean']:.4f} | {r['seed_std']:.4f} |")
    lines+=['','## Diferencias pareadas','', '| Contraste | Diferencia Dice | IC 95 % por secuencia |','|---|---:|---:|']
    for name,r in contrasts.items():lines.append(f"| {name} | {r['frame_mean']:.4f} | [{r['cluster_ci95'][0]:.4f}, {r['cluster_ci95'][1]:.4f}] |")
    lines+=['','Los IC remuestrean las 29 secuencias completas. CVC-ClinicDB no contiene negativos y no mide especificidad.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan_only',action='store_true');a=ap.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);plan=make_plan()
    if (RUN/'plan.json').exists() and json.loads((RUN/'plan.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen plan/code/data changed')
    save_json(plan,RUN/'plan.json');save_json(plan,OUT/'plan.json')
    if a.plan_only:print('PLAN FROZEN',RUN/'plan.json');return
    set_threads(4);probes,unets=load_models();sam=load_sam();pre=SamPreprocessor(sam);helper=CachedSAM(sam)
    progress=RUN/'rows.jsonl';rows=[]
    if progress.exists():rows=[json.loads(x) for x in progress.read_text(encoding='utf-8').splitlines() if x.strip()]
    done={int(r['frame']) for r in rows};started=time.perf_counter()
    with progress.open('a',encoding='utf-8') as f:
        for frame in range(1,613):
            if frame in done:continue
            image,gt=load_pair(frame);row=evaluate_frame(frame,image,gt,sam,helper,pre,probes,unets)
            f.write(json.dumps(row)+'\n');f.flush();rows.append(row)
            save_json({'state':'running','completed':len(rows),'total':612,'last_frame':frame,'external_tuning':False},OUT/'status.json')
            print(json.dumps({'frame':frame,'completed':len(rows),'elapsed_min':(time.perf_counter()-started)/60}),flush=True)
    rows=sorted(rows,key=lambda r:int(r['frame']))
    if [int(r['frame']) for r in rows]!=list(range(1,613)):raise ValueError('Incomplete or duplicate rows')
    with (OUT/'per_frame.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summarize(rows);save_json({'state':'complete','completed':612,'sequences':29,'external_tuning':False},OUT/'status.json')
    print('COMPLETE',OUT/'report.md',flush=True)
if __name__=='__main__':main()
