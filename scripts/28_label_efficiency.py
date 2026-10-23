"""Nested 25/100/400-label study for frozen-SAM adaptation and RGB U-Net; dev only."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from samcost.common import CACHE, load_sam, set_seed, set_threads
from samcost.data import KVASIR, load_image, load_mask, read_split
from samcost.decode import CachedSAM
from samcost.learned import seg_loss
from samcost.light_baselines import build_model
from samcost.metrics import dice, iou
from samcost.prompts import bbox, interior_point
from samcost.unet import prepare_arrays, restore_logits

spec = importlib.util.spec_from_file_location('light20_labels', ROOT/'scripts/20_compare_light_models.py')
light = importlib.util.module_from_spec(spec)
spec.loader.exec_module(light)

BUDGETS = (25, 100, 400)
SEEDS = (0, 1, 2)
PROBE_STEPS = 300
UNET_STEPS = 1500
EVAL_EVERY = 100
PROTOCOL = ROOT/'docs/protocolo_eficiencia_etiquetas_20261017.md'
SOURCES = [Path(__file__), PROTOCOL, ROOT/'scripts/06_probe.py', ROOT/'scripts/20_compare_light_models.py',
           *[ROOT/'src/samcost'/name for name in ['common.py','data.py','decode.py','learned.py',
                                                   'light_baselines.py','metrics.py','prompts.py','unet.py']]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    tmp.replace(path)


def save_torch(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp); tmp.replace(path)


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def nested_subsets(ids, seeds=SEEDS, budgets=BUDGETS):
    if len(ids) != max(budgets) or len(set(ids)) != len(ids):
        raise ValueError('Expected unique train400 IDs')
    result = {}
    for seed in seeds:
        order = np.asarray(ids)[np.random.default_rng(seed).permutation(len(ids))].tolist()
        result[str(seed)] = {str(b): order[:b] for b in budgets}
        for left, right in zip(budgets, budgets[1:]):
            if result[str(seed)][str(left)] != result[str(seed)][str(right)][:left]:
                raise AssertionError('Subsets must be ordered prefixes')
    return result


def area_summary(ids):
    values = np.asarray([load_mask(i).mean() for i in ids])
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
            'small_lt_0.05': int((values < .05).sum()),
            'medium_0.05_0.15': int(((values >= .05) & (values < .15)).sum()),
            'large_ge_0.15': int((values >= .15).sum())}


def make_plan(run_root, output):
    train, dev = read_split('train400'), read_split('dev')
    light.validate_ids(train, dev)
    subsets = nested_subsets(train)
    return {'question': 'label efficiency of frozen-SAM adaptation versus RGB U-Net from scratch',
            'ids': {'train400': train, 'dev': dev, 'test': []}, 'subsets': subsets,
            'subset_area': {s: {b: area_summary(ids) for b, ids in groups.items()}
                            for s, groups in subsets.items()},
            'budgets': list(BUDGETS), 'seeds': list(SEEDS), 'test_evaluated': False,
            'primary': {'probe': f'fixed endpoint at {PROBE_STEPS} Adam updates',
                        'unet': f'fixed endpoint at {UNET_STEPS} Adam updates'},
            'secondary': 'U-Net best development checkpoint among step0 and every 100 updates',
            'probe': {'tokens_per_image': 1500, 'batch_size': 8192, 'steps': PROBE_STEPS,
                      'lr': .01, 'threshold_grid': np.linspace(-3,3,25).tolist()},
            'unet': {'size': 256, 'width': 8, 'batch_size': 8, 'steps': UNET_STEPS,
                     'lr': .001, 'eval_every': EVAL_EVERY, 'pretraining': 'none',
                     'loss': 'BCE + soft Dice', 'normalization': 'GroupNorm'},
            'sampling': 'with replacement for fixed update counts; nested subsets without replacement',
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
            'data_sha256': {i: {kind: sha(KVASIR/kind/f'{i}.jpg') for kind in ['images','masks']}
                            for i in train+dev},
            'run_root': str(run_root.resolve()), 'output': str(output.resolve()),
            'scope': 'development-only mechanistic/data-efficiency study; no SOTA claim'}


def token_labels(gt, input_size, n=64):
    h, w = input_size
    hh, ww = int(round(h*n/1024)), int(round(w*n/1024))
    y = F.interpolate(torch.from_numpy(gt[None,None].astype(np.float32)), size=(hh,ww), mode='area')[0,0]
    return y.numpy(), hh, ww


def collect_tokens(ids, meta, per_image=1500):
    xs, ys = [], []
    # Per-ID generators make overlapping examples identical across all budgets.
    for iid in ids:
        emb = np.load(CACHE/'emb'/f'{iid}.npy').astype(np.float32)
        lab, hh, ww = token_labels(load_mask(iid), tuple(meta[iid]['input_size']))
        x, y = emb[:,:hh,:ww].reshape(256,-1).T, lab.ravel()
        if len(y) > per_image:
            key = int(hashlib.sha256(iid.encode()).hexdigest()[:16], 16)
            take = np.random.default_rng(key).choice(len(y), per_image, replace=False)
            x, y = x[take], y[take]
        xs.append(x); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def train_probe(ids, seed, meta, out):
    summary_path = out/'train_summary.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True); set_seed(seed)
    x, y = collect_tokens(ids, meta)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y).float()
    mu, sd = xt.mean(0), xt.std(0)+1e-6
    model = torch.nn.Linear(256,1); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    generator = torch.Generator().manual_seed(1000+seed)
    started = time.perf_counter(); losses=[]
    for step in range(1, PROBE_STEPS+1):
        idx = torch.randint(len(xt),(8192,),generator=generator)
        loss = F.binary_cross_entropy_with_logits(model((xt[idx]-mu)/sd)[:,0],yt[idx])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 50 == 0: losses.append({'step':step,'loss':float(loss.detach())})
    state = {'model':model.state_dict(),'mu':mu,'sd':sd,'seed':seed,'steps':PROBE_STEPS,
             'n_tokens':len(yt),'train_ids':ids}
    save_torch(state,out/'endpoint.pt')
    result={'steps':PROBE_STEPS,'n_tokens':len(yt),'seconds':time.perf_counter()-started,'losses':losses}
    save_json(result,summary_path); return result


@torch.inference_mode()
def probe_logits(cs, model, mu, sd, iid):
    emb, osz, isz = cs.emb(iid)
    logits = model((emb[0].permute(1,2,0)-mu)/sd)[...,0]
    return F.interpolate(logits[None,None],size=(256,256),mode='bilinear',align_corners=False),emb,osz,isz


def evaluate_probe(cs, dev_ids, run):
    target = run/'dev_per_image.csv'
    if target.exists():
        return list(csv.DictReader(target.open(encoding='utf-8')))
    state=torch.load(run/'endpoint.pt',weights_only=False,map_location='cpu')
    model=torch.nn.Linear(256,1);model.load_state_dict(state['model']);model.eval()
    grid=np.linspace(-3,3,25); totals=np.zeros(len(grid))
    for iid in dev_ids:
        lr,_,osz,isz=probe_logits(cs,model,state['mu'],state['sd'],iid)
        full=cs.full(lr,osz,isz)[0,0].numpy();gt=load_mask(iid)
        totals += [dice(full>t,gt) for t in grid]
    threshold=float(grid[int(np.argmax(totals))]); rows=[]
    for iid in dev_ids:
        lr,emb,osz,isz=probe_logits(cs,model,state['mu'],state['sd'],iid)
        full=cs.full(lr,osz,isz)[0,0].numpy();gt=load_mask(iid);pm=full>threshold
        row={'id':iid,'probe_dice':dice(pm,gt),'probe_iou':iou(pm,gt)}
        labels,n=ndi.label(pm)
        if n:
            sizes=ndi.sum(pm,labels,range(1,n+1));big=labels==(int(np.argmax(sizes))+1)
            point,_=interior_point(big)
            low,_=cs.decode(emb,osz,points=point[None],labels=[1],box=bbox(big),multimask=False)
            pred=(cs.full(low,osz,isz)[0,0]>0).numpy()
        else: pred=np.zeros_like(gt)
        row.update(selfboxpoint_dice=dice(pred,gt),selfboxpoint_iou=iou(pred,gt),probe_empty=int(n==0))
        rows.append(row)
    write_rows(rows,target)
    save_json({'threshold':threshold,'threshold_dev_mean_dice':float(totals.max()/len(dev_ids)),
               'probe_dice':float(np.mean([r['probe_dice'] for r in rows])),
               'selfboxpoint_dice':float(np.mean([r['selfboxpoint_dice'] for r in rows])),
               'test_evaluated':False},run/'evaluation_summary.json')
    return rows


class Prepared:
    def __init__(self,ids,size=256):
        samples=[prepare_arrays(load_image(i),load_mask(i),size) for i in ids]
        self.x=torch.stack([x[0] for x in samples]);self.y=torch.stack([x[1] for x in samples])
        self.valid=torch.stack([x[2] for x in samples]);self.sizes=[x[3] for x in samples]


def sample_batch(data,batch_size,generator):
    indices=torch.randint(len(data.x),(batch_size,),generator=generator)
    x,y,v=data.x[indices].clone(),data.y[indices].clone(),data.valid[indices]
    for j,index in enumerate(indices.tolist()):
        dims=[]
        if torch.rand((),generator=generator)<.5:dims.append(-1)
        if torch.rand((),generator=generator)<.5:dims.append(-2)
        if dims:
            h,w=data.sizes[index]
            x[j,:,:h,:w]=torch.flip(x[j,:,:h,:w],dims)
            y[j,:h,:w]=torch.flip(y[j,:h,:w],dims)
    return x,y,v


@torch.inference_mode()
def evaluate_unet(model,ids,data,masks,batch_size=8):
    model.eval();rows=[]
    for start in range(0,len(ids),batch_size):
        logits=model(data.x[start:start+batch_size])
        for offset,logit in enumerate(logits):
            index=start+offset;pred=(restore_logits(logit,data.sizes[index],masks[index].shape)>0).numpy()
            rows.append({'id':ids[index],'dice':dice(pred,masks[index]),'iou':iou(pred,masks[index]),
                         'pred_area':float(pred.mean())})
    return rows


def train_unet(ids,seed,train,dev_ids,dev_data,dev_masks,out):
    final=out/'summary.json'
    if final.exists():return json.loads(final.read_text(encoding='utf-8'))
    out.mkdir(parents=True,exist_ok=True);set_seed(seed)
    model=build_model('unet',8);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    generator=torch.Generator().manual_seed(2000+seed);state_path=out/'resume.pt'
    history=[];step=0;seconds=0.;best=-1.;best_step=0
    if state_path.exists():
        state=torch.load(state_path,weights_only=False,map_location='cpu')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['generator']);torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        history,step,seconds,best,best_step=[state[k] for k in ['history','step','seconds','best','best_step']]
    else:
        rows=evaluate_unet(model,dev_ids,dev_data,dev_masks);best=float(np.mean([r['dice'] for r in rows]))
        write_rows(rows,out/'best_dev_per_image.csv');save_torch(model.state_dict(),out/'best.pt')
        history.append({'step':0,'train_loss':None,'dev_dice':best,'elapsed_seconds':0.})
    block_losses=[];block_start=time.perf_counter()
    while step<UNET_STEPS:
        model.train();x,y,v=sample_batch(train,8,generator)
        loss=seg_loss(model(x),y,v)
        if not torch.isfinite(loss):raise ValueError('Nonfinite U-Net loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        step+=1;block_losses.append(float(loss.detach()))
        if step%EVAL_EVERY==0 or step==UNET_STEPS:
            seconds+=time.perf_counter()-block_start
            rows=evaluate_unet(model,dev_ids,dev_data,dev_masks)
            score=float(np.mean([r['dice'] for r in rows]))
            history.append({'step':step,'train_loss':float(np.mean(block_losses)),
                            'dev_dice':score,'elapsed_seconds':seconds})
            if score>best:
                best,best_step=score,step;write_rows(rows,out/'best_dev_per_image.csv')
                save_torch(model.state_dict(),out/'best.pt')
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'generator':generator.get_state(),
                   'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                   'history':history,'step':step,'seconds':seconds,'best':best,'best_step':best_step}
            save_torch(state,state_path);save_json({'history':history,'step':step,'best':best,'best_step':best_step},out/'progress.json')
            print(json.dumps({'method':'unet','n_train':len(ids),'seed':seed,**history[-1]}),flush=True)
            block_losses=[];block_start=time.perf_counter()
    endpoint=evaluate_unet(model,dev_ids,dev_data,dev_masks);write_rows(endpoint,out/'endpoint_dev_per_image.csv')
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True));selected=evaluate_unet(model,dev_ids,dev_data,dev_masks)
    write_rows(selected,out/'selected_dev_per_image.csv')
    result={'n_train':len(ids),'seed':seed,'steps':step,'endpoint_dice':float(np.mean([r['dice'] for r in endpoint])),
            'selected_dice':float(np.mean([r['dice'] for r in selected])),'best_step':best_step,
            'seconds':seconds,'parameters':sum(p.numel() for p in model.parameters()),'history':history,'test_evaluated':False}
    save_json(result,final);return result


def read_metric(path,column,ids):
    rows=list(csv.DictReader(path.open(encoding='utf-8')));by_id={r['id']:float(r[column]) for r in rows}
    if len(by_id)!=len(rows) or set(by_id)!=set(ids):raise ValueError(f'IDs mismatch: {path}')
    return np.array([by_id[i] for i in ids])


def interval(values,seed=1701):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}


def summarize(plan,run_root,output):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    dev=plan['ids']['dev'];vectors={};table=[]
    definitions=[('probe','probe','probe_dice'),('sam_auto','probe','selfboxpoint_dice'),
                 ('unet_endpoint','unet','dice'),('unet_selected','unet_selected','dice')]
    for budget in BUDGETS:
        for method,folder,column in definitions:
            arrays=[]
            for seed in SEEDS:
                if folder=='probe':path=run_root/'probe'/f'n{budget}'/f'seed{seed}'/'dev_per_image.csv'
                elif folder=='unet':path=run_root/'unet'/f'n{budget}'/f'seed{seed}'/'endpoint_dev_per_image.csv'
                else:path=run_root/'unet'/f'n{budget}'/f'seed{seed}'/'selected_dev_per_image.csv'
                arrays.append(read_metric(path,column,dev))
            vector=np.stack(arrays).mean(0);vectors[(method,budget)]=vector
            table.append({'method':method,'budget':budget,**interval(vector),
                          'seed_means':[float(x.mean()) for x in arrays],
                          'seed_std':float(np.std([x.mean() for x in arrays],ddof=1))})
    contrasts={}
    for budget in BUDGETS:
        contrasts[f'sam_auto_minus_probe_n{budget}']=interval(vectors['sam_auto',budget]-vectors['probe',budget])
        contrasts[f'sam_auto_minus_unet_endpoint_n{budget}']=interval(vectors['sam_auto',budget]-vectors['unet_endpoint',budget])
    for method in ['probe','sam_auto','unet_endpoint']:
        for left,right in zip(BUDGETS,BUDGETS[1:]):
            contrasts[f'{method}_n{right}_minus_n{left}']=interval(vectors[method,right]-vectors[method,left])
    result={'test_evaluated':False,'table':table,'paired_differences_exploratory':contrasts,
            'primary_methods':['probe','sam_auto','unet_endpoint'],'secondary':['unet_selected'],
            'warning':'Development-only; bootstrap does not adjust selection, multiplicity, or image dependence.'}
    save_json(result,output/'summary.json')
    lines=['# Eficiencia de etiquetas: resultados de desarrollo','',
           'Subconjuntos anidados de 25, 100 y 400 máscaras; tres semillas; test no evaluado.','',
           '| Método | Máscaras | Dice | IC 95 % por imagen | DE semillas |','|---|---:|---:|---:|---:|']
    for row in table:
        lo,hi=row['ci95'];lines.append(f"| {row['method']} | {row['budget']} | {row['mean']:.4f} | [{lo:.4f}, {hi:.4f}] | {row['seed_std']:.4f} |")
    lines += ['','## Diferencias pareadas exploratorias','',
              '| Contraste | Diferencia Dice | IC 95 % |','|---|---:|---:|']
    for name,value in contrasts.items():
        lo,hi=value['ci95'];lines.append(f"| {name} | {value['mean']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines += ['','El endpoint fijo es primario; `unet_selected` es secundario porque usa selección en desarrollo.',
              'Los IC remuestrean imágenes después de promediar semillas. No se evaluó test.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4.5))
    for method,label in [('probe','Probe'),('sam_auto','Probe → caja+punto → SAM'),('unet_endpoint','U-Net endpoint')]:
        rows=[next(r for r in table if r['method']==method and r['budget']==b) for b in BUDGETS]
        means=[r['mean'] for r in rows];errors=np.array([[m-r['ci95'][0],r['ci95'][1]-m] for m,r in zip(means,rows)]).T
        ax.errorbar(BUDGETS,means,yerr=errors,marker='o',capsize=3,label=label)
    ax.set(xscale='log',xticks=BUDGETS,xticklabels=BUDGETS,xlabel='Máscaras de entrenamiento',ylabel='Dice en dev100',ylim=(0,1))
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/'label_efficiency.png',dpi=180);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run_root',type=Path,default=ROOT/'runs/model_comparison/label_efficiency_20261017')
    ap.add_argument('--output',type=Path,default=ROOT/'results/model_comparison/label_efficiency_20261017')
    ap.add_argument('--part',choices=['plan','probe','unet','summarize','all'],default='all')
    args=ap.parse_args();args.run_root.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    plan=make_plan(args.run_root,args.output);plan_path=args.run_root/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8'))!=plan:
        raise ValueError('Frozen plan/code/data differ; use a new run root')
    save_json(plan,plan_path);save_json(plan,args.output/'plan.json')
    if args.part=='plan':print('PLAN FROZEN',plan_path);return
    set_threads(4);meta=json.loads((CACHE/'meta.json').read_text());dev=plan['ids']['dev']
    if args.part in ('probe','all'):
        for budget in BUDGETS:
            for seed in SEEDS:
                ids=plan['subsets'][str(seed)][str(budget)];run=args.run_root/'probe'/f'n{budget}'/f'seed{seed}'
                save_json({'state':'running','part':'probe','budget':budget,'seed':seed},args.output/'status.json')
                print(f'PROBE n{budget} seed{seed}',flush=True);train_probe(ids,seed,meta,run)
        sam=load_sam();cs=CachedSAM(sam);cs.require_ids(dev)
        for budget in BUDGETS:
            for seed in SEEDS:
                print(f'PROBE EVAL n{budget} seed{seed}',flush=True)
                evaluate_probe(cs,dev,args.run_root/'probe'/f'n{budget}'/f'seed{seed}')
        del cs,sam
    if args.part in ('unet','all'):
        dev_data=Prepared(dev);dev_masks=[load_mask(i) for i in dev]
        for budget in BUDGETS:
            for seed in SEEDS:
                ids=plan['subsets'][str(seed)][str(budget)];run=args.run_root/'unet'/f'n{budget}'/f'seed{seed}'
                save_json({'state':'running','part':'unet','budget':budget,'seed':seed},args.output/'status.json')
                print(f'UNET n{budget} seed{seed}',flush=True)
                train_unet(ids,seed,Prepared(ids),dev,dev_data,dev_masks,run)
    if args.part in ('summarize','all'):
        summarize(plan,args.run_root,args.output)
        save_json({'state':'complete','completed_probe_runs':9,'completed_unet_runs':9,'test_evaluated':False},args.output/'status.json')
        print('COMPLETE',args.output/'report.md',flush=True)


if __name__=='__main__':main()
