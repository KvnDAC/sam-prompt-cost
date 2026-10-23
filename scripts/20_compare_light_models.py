"""Frozen, resumable train400/dev100 CPU comparison; never evaluates test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import psutil
import torch
from torch.utils.data import Dataset, DataLoader
from samcost.common import set_seed, set_threads
from samcost.data import KVASIR, read_split, load_image, load_mask
from samcost.learned import seg_loss
from samcost.light_baselines import build_model
from samcost.metrics import dice, iou
from samcost.unet import prepare_arrays, restore_logits

MODELS = ('fcn_lite', 'ds_unet', 'unet')
SOURCES = [Path(__file__), *[ROOT / 'src/samcost' / f for f in
           ['light_baselines.py', 'unet.py', 'data.py', 'learned.py', 'common.py', 'metrics.py']]]


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def save_checkpoint(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def write_rows(rows, path):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def validate_ids(train, dev):
    if len(train) != 400 or len(dev) != 100 or len(set(train)) != 400 or len(set(dev)) != 100 or set(train) & set(dev):
        raise ValueError('Require 400 unique train and 100 unique disjoint dev IDs')


class MemoryImages(Dataset):
    def __init__(self, ids, size):
        samples = [prepare_arrays(load_image(i), load_mask(i), size) for i in ids]
        self.x = torch.stack([s[0] for s in samples])
        self.y = torch.stack([s[1] for s in samples])
        self.valid = torch.stack([s[2] for s in samples])
        self.sizes = [s[3] for s in samples]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y, valid = self.x[index], self.y[index], self.valid[index]
        # Flip content within its valid rectangle; keep top-left padding convention.
        h, w = self.sizes[index]
        dims = []
        if torch.rand(()) < .5:
            dims.append(-1)
        if torch.rand(()) < .5:
            dims.append(-2)
        if dims:
            x, y = x.clone(), y.clone()
            x[:, :h, :w] = torch.flip(x[:, :h, :w], dims)
            y[:h, :w] = torch.flip(y[:h, :w], dims)
        return x, y, valid


@torch.inference_mode()
def evaluate(model, ids, prepared, masks, batch_size):
    model.eval()
    rows = []
    for start in range(0, len(ids), batch_size):
        logits = model(prepared.x[start:start+batch_size])
        for j, logit in enumerate(logits):
            idx = start+j; mask = masks[idx]
            pred = (restore_logits(logit, prepared.sizes[idx], mask.shape) > 0).numpy()
            rows.append({'id': ids[idx], 'dice': dice(pred, mask), 'iou': iou(pred, mask),
                         'pred_area': float(pred.mean())})
    return rows


def checkpoint_state(model, optimizer, generator):
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(), 'loader_rng': generator.get_state(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}


def restore_state(state, model, optimizer, generator):
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']); generator.set_state(state['loader_rng'])
    random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])


def interval(values, seed=1234, n=3000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(x.mean()), *np.quantile(means, [.025, .975]).tolist()]


def paired_difference(left, right):
    a, b = {r['id']: float(r['dice']) for r in left}, {r['id']: float(r['dice']) for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError('Paired comparison requires identical unique IDs')
    return interval([a[i]-b[i] for i in sorted(a)])


@torch.inference_mode()
def latency(model, ids, size):
    """RGB preproc + network + resize + threshold, batch1; decoded images in RAM."""
    images = [load_image(i) for i in ids[:20]]
    samples = []
    for index in range(25):
        image = images[index % len(images)]
        started = time.perf_counter()
        x, _, _, shape = prepare_arrays(image, np.zeros(image.shape[:2], bool), size)
        output = model(x[None])[0]
        _ = restore_logits(output, shape, image.shape[:2]) > 0
        elapsed = (time.perf_counter()-started)*1000
        if index >= 5:
            samples.append(elapsed)
    return {'median_ms': float(np.median(samples)), 'p95_ms': float(np.quantile(samples,.95)),
            'samples_ms': samples, 'warmup': 5, 'repeats': 20,
            'includes': 'RGB preprocessing + complete RGB model + original-size resize + threshold; excludes disk decode, metrics',
            'scope': 'CPU batch1; do not compare with SAM decoder-only timings'}


def train_one(name, seed, a, train_data, dev_data, dev_masks, dev_ids, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        return json.loads((out/'summary.json').read_text(encoding='utf-8'))
    set_seed(seed)
    model = build_model(name, a.width)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    generator = torch.Generator().manual_seed(seed)
    state_path = out/'resume.pt'
    history, best, best_epoch, epoch, stale, seconds = [], -1., 0, 0, 0, 0.
    peak = psutil.Process().memory_info().rss
    if state_path.exists():
        state = torch.load(state_path, weights_only=False, map_location='cpu')
        restore_state(state, model, optimizer, generator)
        history, best, best_epoch, epoch, stale, seconds, peak = [state[k] for k in
           ['history','best','best_epoch','epoch','stale','seconds','peak']]
        print(f'RESUME {name} seed{seed} epoch{epoch}', flush=True)
    else:
        initial = evaluate(model, dev_ids, dev_data, dev_masks, a.batch_size)
        best = float(np.mean([r['dice'] for r in initial]))
        write_rows(initial, out/'initial_dev_per_image.csv')
        save_checkpoint(model.state_dict(), out/'best.pt')
        write_rows(initial, out/'best_dev_per_image.csv')
    loader = DataLoader(train_data, batch_size=a.batch_size, shuffle=True, num_workers=0, generator=generator)
    for epoch_index in range(epoch+1, a.epochs+1):
        if epoch >= a.min_epochs and stale >= a.patience:
            break
        started = time.perf_counter(); model.train(); total, count = 0., 0
        for x,y,v in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = seg_loss(model(x), y, v)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(x); count += len(x)
            peak = max(peak, psutil.Process().memory_info().rss)
        rows = evaluate(model, dev_ids, dev_data, dev_masks, a.batch_size)
        score = float(np.mean([r['dice'] for r in rows]))
        stale += 1
        if score > best:
            best, best_epoch, stale = score, epoch_index, 0
            save_checkpoint(model.state_dict(), out/'best.pt')
            write_rows(rows, out/'best_dev_per_image.csv')
        seconds += time.perf_counter()-started; epoch = epoch_index
        history.append({'epoch':epoch,'train_loss':total/count,'dev_dice':score,
                        'dev_empty_rate':float(np.mean([r['pred_area']==0 for r in rows])),
                        'elapsed_seconds':seconds})
        state = checkpoint_state(model, optimizer, generator)
        state.update(history=history,best=best,best_epoch=best_epoch,epoch=epoch,
                     stale=stale,seconds=seconds,peak=peak)
        save_checkpoint(state,state_path)
        save_json({'history':history,'best_dev_dice':best,'best_epoch':best_epoch},out/'progress.json')
        print(json.dumps({'model':name,'seed':seed,**history[-1]}),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
    selected = evaluate(model, dev_ids, dev_data, dev_masks, a.batch_size)
    original = list(csv.DictReader((out/'best_dev_per_image.csv').open(encoding='utf-8')))
    for old,new in zip(original,selected):
        if old['id']!=new['id'] or abs(float(old['dice'])-new['dice'])>1e-8:
            raise ValueError('Selected checkpoint failed per-image reproduction')
    write_rows(selected,out/'dev_per_image.csv')
    result={'model':name,'seed':seed,'n_train':400,'n_dev':100,'test_evaluated':False,
            'parameters':sum(p.numel() for p in model.parameters()),'best_epoch':best_epoch,
            'epochs_run':epoch,'stopped_early':epoch<a.epochs,'best_dev_dice':best,
            'mean_iou':float(np.mean([r['iou'] for r in selected])), 'training_and_dev_seconds':seconds,
            'process_rss_peak_sampled_bytes':peak,'checkpoint_reproduced_per_image':True,
            'latency':latency(model,dev_ids,a.size),'history':history}
    save_json(result,out/'summary.json')
    return result


def summarize(a, plan, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    export=a.output.resolve();export.mkdir(parents=True,exist_ok=True)
    save_json(plan,export/'plan.json')
    means,table,averaged={},[],{}
    for name in MODELS:
        arrays=[]
        for seed in a.seeds:
            run=a.run_root/name/f'seed{seed}'
            for filename in ['summary.json','progress.json','dev_per_image.csv','initial_dev_per_image.csv']:
                (export/name/f'seed{seed}').mkdir(parents=True,exist_ok=True)
                (export/name/f'seed{seed}'/filename).write_bytes((run/filename).read_bytes())
            rows=list(csv.DictReader((run/'dev_per_image.csv').open(encoding='utf-8')))
            arrays.append({r['id']:float(r['dice']) for r in rows})
        ids=plan['ids']['dev'];scores=np.array([[x[i] for i in ids] for x in arrays])
        averaged[name]=[{'id':i,'dice':float(d)} for i,d in zip(ids,scores.mean(0))]
        means[name]=scores.mean(0)
        runs=[r for r in results if r['model']==name]
        table.append({'model':name,'dice':interval(scores.mean(0)),'seed_std':float(scores.mean(1).std(ddof=1)),
                      'seed_means':scores.mean(1).tolist(),'parameters':runs[0]['parameters'],
                      'latency_ms_median_across_seeds':float(np.median([r['latency']['median_ms'] for r in runs]))})
    paired={f'{x} minus {y}':paired_difference(averaged[x],averaged[y])
            for x,y in [('ds_unet','unet'),('fcn_lite','unet')]}
    result={'test_evaluated':False,'table':table,'paired_differences_exploratory':paired,
            'selection_warning':'Development-selected checkpoints; not independent generalization. Different pretraining/input size from SAM.'}
    save_json(result,export/'summary.json')
    lines=['# Comparación ligera: resultados de desarrollo','',
           '400 train / 100 dev; tres semillas; RGB 128, sin preentrenamiento. No se evaluó test.',
           '','| Modelo | Dice [IC 95 % imagen] | DE semillas | Parámetros | Latencia mediana (ms) |',
           '|---|---:|---:|---:|---:|']
    for row in table:
        m,l,h=row['dice'];lines.append(f"| {row['model']} | {m:.4f} [{l:.4f}, {h:.4f}] | {row['seed_std']:.4f} | {row['parameters']} | {row['latency_ms_median_across_seeds']:.2f} |")
    lines+=['','Los IC remuestrean imágenes después de promediar semillas. No ajustan selección ni multiplicidad.',
            'Latencia CPU batch1 incluye preprocesado RGB, red completa y restauración; excluye lectura de disco.',
            'Variantes locales compactas: no son reproducciones exactas de U-Net/FCN publicados ni modelos de estado del arte.',
            'No comparar esta latencia con tiempos de decoder SAM sobre caché. SAM incorpora otro preentrenamiento y resolución.',
            'El tope de épocas es un presupuesto, no prueba de convergencia. Curvas y épocas elegidas deben acompañar la tabla.']
    (export/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in MODELS:
        for r in [r for r in results if r['model']==name]:
            h=r['history'];axes[0].plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label=f"{name} s{r['seed']}")
            axes[1].plot([x['epoch'] for x in h],[x['dev_dice'] for x in h],label=f"{name} s{r['seed']}")
    axes[0].set(ylabel='Train BCE + Dice',xlabel='Epoch');axes[1].set(ylabel='Development Dice',xlabel='Epoch',ylim=(0,1))
    axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(export/'curves.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run_root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--min_epochs',type=int,default=15)
    ap.add_argument('--patience',type=int,default=8);ap.add_argument('--size',type=int,default=128)
    ap.add_argument('--width',type=int,default=8);ap.add_argument('--batch_size',type=int,default=8)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    a=ap.parse_args()
    if min(a.epochs,a.min_epochs,a.patience,a.width,a.batch_size,a.threads)<1 or a.min_epochs>a.epochs or a.size<32 or a.size%16 or a.lr<=0 or len(set(a.seeds))<2 or len(set(a.seeds))!=len(a.seeds):
        ap.error('Invalid training configuration; require at least two unique seeds')
    set_threads(a.threads)
    train,dev=read_split('train400'),read_split('dev');validate_ids(train,dev)
    plan={'models':list(MODELS),'ids':{'train':train,'dev':dev,'test':[]},'test_evaluated':False,
          'args':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()},
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
          'data_sha256':{i:{kind:hashlib.sha256((KVASIR/kind/f'{i}.jpg').read_bytes()).hexdigest() for kind in ['images','masks']} for i in train+dev},
          'selection':'max mean dev Dice incl epoch0, strict greater, fixed logit0; stop after patience epochs without improvement and min_epochs',
          'augmentation':'independent H/V flips p=.5 inside valid rectangle, train only',
          'pretraining':'none; GroupNorm; BCE+Dice; Adam constant learning rate',
          'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','threads':a.threads,'processor':platform.processor()},
          'scope':'Exploratory common-budget lightweight controls, not SOTA; no new test evaluation'}
    root=a.run_root.resolve();a.run_root=root;root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():
        if json.loads((root/'plan.json').read_text(encoding='utf-8'))!=plan:
            raise ValueError('Plan/data/code changed; use a new output root')
    else:
        save_json(plan,root/'plan.json')
        for path in SOURCES:
            dest=root/'source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    save_json(plan,a.output/'plan.json')
    print('PRELOAD 400 train / 100 dev',flush=True)
    train_data=MemoryImages(train,a.size);dev_data=MemoryImages(dev,a.size);masks=[load_mask(i) for i in dev]
    results=[]
    for name in MODELS:
        for seed in a.seeds:
            save_json({'state':'running','model':name,'seed':seed,'completed_runs':len(results)},root/'status.json')
            results.append(train_one(name,seed,a,train_data,dev_data,masks,dev,root/name/f'seed{seed}'))
    summarize(a,plan,results)
    save_json({'state':'complete','completed_runs':len(results),'test_evaluated':False},root/'status.json')
    save_json({'state':'complete','completed_runs':len(results),'test_evaluated':False},a.output/'status.json')
    print('COMPLETE',a.output/'report.md',flush=True)


if __name__=='__main__':
    main()
