"""Train-derived fixed spatial priors: automatic SAM controls, dev only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from samcost.common import load_sam, set_threads
from samcost.data import read_split, load_mask
from samcost.decode import CachedSAM
from samcost.metrics import dice,iou,bootstrap_ci,paired_test
from samcost.prompts import bbox,interior_point


def derive_prior(ids):
    boxes,points=[],[]
    for iid in ids:
        mask=load_mask(iid)
        if not mask.any():raise ValueError('Empty training GT')
        h,w=mask.shape;scale=np.array([w-1,h-1],float)
        boxes.append(bbox(mask)/np.tile(scale,2))
        points.append(interior_point(mask)[0]/scale)
    return np.median(boxes,axis=0),np.median(points,axis=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=4);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    set_threads(a.threads);train,dev=read_split('train400'),read_split('dev')
    if len(train)!=400 or len(dev)!=100 or set(train)&set(dev):raise ValueError('Invalid splits')
    fixed_box,fixed_point=derive_prior(train)
    cs=CachedSAM(load_sam());cs.require_ids(dev)
    for iid in dev:
        if cs.meta[iid]['precision']!='fp32' or cs.meta[iid]['storage']!='fp32':
            raise ValueError('Require FP32 encoding and cache')
    a.output.mkdir(parents=True)
    manifest={'train_ids':train,'dev_ids':dev,'test_evaluated':False,'threads':a.threads,
              'prior_box_normalized':fixed_box.tolist(),'prior_point_normalized':fixed_point.tolist(),
              'rule':'coordinatewise median train400 GT whole-mask bbox and deepest interior point, normalized by width/height minus one',
              'modes':['fixed_point','fixed_box','fixed_boxpoint'],
              'selection':'single output token0, fixed logit0, one query per image; no evaluated GT enters prompt generation',
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cache_meta':{i:cs.meta[i] for i in dev}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    # Read the target GT only after prompts have been generated from image dimensions.
    rows=[]
    with torch.inference_mode():
        for n,iid in enumerate(dev):
            emb,osz,isz=cs.emb(iid);h,w=osz;scale=np.array([w-1,h-1])
            box=fixed_box*np.tile(scale,2);point=fixed_point*scale
            predictions={}
            for mode in manifest['modes']:
                kwargs={}
                if mode in ('fixed_box','fixed_boxpoint'):kwargs['box']=box
                if mode in ('fixed_point','fixed_boxpoint'):kwargs.update(points=point[None],labels=[1])
                logits,_=cs.decode(emb,osz,multimask=False,**kwargs)
                predictions[mode]=(cs.full(logits,osz,isz)[0,0]>0).numpy()
            gt=load_mask(iid)
            for mode,pred in predictions.items():rows.append({'id':iid,'mode':mode,'dice':dice(pred,gt),'iou':iou(pred,gt),'empty':int(not pred.any())})
            if (n+1)%10==0:print(f'FIXED_PRIOR {n+1}/100',flush=True)
    with (a.output/'dev_per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'test_evaluated':False,'n_train':400,'n_dev':100,'modes':{}}
    for mode in manifest['modes']:
        values=np.array([r['dice'] for r in rows if r['mode']==mode])
        summary['modes'][mode]={'dice':bootstrap_ci(values),'empty_rate':float(np.mean([r['empty'] for r in rows if r['mode']==mode]))}
    matrices=[]
    for seed in [0,1,2]:
        file=ROOT/f'results/phase_b/full_dev_20261012/probes/seed{seed}/probe_dev_per_image.csv'
        byid={r['id']:float(r['selfboxpoint_dice']) for r in csv.DictReader(file.open(encoding='utf-8'))}
        if set(byid)!=set(dev):raise ValueError('Probe IDs differ')
        matrices.append([byid[i] for i in dev])
    fixed_byid={r['id']:r['dice'] for r in rows if r['mode']=='fixed_boxpoint'}
    summary['probe_boxpoint_minus_fixed_boxpoint']=paired_test(np.mean(matrices,axis=0),[fixed_byid[i] for i in dev])
    summary['interpretation']='Exploratory fixed-prior control, supervised by train masks; not zero-supervision, no retraining/capacity-matched generator control.'
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
