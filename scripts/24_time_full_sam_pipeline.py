"""Measure full SAM+probe auto-boxpoint runtime on decoded RGB; dev only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi
from samcost.common import CKPT,load_sam,set_threads
from samcost.data import load_image,load_mask,read_split,SamPreprocessor
from samcost.decode import CachedSAM
from samcost.prompts import bbox,interior_point
from samcost.metrics import dice


@torch.inference_mode()
def autoprompt(cs,emb,osz,isz,linear,mu,sd,threshold):
    z=linear((emb[0].permute(1,2,0)-mu)/sd)[...,0]
    low=F.interpolate(z[None,None],size=(256,256),mode='bilinear',align_corners=False)
    pred=(cs.full(low,osz,isz)[0,0].numpy()>threshold)
    labels,n=ndi.label(pred)
    if not n:return np.zeros(osz,bool)
    sizes=ndi.sum(pred,labels,range(1,n+1));big=labels==(int(np.argmax(sizes))+1)
    point,_=interior_point(big)
    low,_=cs.decode(emb,osz,points=point[None],labels=[1],box=bbox(big),multimask=False)
    return (cs.full(low,osz,isz)[0,0]>0).numpy()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    set_threads(4);ids=read_split('dev')[:20]
    sam=load_sam();cs=CachedSAM(sam);preprocess=SamPreprocessor(sam)
    with CKPT.open('rb') as file:
        checkpoint_sha=hashlib.file_digest(file,'sha256').hexdigest()
    if any(cs.meta[i]['checkpoint_sha256']!=checkpoint_sha for i in ids):
        raise ValueError('Loaded SAM checkpoint differs from cached experiment')
    run=ROOT/'results/phase_b/full_dev_20261012/probes/seed0'
    state=torch.load(run/'probe_state.pt',weights_only=True,map_location='cpu')
    linear=torch.nn.Linear(256,1);linear.load_state_dict(state['linear']);linear.eval()
    mu,sd,thr=state['mu'],state['sd'],state['threshold']
    expected={r['id']:float(r['selfboxpoint_dice']) for r in csv.DictReader((run/'probe_dev_per_image.csv').open(encoding='utf-8'))}
    images={i:load_image(i) for i in ids}
    # Verify the reimplemented runtime path on existing FP32 cache before timing.
    with torch.inference_mode():
        for iid in ids:
            emb,osz,isz=cs.emb(iid)
            pred=autoprompt(cs,emb,osz,isz,linear,mu,sd,thr)
            if abs(dice(pred,load_mask(iid))-expected[iid])>1e-8:raise ValueError('Cached pipeline reproduction failed')
    rows=[]
    with torch.inference_mode():
        for index in range(25):
            iid=ids[index%20];started=time.perf_counter()
            x,osz,isz=preprocess(images[iid]);after_pre=time.perf_counter()
            emb=sam.image_encoder(x);after_encoder=time.perf_counter()
            pred=autoprompt(cs,emb,osz,isz,linear,mu,sd,thr);ended=time.perf_counter()
            # Not included in timing: metric agreement with the cached benchmark.
            measured_dice=dice(pred,load_mask(iid))
            if abs(measured_dice-expected[iid])>1e-8:raise ValueError('Fresh RGB pipeline differs from benchmark prediction')
            if index>=5:
                rows.append({'id':iid,'total_ms':(ended-started)*1000,
                             'preprocess_ms':(after_pre-started)*1000,
                             'encoder_ms':(after_encoder-after_pre)*1000,
                             'probe_prompts_decoder_ms':(ended-after_encoder)*1000})
            print(f'SAM_RUNTIME {index+1}/25',flush=True)
    a.output.mkdir(parents=True)
    result={'method':'SAM ViT-B + linear probe -> largest-component box+point -> SAM',
            'seed':0,'threads':4,'batch_size':1,'warmup':5,'repeats':20,'test_evaluated':False,
            'input_longest_side':1024,'ids':ids,'rows':rows,
            'median_ms':float(np.median([r['total_ms'] for r in rows])),
            'p95_ms':float(np.quantile([r['total_ms'] for r in rows],.95)),
            'parameters_total_unique':sum(p.numel() for p in sam.parameters())+sum(p.numel() for p in linear.parameters()),
            'parameters_trainable':257,'cached_and_fresh_predictions_reproduced':True,
            'includes':'RGB preprocessing + image encoder + probe + connected components + prompt encoder + mask decoder + original-size restore + threshold',
            'excludes':'disk decode, load weights, GT access and metric computation',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},
            'sam_checkpoint_sha256':checkpoint_sha,
            'probe_checkpoint_sha256':hashlib.sha256((run/'probe_state.pt').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ['median_ms','p95_ms','parameters_total_unique']}),flush=True)


if __name__=='__main__':main()
