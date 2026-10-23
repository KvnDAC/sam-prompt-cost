"""Post-hoc check on the frozen HyperKvasir rows: decoder growth and largest-component control.

Reads only per_image.csv from script 32; no model is run and no threshold changes.
Exploratory: added after inspecting the negative-control results (2026-10-22).
"""
import csv,json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/model_comparison/negative_specificity_20261017/per_image.csv'
OUT=ROOT/'results/model_comparison/negative_postprocessing_20261022'

def bootstrap_mean(x,n=10000,seed=1710):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);means=np.empty(n)
    for i in range(n):means[i]=x[rng.integers(0,len(x),len(x))].mean()
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}
def column(rows,prefix):return np.array([[float(r[f'{prefix}_s{s}']) for s in range(3)] for r in rows])

def main():
    rows=list(csv.DictReader(SOURCE.open(encoding='utf-8')))
    probe_largest=column(rows,'probe_largest');sam=column(rows,'sam_auto_area')
    unet_area=column(rows,'unet_area');unet_largest=column(rows,'unet_largest')
    result={'source':SOURCE.relative_to(ROOT).as_posix(),'n_images':len(rows),'post_hoc':True,
      'unit_of_resampling':'image after seed averaging (patient/video IDs unavailable)',
      'means':{k:bootstrap_mean(v.mean(1)) for k,v in [('probe_largest_component',probe_largest),
        ('sam_auto_area',sam),('unet_area',unet_area),('unet_largest_component',unet_largest)]},
      'paired_differences':{
        # SAM is prompted from the probe's largest component only: positive values mean the decoder grows it.
        'sam_auto_area_minus_probe_largest':bootstrap_mean((sam-probe_largest).mean(1)),
        # Applies the same single-component rule to U-Net before comparing false area.
        'sam_auto_area_minus_unet_largest':bootstrap_mean((sam-unet_largest).mean(1))},
      'image_seed_rate_sam_larger_than_probe_component':float((sam>probe_largest).mean()),
      'interpretation':'Exploratory post-hoc control; not part of the frozen negative-control protocol.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
