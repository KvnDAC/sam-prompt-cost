"""Audit train400/dev image duplication; no predictions or test-set metrics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from scipy.fft import dctn
from samcost.data import load_image, read_split, KVASIR


def signature(image):
    # A DCT perceptual hash is a screening heuristic, not a duplicate diagnosis.
    small=np.asarray(Image.fromarray(image).convert('L').resize((32,32),Image.Resampling.LANCZOS),float)
    low=dctn(small,type=2,norm='ortho')[:8,:8].ravel()[1:]
    return low>np.median(low)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    splits={s:read_split(s) for s in ['train400','dev']}
    files,pixels,hashes,attributes=defaultdict(list),defaultdict(list),{},[]
    for split,ids in splits.items():
        for iid in ids:
            image=load_image(iid);raw=(KVASIR/'images'/f'{iid}.jpg').read_bytes()
            files[hashlib.sha256(raw).hexdigest()].append([split,iid])
            pixels[hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()].append([split,iid])
            hashes[iid]=signature(image)
            attributes.append({'id':iid,'split':split,'height':image.shape[0],'width':image.shape[1]})
    train=np.stack([hashes[i] for i in splits['train400']]);dev=np.stack([hashes[i] for i in splits['dev']])
    distances=np.count_nonzero(train[:,None,:]!=dev[None,:,:],axis=-1)
    candidates=[{'train_id':splits['train400'][i],'dev_id':splits['dev'][j],'hamming_63':int(distances[i,j])}
                for i,j in zip(*np.where(distances<=4))]
    candidates.sort(key=lambda x:(x['hamming_63'],x['train_id'],x['dev_id']))
    exact=[v for v in pixels.values() if len({s for s,_ in v})>1]
    result={'splits':{s:len(ids) for s,ids in splits.items()},'test_evaluated':False,
            'id_overlap':sorted(set(splits['train400'])&set(splits['dev'])),
            'duplicate_file_groups':[v for v in files.values() if len(v)>1],
            'duplicate_decoded_groups':[v for v in pixels.values() if len(v)>1],
            'exact_cross_split_groups':exact,'phash_cross_split_candidates':candidates,
            'phash':'63 low-frequency DCT bits excluding DC, median threshold, Hamming <=4; heuristic only',
            'interpretation':'Candidates require manual review; no confirmed patient/video independence. No split altered.',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.output.mkdir(parents=True);(a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (a.output/'image_shapes.json').write_text(json.dumps(attributes,indent=2),encoding='utf-8')
    print(json.dumps({'exact_cross_split_groups':len(exact),'phash_candidates':len(candidates)}))
    if exact:raise ValueError('Exact train/dev image duplicates detected; review before interpreting comparison')


if __name__=='__main__':main()
