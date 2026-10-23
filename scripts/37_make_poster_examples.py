"""Real-image panels for the poster method figure, from a saved Phase B spatial case.

The raw Kvasir-SEG images are not versioned, so the RGB frame is recovered from the
published case figure (results/phase_b/.../spatial_cases/predictions.png): the cyan GT
contour is filled to get the GT mask and inpainted away to get a clean frame. SAM
predictions come from the saved masks (predictions.npz) and prompts from the yellow
marks in the same figure. Nothing is re-run; the case was selected post hoc in Phase B
(largest GT-box minus GT-point Dice), so panels are illustrative, not representative.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/'results/phase_b/full_dev_20261012/spatial_cases'
OUT=ROOT/'docs/waimlap_2026/poster_figures'
CASE='cju40sdwukv3k0755y99ug1k8'
ROW=(1218,1613)  # third row of predictions.png; tiles are the full frame resized to 460 px wide
COLS={'input':(23,483),'noprompt':(504,964),'point':(986,1446),'box':(1468,1928)}
SCALE=960/460  # panels are drawn at about 3x their poster size
CYAN,MAGENTA,YELLOW=(0,229,255),(232,23,232),(255,214,0)

def tile(fig,name):
    (y0,y1),(x0,x1)=ROW,COLS[name];return fig[y0:y1,x0:x1].astype(float)

def gt_and_clean_rgb(rgb):
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    contour=(r<140)&(g>150)&(b>150)&(g-r>60)
    gt=ndi.binary_fill_holes(ndi.binary_closing(contour,iterations=2))
    hole=ndi.binary_dilation(contour,iterations=2);img=rgb.copy();known=~hole
    for _ in range(12):  # fill the contour pixels from their known neighbours
        w=ndi.uniform_filter(known.astype(float),5);s=np.stack([ndi.uniform_filter(img[...,c]*known,5) for c in range(3)],-1)
        grow=(~known)&(w>0);img[grow]=s[grow]/w[grow,None];known=known|grow
    return gt,img

def star_center(rgb,gt):
    # Pure-yellow marker only; the frame itself has yellowish debris.
    mask=(rgb[...,0]>220)&(rgb[...,1]>210)&(rgb[...,2]<90);lab,n=ndi.label(mask)
    if not n:raise ValueError('No prompt marker found')
    big=lab==1+int(np.argmax(ndi.sum(mask,lab,range(1,n+1))));y,x=np.argwhere(big).mean(0)
    if not gt[int(round(y)),int(round(x))]:raise ValueError('Prompt marker falls outside the GT polyp')
    return x,y

def up(mask,size):return np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(size,Image.NEAREST))>127
def edge(mask,width):return mask&~ndi.binary_erosion(mask,iterations=width)

def panel(rgb,gt,pred=None,point=None,box=None):
    h,w=rgb.shape[:2];size=(round(w*SCALE),round(h*SCALE))
    base=np.asarray(Image.fromarray(rgb.clip(0,255).astype(np.uint8)).resize(size,Image.LANCZOS)).astype(float)
    if pred is not None:
        p=up(pred,size);base[p]=base[p]*.45+np.array(MAGENTA)*.55;base[edge(p,5)]=MAGENTA
    base[edge(up(gt,size),6)]=CYAN
    im=Image.fromarray(base.astype(np.uint8));d=ImageDraw.Draw(im)
    if box is not None:
        x0,y0,x1,y1=[v*SCALE for v in box]
        for k in range(int(x0),int(x1),34):d.line([(k,y0),(min(k+20,x1),y0)],YELLOW,7);d.line([(k,y1),(min(k+20,x1),y1)],YELLOW,7)
        for k in range(int(y0),int(y1),34):d.line([(x0,k),(x0,min(k+20,y1))],YELLOW,7);d.line([(x1,k),(x1,min(k+20,y1))],YELLOW,7)
    if point is not None:
        cx,cy=point[0]*SCALE,point[1]*SCALE;R,r=34,14
        star=[(cx+(R if i%2==0 else r)*np.sin(i*np.pi/5),cy-(R if i%2==0 else r)*np.cos(i*np.pi/5)) for i in range(10)]
        d.polygon(star,fill=YELLOW,outline=(40,40,45),width=2)
        if pred is not None:  # the point's tiny output would otherwise hide under the marker
            out=np.asarray(im).copy();out[edge(up(pred,size),4)]=MAGENTA;im=Image.fromarray(out)
    return im

def main():
    fig=np.asarray(Image.open(CASES/'predictions.png').convert('RGB'))
    preds=np.load(CASES/'predictions.npz');gt,rgb=gt_and_clean_rgb(tile(fig,'input'));size=(rgb.shape[1],rgb.shape[0])
    point=star_center(tile(fig,'point'),gt)
    bys,bxs=np.nonzero(gt);box=(bxs.min(),bys.min(),bxs.max(),bys.max())  # the oracle box is the tight GT bounding box
    masks={k:up(preds[f'{CASE}_{k}'],size) for k in ['noprompt_0','point_0','box_1']}
    OUT.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb.clip(0,255).astype(np.uint8)).save(OUT/'method_input.png')
    panel(rgb,gt).save(OUT/'example_input.png')
    panel(rgb,gt,masks['noprompt_0']).save(OUT/'example_noprompt.png')
    panel(rgb,gt,masks['point_0'],point=point).save(OUT/'example_point.png')
    panel(rgb,gt,masks['box_1'],box=box).save(OUT/'example_box.png')
    dice={r.split(',')[1]:float(r.split(',')[2]) for r in (CASES/'metrics.csv').read_text().splitlines()[1:] if r.startswith(CASE)}
    (OUT/'examples_provenance.json').write_text(json.dumps({'case':CASE,'split':'dev','dice':dice,
      'sources':['results/phase_b/full_dev_20261012/spatial_cases/predictions.png','results/phase_b/full_dev_20261012/spatial_cases/predictions.npz',
                 'results/phase_b/full_dev_20261012/spatial_cases/metrics.csv'],
      'selection':'post hoc extreme case (largest GT-box minus GT-point Dice); illustrative only',
      'reconstruction':'RGB recovered from the case figure by inpainting the GT contour; GT mask = filled contour'},indent=2),encoding='utf-8')
    print(OUT,dice)
if __name__=='__main__':main()
