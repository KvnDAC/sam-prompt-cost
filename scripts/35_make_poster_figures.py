"""Build the four WAIMLAp poster result figures from the frozen experiment summaries.

Each figure is drawn at the physical size of its slot in the A0 Canva poster
(slot pixels at 96 px/in), so font sizes below are the printed point sizes.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/poster_20261023'
SLOT=(660/96,405/96)  # width, height in inches of each Results cell figure
DPI=300
INK,MUTED='#28282D','#64646E'
C={'sam':'#4B2A82','probe':'#238B8E','unet':'#E76F51','oracle':'#7A7A7A','weak':'#B8A1D9','none':'#D9D3E6'}
SOURCES={'phase_b':'results/phase_b/full_dev_20261012/summary.json',
  'fixed_prior':'results/model_comparison/fixed_prior_20261016/summary.json',
  'output_selection':'results/phase_b/full_dev_20261012/output_selection/summary.json',
  'label_efficiency':'results/model_comparison/label_efficiency_20261017/summary.json',
  'resolution':'results/model_comparison/unet_resolution_20261017/summary.json',
  'external_positive':'results/model_comparison/external_clinicdb_20261017/summary.json',
  'external_negative':'results/model_comparison/negative_specificity_20261017/summary.json'}

def read(key):return json.loads((ROOT/SOURCES[key]).read_text(encoding='utf-8'))
def pick(rows,key,value):return next(x for x in rows if x[key]==value)
def style():
    plt.rcParams.update({'font.family':'Arial','font.size':16,'axes.edgecolor':MUTED,'axes.labelcolor':INK,
      'xtick.color':INK,'ytick.color':INK,'text.color':INK,'axes.linewidth':1.0,'savefig.transparent':True})
def finish(ax):
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=15,length=4)
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=DPI);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)

def prompt_dependence():
    rows=read('phase_b')['table'];prior=read('fixed_prior')['modes']['fixed_boxpoint']['dice']
    items=[('No prompt',pick(rows,'method','Sin prompt'),C['none']),('Fixed prior',None,C['none']),
      ('Static tokens',pick(rows,'method','Tokens estáticos'),C['weak']),('Cond. tokens',pick(rows,'method','Tokens condicionados'),C['weak']),
      ('Linear probe',pick(rows,'method','Probe lineal'),C['probe']),('Auto SAM',pick(rows,'method','Probe → caja + punto → SAM'),C['sam']),
      ('GT point*',pick(rows,'method','Punto GT'),C['oracle']),('GT box*',pick(rows,'method','Caja GT'),C['oracle'])]
    vals=[prior[0] if r is None else r['dice_mean'] for _,r,_ in items]
    lo=[prior[1] if r is None else r['ci95_low'] for _,r,_ in items];hi=[prior[2] if r is None else r['ci95_high'] for _,r,_ in items]
    fig,ax=plt.subplots(figsize=SLOT,constrained_layout=True);y=np.arange(len(items))[::-1]
    ax.barh(y,vals,color=[c for *_,c in items],height=.72)
    ax.errorbar(vals,y,xerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='none',ecolor=INK,capsize=3,lw=1)
    for yi,v,h in zip(y,vals,hi):ax.text(h+.015,yi,f'{v:.3f}',va='center',fontsize=14,color=INK)
    ax.set_yticks(y,[n for n,*_ in items],fontsize=15);ax.set_xlim(0,1.08);ax.set_xlabel('Dev Dice (95% CI)    *oracle prompt',fontsize=16)
    finish(ax);save(fig,'fig1_prompt_dependence')

def candidates():
    m=read('output_selection')['metrics']
    groups=[('No prompt',['np_dice_single','np_dice_multi_ioupick','np_dice_oracle4']),('GT point',['pt_dice_single','pt_dice_ioupick','pt_dice_oracle3'])]
    series=[('Single output',C['weak']),('IoU-selected',C['sam']),('Best candidate (GT)*',C['oracle'])]
    fig,ax=plt.subplots(figsize=SLOT,constrained_layout=True);w=.26;x=np.arange(len(groups))
    for j,(label,color) in enumerate(series):
        v=np.array([m[keys[j]][0] for _,keys in groups]);lo=np.array([m[keys[j]][1] for _,keys in groups]);hi=np.array([m[keys[j]][2] for _,keys in groups])
        xs=x+(j-1)*w;ax.bar(xs,v,w*.92,color=color,label=label)
        ax.errorbar(xs,v,yerr=[v-lo,hi-v],fmt='none',ecolor=INK,capsize=3,lw=1)
        for xi,vi,h in zip(xs,v,hi):ax.text(xi,h+.025,f'{vi:.2f}',ha='center',fontsize=13)
    ax.set_xticks(x,[g for g,_ in groups],fontsize=16);ax.set_ylim(0,1.12);ax.set_ylabel('Dev Dice',fontsize=16)
    ax.legend(frameon=False,fontsize=13,loc='upper left',ncol=1);ax.text(1.6,-.13,'*oracle choice',ha='right',fontsize=13,color=MUTED,transform=ax.get_xaxis_transform())
    finish(ax);save(fig,'fig2_candidates_vs_selection')

def label_efficiency():
    table=read('label_efficiency')['table'];mature=pick(read('resolution')['table'],'method','256_endpoint');budgets=[25,100,400]
    fig,ax=plt.subplots(figsize=SLOT,constrained_layout=True)
    for method,name,color,marker,shift in [('probe','Linear probe',C['probe'],'o',.94),('sam_auto','Auto SAM',C['sam'],'s',1.),('unet_endpoint','U-Net, 1,500 updates',C['unet'],'^',1.06)]:
        cells=[pick([r for r in table if r['method']==method],'budget',b) for b in budgets];v=np.array([c['mean'] for c in cells])
        ax.errorbar([b*shift for b in budgets],v,yerr=[v-[c['ci95'][0] for c in cells],[c['ci95'][1] for c in cells]-v],
          color=color,marker=marker,lw=2.4,ms=8,capsize=3,elinewidth=1,label=name)
    # Same 400 masks and architecture, longer two-stage recipe: the 400-mask gap is a training-budget effect.
    ax.errorbar([400*1.13],[mature['mean']],yerr=[[mature['mean']-mature['ci95'][0]],[mature['ci95'][1]-mature['mean']]],
      color=C['unet'],marker='^',mfc='white',mew=2,ms=10,ls='none',capsize=3,elinewidth=1,label='U-Net, longer recipe')
    ax.set_xscale('log');ax.set_xticks(budgets,[str(b) for b in budgets],fontsize=15);ax.minorticks_off()
    ax.set_xlim(20,520);ax.set_ylim(.45,.9);ax.set_xlabel('Training masks',fontsize=16);ax.set_ylabel('Dev Dice',fontsize=16)
    ax.legend(frameon=False,fontsize=12.5,loc='lower right');finish(ax);save(fig,'fig3_label_efficiency')

def external():
    methods=['probe','sam_auto','unet'];names=['Probe','Auto SAM','U-Net'];colors=[C['probe'],C['sam'],C['unet']]
    pos=read('external_positive')['table'];neg=read('external_negative')['table']
    fig,(a,b)=plt.subplots(1,2,figsize=SLOT,constrained_layout=True);x=np.arange(3)
    v=np.array([pick(pos,'method',m)['frame_mean'] for m in methods]);ci=np.array([pick(pos,'method',m)['cluster_ci95'] for m in methods])
    a.bar(x,v,.7,color=colors);a.errorbar(x,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='none',ecolor=INK,capsize=3,lw=1)
    for xi,vi,h in zip(x,v,ci[:,1]):a.text(xi,h+.015,f'{vi:.2f}',ha='center',fontsize=13)
    a.set_ylim(0,.62);a.set_ylabel('Dice',fontsize=15);a.set_title('CVC-ClinicDB',fontsize=16,fontweight='bold')
    rows=[pick(neg,'method',m) for m in methods];v=np.array([r['area_fraction']['mean'] for r in rows])*100
    ci=np.array([r['area_fraction']['ci95'] for r in rows])*100;rate=[r['area_threshold_rates']['0.01']['mean']*100 for r in rows]
    b.bar(x,v,.7,color=colors);b.errorbar(x,v,yerr=[v-ci[:,0],ci[:,1]-v],fmt='none',ecolor=INK,capsize=3,lw=1)
    for xi,vi,r in zip(x,v,rate):b.text(xi,vi+.8,f'{vi:.1f}%',ha='center',fontsize=13)
    b.set_ylim(0,21);b.set_ylabel('Mean false area (%)',fontsize=15);b.set_title('HyperKvasir negatives',fontsize=16,fontweight='bold')
    for ax in (a,b):ax.set_xticks(x,names,fontsize=13.5);finish(ax);ax.tick_params(axis='x',labelsize=13.5)
    save(fig,'fig4_external')

def main():
    OUT.mkdir(parents=True,exist_ok=True);style()
    prompt_dependence();candidates();label_efficiency();external()
    (OUT/'figure_provenance.json').write_text(json.dumps({'sources':SOURCES,'slot_inches':SLOT,'dpi':DPI,
      'note':'Development results are exploratory; CVC-ClinicDB and HyperKvasir use the frozen external protocols.'},indent=2),encoding='utf-8')
    print(OUT)
if __name__=='__main__':main()
