"""Build publication figures directly from the frozen experiment summaries."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/paper_20261018'

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def pick(rows,key,value):return next(x for x in rows if x[key]==value)

def main():
    phase=read(Path('results/phase_b/full_dev_20261012/summary.json'))
    label=read(Path('results/model_comparison/label_efficiency_20261017/summary.json'))
    resolution=read(Path('results/model_comparison/unet_resolution_20261017/summary.json'))
    external=read(Path('results/model_comparison/external_clinicdb_20261017/summary.json'))
    negative=read(Path('results/model_comparison/negative_specificity_20261017/summary.json'))
    OUT.mkdir(parents=True,exist_ok=True)

    colors={'oracle':'#7A7A7A','sam':'#6F2DBD','probe':'#238B8E','unet':'#E76F51','weak':'#B8A1D9'}
    fig,axes=plt.subplots(1,4,figsize=(13.2,3.15),constrained_layout=True)

    # A. Controlled prompt conditions on the fixed development set.
    rows=phase['table']; names=['No prompt','Static\ntokens','Conditioned\ntokens','Linear\nprobe','Auto SAM','GT box']
    source=['Sin prompt','Tokens estáticos','Tokens condicionados','Probe lineal','Probe → caja + punto → SAM','Caja GT']
    vals=[pick(rows,'method',x)['dice_mean'] for x in source]
    lo=[pick(rows,'method',x)['ci95_low'] for x in source];hi=[pick(rows,'method',x)['ci95_high'] for x in source]
    cs=[colors['weak'],colors['weak'],colors['weak'],colors['probe'],colors['sam'],colors['oracle']]
    x=np.arange(len(vals));axes[0].bar(x,vals,color=cs,width=.72)
    axes[0].errorbar(x,vals,yerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='none',ecolor='#222',capsize=2,lw=.8)
    axes[0].set_xticks(x,names,rotation=35,ha='right');axes[0].set_ylim(0,1);axes[0].set_ylabel('Dice')
    axes[0].set_title('A  Prompt dependence',loc='left',fontweight='bold')

    # B. Label efficiency under fixed update counts.
    budgets=[25,100,400]
    for method,label_name,color,marker in [('probe','Probe','probe','o'),('sam_auto','Auto SAM','sam','s'),('unet_endpoint','U-Net','unet','^')]:
        ys=[pick([r for r in label['table'] if r['method']==method],'budget',b)['mean'] for b in budgets]
        axes[1].plot(budgets,ys,label=label_name,color=colors[color],marker=marker,lw=2,ms=5)
    axes[1].set_xscale('log');axes[1].set_xticks(budgets,[str(x) for x in budgets]);axes[1].set_ylim(.45,.86)
    axes[1].set_xlabel('Training masks');axes[1].set_ylabel('Development Dice');axes[1].legend(frameon=False,fontsize=8)
    axes[1].set_title('B  Label efficiency',loc='left',fontweight='bold')

    # C. Frozen positive-domain transfer.
    methods=['probe','sam_auto','unet'];labels=['Probe','Auto SAM','U-Net'];cs=[colors['probe'],colors['sam'],colors['unet']]
    erows=external['table'];vals=[pick(erows,'method',m)['frame_mean'] for m in methods]
    lo=[pick(erows,'method',m)['cluster_ci95'][0] for m in methods];hi=[pick(erows,'method',m)['cluster_ci95'][1] for m in methods]
    x=np.arange(3);axes[2].bar(x,vals,color=cs,width=.65)
    axes[2].errorbar(x,vals,yerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='none',ecolor='#222',capsize=3,lw=.9)
    axes[2].set_xticks(x,labels,rotation=25,ha='right');axes[2].set_ylim(0,.62);axes[2].set_ylabel('Dice')
    axes[2].set_title('C  CVC-ClinicDB transfer',loc='left',fontweight='bold')

    # D. Negative-control false area; annotations give rate with >=1% false area.
    nrows=negative['table'];vals=[pick(nrows,'method',m)['area_fraction']['mean'] for m in methods]
    lo=[pick(nrows,'method',m)['area_fraction']['ci95'][0] for m in methods];hi=[pick(nrows,'method',m)['area_fraction']['ci95'][1] for m in methods]
    rates=[pick(nrows,'method',m)['area_threshold_rates']['0.01']['mean'] for m in methods]
    axes[3].bar(x,np.array(vals)*100,color=cs,width=.65)
    axes[3].errorbar(x,np.array(vals)*100,yerr=[(np.array(vals)-lo)*100,(np.array(hi)-vals)*100],fmt='none',ecolor='#222',capsize=3,lw=.9)
    for i,(v,r) in enumerate(zip(vals,rates)):axes[3].text(i,v*100+.8,f'{r*100:.1f}% ≥1%',ha='center',fontsize=7.5)
    axes[3].set_xticks(x,labels,rotation=25,ha='right');axes[3].set_ylim(0,21);axes[3].set_ylabel('Mean false area (%)')
    axes[3].set_title('D  HyperKvasir negatives',loc='left',fontweight='bold')

    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18,lw=.6);ax.tick_params(labelsize=8)
    fig.suptitle('Controlled evidence across prompts, supervision, and domain shift',fontsize=12,fontweight='bold')
    fig.savefig(OUT/'figure_experimental_summary.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/'figure_experimental_summary.pdf',bbox_inches='tight')
    provenance={'phase_b':'results/phase_b/full_dev_20261012/summary.json',
      'label_efficiency':'results/model_comparison/label_efficiency_20261017/summary.json',
      'resolution':'results/model_comparison/unet_resolution_20261017/summary.json',
      'external_positive':'results/model_comparison/external_clinicdb_20261017/summary.json',
      'external_negative':'results/model_comparison/negative_specificity_20261017/summary.json'}
    (OUT/'figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(OUT/'figure_experimental_summary.png')
if __name__=='__main__':main()
