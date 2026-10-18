"""Run one independent U-Net cell from the frozen label-efficiency plan."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('labels28_cell',ROOT/'scripts/28_label_efficiency.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--budget',type=int,choices=study.BUDGETS,required=True)
    ap.add_argument('--seed',type=int,choices=study.SEEDS,required=True)
    ap.add_argument('--run_root',type=Path,default=ROOT/'runs/model_comparison/label_efficiency_20261017')
    ap.add_argument('--output',type=Path,default=ROOT/'results/model_comparison/label_efficiency_20261017')
    a=ap.parse_args();frozen=json.loads((a.run_root/'plan.json').read_text(encoding='utf-8'))
    current=study.make_plan(a.run_root,a.output)
    if current!=frozen:raise ValueError('Frozen plan/code/data differ')
    study.set_threads(4);dev=frozen['ids']['dev'];ids=frozen['subsets'][str(a.seed)][str(a.budget)]
    dev_data=study.Prepared(dev);dev_masks=[study.load_mask(i) for i in dev]
    run=a.run_root/'unet'/f'n{a.budget}'/f'seed{a.seed}'
    study.train_unet(ids,a.seed,study.Prepared(ids),dev,dev_data,dev_masks,run)


if __name__=='__main__':main()
