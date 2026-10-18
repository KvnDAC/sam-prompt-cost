"""Matched continuation from epoch 30: 128 versus 256, three seeds, dev only."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('light20', ROOT/'scripts/20_compare_light_models.py')
light = importlib.util.module_from_spec(spec)
spec.loader.exec_module(light)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_stage(parent_path, out, args, dev_data, masks, ids):
    """Keep endpoint weights/Adam/RNG; reset selection at this arm's resolution.

    Historical 128-pixel best scores cannot be reused to select 256-pixel weights.
    Both arms select from the same parent endpoint plus 15 new checkpoints.
    """
    if (out/'resume.pt').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    parent = torch.load(parent_path, weights_only=False, map_location='cpu')
    if parent['epoch'] != 30:
        raise ValueError('Continuation requires the epoch-30 endpoint, not best.pt')
    model = light.build_model('unet', args.width)
    model.load_state_dict(parent['model'])
    initial = light.evaluate(model, ids, dev_data, masks, args.batch_size)
    best = float(np.mean([r['dice'] for r in initial]))
    # No random operation after importing the parent's exact RNG/Adam state.
    state = {key: parent[key] for key in
             ['model', 'optimizer', 'torch_rng', 'loader_rng', 'python_rng', 'numpy_rng']}
    state.update(history=[], best=best, best_epoch=0, epoch=0, stale=0,
                 seconds=0., peak=psutil.Process().memory_info().rss)
    light.write_rows(initial, out/'initial_dev_per_image.csv')
    light.write_rows(initial, out/'best_dev_per_image.csv')
    light.save_checkpoint(parent['model'], out/'best.pt')
    light.save_json({'parent_epoch': 30, 'parent_checkpoint_sha256': sha(parent_path),
                     'size': args.size, 'initial_dice': best,
                     'selection': 'stage 0 (parent endpoint at arm resolution) plus new epochs'},
                    out/'initialization.json')
    # Commit resume last so interrupted initialization can safely be repeated.
    light.save_checkpoint(state, out/'resume.pt')


def scores(path, ids):
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    by_id = {r['id']: float(r['dice']) for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError(f'Unique identical image IDs required: {path}')
    return np.array([by_id[i] for i in ids])


def ci(values):
    mean, lo, hi = light.interval(values, seed=0, n=10000)
    return {'mean': mean, 'ci95': [lo, hi]}


def summarize(plan, run_root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = plan['ids']['dev']
    vectors, rows = {}, []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for size in plan['sizes']:
        for kind, filename in [('initial', 'initial_dev_per_image.csv'),
                               ('endpoint', 'endpoint_dev_per_image.csv'),
                               ('selected', 'dev_per_image.csv')]:
            matrix = np.array([scores(run_root/f'size{size}/seed{s}'/filename, ids)
                               for s in plan['seeds']])
            name = f'{size}_{kind}'
            vectors[name] = matrix.mean(0)
            rows.append({'method': name, **ci(vectors[name]),
                         'seed_means': matrix.mean(1).tolist(),
                         'seed_std': float(matrix.mean(1).std(ddof=1))})
        for seed in plan['seeds']:
            run = run_root/f'size{size}/seed{seed}'
            dest = output/f'size{size}/seed{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            for filename in ['summary.json', 'initialization.json', 'progress.json',
                             'initial_dev_per_image.csv', 'dev_per_image.csv',
                             'endpoint_dev_per_image.csv']:
                shutil.copyfile(run/filename, dest/filename)
            summary = json.loads((run/'summary.json').read_text())
            init = json.loads((run/'initialization.json').read_text())
            h = summary['history']
            axes[0].plot([30]+[30+x['epoch'] for x in h],
                         [init['initial_dice']]+[x['dev_dice'] for x in h],
                         label=f'{size}, seed {seed}')
            axes[1].plot([30+x['epoch'] for x in h], [x['train_loss'] for x in h],
                         label=f'{size}, seed {seed}')
    baseline = np.array([scores(Path(plan['parent_run_root'])/f'unet/seed{s}/dev_per_image.csv', ids)
                         for s in plan['seeds']]).mean(0)
    sam = np.array([scores_sam(ROOT/f'results/phase_b/full_dev_20261012/probes/seed{s}/probe_dev_per_image.csv', ids)
                    for s in plan['seeds']]).mean(0)
    vectors['128_previous_selected'] = baseline
    vectors['sam_auto'] = sam
    rows.extend([{'method': '128_previous_selected', **ci(baseline)},
                 {'method': 'sam_auto', **ci(sam)}])
    contrasts = [('256_initial', '128_initial'), ('256_endpoint', '128_endpoint'),
                 ('256_selected', '128_selected')]
    for size in plan['sizes']:
        contrasts += [(f'{size}_endpoint', f'{size}_initial'),
                      (f'{size}_selected', '128_previous_selected'),
                      ('sam_auto', f'{size}_selected')]
    result = {'test_evaluated': False, 'table': rows,
              'paired_differences': {f'{a} minus {b}': ci(vectors[a]-vectors[b]) for a,b in contrasts},
              'selection': 'Fresh within-stage dev selection, including epoch 30 endpoint; no historical best carried into either arm',
              'uncertainty': '10000 paired image bootstrap samples after averaging seeds; not patient-level or independent test inference',
              'runs': {f'{size}/seed{s}': json.loads((run_root/f'size{size}/seed{s}/summary.json').read_text())
                       for size in plan['sizes'] for s in plan['seeds']}}
    light.save_json(result, output/'summary.json')
    text = ['# U-Net: entrenamiento adicional y resolución', '',
            '400 train / 100 dev; tres semillas; sin nueva evaluación de test.',
            'Cada par parte del mismo estado de la época 30 (pesos, Adam y RNG).',
            '15 épocas adicionales por brazo, 50 pasos por época, LR 0.001, batch 8.',
            'La selección se reinicia en ambos brazos: estado inicial más las 15 épocas nuevas.',
            'El endpoint usa exactamente 750 pasos adicionales; selected elige por Dice dev.', '',
            '| Método | Dice medio | IC 95 % imágenes |', '|---|---:|---:|']
    for row in rows:
        lo, hi = row['ci95']
        text.append(f"| {row['method']} | {row['mean']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    text += ['', '## Diferencias pareadas exploratorias', '',
             '| Contraste | Diferencia media | IC 95 % |', '|---|---:|---:|']
    for name, value in result['paired_differences'].items():
        lo, hi = value['ci95']
        text.append(f"| {name} | {value['mean']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    text += ['', 'Los IC remuestrean imágenes tras promediar semillas. No corrigen selección, multiplicidad ni dependencia entre imágenes.',
             'La comparación es de desarrollo y fue motivada por resultados anteriores. No demuestra generalización.',
             'Llegar a la época 45 no prueba convergencia. Inspeccionar curvas y épocas seleccionadas.',
             'SAM usa preentrenamiento genérico y entrada 1024; U-Net se entrena desde cero.',
             'La intervención de resolución cambia también la geometría efectiva de la red y su coste de cómputo.', '']
    (output/'report.md').write_text('\n'.join(text), encoding='utf-8')
    axes[0].set(xlabel='Total optimization epochs', ylabel='Development Dice', ylim=(0, 1))
    axes[1].set(xlabel='Total optimization epochs', ylabel='Train BCE + Dice')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'curves.png', dpi=160); plt.close(fig)


def scores_sam(path, ids):
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    mapping = {r['id']: float(r['selfboxpoint_dice']) for r in rows}
    if len(mapping) != len(rows) or set(mapping) != set(ids):
        raise ValueError('SAM IDs differ')
    return np.array([mapping[i] for i in ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', type=Path, default=ROOT/'runs/model_comparison/light_train400_20261016')
    ap.add_argument('--run_root', type=Path, default=ROOT/'runs/model_comparison/unet_resolution_20261017')
    ap.add_argument('--output', type=Path, default=ROOT/'results/model_comparison/unet_resolution_20261017')
    ap.add_argument('--plan_only', action='store_true')
    a = ap.parse_args()
    parent = json.loads((a.parent/'plan.json').read_text())
    light.validate_ids(parent['ids']['train'], parent['ids']['dev'])
    if parent['ids']['train'] != light.read_split('train400') or parent['ids']['dev'] != light.read_split('dev'):
        raise ValueError('Splits changed')
    for path, digest in parent['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Frozen source changed: {path}')
    for image_id, groups in parent['data_sha256'].items():
        for folder, digest in groups.items():
            if sha(light.KVASIR/folder/f'{image_id}.jpg') != digest:
                raise ValueError(f'Data changed: {image_id}, {folder}')
    parents = {f'seed{s}': sha(a.parent/f'unet/seed{s}/resume.pt') for s in [0, 1, 2]}
    plan = {'ids': parent['ids'], 'seeds': [0, 1, 2], 'sizes': [128, 256],
            'parent_run_root': str(a.parent.resolve()), 'parent_plan_sha256': sha(a.parent/'plan.json'),
            'parent_checkpoint_sha256': parents, 'source_sha256': {
                **parent['source_sha256'], str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
            'additional_epochs': 15, 'parent_epoch': 30, 'batch_size': 8, 'width': 8,
            'lr': .001, 'threads': 4, 'test_evaluated': False,
            'selection': 'reset identically in both arms at parent LAST weights, evaluated at arm resolution',
            'early_stopping': False, 'optimizer_and_rng': 'inherited identically within each seed',
            'primary': 'fixed endpoint at +750 Adam updates; dev-selected checkpoints secondary',
            'scope': 'Follow-up development experiment; budget 45 total epochs is not convergence proof'}
    a.run_root.mkdir(parents=True, exist_ok=True); a.output.mkdir(parents=True, exist_ok=True)
    plan_path = a.run_root/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Existing plan differs; use a new run root')
    light.save_json(plan, plan_path); light.save_json(plan, a.output/'plan.json')
    if a.plan_only:
        print('PLAN FROZEN', plan_path, flush=True)
        return
    light.set_threads(4)
    masks = [light.load_mask(i) for i in parent['ids']['dev']]
    completed = 0
    for size in plan['sizes']:
        train = light.MemoryImages(parent['ids']['train'], size)
        dev = light.MemoryImages(parent['ids']['dev'], size)
        args = SimpleNamespace(width=8, lr=.001, batch_size=8, size=size,
                               epochs=15, min_epochs=15, patience=15)
        for seed in plan['seeds']:
            run = a.run_root/f'size{size}/seed{seed}'
            light.save_json({'state': 'running', 'completed_runs': completed,
                             'current': f'{size}/seed{seed}', 'test_evaluated': False}, a.output/'status.json')
            initialize_stage(a.parent/f'unet/seed{seed}/resume.pt', run, args, dev, masks, parent['ids']['dev'])
            print(f'ARM size{size} seed{seed}', flush=True)
            light.train_one('unet', seed, args, train, dev, masks, parent['ids']['dev'], run)
            endpoint = torch.load(run/'resume.pt', map_location='cpu', weights_only=False)
            if endpoint['epoch'] != 15:
                raise ValueError('Fixed endpoint requires 15 completed epochs')
            model = light.build_model('unet', 8); model.load_state_dict(endpoint['model'])
            light.write_rows(light.evaluate(model, parent['ids']['dev'], dev, masks, 8), run/'endpoint_dev_per_image.csv')
            completed += 1
        del train, dev
    summarize(plan, a.run_root, a.output)
    status = {'state': 'complete', 'completed_runs': completed, 'test_evaluated': False}
    light.save_json(status, a.output/'status.json'); light.save_json(status, a.run_root/'status.json')
    print('COMPLETE', a.output, flush=True)


if __name__ == '__main__':
    main()
